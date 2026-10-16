"""Measure the two data-statistics exponents that set a scaling law, on DNA corpora.

Ganguli et al derive the data-scaling exponent of an autoregressive model from two
properties of the corpus itself:

    conditional entropy decay      h(k) - h_inf  ~  k^(-gamma)
    token-token correlation decay  I(d)          ~  d^(-beta)

and predict  L(P) ~ P^(-gamma / 2 beta)  for a model trained on P tokens.

Both exponents are properties of the data, not of any network, so they are measurable
by counting. That is the whole reason this script exists: biologos' corpus-breadth
contrast needs 10 GB of checkpoints per arm and has never run. If the exponent ratio
is the same across corpora, the theory predicts breadth is flat before any
checkpoint is downloaded.

No model, no GPU. Everything here is plug-in estimation off k-mer counts.
"""
import argparse, gzip, json, pathlib, sys
import numpy as np
from scipy.optimize import curve_fit

MAP = np.full(256, 255, dtype=np.uint8)
for i, c in enumerate("ACGT"):
    MAP[ord(c)] = i
    MAP[ord(c.lower())] = i


def load_fasta(path):
    """Return a uint8 array of 0..3, dropping every non-ACGT character."""
    op = gzip.open if str(path).endswith(".gz") else open
    chunks = []
    with op(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                continue
            chunks.append(np.frombuffer(line.strip(), dtype=np.uint8))
    if not chunks:
        return np.empty(0, dtype=np.uint8)
    raw = np.concatenate(chunks)
    seq = MAP[raw]
    return seq[seq < 4]


def mm_entropy(counts):
    """Miller-Madow corrected plug-in entropy in bits.

    Plug-in entropy is biased low by roughly (m-1)/(2N log 2) where m is the number
    of *observed* states. The correction is what makes entropies at different k
    comparable at all; without it the h(k) curve bends downward from bias alone and
    any gamma fitted to it is measuring sample size.
    """
    n = counts.sum()
    if n == 0:
        return 0.0
    nz = counts[counts > 0]
    p = nz / n
    plug = -(p * np.log2(p)).sum()
    return plug + (nz.size - 1) / (2.0 * n * np.log(2.0))


def block_entropies(seq, kmax):
    """H(k) for k = 1..kmax, Miller-Madow corrected, in bits."""
    out = {}
    for k in range(1, kmax + 1):
        n = seq.size - k + 1
        code = np.zeros(n, dtype=np.int64)
        for j in range(k):
            code = code * 4 + seq[j:j + n]
        counts = np.bincount(code, minlength=4 ** k)
        out[k] = mm_entropy(counts)
        del code, counts
    return out


def mutual_information(seq, d):
    """I(X_i ; X_{i+d}) in bits, Miller-Madow corrected on the 4x4 joint."""
    a, b = seq[:-d], seq[d:]
    joint = np.bincount(a.astype(np.int64) * 4 + b, minlength=16)
    n = joint.sum()
    if n == 0:
        return 0.0
    hj = mm_entropy(joint)
    ha = mm_entropy(np.bincount(a, minlength=4))
    hb = mm_entropy(np.bincount(b, minlength=4))
    return ha + hb - hj


def fit_beta(dists, info, dmin, dmax):
    """Power-law slope of I(d) over a stated window. Returns beta, r2, n_points."""
    m = (dists >= dmin) & (dists <= dmax) & (info > 0)
    if m.sum() < 4:
        return None, None, int(m.sum())
    x, y = np.log(dists[m]), np.log(info[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(-coef[0]), float(r2), int(m.sum())


def fit_gamma(ks, h):
    """Fit h(k) = h_inf + c * k^(-gamma). Three parameters, so report the fit quality."""
    ks, h = np.asarray(ks, float), np.asarray(h, float)
    if ks.size < 5:
        return None, None, None

    def model(k, h_inf, c, g):
        return h_inf + c * k ** (-g)

    try:
        p0 = [h.min(), max(h[0] - h.min(), 1e-3), 0.5]
        popt, _ = curve_fit(model, ks, h, p0=p0, maxfev=40000,
                            bounds=([-1.0, 0.0, 0.01], [2.1, 10.0, 6.0]))
    except Exception:
        return None, None, None
    pred = model(ks, *popt)
    ss_res = ((h - pred) ** 2).sum()
    ss_tot = ((h - h.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(popt[2]), float(popt[0]), float(r2)


def markov_surrogate(seq, order, n_out, rng):
    """Sample from the order-k Markov chain fitted to seq.

    This is the control that decides whether a power law is real. A finite-order
    Markov chain has *exponentially* decaying correlations, so if real DNA and its
    own order-5 surrogate give the same beta, the power law is an artifact of the
    fitting window rather than long-range structure in the genome.
    """
    k = order
    n = seq.size - k
    ctx = np.zeros(n, dtype=np.int64)
    for j in range(k):
        ctx = ctx * 4 + seq[j:j + n]
    nxt = seq[k:k + n]
    tab = np.bincount(ctx * 4 + nxt, minlength=4 ** k * 4).reshape(4 ** k, 4).astype(np.float64)
    tab += 0.5
    tab /= tab.sum(axis=1, keepdims=True)
    cdf = np.cumsum(tab, axis=1)

    out = np.empty(n_out, dtype=np.uint8)
    out[:k] = seq[:k]
    state = int(ctx[0])
    mask = 4 ** k
    u = rng.random(n_out)
    for i in range(k, n_out):
        b = int(np.searchsorted(cdf[state], u[i]))
        if b > 3:
            b = 3
        out[i] = b
        state = (state * 4 + b) % mask
    return out


def gamma_gate(seq, order, kmax, tol):
    """Find the largest k at which a plug-in h(k) is still trustworthy.

    An order-m Markov chain has h(k) exactly constant for every k > m. So fit that
    chain to the corpus, sample from it, and measure h(k) on the sample: any decline
    past k = m+1 is pure estimation bias, because the generating process has no
    structure there to find. The largest k whose drift stays inside `tol` is the
    largest k a gamma fit may use.

    Without this gate the h(k) curve bends downward from sparsity alone and gamma
    comes out of the fit looking like a property of the genome.
    """
    rng = np.random.default_rng(12345)
    sur = markov_surrogate(seq, order, seq.size, rng)
    H = block_entropies(sur, kmax)
    ks = list(range(2, kmax + 1))
    h = {k: H[k] - H[k - 1] for k in ks}
    ref_k = order + 1
    if ref_k not in h:
        return None, {"reason": f"kmax {kmax} below markov order {order} + 1"}
    ref = h[ref_k]
    drift = {k: h[k] - ref for k in ks if k >= ref_k}
    usable = ref_k
    for k in sorted(drift):
        if abs(drift[k]) <= tol:
            usable = k
        else:
            break
    return usable, {
        "markov_order": order,
        "tol_bits": tol,
        "reference_k": ref_k,
        "surrogate_h": {str(k): float(v) for k, v in h.items()},
        "drift_from_reference_bits": {str(k): float(v) for k, v in drift.items()},
        "usable_kmax": int(usable),
        "note": ("h(k) on an order-%d surrogate must be flat for k > %d; the largest k "
                 "whose drift stays within %.4f bits is the gamma fit ceiling"
                 % (order, order, tol)),
    }


def characterise(name, seq, args, rng, gamma_kmax):
    dists = np.unique(np.round(np.logspace(0, np.log10(args.dmax), 40)).astype(int))
    dists = dists[dists >= 1]
    info = np.array([mutual_information(seq, int(d)) for d in dists])
    beta, beta_r2, npts = fit_beta(dists, info, args.beta_dmin, args.dmax)

    H = block_entropies(seq, args.kmax)
    ks_all = list(range(2, args.kmax + 1))
    h_all = [H[k] - H[k - 1] for k in ks_all]

    # gamma is fitted only inside the gated window, and only if it is wide enough
    ks_fit = [k for k in ks_all if k <= gamma_kmax] if gamma_kmax else []
    gamma = h_inf = gamma_r2 = None
    gamma_blocked = None
    if len(ks_fit) >= 5:
        h_fit = [H[k] - H[k - 1] for k in ks_fit]
        gamma, h_inf, gamma_r2 = fit_gamma(ks_fit, h_fit)
    else:
        gamma_blocked = ("gated k window is %d wide, under the 5 points a three-parameter "
                         "fit needs" % len(ks_fit))

    ratio = None
    if gamma is not None and beta not in (None, 0):
        ratio = gamma / (2.0 * beta)

    return {
        "corpus": name,
        "n_bases": int(seq.size),
        "beta": beta, "beta_r2": beta_r2, "beta_points": npts,
        "gamma": gamma, "gamma_r2": gamma_r2, "h_inf_bits": h_inf,
        "gamma_fit_k_range": [ks_fit[0], ks_fit[-1]] if ks_fit else None,
        "gamma_blocked": gamma_blocked,
        "scaling_exponent_gamma_over_2beta": ratio,
        "h_curve_all_k": {str(k): float(v) for k, v in zip(ks_all, h_all)},
        "mi_curve": {str(int(d)): float(v) for d, v in zip(dists, info)},
    }


def take(seq, n, rng):
    """Trim to exactly n bases from a random offset, or return None if too short.

    Every corpus is cut to the same n. Plug-in entropy bias depends on sample size,
    so comparing a 40 Mb corpus against a 4 Mb one would show an exponent difference
    that is entirely bias. Matching the budget is the control for that.
    """
    if seq.size < n:
        return None
    off = int(rng.integers(0, seq.size - n + 1))
    return seq[off:off + n]


def build_corpora(g, bases):
    """The three corpus axes that stand in for biologos Stage 0, plus the N set."""
    human = sorted(g.glob("H_*.fa.gz")) + sorted(g.glob("H_*.fa"))
    micro = sorted(g.glob("M_*.fna.gz")) + sorted(g.glob("M_*.fna"))
    nset = sorted(g.glob("N_*.fa"))
    out, prov = {}, {}

    if human:
        out["human_chr21"] = load_fasta(human[0])
        prov["human_chr21"] = [human[0].name]
    if micro:
        out["microbe_single"] = load_fasta(micro[0])
        prov["microbe_single"] = [micro[0].name]
        per = bases // len(micro) + 1
        parts, used = [], []
        for p in micro:
            s = load_fasta(p)
            if s.size:
                parts.append(s[:per]); used.append(p.name)
        if parts:
            # named by the number of genomes that actually cleared the slice length,
            # not by how many files matched the glob
            key = f"microbe_pooled_{len(used)}sp"
            out[key] = np.concatenate(parts)
            prov[key] = used
    if nset:
        out["nset_single"] = load_fasta(nset[0])
        prov["nset_single"] = [nset[0].name]
        take_n = min(len(nset), 12)
        per = bases // take_n + 1
        parts, used = [], []
        for p in nset[:take_n]:
            s = load_fasta(p)
            if s.size:
                parts.append(s[:per]); used.append(p.name)
        if parts:
            out["nset_pooled"] = np.concatenate(parts)
            prov["nset_pooled"] = used
    return out, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genomes", default="../crosstalk/data/genomes")
    ap.add_argument("--out", default="results/corpus_exponents.json")
    ap.add_argument("--bases", type=int, default=4_000_000,
                    help="matched base budget per corpus")
    ap.add_argument("--kmax", type=int, default=9)
    ap.add_argument("--dmax", type=int, default=2048)
    ap.add_argument("--beta-dmin", type=int, default=4,
                    help="d below this is excluded; codon periodicity dominates there")
    ap.add_argument("--markov-order", type=int, default=5)
    ap.add_argument("--gate-tol", type=float, default=0.005,
                    help="bits of allowed h(k) drift on the Markov surrogate")
    ap.add_argument("--gamma-tol", type=float, default=0.01,
                    help="gamma must beat the shuffled control by this to be reported")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    g = pathlib.Path(args.genomes)
    if not g.is_dir():
        sys.exit(f"genome directory not found: {g}")

    corpora, prov = build_corpora(g, args.bases)
    if not corpora:
        sys.exit(f"no genomes matched H_*/M_*/N_* in {g}")

    cuts, skipped = {}, []
    for name, seq in corpora.items():
        c = take(seq, args.bases, rng)
        if c is None:
            skipped.append({"corpus": name, "n_bases": int(seq.size),
                            "reason": f"shorter than matched budget {args.bases}"})
            print(f"  skip {name}: {seq.size} bases < {args.bases}", flush=True)
        else:
            cuts[name] = c
    if not cuts:
        sys.exit("every corpus was shorter than the matched budget")

    # The gate runs before any gamma is reported.
    gate_on = sorted(cuts)[0]
    print(f"gate: order-{args.markov_order} surrogate of {gate_on}", flush=True)
    usable, gate = gamma_gate(cuts[gate_on], args.markov_order, args.kmax, args.gate_tol)
    gate["fitted_on"] = gate_on
    print(f"  usable kmax for gamma = {usable}", flush=True)

    rows = []
    for name, cut in cuts.items():
        print(f"  {name}: {cut.size} bases", flush=True)
        r = characterise(name, cut, args, rng, usable)
        r["sources"] = prov.get(name, [])
        rows.append(r)

    base = cuts[gate_on]
    print(f"  control: shuffled {gate_on}", flush=True)
    sh = base.copy(); rng.shuffle(sh)
    r = characterise(f"CONTROL_shuffled_{gate_on}", sh, args, rng, usable)
    r["control"] = "mononucleotide shuffle: destroys all correlation, I(d) must collapse"
    rows.append(r)

    print(f"  control: order-{args.markov_order} Markov surrogate of {gate_on}", flush=True)
    mk = markov_surrogate(base, args.markov_order, args.bases, rng)
    r = characterise(f"CONTROL_markov{args.markov_order}_{gate_on}", mk, args, rng, usable)
    r["control"] = ("finite-order Markov: correlations decay exponentially, so a matching "
                    "beta means the power-law fit is not reading long-range structure")
    rows.append(r)

    # gamma informativeness: a shuffled corpus has no structure, so whatever gamma
    # the estimator returns there is its noise level. Any corpus within that of the
    # shuffle has an uninformative gamma, and its predicted scaling exponent is void.
    # The check lives here rather than in the write-up so the number cannot be quoted
    # out of the JSON by someone who has not read the caveat.
    shuf = next((r for r in rows if r["corpus"].startswith("CONTROL_shuffled")), None)
    informative = None
    if shuf and shuf.get("gamma") is not None:
        g_null = shuf["gamma"]
        informative = {"shuffle_gamma": g_null, "tol": args.gamma_tol}
        for r in rows:
            if r.get("gamma") is None:
                continue
            sep = abs(r["gamma"] - g_null)
            ok = sep > args.gamma_tol
            r["gamma_separation_from_shuffle"] = float(sep)
            r["gamma_informative"] = bool(ok)
            if not ok:
                r["scaling_exponent_gamma_over_2beta"] = None
                r["scaling_exponent_void_reason"] = (
                    "gamma is %.4f against %.4f on a corpus with every correlation "
                    "destroyed, a separation of %.4f inside the %.4f tolerance, so "
                    "gamma carries no information about structure and gamma/2beta is "
                    "not reportable" % (r["gamma"], g_null, sep, args.gamma_tol))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "settings": vars(args), "gamma_gate": gate,
        "gamma_informativeness": informative,
        "skipped": skipped, "results": rows,
    }, indent=2))

    print(f"\n{'corpus':<42} {'bases':>9} {'beta':>7} {'r2':>6} "
          f"{'gamma':>7} {'r2':>6} {'g/2b':>9}")
    for r in rows:
        f = lambda v, p=4: ("%.*f" % (p, v)) if isinstance(v, float) else "--"
        ratio = (f(r['scaling_exponent_gamma_over_2beta'], 4)
                 if r.get('scaling_exponent_gamma_over_2beta') is not None else 'VOID')
        print(f"{r['corpus']:<42} {r['n_bases']:>9} {f(r['beta'],3):>7} "
              f"{f(r['beta_r2'],3):>6} {f(r['gamma'],3):>7} "
              f"{f(r['gamma_r2'],3):>6} {ratio:>9}")
    if informative:
        print(f"\ngamma on a fully shuffled corpus is {informative['shuffle_gamma']:.4f}. "
              f"Any corpus within {informative['tol']} of that has an uninformative "
              f"gamma,\nso its scaling exponent is VOID rather than small. See "
              f"docs/FINDINGS.md section 45.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
