#!/usr/bin/env python3
"""Is the paper's "codon marginalisation" a marginalisation? Measure, don't assume.

`run_dms_transfer.py` computes the codon-marginalised score as `np.mean(syn)`,
where `syn` holds log-probability *differences* against the wild-type codon. That
is the arithmetic mean of log-ratios, i.e. the log of the geometric mean of the
ratios -- not a marginal likelihood. A marginal over synonymous encodings is a
logsumexp.

The two are not related by a rank-preserving transform across a DMS panel. The
offset between them grows with the number of synonymous codons the target amino
acid has, and that count runs 1 to 6 across the 21 amino acids, a spread of
log(6) = 1.792 nats. Because Spearman here is taken across mutations to
*different* amino acids, an amino-acid-dependent offset can reorder the panel.

Whether it does is an empirical question, so this script scores four variants off
one masked pass and reports all four:

  pref       the single preferred codon -- the paper's `rho_nt`, as a control
  mean_logs  mean of log-ratios -- the paper's `rho_nt_codon_marginalised`
  lse_mut    logsumexp over mutant codons, single wild-type codon denominator
  lse_sym    logsumexp over mutant codons minus logsumexp over wild-type-amino-
             acid codons: the symmetric marginal, log P(aa=mut)-log P(aa=wt)

`lse_mut` is the minimal repair and keeps the asymmetric denominator the existing
code uses. `lse_sym` is the quantity the paper's prose actually describes.

Two diagnostics, because a silent failure here would corrupt logsumexp more than
a mean: a single bad term dominates a logsumexp and is averaged away by a mean.
We count vocabulary misses rather than letting them fall back to UNK, and we
report the realised offset spread.

  ~/biologos-deps-run python3 scripts/run_codon_marginal_audit.py --nt MODEL
"""
import os, argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from biologos import glm
from run_proxy_ladder import spearman
from run_dms_transfer import load_singles, nt_token_logprobs

GAUNTLET = Path(os.environ.get("PROTEINGYM_DIR", ROOT / "data/proteingym"))


def lse(a):
    return float(torch.logsumexp(torch.tensor(a, dtype=torch.float64), 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", nargs="+",
                    default=["InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"])
    ap.add_argument("--max-len", type=int, default=420)
    ap.add_argument("--out", default="results/codon_marginal_audit.csv")
    args = ap.parse_args()

    cdsmap = json.loads((ROOT / "data" / "cds" / "dms_cds.json").read_text())
    ref = {r["DMS_id"]: r for r in csv.DictReader((GAUNTLET / "reference.csv").open())}
    pref = glm.preferred_codons()

    rows, misses, offsets = [], 0, []
    hdr = (f"{'assay':30s} {'n':>5s} {'pref':>8s} {'meanlog':>8s} "
           f"{'lse_mut':>8s} {'lse_sym':>8s}")
    for nt_name in args.nt:
        sc = glm.NTScorer(nt_name)
        vocab = sc.tok.get_vocab()
        print(f"\n=== {nt_name}", flush=True)
        print(hdr); print("-" * len(hdr))
        for dms, rec in sorted(cdsmap.items()):
            tgt = ref[dms]["target_seq"]
            if len(tgt) > args.max_len:
                continue
            muts = load_singles(dms, tgt)
            if len(muts) < 200:
                continue
            off = rec["aa_offset"]
            cds = rec["cds"][off * 3: (off + len(tgt)) * 3]
            if glm.translate(cds) != tgt:
                continue
            positions = sorted({p for p, _, _, _ in muts})
            nlp, _ = nt_token_logprobs(sc, cds, positions, 0)

            s_pref, s_mean, s_lsem, s_lses, y = [], [], [], [], []
            for p, wt, mu, score in muts:
                t, slot = sc.token_index(0, p)
                if t not in nlp:
                    continue
                base = list(cds[t * sc.k:(t + 1) * sc.k])
                wt_tok = "".join(base)
                if wt_tok not in vocab:
                    misses += 1; continue
                wt_lp = nlp[t][vocab[wt_tok]]

                def codon_lps(aa):
                    """Log-prob of every token that puts a codon for `aa` in this slot."""
                    nonlocal misses
                    out = []
                    for codon in glm.SYNONYMOUS[aa]:
                        tk = list(base); tk[slot * 3:slot * 3 + 3] = list(codon)
                        key = "".join(tk)
                        if key not in vocab:
                            misses += 1; continue
                        out.append(float(nlp[t][vocab[key]]) - float(wt_lp))
                    return out

                syn_mut, syn_wt = codon_lps(mu), codon_lps(wt)
                if not syn_mut or not syn_wt:
                    continue
                tk = list(base); tk[slot * 3:slot * 3 + 3] = list(pref[mu])
                pk = "".join(tk)
                if pk not in vocab:
                    misses += 1; continue

                m_mean, m_lsem = float(np.mean(syn_mut)), lse(syn_mut)
                s_pref.append(float(nlp[t][vocab[pk]]) - float(wt_lp))
                s_mean.append(m_mean)
                s_lsem.append(m_lsem)
                s_lses.append(m_lsem - lse(syn_wt))
                offsets.append(m_lsem - m_mean)
                y.append(score)

            if len(y) < 200:
                continue
            y = np.array(y)
            r = [spearman(np.array(v), y) for v in (s_pref, s_mean, s_lsem, s_lses)]
            rows.append(dict(dms_id=dms, nt_model=nt_name, n=len(y),
                             rho_pref=r[0], rho_mean_logs=r[1],
                             rho_lse_mut=r[2], rho_lse_sym=r[3]))
            print(f"{dms[:30]:30s} {len(y):5d} {r[0]:+8.4f} {r[1]:+8.4f} "
                  f"{r[2]:+8.4f} {r[3]:+8.4f}", flush=True)

        g = [x for x in rows if x["nt_model"] == nt_name]
        if g:
            print("-" * len(hdr))
            print(f"{'MEAN':30s} {len(g):5d} "
                  + " ".join(f"{np.mean([x[k] for x in g]):+8.4f}" for k in
                             ("rho_pref", "rho_mean_logs", "rho_lse_mut", "rho_lse_sym")))

    print(f"\nvocabulary misses: {misses}")
    if offsets:
        o = np.array(offsets)
        print(f"logsumexp minus mean-of-logs offset: mean {o.mean():+.4f} nats, "
              f"range [{o.min():+.4f}, {o.max():+.4f}], spread {o.max()-o.min():.4f}")
    if rows:
        out = ROOT / args.out
        keys = sorted({k for r in rows for k in r})
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
