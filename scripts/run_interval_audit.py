#!/usr/bin/env python3
"""Every paired contrast, at both unit levels, with its multiplicity family declared.

Three defects motivate this script, all of them ours.

First, the project's stated rule counts units at the level of independence, but
cluster-level statistics were reported for exactly one contrast. The Stage 0
tokenizer and corpus results were assay-level only.

Second, the headline table used m +/- 1.96*SE, a method this repository already
lists among its own corrected defects, so the same quantity appeared with two
interval methods in two files.

Third, no multiplicity family was ever declared. This script fixes the family at
the eight paired contrasts below, states it in the output, and reports which
survive Bonferroni correction. Most conclusions here are nulls, which
multiplicity makes harder rather than easier, so the correction is reported for
honesty rather than because it rescues anything.

p-values are two-sided sign-flip permutation tests on the paired differences,
which assume only exchangeability of sign under the null -- appropriate at 21
clusters, where a t-test's normality assumption is doing real work.

  ./.venv-glm/bin/python scripts/run_interval_audit.py
"""
import csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from biologos.biointerp.battery import _tcrit

N_PERM, N_BOOT, SEED, POWER_Z = 20000, 20000, 0, 2.801582  # z(0.975)+z(0.80), two-sided 80% power


def cluster_of(dms_id: str) -> str:
    """Protein-level cluster key: everything before the first author-year field.

    Mirrors run_dms_trivial_baselines.cluster_of, duplicated because that module
    imports run_skempi_transfer, which is not in the repository.

    This is a NAME PREFIX, not a sequence-identity cluster. It cannot catch two
    assays on homologous proteins with different id prefixes, which is the case
    the independence rule was written for. Replacing it with an mmseqs run is
    open work; until then "independent units" is asserted, not measured.
    """
    return "_".join(dms_id.split("_")[:2])


def load():
    """scorer -> {dms_id: rho}, over assays every source shares."""
    s, cl = {}, {}
    base = list(csv.DictReader((ROOT / "results/dms_trivial_baselines.csv").open()))
    for col, name in [("rho_esm", "ESM-2 650M"), ("rho_blosum62", "BLOSUM62"),
                      ("rho_chem_locco", "chemistry ridge"), ("rho_nt", "NT-v2 50M"),
                      ("rho_nt_codon_marginalised", "NT-v2 50M codon-marg")]:
        s[name] = {r["dms_id"]: float(r[col]) for r in base}
    for r in base:
        cl[r["dms_id"]] = r["cluster"] if r.get("cluster") else cluster_of(r["dms_id"])
    s["Evo 2 7B"] = {r["dms_id"]: float(r["rho_evo2"])
                     for r in csv.DictReader((ROOT / "results/evo2_dms.csv").open())}
    for f, key, names in [
        ("results/stage0_tokenizer.csv", "3mer", ("NT-v2 50M 3-mer", "NT-v2 50M 6-mer")),
        ("results/stage0_corpus500m.csv", "1000g", ("NT 500M 1000G", "NT 500M human-ref")),
    ]:
        hit, miss = {}, {}
        for r in csv.DictReader((ROOT / f).open()):
            (hit if key in r["nt_model"].lower() else miss)[r["dms_id"]] = float(r["rho_nt"])
        s[names[0]], s[names[1]] = hit, miss
    return s, cl


# The multiplicity family, fixed here and nowhere else. Each entry is
# (label, treatment, baseline); the difference is always treatment - baseline,
# so a positive number always means the first-named scorer did better.
FAMILY = [
    ("Evo 2 vs BLOSUM62",        "Evo 2 7B",         "BLOSUM62"),
    ("Evo 2 vs chemistry ridge", "Evo 2 7B",         "chemistry ridge"),
    ("Evo 2 vs ESM-2",           "Evo 2 7B",         "ESM-2 650M"),
    ("ESM-2 vs BLOSUM62",        "ESM-2 650M",       "BLOSUM62"),
    ("BLOSUM62 vs NT-v2 50M",    "BLOSUM62",         "NT-v2 50M"),
    ("codon-marg vs raw NT",     "NT-v2 50M codon-marg", "NT-v2 50M"),
    ("tokenizer: 3-mer vs 6-mer", "NT-v2 50M 3-mer", "NT-v2 50M 6-mer"),
    ("corpus: 1000G vs human-ref", "NT 500M 1000G",  "NT 500M human-ref"),
]


def stats(d, rng):
    """Mean, t interval, sign count, permutation p, 80%-power MDE, equivalence bound.

    The equivalence bound is the TOST result stated as a quantity rather than a
    verdict. Equivalence within a symmetric margin Delta holds at the 5% level
    exactly when the 90% interval lies inside +/-Delta, so the smallest margin
    these data could support is the larger endpoint of that interval in absolute
    value. Reporting the bound instead of a yes/no answer avoids inventing a
    margin nobody pre-specified, and makes the honest point directly: if the
    bound is a large fraction of the baseline's own score, the study cannot
    establish equivalence at any margin a reader would accept.

    The 90% interval is a bootstrap percentile interval, which needs no
    distribution function the repository does not have.
    """
    n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n)
    t = _tcrit(n - 1)
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, n))
    null = (d * flips).mean(1)
    p = float((np.abs(null) >= abs(m) - 1e-15).mean())
    bs = d[rng.integers(0, n, size=(N_BOOT, n))].mean(1)
    lo90, hi90 = np.percentile(bs, 5), np.percentile(bs, 95)
    return dict(n=n, mean=m, lo=m - t * se, hi=m + t * se,
                pos=int((d > 0).sum()), p=p, mde=POWER_Z * sd / np.sqrt(n),
                eq=max(abs(lo90), abs(hi90)))


def main():
    s, cl = load()
    rng = np.random.default_rng(SEED)
    alpha, k = 0.05, len(FAMILY)
    rows = []

    print(f"\nMultiplicity family: {k} paired contrasts, declared in FAMILY.")
    print(f"Bonferroni threshold: {alpha}/{k} = {alpha/k:.5f}\n")
    head = (f"{'contrast':29s} {'unit':8s} {'n':>3s} {'mean':>8s} "
            f"{'95% t interval':>20s} {'+/n':>7s} {'p':>7s} {'MDE':>7s} {'eq':>7s}")
    print(head); print("-" * len(head))

    for label, a, b in FAMILY:
        ids = sorted(set(s[a]) & set(s[b]))
        d_assay = np.array([s[a][i] - s[b][i] for i in ids])
        byc = {}
        for i in ids:
            byc.setdefault(cl.get(i) or cluster_of(i), []).append(s[a][i] - s[b][i])
        d_clust = np.array([np.mean(v) for v in byc.values()])
        for unit, d in (("assay", d_assay), ("cluster", d_clust)):
            r = stats(d, rng)
            surv = "yes" if r["p"] < alpha / k else ("marginal" if r["p"] < alpha else "no")
            print(f"{label:29s} {unit:8s} {r['n']:3d} {r['mean']:+8.4f} "
                  f"[{r['lo']:+7.4f}, {r['hi']:+7.4f}] {r['pos']:3d}/{r['n']:<3d} "
                  f"{r['p']:7.4f} {r['mde']:7.4f} {r['eq']:7.4f}")
            rows.append([label, a, b, unit, r["n"], r["mean"], r["lo"], r["hi"],
                         r["pos"], r["p"], alpha / k, surv, r["mde"], r["eq"]])
        print()

    out = ROOT / "results/interval_audit.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "treatment", "baseline", "unit", "n", "mean_diff",
                    "ci_lo_t", "ci_hi_t", "n_positive", "p_permutation",
                    "bonferroni_alpha", "survives_bonferroni", "mde_80pct_power",
                    "smallest_equivalence_margin"])
        w.writerows(rows)

    core = ["ESM-2 650M", "chemistry ridge", "Evo 2 7B", "BLOSUM62", "NT-v2 50M"]
    panel = sorted(set.intersection(*(set(s[n]) for n in core)))
    # Cluster-level, to match the cluster-level bound it is compared against.
    b62 = float(np.mean([np.mean(v) for v in
                         _group(sorted(s["BLOSUM62"]), s["BLOSUM62"], cl).values()]))
    key = [r for r in rows if r[0] == "Evo 2 vs BLOSUM62" and r[3] == "cluster"][0]
    print(f"Equivalence, the test nobody ran. For Evo 2 against BLOSUM62 at the")
    print(f"cluster level the smallest symmetric margin these data could support is")
    print(f"{key[13]:.4f}, which is {key[13]/b62:.0%} of BLOSUM62's own score of {b62:+.4f}.")
    print("No margin that large is one a reader would accept, so the honest statement")
    print("is that this study cannot establish equivalence, only fail to reject")
    print("difference. The 'eq' column reports the same bound for every contrast.\n")

    print("Marginal means with t intervals (these replace the m +/- 1.96*SE headline table).")
    print(f"Restricted to the {len(panel)} assays every scorer shares; Evo 2 covers "
          f"{len(s['Evo 2 7B'])} in all, and mixing panels is how a marginal stops "
          "matching its own paired contrast.\n")
    print(f"{'scorer':24s} {'unit':8s} {'n':>3s} {'mean':>8s} {'95% t interval':>20s}")
    for name in core:
        ids = panel
        for unit, v in (("assay", np.array([s[name][i] for i in ids])),
                        ("cluster", np.array([np.mean(x) for x in
                          _group(ids, s[name], cl).values()]))):
            n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n); t = _tcrit(n - 1)
            print(f"{name:24s} {unit:8s} {n:3d} {m:+8.4f} [{m-t*se:+7.4f}, {m+t*se:+7.4f}]")
    print(f"\nwrote {out}")


def _group(ids, vals, cl):
    g = {}
    for i in ids:
        g.setdefault(cl.get(i) or cluster_of(i), []).append(vals[i])
    return g


if __name__ == "__main__":
    main()
