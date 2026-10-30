#!/usr/bin/env python3
"""Re-derive every headline number from the result files and fail on drift.

Six numbers in this repository were wrong at once on 2026-10-08, and all six
were wrong in the same way: a value computed correctly at some point, written
into prose, and then left behind when the computation changed or was found to be
buggy. The scaling figures survived a substring bug for weeks. Both Stage 0
contrasts were published in the direction opposite to the one their sentences
implied. A marginal was quoted over 32 assays while its own paired contrast used
25.

Prose cannot be trusted to track data by hand, so this script does it. It has two
jobs:

  CANONICAL  recompute each headline quantity from results/*.csv and check the
             string the documents use still matches.
  RETIRED    scan the documents for values known to be wrong and fail if any
             survives anywhere.

Run it before quoting a number, and after changing any analysis.

  ./.venv-glm/bin/python scripts/check_reported_numbers.py
"""
import csv, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "paper/main.tex", "paper/DESIGN.md", "docs/FINDINGS.md"]


def load():
    s, cl = {}, {}
    base = list(csv.DictReader((ROOT / "results/dms_trivial_baselines.csv").open()))
    for col, name in [("rho_esm", "esm"), ("rho_blosum62", "b62"),
                      ("rho_chem_locco", "chem"), ("rho_nt", "nt50")]:
        s[name] = {r["dms_id"]: float(r[col]) for r in base}
    for r in base:
        cl[r["dms_id"]] = r["cluster"] or "_".join(r["dms_id"].split("_")[:2])
    s["evo2"] = {r["dms_id"]: float(r["rho_evo2"])
                 for r in csv.DictReader((ROOT / "results/evo2_dms.csv").open())}
    sweep = list(csv.DictReader((ROOT / "results/dms_scale_sweep.csv").open()))
    panel = set(s["b62"])
    nt = {}
    for tag in ("50m", "100m", "250m", "500m"):
        v = [float(r["rho_nt"]) for r in sweep
             if f"-v2-{tag}-" in r["nt_model"] and r["dms_id"] in panel]
        nt[tag] = np.array(v)
    tok = defaultdict(dict)
    for r in csv.DictReader((ROOT / "results/stage0_tokenizer.csv").open()):
        tok["3mer" if "3mer" in r["nt_model"] else "6mer"][r["dms_id"]] = float(r["rho_nt"])
    cor = defaultdict(dict)
    for r in csv.DictReader((ROOT / "results/stage0_corpus500m.csv").open()):
        cor["1000g" if "1000g" in r["nt_model"] else "href"][r["dms_id"]] = float(r["rho_nt"])
    ctrl = {r["a"]: float(r["rho_within_bg"])
            for r in csv.DictReader((ROOT / "results/transfer_analysis.csv").open())
            if r["stage"] == "causal_control"}
    return s, cl, nt, tok, cor, ctrl


def codon_audit_spearman():
    """Spearman of mean rho against parameter count, per scoring convention.

    The monotonicity claim in Section 4.2 is a property of the preferred-codon
    score and not of the family: under any marginalisation the 100M rung falls
    below the 50M one. Both coefficients are checked here because +0.800 was
    also, separately, the value the pre-fix scaling bug produced for the
    preferred-codon score, and a bare number cannot tell the two apart.
    """
    from scipy.stats import spearmanr
    rows = []
    for f in ("results/codon_marginal_audit.csv",
              "results/codon_marginal_audit_ladder.csv"):
        rows += list(csv.DictReader((ROOT / f).open()))
    size = lambda m: int(m.split("-v2-")[1].split("m-")[0])
    sizes = sorted({size(r["nt_model"]) for r in rows})
    out = {}
    for col in ("rho_pref", "rho_mean_logs", "rho_lse_mut", "rho_lse_sym"):
        means = [np.mean([float(r[col]) for r in rows if size(r["nt_model"]) == z])
                 for z in sizes]
        out[col] = spearmanr(sizes, means).statistic
    return out


def paired(a, b):
    k = sorted(set(a) & set(b))
    return np.array([a[i] - b[i] for i in k])


def main():
    s, cl, nt, tok, cor, ctrl = load()
    aud = codon_audit_spearman()
    panel = sorted(set(s["b62"]) & set(s["evo2"]))
    x = np.log10([5e7, 1e8, 2.5e8, 5e8])
    y = np.array([nt[t].mean() for t in ("50m", "100m", "250m", "500m")])
    slope, icept = np.polyfit(x, y, 1)
    pred7b = slope * np.log10(7e9) + icept
    evo = np.mean([s["evo2"][i] for i in panel])
    b62 = np.mean([s["b62"][i] for i in panel])

    # (label, recomputed value, the string the documents must contain)
    CANON = [
        ("Evo 2 mean rho",            evo,                     "+0.266"),
        ("BLOSUM62 mean rho",         b62,                      "+0.228"),
        ("ESM-2 mean rho",            np.mean([s["esm"][i] for i in panel]),  "+0.465"),
        ("NT-v2 50M mean rho",        nt["50m"].mean(),         "-0.0132"),
        ("Evo 2 minus BLOSUM62",      paired(s["evo2"], s["b62"]).mean(),     "+0.0376"),
        ("Evo 2 minus chem ridge",    paired(s["evo2"], s["chem"]).mean(),    "-0.0202"),
        ("tokenizer 3mer-6mer",       paired(tok["3mer"], tok["6mer"]).mean(), "-0.0156"),
        ("corpus 1000G-human-ref",    paired(cor["1000g"], cor["href"]).mean(), "-0.0054"),
        ("NT slope per decade",       slope,                    "+0.0207"),
        ("NT prediction at 7B",       pred7b,                   "+0.028"),
        ("Evo 2 shortfall factor",    evo / pred7b,             "9.5"),
        ("params to reach BLOSUM62",  (b62 - icept) / slope,    "19.5"),
        ("scaling Spearman, pref",    aud["rho_pref"],          "+1.000"),
        ("scaling Spearman, marginal", aud["rho_lse_sym"],      "+0.800"),
        ("map control, canonical",    ctrl.get("canonical", float("nan")),    "+0.141"),
        ("map control, frameshift",   ctrl.get("frameshift+1", float("nan")), "+0.143"),
        ("map control, revcomp",      ctrl.get("reverse_complement", float("nan")), "+0.129"),
    ]

    # Values that were published and are now known wrong. Any survivor is a bug.
    RETIRED = [
        ("+0.0179", "pre-fix scaling slope"),
        ("Spearman $+0.800$", "pre-fix scaling Spearman; the marginalised "
                              "coefficient is a different quantity and is "
                              "checked above"),
        ("+0.0241", "pre-fix 7B prediction"),
        ("factor of eleven", "pre-fix shortfall"),
        ("11x",     "pre-fix shortfall"),
        ("10^{21.3}", "pre-fix parameters to BLOSUM62"),
        ("1.9e21",  "pre-fix parameters to BLOSUM62"),
        ("-0.0099", "pooled 50M/250M mean from the substring bug"),
        ("+0.0156", "Stage 0 tokenizer stated baseline-minus-treatment"),
        ("+0.0054", "Stage 0 corpus stated baseline-minus-treatment"),
        ("statistically indistinguishable from a substitution matrix",
                    "equivalence asserted from a failure to reject"),
        ("statistically indistinguishable from BLOSUM62",
                    "equivalence asserted from a failure to reject"),
    ]

    text = {d: (ROOT / d).read_text() for d in DOCS}
    joined = "\n".join(text.values())
    fails = []

    print(f"{'quantity':28s} {'recomputed':>12s}  {'documents say':>14s}   status")
    print("-" * 74)
    for label, value, token in CANON:
        here = [d for d in DOCS if token in text[d]]
        # Compare at the precision the documents actually write, not at full
        # precision: a value printed as "9.5" is correct for anything in
        # [9.45, 9.55), and a tolerance tighter than the rounding would flag
        # every correctly-rounded number as drift.
        num = token.rstrip("x")
        try:
            want = float(num)
            decimals = len(num.split(".")[1]) if "." in num else 0
            # 1e-9 of slack so a value sitting exactly on the rounding
            # boundary is accepted rather than failing on float
            # representation error.
            close = abs(want - value) <= 0.5 * 10 ** (-decimals) + 1e-9
        except ValueError:
            close = True
        ok = bool(here) and close
        print(f"{label:28s} {value:+12.4f}  {token:>14s}   "
              f"{('ok, in %d file(s)' % len(here)) if ok else 'MISMATCH'}")
        if not here:
            fails.append(f"{label}: {token!r} appears in no document, "
                         f"but recomputes to {value:+.4f}")
        elif not close:
            fails.append(f"{label}: documents say {token!r}, data say {value:+.4f}")

    print("\nretired values still present:")
    clean = True
    for token, why in RETIRED:
        hits = [d for d in DOCS if token in text[d]]
        if hits:
            clean = False
            print(f"  FOUND {token!r} in {', '.join(hits)}  -- {why}")
            fails.append(f"retired value {token!r} survives in {', '.join(hits)}")
    if clean:
        print("  none")

    if fails:
        print(f"\n{len(fails)} problem(s):")
        for f in fails:
            print("  - " + f)
        return 1
    print("\nAll headline numbers recompute from results/ and no retired value survives.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
