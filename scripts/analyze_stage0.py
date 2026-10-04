#!/usr/bin/env python3
"""Stage 0: what released checkpoints already answer, before any pretraining.

Three contrasts are available at matched architecture and cost inference only:

  tokenizer   NT-v2 50M 6-mer vs its 3-mer twin. Identical depth, width, head
              count and training corpus; vocabulary 4107 against 75. With k=3 a
              token is exactly a codon, so the masked token IS the codon under
              test. Boshar et al. (2024) trained this model because codon-level
              resolution was where genomic models lost to protein models, and
              reported it helped their two codon-sensitive FINE-TUNED tasks. It
              has never been scored zero-shot.
  corpus      NT v1 500M human-ref vs 500M 1000G (one reference genome against
              3,202 human genomes) and 2.5B 1000G vs multi-species.
  parameters  NT v1 500M vs 2.5B at fixed 1000G corpus.

Every contrast is read against the same bar as the paper: BLOSUM62 at +0.228,
not zero. Paired bootstrap over assays, and over protein clusters where the
cluster map is available.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLOSUM62 = 0.2282
ESM2 = 0.4655


def boot_paired(a, b, B=20000, seed=0):
    """Bootstrap the paired mean difference a - b over assays."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed); n = len(a); d = a - b
    s = [d[rng.integers(0, n, n)].mean() for _ in range(B)]
    return float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def boot_mean(x, B=20000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); n = len(x)
    s = [x[rng.integers(0, n, n)].mean() for _ in range(B)]
    return float(x.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def load(path, key="nt_model"):
    rows = list(csv.DictReader(Path(path).open()))
    by = {}
    for r in rows:
        by.setdefault(r[key], {})[r["dms_id"]] = r
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/stage0_tokenizer.csv")
    ap.add_argument("--field", default="rho_nt")
    args = ap.parse_args()

    by = load(ROOT / args.csv)
    names = list(by)
    if len(names) < 2:
        print(f"only {len(names)} model(s) in {args.csv}; nothing to contrast")
        for n in names:
            v = [float(r[args.field]) for r in by[n].values()]
            m, lo, hi = boot_mean(v)
            print(f"  {n.split('/')[-1]:48s} {m:+.4f} [{lo:+.4f},{hi:+.4f}] n={len(v)}")
        return

    common = sorted(set.intersection(*[set(by[n]) for n in names]))
    print(f"{len(common)} assays common to all {len(names)} models\n")

    print(f"{'model':50s} {'mean rho':>9s} {'95% CI':>20s} {'ceiling':>8s} {'>0':>6s}")
    stats = {}
    for n in names:
        v = np.array([float(by[n][d][args.field]) for d in common])
        c = np.array([float(by[n][d]["attenuation_ceiling"]) for d in common])
        m, lo, hi = boot_mean(v); stats[n] = v
        print(f"{n.split('/')[-1]:50s} {m:+9.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{np.nanmean(c):8.3f} {int((v>0).sum()):4d}/{len(v)}")

    print("\npaired contrasts, bootstrap over assays")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d, lo, hi = boot_paired(stats[a], stats[b])
            verdict = "contains zero" if lo < 0 < hi else "excludes zero"
            print(f"  {a.split('/')[-1][:30]:32s} minus {b.split('/')[-1][:30]:32s}"
                  f" {d:+.4f} [{lo:+.4f},{hi:+.4f}]  {verdict}")

    print(f"\nagainst the real bar (BLOSUM62 = {BLOSUM62:+.4f}), not zero")
    for n in names:
        d, lo, hi = boot_mean(stats[n] - BLOSUM62)
        print(f"  {n.split('/')[-1]:50s} {d:+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"{'CLEARS' if lo > 0 else 'does not clear'}")


if __name__ == "__main__":
    main()
