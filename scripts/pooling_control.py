"""Why the pooled corpora lose their correlation power law, and whether beta survives.

The matched run found that single genomes have a clean power-law I(d) while pooled
multi-species corpora do not: their I(d) stops decaying and starts *rising* with
distance. Two explanations:

  (a) concatenating slices at all -- junctions and reordering break the decay
  (b) pooling sources of different base composition -- each slice is internally
      homogeneous and differs from its neighbours, which injects a distance-
      independent correlation floor that dominates once the true decay falls below it

These make opposite predictions for one genome sliced and reordered the same way.
Under (a) beta dies there too. Under (b) beta survives, because composition is
constant. That is the control this script runs, with a block bootstrap on beta so
the comparison has intervals.
"""
import json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from corpus_exponents import load_fasta, mutual_information, fit_beta  # noqa: E402

BASES, NSLICE, DMAX, DMIN, NBOOT = 4_000_000, 14, 2048, 4, 60


def beta_of(seq, dmax=DMAX):
    dists = np.unique(np.round(np.logspace(0, np.log10(dmax), 40)).astype(int))
    info = np.array([mutual_information(seq, int(d)) for d in dists])
    b, r2, n = fit_beta(dists, info, DMIN, dmax)
    return b, r2, dists, info


BOOT_DISTS = np.unique(np.round(np.logspace(0, np.log10(512), 18)).astype(int))


def boot_beta(seq, rng, nboot=NBOOT, nblock=40):
    """Block bootstrap: resample contiguous blocks, so within-block correlation is kept.

    A coarser distance grid and a 512 ceiling than the point estimate uses, because
    the interval only has to bound the slope, and 60 full sweeps is already the bulk
    of this script's runtime.
    """
    out = []
    blen = seq.size // nblock
    for _ in range(nboot):
        idx = rng.integers(0, nblock, nblock)
        res = np.concatenate([seq[i * blen:(i + 1) * blen] for i in idx])
        info = np.array([mutual_information(res, int(d)) for d in BOOT_DISTS])
        b, _, _ = fit_beta(BOOT_DISTS, info, DMIN, 512)
        if b is not None:
            out.append(b)
    if not out:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def gc(seq):
    return float(((seq == 1) | (seq == 2)).mean())


def main():
    g = pathlib.Path("../crosstalk/data/genomes")
    micro = sorted(g.glob("M_*.fna.gz")) + sorted(g.glob("M_*.fna"))
    if len(micro) < 2:
        sys.exit("need the microbial set")
    rng = np.random.default_rng(0)
    per = BASES // NSLICE + 1

    # per-species composition, which is the quantity explanation (b) blames
    comp = []
    slices = []
    for p in micro:
        s = load_fasta(p)
        if s.size >= per:
            slices.append(s[:per])
            comp.append({"genome": p.name, "gc": gc(s[:per]), "n": int(s.size)})
    if len(slices) < 2:
        sys.exit("not enough long-enough genomes")

    gcs = np.array([c["gc"] for c in comp])
    print(f"{len(comp)} genomes, GC range {gcs.min():.3f} to {gcs.max():.3f}, "
          f"sd {gcs.std():.4f}", flush=True)

    one = load_fasta(micro[0])
    arms = {}

    # the real pooled corpus: many species, fixed order
    arms["pooled_multispecies"] = np.concatenate(slices)[:BASES]

    # control: ONE species, cut into the same number of slices, reordered randomly.
    # Same junction count, same slice length, composition held constant.
    if one.size >= per * NSLICE:
        parts = [one[i * per:(i + 1) * per] for i in range(NSLICE)]
        order = rng.permutation(NSLICE)
        arms["pooled_onespecies_reordered"] = np.concatenate(
            [parts[i] for i in order])[:BASES]
        arms["contiguous_onespecies"] = one[:BASES]
    else:
        print(f"  note: {micro[0].name} has {one.size} bases, under {per*NSLICE} "
              f"needed for the reordered control", flush=True)

    rows = []
    for name, seq in arms.items():
        b, r2, dists, info = beta_of(seq)
        lo, hi = boot_beta(seq, rng)
        rows.append({"arm": name, "n_bases": int(seq.size), "gc": gc(seq),
                     "beta": b, "beta_r2": r2, "beta_ci95": [lo, hi],
                     "mi_curve": {str(int(d)): float(v) for d, v in zip(dists, info)}})
        print(f"  {name:<34} beta {b:+.3f}  r2 {r2:.3f}  "
              f"CI [{lo:+.3f}, {hi:+.3f}]", flush=True)

    out = pathlib.Path("results/pooling_control.json")
    out.write_text(json.dumps({
        "settings": {"bases": BASES, "n_slices": NSLICE, "dmax": DMAX,
                     "beta_dmin": DMIN, "nboot": NBOOT},
        "per_genome_composition": comp, "results": rows,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
