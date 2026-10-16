"""A valid spread for beta: across independent genomes, not across bootstrap resamples.

The block bootstrap in pooling_control.py is withdrawn. Resampling contiguous blocks
*with replacement* puts the same block into one sequence more than once, which plants
identical segments at assorted separations and inflates I(d) in a distance-dependent
way. It biases the slope it is supposed to bound, and no interval it produced
contained its own point estimate.

The replacement is a real population: the twelve microbial genomes are independent
draws, so beta measured on each one, at a matched base budget, gives a spread that
owes nothing to resampling. The pooled corpus is the concatenation of exactly these
slices, so the comparison is at the same slice length.
"""
import json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from corpus_exponents import load_fasta, mutual_information, fit_beta  # noqa: E402

PER, DMAX, DMIN = 285_715, 512, 4
DISTS = np.unique(np.round(np.logspace(0, np.log10(DMAX), 30)).astype(int))


def beta_of(seq):
    info = np.array([mutual_information(seq, int(d)) for d in DISTS])
    b, r2, n = fit_beta(DISTS, info, DMIN, DMAX)
    return b, r2


def main():
    g = pathlib.Path("../crosstalk/data/genomes")
    micro = sorted(g.glob("M_*.fna.gz")) + sorted(g.glob("M_*.fna"))
    per_genome, slices = [], []
    for p in micro:
        s = load_fasta(p)
        if s.size < PER:
            continue
        sl = s[:PER]
        b, r2 = beta_of(sl)
        gc = float(((sl == 1) | (sl == 2)).mean())
        per_genome.append({"genome": p.name, "gc": gc, "beta": b, "beta_r2": r2})
        slices.append(sl)
        print(f"  {p.name:<46} gc {gc:.3f}  beta {b:+.3f}  r2 {r2:.3f}", flush=True)

    if len(slices) < 3:
        sys.exit("not enough genomes at the matched budget")

    bs = np.array([r["beta"] for r in per_genome], float)
    pooled = np.concatenate(slices)
    pb, pr2 = beta_of(pooled)

    # the one-species control at the identical slice count and length
    one = load_fasta(micro[0])
    n = len(slices)
    ctrl = None
    if one.size >= PER * n:
        rng = np.random.default_rng(0)
        parts = [one[i * PER:(i + 1) * PER] for i in range(n)]
        o = rng.permutation(n)
        cb, cr2 = beta_of(np.concatenate([parts[i] for i in o]))
        ctrl = {"beta": cb, "beta_r2": cr2, "n_slices": n}

    summary = {
        "settings": {"bases_per_slice": PER, "dmax": DMAX, "beta_dmin": DMIN,
                     "n_genomes": len(per_genome)},
        "per_genome": per_genome,
        "single_genome_beta": {
            "mean": float(bs.mean()), "sd": float(bs.std(ddof=1)),
            "min": float(bs.min()), "max": float(bs.max()),
            "pct2.5": float(np.percentile(bs, 2.5)),
            "pct97.5": float(np.percentile(bs, 97.5)),
        },
        "pooled_multispecies": {"beta": pb, "beta_r2": pr2, "n_bases": int(pooled.size)},
        "pooled_onespecies_reordered_control": ctrl,
        "gc_sd_across_genomes": float(np.std([r["gc"] for r in per_genome], ddof=1)),
    }
    pathlib.Path("results/beta_across_genomes.json").write_text(json.dumps(summary, indent=2))

    s = summary["single_genome_beta"]
    print(f"\nsingle genomes (n={len(bs)}): beta mean {s['mean']:+.3f} "
          f"sd {s['sd']:.3f}, range {s['min']:+.3f} to {s['max']:+.3f}")
    print(f"pooled across the same {len(bs)} genomes: beta {pb:+.3f} (r2 {pr2:.3f})")
    if ctrl:
        print(f"one genome, {ctrl['n_slices']} slices reordered: beta "
              f"{ctrl['beta']:+.3f} (r2 {ctrl['beta_r2']:.3f})")
    print(f"GC sd across genomes: {summary['gc_sd_across_genomes']:.4f}")
    print("\nwrote results/beta_across_genomes.json")


if __name__ == "__main__":
    main()
