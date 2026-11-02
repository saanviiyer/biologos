#!/usr/bin/env python3
"""The trivial baselines the memorisation run should be compared against.

`analyze_memorisation_dna.py` reads three columns per scored span -- `mk5_nt`,
`floor_nt` and `mk5_logp_tok` -- from `results/memorisation_dna_baselines.csv`.
That file was never in this repository, so every Markov and composition number
in FINDINGS.md 32 and 36, and every ratio-corrected membership AUC, came back as
nan and could not be confirmed. This rebuilds it.

Nothing here needs a GPU or a model. Each row of the reconstruction CSVs records
the contig and offset its window came from, so the window and the masked span
are both recoverable exactly from the genome, and the baselines are a property
of the sequence alone. `scripts/fetch_memdna_genomes.py` puts the genomes on
disk; this reconstructs 480 windows and checks each one's GC against the value
the run recorded before trusting any of it.

  python3 scripts/fetch_memdna_genomes.py
  python3 scripts/make_memorisation_baselines.py

The baselines, on the identical spans the model was scored on:

  floor_nt      always emit the window's commonest base.
  mk5_nt        order-5 Markov chain fitted to the source genome, filling the
                span from the flanking context and then from its own output.
                The model reconstructs every masked position in one pass and
                sees neither the span nor its own predictions, so a chain that
                is fed the true bases as it goes is being given information the
                model did not have. This column withholds it.
  mk5_nt_tf     the same chain fed the true preceding bases throughout. This is
                the optimistic bound, reported beside the honest one because
                which of the two "greedy fill" means is exactly the kind of
                scoring choice this project has been caught by before.
  mk5_logp_tok  the chain's mean log p of the TRUE 6-mer per token. The ratio
                correction divides out how predictable a span is to begin with,
                which is a question about the true sequence, so the chain is
                conditioned on the true bases here by construction.

Markov emission is a baseline predictor over genomes already on disk, in the
same spirit as the BLOSUM62 bar: an order-5 lookup table filling a 60-nt gap to
measure an accuracy. It is not a design loop and the sequences it writes are
never kept.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_memdna_genomes import bare  # noqa: E402

GEN = ROOT / "data" / "genomes"
OUT = ROOT / "results" / "memorisation_dna_baselines.csv"
RUNS = ("results/memorisation_dna_nt50m.csv",
        "results/memorisation_dna_nt500m.csv")
W, K = 3000, 5                      # window nucleotides; Markov order
BASES = "ACGT"
CODE = np.full(256, 255, np.uint8)
for i, b in enumerate(BASES):
    CODE[ord(b)] = i


def encode(s):
    return CODE[np.frombuffer(s.encode(), np.uint8)]


_seq = {}


def contig(acc):
    if acc not in _seq:
        lines = (GEN / f"{acc}.fa").read_text().splitlines()
        _seq[acc] = "".join(l.strip() for l in lines[1:]).upper()
    return _seq[acc]


class Markov5:
    """P(next base | previous 5), counted over a whole genome, add-one smoothed.

    Positions whose 6-mer window touches an ambiguity code are dropped rather
    than mapped to a base, so an N never invents a transition.
    """

    def __init__(self, seqs):
        cnt = np.zeros(4 ** (K + 1), np.int64)
        for s in seqs:
            e = encode(s)
            if len(e) <= K:
                continue
            ok = e != 255
            idx = np.zeros(len(e) - K, np.int64)
            good = np.ones(len(e) - K, bool)
            for j in range(K + 1):
                idx = idx * 4 + np.where(ok[j:len(e) - K + j], e[j:len(e) - K + j], 0)
                good &= ok[j:len(e) - K + j]
            cnt += np.bincount(idx[good], minlength=4 ** (K + 1))
        tab = (cnt.reshape(-1, 4) + 1).astype(np.float64)
        self.p = tab / tab.sum(1, keepdims=True)
        self.logp = np.log(self.p)
        self.best = self.p.argmax(1).astype(np.uint8)

    @staticmethod
    def _ctx(e):
        """Index of a 5-base context; None if it contains an ambiguity code."""
        if (e == 255).any():
            return None
        v = 0
        for x in e:
            v = v * 4 + int(x)
        return v

    def fill(self, prefix, n, truth=None):
        """Greedy fill of n bases. truth=None self-feeds; otherwise teacher-forced."""
        ctx = list(prefix[-K:])
        out = np.empty(n, np.uint8)
        for i in range(n):
            c = self._ctx(np.array(ctx, np.uint8))
            out[i] = self.best[c] if c is not None else 255
            ctx = ctx[1:] + [int(truth[i]) if truth is not None else int(out[i])]
        return out

    def logp_true(self, prefix, true):
        """Mean log p of each true base given the true 5 before it."""
        ctx, tot, n = list(prefix[-K:]), 0.0, 0
        for b in true:
            c = self._ctx(np.array(ctx, np.uint8))
            if c is not None and b != 255:
                tot += self.logp[c, int(b)]
                n += 1
            ctx = ctx[1:] + [int(b)]
        return tot / n if n else np.nan


def main():
    if not GEN.is_dir():
        sys.exit("data/genomes is missing; run scripts/fetch_memdna_genomes.py")

    rows = []
    for f in RUNS:
        rows += [r for r in csv.DictReader((ROOT / f).open())
                 if r["contig"] not in ("-", "")]

    # Reconstruct every window and check it against the GC the run recorded. A
    # silent off-by-one here would produce baselines for the wrong sequence and
    # nothing downstream would notice.
    wins, bad = {}, 0
    for r in rows:
        key = (r["contig"], int(r["start"]))
        if key in wins:
            continue
        w = contig(bare(r["contig"]))[key[1]:key[1] + W]
        gc = (w.count("G") + w.count("C")) / len(w) if w else -1.0
        if len(w) != W or abs(gc - float(r["gc"])) > 5e-4:
            bad += 1
        wins[key] = w
    if bad:
        sys.exit(f"{bad} of {len(wins)} windows do not reproduce their recorded GC")
    print(f"{len(wins)} windows reconstructed, all GC-identical to the run")

    by_genome = {}
    for r in rows:
        by_genome.setdefault(r["genome"], set()).add(bare(r["contig"]))
    chains = {}
    for g, accs in sorted(by_genome.items()):
        chains[g] = Markov5([contig(a) for a in sorted(accs)])
        print(f"  chain fitted: {g:22s} {len(accs):3d} contig(s)", flush=True)

    enc = {k: encode(v) for k, v in wins.items()}
    floors = {}
    for k, e in enc.items():
        v = e[e != 255]
        floors[k] = np.bincount(v, minlength=4).argmax() if len(v) else 255

    seen, out = set(), []
    for r in rows:
        L, st = int(r["L"]), int(r["span_start"])
        key = (r["wid"], L, st)
        if key in seen:
            continue
        seen.add(key)
        e = enc[(r["contig"], int(r["start"]))]
        a, b = st * 6, (st + L) * 6
        true = e[a:b]
        valid = true != 255
        n = int(valid.sum())
        if n == 0:
            continue
        mk = chains[r["genome"]]
        prefix = e[max(0, a - K):a]
        if len(prefix) < K:                       # span at the window's edge
            prefix = np.concatenate([np.zeros(K - len(prefix), np.uint8), prefix])
        out.append(dict(
            wid=r["wid"], L=L, span_start=st,
            floor_nt=round(float((true[valid] == floors[(r["contig"], int(r["start"]))]
                                  ).mean()), 6),
            mk5_nt=round(float((mk.fill(prefix, b - a)[valid] == true[valid]).mean()), 6),
            mk5_nt_tf=round(float((mk.fill(prefix, b - a, truth=true)[valid]
                                   == true[valid]).mean()), 6),
            mk5_logp_tok=round(float(mk.logp_true(prefix, true) * 6), 6)))

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print(f"\n{len(out)} spans -> {OUT}")
    for col in ("floor_nt", "mk5_nt", "mk5_nt_tf"):
        print(f"  mean {col:12s} {np.mean([r[col] for r in out]):.4f}")


if __name__ == "__main__":
    sys.exit(main())
