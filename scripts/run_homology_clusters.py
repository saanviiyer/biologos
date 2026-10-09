#!/usr/bin/env python3
"""Measure the homology clusters this project has been asserting.

Every interval in this repository that counts units at the "protein cluster"
level uses a NAME PREFIX: everything in a ProteinGym id before the first
author-year field. That is a convention, not a measurement, and it cannot catch
the case the independence rule exists for -- two assays on homologous proteins
whose ids do not share a prefix. The influenza-NP example in the README, two
strains at 94% identity, is exactly that case.

This script replaces the assertion with a measurement. It translates each
assay's coding sequence, aligns all pairs, and clusters by sequence identity,
then reports whether the unit count the project has been using is right.

Method: a 3-mer Jaccard prefilter, then Smith-Waterman local alignment with
BLOSUM62 and affine gaps (-11 open, -1 extend) on every pair the filter passes.
Percent identity is counted over aligned columns. Clusters are single-linkage.

Honest limit, stated because it bounds what the result licenses: a k-mer
prefilter plus pairwise alignment detects close homology reliably and remote
homology poorly. This rules out the 94%-identity case. It does not substitute
for a profile search, and a pair related below roughly 25% identity could still
be missed.

  ./.venv-glm/bin/python scripts/run_homology_clusters.py
"""
import csv, itertools, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

AA = "ARNDCQEGHILKMFPSTWYV"
_B62 = """4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0
-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3
-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3
-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3
0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2
-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2
0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3
-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3
-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1
-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2
-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1
-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2
1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2
0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3
-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1
0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4"""
B62 = {(a, b): int(v) for a, row in zip(AA, _B62.strip().split("\n"))
       for b, v in zip(AA, row.split())}

BASES = "TCAG"
_AAS = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
CODON = {a + b + c: _AAS[i] for i, (a, b, c) in
         enumerate(itertools.product(BASES, BASES, BASES))}

GAP_OPEN, GAP_EXT = -11, -1
ID_THRESHOLD, MIN_COLS, JACCARD_MIN = 0.30, 50, 0.005
# Coverage of the shorter sequence. Identity over a short window is the classic
# twilight-zone artifact: a local aligner searching a 7,000-residue polyprotein
# will find a 70-residue window at 31% identity by chance. Requiring the
# alignment to span half the shorter protein is what separates homology from a
# well-scoring fragment, and it is the criterion that changes the answer here.
COVERAGE_MIN = 0.50


def translate(cds):
    aa = [CODON.get(cds[i:i + 3].upper(), "X") for i in range(0, len(cds) - 2, 3)]
    return "".join(aa).split("*")[0]


def smith_waterman(a, b):
    """Local alignment score, aligned columns and identities. Affine gaps."""
    n, m = len(a), len(b)
    NEG = -10**9
    H = np.zeros(m + 1); E = np.full(m + 1, NEG, float)
    ptr = np.zeros((n + 1, m + 1), np.int8)   # 0 stop, 1 diag, 2 up, 3 left
    best, bi, bj = 0.0, 0, 0
    for i in range(1, n + 1):
        prevH, H_new = H, np.zeros(m + 1)
        F = NEG
        ai = a[i - 1]
        for j in range(1, m + 1):
            E[j] = max(E[j] + GAP_EXT, prevH[j] + GAP_OPEN)
            F = max(F + GAP_EXT, H_new[j - 1] + GAP_OPEN)
            diag = prevH[j - 1] + B62.get((ai, b[j - 1]), -4)
            v = max(0.0, diag, E[j], F)
            H_new[j] = v
            ptr[i, j] = 0 if v == 0 else (1 if v == diag else (2 if v == E[j] else 3))
            if v > best:
                best, bi, bj = v, i, j
        H = H_new
    cols = ident = 0
    i, j = bi, bj
    while i > 0 and j > 0 and ptr[i, j] != 0:
        p = ptr[i, j]
        if p == 1:
            cols += 1; ident += a[i - 1] == b[j - 1]; i -= 1; j -= 1
        elif p == 2:
            cols += 1; i -= 1
        else:
            cols += 1; j -= 1
    return best, cols, ident


def kmers(s, k=3):
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def prefix_cluster(dms_id):
    return "_".join(dms_id.split("_")[:2])


def main():
    cds = json.load((ROOT / "data/cds/dms_cds.json").open())
    ids = sorted({r["dms_id"] for r in
                  csv.DictReader((ROOT / "results/dms_trivial_baselines.csv").open())})
    prot = {i: translate(cds[i]["cds"]) for i in ids if i in cds}
    missing = [i for i in ids if i not in prot]
    print(f"{len(prot)} assays translated" + (f", MISSING {missing}" if missing else ""))

    km = {i: kmers(s) for i, s in prot.items()}
    pairs, skipped = [], 0
    for a, b in itertools.combinations(sorted(prot), 2):
        j = len(km[a] & km[b]) / max(1, len(km[a] | km[b]))
        if j >= JACCARD_MIN:
            pairs.append((a, b, j))
        else:
            skipped += 1
    print(f"{len(pairs)} pairs pass the 3-mer prefilter, {skipped} screened out\n")

    rows, edges, near = [], [], []
    for a, b, j in sorted(pairs):
        score, cols, ident = smith_waterman(prot[a], prot[b])
        pid = ident / cols if cols else 0.0
        cov = cols / min(len(prot[a]), len(prot[b]))
        homolog = pid >= ID_THRESHOLD and cols >= MIN_COLS and cov >= COVERAGE_MIN
        rows.append([a, b, len(prot[a]), len(prot[b]), round(j, 4), int(score),
                     cols, ident, round(pid, 4), round(cov, 4), int(homolog),
                     int(prefix_cluster(a) == prefix_cluster(b))])
        if homolog:
            edges.append((a, b))
            print(f"  HOMOLOG   {pid:6.1%} id over {cols:4d} cols "
                  f"({cov:.0%} of the shorter)  {a} / {b}")
        elif pid >= ID_THRESHOLD and cols >= MIN_COLS:
            near.append((a, b, pid, cols, cov))

    if near:
        print("\n  Rejected on coverage -- identity over a fragment, not a protein:")
        for a, b, pid, cols, cov in near:
            print(f"    {pid:6.1%} id over {cols:4d} cols but only {cov:.0%} of the "
                  f"shorter sequence  {a} / {b}")

    parent = {i: i for i in prot}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    measured = {}
    for i in prot:
        measured.setdefault(find(i), []).append(i)
    pref = {}
    for i in prot:
        pref.setdefault(prefix_cluster(i), []).append(i)

    print(f"\nname-prefix clusters : {len(pref)}")
    print(f"measured clusters    : {len(measured)}  "
          f"(identity >= {ID_THRESHOLD:.0%} over >= {MIN_COLS} aligned columns)")
    disagree = [sorted(v) for v in measured.values()
                if len({prefix_cluster(x) for x in v}) > 1]
    if disagree:
        print("\nPAIRS THE PREFIX KEY MISSES:")
        for g in disagree:
            print("  " + ", ".join(g))
    else:
        print("\nNo cluster merges two different prefixes: on this panel the prefix")
        print("key and the measured clustering give the same units. The shortcut was")
        print("adequate here, which is a fact about this panel, not about the method.")

    out = ROOT / "results/homology_clusters.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["assay_a", "assay_b", "len_a", "len_b", "kmer_jaccard",
                    "sw_score", "aligned_cols", "identities", "percent_identity",
                    "coverage_shorter", "homologous", "same_name_prefix"])
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
