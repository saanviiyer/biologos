#!/usr/bin/env python3
"""Re-fetch the contigs the memorisation run scored, from their accessions.

`scripts/memdna_data.py` and `data/genomes/` are not in this repository, so
`run_memorisation_dna.py` cannot be run and the Markov and composition baselines
it needs could not be computed. The accessions survive anyway: every row of
`results/memorisation_dna_nt*.csv` carries the contig it was drawn from, so the
genome set is recoverable from the results themselves.

This fetches each distinct contig from NCBI efetch into `data/genomes/`, one
FASTA per accession, skipping any already on disk. NCBI allows three requests a
second without a key, so the pace below is deliberate and the whole set takes a
few minutes.

  python3 scripts/fetch_memdna_genomes.py

The synthetic arms (mono, dinuc, markov5) carry contig "-" and are not fetched:
their sequences were generated at run time and are not recoverable. Baselines
for those arms stay undefined, which is correct -- a composition baseline on a
composition control is not a quantity anyone should quote.
"""
import csv
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "genomes"
EFETCH = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
          "?db=nuccore&id={}&rettype=fasta&retmode=text")
PAUSE = 0.4           # under NCBI's 3/s limit for unauthenticated clients


def bare(contig):
    """The accession efetch wants, from the identifier the CSV happens to carry.

    The 2025-26 arm was assembled from ENA and its contigs are recorded in ENA's
    compound form, "ENA|CP166784|CP166784.1". efetch wants the last field alone.
    The member arm carries plain RefSeq accessions and passes through.
    """
    return contig.rsplit("|", 1)[-1] if contig.startswith("ENA|") else contig


def accessions():
    """Every distinct (contig -> genome) the two reconstruction runs scored.

    Keyed by the accession fetched, so the file on disk is named for what was
    requested; the CSV's own spelling is recovered by bare() when reading back.
    """
    seen = {}
    for f in ("results/memorisation_dna_nt50m.csv",
              "results/memorisation_dna_nt500m.csv"):
        for r in csv.DictReader((ROOT / f).open()):
            if r["contig"] not in ("-", ""):
                seen.setdefault(bare(r["contig"]), r["genome"])
    return dict(sorted(seen.items()))


def fetch(acc, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(EFETCH.format(acc), timeout=90) as r:
                txt = r.read().decode()
            if txt.startswith(">") and len(txt) > 200:
                return txt
            raise ValueError(f"short or non-FASTA reply ({len(txt)} bytes)")
        except (urllib.error.URLError, ValueError, TimeoutError) as e:
            if i == tries - 1:
                return None
            time.sleep(2 ** i)
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    accs = accessions()
    print(f"{len(accs)} contigs across {len(set(accs.values()))} genomes")
    done = failed = 0
    for i, (acc, genome) in enumerate(accs.items(), 1):
        dest = OUT / f"{acc}.fa"
        if dest.exists() and dest.stat().st_size > 200:
            done += 1
            continue
        txt = fetch(acc)
        if txt is None:
            print(f"  [{i:3d}/{len(accs)}] FAILED {acc} ({genome})")
            failed += 1
            continue
        dest.write_text(txt)
        done += 1
        nt = sum(len(l.strip()) for l in txt.splitlines() if not l.startswith(">"))
        print(f"  [{i:3d}/{len(accs)}] {acc:24s} {genome:16s} {nt:>9,} nt",
              flush=True)
        time.sleep(PAUSE)
    print(f"\n{done} on disk, {failed} failed -> {OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
