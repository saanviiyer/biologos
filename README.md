# biologos

**What transfers between biological sequence models, and what only looks like it does.**

DNA and protein are two encodings of one molecule, related by a map that is exactly
known. That makes biology an unusually good place to ask a question the rest of
machine learning can only ask loosely: when a model trained on one representation
appears to know something about the other, is that transfer, or is it an artifact
of how the question was asked?

This repository is the measurement. It is not an attempt to build a better model.

## The headline, stated as the evidence supports it

**Genomic-to-protein transfer is a function of scale and architecture, and where it
exists it does not clear a substitution matrix.**

Matched on the same 25 deep mutational scanning assays:

| scorer | mean Spearman | 95% CI |
|---|---|---|
| ESM-2 650M, a protein model | +0.4655 | [+0.395, +0.536] |
| chemistry ridge, leave-one-cluster-out | +0.2860 | [+0.248, +0.324] |
| **Evo 2 7B, a genomic model** | **+0.2658** | [+0.202, +0.330] |
| BLOSUM62 alone | +0.2282 | [+0.195, +0.261] |
| Nucleotide Transformer v2 50M | -0.0132 | [-0.033, +0.007] |

Evo 2 minus BLOSUM62 is +0.0376, interval [-0.012, +0.087], higher in 14 of 25
assays. That interval contains zero. Against five fitted chemistry features Evo 2
is *behind*, at -0.0202.

So a genomic language model at Evo 2 scale does acquire protein-fitness signal.
It is statistically indistinguishable from a substitution matrix, it is beaten by
five hand-chosen chemical properties, and it sits far below a protein model. The
Nucleotide Transformer family carries no signal at all at any scale tested, and its
scaling trend does not reach Evo 2: extrapolated to 7B it predicts +0.024 against
the +0.266 measured, short by a factor of eleven.

## Four things this repository got wrong first

The corrections are the point, so they are listed before the results.

**A reading-frame result was retracted.** A genomic model appeared unable to tell a
gene from the same gene read one nucleotide out of frame. The "frameshift" was a
rotation, which apart from one wrap junction is real genomic sequence read one base
later. Rank the four conditions by whether they produce a real genomic string and
the whole table is reproduced with no appeal to reading frame. The replacement
probe, calibrated against a 3-periodic Markov positive control, turned out to be
underpowered, not null. See section 26 and section 29.

**A representation-level transfer claim was falsified by its own control.** A probe
on genomic embeddings scored +0.141 on unseen residues, which read as weak
transfer. It scores +0.143 after frameshifting and +0.129 on the reverse
complement. A representation invariant to destroying the reading frame is not
protein-level. Under any encoding the codon determines the amino acid, so the
nucleotides at a varying site already are residue identity. See section 17.

**A context dose-response was measuring composition.** Real genomic context made
one model monotonically worse, trend -0.771. Composition-matched shuffled flanks,
carrying no genomic information at all, reproduce it at -0.633. A second model with
a 30 kb window produces a clean *positive* trend of +0.536 that its shuffled twins
fully account for. Context trends on this landscape run in both directions from
noise. See section 35.

**A general claim was too general.** "A genomic model carries no protein-fitness
signal" held for the Nucleotide Transformer family and was written as though it
held for genomic models. Evo 2 refutes it. See section 30.

## What is measured here

| question | answer | section |
|---|---|---|
| Does genomic-LM likelihood predict protein fitness? | No for NT at four scales; yes but sub-baseline for Evo 2 | 18, 20, 30 |
| Is that a scaling phenomenon? | No. The NT trend under-predicts Evo 2 by 11x | 31 |
| Does the genomic representation carry protein-level information? | No. The signal survives frameshift and reverse complement | 17 |
| Does more genomic context help? | Unresolved. Both signs reproduce from shuffled flanks | 19, 35 |
| Does a genomic model memorise its training genomes? | No membership signal at 50M; confounded and open at 500M | 32, 36 |
| How good is a genomic model at DNA itself? | 50M beats an order-5 Markov chain by 2 points; 500M by 13 | 32, 36 |
| Do genomic models beat a lookup table on regulatory tasks? | Mostly no. Two wins under 0.03 AUC against one loss of 0.20 | 34 |

## The measurement rules this repository follows

Each was paid for by a result that had to be withdrawn.

1. **A trivial baseline in every table.** BLOSUM62 reaches 49% of ESM-2. A
   positional one-hot beats both genomic models on splice sites. A 60-parameter
   one-hot beat a 650M protein model under a leaky split.
2. **A noise ceiling before any null.** Synonymous encodings share a label by
   construction and bound what could have been detected. Without one, a null says
   nothing.
3. **A positive control before any negative claim.** A 3-periodic Markov model
   represents reading frame by construction, and calibrating against it turned a
   null into an underpowered non-detection.
4. **A matched control for every intervention.** Shuffled flanks of matched length
   and composition; frameshift against reverse complement; an aperiodic twin at the
   same Markov order.
5. **Correlated units counted once.** 194 ProteinGym assays are 165 independent
   proteins. 29 coding sequences were 24.

## Layout

```
biologos/      glm.py (genomic scorers), coverage.py, biointerp/ (typed interventions)
scripts/       one runner per experiment, plus its analyzer where separable
results/       the tables every number in docs/FINDINGS.md is read from
data/cds/      verified coding sequences, each accepted only on exact translation
docs/          the full findings log for the transfer arm
```

Large inputs are not vendored. ProteinGym v1 comes from the HuggingFace dataset
`OATML-Markslab/ProteinGym_v1` (note that the older `ProteinGym` repository is the
87-assay v0.1 release). Genome sequence is fetched from ENA by
`scripts/build_genomic_context.py`.

## Running it

Genomic-model code needs `transformers==4.44.2`. Version 5 removes
`find_pruneable_heads_and_indices` and the `is_decoder` config attribute that
Nucleotide Transformer v2's bundled remote code reads, and the failure looks like a
missing model, not a version skew.

```bash
python scripts/fetch_cds.py              # verified coding sequences, ENA
python scripts/build_genomic_context.py  # genome and operon context
python scripts/run_dms_transfer.py       # the headline comparison
python scripts/run_transfer_scaling.py   # the scaling analysis
```

## Scope

Evo 2 numbers come from a single run on one GPU host and describe `evo2_7b`, since
`evo2_1b_base` needs Transformer Engine. The regulatory arm covers two small models
at 50M and roughly 1.6M and licenses nothing about larger ones. Every assay used
here has homologs, so the regime where a sequence has no relatives is untested
throughout.

## License

MIT. See [LICENSE](LICENSE).
