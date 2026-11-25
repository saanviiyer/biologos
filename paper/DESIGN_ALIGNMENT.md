# Appendix C / third arm: the alignment axis

Scoped 24 Nov 2026. Nothing new here has been run. What *has* been run is in
section 17 of `docs/FINDINGS.md` and is restated below with its limits, because
the gap between it and a claim about representational convergence is the whole
content of this document.

## The question

`main.tex` now frames the paper against the Platonic Representation Hypothesis
(Huh et al., 2024): models trained on different modalities converge toward a
shared representation of reality, and alignment rises with capability. The
introduction argues that biological sequence models are its sharpest test, since
DNA and protein are two encodings of one molecule under an exact map rather than
correlated observations of an inferred reality.

The paper then does not measure alignment. It measures transfer, which is a
consequence of alignment and not the thing itself. This arm closes that gap.

**The question for this arm: does cross-modal alignment between a genomic and a
protein representation exceed what a trivial encoding of the same molecule
produces, does it rise with capability, and does it survive destroying the map?**

The third clause is the contribution. The convergence literature reads alignment
against a random or shuffled baseline. No published alignment measurement has
been asked whether it survives a manipulation that preserves surface statistics
and destroys the correspondence, because in vision and language no such
manipulation is constructible. Here it is one line.

## What is already measured, and why it is not yet an answer

Section 17 of `docs/FINDINGS.md`, from `results/transfer_analysis.csv`, rows with
`stage=alignment`:

| pair | linear CKA | held-out R² |
|---|---|---|
| NT → ESM-2 | 0.663 | +0.412 |
| **one-hot → ESM-2** | **0.819** | **+0.553** |
| one-hot → NT | 0.713 | +0.533 |

A 60-parameter one-hot residue encoding aligns with ESM-2 better than the genomic
model does. That is the alignment-space analogue of BLOSUM62 clearing the genomic
score, and it is the right shape of result for this arm. It is not yet a claim
about convergence, for five separable reasons.

1. **It is not the metric the hypothesis is stated in.** Huh et al. measure
   mutual *k*-nearest-neighbour overlap between kernels. Linear CKA is a global,
   linear similarity and answers a different question. A follow-up literature
   already argues that what converges is local neighbourhood structure rather
   than global geometry, so a CKA-only result invites the reply that we measured
   the wrong thing and it would be correct.
2. **It is one model at one size.** The hypothesis's central prediction is that
   alignment rises with capability. A single NT arm cannot test a trend, and the
   paper's own finding that NT-v2 50M beats an order-5 Markov chain by +0.018
   hands any defender the reply that an incompetent model has not converged to
   anything yet.
3. **It has no intervals and no unit accounting.** Three bare numbers. The paper
   refuses to read a bare number anywhere else.
4. **The map control was never applied to it.** `stage=causal_control` exists for
   the probe (+0.141 canonical, +0.143 frameshifted) and not for the alignment.
   The one control that makes this arm novel is the one not yet run on the
   quantity it is about.
5. **The choice of datapoints is undeclared.** Alignment is a property of a point
   set. Variants within a protein and distinct proteins are different questions
   and the existing rows do not say which was taken.

## Design

**Headline point set: distinct proteins.** One vector per protein, from the
wild-type sequence, across the 165 protein clusters behind the 194-assay panel.
This is the analogue of the hypothesis's own construction, where points are
distinct items in a dataset. Correlated-unit accounting applies as it does
everywhere else in this paper: 194 assays are 165 clusters and 29 effective
units, and the interval is taken at the cluster level.

**Second point set, reported beside it: variants within a protein.** Pooled
across the panel, one alignment per protein, averaged at the cluster level. This
asks whether the two models agree about the *local* geometry of a single
protein's mutational neighbourhood, which is the quantity the Aristotelian
reading of the hypothesis says actually converges, and it is also the geometry
that would have to agree for transfer to work. Neither set is the other's
robustness check; they are two questions and both get reported.

**Arms.** Protein side: ESM-2 650M, which the paper already uses. Genomic side:
NT-v2 at 50M, 100M, 250M and 500M, plus Evo 2 7B. Trivial side: one-hot residue
identity, and a *k*-mer composition vector over the coding sequence. Evo 2 is the
only arm that needs a GPU host and it is the arm the trend rests on.

**Manipulations, applied to the genomic side only.** Canonical, frameshift +1,
reverse complement, synonymous scramble. `scripts/run_transfer_analysis.py`
already implements all four and `nt_embed` already produces the representation
they act on, so this is a loop over existing functions.

## Scoring, which is where this arm can most easily go wrong

**Mutual *k*-NN is the headline and CKA is reported beside it.** The headline
must be stated in the metric the hypothesis is stated in. CKA stays because it is
already measured and because a disagreement between the two is informative about
whether local or global structure is what fails.

**Both are sensitive to *k* and to *n*, and neither gets a single value.**
Koepke et al. (2026) report that measured alignment degrades substantially as the
point set grows from ~1K to millions, which means any single-*n* number is a
statement about the *n* it was taken at. Pre-register a *k* sweep over
{1, 3, 5, 10, 20} and report alignment against *n* as a curve rather than a
point. A result that holds at one *k* and not the others is reported as such.

**The trivial baseline is composition, not randomness.** One-hot residue identity
and *k*-mer composition. A random projection is not a control here for the same
reason a random steering vector is not: both representations encode the same
molecule's composition, so some alignment is guaranteed for reasons that have
nothing to do with convergence, and the question is only whether the models beat
that floor.

**The noise ceiling comes from the same place it comes from everywhere else in
this paper.** Rewrite each coding sequence with different synonymous codons and
recompute the genomic embedding. A protein-level representation is invariant to
that; the spread of alignment across synonymous encodings of the same protein
bounds how much alignment is readable at all. Report it as κ, as elsewhere, and
refuse to call a cell null below the threshold in G1.

**Pooling is a declared choice, not a default.** `nt_embed` mean-pools the last
hidden layer. Mean pooling over a coding sequence is close to a composition
statistic by construction, which biases *toward* the trivial baseline's
explanation. Run one alternative — the best single layer by the paper's existing
selection rule — and report both. If the conclusion turns on pooling, say so.

## Pre-registered gates

Each is a stop condition, written before any run. G1 to G3 are `DESIGN.md`'s,
restated so this arm is self-contained; G6 to G8 are new and specific to
alignment.

- **G1, power.** Every arm's synonymous attenuation ceiling must exceed 0.80. An
  arm whose alignment varies more across synonymous encodings of one protein than
  across different proteins cannot be read, and its cell is reported as
  uninformative, never as a null.
- **G2, the bar.** The comparison is against the better of one-hot and *k*-mer
  composition, not against zero and not against a random projection. A cell shows
  alignment only if its paired difference against that baseline excludes zero at
  the cluster level.
- **G3, the map control.** Any cell clearing G2 is re-measured frameshifted and
  reverse-complemented. **Pre-registered prediction: if the alignment reflects
  convergence it must fall; if it holds it was composition.** A cell whose
  alignment survives either manipulation is reported as composition, never as
  convergence. This gate is the reason the arm is worth running.
- **G6, the metric is the stated metric.** The headline is mutual *k*-NN. No
  claim about the hypothesis is made from CKA alone, and no claim is made from a
  single *k* or a single *n*.
- **G7, a trend needs the ladder, not two points.** The capability prediction is
  tested across all four NT rungs plus Evo 2, by the paper's existing
  permutation test over orderings, and is reported with the floor that test can
  attain at five points. Two points are a line and not a trend, and this paper has
  already been wrong once about a monotone ordering that belonged to a scoring
  convention rather than to a family.
- **G8, no asymmetry claim without both directions.** Alignment is symmetric;
  transfer is not. If the paper says genomic-to-protein transfer fails while
  protein-to-genomic alignment holds, that asymmetry is a claim and needs the
  reverse ridge map as well as the forward one.

## What a result looks like, in advance

Three outcomes and what each licenses.

1. **Alignment is at or below the composition floor at every size.** The arm
   confirms and sharpens the existing CKA result in the metric the hypothesis is
   stated in, and the paper can say that in the one setting where the shared
   reality is exactly known, measured convergence does not exceed composition.
   This is the outcome the current evidence points to.
2. **Alignment clears the floor and falls under the map control.** Convergence is
   real and bounded, and the paper reports how much of it the exact map accounts
   for. This would be the most interesting outcome and the paper should want it.
3. **Alignment clears the floor and survives the map control.** The alignment is
   composition that the floor failed to capture, and the honest report is that
   our own baseline was too weak, not that convergence was found.

No outcome licenses a claim about convergence outside biology. The arm tests one
hypothesis in the setting most favourable to it.

## Budget

Inference only. No training, no fine-tuning, no sampling.

- NT ladder embeddings over 165 wild-type sequences, four sizes, four
  manipulations: small, runs on the machine as it stands.
- Variant-level embeddings over the 25-assay panel at four sizes: the dominant NT
  cost, and still inside what `run_transfer_analysis.py` already does per arm.
- ESM-2 650M embeddings: already computed for the existing rows.
- Evo 2 7B: the only GPU-host cost, one pass per manipulation over 165 sequences
  plus the variant panel. This is the arm that carries G7 and it should be run
  last, after the NT ladder has shown whether there is a trend to extend.

Staging so it can fail cheaply: run G1 on one protein first. If the synonymous
attenuation ceiling is below 0.80 for mean-pooled NT embeddings, this arm is not
measurable in that representation and the pooling alternative has to come before
anything else.

## Biosafety

Embedding extraction and similarity measurement over sequences that already
exist, aggregate outputs only. No sampling, no denoising to completion, no
conditional generation, no optimisation toward any objective, no per-instance
release. Consistent with the standing rule in `DESIGN_OBJECTIVE.md`: score
sequences that already exist, do not use a model to produce sequences that do
not. Nothing in this arm produces a sequence.

## What this does not fix

The token budget is still closed, since Evo 2 and the Nucleotide Transformer
publish one checkpoint each. Alignment measured on frozen checkpoints cannot
separate architecture from corpus from token budget any better than transfer can,
so G7's trend is a statement about released models and not about scale as a
variable. The arm also tests the hypothesis only where the map is exact, which is
the setting it is most likely to hold in and the one least like the vision and
language settings the hypothesis is usually argued on. A negative result here is
therefore stronger than a negative result elsewhere would be, and a positive one
is weaker.
