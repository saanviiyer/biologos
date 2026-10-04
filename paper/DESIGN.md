# Appendix A / successor project: the pretraining ladder

## Why frozen checkpoints cannot answer the question

The paper rejects the scaling account of genomic-to-protein transfer: the
Nucleotide Transformer trend predicts +0.024 at 7B against Evo 2's measured
+0.266, short by 11x. Rejecting that account is all the evidence supports, because
Evo 2 differs from the Nucleotide Transformer in architecture, corpus, tokenizer,
objective and token budget at once, and no frozen checkpoint lets any of those be
varied.

This is structural. Public releases fix tokens and corpus at one value per model.
Evaluating more checkpoints adds points on the parameter axis and none on the
others. A law of the form L(N, D, corpus) is not estimable from released models
however many are scored.

**Checked, 4 Oct 2026.** Neither family publishes intermediate-step checkpoints.
Every Nucleotide Transformer and Evo 2 repository on HuggingFace carries a single
`main` branch, no tags and one weight file, and the `savanna_evo2_*` repositories
are native-format copies of the same finals. The token axis is closed and the
ladder must buy it.

**But three axes are already open and free**, which shrinks Stage 1 and must be run
before any GPU is requested:

| contrast | models | what it isolates |
|---|---|---|
| tokenizer | NT-v2 50M 6-mer vs `nucleotide-transformer-v2-50m-3mer-multi-species` | vocabulary only: identical 512 hidden, 12 layers, 16 heads, same 850-genome corpus, vocab 4107 vs 75 |
| corpus diversity | NT 500M `human-ref` vs `500m-1000g` | one reference genome vs 3,202 human genomes, identical 1280/24/20 |
| corpus breadth | NT 2.5B `1000g` vs `2.5b-multi-species` | human vs 850 species, identical 2560/32/20 |
| parameters at fixed corpus | NT `500m-1000g` vs `2.5b-1000g` | size alone |
| training stage | `evo2_7b_base` vs `evo2_7b` | midtraining and context extension |

Running these costs inference only. They cannot settle the token axis, but they
bound how much of the 11x Evo 2 gap corpus and tokenization can account for, and a
corpus gradient here is the signal that justifies Stage 2. The 3-mer model is the
highest-value single run: Boshar et al. (2024) trained it specifically because
codon-level resolution was where genomic models lost to protein models, and it has
never been scored zero-shot.

## The question the ladder answers

Does genomic-to-protein transfer emerge at any reachable (N, D, corpus), and if so
which axis buys it?

Three outcomes, all publishable:

1. **Transfer emerges with coding density.** Models trained on coding-rich corpora
   transfer and matched models on genome-wide corpora do not, at equal N and D.
   This would say Evo 2's signal is corpus, and would give practitioners a recipe.
2. **Transfer emerges with N or D along a fittable curve.** Then there is a law and
   a crossover point, and the paper becomes the ATLAS-shaped result.
3. **Transfer stays at the BLOSUM62 bar everywhere reachable.** Then the strongest
   version of the negative result is established causally, not by extrapolation,
   and the 11x gap is attributed to something outside the reachable box.

Outcome 3 is the modal one on current evidence and must be acceptable before
starting, or the project is a bet on a positive.

## Design

**Axes.** Fully crossed except where noted.

| axis | levels | note |
|---|---|---|
| parameters N | 14M, 44M, 140M, 440M | four points, ~0.5 decade apart |
| tokens D | 2B, 8B, 32B | three points, Chinchilla-ish spread |
| corpus composition | CDS-only, genome-wide, 50/50, multi-species genome-wide | four |
| seeds | 2 at the smallest two N, 1 above | variance estimate where it is cheap |

Architecture is held fixed, a single decoder-only transformer with single-
nucleotide tokenization, so that architecture is not confounded with corpus the way
it is between NT and Evo 2. Tokenization is single-nucleotide because 6-mer
tokenization is itself a candidate explanation and must not be entangled.

**Run count.** 4 x 3 x 4 = 48 cells; with the seed schedule, 72 runs. Reserve 20%
for restarts and a held-out axis extension, giving ~86 runs.

**Held-out evaluation axes**, declared before training, mirroring the ATLAS
protocol: R^2(N) on the largest N, R^2(D) on the longest token budget, R^2(C) on
the largest compute, R^2(corpus) on the 50/50 mixture never seen during fitting.

**Evaluation.** The paper's protocol unchanged: 25 DMS assays, masked-marginal or
autoregressive log-likelihood on the coding sequence, cluster-level statistics over
21 clusters, synonymous attenuation ceiling per model, BLOSUM62 and the fitted
chemistry ridge in every table.

## Pre-registered gates

Each gate is a stop condition, written before any run.

- **G1, power.** Every model's attenuation ceiling must exceed 0.80. A model whose
  synonymous spread swamps its between-variant spread cannot be read, and its cell
  is reported as uninformative, never as a null.
- **G2, the bar.** The comparison is against BLOSUM62 at +0.228, not against zero
  and not against ESM-2. A cell "shows transfer" only if its paired difference
  against BLOSUM62 excludes zero at the cluster level.
- **G3, the map control.** Any cell clearing G2 is re-scored frameshifted and
  reverse-complemented. A cell whose signal survives either is codon leakage and is
  reported as such, not as transfer. This gate is why the ladder is worth running:
  it is the one check Evo 2 has never been given.
- **G4, codon marginalisation.** Any cell clearing G2 and G3 is re-scored
  marginalised over synonymous codons. The marginalised score is the headline.
- **G5, no trend without a control.** No trend over N, D or corpus is reported
  without the matched-control twin that Section 3.3 of the paper requires.

## Budget

Token-parameter products, at roughly 6ND FLOPs:

| N | D=2B | D=8B | D=32B |
|---|---|---|---|
| 14M | 0.17 PF-days | 0.67 | 2.7 |
| 44M | 0.53 | 2.1 | 8.4 |
| 140M | 1.7 | 6.7 | 27 |
| 440M | 5.3 | 21 | 84 |

Summed over the 48 cells and the seed schedule: roughly **250 PF-days**, which is
about **6,000 A100-hours** at 40% MFU, call it 8,000 with restarts and evaluation.
At commodity spot pricing that is a few thousand dollars; on an institutional
cluster it is a medium allocation request. This is the honest number and it is two
orders of magnitude below ATLAS, which trained to 8B.

**What the budget buys that money cannot otherwise buy:** causal control over the
three axes that frozen checkpoints confound. Nothing about evaluating more public
models substitutes.

## Staging, so the project can fail cheaply

**Stage 0, RUN 4 Oct 2026. Two of three axes null; the gate is not passed.**

| axis | contrast | paired difference | verdict |
|---|---|---|---|
| tokenization | NT-v2 50M 6-mer vs 3-mer | +0.0156 [-0.0119, +0.0426] | flat |
| corpus diversity | NT 500M human-ref vs 1000G | +0.0054 [-0.0131, +0.0246] | flat |
| corpus breadth | NT 2.5B 1000G vs multi-species | not run, disk | **open** |

All four models land 0.225 to 0.257 below BLOSUM62 with attenuation ceilings of
0.866 to 0.901, so none of these is a power failure. The 2.5B pair needs ~20 GB of
free disk and the machine had 138 MB.

**The gate as written said to make the case explicitly before proceeding, so here
it is: do not fund the ladder yet.** Two of the three factors separable without
training are flat. If breadth is flat too, every reachable axis is null and the
entire argument for 8,000 A100-hours rests on the token budget alone. That is a
single unmeasured axis carrying a five-figure compute request, and the honest move
is to stop at the measurement-validity paper, which is now stronger for carrying
three controlled negative contrasts.

**Run breadth first.** It costs one download and it is the most informative of the
three, since Evo 2 trains across all domains of life and cross-species range is the
most plausible surviving account of its margin. A gradient there reopens the ladder;
a flat result closes it. No decision on the GPU spend should be made before it.

**Stage 1, ~400 A100-hours.** The 14M and 44M rows only, all corpora, D=2B and 8B.
If no cell clears G2 and the corpus axis shows no gradient, the modal outcome is
confirmed at 5% of the budget and the result is written as the causal version of the
negative. Stop here and the paper is still stronger than the current one.

**Stage 2.** Extend to 140M and D=32B only if Stage 1 shows a corpus gradient or an
N trend whose extrapolation reaches +0.228 within the box.

**Stage 3.** The 440M row, only to pin a crossover that Stage 2 located.

This staging is the point. The ATLAS form is reachable only in the branch where
Stage 1 finds something, and Stage 1 is cheap enough to settle that in a week.

## What this does not fix

Evo 2 is 7B and the ladder tops out at 440M, so a positive at Evo 2 scale and null
throughout the box remains consistent with both an emergent threshold and a corpus
difference. The ladder narrows the explanation; it does not close it. Saying so in
the paper is better than implying a box this size settles a 7B observation.
