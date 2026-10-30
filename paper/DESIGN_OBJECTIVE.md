# Appendix B / second arm: the objective axis

Scoped 15 Oct 2026. Nothing here has been run. Every checkpoint claim carries the
date it was checked and the ones that are still unverified say so.

## The question

Everything this project has measured scores a model by a **masked or autoregressive
likelihood**. Sections 18, 20, 30 and 31 read next-token or masked-marginal
probability; section 17 reads a frozen representation. The paper's claim is
therefore about one family of training objective, and it is written as though it
were about modality.

The question for this arm: **does the asymmetry survive a denoising objective?**

Put plainly. Protein models transfer to protein fitness and genomic models do not
clear a substitution matrix. Both halves were established on models trained to
predict a masked or next token. Discrete diffusion models are trained to denoise at
every corruption level instead, and the claim that is sometimes made for them --
that they learn the distribution rather than a conditional -- predicts that the
transfer picture should change. This arm tests that, and it is the cheapest
remaining way to find out whether the paper's headline is a statement about
modality or a statement about one objective.

## Why this is well-posed where the ladder is not

`DESIGN.md` rejects the scaling account of transfer and then concedes the reason it
cannot do better: Evo 2 differs from the Nucleotide Transformer in architecture,
corpus, tokenizer, objective and token budget at once, and no frozen checkpoint
lets any of those vary alone. That is why the ladder wants 8,000 A100-hours.

This arm does not have that problem, because the objective contrast is already
matched on both sides by the model authors themselves.

| row | masked / AR likelihood | masked diffusion | what is held fixed |
|---|---|---|---|
| protein | ESM-2 650M, 33 layers | DPLM 650M, 33 layers | architecture, parameter count, UniRef corpus family |
| nucleic acid | NT-v2 50M multi-species | D3LM ~55.9M | NT-v2 architecture, parameter count, by the D3LM paper's own construction |

DPLM is published as architecture- and scale-matched to ESM-2 at 150M, 650M and 3B,
and its 650M checkpoint loads through transformers as `EsmForDPLM`, which is the ESM
backbone carrying a diffusion objective. D3LM is published as the NT-v2 architecture
trained with a masked diffusion objective and is compared in its own paper against
an NT-v2 model of the same size. So in both rows the thing that differs is the
objective, and the thing that differs is the thing being measured. This is the
contrast the ladder cannot buy at any price, available for the cost of inference.

**And the two flat Stage 0 axes are what license it.** Residual confounds between
D3LM and NT-v2 are tokenization and corpus. Section 39 measured 3-mer against 6-mer
tokenization at -0.0156, interval [-0.0426, +0.0119]. Section 40 measured corpus
diversity at -0.0054, interval [-0.0246, +0.0131]. Both flat, both with attenuation
ceilings of 0.866 to 0.901 so neither is a power failure. Those two nulls read as
dead ends when they were run. Here they are calibration: they bound how much of any
D3LM-minus-NT difference tokenization or corpus could account for, which is the
reason a difference in this arm can be attributed to the objective at all. The
negative results are load-bearing.

## What is already measured

Two of the four cells are in `results/dms_transfer.csv` and need no new run:

| scorer | mean Spearman | 95% CI |
|---|---|---|
| ESM-2 650M | +0.4655 | [+0.395, +0.536] |
| NT-v2 50M | -0.0132 | [-0.033, +0.007] |
| BLOSUM62, the bar | +0.2282 | [+0.195, +0.261] |
| chemistry ridge, LOCO | +0.2860 | [+0.248, +0.324] |

So the arm is two new scorers, not four, and the comparison it enters is one this
repository has already run twenty-five assays of.

## The checkpoints

**Checked 15 Oct 2026.**

| checkpoint | size | claimed | verified |
|---|---|---|---|
| `airkingbd/dplm_650m` | 650M, 33 layers | masked diffusion, ESM-matched | card confirms size, layers, `EsmForDPLM` loading, 150M/650M/3B siblings |
| `Hengchang-Liu/D3LM-from-nt` | 55.9M | masked diffusion, initialised from NT | listed in the `Hengchang-Liu/d3lm` collection |
| `Hengchang-Liu/D3LM-scratch` | 55.9M | masked diffusion, trained from scratch | listed in the same collection |

**Not yet verified and must be settled in Stage 0 below:** whether weight files are
actually present in the two D3LM repositories, their licenses, D3LM's tokenization
(NT-v2 is 6-mer; single-nucleotide tokenization would confound the objective with
the tokenization axis, which section 39 bounds but does not erase), D3LM's training
corpus, whether its denoiser is conditioned on a timestep, and DPLM's corpus and
token budget, which its card does not state.

The two D3LM checkpoints are a bonus the design should use rather than collapse.
`from-nt` and `scratch` separate the objective from the initialisation at the same
size, so the nucleic-acid row is a three-point ladder:

- NT-v2 50M, masked LM, NT init -- measured, null at -0.0132
- D3LM-from-nt, masked diffusion, NT init -- objective only
- D3LM-scratch, masked diffusion, fresh init -- objective and initialisation

At roughly 56M parameters these are a few hundred megabytes each, not the ~20 GB
per arm that blocked the corpus-breadth axis. This arm runs on the machine as it is.

## Scoring, which is where this arm can most easily go wrong

**One score is admissible and it is the masked marginal.** Mask the token, read
`log_softmax` at the masked position, take the mutant-minus-wildtype difference.
That is what `esm_scores()` and `nt_token_logprobs()` in
`scripts/run_dms_transfer.py` already compute, and it is available unchanged from a
masked diffusion model, because the per-position denoiser output at a given
corruption level *is* a masked marginal. DPLM loading as an ESM backbone means
`esm_scores()` should apply with no change beyond the model class, and D3LM being
NT-v2 architecture means `glm.NTScorer.token_index`, `_masked_logprobs` and the
`_patch_transformers_for_nt_v2` shim should carry over.

**The ELBO is not admissible and must not be substituted for it.** A diffusion
model's natural score is a variational bound, and a bound's tightness is a property
of the model, not of the sequence. Comparing ESM-2's exact masked marginal against
DPLM's ELBO would compare a likelihood with an upper bound on a different
likelihood and attribute the gap to the objective. The published literature is
explicit that the masked-diffusion ELBO is loose enough that recent work recovers
20 to 31 percent of the gap to exact perplexity, and that the ELBO is invariant to
the unmasking rule, so it cannot even rank samplers. If a cell admits no masked
marginal, **the cell is reported as not runnable, never as a null.**

**Two degrees of freedom a masked LM does not have, both pre-registered rather than
averaged away:**

1. **Timestep conditioning.** If a denoiser takes `t`, the masked marginal is a
   function of it. The headline fixes `t` at the value corresponding to a single
   masked token out of the full sequence, which is the setting that matches the
   existing protocol, and a sweep over `t` is reported beside it as a sensitivity
   band. A score that moves materially across `t` is reported as a band, not
   collapsed to a mean.
2. **Mask rate.** Single-token masking is in distribution for ESM-2, trained at
   roughly 15 percent, and at the edge of the training distribution for a diffusion
   model trained across all corruption levels. This asymmetry favours the masked LM
   and must be stated, not hidden. The headline stays at single-mask for
   comparability with the twenty-five-assay protocol; a mask-rate sweep is reported
   alongside, and if a diffusion cell clears the bar only at high mask rate, that is
   the finding and it is reported as conditional on mask rate.

Neither of these is a nuisance parameter. They are the two places where "the same
protocol" stops being the same thing across objectives, and declaring them in
advance is what keeps the comparison from being chosen after the fact.

## Pre-registered gates

Each is a stop condition, written before any run. G1 to G5 are `DESIGN.md`'s,
restated so this arm is self-contained; G6 is new and specific to diffusion.

- **G1, power.** Every model's synonymous attenuation ceiling must exceed 0.80,
  from `noise_ceiling` over synonymous fibres. A model whose within-fibre spread
  swamps its between-variant spread cannot be read and its cell is reported as
  uninformative, never as a null. The four NT-family models already run land at
  0.866 to 0.901, so this gate is expected to pass, but it is computed per model and
  not inherited.
- **G2, the bar.** The comparison is against BLOSUM62 at +0.2282 and the fitted
  chemistry ridge at +0.2860, not against zero and not against ESM-2. A cell shows
  transfer only if its paired difference against BLOSUM62 excludes zero at the
  cluster level, over the 21 clusters measured in section 42, via
  `paired_contrast(..., clusters=...)`.
- **G3, the map control.** Any nucleic-acid cell clearing G2 is re-scored
  frameshifted and reverse-complemented through `map_control`. A cell whose signal
  survives either is codon leakage and is reported as such. Section 17 is the
  precedent and the reason this gate exists: a probe scored +0.141 canonical, +0.143
  frameshifted and +0.129 reverse-complemented, and a representation invariant to
  destroying the reading frame is not protein-level. **No diffusion-based scorer has
  ever been given this control.** It is the single most likely place for this arm to
  produce a correction rather than a result.
- **G4, codon marginalisation.** Any nucleic-acid cell clearing G2 and G3 is
  re-scored marginalised over synonymous codons, via the existing
  `glm.synonymous_ensemble`. The marginalised score is the headline, and it must be
  an actual marginal: log-sum-exp over the codon fibre in *both* numerator and
  denominator. **Do not inherit the `rho_nt_codon_marginalised` column**, which
  averages log-ratios and is therefore a geometric mean of codon likelihoods rather
  than a marginal; `scripts/run_codon_marginal_audit.py` measures the difference and
  `results/codon_marginal_audit{,_ladder}.csv` record it. On the NT ladder the two
  specifications differ by +0.001 to +0.011, wider than the within-species corpus
  contrast the paper reports, they agree across assays at only +0.59, and the
  parameter-count monotonicity holds under one and not the other. Any cell whose
  headline is a mean of logs has not cleared this gate. A second trap the NT arm
  paid for: resolve codon tokens against the vocabulary explicitly and count the
  misses, rather than falling back to `unk_token_id`, which silently scored 369
  absent tokens across the ladder.
- **G5, no trend without a control.** No trend over objective, scale or mask rate is
  reported without a matched control twin, per section 3.3 of the paper.
- **G6, the score is the same score.** Every cell in a table must be scored by the
  masked marginal at the pre-registered `t` and mask rate. A cell scored any other
  way does not enter the table. This gate is what stops the arm from becoming a
  comparison of scoring conventions wearing the clothes of a comparison of
  objectives.

## Outcomes, and the modal one named in advance

Four, and each is reportable.

1. **The objective is flat in both rows.** DPLM lands on ESM-2 and D3LM lands on
   NT-v2, both intervals containing zero. The paper's headline generalises from one
   objective to two and the modality statement is strengthened, not weakened.
2. **The objective matters where transfer exists and not where it does not.** DPLM
   separates from ESM-2 but D3LM stays below BLOSUM62. Then denoising changes how
   well a model reads a landscape it can already read, and buys nothing where the
   signal was absent. This is the most interesting outcome and the one with a
   mechanism worth arguing about.
3. **A nucleic-acid diffusion cell clears BLOSUM62.** Then G3 and G4 decide whether
   it is transfer or codon leakage, and section 17 says what the prior should be.
   Either way it is the first time a denoising objective has been put through the
   map control, and both branches are publishable.
4. **The arm is not runnable.** D3LM has no usable weights, no accessible
   per-position marginal, or a tokenization that confounds the objective beyond what
   sections 39 and 40 bound. Then this is reported as a feasibility note, the
   nucleic-acid row is declared empty, and the protein row runs alone as a
   matched-pair objective contrast, which is still a result.

**Outcome 1 is the modal one on current evidence and must be acceptable before
starting.** NT-v2 50M sits 0.24 below BLOSUM62 with a healthy ceiling, so the
nucleic-acid row's prior is null-on-null and the honest expectation is that
denoising does not rescue it. If a flat result is not acceptable, this arm is a bet
on a positive and should not be run.

## Budget

Inference only. Four scorers over 25 assays and the single-mutant subset of each,
plus the synonymous ensembles that G1 and G4 need. DPLM 650M is the same cost as the
ESM-2 650M pass already run; the two D3LM checkpoints are smaller than any model in
the existing tables. The `t` and mask-rate sweeps multiply the diffusion cells by
the number of swept values and nothing else.

Call it a day of GPU time and a few gigabytes of download, against the ladder's
8,000 A100-hours.

**Updated 26 Oct 2026: the ladder is closed.** Section 46 ran the corpus-breadth
axis and it came back flat, which makes all three axes separable without training
null and settles the funding question on DESIGN.md's own pre-registered gate. That
changes this arm's standing rather than its design: it is no longer the cheaper thing
to do before a five-figure compute decision, it is **the only open axis of the
paper's generality left**. The paper's claim rests on one family of training
objective, and whether it survives a denoising objective is now the question that
has not been asked rather than one of two.

## Stage 0, to be run before any scoring

Mirroring `DESIGN.md`'s staging, so the arm can fail in an afternoon.

1. Confirm weight files, config and license in both D3LM repositories. A collection
   listing is not a released checkpoint.
2. Read D3LM's tokenization and corpus out of the paper and the configs. 6-mer and
   the NT multi-species corpus means sections 39 and 40 bound the confound.
   Single-nucleotide tokenization or a different corpus means the confound is
   unbounded and must be declared in every table.
3. Establish whether either denoiser is conditioned on a timestep, and fix the
   headline `t` before scoring anything.
4. Score one assay with each new checkpoint and compute its attenuation ceiling. G1
   on one assay costs minutes and decides whether the cell can be read at all.
5. Verify that the existing masked-marginal code paths carry over, and that the
   `transformers==4.44.2` pin holds for D3LM as it does for NT-v2. Version 5 removes
   `find_pruneable_heads_and_indices` and the `is_decoder` config attribute that
   NT-v2's bundled remote code reads, and the failure presents as a missing model
   rather than a version skew. A model carrying the NT-v2 architecture should be
   assumed to inherit this.

If step 1 or 4 fails, outcome 4 is the result and the protein row runs alone.

## Safety posture, stated because this arm changes the inputs

Every model in the paper's transfer tables is a scorer. This arm is the first to
load checkpoints trained to **generate sequence**, and two of the candidates are
models whose headline application is designing DNA with tunable regulatory
activity.

That is the accurate form of the claim, and an earlier draft of this section
overstated it. `biologos/boltz.py` already drives Boltz, a diffusion model, and
passes `--diffusion_samples` and `--sampling_steps` to it, so the repository has
run a generative model before. Boltz generates **structure conditioned on a given
sequence** and proposes no sequence, which is why it raises none of what follows;
D3LM and DPLM generate sequence itself. The distinction is between the two, not
between this arm and a repository that had never loaded a generative model.

The rule this arm runs under, which is the rule every other section of this
repository already satisfies by construction:

**Score sequences that already exist. Do not use a model to produce sequences that
do not.**

Concretely, for this arm:

- **Likelihood evaluation only.** Masked marginals at fixed positions in sequences
  that already exist. No sampling, no denoising to completion, no conditional
  generation, no infilling.
- **No optimisation toward a fitness objective**, and no fine-tuning on assay
  labels. A model fitted to a landscape and then used to rank unmeasured variants is
  a design tool, not a measurement.
- **No new sequence data.** The inputs are the ProteinGym panel and the verified
  coding sequences in `data/cds/`, admitted only on exact translation. This arm adds
  checkpoints, not organisms and not sequences.
- **Aggregate outputs only**: per-assay correlations, intervals, ceilings and the
  control tables, which is what every other section produces.

One qualification, because the unqualified version would be false.
`biologos/landscape.py` is a data loader and a noise model, and its docstring points
at `crosstalk.solve`, which computes exact regret. So no search runs *in this
repository*, but the module does feed one next door, over a fully enumerated
three-site landscape whose fitness values are published measurements. Saying "this
project contains no optimisation over sequences" without that qualification would
misdescribe the tree.

The failure mode to name in advance, because it will be proposed as a natural
extension: a cell that clears G2 invites sampling from that checkpoint to see what
it generates, and a cell that fails invites fine-tuning it on the assay. Both cross
the line above and neither is needed for any of the four outcomes. **The arm is
answerable entirely by scoring existing sequences, and that is the reason it is
worth running in this form.** If a result here makes generation look informative,
the correct response is to write down why and stop, not to generate.

**On the review question, settled 26 Oct 2026.** An earlier version of this section
deferred to a `biologos/SAFETY.md` written against HARNESS.md B1 to B7, and blocked
this arm behind a faculty-advisor disclosure that gate B7 requires. That file has
been deleted and the block lifted, for a reason worth recording so it is not
reintroduced: **HARNESS.md's B-gates are scoped to safeguard auditing.** That
supplement opens by saying every gate above it assumes you can hold the positive
class and that in biosecurity you cannot -- the problem of probing a screener when
the material defining a correct flag is material you must not acquire. `shibboleth`
has that problem and its protocol is its method. This project has no positive class:
it scores public, published assay data and its outputs are correlations. The tell was
that B1's own input categories had to be rewritten to apply here at all, which is
evidence a gate does not fit rather than licence to redefine it.

There is also no faculty advisor on this project, so B7 could not have been met as
written and the blank it left behind was asserting a review that was never going to
happen. The author is the only reviewer, and that is stated here rather than
concealed behind a filled-in field.

What survives from that exercise is the rule above and one correction below. The two
questions the biosafety gate declined to answer in writing -- whether this boundary
holds against what the code paths make available once generative weights are local,
and what unrefused extensions the arm invites -- remain unanswered by anyone. The
gate's reasoning for refusing was that writing them down produces a map from a
likelihood-only posture to generation from a regulatory-DNA design model, which
fails HARNESS.md B3: delete every result, keep the artifact, ask whether the artifact
alone lowers the cost of misuse. That reasoning is accepted and the enumeration has
not been produced anywhere in this repository. It is a known gap, not a cleared one.

## What this does not fix

The protein row is matched at 650M and the nucleic-acid row at roughly 56M, so the
two rows are not matched to each other and no cross-row difference can be read as
modality at fixed scale. The arm measures the objective **within** each modality and
nothing else. Reading it as a protein-versus-nucleic-acid comparison would repeat
exactly the confound that `DESIGN.md` rejects the scaling account for.

Masked diffusion is also one objective family among several. Order-agnostic
autoregressive models such as EvoDiff-Seq, and the score-entropy and
continuous-latent DNA models, are not covered, and the first of those needs
averaging over decoding orders whose Monte-Carlo variance would have to enter the
noise ceiling before any null could be read. Those are a later arm, if this one
finds anything.

And D3LM at 56M cannot speak for a nucleic-acid diffusion model at Evo 2's scale. A
null throughout this arm stays consistent with a positive at 7B, for the same reason
the ladder's box does not close the Evo 2 observation.
