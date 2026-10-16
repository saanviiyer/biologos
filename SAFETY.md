# Safety protocol

Drafted 15 Oct 2026, from the template in `shibboleth/SAFETY.md` and the
biosecurity supplement in `RESEARCH/HARNESS.md` (gates B1 to B7). **This file is
not yet in force: the advisor field under "Institutional review" is unfilled, and
HARNESS.md B7 is unmet until it is filled and the disclosure has happened.**

This project measures what transfers between biological sequence models. It is not
an attempt to build a better model and it designs nothing. That makes the protocol
short, but it does not make it optional, because the arm scoped in
`paper/DESIGN_OBJECTIVE.md` would be the first to load checkpoints trained to
generate sequence, and the boundary has to exist before those weights are on the
machine rather than after.

## The governing rule

**Non-uplifting measurement: a model may be evaluated only by reading its scores on
sequences that already exist, never by using it to produce sequences that do not.**

Every run in this repository to date satisfies this by construction. A masked
marginal, a pseudo-likelihood, a frozen embedding and a k-mer entropy are all
readings of a given string. The rule is what separates them from the obvious next
experiment in each case, and it is the reason this work is publishable in the open.

## Admissible inputs

Every input must fall in one of four classes. Anything else is out of scope
regardless of how informative it would be.

- **Class P, published assay data.** ProteinGym v1 (`OATML-Markslab/ProteinGym_v1`),
  and the ParD3 antitoxin specificity landscape of Lite et al., *eLife* 2020, GEO
  GSE153897. Public, peer-reviewed deep mutational scanning on characterised
  proteins. The measurements are the ground truth; this project adds none.
- **Class V, verified native coding sequence.** `data/cds/`, fetched from ENA by
  `scripts/fetch_cds.py` and **accepted only on exact translation to the assayed
  protein**. The verification is the admission criterion, not a convenience.
- **Class R, public reference genome.** Genome and operon context from ENA via
  `scripts/build_genomic_context.py`, and the genomes under
  `../crosstalk/data/genomes` used for corpus statistics in section 45: human
  chr21, twelve named microbial species, and the `N_` contig set.
- **Class D, destructively derived.** Sequences obtained from Class V or R by an
  operation that provably removes the property under test while holding a nuisance
  variable fixed: frameshift, reverse complement, mononucleotide and
  composition-matched shuffles, order-*k* Markov surrogates, and synonymous
  recodings. These are the controls, and they are the methodological core of the
  project. A composition-preserving shuffle encodes nothing and is strictly less
  informative than its source.

**B4, clean panel.** No class here is a benign panel standing in for a hazardous
one, so B4's contamination check does not apply in the form shibboleth needs it.
The analogous check that does apply is the translation gate on Class V and the
homology clustering in section 42, both of which are run and reported.

## What is explicitly excluded

- **No generation of sequence.** No sampling, no denoising to completion, no
  conditional generation, no infilling, no inverse folding, by any model, including
  as an intermediate that is discarded.
- **No optimisation toward a fitness or activity objective.** No search, gradient,
  evolutionary loop or guided sampling anywhere in this repository whose objective
  is a higher predicted fitness, binding, activity or expression. See the exception
  below, which is real and must be read.
- **No fine-tuning on assay labels.** Every score in this repository is zero-shot or
  a frozen-representation probe. A model fitted to a landscape and then used to rank
  unmeasured variants is a design tool, not a measurement.
- **No new organism or sequence data sources** beyond the four classes above. Adding
  one is a review trigger.
- **No per-sequence release.** Outputs are aggregate: per-assay correlations,
  intervals, attenuation ceilings, control tables. No ranked variant lists, no
  per-variant predicted scores on anything unmeasured. This is HARNESS.md B5.
- **No enumeration of evasion or bypass routes**, for any safeguard or for this
  file's own boundary. HARNESS.md B3 is the reason: delete every result, keep the
  artifact, and ask whether the artifact alone lowers the cost of misuse. A written
  list of the unguarded routes out of a containment posture fails that test, and the
  biosafety gate declined to produce one for `paper/DESIGN_OBJECTIVE.md` on exactly
  this ground on 15 Oct 2026. That refusal is recorded in that file and is upheld
  here.

### The optimisation exception, stated because a flat claim would be false

`biologos/landscape.py` exists to supply a calibrated noisy oracle over the ParD3
landscape, and its own docstring points at `crosstalk.solve`, which computes exact
regret. So a sentence reading "this project contains no optimisation over
sequences" would be untrue of the tree the module serves.

The accurate statement: **no search or optimisation loop is run in this
repository.** `landscape.py` is a data loader and a noise model. The search that
consumes it lives in `crosstalk`, over a fully enumerated three-site landscape
(positions 61, 64 and 80 of a bacterial antitoxin, 8,000 states) whose fitness
values are published measurements, which is why exact regret is computable at all
and why there is nothing to discover by searching it. **`crosstalk` has no
`SAFETY.md` of its own**, and whether that search needs one is an open item below,
not something this file can settle for another project.

## The generative-model boundary

This is the part that is new, and the distinction it draws is the one that matters.

`biologos/boltz.py` already drives Boltz, a diffusion model, and passes
`--diffusion_samples` and `--sampling_steps` to it. **Boltz generates structure
conditioned on a sequence that is given to it and proposes no sequence**, so it sits
inside the governing rule: the sequence is an input, the structure is a reading of
it. An earlier draft of `paper/DESIGN_OBJECTIVE.md` claimed that arm would be the
first generative checkpoint in the repository; that was wrong and is corrected
there.

The models proposed in that arm are different in kind. DPLM and D3LM generate
**sequence**, and D3LM's headline application is designing DNA with tunable
regulatory activity. They are admissible here **only as scorers**: a masked
marginal read at a fixed position of a Class P or Class V sequence, which is the
same quantity already read from ESM-2 and the Nucleotide Transformer. Loading such
a checkpoint does not license using it for what it was built for.

The failure mode to name in advance, because it will be proposed as a natural
extension: a cell that clears the BLOSUM62 bar invites sampling from that
checkpoint to see what it produces, and a cell that fails invites fine-tuning it on
the assay. Both are excluded above, and neither is needed for any of the four
outcomes that arm pre-registers. **The arm is answerable entirely by scoring
sequences that already exist, and that is the reason it is worth running in this
form.** If a result makes generation look informative, the correct response is to
write down why and stop.

## Why the exclusion is a feature

The constraint is not a tax on the science; it selects the question.

A design-oriented version of this project would ask whether a genomic model can be
made to propose better proteins. That question needs generation, needs a wet-lab
loop to mean anything, and has a large field working on it. The question this
project asks instead -- whether an apparent cross-modality transfer survives its own
controls -- needs no generation at all, and it is the half nobody is doing. Four
claims in this repository have been retracted by their own controls, including a
representation-level transfer result that scored +0.141 canonically, +0.143
frameshifted and +0.129 reverse-complemented. None of those corrections required
producing a single new sequence.

## Disclosure posture

This is **ordinary publication, not coordinated vulnerability disclosure.** Nothing
here audits a deployed safeguard, so shibboleth's notify-before-posting protocol and
90-day embargo do not apply, and HARNESS.md B6 is satisfied by saying so explicitly
rather than by naming a recipient.

What does carry over:

1. **Aggregate over per-sequence**, per B5 and the exclusion list above.
2. **Corrections are published, not quietly dropped.** Four retractions are in
   `docs/FINDINGS.md` and in the README above the results, which is where HARNESS.md
   H14 says stated blindness belongs.
3. **If any arm ever produces a result about a deployed screening tool or a
   safeguard**, this section is void and `shibboleth/SAFETY.md`'s disclosure
   protocol applies instead. Re-read before proceeding.

## Institutional review

This is computational work on public, published sequence data, with no human
subjects, no pathogens, no select agents and no wet lab, so it is not IRB or IBC
territory.

It must still be **disclosed to a faculty advisor before the first run of any arm
that loads a sequence-generative checkpoint**, and the advisor named in the paper's
responsible-use statement. Several target venues require the statement; write it
from this file.

> **Advisor: _____________________  (UNFILLED -- to be completed by S. Iyer)**
>
> **Date disclosed: _____________________  (UNFILLED)**
>
> Until both fields are filled, HARNESS.md B7 is **unmet**, and
> `paper/DESIGN_OBJECTIVE.md` is blocked at its own Stage 0 step 1: no DPLM or D3LM
> checkpoint is downloaded.

Two questions belong in that conversation and are deliberately not written down
anywhere in this tree, per B3: whether the boundary in this file holds against what
the code paths make available once generative weights are local, and what the
unrefused extensions of the objective arm are. A human reviewer can hold that
analysis in conversation, scoped to this repository, without it becoming a
transferable document.

## Review trigger

Stop and re-read this file before any experiment that:

- introduces an input source not provably Class P, V, R or D,
- loads a checkpoint trained to generate sequence,
- introduces any search, optimisation, guided-sampling or fine-tuning loop,
- would produce a per-sequence artifact intended for release,
- or extends an arm to a system other than the one it was designed for.

## Open items this file does not settle

1. **The advisor fields above.** Blocking.
2. **`crosstalk` has no `SAFETY.md`**, and it is the project that runs search over
   the ParD3 landscape this repository's `landscape.py` loads. Whether that needs
   its own posture is a question for the advisor, and it is not this file's to
   answer.
3. **Licenses for the proposed checkpoints are unverified.** `airkingbd/dplm_650m`
   states none on its card, and neither D3LM repository states one. Stage 0 step 1
   of `paper/DESIGN_OBJECTIVE.md` covers this.
4. **This file has never been tested against a gate that could fail it**, which
   HARNESS.md H16 says is the condition under which a gate is indistinguishable from
   one that works. The honest status is that it is a written posture, not a
   demonstrated one.
