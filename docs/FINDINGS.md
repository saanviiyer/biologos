# biologos findings

Section numbers follow the parent `crosstalk` research log, so they are not
contiguous and two of them collide with sections on unrelated topics in that log.
Only the sections bearing on transfer between biological sequence models appear
here, in the order they were produced.

**Read the retraction notice before citing anything about reading frame.** Section
26 is retracted in its central claim, the notice is inline above it, and sections
29 and 35 carry the corrections that replaced it.

---

## 14. The genomic rung: crossing to DNA, where the anti-correlation does not follow

Sections 9 and 12 are all protein models reading protein sequences, so the
standing alternative explanation was that something about protein-LM pretraining
is at fault. This tests the claim one modality down, on the DNA encoding the very
same variants.

**Data, verified rather than assumed.** All four CDS were fetched from ENA via
UniProt cross-references and each was accepted only after translating to an exact
match against the protein sequence already pinned in `crosstalk.boltz`
(`scripts/fetch_cds.py`): ParD3 `AEH90010.1` 282 nt, ParE3 `AEH90009.1` 312 nt,
ParD2 `AEH89587.1`, ParE2 `AEH89588.1`. Codon usage is the organism's own,
computed from all 6,508 CDS of the WSM2075 chromosome (CP002279, 1,980,535
codons, GC 63.6% with the expected GC3 bias).

Two things are possible here that the protein rung could not do.

**The partner-aware context is the real operon.** Locating the genes in the
genome (`scripts/build_genomic_context.py`) shows ParD3 and ParE3 do not merely
sit adjacent, they **overlap by 11 nt** -- the translationally-coupled
arrangement typical of toxin-antitoxin pairs. So where the protein rung had to
invent a 25-glycine linker to get two chains into one context, the genomic rung
scores the locus the organism carries and the model was trained on. Only the
deliberately non-natural ParD3:ParE2 control is constructed.

**Tokenization transfers exactly.** The ParD3 CDS is 282 nt = 47 non-overlapping
6-mers with no offset, so a token is exactly two codons and the three mutated
codons fall in three distinct tokens (30, 31, 39). The masked-marginal shortcut
therefore costs three forward passes for the whole landscape, the same as ESM-2.
Wild type scores exactly 0.0 by construction, which is the check that the token
arithmetic is right.

### Result: non-predictive, not anti-predictive

Partner-blind AUC on the same 399-variant discrimination set:

| model | AUC | |
|---|---|---|
| trivial baseline: mutation count | **0.664** [0.625, 0.702] | |
| ESM-2 650M likelihood (section 9) | **0.151** | anti-predictive |
| AIDO.Protein-RAG-3B (section 12) | 0.212 | anti-predictive |
| NT-v2 50M multi-species | 0.505 [0.445, 0.561] | chance |
| NT-v2 100M multi-species | 0.460 [0.401, 0.518] | chance |
| NT-v2 250M multi-species | 0.599 [0.543, 0.652] | chance |
| NT-v2 500M multi-species | 0.393 [0.334, 0.451] | chance |
| HyenaDNA (autoregressive, single-nt) | 0.483 [0.424, 0.541] | chance |

**This is a dissociation, not a replication.** Protein-LM likelihood points the
wrong way; genomic-LM likelihood points nowhere. Where ESM-2 650M had rho +0.510
on-target and +0.540 off-target, NT sits at -0.10 / -0.08 -- it is not tracking
protein binding at all, in either direction.

The scale ladder makes the point sharper by *failing* to have a trend. The
protein ladder fell monotonically (0.296 -> 0.151 from 8M to 650M): more capacity,
more naturalness, more harm. The genomic ladder wanders -- 0.505, 0.460, 0.599,
0.393 -- with every CI straddling or near chance and no monotone direction. That
is the signature of noise, not of a signal pointing anywhere.

HyenaDNA closes the two remaining loopholes at once: single-nucleotide
tokenization (no codon is ever straddled) and an autoregressive rather than
masked objective. It sits at 0.483. Neither tokenization nor training objective
explains the result.

### The synonymous floor: how much of a genomic score could possibly be signal

A protein variant does not determine its DNA. Synonymous encodings translate
identically and therefore carry an *identical* measured fitness -- the assay reads
protein binding, so codon choice cannot move the label. Any score variance across
them is provably specificity-irrelevant. No protein rung can measure this.

Sampling synonymous encodings weighted by genome codon usage (the choice a real
pipeline would make), full pseudo-likelihood, ~8 encodings per variant:

| model | within-variant SD | between-variant SD | ratio | attenuation ceiling |
|---|---|---|---|---|
| NT-v2 50M | 3.46 | 6.26 | 0.552 | 0.834 |
| NT-v2 100M | 2.66 | 5.71 | 0.465 | 0.885 |
| NT-v2 250M | 2.25 | 4.46 | 0.505 | 0.863 |
| NT-v2 500M | 3.26 | 6.91 | 0.471 | 0.882 |

Roughly **half the score spread between different variants is reproduced by
changing codons that cannot change the answer**, i.e. about a quarter of the
variance is label-irrelevant by construction. That is a real and previously
unquantified cost of using a genomic LM as a protein-fitness proxy.

It is not, however, the explanation. The implied attenuation on any correlation
is only 0.83-0.89, so noise-correcting the observed rho of about -0.10 still
leaves about -0.11. **The genomic model is not a good proxy that codon noise
spoiled; it carries essentially no protein-binding signal to spoil.**

### Scope

One landscape, one organism, one gene pair. NT-v2 multi-species includes bacterial
genomes but this specific locus's representation in pretraining is unknown, and
Evo 2 -- the model most likely to do better -- needs a GPU and has not been run.

## 15. Readout versus representation, and how much the split decides

Every rung so far reads one number out of a model: a likelihood. That is a claim
about a *readout*, not about a *representation*, and the two can come apart. A
linear probe on frozen embeddings tests the second directly
(`scripts/run_readout_probe.py`): no fine-tuning, ridge regression, same
discrimination set, same metrics.

**The split turned out to be the entire experiment**, and getting it wrong the
first time is the most useful thing in this section.

| features | random split | residue split | within-background |
|---|---|---|---|
| ESM-2 650M embedding | rho 0.923 / AUC 0.997 | 0.638 / 0.862 | **+0.155** |
| NT-v2 50M embedding (DNA) | 0.899 / 0.995 | 0.614 / 0.852 | **+0.088** |
| **trivial: one-hot over 3 sites** | 0.885 / 0.974 | **0.849 / 0.952** | **0.000** |

Read the columns right to left.

**Random splits are meaningless here.** Variants differ at three positions only,
so a random split leaves most (position, residue) combinations present in
training. Everything scores ~0.99, including a 60-parameter lookup table.

**Holding out amino acids is not enough either.** Hold out a variant because it
contains an unseen residue and it still carries two *seen* residues, and an
additive lookup predicts it from those. That is why one-hot scores 0.952 and
**beats both learned representations** -- not because it generalised, but because
two thirds of every test variant was familiar. Any paper reporting an embedding
win on this kind of split against a weaker baseline is reporting the baseline's
absence.

**The clean test** groups test variants by (mutated position, the two residues at
the other positions), so within a group every variant shares an identical
background and differs only at a position holding a residue never seen in
training. One-hot is uninformative *by construction* there: those weights were
never updated, so every member of a group gets the same prediction. Anything
above zero is transferred chemistry.

Under that test the ordering finally means something: **ESM-2 +0.155, the DNA
model +0.088, the lookup table 0.000** (3,814 background groups, 15,087 variants).

Two things follow, and the second is the one worth the section:

1. **The dissociation is real but modest.** The same ESM-2 650M whose likelihood
   scores AUC 0.151 -- actively anti-predictive -- has a representation that is
   positively predictive of the same quantity. The information is in the model;
   the likelihood inverts it. But the margin over a lookup table is +0.155, not
   the +0.86 the leaky split advertised.
2. **The genomic model's representation knows something its likelihood does not.**
   NT's likelihood is at chance (section 14) while its embeddings reach +0.088 on
   unseen residues. Even a DNA model carries some protein-level signal that
   scoring the sequence discards entirely.

**Split choice moved the same features, on the same task, from rho 0.155 to
0.997 -- a factor of six.** That is the practical warning: on a combinatorially
complete landscape, the evaluation protocol determines the conclusion more than
the model does.

## 16. The generality test on measured data, and it does not reproduce

Sections 9-15 are one protein. The obvious test is whether any of it holds
elsewhere, on measured rather than simulated data.

**The dataset.** SKEMPI 2.0 curates measured binding affinities for mutations in
solved complexes. Buried in it is exactly the needed structure: proteins whose
*same* mutations were measured against *two different partners*. Extracting it
(`crosstalk/skempi.py`) gives **15 system pairs, 249 verified two-sided single
mutations**, across about ten independent biological systems -- protease/inhibitor
(BPTI vs trypsin and chymotrypsin), hormone/receptor (hGH vs its own receptor and
the prolactin receptor), enzyme/inhibitor (BLIP vs TEM-1 and SHV-1), TCR/pMHC,
antibody/antigen, and RNase inhibitor vs angiogenin and RNase A.

**A numbering bug that quietly destroyed most of the data.** SKEMPI carries two
mutation numberings and they disagree on 63% of rows: `Mutation(s)_cleaned` is
renumbered to be comparable across entries, `Mutation(s)_PDB` is author numbering
in that row's own structure. Matching structures with the cleaned numbering --
the natural choice, since it is the one that joins across entries -- silently
dropped BPTI from 31 mutations to 0, MT-SP1 from 27 to 1, and the whole TCR set to
0, leaving 6 systems and 103 mutations. Using PDB numbering for the structural
lookup and cleaned numbering for the join recovers 15 systems and 249 mutations.
Every mutation is still accepted only if the residue actually present at that
position in that chain matches the stated wild type.

**Result: nothing works, and the trivial baseline wins.** Leave-one-system-out,
so the probe never sees the system it is scored on, target = ddG_B - ddG_A:

| method | pooled rho (within-system z, n=249) | systems positive |
|---|---|---|
| ESM-2 650M likelihood | **+0.027** | 8/15 |
| ESM-2 650M embedding, LOSO probe | **+0.020** | 9/15 |
| trivial: BLOSUM62 + hydropathy + volume, LOSO | **+0.159** | -- |

Per-system correlations range from -0.435 to +0.936, which at n=10-31 is noise:
n=20 needs |rho| > 0.44 to clear p=0.05.

**The null is real, not underpowered.** The obvious defence is that SKEMPI pools
assays from many labs and the margin is a difference of two noisy numbers, so
nothing could correlate. That is testable the same way section 0 calibrated the
ParD3 oracle. 365 mutations in SKEMPI are measured by more than one independent
reference; the cross-reference spread gives a single-measurement SD of **0.459
kcal/mol**, so margin noise is **0.649**. Against a two-sided margin SD of
**2.490**, the noise-to-signal ratio is 0.261 and the **attenuation ceiling on any
correlation is 0.965**. A true correlation of 0.5 would have shown up as 0.48.
It did not show up.

> **Superseded in part by section 26.** The SKEMPI null here is real for SKEMPI's
> *mutation sampling*, but it is not a refutation. Section 26 shows the two
> sources agree at rho +0.919 on the 27 mutations they share, and that the
> effect is absent from those 27 and present in the 201 dense-DMS mutations
> SKEMPI lacks. SKEMPI's BPTI entry is 48% alanine substitutions with 17 of 31
> at a single position. Read the paragraph below as "curated affinity data
> cannot see this", not as "the effect is not there".

**What this costs the argument.** The ParD3 findings do not currently generalise.
On measured data from ten other systems, ESM-2 likelihood is not anti-predictive
(it is ~0), a frozen-embedding probe is not predictive, and a substitution matrix
plus two physicochemical scalars beats both. The honest position is that sections
9-15 are established *for the ParD3 landscape* and are contradicted, not merely
unconfirmed, as a general claim about protein language models and specificity.

Two differences that may matter and are not yet separated: SKEMPI is single point
mutations at diverse interfaces with a modest dynamic range, whereas ParD3 is a
combinatorially complete three-site landscape scored on a sharper task
(specific-versus-promiscuous among *strong* binders). Whether the ParD3 effect
needs that task structure, or is simply particular to ParD3, is unresolved and is
the thing to resolve before any of it is written up as general.

## 17. The transfer claim, falsified: what looked protein-level is codon leakage

Section 15 reported that the genomic model's *representation* reaches rho +0.088
on unseen residues while its likelihood sits at chance, and read that as weak
genomic-to-protein transfer. Three tests were built to try to break that reading.
All three broke it (`scripts/run_transfer_analysis.py`).

### The causal controls, which are the decisive part

Two manipulations move in opposite directions, and a genuinely protein-level
representation makes an unambiguous prediction about each: destroying the reading
frame must destroy the signal, and changing the DNA while holding the protein
fixed must not.

Within-background rho for a probe trained on canonical encodings (mean-pooled
NT-v2 50M embeddings, 3,814 background groups):

| condition | protein | DNA | rho |
|---|---|---|---|
| canonical | reference | reference | **+0.141** |
| random mutant codon | identical | codon at the varying site randomised | **+0.143** |
| frameshift +1 | **destroyed** | nearly unchanged | **+0.143** |
| reverse complement | **destroyed** | wrong strand | **+0.129** |
| synonymous scramble | identical | all 94 codons replaced | **+0.064** |

**The signal is completely insensitive to whether the sequence is even in frame,
or on the coding strand.** Frameshifting by one nucleotide destroys every codon
downstream and changes nothing (+0.143 against +0.141). Reading the opposite
strand, where the sequence is not a coding sequence at all, costs almost nothing.
No protein-level representation can be invariant to that. This is the finding,
and it does not depend on any of the other rows.

The mechanism is mundane once seen, and it is simpler than it first appears.
Under *any* encoding rule the codon determines the amino acid, so the nucleotides
at the varying site already *are* residue identity. Nothing protein-level has to
be represented for a linear probe to recover them, and re-framing or
complementing the sequence leaves that information fully intact.

**A correction, since the obvious fix does not work.** The first version of this
section attributed the leak to the preferred-codon rule making the codon a
deterministic function of the amino acid, and predicted that randomising the
substituted codon would remove it. It does not: `random mutant codon` scores
+0.143, indistinguishable from canonical. The direction that matters is
codon-to-amino-acid, which is deterministic in every encoding, so randomising
which synonymous codon is used makes the map one-to-many without hiding anything.

**And the one row that does drop should not be read as leak removal.** Synonymous
scramble replaces all 94 codons rather than the one under test, so it is a global
perturbation far off the model's training distribution as well as a
protein-preserving one. Its fall to +0.064 is therefore confounded and is
reported only for completeness. The frameshift and reverse-complement rows carry
the argument; the scramble does not.

### The variance decomposition agrees

Synonymous encodings hold the protein fixed, so the fraction of variance
explained by amino-acid identity measures how protein-level a quantity is. A
protein model scores 1.000 by construction, since it never sees DNA.

| quantity | eta^2 from amino-acid identity |
|---|---|
| NT likelihood | 0.643 |
| NT representation | 0.672 |
| any protein model | 1.000 |

About a third of the genomic representation's variance is synonymous codon
choice, which cannot move a measured label. And the representation is barely more
protein-level than the likelihood (0.672 vs 0.643), so the readout-versus-
representation gap that motivated section 15 is not there on the genomic side.

### The alignment is weaker than a lookup table's

If the two modalities converged on a shared account of these molecules, the DNA
representation should align with the protein representation better than a trivial
encoding does. Linear CKA and a cross-modal ridge map fitted with whole amino
acids held out:

| pair | CKA | held-out R^2 |
|---|---|---|
| NT -> ESM-2 | 0.663 | +0.412 |
| **one-hot -> ESM-2** | **0.819** | **+0.553** |
| one-hot -> NT | 0.713 | +0.533 |

**A 60-parameter one-hot vector aligns with ESM-2 better than the genomic model
does.** Whatever NT shares with ESM-2 is less than what residue identity alone
shares with it, so there is nothing here that deserves the name cross-modal
transfer. NT is best described as a noisy re-encoding of which residue sits
where.

### Scope and one honest wrinkle

The canonical rho here is +0.141 against +0.088 in section 15 because the feature
set differs: section 15 concatenated pooled and per-site embeddings (2,048 dims),
this uses mean-pooled only (512). The comparison across conditions is internally
matched, which is what the controls require.

The controls establish that the apparent transfer on *this* landscape is codon
leakage. They do not establish that no genomic model could transfer: a
codon-randomised scoring rule would remove the leak by construction, and Evo 2 --
the model most likely to behave differently -- still has not been run.

## 18. Genomic-to-protein transfer across 25 DMS assays: no transfer, and the null is real

Sections 14 and 17 are one landscape. This is the generality test
(`scripts/run_dms_transfer.py`), on measured deep mutational scans.

**Setup.** Every ProteinGym assay for which a native coding sequence could be
verified (`scripts/fetch_dms_cds.py`: 29 of 41 resolved -- an assay is kept only
if some ENA CDS for its UniProt accession translates to a protein *containing*
the assayed target exactly, with the offset recorded). After dropping assays
longer than 420 aa or with fewer than 200 usable single mutants, **25 assays
across 9 organisms** remain: human, yeast, *E. coli*, *B. subtilis*,
*P. aeruginosa*, *K. pneumoniae*, *S. acidocaldarius*, SARS-CoV-2 and HIV-1.

The comparison is like-for-like: identical masked-marginal protocol, same region,
the assayed target sequence and the CDS encoding exactly it. Only the modality
differs.

| proxy | mean rho over 25 assays | 95% CI | assays positive |
|---|---|---|---|
| ESM-2 650M on the protein | **+0.466** | [+0.395, +0.536] | **25/25** |
| NT-v2 50M on the CDS | **-0.013** | [-0.033, +0.007] | 10/25 |
| NT-v2 50M, codon-marginalised | **-0.019** | [-0.041, +0.004] | -- |

Paired difference **+0.479** [+0.404, +0.555], t=12.3. ESM-2 has the larger |rho|
on **24 of 25** assays. The genomic model is positive on 10 of 25, a coin flip
(sign test p=0.42), its confidence interval contains zero, and its largest |rho|
on any assay is 0.146 against ESM-2's 0.737.

**The null is not a noise artifact.** Scoring a variant with a genomic LM requires
choosing a codon, and the same masked pass prices every synonymous alternative,
so the spread across encodings that translate identically -- and therefore share
a DMS score exactly -- comes free. Averaged over assays that spread implies a
**mean attenuation ceiling of 0.900**: a true correlation of 0.5 would still have
appeared as 0.45. Nothing appeared.

**Marginalising over codons does not rescue it.** Section 17 showed that
committing to one codon lets a proxy read residue identity off the nucleotides.
Averaging the score over all synonymous encodings removes that channel by
construction and is the genomic score actually entitled to be called a
protein-level proxy. It scores -0.019. The leak was never doing useful work; it
was only ever an alternative route to residue identity.

### The trivial baselines, added 1 Sept 2026, which change how to state this

Section 18 originally reported only ESM-2 and NT-v2, with no trivial baseline. That
was the project's own standing rule broken. Scoring the identical mutation sets
with the chemistry features already implemented in `run_skempi_transfer.py`
(`scripts/run_dms_trivial_baselines.py`, mutation counts matched 25/25):

| scorer | per assay (n=25) | per cluster (n=21) |
|---|---|---|
| ESM-2 650M likelihood | +0.465 [+0.391, +0.540] | +0.442 |
| **BLOSUM62 alone** | **+0.228** [+0.193, +0.263], 24/25 | +0.218 |
| signed Kyte-Doolittle change | +0.175 [+0.142, +0.209] | +0.173 |
| -\|volume change\| | +0.107 [+0.086, +0.127] | +0.100 |
| **chem ridge, leave-one-cluster-out** | **+0.286** [+0.245, +0.326] | +0.276 |
| NT-v2 50M likelihood | -0.013 [-0.034, +0.008] | -0.012 |
| NT-v2 codon-marginalised | -0.018 [-0.042, +0.006] | -0.017 |

A substitution matrix alone reaches 49% of ESM-2, and five fitted chemistry
features reach 61%. ESM-2's margin over BLOSUM62, paired within assay, is
**+0.237** [+0.178, +0.296], higher in 23 of 25.

**This makes the claim cheaper and stronger rather than weaker.** The bar a
genomic model has to clear is **+0.228, not +0.466**, and it does not clear it:
NT-v2 loses to BLOSUM62 by **+0.241** [+0.200, +0.283] in 24 of 25 assays. The
headline is therefore best stated as *a genomic language model scores below a
substitution matrix on protein fitness*, which needs no appeal to a large protein
model at all.

It also sets the honest bar for any future protein-model claim here: ESM-2's
genuinely learned margin over fitted chemistry is +0.179, not +0.465.

### Where this leaves the transfer claim

Stated plainly, and it is a negative result: **on 25 measured DMS assays spanning
9 organisms, a genomic language model scoring the native coding sequence carries
no protein-fitness signal, while a protein language model on the identical region
reaches rho +0.47.** On ParD3 the same holds for binding specificity across four
NT scales and a second architecture with a different tokenization and training
objective (section 14), and what looked like representation-level transfer there
survives frameshifting and reverse-complementing the sequence, so it was never
protein-level to begin with (section 17).

Three caveats, none of which the evidence currently supports leaning on:

- **Scale.** Only NT-v2 50M was run across all 25 assays. The ParD3 ladder covers
  50M-500M and shows no trend, but the DMS sweep does not.
- **Evo 2 is untested** and is the model most likely to differ; it needs a GPU.
- **Coding sequences are short.** Genomic LMs are built for long-range genomic
  context, and a 300-1,200 nt CDS in isolation may simply be the wrong input for
  them, even though it is the input any protein-fitness application would supply.

What the result does establish is that the obvious way to use a genomic model as
a protein-fitness or specificity proxy -- score the coding sequence -- does not
work, is not rescued by marginalising codons, and is not limited by measurement
noise.

## 19. The context objection, tested directly: more genome makes it worse

Section 18's standing objection is that genomic language models are built for
long-range context and were handed an isolated 300-1,200 nt coding sequence -- the
wrong input, so the null is uninformative. Two tests, one weak and one decisive.

**The weak one.** Refetching each DMS assay's parent nucleotide record
(`scripts/fetch_dms_context.py`) resolves 21 of 29 assays, only 10 of them
genuine `Genomic_DNA`. Most EMBL cross-references are gene-only entries, so the
available flanks are tiny -- IF1_ECOLI's whole record is 328 nt, RASH_HUMAN's is
an mRNA with zero upstream. Only a handful (BLAT 204/3000, ESTA_BACSU 3000/3000,
TRPC_SACS2 2232/4) carry real neighbourhood. This arm cannot settle the question
and is not asked to.

**The decisive one.** ParD3 can settle it, because the entire 6.9 Mb
*M. opportunistum* chromosome is on disk and the locus's coordinates are known, so
the real upstream and downstream sequence can be dialled up to the model's full
context (`scripts/run_context_doseresponse.py`). Flanks are snapped to multiples
of 6 so codons never straddle 6-mer token boundaries, which would otherwise
confound a context effect with a tokenisation artifact.

| real flank each side | total nt | tokens | AUC | 95% CI |
|---|---|---|---|---|
| 0 | 282 | 47 | 0.505 | [0.445, 0.561] |
| 60 | 402 | 67 | 0.511 | [0.453, 0.569] |
| 300 | 882 | 147 | 0.492 | [0.434, 0.552] |
| 1,200 | 2,682 | 447 | **0.379** | [0.325, 0.433] |
| 3,000 | 6,282 | 1,047 | **0.365** | [0.311, 0.421] |
| 5,400 | 11,082 | 1,847 | 0.410 | [0.354, 0.468] |
| *trivial baseline: mutation count* | | | **0.664** | |

**MECHANISM CORRECTED BY SECTION 35.** The trend below is real but it is NOT
evidence that *genomic* context hurts. Composition-matched shuffled flanks, which
carry no genomic information at all, reproduce it: mononucleotide-shuffled trend
**-0.633** across three seeds against real's -0.771. It is a length and composition
effect on the score scale. The trend is also grid-dependent, -0.430 on a denser
grid, with the curve turning back up past 3 kb. The conclusion that context does
not rescue NT survives; the sentence below does not, and should not be quoted.

**More real genomic context monotonically hurts** (Spearman AUC against flank
size **-0.771**). At 11 kb the model is being given 1,847 tokens -- close to its
full 2,048-token context, and all of it the organism's actual chromosome -- and
the specificity signal is still absent, in fact further below chance than with no
context at all. Nothing approaches the mutation-count baseline at any context
size.

The objection is therefore answered on this landscape: the failure in sections 14
and 18 is not an artifact of withholding genomic context. Supplying the real thing,
up to the model's capacity, does not produce a signal. Together with the real
ParD3:ParE3 operon arm of section 14 -- the natural locus, 11 nt gene overlap
included, also at chance -- there is no version of "give it proper genomic input"
left that has not been tried at this scale.

What remains untested is a model built for much longer range and much larger
capacity. That is Evo 2, which needs a GPU;
`notebooks/Evo2_crosstalk_GPU.ipynb` runs every arm above under the identical
protocol and is the one outstanding arm.

## 20. The DMS null holds at every genomic model scale

Section 18 ran one genomic model across the 25 assays, leaving open whether a
larger one would recover the signal. The ParD3 ladder showed no trend, but ParD3
is one protein. Repeating the full sweep at four scales
(`scripts/run_dms_transfer.py --nt ...`, `results/dms_scale_sweep.csv`):

| genomic model | mean rho | 95% CI | codon-marginalised | attenuation ceiling | ESM-2 larger \|rho\| |
|---|---|---|---|---|---|
| NT-v2 50M | -0.013 | [-0.033, +0.007] | -0.018 | 0.900 | 24/25 |
| NT-v2 100M | -0.013 | [-0.037, +0.011] | -0.025 | 0.899 | 24/25 |
| NT-v2 250M | -0.007 | [-0.025, +0.012] | -0.017 | 0.880 | 24/25 |
| NT-v2 500M | +0.008 | [-0.016, +0.032] | +0.010 | 0.880 | 24/25 |
| *ESM-2 650M on the protein* | **+0.465** | [+0.395, +0.536] | | | |

Every interval contains zero. A tenfold increase in parameters moves the mean
correlation by 0.021, from -0.013 to +0.008, which is inside the noise of a single
assay. The protein model has the larger absolute correlation on 24 of 25 assays at
every scale, and the same assay is the exception each time.

Two readings are now excluded together. Section 19 excluded missing genomic
context on ParD3; this excludes insufficient capacity across 25 assays and nine
organisms. What remains is a model class built for much longer range than any
tested here, which is Evo 2.

## 26. RETRACTED IN ITS CENTRAL CLAIM. See the correction below before reading

**Do not cite this section's diagnostic.** An adversarial audit on 2026-08-31 found
a confound that reproduces the entire result and a logical error in its headline
claim. The numbers below stand as measurements; the interpretation does not.

**The confound.** `frameshift(cds, k=1)` is a rotation, `cds[1:] + cds[:1]`. Apart
from one wrap junction that string is real genomic sequence read one base later:
`cds[1:]` is verifiably a substring of the chromosome. The reverse complement is
also real genomic sequence, on the other strand. The two conditions that were NOT
penalised are exactly the two that yield genuine genomic sequence, and the two
that were penalised, codon-order shuffle and synonymous recode, are exactly the two
that yield synthetic sequence. Rank the four conditions by whether they produce a
real genomic string and the table is reproduced without any appeal to reading
frame. NT-v2 behaves as a correct density model over genomes.

**The logical error.** "Reading-frame discrimination is a necessary condition for
ranking protein variants" is false. Ranking point mutants of one gene compares
variants that all share the same frame, and a discriminative ranker never needs to
represent a quantity that is constant across the comparison. Nucleotide
conservation scores with no frame variable at all (phyloP, GERP, CADD) predict
variant effect. Frame is also not identifiable from the string, since a genomic LM
sees all six phases at arbitrary chunk offsets during training.

**Pseudoreplication.** The 29 rows are **24 unique coding sequences**. BLAT_ECOLX
appears three times and CP2C9, PTEN and RL401 twice each, with scores identical to
five decimals. Recomputed on unique sequences with t-based intervals: synonymous
+0.2595 [+0.077, +0.442], codon shuffle +0.3587 [+0.211, +0.506], frameshift
+0.0505 [-0.078, +0.179], reverse complement -0.0375 [-0.169, +0.094]. **The
"twelve times more strongly" ratio becomes 7.1 and its bootstrap interval spans
zero, so that sentence is withdrawn.**

**Power.** At n=24 the minimum detectable effect is about 0.181 nats per token, so
a real frame effect of 0.15 would be missed more often than not. This is a
non-detection at a resolution too coarse to be informative, not a null. About 283
genes would be needed for 80% power at the observed effect size.

**Two further defects.** The synonymous recode excludes the native codon and draws
uniformly over alternatives, so it over-represents rare codons and conflates
protein preservation with wrecked codon usage. And pseudo-likelihood is a sum of
local conditionals, so rotation invariance may be a property of the scoring rule
rather than of the model; the matched protein-side control has not been run.

**The frame question is OPEN, not answered negatively (added after section 29).**
The confound-free replacement probe has now been built and calibrated against a
3-periodic Markov positive control with a matched aperiodic twin. Normalised by
each model's own reference effect, the positive control shows **+3.42%**
[+2.63, +4.20] on the matched contrast and the aperiodic twin shows -0.54%.
Against that yardstick:

- **NT-v2 50M is UNDERPOWERED, not null.** Its interval reaches **+11.59%**, well
  above the 3.42% a frame-representing model shows, and its point estimate
  (+4.38%) is larger than the positive control's. A frame effect of the size a
  frame-aware model exhibits would not have been detected here. Bounding it would
  take roughly **214 unique coding sequences**, against the 24 available.
- **HyenaDNA is bounded, but only just.** Its interval tops out at +3.35% against
  the control's +3.42%, a margin of 1.0x. That is a tie at the boundary, not a
  clean exclusion.
- ESM-2 35M detects the contrast at +4.88%, so the probe works.

**So no claim that a genomic language model lacks reading-frame representation is
supported by anything in this file.** The rotation probe was confounded and the
confound-free probe lacks the power to replace it at this sample size. Earlier
statements in this project that HyenaDNA "fails the frame test", and that its
single-nucleotide tokenisation closes the tokenisation loophole, overstated what
the data support and are withdrawn.

A further narrowing from the order sweep: every 3-periodic Markov order from 0 to
5 passes the rotation test, but orders 1 and 5 fail the stop contrast. A null on
the contrast therefore means "has not learned in-frame stop depletion", which is
narrower than "does not represent the reading frame".

**What survives.** Sections 18, 19 and 20, the DMS null, do not depend on this
section and are unaffected. Section 17 also stands, since its mechanism (codon to
amino acid is deterministic under every encoding, so a probe reads residue
identity off the nucleotides) does not rest on the frameshift row.

**Citation corrections.** Merchant et al. (Nature 2025) used Evo 1, not Evo 2. King
et al. (Science 2026) used Evo 1 and Evo 2 with fine-tuning, prompt engineering and
inference-time guidance from separate predictive models, so their viable phages did
not come from bare likelihood ranking, and the tension this section claimed is
correspondingly smaller.

**The replacement experiment** must match realness across arms: a 1-nucleotide
insertion against a 3-nucleotide insertion at the same site, both synthetic with
only one breaking frame; and an in-frame nonsense SNV against a synonymous SNV in
the same codon, both single-base edits to a real gene with only one destroying the
protein. A GeneMark-style 3-periodic Markov model is the missing positive control
and the missing trivial baseline.

---

## 26. The reading-frame test: a genomic LM that cannot tell a gene from its frameshift

Merchant et al. (Nature 2025) design functional de novo genes with a genomic
language model and King et al. (Science 2026) design whole bacteriophages the same
way, while sections 14 to 20 find that genomic-LM likelihood carries no
protein-fitness signal. The obvious reconciliation is a difference of grain:
generating a plausible gene needs coarse competence, ranking two point mutants
needs fine competence, and a model could have the first without the second.

`scripts/run_granularity_ladder.py` tests that on 29 verified coding sequences
with a factorial of corruptions, each isolating one ingredient. Scores are mean
per-token log-likelihood so lengths are comparable, and every comparison is paired
within gene.

| corruption | what it preserves | what it destroys | mean delta from the real gene | real higher |
|---|---|---|---|---|
| synonymous recode | the protein, exactly | native codon choice | **+0.2334** [+0.0849, +0.3820] | 20/29 |
| codon-order shuffle | codon usage, exactly | the protein | **+0.3218** [+0.1954, +0.4482] | 24/29 |
| frameshift +1 | nucleotide composition | the reading frame | **+0.0264** [-0.0778, +0.1307] | 14/29 |
| reverse complement | composition | coding status entirely | **-0.0643** [-0.1742, +0.0455] | 11/29 |

**The coarse-competence hypothesis is wrong, and the result is worse than that.**
The model cannot distinguish a real gene from the same gene read out of frame: the
difference is +0.026 nats per token with a confidence interval spanning zero, and
the real gene scores higher in 14 of 29 cases, which is a coin flip. It also
cannot distinguish a gene from its reverse complement, where the point estimate
actually favours the complement.

What the model is sensitive to is local sequence statistics. Shuffling codon order
costs 0.322 nats per token even though codon usage is preserved exactly, so the
penalty is for disrupted local context rather than for composition. Synonymous
recoding costs 0.233. The model discriminates codon context **twelve times more
strongly than it discriminates reading frame**.

That is a coherent account of everything in sections 14 to 20. A likelihood that
does not represent the reading frame cannot rank protein variants, because which
protein a sequence encodes is exactly the thing the reading frame determines. The
apparent representation-level signal in section 17 survived frameshifting for the
same reason: there was never a frame-dependent representation for frameshifting to
destroy.

### The diagnostic this yields

Reading-frame discrimination is a **necessary condition** for a genomic model's
likelihood to be usable as a protein-level proxy, and it is cheap to test. Score a
set of real coding sequences and their one-nucleotide rotations, and compare
paired. A model that fails has no gene-level representation, whatever its
performance on any benchmark, and its likelihood cannot be scoring protein
function.

The test needs no labels, no assay, and no fitness data. It costs two forward
passes per gene. Nothing in the genomic-LM literature reports it.

### Scope, and what it does not say about Evo 2

This is the Nucleotide Transformer v2 family at 50M parameters, on 29 genes. It
does not test Evo 2, which is far larger, trained on far more sequence, and built
for much longer context, and which is the model behind both the Nature and the
Science results. The honest reconciliation is therefore not settled here: it is
possible that Evo 2 passes the reading-frame test and NT does not, in which case
frame representation is a capability that appears with scale and the two
literatures are measuring different models rather than different grains.

That makes the test the sharpest available prediction to run on a GPU, and it has
been added to `notebooks/Evo2_crosstalk_GPU.ipynb`. If Evo 2 fails it too, the
generative results rest on something other than the likelihood this literature
reports.

## 29. The frame test has a working positive control, and it changes what the nulls mean

Section 26 was retracted because its frameshift diagnostic was confounded: a
one-nucleotide rotation of a gene is still real genomic sequence, so a correct
density model over genomes has no reason to penalise it. The replacement probe
is the matched in-frame / out-of-frame stop contrast — write TAA either on a
codon boundary or one base past it, two three-base edits of the same
trinucleotide at the same site, only one of which can terminate the protein.
ESM-2 passes it. But ESM-2 reads the translated protein, so its pass shows the
contrast works on a **protein** model and says nothing about whether it works on
a model that eats nucleotides. The order-1 Markov chain already in the repository
cannot settle it either: with one transition table it has no parameter that
depends on position modulo three, so it is a negative control by construction.

**A 3-periodic Markov chain is the missing case.** Separate transition tables for
codon positions 1, 2 and 3, the GeneMark device (Borodovsky & McIninch 1993),
represent the reading frame explicitly, in the DNA alphabet, with no protein
anywhere in the model. `scripts/run_biointerp_periodic.py` fits it on the 24
unique coding sequences leave-one-sequence-out, so it is a generative model of
coding DNA rather than a fit to the string in front of it, and runs it against a
**non-periodic twin at the same order, on the same corpus, by the same
leave-one-out procedure, through the same scoring code**, so periodicity is the
single difference between the two arms. Order 2 was chosen by held-out
log-likelihood on the real sequences, before any intervention was scored.

| probe | 3-periodic (frame-aware) | aperiodic twin (frame-blind) |
|---|---|---|
| frameshift +1 (rotation) | **+0.0861** [+0.0701, +0.1022], 23/24 | -0.0001 [-0.0002, -0.0000], 10/24 |
| matched stop contrast | **+0.0035** [+0.0027, +0.0043], 24/24, p=1.2e-07 | -0.0002 [-0.0005, +0.0001], 11/24, p=0.84 |

**The contrast has power against a DNA-alphabet model.** A model whose only frame
machinery is three phase-indexed tables passes it decisively; the identical model
without them fails. So a null on this contrast is evidence about the model rather
than about the instrument, and the genomic frame material can support a
conclusion. The edit-size asymmetry between the two arms (2.25 versus 2.12 bases
actually changed) is not what produces the pass, because the frame-blind twin
sees the same asymmetry and returns a slightly negative contrast.

### But the contrast is low-gain, and that is the finding

Nats per token are not comparable across a 6-mer tokenizer, a single-nucleotide
one and an amino-acid one, so every effect below is divided by that model's own
reference effect, the mononucleotide shuffle. All batteries are re-adjudicated on
the 24 unique coding sequences with t intervals.

| model | matched stop contrast, as % of its own reference | verdict |
|---|---|---|
| Markov order-2 3-periodic (positive control) | **+3.42%** [+2.63, +4.20] | REPRESENTS |
| Markov order-2 aperiodic (negative control) | -0.54% [-1.28, +0.20] | NULL |
| ESM-2 35M | +4.88% [+1.47, +8.29] | REPRESENTS |
| NT-v2 50M | +4.38% [-2.83, **+11.59**] | **underpowered** |
| HyenaDNA 32k | +1.21% [-0.93, **+3.35**] | at the boundary |

A frame-representing DNA model shows only 3.4% of its own reference effect on
this contrast, against 83.6% on the rotation. The contrast is roughly
twenty-five times weaker, because it edits three bases of a thousand while the
score is averaged over the whole sequence. At that gain and n=24:

- **NT-v2's contrast null is underpowered, not bounded.** Its interval reaches
  11.6% of its own reference, above the 3.4% a frame-representing model shows,
  and its point estimate (+4.38%) is in fact larger than the positive control's.
  It would take about **214 unique coding sequences** to bound it at the positive
  control's effect size. *NT-v2 has not been shown to lack frame representation.*
- **HyenaDNA's interval tops out at 3.35% against the control's 3.42%.** That is
  a tie, not an exclusion, and must not be quoted as a bounded null.
- On the rotation probe both are bounded (NT-v2 +11.98% [-18.40, +42.36],
  HyenaDNA -0.48% [-1.66, +0.71], against the control's +83.56%) — but that is
  the probe section 26 retracted, so the bound is on a gene-model quantity a
  genome model has no obligation to have.

**The contrast is also not a complete test of frame representation.** Sweeping
Markov order (`results/frame_power_markov_orders_full.csv`), every 3-periodic
model from order 0 to order 5 passes the rotation test at 71–134% of its
reference, but orders 1 and 5 **fail the stop contrast**. Order 1 has no context
for the third base of a stop codon; order 5 has 4096 contexts per phase against
17.8 kb of training sequence. So failing the contrast means "has not learned
in-frame stop-codon depletion", which is narrower than "does not represent the
frame".

### Corrections carried by this section

The 29 rows of every earlier battery are 24 unique coding sequences
(BLAT_ECOLX three times, CP2C9, PTEN and RL401 twice each) and the ESM-2 control's
10 are 9 (RL401 twice). All intervals were m ± 1.96·SE. Both are fixed here by
re-adjudication from the saved per-sequence deltas, with no rescoring; the full
before/after table is `results/biointerp_dedup_corrections.txt`. Four numbers
quoted elsewhere move:

| number | before | after |
|---|---|---|
| ESM-2 matched contrast | +0.0511 [+0.0243, +0.0779], 9/10, p=0.021 | **+0.0504 [+0.0151, +0.0856], 8/9, p=0.039** |
| NT-v2 matched contrast | +0.0191 [-0.0047, +0.0430], 18/29 | **+0.0185 [-0.0119, +0.0488], 14/24** |
| NT-v2 frameshift +1 | +0.0264 [-0.0778, +0.1307], 14/29 | **+0.0505 [-0.0775, +0.1784], 13/24** |
| HyenaDNA 'stop out of frame' | +0.0009 [+0.0000, +0.0018] REPRESENTS | **+0.0011 [-0.0000, +0.0022] NULL** |

One verdict changes (HyenaDNA's unmatched stop control) and no headline claim in
sections 18 to 20 is affected, since none depends on this material.

### What to say and what not to say

Say: the frame diagnostic is a validated instrument on DNA models, and it is
cheap — two forward passes per gene, no labels. Say: at the sample size the
literature would plausibly use, the confound-free version of it is too coarse to
convict a genomic LM, and the version that is not too coarse is confounded.

Do not say that NT-v2 or HyenaDNA fails to represent the reading frame. The
evidence does not reach that.

`notebooks/Evo2_crosstalk_GPU.ipynb` cell 7b is not yet usable as written: it
runs the four retracted section-26 conditions, has no matched stop contrast, and
does not deduplicate its genes. Re-cut it against
`scripts/run_biointerp_periodic.py` — the matched contrast, unique coding
sequences, t intervals, and the 3-periodic Markov control fitted on the same
genes as the yardstick — and give it at least 214 unique coding sequences before
any Evo 2 result is read as a null. On fewer than that, an Evo 2 null on this
contrast would mean nothing, exactly as NT-v2's does not.

## 30. Evo 2 refutes the general form of the genomic null, and the trivial baseline says by how much

Run on Dartmouth `discovery` (NVIDIA L40S, evo2 0.6.0) from
`notebooks/Evo2_crosstalk_GPU.ipynb`, with four repairs recorded in the handoff
README, the load-bearing one being that `evo2_1b_base` needs Transformer Engine and
was replaced by **`evo2_7b`**. The numbers therefore describe the 7B model. The
collaborator also found a real bug in my scorer: `Evo2.__call__` nests two deep, so
the single unwrap raised a TypeError.

### The DMS panel, matched assay-for-assay

Their run scored 32 assays against my 25. All 25 of mine are inside their 32, so
the comparison can be made exactly rather than approximately. On those 25 assays
(21 protein clusters):

| scorer | mean rho | 95% CI |
|---|---|---|
| ESM-2 650M (protein LM) | +0.4655 | [+0.3950, +0.5359] |
| chemistry ridge, leave-one-cluster-out | +0.2860 | [+0.2475, +0.3244] |
| **Evo 2 7B (genomic LM)** | **+0.2658** | [+0.2022, +0.3295] |
| BLOSUM62 alone | +0.2282 | [+0.1950, +0.2614] |
| NT-v2 50M (genomic LM) | -0.0132 | [-0.0333, +0.0069] |

**The blanket claim in sections 18 to 20 is refuted.** "A genomic language model
scoring a native coding sequence carries no protein-fitness signal" is false at Evo
2 scale: +0.266 with an interval excluding zero and positive on 23 of 25 assays.
That claim now holds only for the Nucleotide Transformer family at the scales
tested, and the original framing was too general.

**But the trivial baseline decides how much was gained**, and this is why it was
worth adding:

| paired contrast | difference | higher in | verdict |
|---|---|---|---|
| Evo 2 minus BLOSUM62 | +0.0376 [-0.0118, +0.0870] | 14/25 | **contains zero** |
| Evo 2 minus chemistry ridge | **-0.0202** [-0.0755, +0.0352] | 9/25 | contains zero, and negative |
| ESM-2 minus Evo 2 | +0.1997 [+0.1532, +0.2461] | 24/25 | excludes zero |

Per cluster the picture is the same: Evo 2 +0.2581 against BLOSUM62 +0.2183, a
difference of +0.0398 [-0.0219, +0.1014], higher in 11 of 21.

**So the corrected claim is narrower and more useful than either the original null
or its refutation.** At Evo 2 scale a genomic model does acquire protein-fitness
signal, it is **statistically indistinguishable from a substitution matrix**, it is
not better than five fitted chemistry features, and it remains far below a protein
language model. Without BLOSUM62 in the table, +0.266 would read as a substantial
positive result.

### ParD3 specificity: the null survives, and the noise floor is now much tighter

Every single-sequence arm sits at or below the 0.664 mutation-count baseline
(partner-blind 0.401, real operon 0.495, synthetic ParD3:ParE2 0.390), while
correlating +0.47 to +0.49 with on-target binding. Evo 2 tracks whether a protein
works, not which partner it chooses.

The margin arm (E3 minus E2) was the only arm above baseline at 0.689 and was
reported without an error bar. Bootstrapped here over 4,000 resamples:
**0.689, 95% CI [0.634, 0.740]**. It excludes chance but **contains the 0.664
baseline**, and beats that baseline in 82.4% of resamples. It is therefore not
shown to beat counting mutations.

The synonymous floor improves markedly: within-variant SD 0.983 against
between-variant 3.579, a ratio of 0.275 and an **attenuation ceiling of 0.962**,
where NT sat at 0.83 to 0.89. A true correlation could be attenuated by only about
4%, so codon noise cannot explain the specificity null. **This strengthens the
ParD3 negative result** rather than weakening it.

### The context dose-response reverses

Section 19 found that supplying real genomic context made NT-v2 monotonically
worse, trend rho -0.77. Evo 2 goes the other way: **+0.70**, with AUC rising from
0.401 at no flank to 0.543 at 3,000 nt, saturating by about 1,200 nt, and the gain
coming from off-target correlation falling (+0.363 to +0.219) while on-target stays
flat. On five points p = 0.19 and every interval overlaps, so **the sign reversal is
the finding and the magnitude is not resolved.**

**WITHDRAWN PENDING A CONTROL (section 35).** This arm has **no shuffled-flank
control**, and one is now known to be necessary. HyenaDNA, which holds 30 kb
natively, shows a real-flank trend of +0.536 that its shuffled twins fully account
for (+0.893, +0.750, +0.750, +0.750; real minus shuffled -0.008, real higher in 1
of 6). A positive context dose-response is therefore reproducible from flanks
containing no information. **Evo 2's +0.70 must not be read as long-range genomic
context until the shuffled twin is run.** The control ports directly from
`scripts/run_recursive_context_hyena.py` and is the single highest-value item
outstanding on the Evo 2 arm.

### What still stands from sections 18 to 20

The NT-v2 results are unaffected: it remains at -0.013, losing to BLOSUM62 by
+0.241 in 24 of 25 assays. The correct statement of the whole arm is now that
**genomic-to-protein transfer is a function of scale and architecture, absent in the
Nucleotide Transformer family and present but sub-baseline in Evo 2.**

## 31. Transfer is not a scaling phenomenon: the NT trend does not reach Evo 2

[Longpre2025atlas] fits scaling laws for cross-lingual transfer and locates the
compute crossovers at which one choice overtakes another. DNA and protein are two
encodings of one molecule, so the same question applies: is a genomic model's
protein-fitness skill a function of scale?

Five points are now available on the identical 25 assays, a 140x parameter span
(`scripts/run_transfer_scaling.py`):

| model | parameters | mean rho |
|---|---|---|
| NT-v2 50M | 5.0e7 | -0.0099 |
| NT-v2 100M | 1.0e8 | -0.0129 |
| NT-v2 250M | 2.5e8 | -0.0065 |
| NT-v2 500M | 5.0e8 | +0.0083 |
| **Evo 2 7B** | **7.0e9** | **+0.2658** |
| *BLOSUM62 reference* | | *+0.2282* |
| *ESM-2 650M reference* | | *+0.4655* |

Within the Nucleotide Transformer family the trend is **+0.0179 per decade of
parameters**, Spearman(params, rho) = +0.80 across four points.

**The trend does not explain Evo 2.** Extrapolated to 7B it predicts **+0.024**
against the **+0.266** actually measured, under-predicting by a factor of 11.

**And the extrapolation is absurd, which is the point.** Reaching BLOSUM62 at
+0.228 along the NT line would take about **1.9e21 parameters**. That number is not
an estimate of anything; it is what happens when a flat trend is extended thirteen
orders of magnitude, and it should be quoted only to make that point. The NT values
span -0.013 to +0.008, all within noise of zero, so the honest statement is that
the NT trend is flat and any crossover it implies is meaningless.

**The reading.** Evo 2's protein-fitness signal is not what scaling the Nucleotide
Transformer family would buy. It comes from architecture, corpus and context
length, which is the genomic analogue of the sub-billion result in
[Liu2024mobilellm], where architecture dominated parameter count. Section 30's
context dose-response points the same way: real genomic context helps Evo 2
(trend +0.70) and hurts NT (-0.77), and long-range context is exactly what Evo 2
is built for.

**Caveat that limits this to a demonstration rather than a law.** Four NT points,
one Evo 2 point, and the two families differ in architecture, tokenisation, corpus
and context window simultaneously, so nothing here isolates which of those is
responsible. A genuine scaling law in the sense of [Longpre2025atlas] would need a
single family swept across scales with everything else held fixed, which no public
genomic model family currently provides at this range.

## 32. A genomic model memorises nothing, and barely beats an order-5 Markov chain

Section 28 measured memorisation in a protein model and found no verbatim recall.
The genomic case admits a sharper test, because Nucleotide Transformer v2 was
trained on a known finite genome set, so membership can be checked rather than
assumed (`scripts/run_memorisation_dna.py`, `analyze_memorisation_dna.py`).

**Setup.** 27,787 span reconstructions over 720 windows of 3,000 nt from 25
sources, NT-v2 50M. L is the span in 6-mer tokens, so L=10 is 60 nt, the direct
analogue of section 28's 20-residue span. Arms: 11 genomes verified present in the
NT-v2 corpus, 13 released in 2025-26 and therefore after its training, plus
genus-matched pairs, off-frame controls, and matched Markov, dinucleotide and
mononucleotide shuffles. Window pools are GC-matched in 1% bins and permutations
are over genomes rather than windows.

### There is no membership signal at all

| comparison | member | non-member | difference | p |
|---|---|---|---|---|
| per-nt accuracy, L=1 | 0.3244 | 0.3321 | **-0.0077** | 0.647 |
| per-nt accuracy, L=10 | 0.3256 | 0.3296 | -0.0040 | 0.747 |
| log p(true), nats/token, L=1 | -8.2017 | -8.2278 | +0.0261 | 0.885 |
| log p(true), nats/token, L=10 | -7.8332 | -7.8181 | -0.0151 | 0.912 |

Genomes the model was trained on are reconstructed no better than genomes
published after its training finished, and the point estimates run slightly the
wrong way. The sensitive test agrees: loss-based membership inference, including
the standard order-5 reference-model ratio correction for samples that are
intrinsically easy rather than memorised, gives **AUC 0.49 to 0.55** across every
span length. That is chance.

Exact span recovery is approximately zero at every length beyond a single token, in
every arm, so there is no verbatim recall either, matching section 28's protein
result.

### The number that reframes sections 30 and 31

| scorer | per-nt accuracy |
|---|---|
| NT-v2 50M | ~0.330 |
| order-5 Markov chain fitted to the source genome | ~0.310 |
| always emit the window's commonest base | ~0.320 |
| chance for four bases | 0.250 |

**A 50M-parameter genomic language model beats an order-5 Markov chain by about
two percentage points of nucleotide accuracy on its own pretraining objective**,
and beats "emit the commonest base" by between 0.6 and 2.7 points.

That reframes the protein-fitness nulls in sections 18 to 20 and 31. NT-v2's
failure to carry protein-fitness signal is not a surprising fact about
cross-modality transfer. It is what one should expect from a model that is barely
distinguishable from a Markov chain at modelling DNA in the first place. The
interesting question was never why NT-v2 fails on proteins; it is why anyone
expected a model with this much margin over a 5-mer lookup to succeed at anything.
Section 31's finding that Evo 2 sits 11x above the NT scaling line reads the same
way from this side.

### A confound the audit raised is now closed

The adversarial audit of section 26 argued that the DMS coding sequences were
"well-curated, database-canonical genes of model organisms in NT-v2's 850-genome
training corpus", so both a gene and its rotations would be verbatim training
substrings, which alone could predict the observed null. Two results here retire
that. Membership confers no reconstruction advantage anywhere, and the specific
chromosome used throughout this project, CP002279 (*M. opportunistum*), is **not**
a member of the NT-v2 corpus. The memorisation confound is not what produced those
results.

### Scope

One model family at 50M, with a 500M arm and a HyenaDNA smoke test also on disk and
not yet analysed. Membership is established by corpus documentation and release
date, which is strong evidence but not the same as access to the training set.
Nothing here speaks to Evo 2, whose corpus and scale are both different and which
section 30 shows behaves differently.

## 34. Genomic models on their own home task: mostly beaten by a lookup table

Sections 14 to 32 all grade genomic models on coding sequence and protein fitness,
a task they were not built for. This measures them where they are meant to work
(`scripts/run_regulatory_models.py`, `run_regulatory_headtohead.py`).

**Setup.** Five Nucleotide Transformer benchmark tasks, three splits (shipped,
homology-clustered, random), frozen embeddings with a linear probe, identical
pipeline to the baselines so only the features change. The layer and pooling grid
was **fixed before any test number existed**, the headline configuration is chosen
by grouped cross-validation on training rows only, and a best-of-nine-on-test row
is reported separately purely to bound the sweep's headroom. The trivial
competitor is chosen **on test**, so the baseline gets an oracle the model does not.

### The answer, on the clustered split, paired cluster bootstrap

| task | best trivial | AUC | NT-v2 50M | difference | HyenaDNA | difference |
|---|---|---|---|---|---|---|
| splice_sites_all | one-hot | 0.959 | 0.756 | **-0.203** [-0.216, -0.189] | 0.704 | **-0.254** [-0.270, -0.240] |
| promoter_tata | one-hot | 0.941 | 0.930 | -0.010 [-0.041, +0.022] | 0.915 | -0.025 [-0.063, +0.011] |
| promoter_all | k-mer 5 | 0.925 | 0.934 | **+0.010** [+0.001, +0.018] | 0.938 | **+0.013** [+0.004, +0.022] |
| H3K4me3 | k-mer 3 | 0.875 | 0.887 | +0.012 [-0.001, +0.025] | 0.873 | -0.002 [-0.018, +0.013] |
| enhancers | k-mer 5 | 0.779 | 0.808 | **+0.029** [+0.017, +0.043] | 0.787 | +0.009 [-0.005, +0.022] |

**Two clear wins in ten model-task cells, both under 0.03 AUC, against one loss of
0.20.** A positional one-hot reaches 127% of NT-v2 and 136% of HyenaDNA on splice
sites; a 3-mer count reaches 99% of NT on H3K4me3. This is the genomic counterpart
of BLOSUM62 reaching 49% of ESM-2 in section 18, and it is worse, because here the
lookup table wins outright on two tasks.

### The splice-site loss is the readout, not the representation

Mean pooling is position-blind, which hands a positional one-hot an unearned
advantage on a task defined by a positional consensus. A **post-hoc** windowed
readout, mean over eight equal token blocks, was added after seeing the loss and is
labelled as post-hoc everywhere:

| task, clustered | one-hot | NT mean-pool | NT windowed | Hyena windowed |
|---|---|---|---|---|
| splice_sites_all | 0.959 | 0.756 | **0.938** | 0.869 |
| promoter_tata | 0.941 | 0.930 | **0.954** | 0.936 |

The 0.203 gap collapses to 0.021 and NT overtakes one-hot on promoter_tata. So the
information **is** in the representation and the standard frozen-embedding protocol
discards it. The headline loss stands for the standard protocol, and it must not be
restated as "the model does not represent splice sites".

That is the same lesson as section 15 in a new place: the protocol, not the model,
decided the number. Layer choice alone moves results by up to 0.028 within a model,
against 0.010 to 0.052 between models on four of five tasks.

### The models are nearly strand-blind where the biology is not

AUC(forward) minus AUC(reverse complement), probe trained forward, clustered split:

| task | best trivial | NT | HyenaDNA |
|---|---|---|---|
| **splice_sites_all** | **+0.248** | +0.037 | +0.011 |
| promoter_tata | +0.198 | +0.097 | +0.033 |
| promoter_all | +0.008 | +0.031 | +0.021 |
| H3K4me3 | +0.002 | +0.021 | +0.022 |
| enhancers | +0.027 | +0.033 | +0.016 |

A splice site is strictly strand-oriented. One-hot correctly collapses by 0.248
when the strand is flipped; NT loses 0.037 and HyenaDNA 0.011. On the
reverse-complement task NT **beats** one-hot by +0.047, and that is a failure
rather than a win: it scores the reverse complement of a splice donor almost as
highly as the donor. The control holds, because on the two non-oriented tasks
(enhancers, H3K4me3) trivial and model asymmetries agree at about 0.02, so this is
not an artifact of the protocol.

This independently reproduces the "strand-blind" finding of the Mechanistic
Invariance Test (arXiv:2604.06549), on different tasks and a different protocol.

### Coverage reproduces section 23, on both sides

At a 1% false-positive budget, true positive rate and locus coverage with the
longest run of consecutive missed positives:

| task | best trivial | NT | HyenaDNA |
|---|---|---|---|
| enhancers | 0.057 / 0.062 / 76 | 0.089 / **0.101** / 58 | 0.081 / 0.095 / 78 |
| H3K4me3 | 0.217 / 0.261 / 20 | 0.232 / 0.277 / 20 | 0.214 / 0.254 / 21 |
| splice_sites_all | 0.446 / 0.508 / 7 | 0.064 / 0.079 / 90 | 0.016 / **0.020** / 256 |

**NT's +0.029 AUC win on enhancers buys four points of locus coverage and leaves a
screen that finds 10% of positive-bearing loci while missing 58 in a row.**
HyenaDNA on splice sites is the extreme: AUC 0.704 reads as weak but real, and
locus coverage is 0.020 with a contiguous hole 256 positives long.

### What is reassuring

Split choice barely matters on these benchmarks. Shipped against clustered differs
by at most 0.012 on four of five tasks for both models, with promoter_tata the
outlier for everyone (+0.018 to +0.033). No model win depends on the split. Unlike
ParD3 in section 15, these benchmarks are not badly leaky, and that is worth saying
plainly.

### Scope

Two small models, 50M and roughly 1.6M. Nothing here licenses a claim about
NT-500M or Evo 2. The windowed readout is post-hoc and was run on two of five
tasks; it should be pre-registered and run on all five before being quoted as a
headline number.

## 35. The context dose-response reproduces on shuffled flanks, in both directions

Section 19 could not separate two readings: genomic context genuinely fails to help
Nucleotide Transformer, or NT cannot use context past its roughly 12 kb window and
the sweep was measuring the window. Recursive decomposition in the manner of
[Zhang2025recursive] was built to separate them
(`scripts/recursive_context.py`, `run_recursive_context*.py`).

It did not separate them. It found something more useful: **the measurement the
question was posed against is not stable.**

### The control that decides it

Direct scorer, section 19's own flank grid, real chromosome against
composition-matched shuffles, three seeds each:

| flank content | trend rho | per-seed |
|---|---|---|
| **real chromosome** | **-0.771** | |
| mononucleotide-shuffled | **-0.633** | -0.700, -0.500, -0.700 |
| dinucleotide-shuffled | -0.333 | -0.100, -0.300, -0.600 |

**Section 19's negative dose-response is substantially reproduced by flanks that
carry no genomic information at all.** It is a length and composition effect on the
score scale, not evidence that real genomic context specifically hurts. Section
19's conclusion survives, since context does not rescue NT either way, but **its
stated mechanism does not** and the paragraph asserting that supplying real genomic
context degrades the result should not be quoted.

The trend is also grid-dependent: -0.771 on section 19's grid, **-0.430** on a
denser one, with the curve turning back up past 3 kb.

### The recursion is faithful at one chunk and not past it

| flank | chunks | rho(recursive, direct) | AUC direct | AUC recursive |
|---|---|---|---|---|
| 1,440 | 1 | **1.000** | 0.373 | 0.373 |
| 2,880 | 2 | 0.859 | 0.360 | 0.348 |
| 4,320 | 3 | 0.861 | 0.383 | 0.329 |
| 5,760 | 4 | 0.840 | 0.409 | 0.321 |

Exact at one chunk, so the plumbing is right. Past that the recursive score is only
a rho 0.85 stand-in and the AUC gap widens monotonically. **Everything the
recursion reports beyond NT's window is therefore suspect**, and the sweep out to
80.9 kb (13,487 tokens, 6.6x the window) should not be read as a measurement of
long-range context. The agent reported this rather than proceeding, which is the
right call.

### Two references that set the level

An **order-5 Markov chain** through the identical pipeline returns AUC 0.3845 at
every flank, every mode and every transform, to machine precision. It provably
cannot see the flank, so the pipeline injects no flank dependence, which validates
the harness. It also sets the ceiling: NT's recursive AUC of 0.299 to 0.373 sits
**at or below a model that sees five nucleotides** at every flank past 2,880. The
limit here is the model, not the context mechanism, which is what section 32 would
predict.

**HyenaDNA** holds 30 kb natively and needs no recursion. Its real trend is
**+0.536**, which looks like the Evo 2 direction. Its shuffled twins trend +0.893,
+0.750, +0.750 and +0.750, reaching the same or higher AUC; real minus shuffled is
**-0.008**, real higher in 1 of 6. **A model that genuinely holds 30 kb produces a
clean positive dose-response from flanks containing no information whatsoever.**

### What this establishes

Context dose-response trends on this landscape are reproducible in both directions
by composition-matched noise: negative at 12 kb through NT, positive at 30 kb
through HyenaDNA. The whole comparison also sits in a no-signal regime, every arm
between 0.27 and 0.51 against a 0.664 mutation-count baseline.

Two honest limits on the recursion itself. Chunk i>0 splices distal chromosome
directly against the CDS at a junction that does not occur in nature, so what is
excluded is "NT can use distal context delivered this way", not the general claim.
And the analogy to [Zhang2025recursive] is partial: their root model adaptively
chooses what to examine, whereas a masked LM offers no interface for that, so this
is the map-reduce skeleton without the agent.

## 36. The 500M memorisation arm: much better at DNA, and an apparent membership signal that does not survive matching

Section 32 measured NT-v2 50M. The 500M run was collected at the same time and is
analysed here (`results/memorisation_dna_nt500m.csv`, 11,947 reconstructions over
280 windows). Two things change with scale and they point in opposite directions.

### Scale buys a great deal at the pretraining objective

| model | per-nt accuracy, L=1 | margin over an order-5 Markov chain |
|---|---|---|
| NT-v2 50M | 0.3244 | **+0.018** |
| NT-v2 500M | 0.4372 | **+0.131** |

The 500M model beats a 5-mer lookup by **seven times** the margin the 50M model
manages. Section 32's line that a genomic model "barely beats an order-5 Markov
chain" is a statement about 50M, not about the family, and should be qualified
wherever it is quoted. Scale does buy real competence at modelling DNA. What
section 31 showed is separate: that competence does not convert into
protein-fitness signal along the NT scaling line.

### The apparent membership signal

Unlike at 50M, the all-bacteria comparison shows something:

| measure | member | non-member | difference | p |
|---|---|---|---|---|
| per-nt accuracy, L=1 | 0.4372 | 0.4039 | +0.0333 | 0.22 |
| log p(true), L=1 | -6.5684 | -7.0755 | **+0.5071** | **0.07** |
| loss-based membership inference, L=1 | | | **AUC 0.664** (0.676 ratio-corrected) | |

At 50M the same attack returned AUC 0.522. So on its face, membership becomes
detectable at 500M.

### It does not survive composition matching, and the matching failed

**The GC pools are not matched in this run.** The member pool averages GC 0.6030
against the non-member pool's 0.5531, a five-point gap. The 50M run matched to
0.5763 against 0.5768, four thousandths. Genomic models are strongly
composition-sensitive, so a five-point GC gap is a live alternative explanation for
every number in the table above.

The genus-matched comparisons control composition properly, and they show nothing:

| comparison | per-nt difference, L=1 | membership inference AUC |
|---|---|---|
| M. albiziae (member) vs Mesorhizobium 2026 (non) | **-0.0097** | |
| M. albiziae (member) vs CP002279 (non) | +0.0028 | |
| genus-matched, pooled | | **0.462 raw, 0.509 corrected** |

The member genome is *worse* than its non-member relative on one comparison and
indistinguishable on the other, and the membership attack falls to chance. **The
signal appears only in the comparison whose composition matching failed, and
disappears in the comparison where it holds.**

Power also fell: the 500M membership test uses 40 windows per arm against 136 at
50M, so its intervals are much wider even before the confound is considered.

### What to say and not say

Say: at 500M there is no confound-free evidence of training-set memorisation, and
the apparent signal in the unmatched comparison is most plausibly composition.
There is still no verbatim recall, with exact span recovery at approximately zero
for every arm beyond a single token.

Do not say: that memorisation has been excluded at 500M. The genus-matched
comparison rests on a **single** member genome and 40 windows per arm, which is too
thin to exclude a moderate effect. The honest position is that the question is open
at 500M and answered only at 50M, and closing it needs a properly GC-matched pool
at the larger scale rather than a larger model.

This is the same failure mode as sections 26 and 35 in a third place: a signal that
looks real until the matched control is applied, and this time the control was
present in the design but degraded in execution.

## 38. Evo 2's context arm never had specificity signal to explain

Section 35 left one item outstanding: Evo 2's context dose-response was reported as
a +0.70 trend with no composition-matched control, after shuffled flanks had
reproduced both NT-v2's negative trend and HyenaDNA's positive one. The control is
now built (`scripts/make_evo2_context_control.py`, which emits a standalone
`evo2_context_control.py` for a CUDA host) and validated against the published arm:
it reproduces all five AUCs to 1e-12 and its real flanks are bit-exact slices of
CP002279. It has not been run, because Evo 2 7B needs a GPU this machine does not
have.

Three checks on the per-variant score arrays, which needed no new GPU time, change
what the control is for.

**No arm carries specificity signal above chance.** On the same 399-variant
discrimination set (235 specific, 164 promiscuous):

| flank | AUC | 95% CI | vs chance |
|---|---|---|---|
| 0 nt | 0.4006 | [0.345, 0.456] | **below** chance |
| 300 nt | 0.5211 | [0.465, 0.578] | indistinguishable |
| 1200 nt | 0.5374 | [0.481, 0.593] | indistinguishable |
| 3000 nt | 0.5426 | [0.486, 0.599] | indistinguishable |
| 5400 nt | 0.5368 | [0.481, 0.593] | indistinguishable |

The bare CDS is significantly *anti*-predictive. Every flanked arm is
indistinguishable from chance. So the dose-response runs from anti-predictive to
chance, and no arm on the curve has specificity signal that context could be
credited with supplying.

**The +0.70 is one step, and it is not significant.** Exact permutation over all
120 orderings of five points gives p = 0.1167. Dropping the 0 nt point, the trend
among arms that actually have context is +0.400 with p = 0.375 over 24 orderings.
Paired bootstrap on the same resamples: 300 vs 0 nt is +0.1204 [+0.091, +0.150],
5400 vs 300 nt is +0.0157 [+0.0007, +0.0313], and 5400 vs 1200 nt is -0.0006
[-0.016, +0.015]. An 18x increase in context past the first increment buys +0.016
of AUC and then nothing.

**300 nt and 5400 nt are the same measurement.** Per-variant score arrays
correlate at rho +0.963 across all 7,882 variants, and the flanked arms sit at
+0.959 to +0.981 with each other. Whatever the flank contributes saturates
immediately.

**The AUC change is the non-cognate correlation falling, not the cognate one
rising.** rho with cognate fitness is flat at +0.47 across every flank including
zero (+0.473, +0.490, +0.466, +0.467, +0.468). rho with non-cognate fitness drops
+0.363 to +0.240 at the first increment and then sits flat (+0.206, +0.216,
+0.219). Adding a prefix decouples the score from the wrong partner's fitness
while leaving the right partner's untouched.

**Therefore the control now tests a narrower thing, and one that matters less.**
Not "does long-range genomic context carry specificity signal", which is answered
no by the chance test above, but "is the single 300 nt step genomic at all". Point
four is the reason to expect it is not: the effect appears in full at the shortest
flank and is a decoupling rather than an acquisition, which is what having any
prefix to condition on would do. Until the control runs, **do not cite Evo 2's
+0.70 as a context dose-response.** The defensible statement is that no Evo 2
context arm discriminates specificity above chance, which does not depend on the
control at all.

The job scores the 399 discrimination variants instead of all 7,882, a 20x saving
that changes no reported number because the published AUCs are read off that subset
already: 21 arms, 8,379 sequences, 41.9M tokens, against 167M for the original
context arms. The runner and its analyzer live with the ParD3 landscape they need,
which this repository does not vendor. The table is
`results/evo2_context_analysis.csv`.
