"""The four controls, as functions over arrays. No model, no biology.

A claim of the form "model f, which scores inputs in representation A, captures
a property P defined on modality B" is testable whenever A and B are related by
a known many-to-one map. This module implements the four checks that decide it,
taking plain arrays of scores so that any model and any modality can be put
through them: paraphrase sets and augmentation orbits supply fibres as readily
as synonymous codons do.

    from biologos.validity import noise_ceiling, paired_contrast, map_control

    ceiling = noise_ceiling(fibres)          # what the measurement can show
    result  = paired_contrast(f, baseline)   # is f better than the cheap thing
    leak    = map_control(canonical, frameshift=..., reverse=...)

Each returns a dataclass whose fields are the numbers, and whose `verdict` says
what they license in a sentence. Nothing here knows what a protein is.

Design notes, since they are the part that matters:

* Intervals are t intervals; the bootstrap is used only for the equivalence
  bound, where a percentile interval is what the TOST argument needs.
* p-values are two-sided sign-flip permutation tests. At the sample sizes these
  controls are usually run at, a normality assumption does real work and
  exchangeability of sign does not.
* `paired_contrast` takes an optional `clusters`. Correlated units are the
  commonest way a protocol like this is defeated, so the aggregation is in the
  function rather than left to the caller.
* Nothing returns a bare "significant" flag. Every result carries the minimum
  effect it could have detected, because a null without that number is not a
  null -- it is an unpowered measurement, and the two should never be reported
  in the same words.

numpy only, no scipy: these functions are meant to drop into a repository that
has whatever it has.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

__all__ = ["Ceiling", "PairedContrast", "MapControl", "TwinContrast",
           "noise_ceiling", "attenuated_max", "paired_contrast", "map_control",
           "context_twin", "minimum_detectable_effect", "tcrit"]

# z(0.975) + z(0.80): the two-sided 80%-power constant.
_POWER_Z = 2.801582
_N_PERM = 20000
_N_BOOT = 20000


def tcrit(df: int) -> float:
    """Two-sided 97.5th percentile of Student t, by Hill's (1970) inversion.

    Local so this module keeps no scipy dependency. Agrees with
    scipy.stats.t.ppf to better than 1e-4 for df >= 3.
    """
    if df <= 0:
        return float("nan")
    exact = {1: 12.7062047, 2: 4.30265273, 3: 3.18244631, 4: 2.77644511}
    if df in exact:
        return exact[df]
    a = 1.0 / (df - 0.5)
    b = 48.0 / (a * a)
    c = ((20700 * a / b - 98) * a - 16) * a + 96.36
    d = ((94.5 / (b + c) - 3.0) / b + 1.0) * np.sqrt(a * np.pi / 2) * df
    x = d * 0.05
    y = x ** (2.0 / df)
    if y > 0.05 + a:
        x = 1.95996398454
        y = x * x
        y = (((((0.4 * y + 6.3) * y + 36.0) * y + 94.5) / b - y - 3.0) / b + 1.0) * x
        y = a * y * y
        y = np.expm1(y) if y <= 0.002 else np.exp(y) - 1.0
    else:
        y = ((1.0 / (((df + 6.0) / (df * y) - 0.089 * d - 0.822)
                     * (df + 2.0) * 3.0) + 0.5 / (df + 4.0)) * y - 1.0) \
            * (df + 1.0) / (df + 2.0) + 1.0 / y
    return float(np.sqrt(df * y))


@dataclass
class Ceiling:
    """What a measurement can show, read off the model's own invariance."""
    kappa: float
    sigma_between: float
    sigma_within: float
    n_fibres: int

    def attenuated(self, rho_true: float) -> float:
        """The largest correlation a true rho_true could appear as."""
        return self.kappa * rho_true

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.kappa):
            return "no ceiling: fibres carry no between-group variance"
        if self.kappa >= 0.95:
            return f"kappa {self.kappa:.3f}: the measurement is near-lossless"
        if self.kappa >= 0.7:
            return (f"kappa {self.kappa:.3f}: a true correlation of 0.5 could appear "
                    f"no larger than {0.5 * self.kappa:.2f}")
        return (f"kappa {self.kappa:.3f}: heavily attenuated, a null here is weak "
                f"evidence of absence")


def noise_ceiling(fibres: Sequence[Sequence[float]]) -> Ceiling:
    """Attenuation ceiling from variation inside groups that must share a label.

    `fibres` is a sequence of groups; every member of a group is an input the
    map sends to the same label, so any spread within a group is variance that
    cannot track the label. Synonymous encodings of one protein variant are a
    fibre; so are paraphrases of one sentence, or augmentations of one image.

    kappa = sqrt(var_between / (var_between + var_within)), and a true
    correlation rho can appear no larger than kappa * rho.
    """
    groups = [np.asarray(g, dtype=float) for g in fibres if len(g) > 0]
    if len(groups) < 2:
        raise ValueError("need at least two fibres to separate the variances")
    means = np.array([g.mean() for g in groups])
    var_between = means.var(ddof=1)
    within = [g.var(ddof=1) for g in groups if len(g) > 1]
    var_within = float(np.mean(within)) if within else 0.0
    total = var_between + var_within
    kappa = float(np.sqrt(var_between / total)) if total > 0 else float("nan")
    return Ceiling(kappa, float(np.sqrt(var_between)), float(np.sqrt(var_within)),
                   len(groups))


def attenuated_max(rho_true: float, kappa: float) -> float:
    """Largest observable correlation given a true value and a ceiling."""
    return kappa * rho_true


def minimum_detectable_effect(sd: float, n: int, power: float = 0.80) -> float:
    """Smallest paired difference detectable at `power`, two-sided alpha 0.05."""
    if power != 0.80:
        raise ValueError("only 80% power is tabulated")
    return _POWER_Z * sd / np.sqrt(n)


@dataclass
class PairedContrast:
    """One paired comparison, with everything needed to read it honestly."""
    n: int
    mean: float
    ci_lo: float
    ci_hi: float
    n_positive: int
    p_permutation: float
    mde_80: float
    equivalence_bound: float
    unit: str = "observation"

    @property
    def contains_zero(self) -> bool:
        return self.ci_lo <= 0.0 <= self.ci_hi

    @property
    def verdict(self) -> str:
        if not self.contains_zero:
            return (f"difference of {self.mean:+.4f} over {self.n} {self.unit}s, "
                    f"p={self.p_permutation:.4f}")
        return (f"no difference detected ({self.mean:+.4f}, p={self.p_permutation:.4f}); "
                f"this test could not have caught an effect below {self.mde_80:.4f}, "
                f"and equivalence holds only out to a margin of "
                f"{self.equivalence_bound:.4f}")


def paired_contrast(a, b, clusters=None, *, unit: str = "observation",
                    seed: int = 0) -> PairedContrast:
    """Paired difference a - b, with interval, permutation p, power and bound.

    `clusters`, when given, is one label per observation. Observations sharing a
    label are averaged before any statistic is computed, because correlated
    units are the commonest way a comparison like this is inflated. The reported
    n is then the number of clusters.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"a and b must be paired: {a.shape} vs {b.shape}")
    d = a - b
    if clusters is not None:
        clusters = np.asarray(clusters)
        if clusters.shape != d.shape:
            raise ValueError("clusters must have one label per observation")
        d = np.array([d[clusters == c].mean() for c in dict.fromkeys(clusters.tolist())])
        unit = "cluster"
    n = len(d)
    if n < 2:
        raise ValueError("need at least two units")
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t = tcrit(n - 1)
    rng = np.random.default_rng(seed)
    null = (d * rng.choice([-1.0, 1.0], size=(_N_PERM, n))).mean(1)
    p = float((np.abs(null) >= abs(m) - 1e-15).mean())
    bs = d[rng.integers(0, n, size=(_N_BOOT, n))].mean(1)
    lo90, hi90 = np.percentile(bs, 5), np.percentile(bs, 95)
    return PairedContrast(n=n, mean=m, ci_lo=m - t * se, ci_hi=m + t * se,
                          n_positive=int((d > 0).sum()), p_permutation=p,
                          mde_80=_POWER_Z * sd / np.sqrt(n),
                          equivalence_bound=float(max(abs(lo90), abs(hi90))),
                          unit=unit)


@dataclass
class MapControl:
    """What survives when the representation map is destroyed."""
    canonical: float
    destroyed: Mapping[str, float] = field(default_factory=dict)

    @property
    def retained(self) -> dict:
        """Fraction of the canonical score each manipulation leaves behind."""
        if self.canonical == 0:
            return {k: float("nan") for k in self.destroyed}
        return {k: v / self.canonical for k, v in self.destroyed.items()}

    @property
    def max_retained(self) -> float:
        r = self.retained
        return max(r.values()) if r else float("nan")

    @property
    def verdict(self) -> str:
        if not self.destroyed:
            return "no map-destroying control was run"
        worst, frac = max(self.retained.items(), key=lambda kv: kv[1])
        if frac >= 0.8:
            return (f"signal survives {worst} at {frac:.0%} of canonical: the "
                    f"representation is invariant to destroying the map, so it is "
                    f"not at the claimed level")
        if frac >= 0.4:
            return (f"signal partly survives {worst} at {frac:.0%}: the claim is "
                    f"contaminated, not clean")
        return (f"every manipulation removes the signal, the largest residue being "
                f"{worst} at {frac:.0%}: consistent with a genuine claim")


def map_control(canonical: float, **destroyed: float) -> MapControl:
    """Compare a canonical score against scores with the map destroyed.

    Every keyword is one manipulation that must destroy the property while
    leaving the input inside the model's training distribution. A manipulation
    that also shifts the input distribution confounds a drop with that shift and
    is not admissible evidence, however large the drop.
    """
    return MapControl(float(canonical), {k: float(v) for k, v in destroyed.items()})


@dataclass
class TwinContrast:
    """A real intervention against a matched twin that carries no information."""
    real: float
    twin: float

    @property
    def delta(self) -> float:
        return self.real - self.twin

    @property
    def verdict(self) -> str:
        if self.real == 0:
            return "no real effect to explain"
        share = self.twin / self.real
        if share >= 0.8:
            return (f"the matched twin reproduces {share:.0%} of the effect: what was "
                    f"measured is composition, not the intervention")
        if share >= 0.4:
            return f"the twin reproduces {share:.0%} of the effect: partly composition"
        return (f"the twin reproduces only {share:.0%}: the effect survives its "
                f"composition control")


def context_twin(real: float, twin: float) -> TwinContrast:
    """Effect measured with real context, against a composition-matched twin."""
    return TwinContrast(float(real), float(twin))
