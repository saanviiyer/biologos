"""Tests for the four controls.

These check the instruments, not the biology. Each case is constructed so the
right answer is known in advance: a ceiling with a closed form, a difference
that is exactly zero, a signal that is pure leakage. An instrument that cannot
pass these has no business adjudicating anyone's claim.

The last test is a regression against this project's own published numbers, so
that the module and the paper cannot drift apart silently.
"""
import csv
from pathlib import Path

import numpy as np
import pytest

from biologos.validity import (Ceiling, context_twin, map_control,
                               minimum_detectable_effect, noise_ceiling,
                               paired_contrast, tcrit)

ROOT = Path(__file__).resolve().parents[1]


# --- the t quantile ------------------------------------------------------

@pytest.mark.parametrize("df,want", [(1, 12.7062047), (2, 4.30265273),
                                     (10, 2.2281389), (24, 2.0638986),
                                     (100, 1.9839715)])
def test_tcrit_matches_tables(df, want):
    assert tcrit(df) == pytest.approx(want, abs=1e-4)


# --- control 1: the noise ceiling ---------------------------------------

def test_ceiling_matches_closed_form():
    rng = np.random.default_rng(0)
    between, within = 1.0, 0.25
    fibres = [rng.normal(mu, within, 40)
              for mu in rng.normal(0, between, 60)]
    c = noise_ceiling(fibres)
    expected = np.sqrt(between**2 / (between**2 + within**2))
    assert c.kappa == pytest.approx(expected, abs=0.03)
    assert c.n_fibres == 60


def test_ceiling_is_one_when_fibres_are_tight():
    fibres = [[m, m, m] for m in range(20)]
    c = noise_ceiling(fibres)
    assert c.kappa == pytest.approx(1.0)
    assert "near-lossless" in c.verdict


def test_ceiling_flags_an_unpowered_measurement():
    """Within-fibre noise far above between-fibre spread must warn, not pass."""
    rng = np.random.default_rng(1)
    fibres = [rng.normal(mu, 5.0, 30) for mu in rng.normal(0, 0.2, 40)]
    c = noise_ceiling(fibres)
    assert c.kappa < 0.3
    assert "weak evidence of absence" in c.verdict
    # A true correlation of 0.5 is invisible through this instrument.
    assert c.attenuated(0.5) < 0.15


def test_ceiling_needs_two_fibres():
    with pytest.raises(ValueError):
        noise_ceiling([[1.0, 2.0]])


# --- control 4: paired contrasts, power and correlated units ------------

def test_null_contrast_contains_zero_and_reports_its_floor():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 40)
    r = paired_contrast(a, a + rng.normal(0, 0.3, 40))
    assert r.contains_zero
    assert r.p_permutation > 0.05
    assert r.mde_80 > 0
    assert "could not have caught an effect below" in r.verdict


def test_real_effect_is_detected():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 40)
    r = paired_contrast(base + 2.0, base + rng.normal(0, 0.2, 40))
    assert not r.contains_zero
    assert r.p_permutation < 0.001
    assert r.mean == pytest.approx(2.0, abs=0.1)


def test_clusters_shrink_n_and_widen_the_interval():
    """Duplicating every observation must not buy confidence."""
    rng = np.random.default_rng(4)
    d = rng.normal(0.3, 1.0, 20)
    a = np.repeat(d, 3)
    b = np.zeros_like(a)
    labels = np.repeat(np.arange(20), 3)
    naive = paired_contrast(a, b)
    honest = paired_contrast(a, b, clusters=labels)
    assert naive.n == 60 and honest.n == 20
    assert honest.mean == pytest.approx(naive.mean, abs=1e-9)
    width = lambda r: r.ci_hi - r.ci_lo
    assert width(honest) > width(naive)
    assert honest.unit == "cluster"


def test_mde_is_the_standard_formula():
    assert minimum_detectable_effect(1.0, 25) == pytest.approx(2.801582 / 5.0)
    with pytest.raises(ValueError):
        minimum_detectable_effect(1.0, 25, power=0.9)


def test_effect_at_the_mde_is_detected_about_eighty_percent_of_the_time():
    """The reported floor has to mean what it says."""
    rng = np.random.default_rng(5)
    n, sd = 30, 1.0
    effect = 2.801582 * sd / np.sqrt(n)
    hits = 0
    trials = 400
    for _ in range(trials):
        d = rng.normal(effect, sd, n)
        se = d.std(ddof=1) / np.sqrt(n)
        lo = d.mean() - tcrit(n - 1) * se
        hi = d.mean() + tcrit(n - 1) * se
        hits += not (lo <= 0 <= hi)
    assert 0.70 < hits / trials < 0.90


def test_equivalence_bound_grows_with_noise():
    rng = np.random.default_rng(6)
    tight = paired_contrast(rng.normal(0, 0.1, 30), np.zeros(30))
    loose = paired_contrast(rng.normal(0, 2.0, 30), np.zeros(30))
    assert loose.equivalence_bound > 5 * tight.equivalence_bound


def test_paired_contrast_rejects_unpaired_input():
    with pytest.raises(ValueError):
        paired_contrast(np.zeros(5), np.zeros(6))


# --- control 2: map-destroying manipulations ----------------------------

def test_pure_leakage_is_called_out():
    m = map_control(0.141, frameshift=0.143, reverse_complement=0.129)
    assert m.max_retained > 0.9
    assert "not at the claimed level" in m.verdict


def test_a_signal_that_dies_with_the_map_survives_the_control():
    m = map_control(0.50, frameshift=0.02, reverse_complement=-0.01)
    assert m.max_retained < 0.1
    assert "consistent with a genuine claim" in m.verdict


def test_partial_contamination_is_not_called_clean():
    m = map_control(0.40, frameshift=0.22)
    assert "contaminated" in m.verdict


def test_no_control_run_is_reported_as_such():
    assert "no map-destroying control was run" in map_control(0.3).verdict


# --- control 3: composition-matched twins -------------------------------

def test_twin_reproducing_the_effect_means_composition():
    t = context_twin(real=-0.771, twin=-0.633)
    assert "composition, not the intervention" in t.verdict


def test_effect_surviving_its_twin():
    t = context_twin(real=0.60, twin=0.05)
    assert "survives its composition control" in t.verdict


# --- regression against this project's published numbers ----------------

def _panel():
    base = list(csv.DictReader((ROOT / "results/dms_trivial_baselines.csv").open()))
    evo = {r["dms_id"]: float(r["rho_evo2"])
           for r in csv.DictReader((ROOT / "results/evo2_dms.csv").open())}
    rows = [r for r in base if r["dms_id"] in evo]
    return rows, evo


@pytest.mark.skipif(not (ROOT / "results/evo2_dms.csv").exists(),
                    reason="result files not present")
def test_reproduces_the_published_evo2_contrast():
    rows, evo = _panel()
    a = [evo[r["dms_id"]] for r in rows]
    b = [float(r["rho_blosum62"]) for r in rows]
    r = paired_contrast(a, b)
    assert r.n == 25
    assert r.mean == pytest.approx(0.0376, abs=5e-5)
    assert r.n_positive == 14
    assert r.contains_zero

    clustered = paired_contrast(a, b, clusters=[r_["cluster"] for r_ in rows])
    assert clustered.n == 21
    assert clustered.mean == pytest.approx(0.0398, abs=5e-5)
    assert clustered.mde_80 == pytest.approx(0.0835, abs=5e-4)
    assert clustered.equivalence_bound == pytest.approx(0.0884, abs=3e-3)


@pytest.mark.skipif(not (ROOT / "results/transfer_analysis.csv").exists(),
                    reason="result files not present")
def test_reproduces_the_published_map_control():
    ctrl = {r["a"]: float(r["rho_within_bg"])
            for r in csv.DictReader((ROOT / "results/transfer_analysis.csv").open())
            if r["stage"] == "causal_control"}
    m = map_control(ctrl["canonical"], frameshift=ctrl["frameshift+1"],
                    reverse_complement=ctrl["reverse_complement"])
    assert m.canonical == pytest.approx(0.1413, abs=5e-5)
    assert m.max_retained > 1.0           # frameshift scores HIGHER than canonical
    assert "not at the claimed level" in m.verdict
