"""Tests for the expansion schedules and the late-window instrumentation.

The experiment's whole risk is that fast expansion produces a field that
*looks* alive by a survival test while actually having decoupled. So the tests
care most about two things: that the schedule really is a diluting Laplacian
and nothing more, and that the late-window measures can tell binding apart
from mere motion.
"""

from __future__ import annotations

import numpy as np
import pytest

from project_genesis.expansion import (
    SCHEDULES,
    diffusion_schedule,
    scale_factor,
)
from project_genesis.survival import run_ensemble, step_batch, seed_batch


# ----------------------------------------------------------------- schedules

def test_every_schedule_starts_at_unity():
    for kind in SCHEDULES:
        assert scale_factor(kind, 0.0, 0.1) == pytest.approx(1.0)


def test_zero_rate_is_a_static_box_for_every_schedule():
    for kind in SCHEDULES:
        for t in (0.0, 10.0, 500.0):
            assert scale_factor(kind, t, 0.0) == pytest.approx(1.0)


def test_flat_never_expands_however_fast_the_rate():
    assert scale_factor("flat", 1000.0, 5.0) == pytest.approx(1.0)


def test_schedules_are_monotone_increasing():
    for kind in ("radiation", "matter", "desitter"):
        a = [scale_factor(kind, t, 0.05) for t in (0, 5, 20, 100, 400)]
        assert all(x <= y for x, y in zip(a, a[1:]))


def test_late_time_power_law_exponents_are_right():
    """a ~ t^(1/2) for radiation, t^(2/3) for matter, checked as a log slope."""
    for kind, want in (("radiation", 0.5), ("matter", 2.0 / 3.0)):
        t1, t2 = 1e5, 1e6
        a1 = scale_factor(kind, t1, 1.0)
        a2 = scale_factor(kind, t2, 1.0)
        slope = np.log(a2 / a1) / np.log(t2 / t1)
        assert slope == pytest.approx(want, abs=1e-3)


def test_desitter_grows_faster_than_any_power_law():
    t = 300.0
    assert scale_factor("desitter", t, 0.05) > scale_factor("matter", t, 0.05)
    assert scale_factor("matter", t, 0.05) > scale_factor("radiation", t, 0.05)


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError):
        scale_factor("bounce", 1.0, 0.1)


# ------------------------------------------------------------------ dilution

def test_diffusion_is_one_over_a_squared():
    d = diffusion_schedule("matter", 0.05, 0.06)
    for step in (0, 50, 500):
        a = scale_factor("matter", step * 0.06, 0.05)
        assert d(step) == pytest.approx(1.0 / (a * a))


def test_diffusion_starts_at_one_and_only_falls():
    d = diffusion_schedule("desitter", 0.05, 0.06)
    vals = [d(s) for s in (0, 10, 100, 500, 1000)]
    assert vals[0] == pytest.approx(1.0)
    assert all(x >= y for x, y in zip(vals, vals[1:]))


def test_flat_schedule_leaves_diffusion_at_one():
    d = diffusion_schedule("flat", 0.0, 0.06)
    assert all(d(s) == pytest.approx(1.0) for s in (0, 100, 5000))


def test_floor_clamps_decoupling_when_asked():
    d = diffusion_schedule("desitter", 0.5, 0.06, floor=0.01)
    assert d(10000) == pytest.approx(0.01)


def test_expansion_only_touches_the_laplacian():
    """D scales the gradient term and nothing else: with a uniform field the
    Laplacian vanishes, so every diffusion value must give the same update."""
    fields = np.full((2, 3, 5, 5, 5), 0.4)
    a = step_batch(fields, diffusion=1.0, gamma=1.5, dt=0.06, ndim=3)
    b = step_batch(fields, diffusion=0.001, gamma=1.5, dt=0.06, ndim=3)
    assert np.allclose(a, b)


def test_lower_diffusion_really_does_slow_the_field():
    rng = np.random.default_rng(0)
    f0 = seed_batch(2, 3, 8, rng, ndim=3)
    fast = step_batch(f0, diffusion=1.0, gamma=1.5, dt=0.06, ndim=3)
    slow = step_batch(f0, diffusion=0.05, gamma=1.5, dt=0.06, ndim=3)
    assert np.abs(fast - f0).mean() > np.abs(slow - f0).mean()


# ------------------------------------------------------- late-window metrics

def test_late_metrics_have_one_value_per_run():
    res = run_ensemble(5, 3, 6, 60, ndim=3, noise=0.01, seed=0, early=10,
                       sample=10)
    for key in ("late_churn", "late_integration", "late_ell"):
        assert res[key].shape == (5,)
        assert np.isfinite(res[key]).all()
    assert res["late_samples"] > 0


def test_late_metrics_are_in_range():
    res = run_ensemble(4, 4, 6, 60, ndim=3, noise=0.02, seed=1, early=10,
                       sample=10)
    assert (res["late_churn"] >= 0).all() and (res["late_churn"] <= 1).all()
    assert (res["late_integration"] >= 0).all()
    assert (res["late_integration"] <= 1).all()
    assert (res["late_ell"] >= 1.0).all()


def test_a_frozen_field_has_zero_churn():
    """No noise and no diffusion: nothing can change sector, so churn is 0."""
    res = run_ensemble(3, 3, 6, 60, ndim=3, noise=0.0, seed=2, early=10,
                       sample=10, diffusion_of_step=lambda s: 0.0)
    assert res["late_churn"].max() == pytest.approx(0.0, abs=1e-9)


def test_decoupling_collapses_the_domain_scale():
    """The trap this experiment exists to avoid: with no spatial coupling the
    palette cannot monopolise, but the domain scale falls to the lattice."""
    coupled = run_ensemble(6, 4, 10, 300, ndim=3, noise=0.02, seed=3, early=10,
                           sample=10)
    decoupled = run_ensemble(6, 4, 10, 300, ndim=3, noise=0.02, seed=3, early=10,
                             sample=10, diffusion_of_step=lambda s: 0.0)
    assert decoupled["late_ell"].mean() < coupled["late_ell"].mean()
    assert decoupled["late_ell"].mean() < 1.5


def test_diffusion_schedule_changes_the_outcome():
    common = dict(ndim=3, noise=0.02, seed=4, early=10, sample=10)
    flat = run_ensemble(6, 4, 10, 300, **common)
    slowed = run_ensemble(6, 4, 10, 300,
                          diffusion_of_step=diffusion_schedule("desitter", 0.2, 0.06),
                          **common)
    assert not np.allclose(flat["late_ell"], slowed["late_ell"])


def test_late_window_respects_late_frac():
    a = run_ensemble(3, 3, 6, 200, ndim=3, noise=0.01, seed=5, early=10,
                     sample=10, late_frac=0.1)
    b = run_ensemble(3, 3, 6, 200, ndim=3, noise=0.01, seed=5, early=10,
                     sample=10, late_frac=0.9)
    assert b["late_samples"] > a["late_samples"]
