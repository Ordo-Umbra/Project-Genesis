"""Tests for the capacity-gated selection path.

The experiment's conclusion depends on one thing above all: that the
no-capacity control arm really is the published experiment, not a
re-implementation that happens to look like it. If those two paths ever
diverge, the comparison across the capacity axis is comparing two different
models and means nothing.
"""

from __future__ import annotations

import numpy as np

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.multiphase import step_multiphase, step_multiphase_kappa
from project_genesis.selection import (
    evolve_palette,
    evolve_palette_gated,
    make_stepper,
    measure_window,
    measure_window_gated,
)


# ------------------------------------------------- the control arm is honest


class TestCapacityGating(unittest.TestCase):

    def test_control_arm_is_bit_identical_to_the_published_path(self):
        """The c=None arm must BE n3_selection_sweep's evolution, not a lookalike."""
        a = evolve_palette(3, 20, 25, np.random.default_rng(11), noise=0.25, ndim=2)
        b, kappa = evolve_palette_gated(3, 20, 25, np.random.default_rng(11),
                                        consumption=None, noise=0.25, ndim=2)
        assert np.array_equal(a, b)
        assert kappa is None


    def test_control_arm_matches_in_3d_too(self):
        a = evolve_palette(4, 10, 15, np.random.default_rng(12), noise=0.1, ndim=3)
        b, _ = evolve_palette_gated(4, 10, 15, np.random.default_rng(12),
                                    consumption=None, noise=0.1, ndim=3)
        assert np.array_equal(a, b)


    def test_control_measure_window_matches_the_published_one(self):
        rng_a = np.random.default_rng(13)
        fields_a = evolve_palette(3, 20, 20, rng_a, noise=0.25, ndim=2)
        m_a = measure_window(fields_a, 3, 6, rng_a, noise=0.25)

        rng_b = np.random.default_rng(13)
        fields_b, kappa_b = evolve_palette_gated(3, 20, 20, rng_b, consumption=None,
                                                 noise=0.25, ndim=2)
        m_b = measure_window_gated(fields_b, kappa_b, 3, 6, rng_b, consumption=None,
                                   noise=0.25)
        for key in ("distinction", "integration", "integration_guarded",
                    "churn", "domain_scale"):
            self.assertAlmostEqual(m_a[key], m_b[key], delta=abs(m_b[key]) * (1e-12) + 1e-12)


    # ------------------------------------------------------------ the stepper


    def test_stepper_without_capacity_is_the_plain_law(self):
        rng = np.random.default_rng(1)
        fields = rng.uniform(0, 1, (3, 8, 8))
        step, init = make_stepper(None, gamma=1.5, dt=0.1)
        out, kappa = step(fields, init((8, 8)))
        assert np.array_equal(out, step_multiphase(fields, gamma=1.5, dt=0.1))
        assert kappa is None


    def test_stepper_with_capacity_is_the_repo_capacity_law(self):
        rng = np.random.default_rng(2)
        fields = rng.uniform(0, 1, (3, 8, 8))
        k0 = np.ones((8, 8))
        step, init = make_stepper(5.0, gamma=1.5, dt=0.1)
        got_f, got_k = step(fields, k0)
        want_f, want_k = step_multiphase_kappa(
            fields, k0, gamma=1.5, dt=0.1, kappa_baseline=1.0,
            kappa_recovery=0.1, kappa_consumption=5.0, kappa_diffusion=0.5)
        assert np.array_equal(got_f, want_f)
        assert np.array_equal(got_k, want_k)


    def test_capacity_field_starts_at_baseline(self):
        _, init = make_stepper(1.0, baseline=0.7)
        assert np.allclose(init((5, 5)), 0.7)


    # ------------------------------------------------- the knob has to do work


    def test_free_capacity_stays_pinned_at_baseline(self):
        """c = 0 means load costs nothing, so kappa must not deplete anywhere."""
        _, kappa = evolve_palette_gated(3, 16, 40, np.random.default_rng(3),
                                        consumption=0.0, noise=0.2, ndim=2)
        self.assertAlmostEqual(kappa.mean(), 1.0, delta=1e-6)
        self.assertAlmostEqual(kappa.min(), 1.0, delta=1e-6)


    def test_binding_budget_actually_depletes_capacity(self):
        _, kappa = evolve_palette_gated(3, 16, 40, np.random.default_rng(3),
                                        consumption=20.0, noise=0.2, ndim=2)
        assert kappa.mean() < 0.8
        assert kappa.min() < kappa.max()          # depleted where load is dense


    def test_scarcity_is_monotone_in_the_consumption_knob(self):
        means = []
        for c in (0.0, 1.0, 5.0, 20.0):
            _, k = evolve_palette_gated(3, 16, 40, np.random.default_rng(4),
                                        consumption=c, noise=0.2, ndim=2)
            means.append(float(k.mean()))
        assert means == sorted(means, reverse=True)


    def test_capacity_never_leaves_its_bounds(self):
        _, kappa = evolve_palette_gated(4, 16, 60, np.random.default_rng(5),
                                        consumption=20.0, noise=0.25, ndim=2)
        assert kappa.min() >= 0.0
        assert kappa.max() <= 1.0 + 1e-9


    # ------------------------------------------------------------- measurement


    def test_measure_reports_realised_capacity_not_the_nominal_setting(self):
        rng = np.random.default_rng(6)
        fields, kappa = evolve_palette_gated(3, 16, 40, rng, consumption=20.0,
                                             noise=0.2, ndim=2)
        m = measure_window_gated(fields, kappa, 3, 5, rng, consumption=20.0,
                                 noise=0.2)
        assert 0.0 <= m["kappa_mean"] <= 1.0
        assert m["kappa_mean"] < 1.0            # it really did deplete


    def test_control_arm_reports_unit_capacity(self):
        rng = np.random.default_rng(7)
        fields, kappa = evolve_palette_gated(3, 16, 30, rng, consumption=None,
                                             noise=0.2, ndim=2)
        m = measure_window_gated(fields, kappa, 3, 5, rng, consumption=None,
                                 noise=0.2)
        self.assertAlmostEqual(m["kappa_mean"], 1.0, places=7)


    def test_gated_measure_carries_the_texture_guard(self):
        rng = np.random.default_rng(8)
        fields, kappa = evolve_palette_gated(3, 20, 40, rng, consumption=5.0,
                                             noise=0.25, ndim=2)
        m = measure_window_gated(fields, kappa, 3, 5, rng, consumption=5.0,
                                 noise=0.25)
        assert "resolved" in m and "domain_scale" in m
        assert m["integration_guarded"] <= m["integration"] + 1e-9


if __name__ == "__main__":
    unittest.main()
