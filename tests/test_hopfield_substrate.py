"""Checks for the Hopfield transplant substrate (hopfield_substrate)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.hopfield_substrate import (  # noqa: E402
    best_overlap,
    glauber_sweep,
    hebbian_weights,
    s_functional,
    steady_state_kappa,
    trajectory_label,
    window_metrics,
)


def _patterns(p, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=(p, n)).astype(float)


class TestHebbianNetwork(unittest.TestCase):

    def test_weights_symmetric_zero_diagonal(self):
        j = hebbian_weights(_patterns(3, 40))
        np.testing.assert_allclose(j, j.T)
        np.testing.assert_allclose(np.diag(j), 0.0)

    def test_stored_pattern_is_fixed_point_at_zero_temperature(self):
        # below critical loading, a stored memory is stable under T→0 dynamics
        pats = _patterns(3, 200, seed=1)
        j = hebbian_weights(pats)
        rng = np.random.default_rng(2)
        s = glauber_sweep(pats[0].copy(), j, 1e-9, rng)
        m, idx = best_overlap(s, pats)
        self.assertEqual(idx, 0)
        self.assertGreater(m, 0.98)

    def test_corrupted_pattern_is_repaired(self):
        # attractor dynamics: 10% corruption flows back to the stored memory
        pats = _patterns(2, 200, seed=3)
        j = hebbian_weights(pats)
        rng = np.random.default_rng(4)
        s = pats[0].copy()
        flip = rng.choice(200, size=20, replace=False)
        s[flip] *= -1
        for _ in range(5):
            s = glauber_sweep(s, j, 1e-9, rng)
        m, idx = best_overlap(s, pats)
        self.assertEqual(idx, 0)
        self.assertGreater(m, 0.95)

    def test_high_temperature_destroys_retrieval(self):
        pats = _patterns(2, 200, seed=5)
        j = hebbian_weights(pats)
        rng = np.random.default_rng(6)
        s = pats[0].copy()
        for _ in range(10):
            s = glauber_sweep(s, j, 5.0, rng)
        m, _ = best_overlap(s, pats)
        self.assertLess(m, 0.3)

    def test_glauber_returns_pm_one(self):
        pats = _patterns(2, 50, seed=7)
        j = hebbian_weights(pats)
        s = glauber_sweep(pats[0].copy(), j, 1.0, np.random.default_rng(8))
        self.assertTrue(set(np.unique(s)) <= {-1.0, 1.0})


class TestWindowMetrics(unittest.TestCase):

    def test_frozen_retrieval_has_no_distinction(self):
        # always the same attractor: ΔI high, ΔC exactly zero
        window = [(0.95, 0)] * 20
        out = window_metrics(window, n_patterns=5)
        self.assertAlmostEqual(out["delta_c"], 0.0)
        self.assertAlmostEqual(out["delta_i"], 0.95)

    def test_noise_has_no_distinction(self):
        # nothing structured: both readings near zero
        window = [(0.05, i % 5) for i in range(20)]
        out = window_metrics(window, n_patterns=5)
        self.assertAlmostEqual(out["delta_c"], 0.0)
        self.assertAlmostEqual(out["structured_fraction"], 0.0)

    def test_attractor_hopping_maximises_distinction(self):
        # structured samples spread over all memories: ΔC → structured fraction
        window = [(0.6, i % 5) for i in range(20)]
        out = window_metrics(window, n_patterns=5)
        self.assertAlmostEqual(out["delta_c"], 1.0)
        self.assertAlmostEqual(out["structured_fraction"], 1.0)

    def test_partial_structure_scales_delta_c(self):
        # half the window structured over two memories of four
        window = [(0.6, i % 2) for i in range(10)] + [(0.1, 0)] * 10
        out = window_metrics(window, n_patterns=4)
        expected = (np.log(2) / np.log(4)) * 0.5
        self.assertAlmostEqual(out["delta_c"], expected, places=10)


class TestCapacityAndS(unittest.TestCase):

    def test_kappa_steady_state_matches_engine_law(self):
        # κ = r/(r + c·load): abundant recovery → κ→1, heavy load → κ→0
        self.assertAlmostEqual(steady_state_kappa(0.0, 5.0, 0.5), 1.0)
        self.assertAlmostEqual(steady_state_kappa(1.0, 4.0, 1.0), 0.2)
        self.assertLess(steady_state_kappa(1.0, 100.0, 0.1), 0.01)

    def test_s_functional_form(self):
        self.assertAlmostEqual(s_functional(0.3, 0.8, 0.5, 2.0), 0.3 + 0.8)

    def test_scarcity_taxes_integration_not_distinction(self):
        # the mechanism behind the level crossing, in one assertion:
        # raising consumption lowers the κ·w·ΔI term, leaves ΔC untouched
        s_abundant = s_functional(0.3, 0.8, steady_state_kappa(0.3, 0.1, 1.0), 2.0)
        s_scarce = s_functional(0.3, 0.8, steady_state_kappa(0.3, 50.0, 1.0), 2.0)
        self.assertLess(s_scarce, s_abundant)
        self.assertGreaterEqual(s_scarce, 0.3)   # the ΔC funding survives


class TestTrajectoryTaxonomy(unittest.TestCase):
    """Mirrors the S-compass delta-form labels (s_engine.py, URP repo)."""

    def test_four_quadrants(self):
        self.assertEqual(trajectory_label(0.1, 0.1), "expanding")
        self.assertEqual(trajectory_label(-0.1, 0.1), "consolidating")
        self.assertEqual(trajectory_label(0.1, -0.1), "diverging")
        self.assertEqual(trajectory_label(-0.1, -0.1), "contracting")

    def test_deadband_is_steady(self):
        self.assertEqual(trajectory_label(0.0, 0.0), "steady")
        self.assertEqual(trajectory_label(1e-4, -1e-4), "steady")

    def test_single_live_component(self):
        self.assertEqual(trajectory_label(0.1, 0.0), "expanding")
        self.assertEqual(trajectory_label(0.0, -0.1), "contracting")


if __name__ == "__main__":
    unittest.main()
