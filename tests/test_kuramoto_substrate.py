"""Checks for the Kuramoto transplant substrate (kuramoto_substrate)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.kuramoto_substrate import (  # noqa: E402
    natural_frequencies,
    order_parameter,
    kuramoto_step,
    simulate,
    window_metrics,
    steady_state_kappa,
    s_functional,
    trajectory_label,
)
from project_genesis import hopfield_substrate  # noqa: E402


class TestNaturalFrequencies(unittest.TestCase):

    def test_zero_spread_is_identical(self):
        w = natural_frequencies(50, 0.0, np.random.default_rng(0))
        np.testing.assert_allclose(w, 0.0)

    def test_centred_and_scales_with_spread(self):
        rng = np.random.default_rng(1)
        narrow = natural_frequencies(20000, 0.3, rng, distribution="gaussian")
        wide = natural_frequencies(20000, 1.2, rng, distribution="gaussian")
        self.assertLess(abs(narrow.mean()), 0.05)          # zero-centred
        self.assertGreater(wide.std(), 3.0 * narrow.std())  # scales with spread

    def test_lorentzian_available(self):
        w = natural_frequencies(1000, 0.5, np.random.default_rng(2),
                                distribution="lorentzian")
        self.assertEqual(w.shape, (1000,))
        self.assertLess(abs(np.median(w)), 0.2)   # heavy tails, but centred


class TestOrderParameter(unittest.TestCase):

    def test_perfectly_aligned_is_one(self):
        r, _ = order_parameter(np.full(64, 0.7))
        self.assertAlmostEqual(r, 1.0)

    def test_uniformly_spread_is_near_zero(self):
        phases = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        r, _ = order_parameter(phases)
        self.assertLess(r, 1e-2)

    def test_bounded(self):
        rng = np.random.default_rng(3)
        for _ in range(10):
            r, _ = order_parameter(rng.uniform(-np.pi, np.pi, 200))
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)


class TestDynamics(unittest.TestCase):

    def _run(self, coupling, spread, noise=0.0, n=120, steps=400, seed=0):
        rng = np.random.default_rng(seed)
        omega = natural_frequencies(n, spread, rng, distribution="gaussian")
        phases = rng.uniform(-np.pi, np.pi, n)
        return simulate(phases, omega, coupling, 0.05, steps, noise, rng,
                        burn_in=steps // 2)

    def test_strong_coupling_synchronises(self):
        sim = self._run(coupling=3.0, spread=0.2)
        self.assertGreater(np.mean(sim["r_samples"]), 0.85)

    def test_weak_coupling_stays_incoherent(self):
        sim = self._run(coupling=0.2, spread=2.0)
        self.assertLess(np.mean(sim["r_samples"]), 0.3)

    def test_deterministic_locked_state_is_low_activity(self):
        # the toggle's OFF state: a synchronised rhythm that just rotates pays
        # essentially no co-rotating reconfiguration
        sim = self._run(coupling=3.0, spread=0.2, noise=0.0)
        self.assertLess(sim["activity"], 0.01)

    def test_noise_drive_raises_activity(self):
        # the toggle's ON state: holding the same rhythm against noise costs work
        sim = self._run(coupling=3.0, spread=0.2, noise=0.6)
        self.assertGreater(sim["activity"], 0.05)

    def test_step_shape_and_finiteness(self):
        rng = np.random.default_rng(4)
        omega = natural_frequencies(30, 0.5, rng)
        phases = rng.uniform(-np.pi, np.pi, 30)
        nxt = kuramoto_step(phases, omega, 1.0, 0.05, 0.1, rng)
        self.assertEqual(nxt.shape, phases.shape)
        self.assertTrue(np.all(np.isfinite(nxt)))


class TestWindowMetrics(unittest.TestCase):

    def _run(self, coupling, spread, noise=0.0, n=200, steps=500, seed=0):
        rng = np.random.default_rng(seed)
        omega = natural_frequencies(n, spread, rng, distribution="gaussian")
        phases = rng.uniform(-np.pi, np.pi, n)
        return simulate(phases, omega, coupling, 0.05, steps, noise, rng,
                        burn_in=steps // 2)

    def test_deep_synchrony_has_high_integration_low_distinction(self):
        out = window_metrics(self._run(coupling=3.0, spread=0.2))
        self.assertGreater(out["delta_i"], 0.85)
        self.assertLess(out["delta_c"], 0.1)
        self.assertGreater(out["coherent_fraction"], 0.9)

    def test_incoherence_has_low_coherent_fraction(self):
        out = window_metrics(self._run(coupling=0.2, spread=2.0))
        self.assertLess(out["delta_i"], 0.3)
        self.assertLess(out["coherent_fraction"], 0.3)

    def test_fixed_bins_keep_noisy_locked_state_low_distinction(self):
        # the property that makes the toggle honest: a synchronised-but-noisy
        # population still reads ΔC ≈ 0 (one rhythm), not a spurious spread
        out = window_metrics(self._run(coupling=3.0, spread=0.2, noise=0.6))
        self.assertGreater(out["delta_i"], 0.8)
        self.assertLess(out["delta_c"], 0.1)

    def test_delta_c_peaks_between_the_phases(self):
        # ΔC is larger in the critical neighbourhood than deep in either phase
        deep = window_metrics(self._run(coupling=3.0, spread=0.2))["delta_c"]
        crit = window_metrics(self._run(coupling=2.0, spread=1.0))["delta_c"]
        disordered = window_metrics(self._run(coupling=2.0, spread=2.4))["delta_c"]
        self.assertGreater(crit, deep)
        self.assertGreater(crit, disordered)


class TestOneLawAcrossSubstrates(unittest.TestCase):
    """The capacity law, functional, and taxonomy are literally the Hopfield
    module's objects — reuse in code, not a re-implementation."""

    def test_capacity_law_is_the_same_object(self):
        self.assertIs(steady_state_kappa, hopfield_substrate.steady_state_kappa)
        self.assertIs(s_functional, hopfield_substrate.s_functional)
        self.assertIs(trajectory_label, hopfield_substrate.trajectory_label)

    def test_scarcity_taxes_integration_not_distinction(self):
        # the mechanism behind the level crossing, identical to the Hopfield test
        s_abundant = s_functional(0.3, 0.8, steady_state_kappa(0.3, 0.1, 1.0), 2.0)
        s_scarce = s_functional(0.3, 0.8, steady_state_kappa(0.3, 50.0, 1.0), 2.0)
        self.assertLess(s_scarce, s_abundant)
        self.assertGreaterEqual(s_scarce, 0.3)   # the ΔC funding survives

    def test_diverging_is_the_hallucination_label(self):
        self.assertEqual(trajectory_label(0.1, -0.1), "diverging")


if __name__ == "__main__":
    unittest.main()
