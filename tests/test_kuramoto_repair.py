"""Tests for the drive-free load reading on the oscillator substrate.

Two things have to hold or the correction is not a correction.

The published ``activity`` must be **untouched** — three experiments are stated
in it, and a refactor that silently perturbed it would invalidate them without
anyone noticing. So the first block pins it bit-for-bit against values recorded
before ``drift_field`` was split out.

And the new ``repair_rate`` has to actually have the properties it was
introduced for: no injected noise in it, no dependence on the integrator's step,
and still zero in the deterministic locked state — that last one is the toggle
the whole transplant design rests on, so a reading that quietly charged a
free-running rhythm would break the experiment it was meant to fix.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.kuramoto_substrate import (  # noqa: E402
    drift_field,
    kuramoto_step,
    natural_frequencies,
    order_parameter,
    simulate,
    window_metrics,
)

K, SIGMA = 2.0, 0.6


def run(gamma, dt, noise, *, n=300, duration=45.0, seed=7):
    steps = int(duration / dt)
    rng = np.random.default_rng(seed)
    omega = natural_frequencies(n, gamma, rng, distribution="gaussian")
    phases = rng.uniform(-np.pi, np.pi, n)
    sim = simulate(phases, omega, K, dt, steps, noise, rng, burn_in=steps // 2)
    return window_metrics(sim, freq_tol=0.3)


class TestPublishedActivityIsUnchanged(unittest.TestCase):
    """Recorded before `drift_field` was factored out of `kuramoto_step`."""

    def test_activity_is_bit_identical_to_the_pre_refactor_values(self):
        for dt, expected in ((0.10, 0.19792), (0.05, 0.13659), (0.025, 0.09562)):
            m = run(0.3, dt, SIGMA, n=300, duration=45.0, seed=7)
            self.assertAlmostEqual(m["activity"], expected, places=5,
                                   msg=f"dt={dt}")

    def test_drift_field_reproduces_the_step_it_was_extracted_from(self):
        rng = np.random.default_rng(3)
        omega = natural_frequencies(200, 0.5, rng, distribution="gaussian")
        phases = rng.uniform(-np.pi, np.pi, 200)
        r, psi = order_parameter(phases)
        self.assertTrue(np.allclose(drift_field(phases, omega, K),
                                    omega + K * r * np.sin(psi - phases)))

    def test_a_noiseless_step_is_exactly_the_drift_times_dt(self):
        rng = np.random.default_rng(4)
        omega = natural_frequencies(150, 0.4, rng, distribution="gaussian")
        phases = rng.uniform(-np.pi, np.pi, 150)
        stepped = kuramoto_step(phases, omega, K, 0.05, 0.0, rng)
        self.assertTrue(np.allclose(stepped - phases,
                                    drift_field(phases, omega, K) * 0.05))


class TestRepairRateIsDriveFree(unittest.TestCase):

    def test_the_published_reading_is_essentially_the_injected_noise(self):
        """The defect being corrected, pinned so it cannot be argued away."""
        ratios = [run(0.1, dt, SIGMA)["activity"] / (SIGMA * np.sqrt(dt))
                  for dt in (0.1, 0.025, 0.0125)]
        self.assertTrue(all(abs(r - 1.0) < 0.11 for r in ratios), msg=str(ratios))
        self.assertLess(abs(ratios[-1] - 1.0), 0.02)   # converges to the floor

    def test_repair_rate_is_not_the_injected_noise(self):
        m = run(0.1, 0.05, SIGMA)
        self.assertGreater(abs(m["repair_rate"] / (SIGMA * np.sqrt(0.05)) - 1.0),
                           0.5)

    def test_repair_rate_scales_with_the_coupling_not_the_step(self):
        """It is restoring work: stiffer coupling, more of it."""
        def rep(coupling):
            steps = int(45.0 / 0.05)
            rng = np.random.default_rng(7)
            omega = natural_frequencies(300, 0.1, rng, distribution="gaussian")
            phases = rng.uniform(-np.pi, np.pi, 300)
            sim = simulate(phases, omega, coupling, 0.05, steps, SIGMA, rng,
                           burn_in=steps // 2)
            return window_metrics(sim)["repair_rate"]
        vals = [rep(c) for c in (1.0, 2.0, 4.0)]
        self.assertTrue(all(a < b for a, b in zip(vals, vals[1:])), msg=str(vals))


class TestRepairRateIsUnitFree(unittest.TestCase):

    def test_repair_rate_barely_moves_across_a_16x_change_in_dt(self):
        vals = [run(0.1, dt, SIGMA)["repair_rate"]
                for dt in (0.2, 0.1, 0.05, 0.025, 0.0125)]
        spread = (max(vals) - min(vals)) / np.mean(vals)
        self.assertLess(spread, 0.12, msg=f"{vals} -> {spread:.3f}")

    def test_the_published_reading_moves_a_lot_across_the_same_sweep(self):
        vals = [run(0.1, dt, SIGMA)["activity"]
                for dt in (0.2, 0.1, 0.05, 0.025, 0.0125)]
        self.assertGreater(max(vals) / min(vals), 3.0, msg=str(vals))

    def test_activity_tracks_sqrt_dt_while_repair_rate_does_not(self):
        dts = np.array([0.2, 0.05, 0.0125])
        act = np.array([run(0.1, d, SIGMA)["activity"] for d in dts])
        rep = np.array([run(0.1, d, SIGMA)["repair_rate"] for d in dts])
        # activity / sqrt(dt) is flat; repair_rate / sqrt(dt) is not
        self.assertLess(np.std(act / np.sqrt(dts)) / np.mean(act / np.sqrt(dts)),
                        0.06)
        self.assertGreater(np.std(rep / np.sqrt(dts)) / np.mean(rep / np.sqrt(dts)),
                           0.3)


class TestTheToggleSurvives(unittest.TestCase):
    """A rhythm nobody perturbs must still be free, or the transplant breaks."""

    def test_a_deterministic_locked_state_pays_nothing(self):
        m = run(0.1, 0.05, 0.0)
        self.assertGreater(m["delta_i"], 0.9)          # it really is locked
        self.assertLess(m["repair_rate"], 1e-6)

    def test_it_pays_nothing_at_every_step_size(self):
        for dt in (0.2, 0.05, 0.0125):
            self.assertLess(run(0.1, dt, 0.0)["repair_rate"], 1e-6,
                            msg=f"dt={dt}")

    def test_the_same_state_pays_once_it_is_driven(self):
        free = run(0.1, 0.05, 0.0)["repair_rate"]
        held = run(0.1, 0.05, SIGMA)["repair_rate"]
        self.assertGreater(held, 0.1)
        self.assertLess(free / held, 0.01)

    def test_an_unlocked_population_pays_even_undriven(self):
        """Not everything undriven is free — drifters run at their own rates."""
        m = run(2.0, 0.05, 0.0)
        self.assertLess(m["delta_i"], 0.4)
        self.assertGreater(m["repair_rate"], 0.1)

    def test_window_metrics_passes_both_readings_through(self):
        m = run(0.5, 0.05, SIGMA)
        self.assertIn("activity", m)
        self.assertIn("repair_rate", m)
        self.assertNotAlmostEqual(m["activity"], m["repair_rate"], places=3)


class TestLoadProfile(unittest.TestCase):

    def test_the_corrected_load_separates_order_from_disorder_more_sharply(self):
        """rho is the quantity the mechanism needs to be far from 1."""
        lo_a, lo_r = (run(0.1, 0.05, SIGMA)[k] for k in ("activity", "repair_rate"))
        hi_a, hi_r = (run(1.06, 0.05, SIGMA)[k] for k in ("activity", "repair_rate"))
        self.assertGreater(hi_r / lo_r, hi_a / lo_a)

    def test_the_corrected_load_still_rises_with_disorder(self):
        vals = [run(g, 0.05, SIGMA)["repair_rate"] for g in (0.1, 1.06, 2.4)]
        self.assertTrue(all(a < b for a, b in zip(vals, vals[1:])), msg=str(vals))


if __name__ == "__main__":
    unittest.main()
