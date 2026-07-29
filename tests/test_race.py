"""Tests for the imposed-lead intervention and the growth-law discriminator.

Two things must be right or the experiment says nothing. The intervention has
to actually set the lead while leaving the field a legitimate configuration
(otherwise the dose-response is measuring an artefact of the seeding), and the
growth-law fit has to be able to tell a compounding law from a linear one
(otherwise "it is a race" could never have been refuted, which is what in fact
happened).
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.race import (  # noqa: E402
    growth_rate,
    run_race,
    sector_fractions,
    seed_with_lead,
    top_gap,
)


class TestRace(unittest.TestCase):

    # ------------------------------------------------ the intervention is fair

    def test_zero_lead_reproduces_an_unbiased_seed(self):
        """With no lead the seed must be the plain fair draw."""
        a = seed_with_lead(4, 3, 6, np.random.default_rng(0), lead=0.0)
        rng = np.random.default_rng(0)
        raw = rng.uniform(-0.5, 0.5, (4, 3, 6, 6, 6))
        want = raw / (np.sqrt((raw ** 2).sum(axis=1, keepdims=True)) + 1e-12)
        self.assertTrue(np.allclose(a, want))

    def test_seed_stays_on_the_unit_sphere_at_every_lead(self):
        """The tilt must not produce an illegal configuration."""
        for lead in (0.0, 0.05, 0.5, 2.0):
            f = seed_with_lead(3, 4, 6, np.random.default_rng(1), lead=lead)
            norms = np.sqrt((f ** 2).sum(axis=1))
            self.assertTrue(np.allclose(norms, 1.0, atol=1e-6), msg=f"lead {lead}")

    def test_lead_biases_the_intended_sector_and_only_that_one(self):
        f = seed_with_lead(6, 4, 8, np.random.default_rng(2), lead=0.6, sector=2)
        frac = sector_fractions(np.argmax(f, axis=1), 4)
        self.assertGreater(frac[:, 2].mean(), frac[:, 0].mean())
        self.assertGreater(frac[:, 2].mean(), frac[:, 1].mean())
        self.assertGreater(frac[:, 2].mean(), frac[:, 3].mean())

    def test_larger_lead_gives_a_larger_realised_share(self):
        shares = []
        for lead in (0.0, 0.1, 0.3, 0.8):
            f = seed_with_lead(8, 4, 8, np.random.default_rng(3), lead=lead)
            shares.append(float(sector_fractions(np.argmax(f, axis=1), 4).max(axis=1).mean()))
        self.assertEqual(shares, sorted(shares))

    def test_a_huge_lead_takes_essentially_everything(self):
        f = seed_with_lead(4, 4, 8, np.random.default_rng(4), lead=5.0)
        frac = sector_fractions(np.argmax(f, axis=1), 4)
        self.assertGreater(frac[:, 0].mean(), 0.95)

    # -------------------------------------------------------------- the gap

    def test_top_gap_is_leader_minus_runner_up(self):
        frac = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        self.assertTrue(np.allclose(top_gap(frac), [0.2, 0.0]))

    def test_top_gap_is_zero_for_a_perfect_tie(self):
        frac = np.full((3, 4), 0.25)
        self.assertTrue(np.allclose(top_gap(frac), 0.0))

    def test_sector_fractions_sum_to_one(self):
        rng = np.random.default_rng(5)
        labels = rng.integers(0, 4, (5, 6, 6, 6))
        self.assertTrue(np.allclose(sector_fractions(labels, 4).sum(axis=1), 1.0))

    # ------------------------------------------- the growth law discriminator

    def test_exponential_data_is_identified_as_compounding(self):
        t = np.arange(20, 201, 10, dtype=float)
        gaps = np.exp(0.01 * t)[:, None] * np.ones((1, 5))
        lam, r2_log, r2_lin, r2_sqrt = growth_rate(t, gaps)
        self.assertAlmostEqual(lam, 0.01, delta=1e-6)
        self.assertGreater(r2_log, r2_lin)
        self.assertGreater(r2_log, r2_sqrt)

    def test_linear_data_is_identified_as_drift(self):
        """The refutation has to be reachable, or Q2 was unfalsifiable."""
        t = np.arange(20, 201, 10, dtype=float)
        gaps = (0.001 * t)[:, None] * np.ones((1, 5))
        _, r2_log, r2_lin, r2_sqrt = growth_rate(t, gaps)
        self.assertGreater(r2_lin, r2_log)
        self.assertGreater(r2_lin, r2_sqrt)

    def test_sqrt_data_is_identified_as_diffusion(self):
        t = np.arange(20, 201, 10, dtype=float)
        gaps = (0.01 * np.sqrt(t))[:, None] * np.ones((1, 5))
        _, r2_log, r2_lin, r2_sqrt = growth_rate(t, gaps)
        self.assertGreater(r2_sqrt, r2_log)
        self.assertGreater(r2_sqrt, r2_lin)

    def test_growth_rate_needs_enough_points(self):
        lam, *_ = growth_rate(np.array([10.0, 20.0]), np.ones((2, 3)), lo=0, hi=5)
        self.assertTrue(np.isnan(lam))

    def test_growth_rate_survives_a_zero_gap(self):
        t = np.arange(20, 121, 10, dtype=float)
        gaps = np.zeros((len(t), 4))
        lam, r2_log, *_ = growth_rate(t, gaps)
        self.assertTrue(np.isfinite(lam))

    # -------------------------------------------------------------- ensemble

    def test_run_race_reports_every_replica(self):
        r = run_race(6, 3, 8, 60, lead=0.0, seed=0)
        self.assertEqual(len(r["death"]), 6)
        self.assertTrue(np.array_equal(r["censored"], r["death"] < 0))
        self.assertAlmostEqual(r["survival"], float((r["death"] < 0).mean()), places=12)

    def test_run_race_measures_the_realised_lead(self):
        low = run_race(8, 4, 8, 40, lead=0.0, seed=1)
        high = run_race(8, 4, 8, 40, lead=0.5, seed=1)
        self.assertGreater(high["lead_measured"], low["lead_measured"])

    def test_a_large_imposed_lead_kills_the_ensemble(self):
        """The dose-response must have a top end, or the cliff is unmeasurable."""
        r = run_race(8, 4, 10, 200, lead=1.0, seed=2)
        self.assertEqual(r["survival"], 0.0)

    def test_gap_trajectory_is_one_row_per_sample(self):
        r = run_race(5, 3, 8, 60, lead=0.0, seed=3, sample=10)
        self.assertEqual(r["gaps"].shape[1], 5)
        self.assertEqual(len(r["times"]), r["gaps"].shape[0])
        self.assertTrue(np.isfinite(r["gaps"]).all())


if __name__ == "__main__":
    unittest.main()
