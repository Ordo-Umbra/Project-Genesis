"""Checks for the paired-statistics core of the scarcity power test."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "experiments"))

from n3_scarcity_power import paired_stats  # noqa: E402


class TestPairedStats(unittest.TestCase):

    def test_paired_mean_is_the_difference_mean(self):
        a = np.array([0.81, 0.83, 0.80, 0.85, 0.82])
        b = np.array([0.79, 0.82, 0.78, 0.84, 0.80])
        st = paired_stats(a, b, n_boot=1000)
        self.assertAlmostEqual(st["mean"], float((a - b).mean()), places=10)
        self.assertEqual(st["n"], 5)

    def test_pairing_tightens_error_when_correlated(self):
        # correlated runs (shared "difficulty" per seed): paired SE < unpaired
        rng = np.random.default_rng(0)
        shared = rng.normal(0.8, 0.03, 30)          # per-seed common component
        a = shared + 0.01 + rng.normal(0, 0.003, 30)
        b = shared + rng.normal(0, 0.003, 30)
        st = paired_stats(a, b, n_boot=1000)
        self.assertGreater(st["corr"], 0.8)
        self.assertLess(st["se_paired"], st["se_unpaired"])

    def test_uncorrelated_pairing_does_not_help_much(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0.8, 0.03, 40)
        b = rng.normal(0.8, 0.03, 40)
        st = paired_stats(a, b, n_boot=500)
        # with ~zero correlation, paired and unpaired SE are comparable
        self.assertLess(abs(st["se_paired"] - st["se_unpaired"]),
                        0.5 * st["se_unpaired"])

    def test_t_matches_manual_when_scipy_absent_path(self):
        # the reported t is mean/se_paired to within the scipy convention
        a = np.array([0.82, 0.84, 0.81, 0.86, 0.83, 0.85])
        b = np.array([0.80, 0.81, 0.80, 0.85, 0.80, 0.83])
        st = paired_stats(a, b, n_boot=500)
        approx_t = st["mean"] / st["se_paired"]
        self.assertAlmostEqual(st["t"], approx_t, places=4)

    def test_significant_positive_effect_is_detected(self):
        # a clean +0.01 paired effect with tight per-pair noise → small p
        rng = np.random.default_rng(2)
        shared = rng.normal(0.8, 0.03, 24)
        a = shared + 0.01 + rng.normal(0, 0.002, 24)
        b = shared + rng.normal(0, 0.002, 24)
        st = paired_stats(a, b, n_boot=2000)
        self.assertGreater(st["mean"], 0)
        self.assertLess(st["p"], 0.05)
        self.assertGreater(st["ci_lo"], 0.0)   # bootstrap CI excludes zero

    def test_null_effect_is_not_significant(self):
        rng = np.random.default_rng(3)
        shared = rng.normal(0.8, 0.03, 24)
        a = shared + rng.normal(0, 0.002, 24)
        b = shared + rng.normal(0, 0.002, 24)
        st = paired_stats(a, b, n_boot=2000)
        self.assertGreater(st["p"], 0.05)
        self.assertLessEqual(st["ci_lo"], 0.0)
        self.assertGreaterEqual(st["ci_hi"], 0.0)  # CI straddles zero

    def test_bootstrap_ci_brackets_mean(self):
        rng = np.random.default_rng(4)
        a = rng.normal(0.85, 0.02, 30)
        b = rng.normal(0.83, 0.02, 30)
        st = paired_stats(a, b, n_boot=3000)
        self.assertLessEqual(st["ci_lo"], st["mean"])
        self.assertLessEqual(st["mean"], st["ci_hi"])

    def test_effect_size_sign_follows_mean(self):
        # a positive but noisy paired effect: d_z finite and positive
        rng = np.random.default_rng(5)
        a = 0.9 + rng.normal(0, 0.01, 8)
        b = 0.8 + rng.normal(0, 0.01, 8)
        st = paired_stats(a, b, n_boot=200)
        self.assertGreater(st["mean"], 0)
        self.assertTrue(np.isfinite(st["d_z"]))
        self.assertGreater(st["d_z"], 0)


if __name__ == "__main__":
    unittest.main()
