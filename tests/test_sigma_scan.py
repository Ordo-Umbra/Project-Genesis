"""Fast checks for experiments/confinement_sigma_scan.py.

These exercise the experiment's machinery (jackknife, exact reference
curve, single-point driver) at CI scale; the production scan itself is a
long-running experiment, not a test.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from confinement_sigma_scan import (  # noqa: E402
    jackknife,
    run_point,
    sigma_exact_su2_2d,
)


class TestJackknife(unittest.TestCase):

    def test_linear_estimator_matches_standard_error(self):
        """For the identity estimator, jackknife error = std/sqrt(n)."""
        rng = np.random.default_rng(0)
        x = rng.normal(3.0, 2.0, size=(400, 1))
        mean, err = jackknife(x, lambda m: float(m[0]))
        self.assertAlmostEqual(mean, float(x.mean()), places=12)
        expected = float(x.std(ddof=1) / math.sqrt(len(x)))
        self.assertAlmostEqual(err, expected, delta=0.02 * expected)

    def test_single_sample_gives_nan_error(self):
        x = np.ones((1, 1))
        mean, err = jackknife(x, lambda m: float(m[0]))
        self.assertEqual(mean, 1.0)
        self.assertTrue(math.isnan(err))

    def test_nonlinear_estimator_bias_handled(self):
        """Jackknife of log(mean) stays close to log of the true mean."""
        rng = np.random.default_rng(1)
        x = rng.normal(10.0, 1.0, size=(500, 1))
        mean, err = jackknife(x, lambda m: math.log(m[0]))
        self.assertAlmostEqual(mean, math.log(10.0), delta=3 * err)


class TestExactReference(unittest.TestCase):

    def test_sigma_exact_positive_and_decreasing(self):
        vals = [sigma_exact_su2_2d(b) for b in (0.5, 1.0, 2.0, 4.0)]
        for v in vals:
            self.assertGreater(v, 0.0)
        for lo, hi in zip(vals, vals[1:]):
            self.assertGreater(lo, hi)

    def test_sigma_exact_known_value(self):
        """σ(1.0) = −ln(I₂(2)/I₁(2)) ≈ 0.8367."""
        self.assertAlmostEqual(sigma_exact_su2_2d(1.0), 0.8367, delta=5e-4)


class TestRunPoint(unittest.TestCase):

    def test_micro_point_structure_and_calibration(self):
        """A tiny SU(2) point must return well-formed results near exact σ."""
        res = run_point(
            n=2, ndim=2, size=8, beta_g=2.0, r_max=2,
            n_therm=30, n_meas=40, n_skip=2, n_or=1, seed=99,
        )
        self.assertEqual(res["beta_g"], 2.0)
        self.assertIn("W_1_1", res["loops"])
        self.assertIn("chi_2_2", res["creutz"])
        chi = res["creutz"]["chi_2_2"]
        self.assertFalse(math.isnan(chi["value"]))
        self.assertGreater(chi["value"], 0.0)
        # calibration against the exact 2-D curve, generous 4σ + floor
        exact = sigma_exact_su2_2d(2.0)
        tol = max(4 * chi["error"], 0.1)
        self.assertLess(abs(chi["value"] - exact), tol)
        # Polyakov magnitude is bounded
        self.assertLessEqual(res["polyakov_abs"], 1.0)


if __name__ == "__main__":
    unittest.main()
