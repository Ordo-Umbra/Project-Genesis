"""Fast checks for experiments/n3_scaling_ladder.py fit helpers."""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_scaling_ladder import (  # noqa: E402
    parabolic_peak,
    weighted_loglog_slope,
)


class TestParabolicPeak(unittest.TestCase):

    def test_recovers_smooth_peak(self):
        """A log-parabola sampled on a grid is recovered exactly."""
        x_true, y_true = 0.15, 3.0
        xs = [0.08, 0.11, 0.15, 0.21, 0.3]
        ys = [y_true - 2.0 * (math.log(x) - math.log(x_true)) ** 2
              for x in xs]
        pk = parabolic_peak(xs, ys, [0.1] * len(xs))
        self.assertFalse(pk["at_edge"])
        self.assertAlmostEqual(pk["x_peak"], x_true, delta=0.01)
        self.assertAlmostEqual(pk["y_peak"], y_true, delta=0.01)

    def test_flags_edge_maximum(self):
        pk = parabolic_peak([0.1, 0.2, 0.3], [3.0, 2.0, 1.0], [0.1] * 3)
        self.assertTrue(pk["at_edge"])
        self.assertEqual(pk["x_peak"], 0.1)

    def test_tied_maximum_stays_in_bracket(self):
        """A tie between the middle and last point yields an interior peak
        between them (the parabola vertex), not an extrapolation."""
        pk = parabolic_peak([0.1, 0.2, 0.4], [0.5, 1.0, 1.0], [0.1] * 3)
        self.assertFalse(pk["at_edge"])
        self.assertGreaterEqual(pk["x_peak"], 0.2)
        self.assertLessEqual(pk["x_peak"], 0.4)
        self.assertGreaterEqual(pk["y_peak"], 1.0)


class TestSlopeFit(unittest.TestCase):

    def test_recovers_power_law(self):
        ls = [24, 32, 48, 64]
        b_true = 1.75
        ys = [0.01 * (L / 24) ** b_true for L in ls]
        slope, err = weighted_loglog_slope(ls, ys, [0.1 * y for y in ys])
        self.assertAlmostEqual(slope, b_true, delta=1e-6)
        self.assertGreater(err, 0.0)

    def test_flat_data_gives_zero_slope(self):
        ls = [24, 32, 48, 64]
        ys = [0.02] * 4
        slope, err = weighted_loglog_slope(ls, ys, [0.002] * 4)
        self.assertAlmostEqual(slope, 0.0, places=10)
        # with 10% errors, the slope error should easily separate 0 from 1.75
        self.assertLess(err, 0.5)

    def test_error_shrinks_with_precision(self):
        ls = [24, 32, 48, 64]
        ys = [0.02] * 4
        _, err_loose = weighted_loglog_slope(ls, ys, [0.004] * 4)
        _, err_tight = weighted_loglog_slope(ls, ys, [0.001] * 4)
        self.assertLess(err_tight, err_loose)


if __name__ == "__main__":
    unittest.main()
