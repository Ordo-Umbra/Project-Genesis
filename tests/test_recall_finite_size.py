"""Fast checks for the finite-size recall-vs-order analysis helpers."""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_recall_finite_size import (  # noqa: E402
    edge_gap,
    interp_at,
    weighted_slope,
)


class TestInterpAt(unittest.TestCase):

    def test_midpoint_interpolation(self):
        # halfway between T=0.1 (0.8) and T=0.2 (0.4) → 0.6
        self.assertAlmostEqual(
            interp_at([0.1, 0.2, 0.3], [0.8, 0.4, 0.2], 0.15), 0.6, places=10)

    def test_exact_node_returns_node_value(self):
        self.assertAlmostEqual(
            interp_at([0.1, 0.2, 0.3], [0.8, 0.4, 0.2], 0.2), 0.4, places=10)

    def test_clamps_below_range(self):
        self.assertEqual(interp_at([0.1, 0.2], [0.8, 0.4], 0.0), 0.8)

    def test_clamps_above_range(self):
        self.assertEqual(interp_at([0.1, 0.2], [0.8, 0.4], 0.9), 0.4)

    def test_nonfinite_target_is_nan(self):
        self.assertTrue(math.isnan(
            interp_at([0.1, 0.2], [0.8, 0.4], float("inf"))))


class TestEdgeGap(unittest.TestCase):

    def test_positive_when_recall_edge_higher(self):
        self.assertAlmostEqual(edge_gap(0.12, 0.09), 0.03, places=10)

    def test_negative_when_recall_edge_lower(self):
        self.assertAlmostEqual(edge_gap(0.07, 0.10), -0.03, places=10)

    def test_recall_inf_order_finite_is_plus_inf(self):
        # recall never falls below ½ → overtaking beyond the window
        self.assertTrue(math.isinf(edge_gap(float("inf"), 0.10)))
        self.assertGreater(edge_gap(float("inf"), 0.10), 0)

    def test_both_inf_is_nan(self):
        # neither edge crossed → comparison undefined on this grid
        self.assertTrue(math.isnan(edge_gap(float("inf"), float("inf"))))


class TestWeightedSlope(unittest.TestCase):

    def test_exact_line_recovered(self):
        # y = 0.5 - 2 x  on three points, equal weights
        xs = [0.0, 0.02, 0.05]
        ys = [0.5, 0.46, 0.40]
        errs = [0.01, 0.01, 0.01]
        fit = weighted_slope(xs, ys, errs)
        self.assertAlmostEqual(fit["slope"], -2.0, places=6)
        self.assertAlmostEqual(fit["intercept"], 0.5, places=6)
        self.assertEqual(fit["n"], 3)

    def test_intercept_is_thermodynamic_limit(self):
        # margin measured at 1/L; intercept is the L→∞ value
        inv = [1 / 16, 1 / 24, 1 / 32]
        margin = [0.30, 0.34, 0.36]  # widening toward smaller 1/L
        fit = weighted_slope(inv, margin, [0.02, 0.02, 0.02])
        # extrapolated margin at 1/L = 0 should exceed the largest measured
        self.assertGreater(fit["intercept"], max(margin))

    def test_infinite_points_are_dropped(self):
        fit = weighted_slope([0.0, 0.05, 0.1], [float("inf"), 0.4, 0.3],
                             [0.01, 0.01, 0.01])
        self.assertEqual(fit["n"], 2)
        self.assertTrue(math.isfinite(fit["slope"]))

    def test_too_few_finite_points_is_nan(self):
        fit = weighted_slope([0.0, 0.05], [float("inf"), 0.4],
                             [0.01, 0.01])
        self.assertTrue(math.isnan(fit["slope"]))
        self.assertEqual(fit["n"], 1)


if __name__ == "__main__":
    unittest.main()
