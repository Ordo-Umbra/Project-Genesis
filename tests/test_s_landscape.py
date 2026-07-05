"""Fast checks for the S-landscape phase-diagram analysis helpers."""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_s_landscape import (  # noqa: E402
    analyse_curve,
    split_index,
    zero_crossing,
)


def _pt(t, m, dc, coh, kap):
    return {"temperature": t, "magnetisation": m, "delta_c": dc,
            "coherence": coh, "kappa_mean": kap}


class TestSplitIndex(unittest.TestCase):

    def test_drop_located(self):
        self.assertEqual(split_index([0.9, 0.8, 0.3, 0.1], 0.5), 2)

    def test_never_disorders_all_ordered(self):
        self.assertEqual(split_index([0.9, 0.8, 0.7], 0.5), 2)

    def test_immediately_disordered_clamped_to_one(self):
        self.assertEqual(split_index([0.2, 0.1], 0.5), 1)


class TestAnalyseCurve(unittest.TestCase):

    def test_ordered_optimum_when_coherence_dominates(self):
        # low-T ordered point has high I; with big w it wins
        pts = [_pt(0.03, 0.95, 0.001, 0.9, 1.0),   # ordered
               _pt(0.09, 0.30, 0.03, 0.4, 0.5),    # critical (ΔC peak)
               _pt(0.30, 0.02, 0.02, 0.33, 0.6)]   # disordered
        a = analyse_curve(pts, w=0.5, m_threshold=0.5)
        self.assertEqual(a["regime"], "ordered")
        self.assertEqual(a["t_ordered"], 0.03)
        self.assertGreater(a["delta_s"], 0.0)

    def test_critical_optimum_when_distinction_dominates(self):
        pts = [_pt(0.03, 0.95, 0.001, 0.9, 1.0),
               _pt(0.09, 0.30, 0.03, 0.4, 0.5),
               _pt(0.30, 0.02, 0.02, 0.33, 0.6)]
        a = analyse_curve(pts, w=0.001, m_threshold=0.5)  # ΔC dominates
        self.assertEqual(a["regime"], "critical")
        self.assertEqual(a["t_critical"], 0.09)
        self.assertLess(a["delta_s"], 0.0)

    def test_branch_temperatures_are_from_correct_sides(self):
        pts = [_pt(0.03, 0.95, 0.001, 0.9, 1.0),
               _pt(0.09, 0.30, 0.03, 0.4, 0.5),
               _pt(0.30, 0.02, 0.02, 0.33, 0.6)]
        a = analyse_curve(pts, w=0.05, m_threshold=0.5)
        self.assertLess(a["t_ordered"], 0.09)     # ordered branch below split
        self.assertGreaterEqual(a["t_critical"], 0.09)


class TestZeroCrossing(unittest.TestCase):

    def test_interpolates_downward_crossing(self):
        # crosses zero between x=2 and x=4: 2 + 2*(1)/(1-(-1)) = 3
        self.assertAlmostEqual(zero_crossing([1, 2, 4, 8], [2, 1, -1, -3]),
                               3.0, places=10)

    def test_no_crossing_gives_nan(self):
        self.assertTrue(math.isnan(zero_crossing([1, 2, 3], [1, 2, 3])))

    def test_only_downward_crossing_counts(self):
        # upward crossing (-1 -> +1) is ignored; downward later is caught
        self.assertFalse(math.isnan(
            zero_crossing([1, 2, 3, 4], [-1, 1, 2, -1])))


if __name__ == "__main__":
    unittest.main()
