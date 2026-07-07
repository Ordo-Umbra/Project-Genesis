"""Checks for the 2-D vs 3-D memory-connectivity verdict logic."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_memory_clusters_3d import _dim_stats, summarise  # noqa: E402


def _pt(t, fertile, p_inf, span, chi):
    return {"temperature": t, "fertile_fraction": fertile, "p_inf": p_inf,
            "spanning_prob": span, "chi": chi, "magnetisation": 0.0}


# a 2-D-like backbone that bends but holds, and a 3-D-like one that breaks
HOLDS = [_pt(0.05, 1.00, 1.00, 1.00, 0.0),
         _pt(0.11, 0.57, 0.45, 0.98, 40.0),
         _pt(0.20, 0.88, 0.88, 1.00, 1.0)]
BREAKS = [_pt(0.05, 1.00, 1.00, 1.00, 0.0),
          _pt(0.13, 0.17, 0.02, 0.15, 15.0),
          _pt(0.20, 0.41, 0.39, 1.00, 2.0)]
TEMPS = [0.05, 0.11, 0.20]
TEMPS3 = [0.05, 0.13, 0.20]


class TestDimStats(unittest.TestCase):

    def test_detects_depercolation(self):
        s = _dim_stats(BREAKS, TEMPS3, 0.3116)
        self.assertTrue(s["depercolates"])           # min span 0.15 < ½
        self.assertAlmostEqual(s["fertile_min"], 0.17)
        self.assertAlmostEqual(s["p_inf_min"], 0.02)

    def test_holding_backbone_not_depercolated(self):
        s = _dim_stats(HOLDS, TEMPS, 0.5927)
        self.assertFalse(s["depercolates"])           # min span 0.98 ≥ ½
        self.assertAlmostEqual(s["chi_peak"], 40.0)


class TestSummariseVerdict(unittest.TestCase):

    def test_more_fragile_in_3d_branch(self):
        text = "\n".join(summarise({2: HOLDS, 3: BREAKS}, TEMPS))
        self.assertIn("MORE fragile in 3-D", text)
        self.assertIn("DE-PERCOLATES", text)
        self.assertIn("load physics beats the favourable geometry", text)

    def test_more_robust_in_3d_branch(self):
        # 3-D backbone stays strong, 2-D dips lower → the predicted outcome
        weak2d = [_pt(0.05, 1.0, 1.0, 1.0, 0.0),
                  _pt(0.11, 0.4, 0.20, 0.9, 20.0),
                  _pt(0.20, 0.8, 0.8, 1.0, 1.0)]
        strong3d = [_pt(0.05, 1.0, 1.0, 1.0, 0.0),
                    _pt(0.11, 0.9, 0.88, 1.0, 1.0),
                    _pt(0.20, 0.95, 0.95, 1.0, 0.5)]
        text = "\n".join(summarise({2: weak2d, 3: strong3d}, TEMPS))
        self.assertIn("more robust", text)


if __name__ == "__main__":
    unittest.main()
