"""Regime checks for the memory-percolation verdict helper."""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_memory_clusters import percolation_verdict  # noqa: E402


def _pt(t, fertile, p_inf, span, chi, m):
    return {"temperature": t, "fertile_fraction": fertile, "p_inf": p_inf,
            "spanning_prob": span, "chi": chi, "magnetisation": m}


class TestPercolationVerdict(unittest.TestCase):

    def test_stays_connected_but_dips_below_pc(self):
        # spans everywhere, but the fertile fraction crosses p_c and P∞ dips
        pts = [_pt(0.05, 0.99, 0.99, 1.0, 0.0, 0.95),
               _pt(0.11, 0.55, 0.41, 0.97, 43.0, 0.14),
               _pt(0.20, 0.89, 0.89, 1.0, 0.9, 0.02)]
        text = "\n".join(percolation_verdict(pts, 0.089, float("inf"), 0.104))
        self.assertIn("STAYS CONNECTED", text)
        self.assertIn("BELOW the random", text)
        # bottleneck located by P∞ minimum, not spanning prob
        self.assertIn("thinnest at T=0.11", text)

    def test_stays_connected_above_pc_says_not_in_doubt(self):
        # never crosses p_c → the "connectivity never in doubt" branch
        pts = [_pt(0.05, 0.99, 0.99, 1.0, 0.0, 0.95),
               _pt(0.11, 0.72, 0.70, 1.0, 3.0, 0.14),
               _pt(0.20, 0.90, 0.90, 1.0, 0.5, 0.02)]
        text = "\n".join(percolation_verdict(pts, 0.089, float("inf"),
                                             float("inf")))
        self.assertIn("STAYS CONNECTED", text)
        self.assertIn("never seriously in doubt", text)

    def test_de_percolation_branch(self):
        # spanning probability drops below ½ → genuine de-percolation
        pts = [_pt(0.05, 0.99, 0.99, 1.0, 0.0, 0.95),
               _pt(0.11, 0.45, 0.10, 0.2, 30.0, 0.14),
               _pt(0.20, 0.30, 0.05, 0.1, 5.0, 0.02)]
        text = "\n".join(percolation_verdict(pts, 0.089, 0.10, 0.098))
        self.assertIn("DE-PERCOLATES", text)
        # perc edge (0.10) is above the order edge (0.089)
        self.assertIn("above the order edge", text)

    def test_bottleneck_uses_p_inf_not_spanning(self):
        # two points tie on spanning prob; the P∞ minimum must win
        pts = [_pt(0.10, 0.62, 0.56, 0.97, 16.0, 0.35),
               _pt(0.11, 0.55, 0.41, 0.97, 43.0, 0.14),
               _pt(0.20, 0.89, 0.89, 1.0, 0.9, 0.02)]
        text = "\n".join(percolation_verdict(pts, 0.089, float("inf"), 0.104))
        self.assertIn("thinnest at T=0.11", text)
        self.assertNotIn("thinnest at T=0.1 ", text)


if __name__ == "__main__":
    unittest.main()
