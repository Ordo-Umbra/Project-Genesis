"""Fast checks for the κ-recovery recall-rescue analysis helper."""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_recall_recovery import crossing_below  # noqa: E402


class TestCrossingBelow(unittest.TestCase):

    def test_interpolates_downward_crossing(self):
        # crosses 0.5 between T=0.2 (0.8) and T=0.3 (0.2):
        # 0.2 + 0.1·(0.8−0.5)/(0.8−0.2) = 0.25
        t = crossing_below([0.1, 0.2, 0.3], [0.9, 0.8, 0.2], 0.5)
        self.assertAlmostEqual(t, 0.25, places=10)

    def test_never_below_returns_inf(self):
        # recall stays above ½ across the whole window → rescued beyond scan
        self.assertTrue(math.isinf(
            crossing_below([0.1, 0.2, 0.3], [0.9, 0.8, 0.7], 0.5)))

    def test_starts_below_returns_first_temperature(self):
        self.assertEqual(
            crossing_below([0.1, 0.2, 0.3], [0.3, 0.2, 0.1], 0.5), 0.1)

    def test_exact_level_at_node_is_the_crossing(self):
        # value is exactly at the level at T=0.2 then drops → crossing at 0.2
        t = crossing_below([0.1, 0.2, 0.3], [0.6, 0.5, 0.0], 0.5)
        self.assertAlmostEqual(t, 0.2, places=10)


if __name__ == "__main__":
    unittest.main()
