"""Checks for the competing-memories analysis helper."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_memory_competition import crossover_rate  # noqa: E402


def _c(r, p):
    return {"recovery": r, "overwrite_prob": p}


class TestCrossoverRate(unittest.TestCase):

    def test_interpolates_upward_crossing(self):
        # crosses ½ between r=0.2 (0.4) and r=0.8 (1.0):
        # 0.2 + 0.6·(0.5−0.4)/(1.0−0.4) = 0.3
        cross = [_c(0.02, 0.0), _c(0.2, 0.4), _c(0.8, 1.0)]
        self.assertAlmostEqual(crossover_rate(cross), 0.3, places=10)

    def test_none_when_never_reaches_level(self):
        cross = [_c(0.02, 0.0), _c(0.2, 0.1), _c(0.8, 0.3)]
        self.assertIsNone(crossover_rate(cross))

    def test_none_when_already_above(self):
        # starts above the level and stays there → no upward crossing
        cross = [_c(0.02, 0.7), _c(0.2, 0.9), _c(0.8, 1.0)]
        self.assertIsNone(crossover_rate(cross))

    def test_exact_level_at_node_is_the_crossing(self):
        # value hits exactly ½ at the upper node of the first crossing pair
        cross = [_c(0.1, 0.2), _c(0.2, 0.5), _c(0.4, 0.9)]
        self.assertAlmostEqual(crossover_rate(cross), 0.2, places=10)

    def test_custom_level(self):
        cross = [_c(0.1, 0.0), _c(0.3, 0.5), _c(0.9, 1.0)]
        # level 0.75 crosses between r=0.3 (0.5) and r=0.9 (1.0):
        # 0.3 + 0.6·(0.75−0.5)/(1.0−0.5) = 0.6
        self.assertAlmostEqual(crossover_rate(cross, 0.75), 0.6, places=10)

    def test_first_crossing_wins(self):
        # non-monotone: dips back below then up again; take the first crossing
        cross = [_c(0.1, 0.2), _c(0.2, 0.6), _c(0.3, 0.4), _c(0.4, 0.9)]
        self.assertAlmostEqual(crossover_rate(cross), 0.175, places=10)


if __name__ == "__main__":
    unittest.main()
