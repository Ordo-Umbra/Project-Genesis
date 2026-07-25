"""Checks for the selection-sweep instrument."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.selection import (  # noqa: E402
    evolve_palette, measure_window, normalise, selection_score)


class TestScore(unittest.TestCase):

    def test_binding_without_motion_scores_zero(self):
        # a frozen configuration is dead however well it binds
        self.assertEqual(selection_score(0.9, 0.0), 0.0)

    def test_motion_without_binding_scores_zero(self):
        # endless churn that binds nothing is a fragmented mess
        self.assertEqual(selection_score(0.0, 0.9), 0.0)

    def test_both_required(self):
        self.assertGreater(selection_score(0.5, 0.5),
                           max(selection_score(0.9, 0.01),
                               selection_score(0.01, 0.9)))

    def test_negatives_clamped(self):
        self.assertEqual(selection_score(-1.0, 0.5), 0.0)


class TestNormalise(unittest.TestCase):

    def test_scales_to_unit_max(self):
        np.testing.assert_allclose(normalise([1.0, 2.0, 4.0]), [0.25, 0.5, 1.0])

    def test_all_zero_is_safe(self):
        np.testing.assert_allclose(normalise([0.0, 0.0]), [0.0, 0.0])


class TestSweepMechanics(unittest.TestCase):
    """The instrument must treat every palette the same way."""

    def _measure(self, p, seed=0, size=48, steps=300, window=20):
        rng = np.random.default_rng(seed)
        f = evolve_palette(p, size, steps, rng, noise=0.25)
        return measure_window(f, p, window, rng, noise=0.25)

    def test_shapes_and_keys(self):
        m = self._measure(3)
        for k in ("distinction", "integration", "churn", "sectors_alive"):
            self.assertIn(k, m)
        self.assertGreaterEqual(m["churn"], 0.0)
        self.assertGreaterEqual(m["integration"], 0.0)

    def test_every_palette_keeps_its_sectors(self):
        # no palette collapses: the possibility space is genuinely open
        for p in (2, 3, 4):
            self.assertEqual(self._measure(p)["sectors_alive"], p)

    def test_two_sectors_cannot_bind_the_palette(self):
        # P=2 has no trivalent junction, so no full-palette binding
        self.assertAlmostEqual(self._measure(2)["integration"], 0.0, places=6)

    def test_three_sectors_bind_more_than_four(self):
        # the monopoly, on the shared instrument
        self.assertGreater(self._measure(3)["integration"],
                           self._measure(4)["integration"])

    def test_distinction_rises_with_palette(self):
        # the competing axis: more kinds means more structure
        self.assertGreater(self._measure(5)["distinction"],
                           self._measure(3)["distinction"])


if __name__ == "__main__":
    unittest.main()
