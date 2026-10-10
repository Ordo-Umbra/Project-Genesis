"""Tests for the genericity instruments behind the d+1 junction result.

`excess` catches over-valent points; `components` catches over-valent edges,
which `excess` is blind to in 3-D. Both blindnesses are pinned here so the
reason the second instrument exists cannot silently regress.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from n3_junction_genericity import (  # noqa: E402
    blob_size, checkerboard, components, excess,
)
from n3_junction_null import voronoi_labels  # noqa: E402


class ComponentTests(unittest.TestCase):
    def test_components_wrap_periodically(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = mask[9, 0] = True  # neighbours across the boundary
        self.assertEqual(components(mask).tolist(), [2])

    def test_separate_blobs_counted_separately(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[1, 1] = mask[6, 6] = True
        self.assertEqual(sorted(components(mask).tolist()), [1, 1])


class InstrumentTests(unittest.TestCase):
    def test_2d_checkerboard_is_all_excess(self):
        e, _ = excess(checkerboard(36, 2, 9), 3)
        self.assertEqual(e, 1.0)

    def test_3d_excess_is_blind_to_degenerate_edges(self):
        e, _ = excess(checkerboard(32, 3, 8), 4)
        self.assertLess(e, 0.2)

    def test_3d_components_catch_degenerate_edges(self):
        rng = np.random.default_rng(0)
        cb = blob_size([checkerboard(32, 3, 8)], 4)
        vor = blob_size([voronoi_labels(40, 3, 30, 4, rng)], 4)
        self.assertGreater(cb, 20 * vor)


if __name__ == "__main__":
    unittest.main()
