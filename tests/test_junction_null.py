"""Tests for the geometry-vs-dynamics null of the d+1 junction result.

Pins the two facts the null leans on: a random Voronoi tessellation with no
dynamics already selects `P = d + 1` in 2-D, and a non-generic four-fold block
tiling scores `P = 4` at a vertex until something splits it.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from n3_junction_null import (  # noqa: E402
    any_junction_density, block_tiling_2d, voronoi_labels,
)
from project_genesis.junction_scale import (  # noqa: E402
    full_palette_density, selection_profile,
)


class VoronoiNullTests(unittest.TestCase):
    def test_every_colour_present(self):
        lab = voronoi_labels(40, 2, 20, 5, np.random.default_rng(0))
        self.assertEqual(set(np.unique(lab)), set(range(5)))

    def test_null_selects_three_in_the_plane(self):
        rng = np.random.default_rng(1)
        dens = {P: full_palette_density(voronoi_labels(120, 2, 40, P, rng), P,
                                        min_valence=3)
                for P in (2, 3, 4, 5)}
        self.assertEqual(selection_profile(dens)["argmax"], 3)

    def test_point_junctions_exist_above_d_plus_one(self):
        rng = np.random.default_rng(2)
        lab = voronoi_labels(120, 2, 40, 6, rng)
        self.assertGreater(any_junction_density(lab, 3), 0.0)


class DegenerateTilingTests(unittest.TestCase):
    def test_block_tiling_has_four_fold_vertices(self):
        lab = block_tiling_2d(36, 9)
        self.assertGreater(full_palette_density(lab, 4, min_valence=4), 0.0)


if __name__ == "__main__":
    unittest.main()
