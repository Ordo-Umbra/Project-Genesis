"""Checks for the dimensional (0D/1D/2D) CW-census of the sector tessellation."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.dimensional_forms import (  # noqa: E402
    dimensional_census,
    local_dimension,
    scalar_sector_labels,
)


class LocalDimensionTests(unittest.TestCase):
    def test_uniform_field_is_all_interior(self):
        lab = np.zeros((16, 16), dtype=int)
        self.assertTrue(np.all(local_dimension(lab) == 2))

    def test_classes_present_at_a_junction(self):
        # three sectors meeting → interior, wall, and junction all appear
        lab = np.zeros((20, 20), dtype=int)
        lab[:, 10:] = 1
        lab[10:, 10:] = 2
        dim = local_dimension(lab)
        self.assertEqual(sorted(np.unique(dim).tolist()), [0, 1, 2])


class EulerBackboneTests(unittest.TestCase):
    def test_stripes_euler_zero(self):
        # two straight walls on the torus, no junctions
        lab = np.zeros((40, 40), dtype=int)
        lab[20:, :] = 1
        c = dimensional_census(lab)
        self.assertEqual((c["V"], c["E"], c["F"]), (0, 2, 2))
        self.assertEqual(c["euler"], 0)

    def test_four_block_is_degree_four_and_euler_zero(self):
        lab = np.zeros((40, 40), dtype=int)
        lab[:20, :20], lab[:20, 20:] = 0, 1
        lab[20:, :20], lab[20:, 20:] = 2, 3
        c = dimensional_census(lab)
        self.assertEqual(c["V"], 4)
        self.assertEqual(c["F"], 4)
        self.assertEqual(c["euler"], 0)          # 4 - 8 + 4
        self.assertAlmostEqual(c["mean_valence"], 4.0, places=6)

    def test_trivalent_brick_approaches_two_three_one(self):
        n = 60
        lab = np.zeros((n, n), dtype=int)
        for r in range(n):
            row = r // 20
            off = 10 if row % 2 else 0
            for col in range(n):
                lab[r, col] = row * 3 + ((col + off) % n) // 20
        _, lab = np.unique(lab, return_inverse=True)
        lab = lab.reshape(n, n)
        c = dimensional_census(lab)
        self.assertEqual(c["euler"], 0)
        self.assertLess(abs(c["mean_valence"] - 3.0), 0.5)
        # V:E:F within a factor near the ideal 2:3:1
        self.assertLess(abs(c["ratio_to_F"][0] - 2.0), 0.6)
        self.assertLess(abs(c["ratio_to_F"][1] - 3.0), 0.6)


class ThreeDimensionalTests(unittest.TestCase):
    def test_slabs_euler_zero(self):
        # two slab domains, two planar walls, no edges/vertices on T^3
        lab = np.zeros((24, 24, 24), dtype=int)
        lab[12:, :, :] = 1
        c = dimensional_census(lab)
        self.assertEqual((c["V"], c["E"], c["F"], c["C"]), (0, 0, 2, 2))
        self.assertEqual(c["euler"], 0)          # 0 - 0 + 2 - 2

    def test_eight_block_is_a_valid_cw_complex(self):
        # 2x2x2 cubes: a clean CW-decomposition of the 3-torus
        lab = np.zeros((24, 24, 24), dtype=int)
        idx = 0
        for a in (slice(0, 12), slice(12, 24)):
            for b in (slice(0, 12), slice(12, 24)):
                for c_ in (slice(0, 12), slice(12, 24)):
                    lab[a, b, c_] = idx
                    idx += 1
        c = dimensional_census(lab)
        self.assertEqual(c["euler"], 0)          # V-E+F-C = 8-24+24-8
        self.assertEqual(len(c["cells"]), 4)     # four cell dimensions

    def test_four_families_need_the_fourth_sector(self):
        # a 0-cell (vertex) needs 4 distinct sectors to meet in 3-D
        from project_genesis.multiphase import sector_labels, step_multiphase
        import numpy as np2
        def families(palette):
            r = np2.random.default_rng(0)
            f = 0.1 * r.standard_normal((palette, 40, 40, 40))
            for _ in range(120):
                f = step_multiphase(f, diffusion=1.0, gamma=1.5, dt=0.1)
            c = dimensional_census(sector_labels(f))
            return sum(1 for k in c["cells"].values() if k > 0)
        self.assertEqual(families(3), 3)         # one sector short of a vertex
        self.assertEqual(families(4), 4)         # d + 1 = 4 families


class ScalarLabelTests(unittest.TestCase):
    def test_level_banding_gives_requested_sector_count(self):
        field = np.linspace(0, 1, 400).reshape(20, 20)
        lab = scalar_sector_labels(field, n_levels=4)
        self.assertEqual(len(np.unique(lab)), 4)


if __name__ == "__main__":
    unittest.main()
