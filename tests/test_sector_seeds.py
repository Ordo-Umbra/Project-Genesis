"""Tests for the κ-as-soil thermal sector-seed rooting primitives.

These check the thermal analogue of ``engine._plant_seed`` faithfully
reproduces its rule: root only where local capacity ≥ threshold, blend
+ renormalise on success, consume capacity, leave the field untouched on
barren soil.
"""

from __future__ import annotations

import numpy as np

import unittest

from project_genesis.sector_seeds import (
    DEFAULT_KAPPA_COST,
    fertile_fraction,
    local_kappa_mean,
    make_sector_seed,
    order_in_footprint,
    plant_sector_seed,
)


class TestSeedConstruction(unittest.TestCase):

    def test_seed_is_normalised_and_aligned(self):
        for dim in (2, 3):
            seed = make_sector_seed(3, 4, sector=1, dim=dim)
            self.assertEqual(seed.shape, (4,) * dim + (3,))
            self.assertTrue(np.allclose(np.linalg.norm(seed, axis=-1), 1.0))
            self.assertTrue(np.all(np.argmax(np.abs(seed) ** 2, axis=-1) == 1))


class TestFertileFraction(unittest.TestCase):

    def test_counts_sites_above_threshold(self):
        kappa = np.array([[0.1, 0.4], [0.5, 0.2]])
        self.assertEqual(fertile_fraction(kappa, 0.3), 0.5)

    def test_all_barren_is_zero(self):
        self.assertEqual(fertile_fraction(np.full((5, 5), 0.1), 0.3), 0.0)

    def test_all_fertile_is_one(self):
        self.assertEqual(fertile_fraction(np.full((5, 5), 0.9), 0.3), 1.0)


class TestFootprintHelpers(unittest.TestCase):

    def test_local_kappa_mean_periodic(self):
        kappa = np.zeros((6, 6))
        kappa[5, 5] = kappa[5, 0] = kappa[0, 5] = kappa[0, 0] = 1.0
        # 2×2 footprint at the corner wraps to cover all four unit cells
        self.assertAlmostEqual(local_kappa_mean(kappa, (5, 5), 2), 1.0)

    def test_order_in_footprint_ordered_vs_mixed(self):
        psi = np.zeros((8, 8, 3), dtype=complex)
        psi[..., 0] = 1.0
        self.assertAlmostEqual(order_in_footprint(psi, (2, 2), 4), 1.0)
        mixed = np.full((8, 8, 3), 1.0 / np.sqrt(3), dtype=complex)
        self.assertAlmostEqual(order_in_footprint(mixed, (2, 2), 4),
                               1.0 / 3.0, places=6)


class TestPlantingRule(unittest.TestCase):

    def setUp(self):
        self.seed = make_sector_seed(3, 4, sector=1, dim=2)
        # a *disordered* field — the meaningful recall target (regenerating
        # structure where it has been lost, not overwriting a rival domain)
        self.psi = np.full((12, 12, 3), 1.0 / np.sqrt(3), dtype=complex)

    def test_roots_in_fertile_soil_and_consumes_capacity(self):
        kappa = np.full((12, 12), 0.8)
        psi2, kappa2, rooted, lk = plant_sector_seed(
            self.psi, kappa, self.seed, (2, 2))
        self.assertTrue(rooted)
        self.assertAlmostEqual(lk, 0.8)
        # seed's sector now dominates the (previously disordered) footprint
        foot = psi2[2:6, 2:6]
        self.assertTrue(np.all(np.argmax(np.abs(foot) ** 2, axis=-1) == 1))
        # capacity drawn down by the cost, only inside the footprint
        self.assertAlmostEqual(kappa2[2:6, 2:6].mean(), 0.8 * (1 - DEFAULT_KAPPA_COST))
        self.assertAlmostEqual(kappa2[8, 8], 0.8)  # untouched elsewhere

    def test_barren_soil_rejects_seed_and_leaves_field_untouched(self):
        kappa = np.full((12, 12), 0.1)
        psi2, kappa2, rooted, lk = plant_sector_seed(
            self.psi, kappa, self.seed, (2, 2))
        self.assertFalse(rooted)
        self.assertTrue(np.array_equal(psi2, self.psi))
        self.assertTrue(np.array_equal(kappa2, kappa))

    def test_planting_keeps_psi_normalised(self):
        kappa = np.full((12, 12), 0.5)
        psi2, _, rooted, _ = plant_sector_seed(
            self.psi, kappa, self.seed, (5, 5))
        self.assertTrue(rooted)
        self.assertTrue(np.allclose(np.linalg.norm(psi2, axis=-1), 1.0))

    def test_inputs_not_mutated(self):
        kappa = np.full((12, 12), 0.8)
        psi_before, kappa_before = self.psi.copy(), kappa.copy()
        plant_sector_seed(self.psi, kappa, self.seed, (2, 2))
        self.assertTrue(np.array_equal(self.psi, psi_before))
        self.assertTrue(np.array_equal(kappa, kappa_before))

    def test_threshold_zero_always_roots(self):
        kappa = np.zeros((12, 12))
        _, _, rooted, _ = plant_sector_seed(
            self.psi, kappa, self.seed, (2, 2), threshold=0.0)
        self.assertTrue(rooted)


if __name__ == "__main__":
    unittest.main()
