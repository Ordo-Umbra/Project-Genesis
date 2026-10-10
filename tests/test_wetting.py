"""Tests for pairwise-coupled multiphase dynamics.

The wetting experiment only says something about the published model if the
pairwise step *is* the published step when the couplings are uniform, so that
equivalence is pinned first. The rest pins the two levers the experiment pulls:
which sectors are conserved, and the linear growth rate a dormant sector sees.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.multiphase import step_multiphase_conserved  # noqa: E402
from project_genesis.wetting import (  # noqa: E402
    coupling_matrix, growth_rate, step_multiphase_pairwise,
)


class EquivalenceTests(unittest.TestCase):
    def test_uniform_coupling_reproduces_published_step(self):
        for ndim in (2, 3):
            f = np.random.default_rng(ndim).random((3, *([12] * ndim))) * 0.1
            a, b = f.copy(), f.copy()
            for _ in range(50):
                a, _ = step_multiphase_conserved(a, None, gamma=1.5, dt=0.1)
                b = step_multiphase_pairwise(b, coupling_matrix(3, 1.5))
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


class CouplingTests(unittest.TestCase):
    def test_overrides_are_symmetric(self):
        g = coupling_matrix(3, 1.5, {(0, 2): 0.6})
        self.assertEqual(g[0, 2], 0.6)
        self.assertEqual(g[2, 0], 0.6)
        self.assertEqual(g[0, 1], 1.5)
        self.assertTrue(np.all(np.diag(g) == 0))

    def test_self_coupling_rejected(self):
        with self.assertRaises(ValueError):
            coupling_matrix(3, 1.5, {(1, 1): 0.5})

    def test_asymmetric_matrix_rejected(self):
        g = coupling_matrix(2)
        g[0, 1] = 0.3
        with self.assertRaises(ValueError):
            step_multiphase_pairwise(np.zeros((2, 4, 4)), g)


class ConservationTests(unittest.TestCase):
    def test_conserved_sector_keeps_its_total_free_one_does_not(self):
        rng = np.random.default_rng(0)
        f = rng.random((3, 24, 24))
        f[2] *= 0.01
        g = coupling_matrix(3, 1.5, {(0, 2): 0.3, (1, 2): 0.3})
        start = f.sum(axis=(1, 2))
        for _ in range(100):
            f = step_multiphase_pairwise(f, g, conserved=(True, True, False))
        end = f.sum(axis=(1, 2))
        np.testing.assert_allclose(end[:2], start[:2], rtol=1e-10)
        self.assertGreater(end[2], 10 * start[2])


class GrowthRateTests(unittest.TestCase):
    def test_bulk_rate_is_one_minus_two_g(self):
        f = np.zeros((3, 4, 4))
        f[0] = 1.0
        for gv in (0.3, 0.5, 1.5):
            lam = growth_rate(f, coupling_matrix(3, 1.5, {(0, 2): gv}), 2)
            np.testing.assert_allclose(lam, 1 - 2 * gv)

    def test_wall_is_less_suppressing_than_bulk(self):
        f = np.zeros((3, 1, 2))
        f[0, 0, 0] = 1.0                      # bulk A
        f[0, 0, 1] = f[1, 0, 1] = 0.6         # A|B wall
        lam = growth_rate(f, coupling_matrix(3, 1.5, {(0, 2): 0.6, (1, 2): 0.6}), 2)
        self.assertGreater(lam[0, 1], lam[0, 0])


if __name__ == "__main__":
    unittest.main()
