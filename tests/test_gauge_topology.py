"""Checks for the 4-D SU(N) clover topological-charge instrument."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.gauge import random_unitary  # noqa: E402
from project_genesis.gauge_topology import (  # noqa: E402
    cool,
    mean_plaquette,
    staple_field,
    topological_charge,
    topological_charge_density,
    topological_susceptibility,
)


def _identity_links(L=4):
    links = np.zeros((4, L, L, L, L, 3, 3), dtype=complex)
    for mu in range(4):
        links[mu, ..., 0, 0] = links[mu, ..., 1, 1] = links[mu, ..., 2, 2] = 1.0
    return links


class TestTopologicalCharge(unittest.TestCase):

    def test_identity_is_zero(self):
        self.assertEqual(topological_charge(_identity_links()), 0.0)

    def test_pure_gauge_is_zero(self):
        # a gauge transform of the identity (U_μ(x)=g(x)g(x+μ)†) has Q=0
        rng = np.random.default_rng(0)
        L = 4
        g = random_unitary(rng, 3, (L, L, L, L), special=True)
        dag = lambda m: np.conjugate(np.swapaxes(m, -1, -2))
        links = np.stack([g @ dag(np.roll(g, -1, axis=mu)) for mu in range(4)])
        self.assertAlmostEqual(topological_charge(links), 0.0, places=6)

    def test_density_sums_to_total(self):
        rng = np.random.default_rng(1)
        links = random_unitary(rng, 3, (4, 4, 4, 4, 4), special=True, scale=0.7)
        self.assertAlmostEqual(
            float(topological_charge_density(links).sum()),
            topological_charge(links), places=9)

    def test_requires_4d(self):
        rng = np.random.default_rng(2)
        links = random_unitary(rng, 3, (3, 4, 4, 4), special=True)
        with self.assertRaises(ValueError):
            topological_charge_density(links)

    def test_cooling_leaves_identity_trivial(self):
        z = cool(_identity_links(), 3)
        self.assertAlmostEqual(topological_charge(z), 0.0, places=9)

    def test_cooling_reduces_action(self):
        # cooling descends the Wilson action → the mean plaquette rises to ~1
        rng = np.random.default_rng(3)
        links = random_unitary(rng, 3, (4, 6, 6, 6, 6), special=True, scale=0.7)
        p0 = mean_plaquette(links)
        p1 = mean_plaquette(cool(links, 8))
        self.assertGreater(p1, p0)
        self.assertGreater(p1, 0.85)     # cooling drives the field smooth


class TestStapleAndSusceptibility(unittest.TestCase):

    def test_staple_field_shape(self):
        rng = np.random.default_rng(4)
        links = random_unitary(rng, 3, (4, 4, 4, 4, 4), special=True)
        self.assertEqual(staple_field(links, 0).shape, links[0].shape)

    def test_susceptibility_formula(self):
        chi = topological_susceptibility([-1.0, 0.0, 1.0, 0.0], 100.0)
        self.assertAlmostEqual(chi, 0.5 / 100.0, places=12)

    def test_susceptibility_empty(self):
        self.assertEqual(topological_susceptibility([], 50.0), 0.0)


if __name__ == "__main__":
    unittest.main()
