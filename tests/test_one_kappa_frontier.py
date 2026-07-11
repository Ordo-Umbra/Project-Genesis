"""Checks for the shared coherent-fraction operator and the CP² thermal sampler.

Covers ``coherent_fraction`` (one operator, both sectors), ``cp_action_density``,
``cp_coherent_fraction`` and ``cp_metropolis_sweep`` — the pieces the one-κ
frontier experiment rests on.  The experiment's *result* is a deliberate
boundary (the two κ's do not numerically coincide); these tests cover the
machinery, not a forced agreement.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.gauge import random_unitary  # noqa: E402
from project_genesis.gauge_topology import (  # noqa: E402
    action_density,
    self_dual_fraction,
)
from project_genesis.gauge_topology import topological_charge_density as gauge_q  # noqa: E402
from project_genesis.topological_charge import (  # noqa: E402
    coherent_fraction,
    cp_action,
    cp_action_density,
    cp_coherent_fraction,
    cp_metropolis_sweep,
    topological_charge_density,
)


def _random_cp(L, rng):
    z = rng.standard_normal((L, L, 3)) + 1j * rng.standard_normal((L, L, 3))
    return z / np.linalg.norm(z, axis=-1, keepdims=True)


class TestCoherentFractionOperator(unittest.TestCase):

    def test_ratio_and_zero_guard(self):
        e = np.array([2.0, 3.0]); q = np.array([0.5, -0.5])
        self.assertAlmostEqual(coherent_fraction(e, q), 1.0 / 5.0, places=12)
        self.assertEqual(coherent_fraction(np.zeros(3), np.ones(3)), 0.0)

    def test_is_the_gauge_self_dual_fraction(self):
        # one operator: self_dual_fraction ≡ coherent_fraction(e, q) for SU(3)
        rng = np.random.default_rng(0)
        links = random_unitary(rng, 3, (4, 4, 4, 4, 4), special=True, scale=0.7)
        self.assertAlmostEqual(
            self_dual_fraction(links),
            coherent_fraction(action_density(links), gauge_q(links)), places=12)

    def test_is_the_cp_coherent_fraction(self):
        rng = np.random.default_rng(1)
        z = _random_cp(24, rng)
        self.assertAlmostEqual(
            cp_coherent_fraction(z),
            coherent_fraction(cp_action_density(z), topological_charge_density(z)),
            places=12)


class TestCpActionDensity(unittest.TestCase):

    def test_sums_to_cp_action(self):
        rng = np.random.default_rng(2)
        z = _random_cp(20, rng)
        self.assertAlmostEqual(cp_action_density(z).sum(), cp_action(z), places=9)

    def test_bogomolny_bound_total(self):
        # Σe ≥ Σ|q| so the coherent fraction is in [0, 1]
        rng = np.random.default_rng(3)
        z = _random_cp(24, rng)
        f = cp_coherent_fraction(z)
        self.assertGreaterEqual(f, 0.0)
        self.assertLessEqual(f, 1.0)


class TestCpMetropolisSweep(unittest.TestCase):

    def test_preserves_normalisation_and_shape(self):
        rng = np.random.default_rng(4)
        z = _random_cp(16, rng)
        z2 = cp_metropolis_sweep(z, 1.5, rng, 0.5)
        self.assertEqual(z2.shape, z.shape)
        norms = np.linalg.norm(z2, axis=-1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-10))

    def test_thermalises_toward_lower_action(self):
        # from a hot start, sweeps at finite beta reduce the action
        rng = np.random.default_rng(5)
        z = _random_cp(24, rng)
        s0 = cp_action(z)
        for _ in range(15):
            z = cp_metropolis_sweep(z, 2.0, rng, 0.5)
        self.assertLess(cp_action(z), s0)

    def test_does_not_mutate_input(self):
        rng = np.random.default_rng(6)
        z = _random_cp(12, rng)
        z_copy = z.copy()
        _ = cp_metropolis_sweep(z, 1.0, rng, 0.5)
        self.assertTrue(np.array_equal(z, z_copy))


if __name__ == "__main__":
    unittest.main()
