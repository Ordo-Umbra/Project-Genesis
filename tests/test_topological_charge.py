"""Checks for the geometric CP^(N-1) topological-charge estimator."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.topological_charge import (  # noqa: E402
    instanton_density,
    topological_charge,
    topological_charge_density,
    topological_susceptibility,
)


def _winding(L=24, lam=4.0, sector=2, charge=+1):
    """A constructed CP^1 instanton (winding ±1) embedded in ℂ^3."""
    xs = np.arange(L) - L / 2 + 0.5
    X, Y = np.meshgrid(xs, xs)
    w = (X + 1j * Y) / lam
    if charge < 0:
        w = np.conj(w)
    z = np.zeros((L, L, 3), dtype=complex)
    z[..., 0] = 1.0
    z[..., 1] = w
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    return z


class TestTopologicalCharge(unittest.TestCase):

    def test_uniform_field_is_zero(self):
        psi = np.zeros((8, 8, 3), dtype=complex)
        psi[..., 0] = 1.0
        self.assertEqual(topological_charge(psi), 0.0)

    def test_charge_is_integer(self):
        rng = np.random.default_rng(1)
        psi = rng.standard_normal((10, 10, 3)) + 1j * rng.standard_normal((10, 10, 3))
        psi /= np.linalg.norm(psi, axis=-1, keepdims=True)
        Q = topological_charge(psi)
        self.assertAlmostEqual(Q, round(Q), places=9)

    def test_local_phase_invariance(self):
        # the CP gauge freedom: an independent phase at every site leaves Q fixed
        rng = np.random.default_rng(2)
        psi = rng.standard_normal((9, 9, 3)) + 1j * rng.standard_normal((9, 9, 3))
        psi /= np.linalg.norm(psi, axis=-1, keepdims=True)
        theta = rng.uniform(0, 2 * np.pi, (9, 9))
        psi_g = psi * np.exp(1j * theta)[..., None]
        self.assertAlmostEqual(topological_charge(psi), topological_charge(psi_g),
                               places=9)

    def test_constructed_instanton_has_unit_charge(self):
        self.assertAlmostEqual(topological_charge(_winding(charge=+1)), 1.0, places=2)

    def test_anti_instanton_has_negative_charge(self):
        self.assertAlmostEqual(topological_charge(_winding(charge=-1)), -1.0, places=2)

    def test_density_sums_to_total(self):
        psi = _winding()
        self.assertAlmostEqual(
            float(topological_charge_density(psi).sum()),
            topological_charge(psi), places=9)

    def test_instanton_density_nonnegative(self):
        self.assertGreaterEqual(instanton_density(_winding()), 0.0)

    def test_rejects_non_2d_field(self):
        with self.assertRaises(ValueError):
            topological_charge_density(np.ones((4, 4, 4, 3), dtype=complex))


class TestSusceptibility(unittest.TestCase):

    def test_susceptibility_formula(self):
        # ⟨Q²⟩ / V for charges [-1, 0, 1, 0] over volume 100 → 0.5/100
        chi = topological_susceptibility([-1, 0, 1, 0], 100.0)
        self.assertAlmostEqual(chi, 0.5 / 100.0, places=12)

    def test_empty_is_zero(self):
        self.assertEqual(topological_susceptibility([], 100.0), 0.0)


if __name__ == "__main__":
    unittest.main()
