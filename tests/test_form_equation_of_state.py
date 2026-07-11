"""Checks for the equation of state of the form matter.

Covers ``equation_of_state_from_dilution`` and ``gas_equation_of_state`` and the
physics the ``n3_form_equation_of_state`` experiment rests on: cold forms are
dust (``w → 0``), the ultra-relativistic gas is radiation (``w → 1/dim``), the
capacity vacuum is a cosmological constant (``w = −1``), and ``w`` is intensive.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_dynamics import (  # noqa: E402
    equation_of_state_from_dilution,
    gas_equation_of_state,
)


class TestEquationOfStateFromDilution(unittest.TestCase):

    def test_dust_vacuum_radiation(self):
        dim = 3
        self.assertAlmostEqual(equation_of_state_from_dilution(dim, dim), 0.0,
                               places=12)        # ρ ∝ a^{-dim} → dust
        self.assertAlmostEqual(equation_of_state_from_dilution(0.0, dim), -1.0,
                               places=12)        # constant → Λ
        self.assertAlmostEqual(equation_of_state_from_dilution(dim + 1, dim),
                               1.0 / dim, places=12)   # a^{-(dim+1)} → radiation

    def test_matches_continuity_exponent(self):
        # inverse relation: n = dim(1 + w)
        for dim in (2, 3, 4):
            for w in (-1.0, -0.3, 0.0, 0.5, 1.0):
                n = dim * (1.0 + w)
                self.assertAlmostEqual(equation_of_state_from_dilution(n, dim), w,
                                       places=12)


class TestGasEquationOfState(unittest.TestCase):

    def test_cold_gas_is_dust(self):
        m = np.array([2.0, 3.0, 5.0])
        v = np.full((3, 3), 1e-4)
        res = gas_equation_of_state(m, v)
        self.assertLess(abs(res["w"]), 1e-6)      # w → 0

    def test_ultrarelativistic_gas_is_radiation(self):
        dim = 3
        m = np.array([1.0, 1.0])
        # speed → 1 along one axis: w → 1/dim
        v = np.zeros((2, dim)); v[:, 0] = 0.99999
        res = gas_equation_of_state(m, v)
        self.assertAlmostEqual(res["w"], 1.0 / dim, delta=0.02)

    def test_w_increases_with_dispersion(self):
        m = np.ones(20)
        rng = np.random.default_rng(0)
        prev = -1.0
        for sigma in (0.05, 0.15, 0.3, 0.5, 0.7):
            v = rng.normal(0.0, sigma, (20, 3))
            v = np.clip(v, -0.5, 0.5)              # keep speeds < 1
            w = gas_equation_of_state(m, v)["w"]
            self.assertGreater(w, prev)
            prev = w

    def test_w_is_intensive(self):
        # w = p/ρ does not depend on the box volume
        m = np.array([1.0, 2.0])
        v = np.array([[0.3, 0.0, 0.0], [0.0, 0.4, 0.0]])
        w1 = gas_equation_of_state(m, v, volume=1.0)["w"]
        w2 = gas_equation_of_state(m, v, volume=7.5)["w"]
        self.assertAlmostEqual(w1, w2, places=12)

    def test_density_and_pressure_scale_with_volume(self):
        m = np.array([1.0, 2.0])
        v = np.array([[0.3, 0.0], [0.0, 0.4]])
        a = gas_equation_of_state(m, v, volume=1.0)
        b = gas_equation_of_state(m, v, volume=2.0)
        self.assertAlmostEqual(a["rho"], 2.0 * b["rho"], places=12)
        self.assertAlmostEqual(a["pressure"], 2.0 * b["pressure"], places=12)

    def test_rejects_superluminal_and_bad_volume(self):
        with self.assertRaises(ValueError):
            gas_equation_of_state([1.0], np.array([[1.0, 0.0, 0.0]]))
        with self.assertRaises(ValueError):
            gas_equation_of_state([1.0], np.array([[0.1]]), volume=0.0)


if __name__ == "__main__":
    unittest.main()
