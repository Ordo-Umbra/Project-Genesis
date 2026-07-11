"""Checks for gravity from the capacity field: the gravitational action as free energy.

Covers ``scale_capacity``, ``capacity_kinetic_energy``,
``capacity_scalar_acceleration`` and ``integrate_capacity_scale`` — that the
minisuperspace gravitational term ``−a ȧ²`` equals the capacity scalar's kinetic
free energy ``−a³ κ̇_s²``, that Friedmann is the balance ``κ̇_s² = ρ`` (preserved
and attracting), and that the capacity scalar rolling reproduces the cosmology.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_dynamics import (  # noqa: E402
    capacity_kinetic_energy,
    capacity_scalar_acceleration,
    integrate_capacity_scale,
    integrate_stress_energy,
    scale_capacity,
)


class TestCapacityScalarPieces(unittest.TestCase):

    def test_scale_capacity_is_log(self):
        self.assertAlmostEqual(scale_capacity(1.0), 0.0, places=12)
        self.assertAlmostEqual(scale_capacity(np.e), 1.0, places=12)

    def test_kinetic_energy_is_gravitational_term(self):
        # a^3 (adot/a)^2 == a adot^2 (whose negative is the minisuperspace term)
        for a, adot in ((2.0, 0.7), (1.3, -0.4), (5.0, 2.1)):
            self.assertAlmostEqual(capacity_kinetic_energy(a, adot), a * adot ** 2,
                                   places=12)

    def test_scalar_acceleration_vacuum_on_constraint(self):
        # on the balance kappa_dot^2 = rho, kappa_ddot = -3/2 (rho + p)
        # pure vacuum (p = -rho) → kappa_ddot = 0 (constant roll = de Sitter)
        rho = 0.02
        self.assertAlmostEqual(capacity_scalar_acceleration(np.sqrt(rho), -rho),
                               0.0, places=12)
        # dust (p = 0) → kappa_ddot = -3/2 kappa_dot^2 < 0
        self.assertLess(capacity_scalar_acceleration(0.3, 0.0), 0.0)


class TestIntegrateCapacityScale(unittest.TestCase):

    def test_gravitational_term_identity_along_trajectory(self):
        h = integrate_capacity_scale([0.05, 0.01], [0.0, -1.0], dt=0.1, steps=700)
        a = np.array(h["a"]); kdot = np.array(h["kappa_dot"])
        adot = a * kdot
        grav_posited = -a * adot ** 2
        grav_capacity = -np.array([capacity_kinetic_energy(ai, adi)
                                   for ai, adi in zip(a, adot)])
        rel = np.abs(grav_posited - grav_capacity) / np.maximum(
            np.abs(grav_posited), 1e-30)
        self.assertLess(np.max(rel), 1e-12)          # identical up to roundoff

    def test_energy_balance_preserved(self):
        # kappa_dot^2 = rho never imposed, yet holds along the roll
        h = integrate_capacity_scale([0.05, 0.01], [0.0, -1.0], dt=0.1, steps=700)
        a = np.array(h["a"]); C = np.array(h["constraint"])
        self.assertLess(np.max(np.abs(C[a <= 6.0])), 1e-6)

    def test_reproduces_stress_energy_history(self):
        cap = integrate_capacity_scale([0.05, 0.01], [0.0, -1.0], dt=0.1, steps=700)
        ten = integrate_stress_energy([0.05, 0.01], [0.0, -1.0], dt=0.1, steps=700)
        a1 = np.array(cap["a"]); a2 = np.array(ten["a"]); m = a1 <= 6.0
        self.assertLess(np.max(np.abs(a1[m] - a2[m]) / a2[m]), 1e-5)

    def test_balance_is_attractor(self):
        rho0 = 0.06
        for f in (0.6, 1.5):
            h = integrate_capacity_scale([0.05, 0.01], [0.0, -1.0], dt=0.1,
                                         steps=700, kappadot0=f * np.sqrt(rho0))
            C = np.array(h["constraint"])
            self.assertGreater(abs(C[0]), 1e-3)
            self.assertLess(abs(C[-1]), abs(C[0]) * 1e-2)

    def test_decelerates_then_accelerates(self):
        h = integrate_capacity_scale([0.05, 0.01], [0.0, -1.0], dt=0.1, steps=700)
        q = np.array(h["q"])
        self.assertAlmostEqual(q[0], 0.25, delta=0.02)
        self.assertLess(q[-1], 0.0)
        self.assertAlmostEqual(q[-1], -1.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()
