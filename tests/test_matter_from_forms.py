"""Checks for the matter source read off the form spectrum.

Covers ``matter_energy_density`` and the physics the ``n3_matter_from_forms``
experiment rests on: a Bogomolny mass ladder ``E ∝ |Q|``, topological protection
of the charge under deformation, the ``ρ_m ∝ a^{-dim}`` dilution law that follows
from conserved rest energy, and the acceleration onset set by the form content.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_dynamics import (  # noqa: E402
    acceleration_onset,
    capacity_vacuum_density,
    matter_energy_density,
)
from project_genesis.stable_forms import structural_mass, winding_form  # noqa: E402
from project_genesis.topological_charge import cool, topological_charge  # noqa: E402


class TestMatterEnergyDensity(unittest.TestCase):

    def test_sum_over_volume(self):
        self.assertAlmostEqual(matter_energy_density([1.0, 2.0, 3.0], 2.0), 3.0,
                               places=12)

    def test_scalar_and_array_agree(self):
        self.assertAlmostEqual(matter_energy_density([2.5, 2.5], 5.0),
                               matter_energy_density(np.array([5.0]), 5.0),
                               places=12)

    def test_rejects_nonpositive_volume(self):
        with self.assertRaises(ValueError):
            matter_energy_density([1.0], 0.0)
        with self.assertRaises(ValueError):
            matter_energy_density([1.0], -3.0)


class TestBogomolnyLadder(unittest.TestCase):

    def test_energy_grows_with_charge(self):
        Es = [structural_mass(winding_form(48, Q, ring=3.0)) for Q in (1, 2, 3, 4)]
        for lo, hi in zip(Es, Es[1:]):
            self.assertLess(lo, hi)               # heavier form for higher charge

    def test_energy_is_linear_in_charge(self):
        q = np.array([1.0, 2.0, 3.0, 4.0])
        e = np.array([structural_mass(winding_form(64, int(Q), ring=3.0))
                      for Q in q])
        slope, floor = np.polyfit(q, e, 1)
        pred = slope * q + floor
        r2 = 1.0 - ((e - pred) ** 2).sum() / ((e - e.mean()) ** 2).sum()
        self.assertGreater(slope, 0.0)
        self.assertGreater(r2, 0.95)              # a clean Bogomolny ladder


class TestTopologicalProtection(unittest.TestCase):

    def test_cooling_recovers_charge_under_deformation(self):
        # a moderate deformation scrambles the raw charge; cooling restores Q
        Q = 2
        z0 = winding_form(64, Q, ring=3.0)
        rng = np.random.default_rng(3)
        z = z0 + 0.3 * (rng.standard_normal(z0.shape)
                        + 1j * rng.standard_normal(z0.shape))
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        self.assertAlmostEqual(topological_charge(cool(z, 80, 1.0)), float(Q),
                               delta=0.4)


class TestDilutionFromTopology(unittest.TestCase):

    def test_conserved_energy_dilutes_as_a_to_minus_dim(self):
        # fixed (topologically conserved) rest energy in a growing comoving
        # volume V0·a^dim → ρ_m ∝ a^{-dim}
        energies = [5.0, 10.0, 15.0]
        v0, dim = 8.0, 3
        a = np.geomspace(1.0, 50.0, 40)
        rho = np.array([matter_energy_density(energies, v0 * ai ** dim) for ai in a])
        slope = np.polyfit(np.log(a), np.log(rho), 1)[0]
        self.assertAlmostEqual(slope, -dim, places=6)


class TestCosmosFromContent(unittest.TestCase):

    def test_more_forms_push_acceleration_later(self):
        # ρ_m0 read from the form content; more matter → larger a_acc
        rho_L = capacity_vacuum_density(0.02, 1.0, 0.5)
        v0, dim = 13.0 ** 3, 3
        spec = float(sum(structural_mass(winding_form(48, Q, ring=3.0))
                         for Q in (1, 2, 3, 4)))
        a1 = acceleration_onset(matter_energy_density([spec], v0), rho_L, dim)
        a5 = acceleration_onset(matter_energy_density([spec] * 5, v0), rho_L, dim)
        self.assertGreater(a5, a1)
        # and the density scales linearly with the form content
        self.assertAlmostEqual(matter_energy_density([spec] * 5, v0),
                               5.0 * matter_energy_density([spec], v0), places=9)


if __name__ == "__main__":
    unittest.main()
