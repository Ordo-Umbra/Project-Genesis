"""Checks for the stable-forms corpus and structural/gravitational masses."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.stable_forms import (  # noqa: E402
    distinction_density,
    form_charge,
    gravitational_mass,
    structural_mass,
    trivial_bump,
    winding_form,
)
from project_genesis.topological_charge import cool  # noqa: E402


class TestCorpus(unittest.TestCase):

    def test_winding_forms_have_integer_charge(self):
        for Q in (1, 2, 3):
            self.assertAlmostEqual(form_charge(winding_form(48, Q)), float(Q),
                                   places=1)

    def test_anticharge(self):
        self.assertAlmostEqual(form_charge(winding_form(48, -2)), -2.0, places=1)

    def test_trivial_bump_is_chargeless(self):
        self.assertAlmostEqual(form_charge(trivial_bump(48)), 0.0, places=2)

    def test_mass_spectrum_is_a_rising_ladder(self):
        # structural mass increases monotonically with charge
        E = [structural_mass(cool(winding_form(56, Q, ring=3.0), 6, rate=0.4))
             for Q in (1, 2, 3)]
        self.assertLess(E[0], E[1])
        self.assertLess(E[1], E[2])

    def test_distinction_density_sums_to_structural_mass(self):
        psi = winding_form(40, 2)
        self.assertAlmostEqual(float(distinction_density(psi).sum()),
                               structural_mass(psi), places=9)

    def test_distinction_density_requires_2d(self):
        with self.assertRaises(ValueError):
            distinction_density(np.ones((4, 4, 4, 3), dtype=complex))


class TestStabilityAndGravity(unittest.TestCase):

    def test_topological_form_preserves_charge_under_cooling(self):
        psi = winding_form(56, 2, ring=3.0)
        self.assertAlmostEqual(form_charge(cool(psi, 20, rate=0.4)), 2.0,
                               places=1)

    def test_trivial_bump_decays_below_a_form(self):
        # a form keeps a large energy floor; the trivial bump relaxes low
        form = cool(winding_form(56, 1, ring=3.0), 30, rate=0.4)
        bump = cool(trivial_bump(56, 0.8, 5.0), 30, rate=0.4)
        self.assertGreater(structural_mass(form), structural_mass(bump))

    def test_gravitational_mass_positive_and_grows_with_charge(self):
        m1 = gravitational_mass(winding_form(48, 1), max_iters=3000)
        m2 = gravitational_mass(winding_form(48, 2), max_iters=3000)
        self.assertGreater(m1, 0.0)
        self.assertGreater(m2, m1)

    def test_structural_and_gravitational_mass_proportional(self):
        # weak-well regime: M_grav ∝ E across the spectrum (equivalence)
        forms = [cool(winding_form(56, Q, ring=3.0), 6, rate=0.4)
                 for Q in (1, 2, 3)]
        E = np.array([structural_mass(p) for p in forms])
        Mg = np.array([gravitational_mass(p, kappa_consumption=0.2,
                                          max_iters=4000) for p in forms])
        slope = (E * Mg).sum() / (E * E).sum()
        r2 = 1.0 - ((Mg - slope * E) ** 2).sum() / ((Mg - Mg.mean()) ** 2).sum()
        self.assertGreater(r2, 0.99)


if __name__ == "__main__":
    unittest.main()
