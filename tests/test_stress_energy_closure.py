"""Checks for the relativistic closure: the stress-energy tensor and expansion.

Covers ``stress_energy_tensor``, ``covariant_conservation_rate``,
``friedmann_acceleration`` and ``integrate_stress_energy`` — that the tensor has
the perfect-fluid structure, that covariant conservation reproduces the
``a^{−3(1+w)}`` dilution laws, and that the coupled system reproduces the earlier
imposed-law cosmology.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_dynamics import (  # noqa: E402
    covariant_conservation_rate,
    friedmann_acceleration,
    integrate_scale_factor,
    integrate_stress_energy,
    stress_energy_tensor,
)


class TestStressEnergyTensor(unittest.TestCase):

    def test_perfect_fluid_structure(self):
        # dust (w=0) + vacuum (w=-1): T = diag(-rho, p, p, p)
        res = stress_energy_tensor([0.05, 0.01], [0.0, -1.0])
        self.assertAlmostEqual(res["rho"], 0.06, places=12)
        self.assertAlmostEqual(res["pressure"], -0.01, places=12)     # 0 + (-0.01)
        T = res["T"]
        self.assertEqual(T.shape, (4, 4))
        self.assertAlmostEqual(T[0, 0], -0.06, places=12)
        for i in (1, 2, 3):
            self.assertAlmostEqual(T[i, i], -0.01, places=12)
        self.assertAlmostEqual(np.sum(np.abs(T - np.diag(np.diag(T)))), 0.0)

    def test_effective_eos_between_components(self):
        # pure dust → w_eff = 0; pure vacuum → w_eff = -1
        self.assertAlmostEqual(stress_energy_tensor([0.1], [0.0])["w_eff"], 0.0)
        self.assertAlmostEqual(stress_energy_tensor([0.1], [-1.0])["w_eff"], -1.0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            stress_energy_tensor([0.1, 0.2], [0.0])


class TestConservationAndAcceleration(unittest.TestCase):

    def test_conservation_rate(self):
        # rho_dot = -3 H rho (1 + w)
        self.assertAlmostEqual(covariant_conservation_rate(0.2, 0.0, 0.5),
                               -3 * 0.5 * 0.2, places=12)         # dust
        self.assertAlmostEqual(covariant_conservation_rate(0.2, -1.0, 0.5), 0.0,
                               places=12)                          # vacuum: frozen
        self.assertAlmostEqual(covariant_conservation_rate(0.2, 1.0 / 3.0, 0.5),
                               -3 * 0.5 * 0.2 * (4.0 / 3.0), places=12)  # radiation

    def test_acceleration_sign(self):
        # dust decelerates (ä/a < 0); vacuum accelerates (ä/a > 0)
        self.assertLess(friedmann_acceleration(0.1, 0.0), 0.0)
        self.assertGreater(friedmann_acceleration(0.1, -0.1), 0.0)   # p = -rho


class TestIntegrateStressEnergy(unittest.TestCase):

    def test_conservation_reproduces_dilution_laws(self):
        h = integrate_stress_energy([0.05, 0.01, 0.02], [0.0, -1.0, 1.0 / 3.0],
                                    dt=0.1, steps=400)
        a = np.array(h["a"]); comp = np.array(h["rho_components"])
        m = a <= 6.0
        for i, w in enumerate((0.0, -1.0, 1.0 / 3.0)):
            rho = comp[:, i]
            analytic = rho[0] * a ** (-3.0 * (1.0 + w))
            err = np.max(np.abs(rho[m] - analytic[m])
                         / np.maximum(analytic[m], 1e-30))
            self.assertLess(err, 1e-4)             # a^{-3(1+w)} to high precision

    def test_decelerates_then_accelerates(self):
        h = integrate_stress_energy([0.05, 0.01], [0.0, -1.0], dt=0.1, steps=700)
        q = np.array(h["q"]); w = np.array(h["w_eff"])
        self.assertGreater(q[0], 0.0)               # matter era
        self.assertLess(q[-1], 0.0)                 # Λ era
        self.assertAlmostEqual(q[-1], -1.0, delta=0.02)     # de Sitter
        self.assertAlmostEqual(w[0], -1.0 / 6.0, delta=0.02)  # ρ=0.06, p=-0.01
        self.assertAlmostEqual(w[-1], -1.0, delta=0.02)

    def test_matches_imposed_law_cosmology(self):
        # coupled tensor reproduces integrate_scale_factor in the physical window
        hb = integrate_stress_energy([0.05, 0.01], [0.0, -1.0], dt=0.05, steps=400)
        hi = integrate_scale_factor(0.05, 0.01, dim=3, dt=0.05, steps=400)
        a1 = np.array(hb["a"]); a2 = np.array(hi["a"])
        m = a1 <= 6.0
        self.assertLess(np.max(np.abs(a1[m] - a2[m]) / a2[m]), 5e-3)


if __name__ == "__main__":
    unittest.main()
