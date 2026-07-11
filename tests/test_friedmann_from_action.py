"""Checks for the variational closure: the Friedmann relations from an action.

Covers ``minisuperspace_lagrangian``, ``hamiltonian_constraint`` and
``integrate_friedmann_action`` — that the acceleration equation (the a-Euler–
Lagrange equation) preserves ``H² = ρ`` as a first integral, that it is an
attractor, and that the resulting ``a(t)`` matches the stress-energy route.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_dynamics import (  # noqa: E402
    hamiltonian_constraint,
    integrate_friedmann_action,
    integrate_stress_energy,
    minisuperspace_lagrangian,
)


class TestMinisuperspacePieces(unittest.TestCase):

    def test_lagrangian_value(self):
        # L = -a adot^2 / N - N a^3 rho
        self.assertAlmostEqual(minisuperspace_lagrangian(2.0, 0.5, 0.1, 1.0),
                               -2.0 * 0.25 - 1.0 * 8.0 * 0.1, places=12)

    def test_constraint_zero_on_friedmann(self):
        # adot = a*sqrt(rho) → H = sqrt(rho) → C = H^2 - rho = 0
        rho = 0.09
        self.assertAlmostEqual(hamiltonian_constraint(2.0, 2.0 * np.sqrt(rho), rho),
                               0.0, places=12)

    def test_constraint_sign(self):
        # over-fast expansion → C > 0; too slow → C < 0
        self.assertGreater(hamiltonian_constraint(1.0, 0.5, 0.09), 0.0)
        self.assertLess(hamiltonian_constraint(1.0, 0.1, 0.09), 0.0)


class TestFriedmannFromAction(unittest.TestCase):

    def test_constraint_is_preserved(self):
        # H^2 = rho never imposed, yet stays satisfied along the a-EOM
        h = integrate_friedmann_action([0.05, 0.01], [0.0, -1.0], dt=0.1,
                                       steps=700)
        a = np.array(h["a"]); C = np.array(h["constraint"])
        self.assertLess(np.max(np.abs(C[a <= 6.0])), 1e-6)

    def test_reproduces_stress_energy_history(self):
        act = integrate_friedmann_action([0.05, 0.01], [0.0, -1.0], dt=0.1,
                                         steps=700)
        ten = integrate_stress_energy([0.05, 0.01], [0.0, -1.0], dt=0.1, steps=700)
        a1 = np.array(act["a"]); a2 = np.array(ten["a"]); m = a1 <= 6.0
        self.assertLess(np.max(np.abs(a1[m] - a2[m]) / a2[m]), 1e-5)

    def test_constraint_is_an_attractor(self):
        # a wrong initial expansion rate relaxes onto H^2 = rho
        rho0 = 0.06
        for f in (0.6, 1.5):
            h = integrate_friedmann_action([0.05, 0.01], [0.0, -1.0], dt=0.1,
                                           steps=700, adot0=f * np.sqrt(rho0))
            C = np.array(h["constraint"])
            self.assertGreater(abs(C[0]), 1e-3)          # starts violated
            self.assertLess(abs(C[-1]), abs(C[0]) * 1e-2)  # decays toward 0

    def test_decelerates_then_accelerates(self):
        h = integrate_friedmann_action([0.05, 0.01], [0.0, -1.0], dt=0.1,
                                       steps=700)
        q = np.array(h["q"])
        self.assertAlmostEqual(q[0], 0.25, delta=0.02)     # matter era
        self.assertLess(q[-1], 0.0)                         # Λ era
        self.assertAlmostEqual(q[-1], -1.0, delta=0.02)     # de Sitter


if __name__ == "__main__":
    unittest.main()
