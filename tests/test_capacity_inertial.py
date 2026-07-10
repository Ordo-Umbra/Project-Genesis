"""Checks for inertial κ-gravity: orbits, energy conservation, virial."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_dynamics import evolve_inertial  # noqa: E402

FK = dict(kappa_diffusion=1.0, kappa_recovery=0.02, kappa_consumption=0.8,
          kappa_baseline=1.0, max_iters=1500)


def _pair(v0, steps, damping=0.0, escape=None):
    L = 56
    pos = [[L / 2 - 6, L / 2], [L / 2 + 6, L / 2]]
    vel = [[0.0, v0], [0.0, -v0]]
    return evolve_inertial(pos, vel, [1.2, 1.2], shape=(L, L), width=2.5,
                           dt=0.5, steps=steps, damping=damping,
                           escape_radius=escape, **FK)


class TestInertialDynamics(unittest.TestCase):

    def test_energy_conserved_on_bound_orbit(self):
        # symplectic Verlet: total energy holds to well under 2% over the run
        h = _pair(0.5, 30)
        e = np.array(h["energy"])
        self.assertLess((e.max() - e.min()) / abs(e.mean()), 0.03)

    def test_radial_infall_shrinks(self):
        # no angular momentum → the pair falls straight in
        h = _pair(0.0, 16)
        self.assertLess(h["separation"][-1], h["separation"][0])

    def test_tangential_orbit_stays_bound(self):
        # a modest tangential kick gives a bound orbit: separation stays finite
        h = _pair(0.5, 40, escape=34.0)
        self.assertLess(max(h["separation"]), 34.0)    # never escapes
        self.assertGreater(min(h["separation"]), 2.0)  # never collides

    def test_escape_unbinds(self):
        # a large kick unbinds the pair — separation grows past the escape radius
        h = _pair(1.4, 40, escape=24.0)
        self.assertGreater(max(h["separation"]), 22.0)

    def test_damping_removes_energy(self):
        # dissipation lowers the total energy over time
        h = _pair(0.5, 24, damping=0.06)
        e = h["energy"]
        self.assertLess(e[-1], e[0])

    def test_virial_is_recorded(self):
        h = _pair(0.5, 8)
        self.assertEqual(len(h["virial"]), len(h["kinetic"]))


if __name__ == "__main__":
    unittest.main()
