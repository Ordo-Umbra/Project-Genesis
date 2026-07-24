"""Checks for the Aharonov-Bohm statistical phase (wilson_loop / ab_phase)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.gauged_vortex import (  # noqa: E402
    ab_phase,
    local_flux,
    relax,
    seed_vortices,
    wilson_loop,
    zero_links,
)

TWO_PI = 2.0 * np.pi


def _relaxed(q, n=64, steps=2200):
    c0 = (n / 2 - 16, n / 2)
    c1 = (n / 2 + 16, n / 2)
    psi = seed_vortices(n, [c0, c1], [q, -q])
    out = relax(psi, zero_links(n), lam=2.0, beta=1.0, dt=0.05, steps=steps)
    return out["theta"], c0


class TestWilsonLoop(unittest.TestCase):

    def test_trivial_links_zero_holonomy(self):
        hol, val = wilson_loop(zero_links(40), (20, 20), 6)
        self.assertAlmostEqual(hol, 0.0)
        self.assertAlmostEqual(val.real, 1.0)

    def test_perimeter_holonomy_equals_enclosed_flux(self):
        # lattice Stokes: the square Wilson-loop holonomy = the enclosed flux
        theta, c0 = _relaxed(2)
        hol, _ = wilson_loop(theta, c0, 6)
        self.assertAlmostEqual(hol, local_flux(theta, c0, 6.0), delta=0.4)

    def test_holonomy_is_the_flux_quantum(self):
        # the robust (circular) AB holonomy = 2π·q
        theta, c0 = _relaxed(1)
        self.assertAlmostEqual(abs(local_flux(theta, c0, 6.0)) / TWO_PI, 1.0,
                               delta=0.15)


class TestAharonovBohmPhase(unittest.TestCase):
    """The statistical transmutation: a half charge sees the flux as (−1)^q."""

    def test_unit_charge_sees_no_phase(self):
        # Dirac: an integer flux quantum is invisible to a unit charge
        theta, c0 = _relaxed(1)
        self.assertGreater(ab_phase(theta, c0, 6, charge=1.0).real, 0.8)

    def test_half_charge_is_a_fermion_at_one_quantum(self):
        theta, c0 = _relaxed(1)
        self.assertLess(ab_phase(theta, c0, 6, charge=0.5).real, -0.8)

    def test_half_charge_is_a_boson_at_two_quanta(self):
        theta, c0 = _relaxed(2)
        self.assertGreater(ab_phase(theta, c0, 6, charge=0.5).real, 0.8)

    def test_statistical_phase_is_half_the_holonomy(self):
        theta, c0 = _relaxed(1)
        hol = local_flux(theta, c0, 6.0)
        stat = np.cos(0.5 * hol)
        self.assertAlmostEqual(stat, ab_phase(theta, c0, 6.0, charge=0.5).real,
                               places=6)


if __name__ == "__main__":
    unittest.main()
