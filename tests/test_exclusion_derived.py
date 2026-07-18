"""Checks for the framework-derived exclusion (no-cloning) term.

The derived density-dependent form ``e(ρ) = c²rρ²/((r+cρ)(r+2cρ))`` —
the extensivity gap ``2F(ρ) − F(2ρ)`` of the homogeneous capacity free
energy — replaces the hand-picked constant contact coefficient.  These
tests guard its four symbolic properties and the sign/Lyapunov
bookkeeping of the ``contact_derived`` force path.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_waves import (  # noqa: E402
    evolve_inertial_retarded,
    exclusion_energy_density,
    exclusion_energy_derivative,
)

R, C = 0.02, 0.8          # the exclusion-core operating point


def e(rho):
    return exclusion_energy_density(rho, kappa_recovery=R,
                                    kappa_consumption=C)


class DerivedFormTests(unittest.TestCase):
    def test_dilute_limit_is_b_over_2_rho_squared(self):
        # e(ρ)/(ρ²/2) → b = 2c²/r as ρ → 0 (rel err < 1% at ρ = 1e-4·r/c)
        rho = 1e-4 * R / C
        b_dilute = 2.0 * C * C / R
        b_eff = float(e(rho) / (0.5 * rho * rho))
        self.assertLess(abs(b_eff - b_dilute) / b_dilute, 0.01)

    def test_clone_refusal_window_and_its_inversion(self):
        # E_x(2ρ) > 2E_x(ρ) exactly for ρ < r/(2c), inverting above
        rho_star = R / (2.0 * C)
        below = rho_star * np.array([1e-3, 0.1, 0.5, 0.99])
        above = rho_star * np.array([1.01, 2.0, 10.0, 100.0])
        gap_below = e(2.0 * below) - 2.0 * e(below)
        gap_above = e(2.0 * above) - 2.0 * e(above)
        self.assertTrue(np.all(gap_below > 0.0))   # clones refused
        self.assertTrue(np.all(gap_above < 0.0))   # inequality inverted

    def test_saturation_is_r_over_2(self):
        rho = 1e3 * R / C
        self.assertLess(abs(float(e(rho)) - R / 2.0) / (R / 2.0), 0.01)

    def test_derivative_matches_and_is_positive(self):
        rho = np.geomspace(1e-3 * R / C, 10.0, 12)
        ana = exclusion_energy_derivative(rho, kappa_recovery=R,
                                          kappa_consumption=C)
        h = 1e-6 * rho
        num = (e(rho + h) - e(rho - h)) / (2.0 * h)
        self.assertTrue(np.all(ana > 0.0))  # repulsive per site, always
        self.assertTrue(np.allclose(ana, num, rtol=1e-5, atol=1e-12))

    def test_contact_b_and_contact_derived_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            evolve_inertial_retarded(
                [[10.0, 10.0], [14.0, 10.0]], [[0.0, 0.0], [0.0, 0.0]],
                [1.0, 1.0], shape=(24, 24), steps=2,
                contact_b=0.2, contact_derived=True,
                kappa_recovery=R, kappa_consumption=C)


class DerivedContactForceTests(unittest.TestCase):
    def test_derived_contact_repels_overlapping_masses(self):
        # Two overlapping Gaussian blobs (width 2.5, separation 3), with
        # amplitudes small enough that the whole stack sits inside the
        # refusal window ρ < r/(2c) = 0.0125 — the regime where the
        # derived term is a genuine exclusion (b_eff ≈ 2c²/r).  The
        # exclusion contribution (derived run minus no-exclusion
        # control, isolating it from κ-gravity) must push the blobs
        # AWAY from each other.  (Above the window the e′-weighted
        # overlap force inverts — measured, and documented in
        # Docs/Deriving_The_Exclusion_Coefficient.md.)
        mass = 0.01
        common = dict(shape=(48, 48), width=2.5, tau=1.0, dt=0.1,
                      steps=2, record_every=1, kappa_recovery=R,
                      kappa_consumption=C)
        pos = [[22.5, 24.0], [25.5, 24.0]]
        vel = [[0.0, 0.0], [0.0, 0.0]]
        hd = evolve_inertial_retarded(pos, vel, [mass, mass],
                                      contact_derived=True, **common)
        hc = evolve_inertial_retarded(pos, vel, [mass, mass], **common)
        dv = (np.asarray(hd["velocities"])[1]
              - np.asarray(hc["velocities"])[1])
        self.assertLess(dv[0, 0], 0.0)   # left blob pushed further left
        self.assertGreater(dv[1, 0], 0.0)  # right blob pushed right

    def test_derived_contact_lyapunov_still_holds(self):
        # The force is the exact gradient of E_x = Σ e(ρ_tot) and E_x is
        # in the recorded energy, so the Lyapunov law survives the
        # density-dependent form (same argument as contact_b).
        hist = evolve_inertial_retarded(
            [[18.0, 24.0], [30.0, 24.0]], [[0.0, 0.6], [0.0, -0.6]],
            [1.2, 1.2], shape=(48, 48), tau=1.0, dt=0.1, steps=400,
            record_every=20, contact_derived=True,
            kappa_recovery=R, kappa_consumption=C)
        e_rec = np.asarray(hist["energy"])
        self.assertLess(e_rec[-1], e_rec[0])          # net dissipation
        self.assertTrue(np.all(np.diff(e_rec) <= 1e-6))  # Lyapunov
        self.assertTrue(all(d >= 0.0 for d in hist["dissipation"]))


if __name__ == "__main__":
    unittest.main()
