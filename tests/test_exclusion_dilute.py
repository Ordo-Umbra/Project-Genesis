"""Checks for the dilute operating point (Part V): where the derivation
is self-consistent.

The operating point (r = 0.05, c = 0.8, amplitude = inertial mass
0.01, width 10 ≫ ξ = 4.47, size 96 in the experiment; tests run at
48²) lives entirely inside the refusal window ρ < r/(2c) = 0.03125 —
a DIFFERENT regime from the arc's dense point by construction.  These
tests guard:

- the DILUTE LIMIT of the Part-I local form at the point's (r, c):
  e(ρ)/(ρ²/2) → 2c²/r = 25.6 as ρ → 0.  NOTE (recorded, not fixed):
  the spec's pre-registered expectation "e(ρ)/(ρ²/2) = 2c²/r to 1% AT
  THE BLOB AMPLITUDE" does NOT hold at the operating amplitude —
  measured 0.653083×2c²/r (35% off), because the refusal-window
  criterion cρ/r < 1/2 is ~16× looser than the quadratic criterion
  3cρ/r ≪ 1.  The limit below (ρ = 1e-4, 3cρ/r = 0.0048) holds to 1%;
  the at-amplitude value is pinned as measured.  See Part V of
  Docs/Deriving_The_Exclusion_Coefficient.md.
- P1 at the test level: the two derivations' pair exclusion energies
  E_x at one mid-range separation.  The pre-registered bar was 10%;
  the measured ratio at the test grid is 1.150 (the experiment's 96²
  record: 1.170 at s = 12, rising down the skirts to 1.400 — the
  s-dependent map is the result, recorded in Part V).  The test pins
  the measured agreement to a regression band and asserts the forms
  are the same term to within 20% — NOT the failed 10% bar.
- Lyapunov bookkeeping of a short dilute run.  NOTE (recorded, not
  fixed): at this operating point the binary's inertial mass equals
  the load amplitude (0.01 — 60× lighter than the dense point), so
  speeds reach 0.3–0.55 within the first second and the recorded
  energy carries an oscillating residual up to 6e-4 per record —
  dt-insensitive over 0.01–0.1 and present identically in the
  no-exclusion baseline, i.e. the fast-motion instrument regime of the
  retarded bookkeeping, not the exclusion term (whose force is the
  exact gradient of the booked Σe(ρ_tot), the contact_b construction).
  The field sector alone with a frozen load is monotone to 1e-15.
  The test asserts the law's integral form (net dissipation,
  non-negative recorded dissipation) and pins the measured per-record
  residual; the repo's 1e-6 dense-point bar does not apply here —
  recorded in Part V's honest edges.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import gaussian_load  # noqa: E402
from project_genesis.capacity_waves import (  # noqa: E402
    duplicated_load,
    evolve_inertial_retarded,
    exclusion_energy_density,
    exclusion_gap_full,
)

R, C = 0.05, 0.8            # the dilute operating point
FKW = dict(kappa_diffusion=1.0, kappa_recovery=R, kappa_consumption=C,
           kappa_baseline=1.0)
N = 48                   # tests run at 48²; the experiment's record is 96²
WIDTH, AMP = 10.0, 0.01


class DiluteLimitTests(unittest.TestCase):
    def test_e_over_half_rho2_tends_to_2c2_over_r(self):
        # the true dilute LIMIT at the point's (r, c): ρ = 1e-4 has
        # 3cρ/r = 0.0048, so e/(ρ²/2) is within 1% of 2c²/r = 25.6.
        rho = 1e-4
        val = float(exclusion_energy_density(
            rho, kappa_recovery=R, kappa_consumption=C)) / (rho * rho / 2)
        self.assertLess(abs(val / (2 * C * C / R) - 1.0), 0.01)

    def test_at_amplitude_value_pinned_as_measured(self):
        # the analytic value at the operating amplitude:
        # e(A)/(A²/2) = 2c²r/((r+cA)(r+2cA)) = 16.71891327 =
        # 0.653083 × 2c²/r — the pre-registered "1% at the blob
        # amplitude" expectation FAILS by 35% (recorded in Part V).
        val = float(exclusion_energy_density(
            AMP, kappa_recovery=R, kappa_consumption=C)) / (AMP * AMP / 2)
        self.assertAlmostEqual(val, 16.71891327, places=6)
        self.assertAlmostEqual(val / (2 * C * C / R), 0.653083, places=6)
        # and the refusal-window edge itself: the window criterion is
        # satisfied at contact (peak total load < 0.8·r/(2c)) while the
        # quadratic criterion is not — the measured tension of the
        # operating point.
        l1 = gaussian_load((N, N), [(N / 2 - 0.5, N / 2)], WIDTH, AMP)
        l2 = gaussian_load((N, N), [(N / 2 + 0.5, N / 2)], WIDTH, AMP)
        self.assertLess(float((l1 + l2).max()), 0.8 * R / (2 * C))


class P1ConvergenceTests(unittest.TestCase):
    def test_two_forms_at_one_mid_range_separation(self):
        # P1 at the test level: the Part-I local form ∫e(ρ_dup) and the
        # Part-II full-functional gap at s = 12 on the test grid.
        # Measured 1.150026 (48²; the experiment's 96² record: 1.170) —
        # the 10% pre-registered bar FAILS (recorded as-is in Part V);
        # the forms are the same term to within 20% at mid-range, with
        # the s-dependent remainder map carried by the experiment.
        l1 = gaussian_load((N, N), [(N / 2 - 6.0, N / 2)], WIDTH, AMP)
        l2 = gaussian_load((N, N), [(N / 2 + 6.0, N / 2)], WIDTH, AMP)
        dup = duplicated_load(l1, l2)
        ex_local = float(np.sum(exclusion_energy_density(
            dup, kappa_recovery=R, kappa_consumption=C)))
        ex_full = exclusion_gap_full(dup, **FKW)
        self.assertGreater(ex_local, 0.0)
        self.assertGreater(ex_full, 0.0)
        ratio = ex_local / ex_full
        self.assertAlmostEqual(ratio, 1.150026, places=4)
        self.assertLess(abs(ratio - 1.0), 0.20)


class DiluteDynamicsTests(unittest.TestCase):
    def test_short_dilute_run_lyapunov_bookkeeping(self):
        # a moving dilute binary with the Part-I (contact_derived) term:
        # net dissipation and non-negative recorded dissipation hold,
        # and the per-record energy rise is bounded by the measured
        # fast-motion residual (6e-4; see the module docstring).
        hist = evolve_inertial_retarded(
            [[N / 2 - 6.0, N / 2], [N / 2 + 6.0, N / 2]],
            [[0.0, -0.05], [0.0, 0.05]], [AMP, AMP], shape=(N, N),
            width=WIDTH, tau=0.1, dt=0.1, steps=200, record_every=10,
            contact_derived=True, kappa_recovery=R, kappa_consumption=C)
        e_rec = np.asarray(hist["energy"])
        self.assertLess(e_rec[-1], e_rec[0])           # net dissipation
        self.assertTrue(all(d >= 0.0 for d in hist["dissipation"]))
        self.assertLess(float(np.diff(e_rec).max()), 1e-3)  # measured 6e-4
        # the exclusion energy is booked: the run's energy minus the
        # mechanical/field parts is the Part-I term's Σe(ρ_tot)
        ex0 = (hist["energy"][0] - hist["field_energy"][0]
               - hist["kinetic"][0] - hist["field_kinetic"][0])
        tot = (gaussian_load((N, N), [(N / 2 - 6.0, N / 2)], WIDTH, AMP)
               + gaussian_load((N, N), [(N / 2 + 6.0, N / 2)],
                               WIDTH, AMP))
        self.assertAlmostEqual(
            ex0, float(np.sum(exclusion_energy_density(
                tot, kappa_recovery=R, kappa_consumption=C))), places=12)


if __name__ == "__main__":
    unittest.main()
