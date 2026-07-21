"""Checks for the 3-D vortex line: spin as a quantised axial vector."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import gaussian_load, relax_capacity  # noqa: E402
from project_genesis.two_field import chiral_detuning, step_chiral_detuned  # noqa: E402
from project_genesis.vortex_chiral_3d import (  # noqa: E402
    evolve_line_molecule,
    evolve_seeded_line,
    line_angular_momentum,
    line_phase_force,
    line_winding,
    vortex_line,
)


class AxialVectorTests(unittest.TestCase):
    def test_L_aligns_with_the_line_any_orientation(self):
        n = (40, 40, 40)
        c = (20, 20, 20)
        for nrm in [(0, 0, 1), (1, 0, 0), (1, 1, 1), (2, -1, 1)]:
            L = line_angular_momentum(vortex_line(n, c, nrm, 1), c, radius=16)
            nh = np.asarray(nrm, float)
            nh = nh / np.linalg.norm(nh)
            self.assertGreater(abs(L @ nh) / np.linalg.norm(L), 0.99)

    def test_L_is_sign_locked_and_quantised(self):
        n = (40, 40, 40)
        c = (20, 20, 20)

        def Lz(q):
            return line_angular_momentum(vortex_line(n, c, (0, 0, 1), q), c, 16)[2]
        self.assertGreater(Lz(1), 0.0)
        self.assertLess(Lz(-1), 0.0)
        self.assertAlmostEqual(Lz(0), 0.0, delta=1.0)
        self.assertAlmostEqual(Lz(1), -Lz(-1), delta=0.02 * abs(Lz(1)))
        self.assertGreater(abs(Lz(2)), abs(Lz(1)))          # a ladder in |q|

    def test_magnitude_is_direction_independent(self):
        n = (40, 40, 40)
        c = (20, 20, 20)
        mags = [np.linalg.norm(line_angular_momentum(
            vortex_line(n, c, nrm, 1), c, 16))
            for nrm in [(0, 0, 1), (1, 1, 1), (2, -1, 1)]]
        self.assertLess((max(mags) - min(mags)) / max(mags), 0.02)


class LineWindingTests(unittest.TestCase):
    def test_reads_the_integer_charge(self):
        n = (40, 40, 40)
        c = (20, 20, 20)
        self.assertAlmostEqual(abs(line_winding(vortex_line(n, c, (0, 0, 1), 1),
                                                c, 2)), 1.0, delta=0.05)
        self.assertAlmostEqual(abs(line_winding(vortex_line(n, c, (0, 0, 1), 2),
                                                c, 2)), 2.0, delta=0.05)
        self.assertAlmostEqual(line_winding(vortex_line(n, c, (0, 0, 1), 0),
                                            c, 2), 0.0, delta=0.05)


class SelfSustainedLineTests(unittest.TestCase):
    def _kappa(self, n, c):
        return relax_capacity(gaussian_load(n, [c], 2.5, 0.6),
                              kappa_diffusion=1.0, kappa_recovery=0.02,
                              kappa_consumption=0.8)

    def test_winding_and_axis_conserved_no_reimprint(self):
        n = (36, 36, 36)
        c = (18, 18, 18)
        h = evolve_seeded_line(n, c, (0, 0, 1), 1, self._kappa(n, c),
                               chiral=0.0, steps=400, record_every=200,
                               noise=0.15, seed=1)
        w = np.asarray(h["winding"])
        self.assertLess(float(np.max(np.abs(w - w[0]))), 0.1)   # conserved
        self.assertAlmostEqual(abs(round(w[-1])), 1)            # integer, |w|=1
        self.assertGreater(min(h["align"]), 0.95)              # axis stable
        self.assertLess(h["amp_min"][-1], 0.2)                 # core survives

    def test_line_antiline_annihilate(self):
        n = (36, 36, 36)
        c0, c1 = (18, 13, 18), (18, 23, 18)
        kappa = relax_capacity(
            gaussian_load(n, [c0], 2.5, 0.6) + gaussian_load(n, [c1], 2.5, 0.6),
            kappa_diffusion=1.0, kappa_recovery=0.02, kappa_consumption=0.8)
        g = chiral_detuning(kappa, detune_gamma=0.8)
        psi = (vortex_line(n, c0, (0, 0, 1), 1)
               * vortex_line(n, c1, (0, 0, 1), -1)
               * np.sqrt(np.clip(1.0 - g, 0.0, 1.0)))
        for _ in range(400):
            psi = step_chiral_detuned(psi, g, chiral=0.0, dt=0.1)
        self.assertLess(abs(line_winding(psi, c0, 2)), 0.2)     # unwound
        self.assertLess(abs(line_winding(psi, c1, 2)), 0.2)
        self.assertGreater(np.abs(psi).min(), 0.4)             # healed


class MoleculeTests(unittest.TestCase):
    """The 3-D molecule carries a real vector spin-torque but is overdamped."""

    def _torque(self, axis, sepdir, charges, n=32, s=7.0):
        mid = n / 2
        d = np.zeros(3); d[sepdir] = s / 2
        c0 = np.array([mid, mid, mid]) - d
        c1 = np.array([mid, mid, mid]) + d
        loads = [gaussian_load((n, n, n), [tuple(c0)], 2.5, 0.6),
                 gaussian_load((n, n, n), [tuple(c1)], 2.5, 0.6)]
        kappa = relax_capacity(loads[0] + loads[1], kappa_diffusion=1.0,
                               kappa_recovery=0.02, kappa_consumption=0.8)
        g = chiral_detuning(kappa, detune_gamma=0.8)
        psi = (vortex_line((n, n, n), tuple(c0), axis, charges[0])
               * vortex_line((n, n, n), tuple(c1), axis, charges[1])
               * np.sqrt(np.clip(1.0 - g, 0.0, 1.0)))
        F = line_phase_force(loads, psi, phase_chi=0.6)
        com = 0.5 * (c0 + c1)
        return np.cross(c0 - com, F[0]) + np.cross(c1 - com, F[1])

    def test_torque_axis_follows_the_line_and_sign_locks(self):
        tz = self._torque((0, 0, 1), 0, (1, 1))     # lines ∥ z, sep along x
        tx = self._torque((1, 0, 0), 1, (1, 1))     # lines ∥ x, sep along y
        # the torque points along the line direction, in each case
        self.assertGreater(abs(tz[2]) / np.linalg.norm(tz), 0.9)
        self.assertGreater(abs(tx[0]) / np.linalg.norm(tx), 0.9)
        # sign-locked to the winding
        tz_neg = self._torque((0, 0, 1), 0, (-1, -1))
        self.assertLess(np.sign(tz[2]) * np.sign(tz_neg[2]), 0)

    def test_antivortex_torque_is_null(self):
        tz = self._torque((0, 0, 1), 0, (1, 1))
        ta = self._torque((0, 0, 1), 0, (1, -1))
        self.assertLess(np.linalg.norm(ta), 0.2 * np.linalg.norm(tz))

    def test_bound_pair_is_overdamped_no_spin(self):
        # the honest negative: driven, floor-bound, but it will not turn
        n = 32; mid = n / 2; s = 7.0
        pos = [[mid - s / 2, mid, mid], [mid + s / 2, mid, mid]]
        vel = [[0, 0.2, 0], [0, -0.2, 0]]
        h = evolve_line_molecule(pos, vel, [0.6, 0.6], shape=(n, n, n),
            axis=(0, 0, 1), width=2.5, tau=0.1, dt=0.1, steps=800,
            record_every=20, vortex_charge=1, phase_chi=0.6, core=3.0,
            detune_gamma=0.8, reimprint=True, chiral_lambda=0.2,
            kappa_diffusion=1.0, kappa_recovery=0.02, kappa_consumption=0.8)
        t = np.asarray(h["time"])
        omega_z = np.asarray(h["omega_vec"])[t > 0.6 * t[-1], 2].mean()
        self.assertLess(abs(omega_z), 0.004)                  # overdamped: no spin
        sep = np.asarray(h["separation"])
        self.assertLess(sep[-3:].max(), 1.6 * s)              # but stays bound
        self.assertGreater(np.mean([abs(w[0]) for w in h["winding"][-3:]]), 0.8)  # holds spin


if __name__ == "__main__":
    unittest.main()
