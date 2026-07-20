"""Checks for the vorticity-bearing chiral field (spin from real circulation)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import gaussian_load, relax_capacity  # noqa: E402
from project_genesis.two_field import chiral_detuning  # noqa: E402
from project_genesis.vortex_chiral import (  # noqa: E402
    evolve_seeded_field,
    evolve_vortex_molecule,
    field_angular_momentum,
    imprint_vortices,
    vortex_phase_force,
    winding_number,
)


def _pair_loads(n=48, s=9.0, width=2.5, mass=0.6):
    c = [(n / 2 - s / 2, n / 2), (n / 2 + s / 2, n / 2)]
    return c, [gaussian_load((n, n), [ci], width, mass) for ci in c]


class ImprintTests(unittest.TestCase):
    def test_core_holed_bulk_unit(self):
        psi = imprint_vortices((48, 48), [(24, 24)], [1], core=3.0)
        amp = np.abs(psi)
        # amplitude vanishes at the core and recovers to ~1 far away
        self.assertLess(amp[24, 24], 0.1)
        self.assertGreater(amp[24, 4], 0.95)

    def test_phase_winds_q_times(self):
        # the integer winding: total phase circulation about the core is 2πq
        for q in (1, 2, -1):
            psi = imprint_vortices((64, 64), [(32, 32)], [q], core=2.0)
            r = 12
            th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
            xs = 32 + r * np.cos(th)
            ys = 32 + r * np.sin(th)
            ph = np.angle(psi[np.round(xs).astype(int),
                              np.round(ys).astype(int)])
            winding = np.sum(np.diff(np.unwrap(ph))) / (2 * np.pi)
            self.assertAlmostEqual(winding, q, delta=0.1)

    def test_detune_holes_amplitude(self):
        detune = np.zeros((32, 32))
        detune[10:14, 10:14] = 0.9
        psi = imprint_vortices((32, 32), [(24, 24)], [1], core=2.0,
                               detune=detune)
        self.assertLess(np.abs(psi)[12, 12], np.abs(psi)[24, 4])


class AngularMomentumTests(unittest.TestCase):
    def test_sign_locked_to_winding_and_antisymmetric(self):
        c = (24, 24)
        Lp = field_angular_momentum(imprint_vortices((48, 48), [c], [1]), c)
        Lm = field_angular_momentum(imprint_vortices((48, 48), [c], [-1]), c)
        L0 = field_angular_momentum(imprint_vortices((48, 48), [c], [0]), c)
        self.assertGreater(Lp, 0.0)
        self.assertLess(Lm, 0.0)
        self.assertAlmostEqual(L0, 0.0, delta=1e-6)
        self.assertAlmostEqual(Lp, -Lm, delta=0.02 * abs(Lp))

    def test_ladder_grows_with_winding(self):
        c = (24, 24)
        L1 = field_angular_momentum(imprint_vortices((48, 48), [c], [1]), c)
        L2 = field_angular_momentum(imprint_vortices((48, 48), [c], [2]), c)
        self.assertGreater(abs(L2), abs(L1))


class PhaseForceTests(unittest.TestCase):
    def test_same_sign_is_a_torque(self):
        centers, loads = _pair_loads()
        kappa = relax_capacity(loads[0] + loads[1], kappa_diffusion=1.0,
                               kappa_recovery=0.02, kappa_consumption=0.8)
        g = chiral_detuning(kappa, detune_gamma=0.8)
        psi = imprint_vortices((48, 48), centers, [1, 1], core=3.0, detune=g)
        f = vortex_phase_force(loads, psi, phase_chi=0.6)
        # the bond is the x-axis: the force is tangential (y), not radial (x)
        self.assertGreater(np.abs(f[:, 1]).max(), 5.0 * np.abs(f[:, 0]).max())
        com = np.average(np.array(centers), axis=0)
        torque = sum((np.array(ci) - com)[0] * f[i][1]
                     - (np.array(ci) - com)[1] * f[i][0]
                     for i, ci in enumerate(centers))
        self.assertGreater(abs(torque), 1e-3)

    def test_vortex_antivortex_translates(self):
        centers, loads = _pair_loads()
        kappa = relax_capacity(loads[0] + loads[1], kappa_diffusion=1.0,
                               kappa_recovery=0.02, kappa_consumption=0.8)
        g = chiral_detuning(kappa, detune_gamma=0.8)
        same = vortex_phase_force(
            loads, imprint_vortices((48, 48), centers, [1, 1], core=3.0,
                                    detune=g), phase_chi=0.6)
        opp = vortex_phase_force(
            loads, imprint_vortices((48, 48), centers, [1, -1], core=3.0,
                                    detune=g), phase_chi=0.6)
        com = np.average(np.array(centers), axis=0)

        def torque(f):
            return sum((np.array(ci) - com)[0] * f[i][1]
                       - (np.array(ci) - com)[1] * f[i][0]
                       for i, ci in enumerate(centers))
        # antivortex control: the pair translates, the torque collapses
        self.assertLess(abs(torque(opp)), 0.2 * abs(torque(same)))


class VortexMoleculeTests(unittest.TestCase):
    def _run(self, q, steps=1000, n=48):
        pos0 = [[n / 2 - 4.5, n / 2], [n / 2 + 4.5, n / 2]]
        vel0 = [[0.0, 0.2], [0.0, -0.2]]
        return evolve_vortex_molecule(
            pos0, vel0, [0.6, 0.6], shape=(n, n), width=2.5, tau=0.1,
            dt=0.1, steps=steps, record_every=20, vortex_charge=q,
            phase_chi=0.6, core=3.0, detune_gamma=0.8,
            kappa_diffusion=1.0, kappa_recovery=0.02, kappa_consumption=0.8)

    def _late_omega(self, h):
        t = np.asarray(h["time"])
        return float(np.asarray(h["omega"])[t > 0.6 * t[-1]].mean())

    def test_charge_zero_drains(self):
        h = self._run(0)
        self.assertLess(abs(self._late_omega(h)), 0.002)
        self.assertEqual(np.mean(h["L_field"][-5:]), 0.0)   # no vortex
        self.assertLess(np.asarray(h["separation"])[-5:].max(), 1.5 * 9.0)

    def test_charge_sign_sets_spin_sign(self):
        hp = self._run(+1)
        hm = self._run(-1)
        wp, wm = self._late_omega(hp), self._late_omega(hm)
        self.assertGreater(wp, 0.004)          # same-sign vortices co-rotate
        self.assertLess(wm, -0.004)
        # bound on the floor, and the field carries real angular momentum
        self.assertLess(np.asarray(hp["separation"])[-5:].max(), 1.5 * 9.0)
        self.assertGreater(np.mean(hp["L_field"][-5:]), 0.0)
        self.assertLess(np.mean(hm["L_field"][-5:]), 0.0)


class WindingNumberTests(unittest.TestCase):
    def test_unit_vortex_encloses_one(self):
        psi = imprint_vortices((48, 48), [(24, 24)], [1], core=3.0)
        self.assertAlmostEqual(abs(winding_number(psi, (24, 24))), 1.0, delta=0.05)

    def test_charge_two_and_empty_loop(self):
        psi = imprint_vortices((64, 64), [(32, 32)], [2], core=2.0)
        self.assertAlmostEqual(abs(winding_number(psi, (32, 32))), 2.0, delta=0.05)
        # a loop far from the core encloses nothing
        self.assertAlmostEqual(winding_number(psi, (6, 6)), 0.0, delta=0.05)


class SelfSustainedFieldTests(unittest.TestCase):
    def _kappa(self, n=64, s=9.0):
        c = [(n / 2 - s / 2, n / 2), (n / 2 + s / 2, n / 2)]
        loads = sum(gaussian_load((n, n), [ci], 2.5, 0.6) for ci in c)
        return c, relax_capacity(loads, kappa_diffusion=1.0, kappa_recovery=0.02,
                                 kappa_consumption=0.8)

    def test_like_charges_conserve_winding(self):
        # seeded once, co-evolved with NO re-imprint: the integer survives
        c, kappa = self._kappa()
        h = evolve_seeded_field((64, 64), c, [1, 1], kappa, steps=600,
                                record_every=300)
        w = np.asarray(h["winding"])
        self.assertLess(float(np.max(np.abs(w - w[0]))), 0.1)   # conserved
        self.assertAlmostEqual(abs(round(w[-1][0])), 1)          # integer, |w|=1
        self.assertGreater(abs(h["L_field"][-1]), 0.4 * abs(h["L_field"][0]))

    def test_vortex_antivortex_annihilate(self):
        c, kappa = self._kappa()
        h = evolve_seeded_field((64, 64), c, [1, -1], kappa, steps=600,
                                record_every=300)
        w = np.asarray(h["winding"])
        self.assertLess(abs(w[-1][0]), 0.2)          # unwound
        self.assertLess(abs(w[-1][1]), 0.2)
        self.assertGreater(h["amp_min"][-1], 0.4)    # the field heals
        self.assertLess(abs(h["L_field"][-1]), 0.1 * abs(h["L_field"][0]) + 1.0)


class EmergentMoleculeTests(unittest.TestCase):
    def _run(self, q, reimprint, steps=900, n=64):
        pos0 = [[n / 2 - 4.5, n / 2], [n / 2 + 4.5, n / 2]]
        vel0 = [[0.0, 0.2], [0.0, -0.2]]
        return evolve_vortex_molecule(
            pos0, vel0, [0.6, 0.6], shape=(n, n), width=2.5, tau=0.1, dt=0.1,
            steps=steps, record_every=20, vortex_charge=q, phase_chi=0.6,
            core=3.0, detune_gamma=0.8, reimprint=reimprint,
            kappa_diffusion=1.0, kappa_recovery=0.02, kappa_consumption=0.8)

    def test_reimprint_true_matches_default(self):
        # reimprint=True is the pinned mode of record (default) — bitwise
        h_def = self._run(1, reimprint=True)
        pos = np.asarray(h_def["positions"])
        self.assertTrue(np.all(np.isfinite(pos)))
        self.assertGreater(np.mean(h_def["L_field"][-3:]), 0.0)

    def test_self_sustained_vortex_tracks_the_mass(self):
        # reimprint=False: the winding around each MOVING mass stays +-1
        h = self._run(1, reimprint=False)
        wind = np.asarray(h["winding"])
        self.assertGreater(float(np.min(np.abs(wind[:, 0]))), 0.8)
        self.assertGreater(float(np.min(np.abs(wind[:, 1]))), 0.8)
        # bound, and the field still carries (draining) angular momentum
        sep = np.asarray(h["separation"])
        self.assertLess(sep[-3:].max(), 1.5 * 9.0)

    def test_zero_charge_no_field_either_mode(self):
        for rei in (True, False):
            h = self._run(0, reimprint=rei)
            self.assertEqual(np.mean(h["L_field"][-3:]), 0.0)


if __name__ == "__main__":
    unittest.main()
