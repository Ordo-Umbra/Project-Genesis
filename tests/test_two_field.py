"""Checks for the derived two-field (chiral ψ × κ) coupling."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import gaussian_load, relax_capacity  # noqa: E402
from project_genesis.capacity_waves import evolve_inertial_retarded  # noqa: E402
from project_genesis.two_field import (  # noqa: E402
    bulk_precession,
    chiral_detuning,
    evolve_two_field,
    phase_current,
    static_phase_force,
    step_chiral_detuned,
)


def _pair_loads(n=48, s=9.0, width=2.5, mass=0.6):
    return [gaussian_load((n, n), [(n / 2 - s / 2, n / 2)], width, mass),
            gaussian_load((n, n), [(n / 2 + s / 2, n / 2)], width, mass)]


class DetuningTests(unittest.TestCase):
    def test_wells_hole_the_field(self):
        loads = _pair_loads()
        kappa = relax_capacity(loads[0] + loads[1], kappa_diffusion=1.0,
                               kappa_recovery=0.02, kappa_consumption=0.8)
        g = chiral_detuning(kappa, detune_gamma=0.8)
        psi = np.ones((48, 48), dtype=complex)
        for _ in range(600):
            psi = step_chiral_detuned(psi, g, chiral=0.2, dt=0.1)
        amp = np.abs(psi)
        self.assertLess(amp.min(), 0.75)   # holed at the wells
        self.assertGreater(amp.max(), 0.9)  # ordered bulk survives

    def test_detuning_bounds_and_zero(self):
        kappa = np.full((16, 16), 0.5)
        g = chiral_detuning(kappa, detune_gamma=0.8)
        self.assertTrue(np.all(g >= 0.0) and np.all(g <= 0.95))
        self.assertTrue(np.allclose(
            chiral_detuning(np.ones((8, 8)), detune_gamma=0.8), 0.0))


class CurrentForceTests(unittest.TestCase):
    def test_phase_force_is_radial_and_odd_in_lambda(self):
        loads = _pair_loads()
        kappa = relax_capacity(loads[0] + loads[1], kappa_diffusion=1.0,
                               kappa_recovery=0.02, kappa_consumption=0.8)
        fp, _ = static_phase_force(loads, kappa, chiral_lambda=0.2,
                                   steps=1000)
        fm, _ = static_phase_force(loads, kappa, chiral_lambda=-0.2,
                                   steps=1000)
        # radial (bond-axis) component: real, and exactly odd in λ
        self.assertGreater(abs(fp[0][0]), 1e-3)
        self.assertAlmostEqual(fp[0][0], -fm[0][0], delta=0.1 * abs(fp[0][0]))
        # tangential component: none (no torque from the static sector)
        self.assertLess(abs(fp[0][1]), 0.05 * abs(fp[0][0]))
        # the partner is pushed opposite (Newton's third law, pairwise)
        self.assertAlmostEqual(fp[0][0], -fp[1][0],
                               delta=0.05 * abs(fp[0][0]))

    def test_no_lambda_no_force(self):
        loads = _pair_loads()
        kappa = relax_capacity(loads[0] + loads[1], kappa_diffusion=1.0,
                               kappa_recovery=0.02, kappa_consumption=0.8)
        f0, _ = static_phase_force(loads, kappa, chiral_lambda=0.0,
                                   steps=1000)
        fp, _ = static_phase_force(loads, kappa, chiral_lambda=0.2,
                                   steps=1000)
        self.assertLess(abs(f0[0][0]), 0.05 * abs(fp[0][0]))


class PrecessionTests(unittest.TestCase):
    def test_measured_bulk_rate_is_minus_lambda(self):
        rng = np.random.default_rng(0)
        psi = np.ones((32, 32), dtype=complex) + 0.01 * (
            rng.standard_normal((32, 32))
            + 1j * rng.standard_normal((32, 32)))
        g = np.zeros((32, 32))
        for _ in range(100):
            psi = step_chiral_detuned(psi, g, chiral=0.2, dt=0.1)
        dpsi = step_chiral_detuned(psi, g, chiral=0.2, dt=0.1) - psi
        self.assertAlmostEqual(bulk_precession(psi, dpsi, dt=0.1), -0.2,
                               delta=0.01)

    def test_real_field_does_not_precess(self):
        psi = np.ones((32, 32), dtype=complex)
        g = np.zeros((32, 32))
        dpsi = step_chiral_detuned(psi, g, chiral=0.0, dt=0.1) - psi
        self.assertAlmostEqual(bulk_precession(psi, dpsi, dt=0.1), 0.0,
                               delta=1e-9)

    def test_current_is_zero_for_uniform_precession(self):
        # a uniformly precessing bulk carries no mechanical current — the
        # reason the static chiral force can be radial only
        psi = np.exp(-0.2j * np.arange(32))[None, :] * np.ones((32, 32))
        j = phase_current(psi.astype(complex))  # ramp runs along axis 1
        # pure temporal precession: spatially uniform phase => j = 0
        uniform = np.exp(-0.5j) * np.ones((32, 32), dtype=complex)
        ju = phase_current(uniform)
        self.assertLess(np.abs(ju[0]).max(), 1e-12)
        self.assertGreater(np.abs(j[1]).max(), 0.0)  # phase gradient carries current


class TwoFieldEvolutionTests(unittest.TestCase):
    def _run_kwargs(self, steps):
        return dict(shape=(48, 48), width=2.5, tau=0.1, dt=0.1,
                    steps=steps, record_every=20, contact_full=True,
                    kappa_diffusion=1.0, kappa_recovery=0.02,
                    kappa_consumption=0.8, kappa_baseline=1.0)

    def test_zero_couplings_bitwise_match_retarded(self):
        pos0 = [[48 / 2 - 4.5, 24.0], [48 / 2 + 4.5, 24.0]]
        vel0 = [[0.0, 0.2], [0.0, -0.2]]
        kw = self._run_kwargs(200)
        old = evolve_inertial_retarded(pos0, vel0, [0.6, 0.6], **kw)
        new = evolve_two_field(pos0, vel0, [0.6, 0.6], chiral_lambda=0.0,
                               detune_gamma=0.0, phase_chi=0.0,
                               circulation_coupling=0.0, **kw)
        for a, b in zip(old["positions"], new["positions"]):
            self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))
        self.assertTrue(np.array_equal(old["kappa"], new["kappa"]))

    def test_lambda_zero_is_achiral_automatically(self):
        # no special casing: with λ = 0 the measured Ω_field is zero, the
        # circulation drive vanishes, and the kicked pair drains (Z2)
        pos0 = [[48 / 2 - 4.5, 24.0], [48 / 2 + 4.5, 24.0]]
        vel0 = [[0.0, 0.2], [0.0, -0.2]]
        h = evolve_two_field(pos0, vel0, [0.6, 0.6], chiral_lambda=0.0,
                             detune_gamma=0.8, phase_chi=0.1,
                             circulation_coupling=0.05,
                             **self._run_kwargs(800))
        t = np.asarray(h["time"])
        pos = np.asarray(h["positions"])
        sv = pos[:, 0, :] - pos[:, 1, :]
        phi = np.unwrap(np.arctan2(sv[:, 1], sv[:, 0]))
        omega = np.gradient(phi, t)
        self.assertLess(abs(omega[t > 0.6 * t[-1]].mean()), 0.002)
        self.assertAlmostEqual(np.mean(h["omega_field"][-5:]), 0.0,
                               delta=0.005)

    def test_chiral_pair_spins_with_field_handedness(self):
        pos0 = [[48 / 2 - 4.5, 24.0], [48 / 2 + 4.5, 24.0]]
        vel0 = [[0.0, 0.2], [0.0, -0.2]]
        kw = self._run_kwargs(1200)
        omegas = {}
        for lam in (0.2, -0.2):
            h = evolve_two_field(pos0, vel0, [0.6, 0.6], chiral_lambda=lam,
                                 detune_gamma=0.8, phase_chi=0.1,
                                 circulation_coupling=0.05, **kw)
            t = np.asarray(h["time"])
            pos = np.asarray(h["positions"])
            sv = pos[:, 0, :] - pos[:, 1, :]
            phi = np.unwrap(np.arctan2(sv[:, 1], sv[:, 0]))
            omega = np.gradient(phi, t)
            omegas[lam] = omega[t > 0.6 * t[-1]].mean()
            self.assertLess(np.asarray(h["separation"])[-5:].max(),
                            1.5 * 9.0)
        self.assertLess(omegas[0.2], -0.003)   # Ω = −λ handedness
        self.assertGreater(omegas[-0.2], 0.003)


if __name__ == "__main__":
    unittest.main()
