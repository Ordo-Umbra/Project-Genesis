"""Checks for the finite-speed (telegrapher) capacity dynamics."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_waves import (  # noqa: E402
    dispersion_omega,
    evolve_inertial_retarded,
    front_radius,
    steady_kappa,
    step_capacity_parabolic,
    step_capacity_wave,
    wave_mass2,
    wave_speed,
)


class WaveStepTests(unittest.TestCase):
    def test_steady_state_is_fixed_point(self):
        rho, r = 0.1, 0.05
        kbar = steady_kappa(r, 2.0, rho)
        kappa = np.full((16, 16), kbar)
        kdot = np.zeros_like(kappa)
        for _ in range(50):
            kappa, kdot = step_capacity_wave(kappa, kdot, rho, tau=4.0,
                                             kappa_recovery=r, dt=0.1)
        self.assertTrue(np.allclose(kappa, kbar, atol=1e-12))
        self.assertTrue(np.allclose(kdot, 0.0, atol=1e-12))

    def test_parabolic_step_matches_relaxation_direction(self):
        kappa = np.full((16, 16), 0.5)
        out = step_capacity_parabolic(kappa, 0.0, kappa_recovery=0.2, dt=0.1)
        self.assertTrue(np.all(out > kappa))   # recovery pulls to baseline

    def test_cfl_guard(self):
        kappa = np.ones((8, 8))
        with self.assertRaises(ValueError):
            step_capacity_wave(kappa, np.zeros_like(kappa), 0.0,
                               tau=0.01, dt=0.1)

    def test_front_respects_causality(self):
        n, tau, dt = 96, 4.0, 0.1
        c = wave_speed(1.0, tau)
        kappa = np.full((n, n), 1.0)
        g = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        d2 = sum((x - n / 2.0) ** 2 for x in g)
        kappa += 1e-3 * np.exp(-d2 / 8.0)
        kdot = np.zeros_like(kappa)
        r0 = 2.0 * np.sqrt(2.0 * np.log(1e-3 / 1e-12))
        for _ in range(400):
            kappa, kdot = step_capacity_wave(kappa, kdot, 0.0, tau=tau,
                                             kappa_recovery=0.05, dt=dt)
        t = 400 * dt
        rf = front_radius(kappa - steady_kappa(0.05, 2.0, 0.0),
                          (n / 2.0, n / 2.0), 1e-12)
        self.assertLessEqual(rf, c * t + r0 + 2.0)   # nothing outruns it

    def test_dispersion_matches_theory(self):
        n, tau, r, rho = 128, 32.0, 0.05, 0.1
        k = 2.0 * np.pi / 16.0
        kbar = steady_kappa(r, 2.0, rho)
        x = np.arange(n)
        kappa = kbar + 1e-4 * np.cos(k * x)[:, None] * np.ones((1, 4))
        kdot = np.zeros_like(kappa)
        proj = np.cos(k * x)
        a = []
        for _ in range(2500):
            kappa, kdot = step_capacity_wave(kappa, kdot, rho, tau=tau,
                                             kappa_recovery=r, dt=0.1)
            a.append(2.0 * float(np.mean((kappa[:, 0] - kbar) * proj)))
        a = np.asarray(a)
        t = 0.1 * (1 + np.arange(len(a)))
        crossings = [t[i - 1] + 0.1 * a[i - 1] / (a[i - 1] - a[i])
                     for i in range(1, len(a)) if a[i - 1] * a[i] < 0]
        omega = np.pi / float(np.mean(np.diff(crossings)))
        theory = dispersion_omega(k, tau=tau, kappa_recovery=r, rho=rho)
        self.assertLess(abs(omega - theory) / theory, 0.05)

    def test_mass_grows_with_density_and_speed_with_update_rate(self):
        self.assertGreater(wave_mass2(0.05, 2.0, 0.2),
                           wave_mass2(0.05, 2.0, 0.0))
        self.assertGreater(wave_speed(1.0, 4.0), wave_speed(1.0, 64.0))
        self.assertTrue(np.isnan(dispersion_omega(0.01, tau=0.5)))


class RetardedGravityTests(unittest.TestCase):
    def test_retarded_lone_mass_is_quiet(self):
        hist = evolve_inertial_retarded(
            [[24.0, 24.0]], [[0.0, 0.0]], [1.2],
            shape=(48, 48), tau=1.0, dt=0.1, steps=200, record_every=20,
            kappa_recovery=0.02, kappa_consumption=0.8)
        e = np.asarray(hist["energy"])
        # a lone mass at rest in its own relaxed well: no motion, no
        # dissipation — energy conserved to fine tolerance
        self.assertLess(abs(e[-1] - e[0]) / abs(e[0]), 1e-6)
        # residual kappa-dot from the finite relax tolerance seeds ~1e-11
        self.assertLess(max(hist["dissipation"]), 1e-9)

    def test_contact_term_repels_overlapping_masses(self):
        hist = evolve_inertial_retarded(
            [[22.5, 24.0], [25.5, 24.0]], [[0.0, 0.0], [0.0, 0.0]],
            [1.2, 1.2], shape=(48, 48), tau=1.0, dt=0.1, steps=150,
            record_every=15, contact_b=0.5, kappa_recovery=0.02,
            kappa_consumption=0.8)
        sep = np.asarray(hist["separation"])
        # deep-overlap pair (sep = 3 ~ footprint): exclusion pushes apart
        self.assertGreater(sep[-1], sep[0] + 0.5)

    def test_contact_lyapunov_still_holds(self):
        hist = evolve_inertial_retarded(
            [[18.0, 24.0], [30.0, 24.0]], [[0.0, 0.6], [0.0, -0.6]],
            [1.2, 1.2], shape=(48, 48), tau=1.0, dt=0.1, steps=400,
            record_every=20, contact_b=0.3, kappa_recovery=0.02,
            kappa_consumption=0.8)
        e = np.asarray(hist["energy"])
        self.assertGreater(np.mean(np.diff(e) <= 1e-9), 0.9)

    def test_retarded_energy_is_lyapunov_for_moving_binary(self):
        hist = evolve_inertial_retarded(
            [[18.0, 24.0], [30.0, 24.0]], [[0.0, 0.8], [0.0, -0.8]],
            [1.2, 1.2], shape=(48, 48), tau=1.0, dt=0.1, steps=400,
            record_every=20, kappa_recovery=0.02, kappa_consumption=0.8)
        e = np.asarray(hist["energy"])
        self.assertLess(e[-1], e[0])                       # net dissipation
        self.assertGreater(np.mean(np.diff(e) <= 1e-9), 0.9)  # Lyapunov
        self.assertTrue(all(d >= 0.0 for d in hist["dissipation"]))


if __name__ == "__main__":
    unittest.main()
