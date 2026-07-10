"""Checks for the FLRW-like background: Friedmann rates, Hubble drag, growth."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_dynamics import (  # noqa: E402
    evolve_cosmological,
    fof_groups,
    friedmann_rates,
)

FK = dict(kappa_diffusion=1.0, kappa_recovery=0.02, kappa_consumption=0.8,
          kappa_baseline=1.0, max_iters=1500)


class TestFriedmann(unittest.TestCase):

    def test_hubble_rate_at_a1(self):
        # H(a=1) = H0 * sqrt(Om + OL)
        H, _ = friedmann_rates(1.0, 0.05, 1.0, 0.0)
        self.assertAlmostEqual(H, 0.05, places=9)
        H2, _ = friedmann_rates(1.0, 0.1, 0.3, 0.7)
        self.assertAlmostEqual(H2, 0.1, places=9)

    def test_matter_decelerates_lambda_accelerates(self):
        # p=3 matter → ä/a < 0; pure Λ → ä/a > 0
        _, acc_m = friedmann_rates(1.0, 0.05, 1.0, 0.0, p=3.0)
        _, acc_l = friedmann_rates(1.0, 0.05, 0.0, 1.0, p=3.0)
        self.assertLess(acc_m, 0.0)
        self.assertGreater(acc_l, 0.0)

    def test_matter_dilutes_with_expansion(self):
        # H falls as the matter universe expands (density dilutes)
        H1, _ = friedmann_rates(1.0, 0.05, 1.0, 0.0)
        H2, _ = friedmann_rates(4.0, 0.05, 1.0, 0.0)
        self.assertLess(H2, H1)


class TestExpansionDynamics(unittest.TestCase):

    def test_scale_factor_grows(self):
        L = 48
        h = evolve_cosmological([[L / 2, L / 2]], [1.0], shape=(L, L),
                                hubble0=0.05, omega_m=1.0, omega_lambda=0.0,
                                dt=0.5, steps=20, **FK)
        self.assertGreater(h["a_final"], 1.0)
        self.assertEqual(len(h["a"]), 20)

    def test_hubble_drag_redshifts_peculiar_velocity(self):
        # a peculiar velocity decays as 1/a: a·|v_pec| is ~constant
        L = 64
        c = np.array([L / 2.0, L / 2.0])
        h = evolve_cosmological([[L / 2.0, L / 2.0]], [1.5], shape=(L, L),
                                hubble0=0.05, omega_m=1.0, omega_lambda=0.0,
                                dt=0.5, steps=80, center=c,
                                peculiar=[[0.5, 0.0]], **FK)
        prod = []
        for i in range(1, len(h["a"])):
            a = h["a"][i]
            r = np.array(h["positions"][i][0])
            v = np.array(h["velocities"][i][0])
            H, _ = friedmann_rates(a, 0.05, 1.0, 0.0)
            prod.append(a * np.linalg.norm(v - H * (r - c)))
        prod = np.array(prod)
        self.assertLess(prod.std() / prod.mean(), 0.05)   # constant to <5%

    def test_dark_energy_suppresses_structure(self):
        # same cloud: matter forms a bigger bound halo than a Λ universe
        L = 72
        c = np.array([L / 2.0, L / 2.0])

        def largest(ol):
            rng = np.random.default_rng(1)
            pos = [c + rng.uniform(-10, 10, 2) for _ in range(10)]
            h = evolve_cosmological(pos, [1.0] * 10, shape=(L, L), hubble0=0.05,
                                    omega_m=1.0 - ol, omega_lambda=ol, dt=0.5,
                                    steps=70, center=c, **FK)
            return fof_groups([p for p in h["final_positions"]], 6.0)[0]

        self.assertGreaterEqual(largest(0.0), largest(1.0))


if __name__ == "__main__":
    unittest.main()
