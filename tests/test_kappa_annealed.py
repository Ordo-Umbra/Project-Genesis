"""Tests for the dynamical capacity field κ in the annealed (ψ, U) model."""

from __future__ import annotations

import unittest

import numpy as np

from project_genesis.gauge import (
    covariant_coherence,
    random_psi,
    random_unitary,
)
from project_genesis.gauge_mc import heatbath_sweep
from project_genesis.annealed_matter import (
    joint_sweep,
    kappa_step,
    matter_metropolis_sweep,
    sector_amplitudes,
)


class TestKappaGatingParity(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.psi = random_psi(rng, 3, (8, 8))
        self.links = random_unitary(rng, 3, (2, 8, 8), special=True)
        self.kappa = np.random.default_rng(1).uniform(0.2, 1.0, (8, 8))

    def test_matter_sweep_python_numba_agree_with_kappa(self):
        kw = dict(temperature=0.2, matter_coupling=3.0, quartic_u=1.0,
                  step_scale=0.3, kappa=self.kappa)
        p_py, a_py = matter_metropolis_sweep(
            self.psi, self.links, np.random.default_rng(5),
            use_numba=False, **kw)
        p_nb, a_nb = matter_metropolis_sweep(
            self.psi, self.links, np.random.default_rng(5),
            use_numba=True, **kw)
        self.assertLess(float(np.max(np.abs(p_py - p_nb))), 1e-12)
        self.assertEqual(a_py, a_nb)

    def test_heatbath_python_numba_agree_with_kappa(self):
        l_py, _ = heatbath_sweep(self.links, 2.0, np.random.default_rng(7),
                                 matter=self.psi, matter_coupling=3.0,
                                 matter_kappa=self.kappa, use_numba=False)
        l_nb, _ = heatbath_sweep(self.links, 2.0, np.random.default_rng(7),
                                 matter=self.psi, matter_coupling=3.0,
                                 matter_kappa=self.kappa, use_numba=True)
        self.assertLess(float(np.max(np.abs(l_py - l_nb))), 1e-10)

    def test_no_kappa_equals_unit_kappa(self):
        ones = np.ones((8, 8))
        p_none, _ = matter_metropolis_sweep(
            self.psi, self.links, np.random.default_rng(9),
            temperature=0.2, matter_coupling=3.0, kappa=None)
        p_ones, _ = matter_metropolis_sweep(
            self.psi, self.links, np.random.default_rng(9),
            temperature=0.2, matter_coupling=3.0, kappa=ones)
        self.assertTrue(np.array_equal(p_none, p_ones))

    def test_rejects_wrong_kappa_shape(self):
        with self.assertRaises(ValueError):
            matter_metropolis_sweep(
                self.psi, self.links, np.random.default_rng(2),
                temperature=0.2, kappa=np.ones((4, 4)))


class TestKappaStep(unittest.TestCase):

    def test_flat_field_recovers_toward_baseline(self):
        psi = np.zeros((6, 6, 3), dtype=complex)
        psi[..., 0] = 1.0  # zero gradient load
        kappa = 0.5 * np.ones((6, 6))
        for _ in range(200):
            kappa = kappa_step(kappa, psi, recovery=0.2, consumption=5.0)
        self.assertGreater(float(kappa.min()), 0.95)

    def test_load_consumes_capacity(self):
        rng = np.random.default_rng(3)
        psi = random_psi(rng, 3, (6, 6))  # noisy field, high load
        kappa = np.ones((6, 6))
        kappa2 = kappa_step(kappa, psi, recovery=0.0, consumption=5.0,
                            diffusion=0.0)
        self.assertLess(float(kappa2.mean()), 1.0)
        self.assertGreaterEqual(float(kappa2.min()), 0.0)

    def test_stays_in_bounds(self):
        rng = np.random.default_rng(4)
        psi = random_psi(rng, 3, (6, 6))
        kappa = np.ones((6, 6))
        for _ in range(100):
            kappa = kappa_step(kappa, psi, consumption=50.0)
        self.assertGreaterEqual(float(kappa.min()), 0.0)
        self.assertLessEqual(float(kappa.max()), 1.0)


class TestJointSweepWithKappa(unittest.TestCase):

    def test_returns_four_tuple_and_evolves(self):
        rng = np.random.default_rng(11)
        psi = random_psi(rng, 3, (8, 8))
        links = random_unitary(rng, 3, (2, 8, 8), special=True)
        kappa = np.ones((8, 8))
        psi, links, kappa2, acc = joint_sweep(
            psi, links, rng, temperature=0.1, beta_g=3.0,
            matter_coupling=4.0, kappa=kappa)
        self.assertEqual(kappa2.shape, (8, 8))
        self.assertLess(float(kappa2.mean()), 1.0)
        self.assertTrue(0.0 <= acc <= 1.0)

    def test_scarcity_suppresses_coherence(self):
        """κ ≈ 0 gates the integration channel off: the thermodynamic
        version of 'capacity governs what structures can emerge'."""
        def coherence_with(kappa_value):
            r = np.random.default_rng(21)
            psi = random_psi(r, 3, (8, 8))
            links = random_unitary(r, 3, (2, 8, 8), special=True)
            kappa = kappa_value * np.ones((8, 8))
            for _ in range(80):
                psi, links, kappa, _ = joint_sweep(
                    psi, links, r, temperature=0.1, beta_g=3.0,
                    matter_coupling=4.0, kappa=kappa,
                    kappa_params=dict(consumption=0.0, recovery=0.0))
            return covariant_coherence(psi, links) / 128

        abundant = coherence_with(1.0)
        scarce = coherence_with(0.05)
        self.assertGreater(abundant, scarce + 0.3)


if __name__ == "__main__":
    unittest.main()
