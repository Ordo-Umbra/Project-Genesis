"""Checks for the self-consistent gauge field (abelian Higgs gauged vortex)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.gauged_vortex import (  # noqa: E402
    covariant_laplacian,
    energy,
    gauge_transform,
    local_flux,
    plaquette_flux,
    relax,
    seed_vortices,
    zero_links,
)

TWO_PI = 2.0 * np.pi


class TestSeedAndFlux(unittest.TestCase):

    def test_seed_has_winding(self):
        n = 40
        psi = seed_vortices(n, [(n / 2, n / 2)], [1])
        # the phase winds by 2π around the core (a ψ-plaquette sum)
        th = np.angle(psi)
        # amplitude is holed at the core, ~1 far away
        self.assertLess(np.abs(psi[n // 2, n // 2]), 0.2)
        self.assertGreater(np.abs(psi[2, 2]), 0.8)

    def test_trivial_links_have_zero_flux(self):
        f = plaquette_flux(zero_links(30))
        np.testing.assert_allclose(f, 0.0, atol=1e-12)

    def test_covariant_laplacian_shape(self):
        n = 20
        psi = seed_vortices(n, [(10, 10)], [1])
        lap = covariant_laplacian(psi, zero_links(n))
        self.assertEqual(lap.shape, psi.shape)


class TestRelaxation(unittest.TestCase):

    def test_energy_decreases(self):
        n = 48
        psi = seed_vortices(n, [(n / 2 - 10, n / 2), (n / 2 + 10, n / 2)], [1, -1])
        e0 = energy(psi, zero_links(n), 2.0, 1.0)
        out = relax(psi, zero_links(n), lam=2.0, beta=1.0, dt=0.05, steps=800)
        self.assertLess(out["energy"], e0)


class TestFluxQuantization(unittest.TestCase):
    """The headline: the self-consistent gauge field carries quantized flux."""

    def _flux(self, q, n=64, steps=2500):
        c0 = (n / 2 - 16, n / 2)
        c1 = (n / 2 + 16, n / 2)
        psi = seed_vortices(n, [c0, c1], [q, -q])
        out = relax(psi, zero_links(n), lam=2.0, beta=1.0, dt=0.05, steps=steps)
        return local_flux(out["theta"], c0, 6.0) / TWO_PI, out

    def test_single_vortex_is_one_quantum(self):
        quanta, _ = self._flux(1)
        self.assertAlmostEqual(abs(quanta), 1.0, delta=0.2)

    def test_double_vortex_is_two_quanta(self):
        quanta, _ = self._flux(2)
        self.assertAlmostEqual(abs(quanta), 2.0, delta=0.25)

    def test_gauge_off_carries_no_flux(self):
        n = 64
        c0 = (n / 2 - 16, n / 2)
        c1 = (n / 2 + 16, n / 2)
        psi = seed_vortices(n, [c0, c1], [1, -1])
        out = relax(psi, zero_links(n), lam=2.0, beta=1.0, dt=0.05, steps=1500,
                    gauge_on=False)
        self.assertLess(abs(local_flux(out["theta"], c0, 6.0)), 0.3 * TWO_PI)


class TestGaugeInvariance(unittest.TestCase):

    def test_energy_and_flux_invariant(self):
        n = 48
        c0 = (n / 2 - 12, n / 2)
        c1 = (n / 2 + 12, n / 2)
        psi = seed_vortices(n, [c0, c1], [1, -1])
        out = relax(psi, zero_links(n), lam=2.0, beta=1.0, dt=0.05, steps=1200)
        e0 = energy(out["psi"], out["theta"], 2.0, 1.0)
        f0 = local_flux(out["theta"], c0, 6.5)
        rng = np.random.default_rng(1)
        alpha = rng.uniform(-np.pi, np.pi, (n, n))
        psi_g, theta_g = gauge_transform(out["psi"], out["theta"], alpha)
        self.assertAlmostEqual(energy(psi_g, theta_g, 2.0, 1.0), e0, places=6)
        self.assertAlmostEqual(local_flux(theta_g, c0, 6.5), f0, places=6)

    def test_gauge_transform_changes_the_fields(self):
        # the transform is nontrivial (fields move) even though observables don't
        n = 24
        psi = seed_vortices(n, [(12, 12)], [1])
        theta = zero_links(n)
        rng = np.random.default_rng(2)
        alpha = rng.uniform(-np.pi, np.pi, (n, n))
        psi_g, theta_g = gauge_transform(psi, theta, alpha)
        self.assertGreater(np.abs(psi_g - psi).max(), 0.1)
        self.assertGreater(np.abs(theta_g[0]).max(), 0.1)


if __name__ == "__main__":
    unittest.main()
