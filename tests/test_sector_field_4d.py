"""Checks for the 4-D sector field instrument (sector_field_4d)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.instanton_scales import cp_flow_step  # noqa: E402
from project_genesis.sector_field_4d import (  # noqa: E402
    action_density_f,
    charge_density_from_f,
    coherent_fraction_4d,
    field_strength,
    link_phases,
    plaquette_angles,
    sector_action,
    sector_flow_diffusion_constant,
    sector_flow_step,
    sector_gradient_flow,
    sector_metropolis_sweep,
    topological_charge_4d,
    topological_charge_density_4d,
)
from project_genesis.topological_charge import (  # noqa: E402
    cp_metropolis_sweep,
    topological_charge as geometric_charge,
)


def _random_field(shape, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(shape + (3,)) + 1j * rng.standard_normal(shape + (3,))
    return z / np.linalg.norm(z, axis=-1, keepdims=True)


def _flux_links(L, plane, m):
    """Constant-flux U(1) links on T⁴: flux 2πm through every ``plane`` plaquette."""
    u = np.ones((4,) + (L,) * 4, dtype=np.complex128)
    a, b = plane
    idx = np.indices((L,) * 4)
    u[b] = np.exp(1j * 2.0 * np.pi * m * idx[a] / L ** 2)
    edge = idx[a] == L - 1
    u[a][edge] = np.exp(-1j * 2.0 * np.pi * m * idx[b][edge] / L)
    return u


class TestCompositeCharge(unittest.TestCase):

    def test_intersection_number_exact(self):
        # factorised fluxes: c₁ = m in (0,1), c₁ = n in (2,3) ⇒ Q = m·n exactly
        L = 6
        for m, n in ((1, 1), (2, 3), (-1, 2)):
            u = _flux_links(L, (0, 1), m) * _flux_links(L, (2, 3), n)
            f = np.stack([plaquette_angles(u, mu, nu)
                          for mu in range(4) for nu in range(mu + 1, 4)])
            self.assertAlmostEqual(charge_density_from_f(f).sum(), m * n, places=9)

    def test_pure_phase_field_carries_no_charge(self):
        # ψ = e^{iα(x)}·ψ₀ is CP-gauge-trivial: f ≡ 0 up to rounding
        rng = np.random.default_rng(1)
        z0 = np.zeros((5,) * 4 + (3,), dtype=np.complex128)
        z0[..., 0] = 1.0
        z = z0 * np.exp(1j * rng.standard_normal((5,) * 4))[..., None]
        self.assertLess(np.abs(topological_charge_density_4d(z)).max(), 1e-12)

    def test_constant_field_carries_no_charge(self):
        z = np.zeros((4,) * 4 + (3,), dtype=np.complex128)
        z[..., 1] = 1.0
        self.assertEqual(topological_charge_4d(z), 0.0)

    def test_bogomolny_bound_pointwise(self):
        z = _random_field((6,) * 4, seed=2)
        q = topological_charge_density_4d(z)
        e = action_density_f(z)
        self.assertTrue(np.all(e >= np.abs(q) - 1e-12))
        self.assertTrue(0.0 <= coherent_fraction_4d(z) <= 1.0)

    def test_charge_needs_all_six_components(self):
        with self.assertRaises(ValueError):
            charge_density_from_f(np.zeros((3, 4, 4)))
        with self.assertRaises(ValueError):
            topological_charge_density_4d(_random_field((6, 6)))

    def test_link_phases_are_unit(self):
        u = link_phases(_random_field((5,) * 4, seed=3))
        np.testing.assert_allclose(np.abs(u), 1.0, atol=1e-12)


class TestTwoDimensionalConsistency(unittest.TestCase):
    """The composite flux is a genuine topological charge already in 2-D."""

    def test_flux_is_exactly_integer_on_any_config(self):
        # Σf/2π counts branch wraps: exactly 2π·ℤ even on a rough thermal field
        rng = np.random.default_rng(4)
        z = _random_field((20, 20), seed=4)
        for _ in range(15):
            z = cp_metropolis_sweep(z, 1.5, rng, 0.5)
        flux = field_strength(z)[0].sum() / (2.0 * np.pi)
        self.assertAlmostEqual(flux, round(flux), places=8)

    def test_matches_geometric_charge_on_smooth_fields(self):
        # after gentle gradient flow the two estimators lock to the same
        # integer, with the sign flip of the opposite plaquette orientation
        rng = np.random.default_rng(3)
        z = _random_field((24, 24), seed=3)
        for _ in range(40):
            z = cp_metropolis_sweep(z, 1.7, rng, 0.5)
        for _ in range(40):
            z = cp_flow_step(z, 0.05)
        flux = field_strength(z)[0].sum() / (2.0 * np.pi)
        geo = geometric_charge(z)
        self.assertAlmostEqual(flux, round(flux), places=8)
        self.assertAlmostEqual(geo, round(geo), places=8)
        self.assertAlmostEqual(flux, -geo, places=8)


class TestSectorDynamics(unittest.TestCase):

    def setUp(self):
        self.psi = _random_field((5,) * 4, seed=7)

    def test_metropolis_preserves_normalisation(self):
        rng = np.random.default_rng(0)
        z = sector_metropolis_sweep(self.psi, 1.0, rng)
        np.testing.assert_allclose(np.linalg.norm(z, axis=-1), 1.0, atol=1e-12)

    def test_metropolis_cools_a_hot_start(self):
        # at strong coupling the sampler must lower the action from random
        rng = np.random.default_rng(1)
        z = self.psi
        s0 = sector_action(z)
        for _ in range(10):
            z = sector_metropolis_sweep(z, 2.0, rng, 0.5)
        self.assertLess(sector_action(z), s0)

    def test_flow_step_preserves_normalisation(self):
        z = sector_flow_step(self.psi, 0.05)
        np.testing.assert_allclose(np.linalg.norm(z, axis=-1), 1.0, atol=1e-12)

    def test_flow_descends_the_action(self):
        z = self.psi
        actions = [sector_action(z)]
        for _ in range(5):
            z = sector_flow_step(z, 0.05)
            actions.append(sector_action(z))
        self.assertTrue(all(b < a for a, b in zip(actions, actions[1:])))

    def test_gradient_flow_trajectory(self):
        _, traj = sector_gradient_flow(self.psi, t_max=0.25, dt=0.05,
                                       measure_every=2)
        self.assertEqual(traj[0]["t"], 0.0)
        self.assertAlmostEqual(traj[-1]["t"], 0.25)
        for rec in traj:
            for key in ("t", "S", "Q", "kappa", "rho_mean", "n_lumps"):
                self.assertIn(key, rec)
        self.assertLess(traj[-1]["S"], traj[0]["S"])

    def test_diffusion_constant_is_unity_in_4d(self):
        d = sector_flow_diffusion_constant(size=10, dim=4, dt=0.05, n_steps=30)
        self.assertAlmostEqual(d, 1.0, delta=0.05)

    def test_dimension_generic_action_matches_2d(self):
        from project_genesis.topological_charge import cp_action
        z2 = _random_field((12, 12), seed=9)
        self.assertAlmostEqual(sector_action(z2), cp_action(z2), places=10)


if __name__ == "__main__":
    unittest.main()
