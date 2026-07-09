"""Checks for the 4-D SU(N) clover topological-charge instrument."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.gauge import _dagger, random_unitary  # noqa: E402
from project_genesis.gauge_topology import (  # noqa: E402
    action_density,
    continuum_limit,
    cool,
    flow_scale,
    flow_step,
    gradient_flow,
    mean_plaquette,
    self_dual_fraction,
    staple_field,
    topological_charge,
    topological_charge_density,
    topological_susceptibility,
)


def _identity_links(L=4):
    links = np.zeros((4, L, L, L, L, 3, 3), dtype=complex)
    for mu in range(4):
        links[mu, ..., 0, 0] = links[mu, ..., 1, 1] = links[mu, ..., 2, 2] = 1.0
    return links


class TestTopologicalCharge(unittest.TestCase):

    def test_identity_is_zero(self):
        self.assertEqual(topological_charge(_identity_links()), 0.0)

    def test_pure_gauge_is_zero(self):
        # a gauge transform of the identity (U_μ(x)=g(x)g(x+μ)†) has Q=0
        rng = np.random.default_rng(0)
        L = 4
        g = random_unitary(rng, 3, (L, L, L, L), special=True)
        dag = lambda m: np.conjugate(np.swapaxes(m, -1, -2))
        links = np.stack([g @ dag(np.roll(g, -1, axis=mu)) for mu in range(4)])
        self.assertAlmostEqual(topological_charge(links), 0.0, places=6)

    def test_density_sums_to_total(self):
        rng = np.random.default_rng(1)
        links = random_unitary(rng, 3, (4, 4, 4, 4, 4), special=True, scale=0.7)
        self.assertAlmostEqual(
            float(topological_charge_density(links).sum()),
            topological_charge(links), places=9)

    def test_requires_4d(self):
        rng = np.random.default_rng(2)
        links = random_unitary(rng, 3, (3, 4, 4, 4), special=True)
        with self.assertRaises(ValueError):
            topological_charge_density(links)

    def test_cooling_leaves_identity_trivial(self):
        z = cool(_identity_links(), 3)
        self.assertAlmostEqual(topological_charge(z), 0.0, places=9)

    def test_cooling_reduces_action(self):
        # cooling descends the Wilson action → the mean plaquette rises to ~1
        rng = np.random.default_rng(3)
        links = random_unitary(rng, 3, (4, 6, 6, 6, 6), special=True, scale=0.7)
        p0 = mean_plaquette(links)
        p1 = mean_plaquette(cool(links, 8))
        self.assertGreater(p1, p0)
        self.assertGreater(p1, 0.85)     # cooling drives the field smooth


class TestGradientFlow(unittest.TestCase):

    def test_flow_reduces_action(self):
        # the Wilson flow descends the action → mean plaquette rises
        rng = np.random.default_rng(5)
        links = random_unitary(rng, 3, (4, 4, 4, 4, 4), special=True, scale=0.7)
        p0 = mean_plaquette(links)
        z, _ = gradient_flow(links, t_max=0.4, dt=0.02, measure_every=10)
        self.assertGreater(mean_plaquette(z), p0)

    def test_flow_step_stays_su3(self):
        # a flow step keeps every link in SU(3): unitary with det 1
        rng = np.random.default_rng(6)
        links = random_unitary(rng, 3, (4, 4, 4, 4, 4), special=True, scale=0.7)
        z = flow_step(links, 0.03)
        eye = np.eye(3)
        for mu in range(4):
            prod = z[mu] @ _dagger(z[mu])
            self.assertTrue(np.allclose(prod, eye, atol=1e-9))
            self.assertTrue(np.allclose(np.linalg.det(z[mu]), 1.0, atol=1e-9))

    def test_flow_identity_stays_trivial(self):
        z, traj = gradient_flow(_identity_links(), t_max=0.1, dt=0.05,
                                measure_every=1)
        self.assertAlmostEqual(topological_charge(z), 0.0, places=9)
        # a flat field carries no action → self-dual fraction is defined as 0
        self.assertEqual(self_dual_fraction(_identity_links()), 0.0)

    def test_self_dual_fraction_bounds(self):
        # e(x) >= |q(x)| pointwise (Bogomolny) → f_SD in [0, 1]
        rng = np.random.default_rng(7)
        links = random_unitary(rng, 3, (4, 4, 4, 4, 4), special=True, scale=0.7)
        e = action_density(links)
        q = topological_charge_density(links)
        self.assertTrue(np.all(e + 1e-9 >= np.abs(q)))
        f = self_dual_fraction(links)
        self.assertGreaterEqual(f, 0.0)
        self.assertLessEqual(f, 1.0)

    def test_flow_raises_self_dual_fraction(self):
        # smoothing removes UV noise → the self-dual fraction climbs
        rng = np.random.default_rng(8)
        links = random_unitary(rng, 3, (4, 6, 6, 6, 6), special=True, scale=0.7)
        f0 = self_dual_fraction(links)
        z, _ = gradient_flow(links, t_max=0.6, dt=0.03, measure_every=20)
        self.assertGreater(self_dual_fraction(z), f0)

    def test_action_density_requires_4d(self):
        rng = np.random.default_rng(9)
        links = random_unitary(rng, 3, (3, 4, 4, 4), special=True)
        with self.assertRaises(ValueError):
            action_density(links)

    def test_flow_scale_interpolates(self):
        traj = [{"t": 0.0, "t2E": 0.0}, {"t": 0.5, "t2E": 0.2},
                {"t": 1.0, "t2E": 0.4}]
        self.assertAlmostEqual(flow_scale(traj, 0.3), 0.75, places=9)

    def test_flow_scale_not_reached(self):
        traj = [{"t": 0.0, "t2E": 0.0}, {"t": 0.5, "t2E": 0.1}]
        self.assertTrue(np.isnan(flow_scale(traj, 0.3)))

    def test_continuum_limit_recovers_line(self):
        # y = 0.4 - 0.1 x exactly → intercept 0.4, slope -0.1
        x = [2.5, 2.0, 1.0, 0.5]
        y = [0.4 - 0.1 * xi for xi in x]
        fit = continuum_limit(x, y)
        self.assertAlmostEqual(fit["intercept"], 0.4, places=9)
        self.assertAlmostEqual(fit["slope"], -0.1, places=9)

    def test_continuum_limit_weighted(self):
        # a wild outlier with tiny weight must not move the fit
        x = [2.0, 1.0, 0.0, 1.5]
        y = [0.2, 0.3, 0.4, 9.9]
        w = [1.0, 1.0, 1.0, 1e-9]
        fit = continuum_limit(x, y, w)
        self.assertAlmostEqual(fit["intercept"], 0.4, places=6)

    def test_continuum_limit_needs_two_points(self):
        with self.assertRaises(ValueError):
            continuum_limit([1.0], [0.5])


class TestStapleAndSusceptibility(unittest.TestCase):

    def test_staple_field_shape(self):
        rng = np.random.default_rng(4)
        links = random_unitary(rng, 3, (4, 4, 4, 4, 4), special=True)
        self.assertEqual(staple_field(links, 0).shape, links[0].shape)

    def test_susceptibility_formula(self):
        chi = topological_susceptibility([-1.0, 0.0, 1.0, 0.0], 100.0)
        self.assertAlmostEqual(chi, 0.5 / 100.0, places=12)

    def test_susceptibility_empty(self):
        self.assertEqual(topological_susceptibility([], 50.0), 0.0)


if __name__ == "__main__":
    unittest.main()
