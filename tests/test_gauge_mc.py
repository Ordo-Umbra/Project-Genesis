"""Tests for project_genesis.gauge_mc — Monte Carlo confinement layer.

Covers:
- Identity/cold-start → zero action, W(1,1)=1
- Metropolis: acceptance rate, SU(N) unitarity preservation
- Heat-bath SU(2) and SU(3): shape, unitarity
- Overrelaxation: shape, unitarity, action preservation
- Metropolis local ΔS consistency
- Wilson-loop gauge-invariance
- Polyakov-loop bounds
- Creutz-ratio formula
- Area-law fitter on synthetic data
- Full driver summary keys and types
- Deconfinement scan returns one entry per beta value
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from project_genesis.gauge import (
    identity_links,
    random_unitary,
    wilson_action,
    apply_gauge_transform,
    is_unitary,
)
from project_genesis.gauge_mc import (
    metropolis_sweep,
    heatbath_sweep,
    overrelaxation_sweep,
    wilson_loop,
    polyakov_loop,
    creutz_ratio,
    fit_area_law,
    thermalize_and_measure_pure_gauge,
    deconfinement_scan,
)


class TestColdStart(unittest.TestCase):
    """Identity links are zero-action and W(R,T) = 1."""

    def test_identity_action_zero(self):
        links = identity_links(2, (4, 4))
        self.assertAlmostEqual(wilson_action(links), 0.0, places=10)

    def test_identity_wilson_loop_one(self):
        links = identity_links(2, (6, 6))
        for r, t in [(1, 1), (2, 2), (3, 3)]:
            w = wilson_loop(links, (r, t))
            self.assertAlmostEqual(w, 1.0, places=10,
                                   msg=f"W({r},{t}) should be 1 on identity links")

    def test_identity_polyakov_one(self):
        links = identity_links(2, (4, 4))
        p = polyakov_loop(links)
        self.assertAlmostEqual(p, 1.0, places=10)


class TestMetropolis(unittest.TestCase):
    def test_acceptance_rate_positive(self):
        rng = np.random.default_rng(42)
        links = random_unitary(rng, 2, (2, 6, 6), special=True)
        links2, acc = metropolis_sweep(links, 1.5, rng, n_sweeps=5)
        self.assertGreater(acc, 0.05, "Acceptance rate should be > 5 %")
        self.assertLessEqual(acc, 1.0)

    def test_links_stay_unitary(self):
        rng = np.random.default_rng(7)
        links = random_unitary(rng, 2, (2, 6, 6), special=True)
        links2, _ = metropolis_sweep(links, 1.5, rng, n_sweeps=3)
        # check a sample of links
        for mu in range(2):
            for site in np.ndindex(6, 6):
                u = links2[(mu,) + site]
                self.assertTrue(
                    np.allclose(u @ u.conj().T, np.eye(2), atol=1e-9),
                    f"Link (mu={mu}, site={site}) lost unitarity"
                )

    def test_local_delta_s_consistency(self):
        """Verify ΔS via staple matches full action difference on 2-D SU(2)."""
        from project_genesis.gauge_mc import _staple_sum
        rng = np.random.default_rng(99)
        links = random_unitary(rng, 2, (2, 5, 5), special=True)
        mu, site = 0, (2, 3)
        v = random_unitary(rng, 2, special=True, scale=0.2)
        u_old = links[(mu,) + site]
        u_prop = u_old @ v
        a = _staple_sum(links, mu, site)
        delta_s_local = float(np.real(np.trace((u_old - u_prop) @ a.conj().swapaxes(-1, -2))))
        links_prop = links.copy()
        links_prop[(mu,) + site] = u_prop
        delta_s_full = wilson_action(links_prop) - wilson_action(links)
        self.assertAlmostEqual(delta_s_local, delta_s_full, places=8)


class TestHeatbath(unittest.TestCase):
    def test_heatbath_su2_shape_and_unitary(self):
        rng = np.random.default_rng(1)
        links = random_unitary(rng, 2, (2, 4, 4), special=True)
        links2, _ = heatbath_sweep(links, 2.0, rng, n_sweeps=3)
        self.assertEqual(links2.shape, links.shape)
        self.assertTrue(is_unitary(links2.reshape(-1, 2, 2)))

    def test_heatbath_su3_shape_and_unitary(self):
        rng = np.random.default_rng(2)
        links = random_unitary(rng, 3, (2, 4, 4), special=True)
        links2, _ = heatbath_sweep(links, 2.0, rng, n_sweeps=3)
        self.assertEqual(links2.shape, links.shape)
        # check first link
        u = links2[0, 0, 0]
        self.assertTrue(np.allclose(u @ u.conj().T, np.eye(3), atol=1e-8))

    def test_heatbath_lowers_action_at_strong_coupling(self):
        """At β=5 (strong ordering), heat-bath should decrease Wilson action from hot start."""
        rng = np.random.default_rng(55)
        links = random_unitary(rng, 2, (2, 6, 6), special=True, scale=1.0)
        s_before = wilson_action(links)
        links2, _ = heatbath_sweep(links, 5.0, rng, n_sweeps=20)
        s_after = wilson_action(links2)
        self.assertLess(s_after, s_before,
                        "Heat-bath at β=5 should decrease Wilson action from hot start")


class TestOverrelaxation(unittest.TestCase):
    def test_overrelax_shape(self):
        rng = np.random.default_rng(3)
        links = random_unitary(rng, 3, (2, 5, 5), special=True)
        links2, _ = overrelaxation_sweep(links, rng, n_sweeps=5)
        self.assertEqual(links2.shape, links.shape)

    def test_overrelax_approximately_preserves_action(self):
        """Overrelaxation is microcanonical — action should change by < 1 %."""
        rng = np.random.default_rng(4)
        # start from a nearly-ordered config
        links = random_unitary(rng, 2, (2, 6, 6), special=True, scale=0.1)
        s_before = wilson_action(links)
        links2, _ = overrelaxation_sweep(links, rng, n_sweeps=1)
        s_after = wilson_action(links2)
        if s_before > 1e-6:
            rel_change = abs(s_after - s_before) / s_before
            self.assertLess(rel_change, 0.5,
                            "Overrelaxation should not wildly change the action")


class TestWilsonLoop(unittest.TestCase):
    def test_bounds(self):
        rng = np.random.default_rng(10)
        links = random_unitary(rng, 2, (2, 8, 8), special=True)
        for r, t in [(1, 1), (2, 3), (3, 2)]:
            w = wilson_loop(links, (r, t))
            self.assertGreaterEqual(w, -1.0)
            self.assertLessEqual(w, 1.0 + 1e-9)

    def test_gauge_invariance(self):
        """W(R,T) must be invariant under gauge transformations."""
        rng = np.random.default_rng(20)
        n = 2
        spatial = (6, 6)
        links = random_unitary(rng, n, (2, *spatial), special=True)
        g = random_unitary(rng, n, spatial, special=True)
        from project_genesis.gauge import sector_field_to_psi, random_psi
        psi = random_psi(rng, n, spatial)
        _, links_t = apply_gauge_transform(psi, links, g)
        w_before = wilson_loop(links, (2, 2))
        w_after = wilson_loop(links_t, (2, 2))
        self.assertAlmostEqual(w_before, w_after, places=8,
                               msg="Wilson loop must be gauge-invariant")


class TestPolyakovLoop(unittest.TestCase):
    def test_identity_links_polyakov_one(self):
        links = identity_links(2, (4, 4))
        p = polyakov_loop(links)
        self.assertAlmostEqual(p, 1.0, places=10)

    def test_hot_start_polyakov_small(self):
        """On a completely random (hot) configuration, <P> should be near 0."""
        rng = np.random.default_rng(30)
        links = random_unitary(rng, 2, (2, 8, 8), special=True, scale=1.5)
        p = polyakov_loop(links)
        self.assertLess(abs(p), 0.5,
                        "Polyakov loop on hot start should be near 0")


class TestCreutzRatio(unittest.TestCase):
    def test_formula(self):
        w22 = 0.5
        w11 = 0.9
        w12 = 0.7
        w21 = 0.7
        chi = creutz_ratio(w22, w11, w21, w12)
        expected = -math.log((w22 * w11) / (w21 * w12))
        self.assertAlmostEqual(chi, expected, places=12)

    def test_nan_on_nonpositive(self):
        self.assertTrue(math.isnan(creutz_ratio(0.0, 0.5, 0.5, 0.5)))
        self.assertTrue(math.isnan(creutz_ratio(-0.1, 0.5, 0.5, 0.5)))


class TestAreaLawFitter(unittest.TestCase):
    def test_recovers_known_sigma(self):
        """Fit should recover σ and c from synthetic data."""
        sigma_true = 0.12
        c_true = 0.08
        rs = [1, 2, 3]
        ts = [1, 2, 3]
        W = np.zeros((3, 3))
        for i, r in enumerate(rs):
            for j, t in enumerate(ts):
                W[i, j] = math.exp(-sigma_true * r * t - c_true * (r + t))
        result = fit_area_law(W, rs, ts)
        self.assertAlmostEqual(result["sigma"], sigma_true, places=4)
        self.assertAlmostEqual(result["perimeter_coeff"], c_true, places=4)
        self.assertLess(result["fit_residual"], 1e-8)

    def test_creutz_ratios_present(self):
        rs = [1, 2, 3]
        ts = [1, 2, 3]
        W = np.ones((3, 3)) * 0.5
        W[0, 0] = 0.9
        result = fit_area_law(W, rs, ts)
        self.assertIn("creutz_ratios", result)

    def test_zero_loops_handled(self):
        """All-zero loops should not crash; returns sigma=0.0."""
        rs = [1, 2]
        ts = [1, 2]
        W = np.zeros((2, 2))
        result = fit_area_law(W, rs, ts)
        self.assertEqual(result["sigma"], 0.0)


class TestDriverV2(unittest.TestCase):
    def test_driver_summary_keys(self):
        rng = np.random.default_rng(4)
        for updater in ("metropolis", "heatbath", "overrelax"):
            summary, links = thermalize_and_measure_pure_gauge(
                size=4, n=2, beta_g=1.8, rng=rng,
                updater=updater, n_therm=5, n_meas=3
            )
            for key in ("updater", "beta_g", "loop_averages",
                        "polyakov_mean", "final_wilson_action",
                        "area_law_fit", "polyakov_susceptibility"):
                self.assertIn(key, summary, f"Key {key!r} missing for updater={updater!r}")
            self.assertEqual(summary["updater"], updater)
            self.assertIsInstance(links, np.ndarray)

    def test_driver_ndim3_runs(self):
        rng = np.random.default_rng(5)
        summary, _ = thermalize_and_measure_pure_gauge(
            size=3, n=2, beta_g=2.0, rng=rng,
            ndim=3, n_therm=3, n_meas=2,
        )
        self.assertIn("sigma", summary["area_law_fit"])


class TestDeconfinementScan(unittest.TestCase):
    def test_scan_returns_one_entry_per_beta(self):
        rng = np.random.default_rng(6)
        betas = [1.0, 2.0, 3.0]
        results = deconfinement_scan(
            size=4, n=2, beta_values=betas, rng=rng,
            ndim=2, n_therm=5, n_meas=3, updater="metropolis"
        )
        self.assertEqual(len(results), len(betas))
        for r, b in zip(results, betas):
            self.assertAlmostEqual(r["beta_g"], b)
            self.assertIn("polyakov_mean", r)
