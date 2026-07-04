"""Tests for the matter-coupled gauge ensemble and SU(N ≥ 4) sampling.

The heat-bath samples ``exp(−β_g·S_W + g_m·Σ Re[ψ†(x)U_μ(x)ψ(x+μ̂)])`` with a
quenched matter field ψ.  The matter term enters each link's weight exactly,
as the staple addition ``(g_m/β_g)·ψ(x+μ̂)ψ†(x)`` — these tests pin that down
against analytic results and independent implementations:

- single-link + matter-only weight vs the exact SU(2) Bessel expectation
- SU(4) Cabibbo–Marinari vs the strong-coupling plaquette expansion
- Python vs Numba matter sweeps (same seed → same links)
- overrelaxation conserves the *full* exponent (gauge + matter) exactly
- matter coupling thermodynamically restores covariant coherence
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import i1, iv

from project_genesis.gauge import (
    covariant_coherence,
    is_unitary,
    random_psi,
    random_unitary,
    wilson_action,
)
from project_genesis import gauge_mc as gm


class TestSingleLinkMatterExact(unittest.TestCase):

    def test_su2_matter_only_matches_bessel(self):
        """Weight exp(g·Re[ψ†Uψ']) = exp(g·ReTr(U·M)); for SU(2) the exact
        expectation is ⟨ReTr(U·M_q)⟩ = 2k·I₂(2gk)/I₁(2gk) with k = √det(M_q)."""
        rng = np.random.default_rng(5)
        psi_a = random_psi(rng, 2, ())
        psi_b = random_psi(rng, 2, ())
        m = np.outer(psi_b, np.conj(psi_a))
        m_q = gm._su2_quaternion_part(m)
        k = float(np.sqrt(abs(np.linalg.det(m_q))))
        g = 2.0
        n_draws = 60000
        total = 0.0
        for _ in range(n_draws):
            u = gm._heatbath_su2_from_staple(m, g, rng)
            total += float(np.real(np.trace(u @ m_q)))
        measured = total / n_draws
        exact = 2 * k * float(iv(2, 2 * g * k) / i1(2 * g * k))
        self.assertAlmostEqual(
            measured, exact, delta=0.01,
            msg=f"single-link matter heat-bath {measured:.5f} vs exact {exact:.5f}",
        )

    def test_matter_outer_identity(self):
        """Re[ψ†(x)·U·ψ(x+μ̂)] must equal Re Tr(U·M) with M = ψ(x+μ̂)ψ†(x)."""
        rng = np.random.default_rng(6)
        psi = random_psi(rng, 3, (4, 4))
        u = random_unitary(rng, 3, (), special=True)
        site, mu, spatial = (1, 2), 0, (4, 4)
        m = gm._matter_outer(psi, mu, site, spatial)
        fwd = ((site[0] + 1) % 4, site[1])
        direct = float(np.real(np.conj(psi[site]) @ u @ psi[fwd]))
        via_trace = float(np.real(np.trace(u @ m)))
        self.assertAlmostEqual(direct, via_trace, places=12)


class TestSU4Sampling(unittest.TestCase):

    def test_su4_strong_coupling_plaquette(self):
        """2-D SU(4) at β_g = 0.4: W(1,1) ≈ β_g/(2N) to leading order."""
        beta = 0.4
        rng = np.random.default_rng(2)
        links = random_unitary(rng, 4, (2, 6, 6), special=True)
        for _ in range(120):
            links, _ = gm.heatbath_sweep(links, beta, rng)
        w = 0.0
        n_meas = 200
        for _ in range(n_meas):
            for _ in range(2):
                links, _ = gm.heatbath_sweep(links, beta, rng)
            w += gm.wilson_loop(links, (1, 1))
        w /= n_meas
        self.assertAlmostEqual(
            w, beta / 8.0, delta=0.01,
            msg=f"SU(4) strong-coupling W11={w:.5f} vs β/2N={beta / 8.0:.5f}",
        )

    def test_su4_links_stay_unitary(self):
        rng = np.random.default_rng(11)
        links = random_unitary(rng, 4, (2, 4, 4), special=True)
        for _ in range(30):
            links, _ = gm.heatbath_sweep(links, 2.0, rng)
        self.assertTrue(is_unitary(links.reshape(-1, 4, 4)))

    def test_su4_overrelax_preserves_action(self):
        rng = np.random.default_rng(12)
        links = random_unitary(rng, 4, (2, 5, 5), special=True, scale=0.4)
        for _ in range(20):
            links, _ = gm.heatbath_sweep(links, 2.0, rng)
        s0 = wilson_action(links)
        links2, _ = gm.overrelaxation_sweep(links, rng)
        self.assertLess(abs(wilson_action(links2) - s0) / max(s0, 1e-12), 1e-10)


class TestMatterSweeps(unittest.TestCase):

    def test_python_numba_matter_sweep_agree(self):
        rng = np.random.default_rng(7)
        links = random_unitary(rng, 3, (2, 5, 5), special=True)
        psi = random_psi(rng, 3, (5, 5))
        lp, _ = gm.heatbath_sweep(links, 1.5, np.random.default_rng(7),
                                  matter=psi, matter_coupling=1.5, use_numba=False)
        ln, _ = gm.heatbath_sweep(links, 1.5, np.random.default_rng(7),
                                  matter=psi, matter_coupling=1.5, use_numba=True)
        self.assertLess(float(np.max(np.abs(lp - ln))), 1e-10)

    def test_matter_requires_positive_beta(self):
        rng = np.random.default_rng(8)
        links = random_unitary(rng, 2, (2, 4, 4), special=True)
        psi = random_psi(rng, 2, (4, 4))
        with self.assertRaises(ValueError):
            gm.heatbath_sweep(links, 0.0, rng, matter=psi)

    def test_overrelax_conserves_full_exponent(self):
        """Reflections about the effective staple conserve −β·S_W + g·coherence."""
        beta, g_m = 2.0, 1.0
        rng = np.random.default_rng(21)
        links = random_unitary(rng, 3, (2, 6, 6), special=True, scale=0.4)
        psi = random_psi(rng, 3, (6, 6))
        for _ in range(20):
            links, _ = gm.heatbath_sweep(links, beta, rng,
                                         matter=psi, matter_coupling=g_m)

        def exponent(lk):
            return -beta * wilson_action(lk) + g_m * covariant_coherence(psi, lk)

        e0 = exponent(links)
        links2, _ = gm.overrelaxation_sweep(links, rng, matter=psi,
                                            matter_coupling=g_m, beta_g=beta)
        moved = float(np.max(np.abs(links2 - links)))
        self.assertGreater(moved, 0.1, "overrelaxation should move the links")
        self.assertLess(abs(exponent(links2) - e0), 1e-8 * max(1.0, abs(e0)))

    def test_matter_coupling_restores_coherence(self):
        """The thermodynamic version of 'the connection restores coherence':
        with matter coupling the ensemble's covariant coherence is large and
        positive; without it, it averages to ~0."""
        rng = np.random.default_rng(31)
        spatial = (6, 6)
        psi = random_psi(rng, 2, spatial)

        def mean_coherence(coupled: bool, seed: int) -> float:
            r = np.random.default_rng(seed)
            kw = dict(matter=psi, matter_coupling=2.0) if coupled else {}
            lk = random_unitary(r, 2, (2, *spatial), special=True)
            for _ in range(150):
                lk, _ = gm.heatbath_sweep(lk, 1.0, r, **kw)
            total = 0.0
            for _ in range(150):
                for _ in range(2):
                    lk, _ = gm.heatbath_sweep(lk, 1.0, r, **kw)
                total += covariant_coherence(psi, lk) / (2 * 36)
            return total / 150

        coupled = mean_coherence(True, 1)
        uncoupled = mean_coherence(False, 2)
        self.assertGreater(coupled, 0.3)
        self.assertLess(abs(uncoupled), 0.1)


if __name__ == "__main__":
    unittest.main()
