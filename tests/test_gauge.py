"""Tests for the lattice gauge connection (URP §2–3 core principle).

Cover the group elements, the gauge-invariance of covariant coherence (and the
non-invariance of the naive one), curvature of flat vs generic connections, and
the link to the Ψ∈ℂ³ sector field.
"""

from __future__ import annotations

import unittest

import numpy as np

from project_genesis.gauge import (
    apply_gauge_transform,
    covariant_coherence,
    curvature_density,
    flow_step,
    gauge_staple,
    identity_links,
    is_unitary,
    naive_coherence,
    plaquette,
    pure_gauge_links,
    random_psi,
    random_unitary,
    relax_matter,
    sector_field_to_psi,
    total_s,
    traceless_antihermitian,
    wilson_action,
    yang_mills_residual,
)


def _setup(n, size=8, seed=7, scale=0.7):
    rng = np.random.default_rng(seed)
    spatial = (size, size)
    psi = random_psi(rng, n, spatial)
    links = random_unitary(rng, n, (2, *spatial), special=(n > 1), scale=scale)
    g = random_unitary(rng, n, spatial, special=(n > 1))
    return psi, links, g


class TestGroupElements(unittest.TestCase):
    def test_unitary_all_groups(self) -> None:
        rng = np.random.default_rng(0)
        for n in (1, 2, 3):
            u = random_unitary(rng, n, (5,), special=(n > 1))
            self.assertTrue(is_unitary(u))

    def test_su_n_has_unit_determinant(self) -> None:
        rng = np.random.default_rng(0)
        for n in (2, 3):
            u = random_unitary(rng, n, (10,), special=True)
            np.testing.assert_allclose(np.linalg.det(u), 1.0, atol=1e-9)

    def test_scale_zero_is_identity(self) -> None:
        rng = np.random.default_rng(0)
        u = random_unitary(rng, 3, (4,), special=True, scale=0.0)
        self.assertTrue(np.allclose(u, np.eye(3), atol=1e-12))


class TestCoherenceInvariance(unittest.TestCase):
    def test_covariant_coherence_is_gauge_invariant(self) -> None:
        for n in (1, 2, 3):
            psi, links, g = _setup(n)
            c0 = covariant_coherence(psi, links)
            psi_g, links_g = apply_gauge_transform(psi, links, g)
            c1 = covariant_coherence(psi_g, links_g)
            self.assertAlmostEqual(c1, c0, places=8)

    def test_naive_coherence_is_not_gauge_invariant(self) -> None:
        # A non-trivial local rotation must change the naive coherence.
        for n in (2, 3):
            psi, _, g = _setup(n)
            n0 = naive_coherence(psi)
            psi_g = np.einsum("...ij,...j->...i", g, psi)
            self.assertGreater(abs(naive_coherence(psi_g) - n0), 1e-3)

    def test_connection_restores_destroyed_coherence(self) -> None:
        # Uniform field: maximal naive coherence; a local rotation destroys it;
        # the pure-gauge connection from the same g restores it covariantly.
        rng = np.random.default_rng(3)
        spatial = (10, 10)
        uniform = np.zeros((*spatial, 3), dtype=np.complex128)
        uniform[..., 0] = 1.0
        g = random_unitary(rng, 3, spatial, special=True)
        rotated = np.einsum("...ij,...j->...i", g, uniform)
        self.assertLess(naive_coherence(rotated), naive_coherence(uniform))
        restored = covariant_coherence(rotated, pure_gauge_links(g, 2))
        self.assertAlmostEqual(restored, naive_coherence(uniform), places=8)


class TestCurvature(unittest.TestCase):
    def test_identity_plaquette_is_identity(self) -> None:
        links = identity_links(3, (6, 6))
        p = plaquette(links, 0, 1)
        self.assertTrue(np.allclose(p, np.eye(3), atol=1e-12))

    def test_flat_connection_has_zero_curvature(self) -> None:
        rng = np.random.default_rng(1)
        g = random_unitary(rng, 3, (8, 8), special=True)
        self.assertAlmostEqual(wilson_action(pure_gauge_links(g, 2)), 0.0, places=6)

    def test_generic_connection_has_positive_curvature(self) -> None:
        _, links, _ = _setup(3, scale=0.8)
        self.assertGreater(wilson_action(links), 1.0)

    def test_wilson_action_gauge_invariant(self) -> None:
        psi, links, g = _setup(3)
        w0 = wilson_action(links)
        _, links_g = apply_gauge_transform(psi, links, g)
        self.assertAlmostEqual(wilson_action(links_g), w0, places=6)


class TestSectorField(unittest.TestCase):
    def test_psi_from_sectors_is_normalised(self) -> None:
        fields = np.random.default_rng(0).random((3, 8, 8))
        psi = sector_field_to_psi(fields)
        self.assertEqual(psi.shape, (8, 8, 3))
        np.testing.assert_allclose(np.linalg.norm(psi, axis=-1), 1.0, atol=1e-9)

    def test_covariant_coherence_invariant_on_sector_psi(self) -> None:
        rng = np.random.default_rng(2)
        fields = rng.random((3, 10, 10))
        psi = sector_field_to_psi(fields)
        links = identity_links(3, psi.shape[:-1])
        g = random_unitary(rng, 3, psi.shape[:-1], special=True)
        c0 = covariant_coherence(psi, links)
        psi_g, links_g = apply_gauge_transform(psi, links, g)
        self.assertAlmostEqual(covariant_coherence(psi_g, links_g), c0, places=8)


class TestForces(unittest.TestCase):
    def test_traceless_antihermitian(self) -> None:
        m = np.random.default_rng(0).standard_normal((4, 3, 3)) + 1j * np.random.default_rng(1).standard_normal((4, 3, 3))
        ta = traceless_antihermitian(m)
        self.assertTrue(np.allclose(ta + np.conjugate(np.swapaxes(ta, -1, -2)), 0, atol=1e-12))  # anti-Hermitian
        self.assertTrue(np.allclose(np.trace(ta, axis1=-2, axis2=-1), 0, atol=1e-12))  # traceless

    def test_matter_staple_identity(self) -> None:
        # Σ_μ Σ_x Re Tr[U_μ ψ(x+μ̂)ψ(x)†] == covariant_coherence.
        psi, links, _ = _setup(3)
        ndim = links.shape[0]
        total = 0.0
        for mu in range(ndim):
            m = np.einsum("...i,...j->...ij", np.roll(psi, -1, axis=mu), np.conjugate(psi))
            total += float(np.sum(np.real(np.trace(links[mu] @ m, axis1=-2, axis2=-1))))
        self.assertAlmostEqual(total, covariant_coherence(psi, links), places=8)

    def test_gauge_staple_identity(self) -> None:
        # Σ_μ Σ_x Re Tr[U_μ A_μ] == 4·(N·n_plaq − wilson_action).
        _, links, _ = _setup(3, scale=0.9)
        ndim, n = links.shape[0], links.shape[-1]
        spatial = links.shape[1:-2]
        lhs = sum(
            float(np.sum(np.real(np.trace(links[mu] @ gauge_staple(links, mu), axis1=-2, axis2=-1))))
            for mu in range(ndim)
        )
        n_plaq = (ndim * (ndim - 1) // 2) * int(np.prod(spatial))
        self.assertAlmostEqual(lhs, 4.0 * (n * n_plaq - wilson_action(links)), places=6)

    def test_curvature_density_sums_to_wilson(self) -> None:
        _, links, _ = _setup(3, scale=0.8)
        self.assertAlmostEqual(float(curvature_density(links).sum()), wilson_action(links), places=8)

    def test_relax_matter_normalized_and_raises_coherence(self) -> None:
        psi, links, _ = _setup(3)
        c0 = covariant_coherence(psi, links)
        psi2 = relax_matter(psi, links, rate=0.1)
        self.assertTrue(np.allclose(np.linalg.norm(psi2, axis=-1), 1.0, atol=1e-9))
        self.assertGreater(covariant_coherence(psi2, links), c0)


class TestYangMillsFlow(unittest.TestCase):
    def test_s_increases_under_flow(self) -> None:
        for n in (2, 3):
            psi, links, _ = _setup(n, scale=0.8)
            s0 = total_s(psi, links, 0.5)
            for _ in range(40):
                psi, links = flow_step(psi, links, coupling=0.5, eps=0.04, matter_rate=0.1)
            self.assertGreater(total_s(psi, links, 0.5), s0)

    def test_residual_decreases(self) -> None:
        psi, links, _ = _setup(3, scale=0.9)
        r0 = yang_mills_residual(psi, links, 0.5)
        for _ in range(80):
            psi, links = flow_step(psi, links, coupling=0.5, eps=0.04, matter_rate=0.1)
        self.assertLess(yang_mills_residual(psi, links, 0.5), 0.4 * r0)

    def test_links_stay_su_n(self) -> None:
        psi, links, _ = _setup(3, scale=0.9)
        for _ in range(30):
            psi, links = flow_step(psi, links, coupling=0.5, eps=0.05)
        for mu in range(links.shape[0]):
            self.assertTrue(is_unitary(links[mu]))
        np.testing.assert_allclose(np.linalg.det(links), 1.0, atol=1e-7)

    def test_pure_gauge_flow_reduces_curvature(self) -> None:
        psi, links, _ = _setup(3, scale=1.0)
        w0 = wilson_action(links)
        for _ in range(40):
            psi, links = flow_step(psi, links, coupling=0.0, eps=0.05, evolve_matter=False)
        self.assertLess(wilson_action(links), 0.1 * w0)


if __name__ == "__main__":
    unittest.main()
