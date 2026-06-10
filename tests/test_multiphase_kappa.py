"""Tests for the κ-coupled three-component sector field.

Validates the capacity dynamics on the multi-phase model, the κ-gated
integration (scarcity arrests coarsening), the domain counter, and the
multi-phase S-functional.
"""

from __future__ import annotations

import unittest

import numpy as np

from project_genesis.multiphase import (
    analyze_multiphase,
    coherence_integration,
    count_domains,
    multiphase_s_functional,
    multiphase_s_standing,
    sector_labels,
    step_multiphase,
    step_multiphase_kappa,
)


def _random_init(p: int = 3, size: int = 48, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).random((p, size, size)) * 0.1


class TestStepKappa(unittest.TestCase):
    def test_shapes_preserved(self) -> None:
        f = _random_init(size=16)
        k = np.ones((16, 16))
        nf, nk = step_multiphase_kappa(f, k, dt=0.05)
        self.assertEqual(nf.shape, f.shape)
        self.assertEqual(nk.shape, k.shape)

    def test_rejects_bad_kappa_shape(self) -> None:
        with self.assertRaises(ValueError):
            step_multiphase_kappa(_random_init(size=16), np.ones((8, 8)))

    def test_kappa_bounded(self) -> None:
        f = _random_init(size=24)
        k = np.ones((24, 24))
        for _ in range(40):
            f, k = step_multiphase_kappa(f, k, kappa_consumption=80.0, dt=0.1)
        self.assertGreaterEqual(float(k.min()), 0.0)
        self.assertLessEqual(float(k.max()), 1.0 + 1e-12)

    def test_capacity_depletes_at_walls(self) -> None:
        f = _random_init(size=48)
        k = np.ones((48, 48))
        for _ in range(120):
            f, k = step_multiphase_kappa(f, k, kappa_consumption=80.0, kappa_recovery=0.02, dt=0.1)
        report = analyze_multiphase(f, kappa_field=k)
        self.assertLess(report["wall_kappa_mean"], report["interior_kappa_mean"])

    def test_scarcity_arrests_coarsening(self) -> None:
        # Same initial field: a capacity-starved run should retain at least as
        # many domains as a fully-integrating run (walls get pinned).
        init = _random_init(size=48, seed=3)
        plain = init.copy()
        for _ in range(200):
            plain = step_multiphase(plain, diffusion=1.0, gamma=1.5, dt=0.1)
        starved, k = init.copy(), np.ones((48, 48))
        for _ in range(200):
            starved, k = step_multiphase_kappa(
                starved, k, diffusion=1.0, gamma=1.5, dt=0.1,
                kappa_consumption=80.0, kappa_recovery=0.02,
            )
        self.assertGreaterEqual(count_domains(sector_labels(starved)), count_domains(sector_labels(plain)))

    def test_deterministic(self) -> None:
        f1, k1 = _random_init(size=24), np.ones((24, 24))
        f2, k2 = f1.copy(), k1.copy()
        for _ in range(20):
            f1, k1 = step_multiphase_kappa(f1, k1, dt=0.1)
            f2, k2 = step_multiphase_kappa(f2, k2, dt=0.1)
        self.assertTrue(np.array_equal(f1, f2))
        self.assertTrue(np.array_equal(k1, k2))


class TestCountDomains(unittest.TestCase):
    def test_uniform_is_one(self) -> None:
        self.assertEqual(count_domains(np.zeros((8, 8), dtype=int)), 1)

    def test_three_periodic_stripes(self) -> None:
        f = np.zeros((3, 12, 12))
        f[0, :4] = 1
        f[1, 4:8] = 1
        f[2, 8:] = 1
        self.assertEqual(count_domains(sector_labels(f)), 3)

    def test_min_size_filters(self) -> None:
        labels = np.zeros((10, 10), dtype=int)
        labels[0, 0] = 1  # a single-cell domain of label 1
        self.assertGreater(count_domains(labels, min_size=1), count_domains(labels, min_size=5))


class TestStandingCoherence(unittest.TestCase):
    def test_survives_static_equilibrium(self) -> None:
        # The key property the transient ΔI lacked: a static coherent field
        # has a positive, non-vanishing coherence (no time derivative needed).
        f = np.zeros((3, 16, 16))
        f[0, :8] = 1.0  # two clean half-domains
        f[1, 8:] = 1.0
        self.assertGreater(coherence_integration(f, radius=3), 0.0)

    def test_falls_with_fragmentation(self) -> None:
        # Few large domains are more coherent than many small ones.
        coarse = np.zeros((3, 24, 24))
        coarse[0, :12] = 1.0
        coarse[1, 12:] = 1.0
        fine = np.zeros((3, 24, 24))
        for col in range(24):
            fine[col % 3, :, col] = 1.0  # thin alternating stripes
        self.assertGreater(coherence_integration(coarse, radius=3), coherence_integration(fine, radius=3))

    def test_standing_s_does_not_vanish_at_rest(self) -> None:
        f = np.zeros((3, 16, 16))
        f[0, :8] = 1.0
        f[1, 8:] = 1.0
        s = multiphase_s_standing(f, integration_weight=1.0)
        self.assertIn("coherence", s)
        self.assertGreater(s["coherence"], 0.0)
        self.assertGreater(s["s_increment"], s["delta_c"])  # integration contributes


class TestSFunctional(unittest.TestCase):
    def test_uses_field_kappa(self) -> None:
        fields = _random_init(size=16)
        kappa = np.full((16, 16), 0.42)
        s = multiphase_s_functional(fields, None, kappa_field=kappa)
        self.assertAlmostEqual(s["kappa"], 0.42, places=12)
        self.assertEqual(s["delta_i"], 0.0)  # no previous frame

    def test_integration_nonnegative(self) -> None:
        f0 = _random_init(size=24)
        f1 = step_multiphase(f0, dt=0.1)
        s = multiphase_s_functional(f1, f0)
        self.assertGreaterEqual(s["delta_i"], 0.0)
        self.assertGreaterEqual(s["s_increment"], 0.0)


class TestAnalyzeWithKappa(unittest.TestCase):
    def test_report_has_domains_and_kappa(self) -> None:
        f = _random_init(size=24)
        k = np.full((24, 24), 0.7)
        report = analyze_multiphase(f, kappa_field=k)
        self.assertIn("n_domains", report)
        self.assertIn("kappa_mean", report)
        self.assertAlmostEqual(report["kappa_mean"], 0.7, places=6)


if __name__ == "__main__":
    unittest.main()
