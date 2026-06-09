"""Tests for β-sectorisation / boundary-formation analysis.

These validate the *measurement instrument* — that it counts known sectors,
locates boundaries, and reports the distinction/integration decomposition —
rather than asserting any particular physics outcome.
"""

from __future__ import annotations

import unittest

import numpy as np

from project_genesis import EngineConfig, GenesisEngine
from project_genesis.numba_kernels import sector_potential_drift_3d
from project_genesis.sectorisation import (
    analyze_sectorisation,
    count_triple_junctions,
    free_energy,
    gradient_magnitude,
    label_sectors,
    sector_statistics,
    stationary_sector_count,
)


def _three_block_field() -> np.ndarray:
    """A 24³ field split into three constant slabs along x, with sharp walls.

    Interiors are flat (zero gradient); the inter-slab faces are high-gradient
    walls. Periodic wrapping means the first and last slabs are NOT adjacent
    along x interior (the wall sits between them), so we expect three sectors.
    """
    field = np.zeros((24, 24, 24), dtype=np.float64)
    field[0:8, :, :] = 0.1
    field[8:16, :, :] = 0.5
    field[16:24, :, :] = 0.9
    return field


class TestGradientMagnitude(unittest.TestCase):
    def test_constant_field_has_zero_gradient(self) -> None:
        field = np.full((8, 8, 8), 0.3)
        gm = gradient_magnitude(field)
        self.assertTrue(np.allclose(gm, 0.0))

    def test_gradient_is_nonnegative(self) -> None:
        rng = np.random.default_rng(0)
        field = rng.random((8, 8, 8))
        gm = gradient_magnitude(field)
        self.assertTrue(np.all(gm >= 0.0))


class TestLabelSectors(unittest.TestCase):
    def test_uniform_field_is_single_sector(self) -> None:
        field = np.full((12, 12, 12), 0.4)
        labels, n, wall = label_sectors(field)
        self.assertEqual(n, 1)
        self.assertFalse(wall.any())

    def test_three_block_field_finds_three_sectors(self) -> None:
        field = _three_block_field()
        labels, n, wall = label_sectors(field, method="std", k=0.5)
        self.assertEqual(n, 3)
        self.assertTrue(wall.any())

    def test_min_size_filters_speckle(self) -> None:
        field = _three_block_field()
        # Inject a tiny isolated high value (a single-voxel "sector").
        field[12, 12, 12] = 5.0
        labels, n_all, _ = label_sectors(field, method="std", k=0.5, min_size=1)
        labels_f, n_filtered, _ = label_sectors(field, method="std", k=0.5, min_size=20)
        self.assertGreaterEqual(n_all, n_filtered)

    def test_rejects_non_3d(self) -> None:
        with self.assertRaises(ValueError):
            label_sectors(np.zeros((4, 4)))

    def test_periodic_merge_unifies_wrapped_sector(self) -> None:
        # A single slab in the middle leaves one wrapped interior region around
        # it; with periodic merging the wrapped exterior counts as one sector.
        field = np.zeros((16, 16, 16))
        field[6:10, :, :] = 1.0
        labels_p, n_p, _ = label_sectors(field, method="std", k=0.5, periodic=True)
        labels_np, n_np, _ = label_sectors(field, method="std", k=0.5, periodic=False)
        self.assertLessEqual(n_p, n_np)


class TestSectorStatistics(unittest.TestCase):
    def test_statistics_cover_all_sectors(self) -> None:
        field = _three_block_field()
        labels, n, _ = label_sectors(field, method="std", k=0.5)
        stats = sector_statistics(field, labels, beta=0.09)
        self.assertEqual(len(stats), n)
        for s in stats:
            self.assertGreater(s["volume"], 0)
            self.assertGreaterEqual(s["integration"], 0.0)
            self.assertLessEqual(s["integration"], 1.0)
            self.assertGreaterEqual(s["distinction"], 0.0)

    def test_flat_sector_is_maximally_integrated(self) -> None:
        field = _three_block_field()
        labels, _, _ = label_sectors(field, method="std", k=0.5)
        stats = sector_statistics(field, labels, beta=0.09)
        # Flat interiors → zero internal gradient → integration == 1.
        for s in stats:
            self.assertAlmostEqual(s["integration"], 1.0, places=6)
            self.assertAlmostEqual(s["distinction"], 0.0, places=6)


class TestTripleJunctions(unittest.TestCase):
    def test_no_junctions_in_uniform_field(self) -> None:
        labels = np.ones((6, 6, 6), dtype=int)
        self.assertEqual(count_triple_junctions(labels), 0)

    def test_detects_a_triple_point(self) -> None:
        # Construct labels where a wall voxel borders three distinct sectors.
        labels = np.zeros((3, 3, 3), dtype=int)
        labels[0, 1, 1] = 1  # -x neighbour of centre
        labels[1, 0, 1] = 2  # -y neighbour of centre
        labels[1, 1, 0] = 3  # -z neighbour of centre
        # Centre (1,1,1) is a wall touching sectors 1, 2, 3.
        self.assertGreaterEqual(count_triple_junctions(labels), 1)


class TestFreeEnergy(unittest.TestCase):
    def test_stationary_point_formula(self) -> None:
        a, b = 3.0, 2.0
        self.assertAlmostEqual(stationary_sector_count(a, b), (2 * a / (3 * b)) ** 3)

    def test_free_energy_vectorised(self) -> None:
        vals = free_energy(np.array([1, 2, 3]), a=1.0, b=0.5)
        self.assertEqual(np.asarray(vals).shape, (3,))

    def test_stationary_rejects_nonpositive_b(self) -> None:
        with self.assertRaises(ValueError):
            stationary_sector_count(1.0, 0.0)


class TestAnalyzeSectorisation(unittest.TestCase):
    def test_report_keys_present(self) -> None:
        field = _three_block_field()
        report = analyze_sectorisation(field, beta=0.09, k=0.5)
        for key in (
            "n_sectors",
            "boundary_fraction",
            "wall_distinction",
            "interior_integration",
            "triple_junctions",
            "sectors",
        ):
            self.assertIn(key, report)
        self.assertEqual(report["n_sectors"], 3)
        self.assertTrue(0.0 <= report["boundary_fraction"] <= 1.0)

    def test_report_is_json_serializable(self) -> None:
        import json

        field = _three_block_field()
        report = analyze_sectorisation(field, beta=0.09, k=0.5)
        # Should not raise.
        json.loads(json.dumps(report))


class TestSectorPotential(unittest.TestCase):
    def test_drift_vanishes_at_well_centres(self) -> None:
        # Wells at φ = n/k; drift = -sin(2π·k·φ) should be ~0 there.
        k = 3
        field = np.full((4, 4, 4), 1.0 / k)  # second well
        out = np.empty_like(field)
        sector_potential_drift_3d(field, k, out)
        self.assertTrue(np.allclose(out, 0.0, atol=1e-12))

    def test_drift_is_nonzero_off_equilibria(self) -> None:
        # φ = 0.1 is neither a well (n/k) nor a barrier top ((2n+1)/2k),
        # so the restoring drift must be nonzero.
        k = 3
        field = np.full((4, 4, 4), 0.1)
        out = np.empty_like(field)
        sector_potential_drift_3d(field, k, out)
        self.assertTrue(np.all(np.abs(out) > 1e-3))

    def test_potential_enables_wall_formation(self) -> None:
        # Without the potential the field smooths to a single sector; with it,
        # domain walls (boundary fraction > 0) and >1 sector can form.
        base = EngineConfig(chunk_size=24, beta=0.09, seed=7, default_dt=0.01)
        with_pot = EngineConfig(
            chunk_size=24,
            beta=0.09,
            seed=7,
            default_dt=0.01,
            use_sector_potential=True,
            sector_count=3,
            sector_potential_weight=0.5,
        )
        e0 = GenesisEngine(config=base)
        e1 = GenesisEngine(config=with_pot)
        e0.evolve_field(steps=150, dt=0.01, record_every=50)
        e1.evolve_field(steps=150, dt=0.01, record_every=50)
        r0 = analyze_sectorisation(e0.field, 0.09)
        r1 = analyze_sectorisation(e1.field, 0.09)
        self.assertEqual(r0["n_sectors"], 1)
        self.assertGreater(r1["n_sectors"], 1)

    def test_config_rejects_bad_sector_params(self) -> None:
        with self.assertRaises(ValueError):
            EngineConfig(sector_count=0)
        with self.assertRaises(ValueError):
            EngineConfig(sector_potential_weight=-1.0)


class TestEngineIntegration(unittest.TestCase):
    def test_engine_hook_runs_on_evolved_field(self) -> None:
        engine = GenesisEngine(config=EngineConfig(chunk_size=16, seed=7))
        engine.evolve_field(steps=10, dt=0.01, record_every=5)
        report = engine.analyze_sectorisation()
        self.assertIn("n_sectors", report)
        self.assertGreaterEqual(report["n_sectors"], 0)
        self.assertIsInstance(report["triple_junctions"], int)


if __name__ == "__main__":
    unittest.main()
