"""Tests for the dynamical capacity field κ(x,t).

Validates the capacity dynamics (consumption under load, recovery with
slack, bounds), the κ-gated integration feedback, persistence, and the
multi-scale / per-sector capacity reporting.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project_genesis import EngineConfig, GenesisEngine
from project_genesis.metrics import compute_s_functional, kappa_by_scale


def _engine(**overrides) -> GenesisEngine:
    base = dict(chunk_size=16, seed=7, use_dynamic_kappa=True)
    base.update(overrides)
    return GenesisEngine(config=EngineConfig(**base))


def _sharp_wall_field(size: int = 16) -> np.ndarray:
    """Half-zero / half-one field with a sharp wall in the middle."""
    field = np.zeros((size, size, size))
    field[size // 2 :, :, :] = 1.0
    return field


class TestConfigValidation(unittest.TestCase):
    def test_rejects_bad_parameters(self) -> None:
        with self.assertRaises(ValueError):
            EngineConfig(kappa_baseline=0.0)
        with self.assertRaises(ValueError):
            EngineConfig(kappa_recovery=-0.1)
        with self.assertRaises(ValueError):
            EngineConfig(kappa_consumption=-1.0)
        with self.assertRaises(ValueError):
            EngineConfig(kappa_diffusion=-0.5)

    def test_rejects_unsupported_combo(self) -> None:
        with self.assertRaises(ValueError):
            EngineConfig(use_dynamic_kappa=True, use_coherence_potential=True)
        with self.assertRaises(ValueError):
            EngineConfig(use_dynamic_kappa=True, use_integration_functional=True)


class TestKappaField(unittest.TestCase):
    def test_initialized_at_baseline(self) -> None:
        engine = _engine(kappa_baseline=0.8)
        self.assertIsNotNone(engine.kappa_field)
        self.assertTrue(np.allclose(engine.kappa_field, 0.8))

    def test_absent_when_disabled(self) -> None:
        engine = GenesisEngine(config=EngineConfig(chunk_size=16, seed=7))
        self.assertIsNone(engine.kappa_field)

    def test_depletes_under_load(self) -> None:
        # A sharp wall is high |∇φ|² load: κ at the wall must drop below κ
        # in the flat interior.
        engine = _engine()
        engine.field = _sharp_wall_field()
        for _ in range(30):
            engine.step(0.05)
        wall_kappa = float(engine.kappa_field[8, 8, 8])
        flat_kappa = float(engine.kappa_field[4, 8, 8])
        self.assertLess(wall_kappa, flat_kappa)

    def test_recovers_toward_baseline_with_slack(self) -> None:
        engine = _engine(kappa_baseline=1.0, kappa_recovery=0.5)
        engine.field = np.full((16, 16, 16), 0.4)  # zero load
        engine.kappa_field[:] = 0.2  # depleted
        for _ in range(50):
            engine.step(0.05)
        self.assertGreater(float(engine.kappa_field.mean()), 0.5)

    def test_stays_bounded(self) -> None:
        engine = _engine(kappa_consumption=50.0)
        for _ in range(50):
            engine.step(0.05)
        self.assertGreaterEqual(float(engine.kappa_field.min()), 0.0)
        self.assertLessEqual(float(engine.kappa_field.max()), 1.0 + 1e-12)

    def test_deterministic(self) -> None:
        e1, e2 = _engine(), _engine()
        for _ in range(10):
            e1.step(0.01)
            e2.step(0.01)
        self.assertTrue(np.array_equal(e1.kappa_field, e2.kappa_field))
        self.assertTrue(np.array_equal(e1.field, e2.field))


class TestFeedback(unittest.TestCase):
    def test_depleted_kappa_slows_integration(self) -> None:
        # With κ ≈ 0 the diffusion term vanishes, so a sharp wall must decay
        # more slowly than at full capacity.
        full = _engine(kappa_recovery=0.0, kappa_consumption=0.0)
        starved = _engine(kappa_recovery=0.0, kappa_consumption=0.0)
        full.field = _sharp_wall_field()
        starved.field = _sharp_wall_field()
        starved.kappa_field[:] = 0.01
        for _ in range(20):
            full.step(0.02)
            starved.step(0.02)
        # Gradient energy across the wall stays higher when capacity is starved.
        def wall_contrast(engine: GenesisEngine) -> float:
            return float(np.abs(np.diff(engine.field[:, 8, 8])).max())

        self.assertGreater(wall_contrast(starved), wall_contrast(full))

    def test_combines_with_sector_potential(self) -> None:
        engine = _engine(use_sector_potential=True, sector_count=3, sector_potential_weight=0.5)
        engine.evolve_field(steps=20, dt=0.01, record_every=10)
        self.assertTrue(np.all(np.isfinite(engine.field)))
        self.assertTrue(np.all(np.isfinite(engine.kappa_field)))


class TestMetricsAndReporting(unittest.TestCase):
    def test_s_functional_uses_field_kappa(self) -> None:
        rng = np.random.default_rng(0)
        field = rng.random((8, 8, 8))
        kappa = np.full((8, 8, 8), 0.37)
        result = compute_s_functional(field, None, 0.09, 0.22, kappa_field=kappa)
        self.assertAlmostEqual(result["kappa"], 0.37, places=12)

    def test_snapshot_includes_kappa_stats(self) -> None:
        engine = _engine()
        engine.evolve_field(steps=4, dt=0.01, record_every=2)
        snap = engine.history[-1]
        for key in ("kappa_field_mean", "kappa_field_min", "kappa_field_std", "kappa_by_scale"):
            self.assertIn(key, snap)
        json.dumps(snap)  # history must stay serializable

    def test_kappa_by_scale_shapes(self) -> None:
        kappa = np.random.default_rng(1).random((16, 16, 16))
        result = kappa_by_scale(kappa, scales=(4, 8, 16, 32))
        self.assertEqual(set(result), {4, 8, 16})  # 32 > field edge, skipped
        for stats in result.values():
            self.assertLessEqual(stats["min"], stats["mean"])

    def test_coarse_scales_smooth_capacity_texture(self) -> None:
        # Fine scales should see more spread than coarse scales on a field
        # with localized depletion.
        kappa = np.ones((16, 16, 16))
        kappa[7:9, :, :] = 0.1  # depleted wall band
        result = kappa_by_scale(kappa, scales=(4, 8))
        self.assertGreaterEqual(result[4]["std"], result[8]["std"])
        self.assertLessEqual(result[4]["min"], result[8]["min"])

    def test_sector_report_includes_kappa_split(self) -> None:
        engine = _engine(use_sector_potential=True, sector_count=3, sector_potential_weight=0.5)
        engine.evolve_field(steps=30, dt=0.01, record_every=10)
        report = engine.analyze_sectorisation()
        self.assertIn("wall_kappa_mean", report)
        self.assertIn("interior_kappa_mean", report)
        for sector in report["sectors"]:
            self.assertIn("mean_kappa", sector)


class TestPersistence(unittest.TestCase):
    def test_save_load_roundtrip_preserves_kappa(self) -> None:
        engine = _engine()
        for _ in range(10):
            engine.step(0.02)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.npz"
            engine.save(path)
            restored = GenesisEngine.load(path)
        self.assertIsNotNone(restored.kappa_field)
        self.assertTrue(np.allclose(restored.kappa_field, engine.kappa_field))

    def test_load_without_kappa_stays_none(self) -> None:
        engine = GenesisEngine(config=EngineConfig(chunk_size=16, seed=7))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.npz"
            engine.save(path)
            restored = GenesisEngine.load(path)
        self.assertIsNone(restored.kappa_field)


if __name__ == "__main__":
    unittest.main()
