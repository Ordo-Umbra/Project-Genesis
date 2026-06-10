"""Tests for the κ-as-soil coupling between the capacity field and corpus.

A recalled seed only roots where local capacity is sufficient (fertile soil);
in depleted ground it does not unfold, and rooting consumes capacity so the
same patch cannot be immediately replanted.
"""

from __future__ import annotations

import unittest

import numpy as np

from project_genesis import EngineConfig, GenesisEngine
from project_genesis.memory_corpus import StableObject


def _corpus_engine(**overrides) -> GenesisEngine:
    base = dict(
        chunk_size=16,
        seed=7,
        enable_memory_corpus=True,
        use_dynamic_kappa=True,
        corpus_kappa_threshold=0.3,
        corpus_kappa_cost=0.5,
    )
    base.update(overrides)
    return GenesisEngine(config=EngineConfig(**base))


def _seed_object(size: int = 4, value: float = 0.9) -> StableObject:
    sub = np.full((size, size, size), value)
    vox = np.full((size, size, size), 4, dtype=int)
    return StableObject(subfield=sub, voxels=vox, avg_s=0.5, stability_steps=10)


class TestConfigValidation(unittest.TestCase):
    def test_rejects_bad_params(self) -> None:
        with self.assertRaises(ValueError):
            EngineConfig(corpus_kappa_threshold=-0.1)
        with self.assertRaises(ValueError):
            EngineConfig(corpus_kappa_cost=1.5)


class TestSeedRooting(unittest.TestCase):
    def test_barren_soil_rejects_seed(self) -> None:
        engine = _corpus_engine()
        engine.kappa_field[:] = 0.05  # below threshold everywhere
        before = engine.field.copy()
        rooted = engine._plant_seed(_seed_object(), 2, 2, 2)
        self.assertFalse(rooted)
        self.assertEqual(engine._corpus_seeds_barren, 1)
        self.assertEqual(engine._corpus_seeds_rooted, 0)
        self.assertTrue(np.array_equal(engine.field, before))  # field untouched

    def test_fertile_soil_roots_and_consumes_capacity(self) -> None:
        engine = _corpus_engine()
        engine.kappa_field[:] = 1.0  # fertile
        rooted = engine._plant_seed(_seed_object(), 2, 2, 2)
        self.assertTrue(rooted)
        self.assertEqual(engine._corpus_seeds_rooted, 1)
        # Planted region's capacity drawn down by the cost fraction (0.5).
        planted_kappa = engine.kappa_field[2:6, 2:6, 2:6]
        self.assertTrue(np.allclose(planted_kappa, 0.5))
        # Untouched region keeps full capacity.
        self.assertAlmostEqual(float(engine.kappa_field[10, 10, 10]), 1.0)

    def test_consumed_soil_blocks_immediate_replant(self) -> None:
        engine = _corpus_engine(corpus_kappa_cost=0.8)  # 1.0 -> 0.2 < 0.3 threshold
        engine.kappa_field[:] = 1.0
        self.assertTrue(engine._plant_seed(_seed_object(), 2, 2, 2))
        # Same spot is now below threshold -> second planting fails.
        self.assertFalse(engine._plant_seed(_seed_object(), 2, 2, 2))
        self.assertEqual(engine._corpus_seeds_rooted, 1)
        self.assertEqual(engine._corpus_seeds_barren, 1)

    def test_without_kappa_planting_always_succeeds(self) -> None:
        # Corpus on, dynamic κ off: original behaviour, no gating.
        engine = GenesisEngine(config=EngineConfig(chunk_size=16, seed=7, enable_memory_corpus=True))
        self.assertIsNone(engine.kappa_field)
        rooted = engine._plant_seed(_seed_object(), 2, 2, 2)
        self.assertTrue(rooted)
        # Counters stay zero (only meaningful with κ active).
        self.assertEqual(engine._corpus_seeds_rooted, 0)


class TestReporting(unittest.TestCase):
    def test_rooting_stats_in_summary(self) -> None:
        engine = _corpus_engine()
        engine.evolve_field(steps=20, dt=0.01, record_every=10)
        summary = engine.summarize_state()
        self.assertIn("corpus_seeds_rooted", summary)
        self.assertIn("corpus_seeds_barren", summary)

    def test_stats_absent_without_both_features(self) -> None:
        engine = GenesisEngine(config=EngineConfig(chunk_size=16, seed=7, enable_memory_corpus=True))
        engine.evolve_field(steps=10, dt=0.01, record_every=5)
        self.assertNotIn("corpus_seeds_rooted", engine.summarize_state())


if __name__ == "__main__":
    unittest.main()
