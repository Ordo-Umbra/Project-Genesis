"""Tests for the stable-structure memory corpus."""

import unittest

import numpy as np

from project_genesis import EngineConfig, GenesisEngine, MemoryCorpus, StableObject
from project_genesis.metrics import compute_local_s


class TestStableObject(unittest.TestCase):
    def test_creation_defaults(self) -> None:
        subfield = np.ones((8, 8, 8))
        voxels = np.zeros((8, 8, 8), dtype=int)
        obj = StableObject(subfield=subfield, voxels=voxels, avg_s=0.05, stability_steps=10)
        self.assertEqual(obj.usage_count, 0)
        self.assertEqual(obj.avg_s, 0.05)
        self.assertEqual(obj.stability_steps, 10)
        self.assertEqual(len(obj.object_id), 8)

    def test_copy_subfield_is_independent(self) -> None:
        subfield = np.ones((4, 4, 4))
        obj = StableObject(subfield=subfield, voxels=np.zeros_like(subfield, dtype=int), avg_s=0.1, stability_steps=5)
        copy = obj.copy_subfield()
        copy[:] = 999.0
        np.testing.assert_array_equal(obj.subfield, np.ones((4, 4, 4)))


class TestMemoryCorpus(unittest.TestCase):
    def _make_patch(self, value: float = 0.5, size: int = 8) -> tuple[np.ndarray, np.ndarray]:
        subfield = np.full((size, size, size), value)
        voxels = np.ones((size, size, size), dtype=int)
        return subfield, voxels

    def test_add_if_stable_below_threshold_returns_none(self) -> None:
        corpus = MemoryCorpus(max_size=10, min_stability=5, min_local_s=0.01)
        subfield, voxels = self._make_patch()
        # Too few stability steps.
        result = corpus.add_if_stable(subfield, voxels, local_s=0.05, stability_steps=3)
        self.assertIsNone(result)
        self.assertEqual(len(corpus), 0)

    def test_add_if_stable_below_s_returns_none(self) -> None:
        corpus = MemoryCorpus(max_size=10, min_stability=5, min_local_s=0.1)
        subfield, voxels = self._make_patch()
        result = corpus.add_if_stable(subfield, voxels, local_s=0.05, stability_steps=10)
        self.assertIsNone(result)
        self.assertEqual(len(corpus), 0)

    def test_add_if_stable_success(self) -> None:
        corpus = MemoryCorpus(max_size=10, min_stability=5, min_local_s=0.01)
        subfield, voxels = self._make_patch()
        result = corpus.add_if_stable(subfield, voxels, local_s=0.05, stability_steps=10)
        self.assertIsNotNone(result)
        self.assertEqual(len(corpus), 1)
        self.assertIsInstance(result, StableObject)

    def test_max_size_enforced(self) -> None:
        corpus = MemoryCorpus(max_size=3, min_stability=1, min_local_s=0.001)
        for i in range(5):
            subfield = np.full((4, 4, 4), 0.1 * (i + 1))
            voxels = np.ones((4, 4, 4), dtype=int) * i
            corpus.add_if_stable(subfield, voxels, local_s=0.01 * (i + 1), stability_steps=5)
        self.assertLessEqual(len(corpus), 3)

    def test_sample_empty_returns_empty(self) -> None:
        corpus = MemoryCorpus()
        self.assertEqual(corpus.sample(n=3), [])

    def test_sample_increments_usage_count(self) -> None:
        corpus = MemoryCorpus(max_size=5, min_stability=1, min_local_s=0.001)
        subfield, voxels = self._make_patch(value=0.3, size=4)
        corpus.add_if_stable(subfield, voxels, local_s=0.02, stability_steps=5)
        rng = np.random.default_rng(42)
        sampled = corpus.sample(n=2, rng=rng)
        self.assertGreater(len(sampled), 0)
        self.assertGreater(sampled[0].usage_count, 0)

    def test_duplicate_rejection(self) -> None:
        corpus = MemoryCorpus(max_size=10, min_stability=1, min_local_s=0.001)
        subfield, voxels = self._make_patch(value=0.5, size=4)
        first = corpus.add_if_stable(subfield, voxels, local_s=0.05, stability_steps=5)
        self.assertIsNotNone(first)
        # Same patch again.
        second = corpus.add_if_stable(subfield.copy(), voxels.copy(), local_s=0.05, stability_steps=5)
        self.assertIsNone(second)
        self.assertEqual(len(corpus), 1)

    def test_invalid_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            MemoryCorpus(max_size=0)
        with self.assertRaises(ValueError):
            MemoryCorpus(min_stability=-1)
        with self.assertRaises(ValueError):
            MemoryCorpus(min_local_s=-0.5)


class TestComputeLocalS(unittest.TestCase):
    def test_uniform_patch_returns_zero(self) -> None:
        patch = np.ones((8, 8, 8)) * 0.5
        s_val = compute_local_s(patch, beta=0.09)
        self.assertAlmostEqual(s_val, 0.0, places=6)

    def test_varied_patch_returns_positive(self) -> None:
        rng = np.random.default_rng(42)
        patch = rng.random((8, 8, 8))
        s_val = compute_local_s(patch, beta=0.09)
        self.assertGreater(s_val, 0.0)


class TestEngineMemoryCorpusIntegration(unittest.TestCase):
    def test_engine_without_corpus_has_none(self) -> None:
        engine = GenesisEngine(config=EngineConfig(chunk_size=8, seed=42))
        self.assertIsNone(engine.memory_corpus)
        self.assertIsNone(engine.stability_map)

    def test_engine_with_corpus_initialises(self) -> None:
        config = EngineConfig(
            chunk_size=16,
            seed=42,
            enable_memory_corpus=True,
            corpus_max_size=20,
            corpus_min_stability=3,
            corpus_min_local_s=0.005,
        )
        engine = GenesisEngine(config=config)
        self.assertIsNotNone(engine.memory_corpus)
        self.assertIsNotNone(engine.stability_map)
        self.assertEqual(engine.memory_corpus.max_size, 20)
        self.assertEqual(engine.stability_map.shape, engine.field.shape)

    def test_evolve_with_corpus_does_not_crash(self) -> None:
        config = EngineConfig(
            chunk_size=16,
            seed=7,
            enable_memory_corpus=True,
            corpus_max_size=10,
            corpus_min_stability=2,
            corpus_min_local_s=0.001,
            default_steps=20,
            default_dt=0.01,
        )
        engine = GenesisEngine(config=config)
        field = engine.evolve_field(record_every=5)
        self.assertTrue(np.isfinite(field).all())

    def test_evolve_with_corpus_and_coherence(self) -> None:
        """Run with --enable-memory-corpus --coherence-potential --integration-functional."""
        config = EngineConfig(
            chunk_size=16,
            seed=42,
            enable_memory_corpus=True,
            corpus_max_size=10,
            corpus_min_stability=2,
            corpus_min_local_s=0.001,
            use_coherence_potential=True,
            use_integration_functional=True,
            default_steps=15,
            default_dt=0.01,
        )
        engine = GenesisEngine(config=config)
        field = engine.evolve_field(record_every=5)
        self.assertTrue(np.isfinite(field).all())
        # Stability map should have been updated.
        self.assertGreaterEqual(engine.stability_map.max(), 0)

    def test_config_round_trip_with_corpus_fields(self) -> None:
        config = EngineConfig(
            enable_memory_corpus=True,
            corpus_max_size=30,
            corpus_min_stability=4,
            corpus_min_local_s=0.02,
        )
        d = config.to_dict()
        restored = EngineConfig.from_dict(d)
        self.assertEqual(config, restored)


if __name__ == "__main__":
    unittest.main()
