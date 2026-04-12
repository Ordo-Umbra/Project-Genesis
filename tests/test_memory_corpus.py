"""Tests for the stable-structure memory corpus."""

import json
import tempfile
import unittest
from pathlib import Path

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
        self.assertEqual(obj.parent_ids, [])

    def test_copy_subfield_is_independent(self) -> None:
        subfield = np.ones((4, 4, 4))
        obj = StableObject(subfield=subfield, voxels=np.zeros_like(subfield, dtype=int), avg_s=0.1, stability_steps=5)
        copy = obj.copy_subfield()
        copy[:] = 999.0
        np.testing.assert_array_equal(obj.subfield, np.ones((4, 4, 4)))

    def test_parent_ids_tracked(self) -> None:
        subfield = np.ones((4, 4, 4))
        voxels = np.zeros((4, 4, 4), dtype=int)
        obj = StableObject(
            subfield=subfield, voxels=voxels, avg_s=0.1, stability_steps=5,
            parent_ids=["aabb1122", "ccdd3344"],
        )
        self.assertEqual(obj.parent_ids, ["aabb1122", "ccdd3344"])

    def test_to_dict_from_dict_round_trip(self) -> None:
        rng = np.random.default_rng(42)
        subfield = rng.random((4, 4, 4))
        voxels = rng.integers(0, 5, size=(4, 4, 4))
        obj = StableObject(
            subfield=subfield, voxels=voxels, avg_s=0.123,
            stability_steps=7, usage_count=3, parent_ids=["parent1"],
        )
        d = obj.to_dict()
        # Must be JSON serializable.
        json_str = json.dumps(d)
        restored = StableObject.from_dict(json.loads(json_str))
        np.testing.assert_allclose(restored.subfield, obj.subfield)
        np.testing.assert_array_equal(restored.voxels, obj.voxels)
        self.assertEqual(restored.avg_s, obj.avg_s)
        self.assertEqual(restored.stability_steps, obj.stability_steps)
        self.assertEqual(restored.usage_count, obj.usage_count)
        self.assertEqual(restored.object_id, obj.object_id)
        self.assertEqual(restored.parent_ids, obj.parent_ids)


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

    def test_add_if_stable_with_parent_ids(self) -> None:
        corpus = MemoryCorpus(max_size=10, min_stability=1, min_local_s=0.001)
        subfield, voxels = self._make_patch(size=4)
        result = corpus.add_if_stable(
            subfield, voxels, local_s=0.05, stability_steps=5,
            parent_ids=["p1", "p2"],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.parent_ids, ["p1", "p2"])

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
        with self.assertRaises(ValueError):
            MemoryCorpus(compose_probability=-0.1)
        with self.assertRaises(ValueError):
            MemoryCorpus(compose_probability=1.5)

    def test_summary_empty_corpus(self) -> None:
        corpus = MemoryCorpus(max_size=10)
        summary = corpus.summary()
        self.assertEqual(summary["corpus_size"], 0)
        self.assertEqual(summary["corpus_max_size"], 10)
        self.assertEqual(summary["corpus_mean_s"], 0.0)
        self.assertEqual(summary["corpus_total_usage"], 0)
        self.assertEqual(summary["corpus_composed_count"], 0)

    def test_summary_populated_corpus(self) -> None:
        corpus = MemoryCorpus(max_size=10, min_stability=1, min_local_s=0.001)
        for i in range(3):
            subfield = np.full((4, 4, 4), 0.1 * (i + 1))
            voxels = np.ones((4, 4, 4), dtype=int)
            corpus.add_if_stable(subfield, voxels, local_s=0.01 * (i + 1), stability_steps=5)
        summary = corpus.summary()
        self.assertEqual(summary["corpus_size"], 3)
        self.assertGreater(summary["corpus_mean_s"], 0.0)
        self.assertGreater(summary["corpus_max_s"], 0.0)

    def test_to_dict_from_dict_round_trip(self) -> None:
        corpus = MemoryCorpus(
            max_size=20, min_stability=3, min_local_s=0.005,
            patch_scales=[4, 8], compose_probability=0.25,
        )
        for i in range(3):
            subfield = np.full((4, 4, 4), 0.1 * (i + 1))
            voxels = np.ones((4, 4, 4), dtype=int) * i
            corpus.add_if_stable(subfield, voxels, local_s=0.02 * (i + 1), stability_steps=5)

        d = corpus.to_dict()
        json_str = json.dumps(d)
        restored = MemoryCorpus.from_dict(json.loads(json_str))

        self.assertEqual(restored.max_size, corpus.max_size)
        self.assertEqual(restored.min_stability, corpus.min_stability)
        self.assertAlmostEqual(restored.min_local_s, corpus.min_local_s)
        self.assertEqual(restored.patch_scales, corpus.patch_scales)
        self.assertAlmostEqual(restored.compose_probability, corpus.compose_probability)
        self.assertEqual(len(restored), len(corpus))
        for orig, rest in zip(corpus.objects, restored.objects):
            np.testing.assert_allclose(rest.subfield, orig.subfield)
            self.assertEqual(rest.object_id, orig.object_id)

    def test_compose_same_shape(self) -> None:
        corpus = MemoryCorpus(max_size=10, min_stability=1, min_local_s=0.001)
        a_sf = np.full((4, 4, 4), 0.3)
        b_sf = np.full((4, 4, 4), 0.7)
        voxels = np.ones((4, 4, 4), dtype=int)
        obj_a = corpus.add_if_stable(a_sf, voxels, local_s=0.02, stability_steps=5)
        obj_b = corpus.add_if_stable(b_sf, voxels, local_s=0.04, stability_steps=5)
        self.assertIsNotNone(obj_a)
        self.assertIsNotNone(obj_b)

        rng = np.random.default_rng(42)
        composed = corpus.compose(obj_a, obj_b, rng=rng)
        self.assertIsNotNone(composed)
        self.assertEqual(len(composed.parent_ids), 2)
        self.assertIn(obj_a.object_id, composed.parent_ids)
        self.assertIn(obj_b.object_id, composed.parent_ids)
        self.assertEqual(composed.subfield.shape, (4, 4, 4))

    def test_compose_different_shapes(self) -> None:
        corpus = MemoryCorpus(max_size=10, min_stability=1, min_local_s=0.001)
        a_sf = np.full((4, 4, 4), 0.3)
        b_sf = np.full((8, 8, 8), 0.7)
        a_vox = np.ones((4, 4, 4), dtype=int)
        b_vox = np.ones((8, 8, 8), dtype=int) * 2
        obj_a = corpus.add_if_stable(a_sf, a_vox, local_s=0.02, stability_steps=5)
        obj_b = corpus.add_if_stable(b_sf, b_vox, local_s=0.04, stability_steps=5)
        self.assertIsNotNone(obj_a)
        self.assertIsNotNone(obj_b)

        rng = np.random.default_rng(7)
        composed = corpus.compose(obj_a, obj_b, rng=rng)
        self.assertIsNotNone(composed)
        # Output shape should match the larger parent.
        self.assertEqual(composed.subfield.shape, (8, 8, 8))

    def test_patch_scales_default(self) -> None:
        corpus = MemoryCorpus()
        self.assertEqual(corpus.patch_scales, [4, 8, 16])

    def test_patch_scales_custom(self) -> None:
        corpus = MemoryCorpus(patch_scales=[2, 4])
        self.assertEqual(corpus.patch_scales, [2, 4])


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

    def test_engine_corpus_uses_patch_scales(self) -> None:
        config = EngineConfig(
            chunk_size=16,
            seed=42,
            enable_memory_corpus=True,
            corpus_patch_scales="4,8",
        )
        engine = GenesisEngine(config=config)
        self.assertEqual(engine.memory_corpus.patch_scales, [4, 8])

    def test_engine_corpus_uses_compose_probability(self) -> None:
        config = EngineConfig(
            chunk_size=16,
            seed=42,
            enable_memory_corpus=True,
            corpus_compose_probability=0.3,
        )
        engine = GenesisEngine(config=config)
        self.assertAlmostEqual(engine.memory_corpus.compose_probability, 0.3)

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
            corpus_patch_scales="4,8",
            corpus_compose_probability=0.2,
        )
        d = config.to_dict()
        restored = EngineConfig.from_dict(d)
        self.assertEqual(config, restored)

    def test_corpus_metrics_in_summary(self) -> None:
        config = EngineConfig(
            chunk_size=16,
            seed=42,
            enable_memory_corpus=True,
            corpus_max_size=10,
            corpus_min_stability=2,
            corpus_min_local_s=0.001,
            default_steps=10,
            default_dt=0.01,
        )
        engine = GenesisEngine(config=config)
        engine.evolve_field(record_every=10)
        summary = engine.summarize_state()
        self.assertIn("corpus_size", summary)
        self.assertIn("corpus_max_size", summary)
        self.assertIn("corpus_mean_s", summary)
        self.assertIn("corpus_composed_count", summary)

    def test_corpus_metrics_not_in_summary_when_disabled(self) -> None:
        config = EngineConfig(chunk_size=8, seed=42, default_steps=3, default_dt=0.01)
        engine = GenesisEngine(config=config)
        engine.evolve_field(record_every=3)
        summary = engine.summarize_state()
        self.assertNotIn("corpus_size", summary)

    def test_corpus_in_world_summary(self) -> None:
        config = EngineConfig(
            chunk_size=8, seed=42,
            enable_memory_corpus=True,
        )
        engine = GenesisEngine(config=config)
        engine.step(0.01)
        summary = engine.get_world_summary()
        self.assertIn("memory_corpus", summary)
        self.assertIn("corpus_size", summary["memory_corpus"])

    def test_save_load_preserves_corpus(self) -> None:
        config = EngineConfig(
            chunk_size=16,
            seed=7,
            enable_memory_corpus=True,
            corpus_max_size=10,
            corpus_min_stability=2,
            corpus_min_local_s=0.001,
            default_steps=25,
            default_dt=0.01,
        )
        engine = GenesisEngine(config=config)
        engine.evolve_field(record_every=25)

        # Manually add a known object to ensure it survives save/load.
        rng = np.random.default_rng(99)
        known_subfield = rng.random((4, 4, 4))
        known_voxels = rng.integers(0, 5, size=(4, 4, 4))
        engine.memory_corpus.add_if_stable(
            known_subfield, known_voxels, local_s=0.05, stability_steps=10,
            parent_ids=["test_parent"],
        )
        corpus_size_before = len(engine.memory_corpus)
        self.assertGreater(corpus_size_before, 0)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "snap.npz"
            engine.save(path)
            loaded = GenesisEngine.load(path)

        self.assertIsNotNone(loaded.memory_corpus)
        self.assertEqual(len(loaded.memory_corpus), corpus_size_before)
        self.assertIsNotNone(loaded.stability_map)
        self.assertEqual(loaded.stability_map.shape, loaded.field.shape)

        # Find the known object and verify it round-tripped.
        loaded_objs = loaded.memory_corpus.objects
        found = [o for o in loaded_objs if o.parent_ids == ["test_parent"]]
        self.assertEqual(len(found), 1)
        np.testing.assert_allclose(found[0].subfield, known_subfield)

    def test_save_load_without_corpus(self) -> None:
        """Save/load with corpus disabled should still work."""
        config = EngineConfig(chunk_size=8, seed=42, default_steps=3, default_dt=0.01)
        engine = GenesisEngine(config=config)
        engine.evolve_field(record_every=3)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "snap.npz"
            engine.save(path)
            loaded = GenesisEngine.load(path)

        np.testing.assert_allclose(engine.field, loaded.field)
        self.assertIsNone(loaded.memory_corpus)

    def test_evolve_with_multi_scale_and_composition(self) -> None:
        """End-to-end test with multi-scale scanning and composition enabled."""
        config = EngineConfig(
            chunk_size=16,
            seed=42,
            enable_memory_corpus=True,
            corpus_max_size=20,
            corpus_min_stability=2,
            corpus_min_local_s=0.001,
            corpus_patch_scales="4,8",
            corpus_compose_probability=0.5,
            default_steps=30,
            default_dt=0.01,
        )
        engine = GenesisEngine(config=config)
        field = engine.evolve_field(record_every=10)
        self.assertTrue(np.isfinite(field).all())


if __name__ == "__main__":
    unittest.main()
