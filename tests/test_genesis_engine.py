import tempfile
import unittest
from pathlib import Path

import numpy as np

from project_genesis import EngineConfig, GenesisEngine


class GenesisEngineTests(unittest.TestCase):
    def test_repeatable_evolution_with_same_seed(self) -> None:
        config = EngineConfig(chunk_size=8, seed=7, default_steps=6, default_dt=0.02)
        first = GenesisEngine(config=config)
        second = GenesisEngine(config=config)

        np.testing.assert_allclose(first.evolve_field(), second.evolve_field())

    def test_save_and_load_round_trip_preserves_state(self) -> None:
        config = EngineConfig(chunk_size=8, seed=11, default_steps=4, default_dt=0.01)
        engine = GenesisEngine(config=config)
        engine.evolve_field(record_every=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "snapshot.npz"
            engine.save(snapshot_path)
            loaded = GenesisEngine.load(snapshot_path)

        np.testing.assert_allclose(engine.field, loaded.field)
        self.assertEqual(engine.config, loaded.config)
        self.assertEqual(engine.history, loaded.history)

    def test_evolved_field_stays_finite_and_yields_multiple_voxel_classes(self) -> None:
        engine = GenesisEngine(config=EngineConfig(chunk_size=10, seed=3, default_steps=10, default_dt=0.02))
        engine.evolve_field()

        self.assertTrue(np.isfinite(engine.field).all())
        voxel_values = set(np.unique(engine.quantize_to_voxels()).tolist())
        self.assertGreaterEqual(len(voxel_values), 2)
        self.assertTrue(voxel_values.issubset({0, 1, 2}))

    def test_parameter_sensitivity_changes_outcome(self) -> None:
        baseline = GenesisEngine(
            config=EngineConfig(chunk_size=8, seed=5, beta=0.09, gravity=0.22, default_steps=8, default_dt=0.02)
        )
        shifted = GenesisEngine(
            config=EngineConfig(chunk_size=8, seed=5, beta=0.2, gravity=0.1, default_steps=8, default_dt=0.02)
        )

        baseline.evolve_field()
        shifted.evolve_field()

        self.assertFalse(np.allclose(baseline.field, shifted.field))


if __name__ == "__main__":
    unittest.main()
