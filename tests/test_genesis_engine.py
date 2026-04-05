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
        self.assertTrue(voxel_values.issubset({0, 1, 2, 3, 4}))

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

    def test_agent_senses_local_terrain(self) -> None:
        engine = GenesisEngine(config=EngineConfig(chunk_size=8, seed=42))
        agent = engine.add_agent(position=(4, 4, 4))
        reading = agent.sense(engine.field)
        self.assertIn("local_value", reading)
        self.assertIn("local_gradient", reading)
        self.assertAlmostEqual(reading["local_value"], float(engine.field[4, 4, 4]))

    def test_agent_moves_during_evolution(self) -> None:
        engine = GenesisEngine(config=EngineConfig(chunk_size=8, seed=42))
        agent = engine.add_agent(position=(4, 4, 4))
        engine.evolve_field(steps=10, dt=0.01)
        self.assertEqual(len(agent.trail), 11)   # initial position + 10 moves
        self.assertEqual(len(agent.sense_log), 10)
        for coord in agent.position:
            self.assertGreaterEqual(coord, 0)
            self.assertLess(coord, 8)

    def test_agent_appears_in_metrics(self) -> None:
        engine = GenesisEngine(config=EngineConfig(chunk_size=8, seed=42))
        engine.add_agent()
        engine.evolve_field(steps=5, dt=0.01, record_every=5)
        last_snapshot = engine.history[-1]
        self.assertIn("agents", last_snapshot)
        self.assertEqual(len(last_snapshot["agents"]), 1)
        self.assertIn("position", last_snapshot["agents"][0])


if __name__ == "__main__":
    unittest.main()
