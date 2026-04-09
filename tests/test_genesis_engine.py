import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from genesis_engine import export_artifacts, main
from project_genesis import EngineConfig, GenesisEngine


class GenesisEngineTests(unittest.TestCase):
    def test_repeatable_evolution_with_same_seed(self) -> None:
        config = EngineConfig(chunk_size=8, seed=7, default_steps=6, default_dt=0.02)
        first = GenesisEngine(config=config)
        second = GenesisEngine(config=config)

        np.testing.assert_allclose(first.evolve_field(), second.evolve_field())

    def test_save_and_load_round_trip_preserves_state(self) -> None:
        config = EngineConfig(
            chunk_size=8,
            seed=11,
            default_steps=4,
            default_dt=0.01,
            agent_count=2,
            agent_goal="s_functional",
        )
        engine = GenesisEngine(config=config)
        engine.evolve_field(record_every=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "snapshot.npz"
            engine.save(snapshot_path)
            loaded = GenesisEngine.load(snapshot_path)

        np.testing.assert_allclose(engine.field, loaded.field)
        self.assertEqual(engine.config, loaded.config)
        self.assertEqual(engine.history, loaded.history)
        self.assertEqual(len(engine.agents), len(loaded.agents))
        self.assertEqual(engine.agent_timelines(), loaded.agent_timelines())

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
        engine.add_agent(position=(5, 4, 4))
        reading = agent.sense(engine.field, agents=engine.agents, beta=engine.BETA)
        self.assertIn("local_value", reading)
        self.assertIn("local_gradient", reading)
        self.assertIn("local_s_signal", reading)
        self.assertEqual(reading["nearby_agents"], 1)
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

    def test_multi_agent_config_populates_agents_and_tracks_summary(self) -> None:
        engine = GenesisEngine(
            config=EngineConfig(
                chunk_size=8,
                seed=9,
                agent_count=3,
                agent_goal="explore",
                default_steps=4,
                default_dt=0.01,
            )
        )
        self.assertEqual(len(engine.agents), 3)
        engine.evolve_field(record_every=2)
        snapshot = engine.history[-1]
        self.assertEqual(snapshot["agent_count"], 3)
        self.assertGreaterEqual(snapshot["agent_unique_positions"], 3)
        self.assertEqual(len(snapshot["agents"]), 3)
        self.assertTrue(all(agent["goal"] == "explore" for agent in snapshot["agents"]))

    def test_agent_influence_changes_field(self) -> None:
        config = EngineConfig(
            chunk_size=8,
            seed=5,
            agent_count=1,
            agent_influence=0.5,
            default_steps=1,
            default_dt=0.1,
        )
        engine = GenesisEngine(config=config)
        before = engine.field.copy()
        agent = engine.agents[0]
        engine.evolve_field()
        self.assertGreater(float(engine.field[tuple(agent.position)]), float(before[tuple(agent.position)]))

    def test_export_artifacts_writes_summary_and_agent_timelines(self) -> None:
        engine = GenesisEngine(
            config=EngineConfig(chunk_size=8, seed=13, agent_count=2, default_steps=3, default_dt=0.01)
        )
        engine.evolve_field(record_every=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "artifacts"
            export_artifacts(output_dir, engine, record_every=1, requested_steps=3)

            self.assertTrue((output_dir / "agent_timelines.json").exists())
            self.assertTrue((output_dir / "run_summary.json").exists())
            self.assertTrue((output_dir / "final_slices" / "final_slice_x.txt").exists())
            self.assertTrue((output_dir / "final_slices" / "final_slice_y.txt").exists())
            self.assertTrue((output_dir / "final_slices" / "final_slice_z.txt").exists())

    def test_cli_main_supports_multi_agent_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "cli-run"
            argv = [
                "genesis_engine.py",
                "--chunk-size",
                "8",
                "--steps",
                "3",
                "--dt",
                "0.01",
                "--seed",
                "4",
                "--record-every",
                "1",
                "--agent-count",
                "2",
                "--agent-goal",
                "s_functional",
                "--output-dir",
                str(output_dir),
            ]
            with patch("sys.argv", argv):
                main()

            self.assertTrue((output_dir / "engine_snapshot.npz").exists())
            self.assertTrue((output_dir / "run_summary.json").exists())
            self.assertTrue((output_dir / "agent_timelines.json").exists())


if __name__ == "__main__":
    unittest.main()
