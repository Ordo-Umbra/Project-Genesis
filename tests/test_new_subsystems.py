"""Tests for the new subsystems introduced in Phases 1-4.

Covers:
- ChunkManager activation / deactivation logic
- WebSocket message serialization / deserialization
- S-compass bridge output consistency
- Headless save / load round-trip integrity
- Agent perception and action queue
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from project_genesis import EngineConfig, GenesisEngine
from project_genesis.agent import Agent
from project_genesis.chunk_manager import ChunkManager
from project_genesis.s_compass_bridge import compute_s_gradient, perception_to_action


# ------------------------------------------------------------------ #
# Phase 1 – ChunkManager
# ------------------------------------------------------------------ #


class ChunkManagerTests(unittest.TestCase):
    def test_initial_state_all_active(self) -> None:
        cm = ChunkManager((64, 64, 64), chunk_edge=32)
        self.assertEqual(cm.grid_shape, (2, 2, 2))
        self.assertEqual(cm.active_count, cm.total_chunks)

    def test_update_deactivates_void_chunks(self) -> None:
        field = np.zeros((64, 64, 64))
        # Only populate the first chunk.
        field[:32, :32, :32] = 0.5
        cm = ChunkManager((64, 64, 64), chunk_edge=32)
        cm.update_active_mask(field, void_threshold=0.15)
        self.assertTrue(cm.is_active(0, 0, 0))
        self.assertFalse(cm.is_active(1, 1, 1))

    def test_agent_position_activates_chunk(self) -> None:
        field = np.zeros((64, 64, 64))
        cm = ChunkManager((64, 64, 64), chunk_edge=32)
        cm.update_active_mask(field, void_threshold=0.15, agent_positions=[(33, 33, 33)])
        self.assertTrue(cm.is_active(1, 1, 1))

    def test_active_bounding_boxes(self) -> None:
        cm = ChunkManager((32, 32, 32), chunk_edge=32)
        boxes = list(cm.active_bounding_boxes())
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], ((0, 0, 0), (32, 32, 32)))

    def test_get_chunk_data(self) -> None:
        field = np.ones((32, 32, 32)) * 0.42
        cm = ChunkManager((32, 32, 32), chunk_edge=32)
        data = cm.get_chunk_data(field, 0, 0, 0)
        np.testing.assert_allclose(data, 0.42)


# ------------------------------------------------------------------ #
# Phase 2 – Headless save / load round-trip with new engine fields
# ------------------------------------------------------------------ #


class HeadlessSaveLoadTests(unittest.TestCase):
    def test_save_load_preserves_step_count(self) -> None:
        config = EngineConfig(chunk_size=8, seed=11, default_steps=4, default_dt=0.01)
        engine = GenesisEngine(config=config)
        engine.evolve_field(steps=4, dt=0.01, record_every=4)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "snap.npz"
            engine.save(path)
            loaded = GenesisEngine.load(path)

        np.testing.assert_allclose(engine.field, loaded.field)
        self.assertEqual(engine.config, loaded.config)


# ------------------------------------------------------------------ #
# Phase 3 – WebSocket serialization helpers (unit-level)
# ------------------------------------------------------------------ #


class WebSocketSerializationTests(unittest.TestCase):
    def test_get_world_summary_is_json_serializable(self) -> None:
        engine = GenesisEngine(config=EngineConfig(chunk_size=8, seed=1, agent_count=1))
        engine.step(0.01)
        summary = engine.get_world_summary()
        # Must not raise.
        serialized = json.dumps(summary, default=_json_default)
        parsed = json.loads(serialized)
        self.assertIn("dimensions", parsed)
        self.assertIn("step_count", parsed)
        self.assertIn("s_functional", parsed)
        self.assertIn("agent_positions", parsed)

    def test_queue_agent_action(self) -> None:
        engine = GenesisEngine(config=EngineConfig(chunk_size=8, seed=1, agent_count=1))
        agent_id = engine.agents[0].agent_id
        ok = engine.queue_agent_action(agent_id, {"type": "move", "direction": [1, 0, 0]})
        self.assertTrue(ok)
        self.assertIsNotNone(engine.agents[0].pending_action)

        not_ok = engine.queue_agent_action("nonexistent", {"type": "move", "direction": [0, 0, 0]})
        self.assertFalse(not_ok)


# ------------------------------------------------------------------ #
# Phase 4 – Agent perception & S-compass bridge
# ------------------------------------------------------------------ #


class AgentPerceptionTests(unittest.TestCase):
    def test_get_perception_structure(self) -> None:
        engine = GenesisEngine(config=EngineConfig(chunk_size=8, seed=42, agent_count=2))
        agent = engine.agents[0]
        perception = agent.get_perception(engine.field, agents=engine.agents, beta=engine.BETA, radius=2)

        self.assertIn("scalar_field", perception)
        self.assertIn("s_field", perception)
        self.assertIn("nearby_agents", perception)
        self.assertIn("energy", perception)
        self.assertIn("position", perception)
        self.assertIn("agent_id", perception)

        # scalar_field should be (5,5,5) for radius=2
        sf = np.array(perception["scalar_field"])
        self.assertEqual(sf.shape, (5, 5, 5))

    def test_execute_pending_action_moves_agent(self) -> None:
        agent = Agent(position=(4, 4, 4), chunk_size=8)
        field = np.random.default_rng(0).random((8, 8, 8))
        agent.pending_action = {"type": "move", "direction": [1, 0, -1]}
        pos = agent.execute_pending_action(field)
        self.assertEqual(pos, (5, 4, 3))
        self.assertIsNone(agent.pending_action)

    def test_pending_action_consumed_during_evolve(self) -> None:
        engine = GenesisEngine(config=EngineConfig(chunk_size=8, seed=1, agent_count=1))
        agent = engine.agents[0]
        start_pos = tuple(int(c) for c in agent.position)
        agent.pending_action = {"type": "move", "direction": [0, 0, 0]}
        engine.evolve_field(steps=1, dt=0.01)
        # Action should be consumed.
        self.assertIsNone(agent.pending_action)


class SCompassBridgeTests(unittest.TestCase):
    def test_compute_s_gradient_flat_field(self) -> None:
        s_field = np.ones((5, 5, 5))
        direction = compute_s_gradient(s_field)
        self.assertEqual(direction, (0, 0, 0))

    def test_compute_s_gradient_toward_high(self) -> None:
        s_field = np.zeros((3, 3, 3))
        s_field[2, 1, 1] = 1.0  # high value at +x neighbour
        direction = compute_s_gradient(s_field)
        self.assertEqual(direction, (1, 0, 0))

    def test_perception_to_action(self) -> None:
        engine = GenesisEngine(config=EngineConfig(chunk_size=8, seed=42, agent_count=1))
        agent = engine.agents[0]
        perception = agent.get_perception(engine.field, agents=engine.agents, beta=engine.BETA)
        action = perception_to_action(perception, beta=engine.BETA)
        self.assertEqual(action["type"], "move")
        self.assertEqual(len(action["direction"]), 3)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _json_default(obj: object) -> object:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    unittest.main()
