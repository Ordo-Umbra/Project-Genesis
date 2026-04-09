from pathlib import Path
from typing import Any

import numpy as np

from .agent import Agent
from .config import EngineConfig
from .io import load_snapshot, save_snapshot
from .metrics import calculate_gradients, compute_s_functional, summarize_field
from .render import render_voxel_slice


class GenesisEngine:
    def __init__(
        self,
        chunk_size: int | None = None,
        *,
        config: EngineConfig | None = None,
        field: np.ndarray | None = None,
        history: list[dict[str, float | int | str]] | None = None,
        agents: list[Agent] | None = None,
        run_metadata: dict[str, Any] | None = None,
    ):
        if config is None:
            config = EngineConfig(chunk_size=32 if chunk_size is None else chunk_size)
        elif chunk_size is not None and chunk_size != config.chunk_size:
            config = EngineConfig.from_dict({**config.to_dict(), "chunk_size": chunk_size})

        self.config = config
        self.chunk_size = config.chunk_size
        self.BETA = config.beta
        self.G = config.gravity
        self.rng = np.random.default_rng(config.seed)
        self.field = field.copy() if field is not None else self.rng.random(
            (self.chunk_size, self.chunk_size, self.chunk_size)
        )
        self.prev_field: np.ndarray | None = None
        self.history = list(history or [])
        self.agents: list[Agent] = list(agents or [])
        self.run_metadata: dict[str, Any] = {
            "seed": config.seed,
            "resumed_from": None,
            "configured_agent_goal": config.agent_goal,
        }
        if run_metadata:
            self.run_metadata.update(run_metadata)
        self.run_metadata.setdefault("created_history_length", len(self.history))
        self.run_metadata.setdefault("initial_agent_count", len(self.agents) or config.agent_count)
        if not self.agents and config.agent_count:
            self.populate_agents(config.agent_count)

    def calculate_S_gradients(self) -> tuple[np.ndarray, np.ndarray]:
        return calculate_gradients(self.field)

    def add_agent(
        self,
        position: tuple[int, int, int] | None = None,
        *,
        goal: str | None = None,
        explore_probability: float | None = None,
        interaction_radius: int | None = None,
    ) -> "Agent":
        """Spawn a new agent in the world at the given position (or random if None)."""
        agent = Agent(
            position=position,
            chunk_size=self.chunk_size,
            agent_id=f"agent-{len(self.agents)}",
            goal=self.config.agent_goal if goal is None else goal,
            explore_probability=(
                self.config.agent_explore_probability
                if explore_probability is None
                else explore_probability
            ),
            interaction_radius=(
                self.config.agent_interaction_radius
                if interaction_radius is None
                else interaction_radius
            ),
            rng=self.rng,
        )
        self.agents.append(agent)
        return agent

    def populate_agents(
        self,
        count: int,
        *,
        positions: list[tuple[int, int, int]] | None = None,
        goal: str | None = None,
    ) -> list[Agent]:
        agents: list[Agent] = []
        for index in range(count):
            position = None if positions is None else positions[index]
            agents.append(self.add_agent(position=position, goal=goal))
        return agents

    def _build_agent_shared_context(
        self,
        readings: list[dict[str, float | int | str | None]],
    ) -> dict[str, float | int | tuple[int, int, int] | None]:
        if not readings:
            return {"best_density_position": None, "best_s_position": None, "mean_local_value": 0.0}

        best_density_index = max(range(len(readings)), key=lambda index: float(readings[index]["local_value"]))
        best_s_index = max(range(len(readings)), key=lambda index: float(readings[index]["local_s_signal"]))
        return {
            "best_density_position": tuple(int(coord) for coord in self.agents[best_density_index].position),
            "best_s_position": tuple(int(coord) for coord in self.agents[best_s_index].position),
            "mean_local_value": float(np.mean([float(reading["local_value"]) for reading in readings])),
        }

    def _apply_agent_influence(self, dt: float) -> None:
        if self.config.agent_influence <= 0.0:
            return
        for agent in self.agents:
            self.field[tuple(agent.position)] += self.config.agent_influence * dt

    def agent_timelines(self) -> list[dict[str, Any]]:
        return [agent.to_dict(include_logs=True) for agent in self.agents]

    def _agent_summary(self) -> dict[str, float | int]:
        if not self.agents:
            return {
                "agent_count": 0,
                "agent_unique_positions": 0,
                "agent_mean_trail_length": 0.0,
                "agent_mean_local_value": 0.0,
            }

        unique_positions = {position for agent in self.agents for position in agent.trail}
        last_local_values = [
            float(agent.sense_log[-1]["local_value"])
            for agent in self.agents
            if agent.sense_log and agent.sense_log[-1].get("local_value") is not None
        ]
        return {
            "agent_count": len(self.agents),
            "agent_unique_positions": len(unique_positions),
            "agent_mean_trail_length": float(np.mean([len(agent.trail) for agent in self.agents])),
            "agent_mean_local_value": float(np.mean(last_local_values)) if last_local_values else 0.0,
        }

    def step(self, dt: float) -> np.ndarray:
        self.prev_field = self.field.copy()
        laplacian, gradient_squared = self.calculate_S_gradients()
        d_rho = laplacian + (self.BETA * gradient_squared) - (self.G * self.field)
        self.field = self.field + (d_rho * dt)
        return self.field

    def evolve_field(
        self,
        steps: int | None = None,
        dt: float | None = None,
        *,
        record_every: int = 1,
    ) -> np.ndarray:
        total_steps = self.config.default_steps if steps is None else steps
        delta_t = self.config.default_dt if dt is None else dt

        if record_every <= 0:
            raise ValueError("record_every must be greater than zero")

        start_step = int(self.history[-1]["step"]) if self.history else 0

        for step in range(1, total_steps + 1):
            self.step(delta_t)
            readings = [
                agent.sense(self.field, agents=self.agents, beta=self.BETA)
                for agent in self.agents
            ]
            shared_context = self._build_agent_shared_context(readings)
            occupied_positions = {tuple(int(coord) for coord in agent.position) for agent in self.agents}
            for agent in self.agents:
                occupied_positions.discard(tuple(int(coord) for coord in agent.position))
                next_position = agent.step(
                    self.field,
                    agents=self.agents,
                    occupied_positions=occupied_positions,
                    shared_context=shared_context,
                    beta=self.BETA,
                )
                occupied_positions.add(next_position)
            self._apply_agent_influence(delta_t)
            absolute_step = start_step + step
            if step % record_every == 0 or step == total_steps:
                snapshot = self.summarize_state(step=absolute_step, prev_field=self.prev_field)
                snapshot["slice_z"] = render_voxel_slice(self.quantize_to_voxels(), axis="z")
                self.history.append(snapshot)

        return self.field

    def quantize_to_voxels(self) -> np.ndarray:
        voxel_chunk = np.zeros_like(self.field, dtype=int)
        cfg = self.config
        voxel_chunk[self.field < cfg.void_threshold] = 0
        voxel_chunk[(self.field >= cfg.void_threshold) & (self.field < cfg.air_threshold)] = 1
        voxel_chunk[(self.field >= cfg.air_threshold) & (self.field < cfg.soil_threshold)] = 2
        voxel_chunk[(self.field >= cfg.soil_threshold) & (self.field < cfg.bedrock_threshold)] = 3
        voxel_chunk[self.field >= cfg.bedrock_threshold] = 4
        return voxel_chunk

    def summarize_state(
        self,
        *,
        step: int | None = None,
        prev_field: np.ndarray | None = None,
    ) -> dict[str, float | int]:
        result = summarize_field(
            self.field,
            self.quantize_to_voxels(),
            self.BETA,
            self.G,
            step=step,
            prev_field=prev_field,
        )
        result.update(self._agent_summary())
        if self.agents:
            result["agents"] = [a.to_dict() for a in self.agents]
        return result

    def save(self, path: str | Path) -> Path:
        return save_snapshot(
            path,
            self.field,
            self.config,
            self.history,
            self.agent_timelines(),
            self.run_metadata,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GenesisEngine":
        field, config, history, agents, run_metadata = load_snapshot(path)
        engine = cls(
            config=config,
            field=field,
            history=history,
            agents=[
                Agent.from_dict(agent_data, chunk_size=config.chunk_size)
                for agent_data in agents
            ],
            run_metadata=run_metadata,
        )
        engine.run_metadata["resumed_from"] = str(path)
        return engine
