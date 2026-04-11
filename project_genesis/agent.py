"""Minimal terrain-sensing agent for the URP world."""

from __future__ import annotations

import numpy as np


class Agent:
    """A simple agent that senses local field properties and moves through the world."""

    def __init__(
        self,
        position: tuple[int, int, int] | None = None,
        *,
        chunk_size: int,
        agent_id: str | None = None,
        goal: str = "density",
        explore_probability: float = 0.2,
        interaction_radius: int = 2,
        rng: np.random.Generator | None = None,
    ):
        self.chunk_size = chunk_size
        self.agent_id = agent_id
        self.goal = goal
        self.explore_probability = explore_probability
        self.interaction_radius = interaction_radius
        self.rng = rng if rng is not None else np.random.default_rng()
        if position is not None:
            self.position = np.array(position, dtype=int)
        else:
            self.position = self.rng.integers(0, chunk_size, size=3)
        self.trail: list[tuple[int, int, int]] = [self._position_tuple()]
        self.sense_log: list[dict[str, float | int | str | None]] = []
        self.visit_counts: dict[tuple[int, int, int], int] = {self._position_tuple(): 1}
        self.pending_action: dict | None = None

    def _neighbor_positions(self) -> list[tuple[int, int, np.ndarray]]:
        """Return (axis, delta, neighbor_coord) tuples for the 6-connected neighborhood."""
        results: list[tuple[int, int, np.ndarray]] = []
        for axis in range(3):
            for delta in (-1, 1):
                neighbor = self.position.copy()
                neighbor[axis] = (neighbor[axis] + delta) % self.chunk_size
                results.append((axis, delta, neighbor))
        return results

    def _position_tuple(
        self,
        position: np.ndarray | tuple[int, int, int] | list[int] | None = None,
    ) -> tuple[int, int, int]:
        target = self.position if position is None else position
        return tuple(int(coord) for coord in target)

    def _wrapped_distance(self, other_position: np.ndarray) -> float:
        deltas = np.abs(self.position - other_position)
        wrapped = np.minimum(deltas, self.chunk_size - deltas)
        return float(np.linalg.norm(wrapped))

    def _local_signal(
        self,
        field: np.ndarray,
        position: np.ndarray,
        *,
        beta: float = 0.0,
    ) -> dict[str, float]:
        local_value = float(field[tuple(position)])
        neighbors: list[float] = []
        squared_deltas: list[float] = []
        for axis in range(3):
            for delta in (-1, 1):
                neighbor = position.copy()
                neighbor[axis] = (neighbor[axis] + delta) % self.chunk_size
                neighbor_value = float(field[tuple(neighbor)])
                neighbors.append(neighbor_value)
                squared_deltas.append((neighbor_value - local_value) ** 2)

        neighbor_mean = float(np.mean(neighbors))
        local_gradient = float(np.max(neighbors) - np.min(neighbors))
        local_gradient_energy = float(np.mean(squared_deltas))
        local_kappa = 1.0 / (1.0 + local_gradient_energy)
        local_delta_i = float(max(0.0, neighbor_mean - local_value))
        local_s_signal = float((beta * local_gradient_energy) + (local_kappa * local_delta_i))

        return {
            "local_value": local_value,
            "neighbor_mean": neighbor_mean,
            "neighbor_max": float(np.max(neighbors)),
            "neighbor_min": float(np.min(neighbors)),
            "local_gradient": local_gradient,
            "local_gradient_energy": local_gradient_energy,
            "local_kappa": local_kappa,
            "local_delta_i": local_delta_i,
            "local_s_signal": local_s_signal,
        }

    def sense(
        self,
        field: np.ndarray,
        *,
        agents: list["Agent"] | None = None,
        beta: float = 0.0,
    ) -> dict[str, float | int | str | None]:
        """Read local field properties at the agent's current position."""
        reading = self._local_signal(field, self.position, beta=beta)
        peer_distances = [
            self._wrapped_distance(other.position)
            for other in agents or []
            if other is not self
        ]
        nearby_agents = [distance for distance in peer_distances if distance <= self.interaction_radius]
        reading.update(
            {
                "agent_id": self.agent_id,
                "goal": self.goal,
                "nearby_agents": len(nearby_agents),
                "nearest_agent_distance": min(peer_distances) if peer_distances else None,
            }
        )
        self.sense_log.append(reading)
        return reading

    def _score_candidate(
        self,
        candidate: np.ndarray,
        field: np.ndarray,
        *,
        occupied_positions: set[tuple[int, int, int]],
        shared_context: dict[str, float | int | tuple[int, int, int] | None] | None,
        beta: float,
    ) -> float:
        signal = self._local_signal(field, candidate, beta=beta)
        candidate_position = self._position_tuple(candidate)
        crowding_penalty = 1.0 if candidate_position in occupied_positions else 0.0
        novelty = 1.0 / (1.0 + self.visit_counts.get(candidate_position, 0))

        shared_anchor = shared_context.get("best_s_position") if shared_context else None
        if shared_anchor is None:
            anchor_bonus = 0.0
        else:
            anchor = np.array(shared_anchor, dtype=int)
            deltas = np.abs(candidate - anchor)
            wrapped = np.minimum(deltas, self.chunk_size - deltas)
            anchor_bonus = 1.0 / (1.0 + float(np.linalg.norm(wrapped)))

        if self.goal == "explore":
            return (0.25 * signal["local_value"]) + (1.5 * novelty) + (0.5 * anchor_bonus) - crowding_penalty
        if self.goal == "s_functional":
            return signal["local_s_signal"] + (0.2 * novelty) + (0.2 * anchor_bonus) - crowding_penalty
        return signal["local_value"] + (0.15 * signal["local_s_signal"]) + (0.1 * anchor_bonus) - crowding_penalty

    def step(
        self,
        field: np.ndarray,
        *,
        agents: list["Agent"] | None = None,
        occupied_positions: set[tuple[int, int, int]] | None = None,
        shared_context: dict[str, float | int | tuple[int, int, int] | None] | None = None,
        beta: float = 0.0,
    ) -> tuple[int, int, int]:
        """Move one step using gradient-ascent with stochastic exploration.

        The agent prefers moving toward higher field density (seeking denser
        terrain) but has a 20% chance to explore randomly.
        """
        explore_prob = self.explore_probability

        if self.rng.random() < explore_prob:
            candidate_positions = [neighbor for _, _, neighbor in self._neighbor_positions()]
            best_position = candidate_positions[int(self.rng.integers(0, len(candidate_positions)))]
        else:
            candidate_positions = [self.position.copy(), *[neighbor for _, _, neighbor in self._neighbor_positions()]]
            occupied = set(occupied_positions or set())
            current_position = self._position_tuple()
            occupied.discard(current_position)
            best_position = max(
                candidate_positions,
                key=lambda candidate: self._score_candidate(
                    candidate,
                    field,
                    occupied_positions=occupied,
                    shared_context=shared_context,
                    beta=beta,
                ),
            )

        self.position = best_position % self.chunk_size
        position_tuple = self._position_tuple()
        self.trail.append(position_tuple)
        self.visit_counts[position_tuple] = self.visit_counts.get(position_tuple, 0) + 1
        return position_tuple

    # ------------------------------------------------------------------
    # Perception interface (Phase 4)
    # ------------------------------------------------------------------

    def get_perception(
        self,
        world: np.ndarray,
        *,
        agents: list["Agent"] | None = None,
        beta: float = 0.0,
        radius: int = 3,
    ) -> dict:
        """Return a structured perception dictionary for AI agent interfaces.

        The returned dict contains:
        - ``scalar_field``: local 3-D sub-grid of scalar field values centred on
          the agent (with periodic wrapping).
        - ``s_field``: local 3-D sub-grid of S-functional proxy values
          (βΔC + κΔI computed per-voxel).
        - ``nearby_agents``: list of nearby agents with id, type, and position.
        - ``energy``: agent's current local field value (proxy for energy).
        - ``position``: agent's current position.
        """
        pos = self.position
        size = 2 * radius + 1
        scalar_subgrid = np.zeros((size, size, size), dtype=float)
        s_subgrid = np.zeros((size, size, size), dtype=float)

        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                for dk in range(-radius, radius + 1):
                    xi = (pos[0] + di) % world.shape[0]
                    xj = (pos[1] + dj) % world.shape[1]
                    xk = (pos[2] + dk) % world.shape[2]
                    li = di + radius
                    lj = dj + radius
                    lk = dk + radius
                    val = float(world[xi, xj, xk])
                    scalar_subgrid[li, lj, lk] = val

                    # Compute per-voxel S-functional proxy.
                    neighbors_vals: list[float] = []
                    for ax in range(3):
                        for delta in (-1, 1):
                            n = [xi, xj, xk]
                            n[ax] = (n[ax] + delta) % world.shape[ax]
                            neighbors_vals.append(float(world[n[0], n[1], n[2]]))
                    n_mean = float(np.mean(neighbors_vals))
                    sq_deltas = [(nv - val) ** 2 for nv in neighbors_vals]
                    grad_energy = float(np.mean(sq_deltas))
                    kappa = 1.0 / (1.0 + grad_energy)
                    delta_i = max(0.0, n_mean - val)
                    s_subgrid[li, lj, lk] = (beta * grad_energy) + (kappa * delta_i)

        nearby: list[dict] = []
        for other in agents or []:
            if other is self:
                continue
            dist = self._wrapped_distance(other.position)
            if dist <= self.interaction_radius:
                nearby.append({
                    "agent_id": other.agent_id,
                    "goal": other.goal,
                    "position": list(int(c) for c in other.position),
                    "distance": dist,
                })

        return {
            "scalar_field": scalar_subgrid.tolist(),
            "s_field": s_subgrid.tolist(),
            "nearby_agents": nearby,
            "energy": float(world[tuple(pos)]),
            "position": list(int(c) for c in pos),
            "agent_id": self.agent_id,
        }

    # ------------------------------------------------------------------
    # External action execution (Phase 4)
    # ------------------------------------------------------------------

    def execute_pending_action(self, field: np.ndarray) -> tuple[int, int, int]:
        """Execute the queued external action and clear it.

        Supported action types:
        - ``{"type": "move", "direction": [dx, dy, dz]}``
        """
        action = self.pending_action
        self.pending_action = None

        if action is None:
            return self._position_tuple()

        action_type = action.get("type", "move")
        if action_type == "move":
            direction = action.get("direction", [0, 0, 0])
            new_pos = self.position.copy()
            for i in range(3):
                new_pos[i] = (new_pos[i] + int(direction[i])) % self.chunk_size
            self.position = new_pos

        position_tuple = self._position_tuple()
        self.trail.append(position_tuple)
        self.visit_counts[position_tuple] = self.visit_counts.get(position_tuple, 0) + 1
        return position_tuple

    def to_dict(self, *, include_logs: bool = False) -> dict:
        """Serialize the agent state for snapshots."""
        payload = {
            "agent_id": self.agent_id,
            "goal": self.goal,
            "position": list(self._position_tuple()),
            "trail_length": len(self.trail),
            "sense_count": len(self.sense_log),
            "explore_probability": self.explore_probability,
            "interaction_radius": self.interaction_radius,
            "last_sense": self.sense_log[-1] if self.sense_log else None,
        }
        if include_logs:
            payload.update(
                {
                    "trail": [list(self._position_tuple(position)) for position in self.trail],
                    "sense_log": self.sense_log,
                    "visit_counts": [
                        {"position": list(self._position_tuple(position)), "count": int(count)}
                        for position, count in sorted(self.visit_counts.items())
                    ],
                }
            )
        return payload

    @classmethod
    def from_dict(
        cls,
        data: dict,
        *,
        chunk_size: int,
        rng: np.random.Generator | None = None,
    ) -> "Agent":
        agent = cls(
            position=tuple(int(c) for c in data["position"]),
            chunk_size=chunk_size,
            agent_id=data.get("agent_id"),
            goal=str(data.get("goal", "density")),
            explore_probability=float(data.get("explore_probability", 0.2)),
            interaction_radius=int(data.get("interaction_radius", 2)),
            rng=rng,
        )
        trail = [tuple(int(c) for c in position) for position in data.get("trail", [])]
        if trail:
            agent.trail = trail
        agent.sense_log = list(data.get("sense_log", []))
        visit_counts = {
            tuple(int(c) for c in item["position"]): int(item["count"])
            for item in data.get("visit_counts", [])
        }
        if visit_counts:
            agent.visit_counts = visit_counts
        else:
            agent.visit_counts = {agent._position_tuple(): 1}
        return agent
