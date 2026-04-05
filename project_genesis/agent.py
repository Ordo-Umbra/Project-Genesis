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
        rng: np.random.Generator | None = None,
    ):
        self.chunk_size = chunk_size
        self.rng = rng if rng is not None else np.random.default_rng()
        if position is not None:
            self.position = np.array(position, dtype=int)
        else:
            self.position = self.rng.integers(0, chunk_size, size=3)
        self.trail: list[tuple[int, int, int]] = [tuple(self.position)]
        self.sense_log: list[dict[str, float]] = []

    def _neighbor_positions(self) -> list[tuple[int, int, np.ndarray]]:
        """Return (axis, delta, neighbor_coord) tuples for the 6-connected neighborhood."""
        results: list[tuple[int, int, np.ndarray]] = []
        for axis in range(3):
            for delta in (-1, 1):
                neighbor = self.position.copy()
                neighbor[axis] = (neighbor[axis] + delta) % self.chunk_size
                results.append((axis, delta, neighbor))
        return results

    def sense(self, field: np.ndarray) -> dict[str, float]:
        """Read local field properties at the agent's current position."""
        x, y, z = self.position
        local_value = float(field[x, y, z])

        neighbors = [float(field[tuple(pos)]) for _, _, pos in self._neighbor_positions()]

        reading = {
            "local_value": local_value,
            "neighbor_mean": float(np.mean(neighbors)),
            "neighbor_max": float(np.max(neighbors)),
            "neighbor_min": float(np.min(neighbors)),
            "local_gradient": float(np.max(neighbors) - np.min(neighbors)),
        }
        self.sense_log.append(reading)
        return reading

    def step(self, field: np.ndarray) -> tuple[int, int, int]:
        """Move one step using gradient-ascent with stochastic exploration.

        The agent prefers moving toward higher field density (seeking denser
        terrain) but has a 20% chance to explore randomly.
        """
        explore_prob = 0.2

        if self.rng.random() < explore_prob:
            # Random walk constrained to 6-connected neighborhood (single-axis)
            axis = int(self.rng.integers(0, 3))
            delta = int(self.rng.choice([-1, 1]))
            direction = np.zeros(3, dtype=int)
            direction[axis] = delta
        else:
            best_direction = np.array([0, 0, 0])
            best_value = float(field[tuple(self.position)])
            for axis, delta, neighbor in self._neighbor_positions():
                val = float(field[tuple(neighbor)])
                if val > best_value:
                    best_value = val
                    best_direction = np.zeros(3, dtype=int)
                    best_direction[axis] = delta
            direction = best_direction

        self.position = (self.position + direction) % self.chunk_size
        self.trail.append(tuple(self.position))
        return tuple(self.position)

    def to_dict(self) -> dict:
        """Serialize the agent state for snapshots."""
        return {
            "position": [int(c) for c in self.position],
            "trail_length": len(self.trail),
            "sense_count": len(self.sense_log),
            "last_sense": self.sense_log[-1] if self.sense_log else None,
        }
