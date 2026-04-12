"""Stable-structure memory corpus for the URP simulation.

Detects locally stable regions, stores them as reusable building blocks,
and periodically re-seeds them into the universe to prevent collapse
into a single global minimum.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StableObject:
    """A small 3-D patch extracted from the universe that was locally stable."""

    subfield: np.ndarray
    voxels: np.ndarray
    avg_s: float
    stability_steps: int
    usage_count: int = 0
    object_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def copy_subfield(self) -> np.ndarray:
        """Return a copy of the stored subfield data."""
        return self.subfield.copy()


class MemoryCorpus:
    """Bounded collection of :class:`StableObject` instances.

    Parameters
    ----------
    max_size:
        Maximum number of objects retained.  When full, the lowest-scoring
        existing object is evicted to make room.
    min_stability:
        Minimum mean stability-map value a patch must reach before it is
        considered stable enough to store.
    min_local_s:
        Minimum local S-functional value a patch must have.
    """

    def __init__(
        self,
        max_size: int = 50,
        min_stability: int = 5,
        min_local_s: float = 0.01,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        if min_stability < 0:
            raise ValueError("min_stability must be >= 0")
        if min_local_s < 0.0:
            raise ValueError("min_local_s must be >= 0.0")

        self.max_size = max_size
        self.min_stability = min_stability
        self.min_local_s = min_local_s
        self._objects: list[StableObject] = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._objects)

    @property
    def objects(self) -> list[StableObject]:
        return list(self._objects)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add_if_stable(
        self,
        subfield: np.ndarray,
        voxels: np.ndarray,
        local_s: float,
        stability_steps: int,
    ) -> StableObject | None:
        """Add a patch to the corpus if it meets stability and S thresholds.

        Returns the new :class:`StableObject` on success, or ``None`` if the
        patch did not qualify.
        """
        if stability_steps < self.min_stability:
            return None
        if local_s < self.min_local_s:
            return None

        # Check for near-duplicate (same shape, very close avg_s).
        for existing in self._objects:
            if existing.subfield.shape == subfield.shape and abs(existing.avg_s - local_s) < 1e-6:
                if np.allclose(existing.subfield, subfield, atol=1e-4):
                    return None

        obj = StableObject(
            subfield=subfield.copy(),
            voxels=voxels.copy(),
            avg_s=local_s,
            stability_steps=stability_steps,
        )

        if len(self._objects) < self.max_size:
            self._objects.append(obj)
        else:
            # Evict the object with the lowest score.
            worst_index = min(
                range(len(self._objects)),
                key=lambda i: self._score(self._objects[i]),
            )
            if self._score(obj) > self._score(self._objects[worst_index]):
                self._objects[worst_index] = obj
            else:
                return None

        return obj

    def sample(
        self,
        n: int = 1,
        *,
        rng: np.random.Generator | None = None,
    ) -> list[StableObject]:
        """Sample *n* objects from the corpus (with replacement).

        Higher-scoring objects are more likely to be selected.
        Each sampled object has its ``usage_count`` incremented.
        """
        if not self._objects:
            return []

        rng = rng or np.random.default_rng()
        scores = np.array([self._score(obj) for obj in self._objects])
        total = scores.sum()
        if total <= 0:
            probs = np.ones(len(scores)) / len(scores)
        else:
            probs = scores / total

        indices = rng.choice(len(self._objects), size=min(n, len(self._objects)), replace=True, p=probs)
        result: list[StableObject] = []
        for idx in indices:
            obj = self._objects[idx]
            obj.usage_count += 1
            result.append(obj)
        return result

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score(obj: StableObject) -> float:
        """Score an object for selection/eviction.  Higher is better."""
        return obj.avg_s * (1.0 + 0.1 * obj.stability_steps) * (1.0 + 0.05 * obj.usage_count)
