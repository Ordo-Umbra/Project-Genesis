"""Stable-structure memory corpus for the URP simulation.

Detects locally stable regions, stores them as reusable building blocks,
and periodically re-seeds them into the universe to prevent collapse
into a single global minimum.

The corpus functions as a digital analog to physical reality: stable
structures are preserved, and then used as foundations for further
expansion and exploration of the configuration space.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

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
    parent_ids: list[str] = field(default_factory=list)

    def copy_subfield(self) -> np.ndarray:
        """Return a copy of the stored subfield data."""
        return self.subfield.copy()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the object to a JSON-compatible dictionary.

        Array data is stored as nested lists for JSON compatibility.
        """
        return {
            "object_id": self.object_id,
            "avg_s": self.avg_s,
            "stability_steps": self.stability_steps,
            "usage_count": self.usage_count,
            "parent_ids": list(self.parent_ids),
            "subfield_shape": list(self.subfield.shape),
            "subfield": self.subfield.tolist(),
            "voxels": self.voxels.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StableObject:
        """Reconstruct a :class:`StableObject` from a dictionary."""
        return cls(
            subfield=np.array(data["subfield"], dtype=np.float64),
            voxels=np.array(data["voxels"], dtype=int),
            avg_s=float(data["avg_s"]),
            stability_steps=int(data["stability_steps"]),
            usage_count=int(data.get("usage_count", 0)),
            object_id=str(data["object_id"]),
            parent_ids=list(data.get("parent_ids", [])),
        )


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
    patch_scales:
        List of cubic patch edge lengths to scan when looking for stable
        structures.  Defaults to ``[4, 8, 16]`` to capture structures at
        multiple spatial scales.
    compose_probability:
        Per-injection probability that two sampled objects are composed
        together (blended) to form a novel building block.  Defaults to
        ``0.15``.
    """

    def __init__(
        self,
        max_size: int = 50,
        min_stability: int = 5,
        min_local_s: float = 0.01,
        patch_scales: list[int] | None = None,
        compose_probability: float = 0.15,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        if min_stability < 0:
            raise ValueError("min_stability must be >= 0")
        if min_local_s < 0.0:
            raise ValueError("min_local_s must be >= 0.0")
        if not 0.0 <= compose_probability <= 1.0:
            raise ValueError("compose_probability must be between 0 and 1")

        self.max_size = max_size
        self.min_stability = min_stability
        self.min_local_s = min_local_s
        self.patch_scales: list[int] = patch_scales if patch_scales is not None else [4, 8, 16]
        self.compose_probability = compose_probability
        self._objects: list[StableObject] = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._objects)

    @property
    def objects(self) -> list[StableObject]:
        return list(self._objects)

    def summary(self) -> dict[str, float | int]:
        """Return summary statistics about the corpus contents."""
        if not self._objects:
            return {
                "corpus_size": 0,
                "corpus_max_size": self.max_size,
                "corpus_mean_s": 0.0,
                "corpus_max_s": 0.0,
                "corpus_total_usage": 0,
                "corpus_mean_stability": 0.0,
                "corpus_composed_count": 0,
            }
        s_values = [obj.avg_s for obj in self._objects]
        return {
            "corpus_size": len(self._objects),
            "corpus_max_size": self.max_size,
            "corpus_mean_s": float(np.mean(s_values)),
            "corpus_max_s": float(np.max(s_values)),
            "corpus_total_usage": sum(obj.usage_count for obj in self._objects),
            "corpus_mean_stability": float(np.mean([obj.stability_steps for obj in self._objects])),
            "corpus_composed_count": sum(1 for obj in self._objects if obj.parent_ids),
        }

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add_if_stable(
        self,
        subfield: np.ndarray,
        voxels: np.ndarray,
        local_s: float,
        stability_steps: int,
        *,
        parent_ids: list[str] | None = None,
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
            parent_ids=list(parent_ids or []),
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

    def compose(
        self,
        obj_a: StableObject,
        obj_b: StableObject,
        *,
        rng: np.random.Generator | None = None,
    ) -> StableObject | None:
        """Compose two objects into a novel building block.

        The composed object blends the subfields of the two parents.
        If the parents have different shapes, the smaller is zero-padded
        to match the larger.  The result is added to the corpus if it
        qualifies.

        Returns the new :class:`StableObject` or ``None``.
        """
        rng = rng or np.random.default_rng()

        # Determine common shape (max of each dimension).
        target_shape = tuple(
            max(a, b) for a, b in zip(obj_a.subfield.shape, obj_b.subfield.shape)
        )

        def _pad_to(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            if arr.shape == shape:
                return arr.copy()
            padded = np.zeros(shape, dtype=arr.dtype)
            slices = tuple(slice(0, s) for s in arr.shape)
            padded[slices] = arr
            return padded

        a_padded = _pad_to(obj_a.subfield, target_shape)
        b_padded = _pad_to(obj_b.subfield, target_shape)

        # Stochastic blend ratio for novelty.
        alpha = float(rng.uniform(0.3, 0.7))
        composed_subfield = alpha * a_padded + (1.0 - alpha) * b_padded

        # Voxels: pick from the parent with higher field density at each voxel.
        va = _pad_to(obj_a.voxels.astype(np.float64), target_shape)
        vb = _pad_to(obj_b.voxels.astype(np.float64), target_shape)
        composed_voxels = np.where(a_padded >= b_padded, va, vb).astype(int)

        composed_s = alpha * obj_a.avg_s + (1.0 - alpha) * obj_b.avg_s
        composed_stability = min(obj_a.stability_steps, obj_b.stability_steps)
        parent_ids = [obj_a.object_id, obj_b.object_id]

        return self.add_if_stable(
            composed_subfield,
            composed_voxels,
            composed_s,
            composed_stability,
            parent_ids=parent_ids,
        )

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
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire corpus to a JSON-compatible dictionary."""
        return {
            "max_size": self.max_size,
            "min_stability": self.min_stability,
            "min_local_s": self.min_local_s,
            "patch_scales": self.patch_scales,
            "compose_probability": self.compose_probability,
            "objects": [obj.to_dict() for obj in self._objects],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryCorpus:
        """Reconstruct a :class:`MemoryCorpus` from a dictionary."""
        corpus = cls(
            max_size=int(data["max_size"]),
            min_stability=int(data["min_stability"]),
            min_local_s=float(data["min_local_s"]),
            patch_scales=list(data.get("patch_scales", [4, 8, 16])),
            compose_probability=float(data.get("compose_probability", 0.15)),
        )
        for obj_data in data.get("objects", []):
            corpus._objects.append(StableObject.from_dict(obj_data))
        return corpus

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score(obj: StableObject) -> float:
        """Score an object for selection/eviction.  Higher is better."""
        return obj.avg_s * (1.0 + 0.1 * obj.stability_steps) * (1.0 + 0.05 * obj.usage_count)
