"""Chunk-based processing for large worlds.

The world is divided into cubic chunks (default 32³).  Only chunks
containing non-Void voxels or active agents are considered *active*.
The simulation step can query the :class:`ChunkManager` for active
bounding boxes to skip empty regions.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


class ChunkManager:
    """Tracks which cubic chunks of the world are active."""

    def __init__(self, world_shape: tuple[int, int, int], chunk_edge: int = 32) -> None:
        if chunk_edge <= 0:
            raise ValueError("chunk_edge must be positive")
        self.world_shape = world_shape
        self.chunk_edge = chunk_edge
        # Number of chunks along each axis (ceiling division).
        self.grid_shape = tuple(
            (s + chunk_edge - 1) // chunk_edge for s in world_shape
        )
        # Active flags: True means the chunk needs processing.
        self._active = np.ones(self.grid_shape, dtype=bool)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def mark_all_active(self) -> None:
        """Mark every chunk as active (e.g. after a full-world event)."""
        self._active[:] = True

    def update_active_mask(
        self,
        field: np.ndarray,
        void_threshold: float,
        agent_positions: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Recompute which chunks are active.

        A chunk is active if it contains at least one non-Void voxel or
        at least one agent.
        """
        ce = self.chunk_edge
        gx, gy, gz = self.grid_shape

        self._active[:] = False

        for ci in range(gx):
            x0 = ci * ce
            x1 = min(x0 + ce, self.world_shape[0])
            for cj in range(gy):
                y0 = cj * ce
                y1 = min(y0 + ce, self.world_shape[1])
                for ck in range(gz):
                    z0 = ck * ce
                    z1 = min(z0 + ce, self.world_shape[2])
                    chunk_data = field[x0:x1, y0:y1, z0:z1]
                    if np.any(chunk_data >= void_threshold):
                        self._active[ci, cj, ck] = True

        if agent_positions:
            for pos in agent_positions:
                ci = min(pos[0] // ce, gx - 1)
                cj = min(pos[1] // ce, gy - 1)
                ck = min(pos[2] // ce, gz - 1)
                self._active[ci, cj, ck] = True

    def active_bounding_boxes(
        self,
    ) -> Iterator[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        """Yield ``(start, end)`` slices for every active chunk."""
        ce = self.chunk_edge
        it = np.ndindex(*self.grid_shape)
        for idx in it:
            if self._active[idx]:
                start = tuple(i * ce for i in idx)
                end = tuple(
                    min(i * ce + ce, s) for i, s in zip(idx, self.world_shape)
                )
                yield start, end  # type: ignore[misc]

    def get_chunk_data(
        self,
        field: np.ndarray,
        cx: int,
        cy: int,
        cz: int,
    ) -> np.ndarray:
        """Return the voxel data for the chunk at grid indices (cx, cy, cz)."""
        ce = self.chunk_edge
        x0 = cx * ce
        x1 = min(x0 + ce, self.world_shape[0])
        y0 = cy * ce
        y1 = min(y0 + ce, self.world_shape[1])
        z0 = cz * ce
        z1 = min(z0 + ce, self.world_shape[2])
        return field[x0:x1, y0:y1, z0:z1].copy()

    def is_active(self, cx: int, cy: int, cz: int) -> bool:
        return bool(self._active[cx, cy, cz])

    @property
    def active_count(self) -> int:
        return int(self._active.sum())

    @property
    def total_chunks(self) -> int:
        return int(np.prod(self.grid_shape))
