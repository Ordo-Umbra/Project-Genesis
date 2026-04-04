from pathlib import Path

import numpy as np

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

    def calculate_S_gradients(self) -> tuple[np.ndarray, np.ndarray]:
        return calculate_gradients(self.field)

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
            absolute_step = start_step + step
            if step % record_every == 0 or step == total_steps:
                snapshot = self.summarize_state(step=absolute_step, prev_field=self.prev_field)
                snapshot["slice_z"] = render_voxel_slice(self.quantize_to_voxels(), axis="z")
                self.history.append(snapshot)

        return self.field

    def quantize_to_voxels(self) -> np.ndarray:
        voxel_chunk = np.zeros_like(self.field, dtype=int)
        voxel_chunk[self.field < self.config.air_threshold] = 0
        voxel_chunk[(self.field >= self.config.air_threshold) & (self.field < self.config.soil_threshold)] = 1
        voxel_chunk[self.field >= self.config.soil_threshold] = 2
        return voxel_chunk

    def summarize_state(
        self,
        *,
        step: int | None = None,
        prev_field: np.ndarray | None = None,
    ) -> dict[str, float | int]:
        return summarize_field(
            self.field,
            self.quantize_to_voxels(),
            self.BETA,
            self.G,
            step=step,
            prev_field=prev_field,
        )

    def save(self, path: str | Path) -> Path:
        return save_snapshot(path, self.field, self.config, self.history)

    @classmethod
    def load(cls, path: str | Path) -> "GenesisEngine":
        field, config, history = load_snapshot(path)
        return cls(config=config, field=field, history=history)
