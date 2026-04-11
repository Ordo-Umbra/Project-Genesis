#!/usr/bin/env python3
"""Benchmark: measure steps-per-second with and without Numba JIT.

Usage::

    python benchmarks/bench_field_step.py [--size 32] [--steps 100]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure repo root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from project_genesis import EngineConfig, GenesisEngine  # noqa: E402


def bench(size: int, steps: int) -> None:
    config = EngineConfig(chunk_size=size, seed=42, default_dt=0.01)
    engine = GenesisEngine(config=config)

    # Warm up JIT (first call compiles).
    engine.step(0.01)
    engine.field = engine.rng.random((size, size, size))

    t0 = time.perf_counter()
    for _ in range(steps):
        engine.step(0.01)
    elapsed = time.perf_counter() - t0

    sps = steps / elapsed
    print(f"World {size}³  |  {steps} steps in {elapsed:.3f}s  |  {sps:.1f} steps/s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    bench(args.size, args.steps)


if __name__ == "__main__":
    main()
