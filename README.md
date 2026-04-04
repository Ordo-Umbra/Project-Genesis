# Project Genesis: A URP-Driven Terrain Sandbox

Project Genesis is an open-source attempt to turn the Universal Recursion Principle (URP) and Recursive Complexity Model (RCM) into a persistent procedural world.

The repository is now centered on a first concrete milestone: a **URP terrain sandbox**. Instead of jumping directly to MMO networking or autonomous inhabitants, the current code focuses on making the field simulation deterministic, inspectable, and testable.

## Current MVP Scope

The sandbox currently provides:

- a configurable 3D scalar field evolved with the prototype URP-inspired update rule,
- deterministic seeding for repeatable terrain runs,
- voxel sectorization into air, soil, and stone bands,
- saved snapshots for resuming or analyzing a run,
- exported metrics and text slices for inspecting intermediate and final terrain states,
- automated checks for repeatability, stability, persistence, and parameter sensitivity.

This keeps the project grounded in observable outputs while the theoretical model continues to mature.

## Repository Layout

```text
project_genesis/
  config.py      Engine configuration and defaults
  engine.py      Field evolution, voxel quantization, save/load
  io.py          Snapshot serialization helpers
  metrics.py     URP terrain summary metrics
  render.py      Text-based slice rendering for terrain inspection
Docs/
  The Universal Recursion Principle (URP) _260312_170343.txt
tests/
  test_genesis_engine.py
genesis_engine.py
requirements.txt
```

## Architecture Overview

The current simulation loop is intentionally small:

1. Initialize a cubic scalar field with seeded primordial noise.
2. Evolve the field using diffusion, a complexity term, and a damping/gravity term.
3. Quantize the resulting field into voxel sectors.
4. Record metrics and center-slice snapshots during evolution.
5. Export artifacts contributors can inspect or compare between runs.

This structure is meant to support the next stage of work: stronger metrics, richer terrain interpretation, and eventually simple agents operating against the same field.

## Setup

Create a Python environment and install the declared runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Sandbox

Run a deterministic sandbox simulation and export inspectable artifacts:

```bash
python genesis_engine.py --chunk-size 24 --steps 40 --dt 0.01 --seed 7 --record-every 5 --output-dir artifacts/run_seed_7
```

This writes:

- `config.json`
- `final_metrics.json`
- `metrics_history.json`
- `final_slice_z.txt`
- `slices/step_XXXX_z.txt`
- `engine_snapshot.npz`

You can resume from a saved state:

```bash
python genesis_engine.py --resume artifacts/run_seed_7/engine_snapshot.npz --steps 10 --output-dir artifacts/resumed_run
```

## Validation

Run the automated validation suite with:

```bash
python -m unittest discover -s tests
```

The current checks verify:

- deterministic repeatability for identical seeds,
- save/load round-trip integrity,
- finite evolved fields with multiple voxel classes,
- sensitivity to URP parameter changes.

## What Exists Now

- A working terrain prototype based on the existing URP field concept.
- A modular Python package instead of a single experimental script.
- Basic artifact export so contributors can inspect runs without adding graphics dependencies.
- A small validation layer to keep iteration grounded in measurable behavior.

## What Comes Next

Recommended next steps for expansion:

1. strengthen the S-functional metrics and calibration logic,
2. add richer terrain classification beyond three voxel sectors,
3. introduce a minimal single-agent inhabitant that senses local terrain metrics,
4. evaluate whether the simulation loop is compelling enough to justify networking and avatars.

## Theory Reference

The foundational theory document remains in:

- `/home/runner/work/Project-Genesis/Project-Genesis/Docs/The Universal Recursion Principle (URP) _260312_170343.txt`

That document describes the broader URP framing this sandbox is intended to explore in executable form.
