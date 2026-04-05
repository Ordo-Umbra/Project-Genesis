# Project Genesis: A URP-Driven Terrain Sandbox

Project Genesis is an open-source attempt to turn the Universal Recursion Principle (URP) and Recursive Complexity Model (RCM) into a persistent procedural world.

The repository is centered on a first concrete milestone: a **URP terrain sandbox**. The code focuses on making the field simulation deterministic, inspectable, and testable — while progressively layering on the deeper structures described in the URP theory.

## Current MVP Scope

The sandbox currently provides:

- a configurable 3D scalar field evolved with the prototype URP-inspired update rule,
- deterministic seeding for repeatable terrain runs,
- five-band voxel sectorization: **void**, **air**, **soil**, **stone**, and **bedrock**,
- **S-functional tracking** — per-step computation of ΔC (distinction), ΔI (integration), κ (capacity), and S = ΔC + κΔI,
- **terrain-sensing agents** that perceive the local field and move via gradient-ascent with stochastic exploration,
- saved snapshots for resuming or analyzing a run,
- exported metrics and text slices for inspecting intermediate and final terrain states,
- automated checks for repeatability, stability, persistence, parameter sensitivity, and agent behavior.

## Repository Layout

```text
project_genesis/
  __init__.py    Package exports
  agent.py       Minimal terrain-sensing agent
  config.py      Engine configuration and defaults
  engine.py      Field evolution, voxel quantization, agent orchestration, save/load
  io.py          Snapshot serialization helpers
  metrics.py     URP terrain summary metrics and S-functional computation
  render.py      Text-based slice rendering for terrain inspection
Docs/
  The Universal Recursion Principle (URP) _260312_170343.txt
tests/
  test_genesis_engine.py
genesis_engine.py
requirements.txt
```

## Architecture Overview

The simulation loop:

1. Initialize a cubic scalar field with seeded primordial noise.
2. Evolve the field using diffusion, a complexity term (β|∇φ|²), and a gravity/damping term (G·φ).
3. At each step, compute the **S-functional** — tracking how structural differentiation (ΔC) and coherent integration (ΔI) evolve under capacity constraints (κ).
4. Agents sense their local neighborhood and move through the field each step.
5. Quantize the resulting field into five voxel sectors.
6. Record metrics (including S-functional components and agent states) and center-slice snapshots.
7. Export artifacts contributors can inspect or compare between runs.

### S-Functional

The S-functional implements the core URP equation **S = ΔC + κΔI**:

| Component | Implementation |
|-----------|---------------|
| **ΔC** (distinction) | Mean of β\|∇φ\|² — structural gradient energy |
| **κ** (capacity) | 1 / (1 + mean(\|∇φ\|²)) — high gradients suppress integration |
| **ΔI** (integration) | Reduction in mean absolute Laplacian between steps (smoothing = integration) |
| **S** | ΔC + κ · ΔI |

### Voxel Sectors

The field is quantized into five terrain bands:

| ID | Name | Symbol | Condition |
|----|------|--------|-----------|
| 0 | Void | ` ` | field < 0.15 |
| 1 | Air | `.` | 0.15 ≤ field < 0.30 |
| 2 | Soil | `+` | 0.30 ≤ field < 0.60 |
| 3 | Stone | `#` | 0.60 ≤ field < 0.80 |
| 4 | Bedrock | `@` | field ≥ 0.80 |

### Agents

Agents are minimal inhabitants that:
- **Sense** the 6-connected neighborhood at their position (local value, neighbor stats, gradient).
- **Move** via gradient ascent toward denser terrain (80%) or random exploration (20%).
- **Log** their trail and sensor readings across steps for analysis.

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
- `final_metrics.json` (includes S-functional components: `delta_c`, `delta_i`, `kappa`, `s_increment`)
- `metrics_history.json`
- `final_slice_z.txt`
- `slices/step_XXXX_z.txt`
- `engine_snapshot.npz`

You can resume from a saved state:

```bash
python genesis_engine.py --resume artifacts/run_seed_7/engine_snapshot.npz --steps 10 --output-dir artifacts/resumed_run
```

## Using Agents

Agents can be spawned programmatically:

```python
from project_genesis import EngineConfig, GenesisEngine

engine = GenesisEngine(config=EngineConfig(chunk_size=24, seed=7))
agent = engine.add_agent(position=(12, 12, 12))
engine.evolve_field(steps=40, dt=0.01)

# Inspect agent trail and sensor log
print(f"Agent visited {len(agent.trail)} positions")
print(f"Last reading: {agent.sense_log[-1]}")
```

## Validation

Run the automated validation suite with:

```bash
python -m unittest discover -s tests
```

The current checks verify:

- deterministic repeatability for identical seeds,
- save/load round-trip integrity,
- finite evolved fields with multiple voxel classes (across all five sectors),
- sensitivity to URP parameter changes,
- agent sensing returns correct local field values,
- agents move and accumulate trails during evolution,
- agent state appears in engine metrics snapshots.

## What Exists Now

- A working terrain prototype based on the URP field equation with S-functional tracking.
- Five-band voxel sectorization (void, air, soil, stone, bedrock) for richer terrain structure.
- Per-step S-functional computation (ΔC, ΔI, κ, S) connecting the simulation to URP theory.
- Minimal terrain-sensing agents that inhabit and explore the world.
- A modular Python package with clean separation of concerns.
- Artifact export so contributors can inspect runs without graphics dependencies.
- A validation layer covering repeatability, persistence, sensitivity, and agent behavior.

## What Comes Next

Recommended next steps for expansion:

1. Implement a richer coherence potential V(x,t) satisfying ∇²V = ρ, replacing the simple G·φ damping with G·∇V·∇φ from the full URP field equation.
2. Add the nonlocal integration functional I[φ] using correlation kernels K(x,x')φ(x)φ(x').
3. Introduce agent-agent interaction — multiple agents that can sense each other and cooperate.
4. Add agent goal-seeking behavior driven by the S-functional (agents that maximize local S).
5. Richer visualization — matplotlib or VTK-based 3D voxel rendering.
6. Evaluate whether the simulation loop is compelling enough to justify networking and avatars.

## Theory Reference

The foundational theory document remains in:

- `Docs/The Universal Recursion Principle (URP) _260312_170343.txt`

That document describes the broader URP framing this sandbox is intended to explore in executable form.
