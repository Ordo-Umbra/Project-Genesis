# Project Genesis: A URP-Driven Terrain Sandbox

Project Genesis is an open-source attempt to turn the Universal Recursion Principle (URP) into a persistent procedural world.

The repository is centered on a first concrete milestone: a **URP terrain sandbox**. The code focuses on making the field simulation deterministic, inspectable, and testable — while progressively layering on the deeper structures described in the URP theory.

## Current MVP Scope

The sandbox currently provides:

- a configurable 3D scalar field evolved with the prototype URP-inspired update rule,
- **optional full URP coherence potential** V(x,t) satisfying ∇²V = ρ, replacing simple gravity damping with the proper G·∇V·∇φ coherence advection term,
- **optional nonlocal integration functional** I[φ] using exponential-decay correlation kernels,
- deterministic seeding for repeatable terrain runs,
- five-band voxel sectorization: **void**, **air**, **soil**, **stone**, and **bedrock**,
- **S-functional tracking** — per-step computation of ΔC (distinction), ΔI (integration), κ (capacity), and S = ΔC + κΔI,
- **multi-agent terrain-sensing inhabitants** with configurable density-seeking, exploration, or S-functional-driven policies,
- agent-agent sensing, shared best-known signals, and optional field influence at visited cells,
- **stable-structure memory corpus** with multi-scale patch scanning, bounded corpus retention, probabilistic recall, compositional injection, and lineage tracking,
- saved snapshots for resuming or analyzing a run,
- exported metrics, run summaries, agent timelines, corpus summaries, and text slices for inspecting intermediate and final terrain states,
- **matplotlib visualization** — 3-D voxel scatter plots, field cross-section heat maps, and S-functional time-series charts,
- automated checks for repeatability, stability, persistence, parameter sensitivity, agent behavior, memory-corpus serialization and recall, artifacts, CLI flows, physics correctness, and visualization output.

## Repository Layout

```text
project_genesis/
  __init__.py          Package exports
  agent.py             Terrain-sensing agent with perception and action queue
  chunk_manager.py     Chunk-based world partitioning for active-region tracking
  config.py            Engine configuration and defaults
  engine.py            Field evolution, voxel quantization, agent orchestration, save/load
  io.py                Snapshot serialization helpers
  metrics.py           URP terrain summary metrics and S-functional computation
  memory_corpus.py     Stable-object corpus, composition, serialization, lineage
  network_server.py    WebSocket server for remote monitoring and control
  numba_kernels.py     Numba JIT-accelerated field evolution kernels
  render.py            Text-based slice rendering for terrain inspection
  s_compass_bridge.py  S-compass connector bridge for AI agent integration
  visualize.py         Matplotlib-based 3-D voxel and S-functional visualization
Docs/
  The Universal Recursion Principle (URP) _260312_170343.txt
tests/
  test_genesis_engine.py
  test_memory_corpus.py
  test_new_subsystems.py
  test_urp_extensions.py
web_viewer/
  index.html           Three.js live voxel viewer
  client.js            WebSocket client for the viewer
benchmarks/
  bench_field_step.py   Steps-per-second benchmark
genesis_engine.py       CLI entry point
run_server.py           Headless simulation server entry point
requirements.txt
```

## Architecture Overview

The simulation loop:

1. Initialize a cubic scalar field with seeded primordial noise.
2. Evolve the field using diffusion, a complexity term (β|∇φ|²), and either the simplified gravity damping (G·φ) or the full URP coherence potential (G·∇V·∇φ where ∇²V = ρ).
3. Optionally compute the nonlocal integration functional I[φ] = ∫∫ K(x,x')φ(x)φ(x') dx dx' using exponential-decay correlation kernels.
4. At each step, compute the **S-functional** — tracking how structural differentiation (ΔC) and coherent integration (ΔI) evolve under capacity constraints (κ).
5. Agents sense their local neighborhood and move through the field each step.
6. Quantize the resulting field into five voxel sectors.
7. Optionally update a stability map, scan multi-scale stable patches into the memory corpus, and probabilistically re-inject recalled or composed structures.
8. Record metrics (including S-functional components, agent states, and corpus summaries) and center-slice snapshots.
9. Share local best signals across agents, apply optional agent field influence, and export inspectable artifacts.

### S-Functional

The S-functional implements the core URP equation **S = ΔC + κΔI**:

| Component | Implementation |
|-----------|---------------|
| **ΔC** (distinction) | Mean of β\|∇φ\|² — structural gradient energy |
| **κ** (capacity) | 1 / (1 + mean(\|∇φ\|²)) — high gradients suppress integration |
| **ΔI** (integration) | Reduction in mean absolute Laplacian between steps (smoothing = integration) |
| **S** | ΔC + κ · ΔI |

### Coherence Potential V(x,t)

The full URP field equation replaces the simple G·φ damping term with a coherence advection term G·∇V·∇φ, where V is a potential satisfying the Poisson equation ∇²V = ρ (with ρ proportional to the field φ). This models gravitational-like coherence forces that drive the field toward configurations maximizing mutual information across boundaries.

Enable with `--coherence-potential`. The Poisson equation is solved iteratively using Numba-accelerated Jacobi relaxation with periodic boundary conditions. Control the solver precision with `--poisson-iterations` (default: 30).

### Nonlocal Integration Functional I[φ]

The integration functional I[φ] = ∫∫ K(x,x')φ(x)φ(x') dx dx' captures nonlocal correlations in the field using an exponential-decay kernel K(x,x') = exp(-decay·|x-x'|). Its functional derivative δI/δφ enters the field equation as an additional driving term that rewards configurations where nearby regions share coherent structure.

Enable with `--integration-functional`. Configure with `--integration-radius` (default: 2), `--integration-decay` (default: 1.0), and `--integration-weight` (default: 0.01).

### Voxel Sectors

The field is quantized into five terrain bands:

| ID | Name | Symbol | Condition |
|----|------|--------|-----------|
| 0 | Void | ` ` | field < `void_threshold` (default: 0.15) |
| 1 | Air | `.` | `void_threshold` ≤ field < `air_threshold` (default: 0.30) |
| 2 | Soil | `+` | `air_threshold` ≤ field < `soil_threshold` (default: 0.60) |
| 3 | Stone | `#` | `soil_threshold` ≤ field < `bedrock_threshold` (default: 0.80) |
| 4 | Bedrock | `@` | field ≥ `bedrock_threshold` (default: 0.80) |

### Agents

Agents are configurable inhabitants that:
- **Sense** the 6-connected neighborhood at their position (local value, neighbor stats, gradient, local S-signal proxy).
- **Detect peers** within a configurable interaction radius and include nearby-agent data in their sensor log.
- **Move** according to one of three policies: density-seeking, exploration-biased novelty, or local S-functional proxy maximization.
- **Share** best-known local signals each step so the agent population can loosely coordinate.
- **Influence** the field after moving when `agent_influence` is enabled.
- **Log** full trails and sensor readings across steps for analysis or resume.

### Stable-Structure Memory Corpus

When `--enable-memory-corpus` is set, the engine maintains a bounded library of locally stable 3-D patches:

- scans the field at configurable patch scales (`--corpus-patch-scales`, default `4,8,16`),
- stores patches that meet local stability and local S thresholds,
- persists the corpus and stability map inside `engine_snapshot.npz`,
- samples stored objects for probabilistic recall during evolution,
- optionally composes two recalled objects into a novel patch before injection (`--corpus-compose-probability`),
- tracks lineage through `parent_ids` so composed objects remain inspectable.

The feature is tuned with:

- `--corpus-max-size`
- `--min-stability`
- `--min-local-s`
- `--corpus-patch-scales`
- `--corpus-compose-probability`

## Setup

Create a Python environment and install the declared runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

You can also install the package with its console entry point:

```bash
python -m pip install .
project-genesis --help
```

## Running the Sandbox

Run a deterministic sandbox simulation and export inspectable artifacts:

```bash
python genesis_engine.py --chunk-size 24 --steps 40 --dt 0.01 --seed 7 --record-every 5 --agent-count 4 --agent-goal s_functional --output-dir artifacts/run_seed_7
```

This writes:

- `config.json`
- `final_metrics.json` (includes S-functional components: `delta_c`, `delta_i`, `kappa`, `s_increment`)
- `metrics_history.json`
- `run_summary.json`
- `agent_timelines.json`
- `final_slice_z.txt`
- `final_slices/final_slice_x.txt`
- `final_slices/final_slice_y.txt`
- `final_slices/final_slice_z.txt`
- `slices/step_XXXX_z.txt`
- `engine_snapshot.npz`

### Running with Full URP Physics

Enable the coherence potential and integration functional for the complete URP field equation:

```bash
python genesis_engine.py --chunk-size 24 --steps 40 --dt 0.01 --seed 7 --coherence-potential --integration-functional --visualize --output-dir artifacts/full_urp_run
```

This adds the full ∂_t φ = ∇²φ + β|∇φ|² + G·∇V·∇φ + w_I·δI/δφ evolution, and generates matplotlib visualizations:

- `voxel_3d.png` — 3-D scatter plot of the voxel terrain
- `field_slices.png` — Centre-slice heat maps along x, y, z axes
- `s_history.png` — S-functional component time series

You can resume from a saved state:

```bash
python genesis_engine.py --resume artifacts/run_seed_7/engine_snapshot.npz --steps 10 --output-dir artifacts/resumed_run
```

### Running with Memory Corpus Recall

Enable the stable-structure memory system and multi-scale patch scanning:

```bash
python genesis_engine.py --chunk-size 24 --steps 40 --dt 0.01 --seed 7 --enable-memory-corpus --corpus-max-size 64 --min-stability 4 --min-local-s 0.01 --corpus-patch-scales 4,8,16 --corpus-compose-probability 0.2 --output-dir artifacts/memory_corpus_run
```

When enabled, `final_metrics.json`, `run_summary.json`, and WebSocket world summaries include corpus metrics such as `corpus_size`, `corpus_mean_s`, `corpus_total_usage`, `corpus_mean_stability`, and `corpus_composed_count`.

## Using Agents

Agents can be spawned programmatically:

```python
from project_genesis import EngineConfig, GenesisEngine

engine = GenesisEngine(
    config=EngineConfig(
        chunk_size=24,
        seed=7,
        agent_count=3,
        agent_goal="explore",
        agent_influence=0.05,
    )
)
agent = engine.add_agent(position=(12, 12, 12), goal="s_functional")
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
- agent sensing returns correct local field values and peer-awareness data,
- agents move and accumulate trails during evolution,
- agent state appears in engine metrics snapshots,
- multi-agent config is applied automatically,
- artifact export writes structured summaries and timelines,
- CLI-driven multi-agent runs produce complete outputs.
- memory corpus objects serialize / deserialize cleanly with lineage metadata,
- engine save / load preserves corpus contents and the stability map,
- multi-scale corpus scanning and compositional injection execute without breaking evolution,
- chunk activation / deactivation logic,
- WebSocket message serialization / deserialization,
- S-compass bridge output consistency,
- headless save / load round-trip integrity,
- agent perception data structure and action queue execution.

## Headless Server Mode

Run the simulation as a persistent headless server with auto-save and optional WebSocket API:

```bash
python run_server.py --world-size 64 --save-interval 100 --port 8765 --agent-count 4
```

The server:

- Runs the simulation loop indefinitely (or up to ``--max-steps``).
- Auto-saves compressed snapshots every ``--save-interval`` steps.
- Traps **SIGINT** / **SIGTERM** for graceful shutdown with a final save.
- Optionally starts a WebSocket server (disable with ``--port 0``).
- Accepts a ``--config`` JSON file for full configuration control.
- Supports ``--resume`` to continue from a saved snapshot.

## WebSocket API

When the headless server runs with a non-zero port, the following commands are available over WebSocket (JSON messages):

| Command | Payload | Response |
|---------|---------|----------|
| `get_state` | — | World dimensions, step count, S-functional, agent positions, chunk info, optional memory-corpus summary |
| `get_chunk` | `{x, y, z}` | Binary voxel data for the requested chunk |
| `get_agent_view` | `{agent_id}` | Full perception dict for the specified agent |
| `send_action` | `{agent_id, action}` | Queues an action for an agent; acknowledged |

The server also pushes `chunk_updated` events to connected clients when voxel data changes.

## Web Viewer

Open `web_viewer/index.html` in a browser while the headless server is running. The viewer:

- Connects to the WebSocket server automatically.
- Renders voxels using Three.js with semi-transparent band materials.
- Displays a live S-functional graph using Chart.js.
- Provides play/pause and speed controls.
- Receives incremental chunk update notifications.

## Perception-Action Interface

Agents now expose a structured perception interface for external AI controllers:

```python
perception = agent.get_perception(engine.field, agents=engine.agents, beta=engine.BETA)
# Returns: scalar_field, s_field, nearby_agents, energy, position, agent_id
```

External actions can be queued via the WebSocket API or programmatically:

```python
engine.queue_agent_action("agent-0", {"type": "move", "direction": [1, 0, 0]})
```

### S-Compass Bridge

The `s_compass_bridge` module computes a recommended action vector from perception data:

```python
from project_genesis.s_compass_bridge import perception_to_action

action = perception_to_action(perception, beta=0.09)
# Returns: {"type": "move", "direction": [dx, dy, dz]}
```

## Numba JIT Acceleration

Field evolution now uses Numba-compiled kernels (`numba_kernels.py`) for the Laplacian, gradient, evolution, Poisson solver, gradient dot product, and correlation kernel steps. The kernels use `@njit(parallel=True)` with `prange` for multi-core parallelism. Run the benchmark:

```bash
python benchmarks/bench_field_step.py --size 64 --steps 200
```

## Matplotlib Visualization

Generate publication-quality plots of the terrain and S-functional evolution:

```python
from project_genesis import GenesisEngine, EngineConfig
from project_genesis.visualize import render_voxels_3d, render_field_slices, plot_s_history, save_visualization

engine = GenesisEngine(config=EngineConfig(chunk_size=24, seed=7))
engine.evolve_field(steps=40, dt=0.01, record_every=5)

# 3-D voxel scatter plot
fig = render_voxels_3d(engine.quantize_to_voxels())
fig.savefig("terrain.png", dpi=150)

# Field cross-sections
fig2 = render_field_slices(engine.field)
fig2.savefig("slices.png", dpi=150)

# S-functional time series
fig3 = plot_s_history(engine.history)
fig3.savefig("s_history.png", dpi=150)

# Or generate all at once:
save_visualization("output/", engine.quantize_to_voxels(), engine.field, engine.history)
```

## Chunk-Based Processing

The `ChunkManager` divides the world into cubic chunks and tracks which contain non-Void voxels or active agents. Only active chunks are processed, improving performance for sparse worlds.

## What Exists Now

- A working terrain prototype based on the URP field equation with S-functional tracking.
- **Full URP coherence potential** V(x,t) satisfying ∇²V = ρ, with Jacobi Poisson solver, replacing simple G·φ damping with G·∇V·∇φ from the complete URP field equation.
- **Nonlocal integration functional** I[φ] using exponential-decay correlation kernels K(x,x')φ(x)φ(x'), adding coherent-integration driving forces to the field evolution.
- **Numba JIT-accelerated** field evolution kernels with parallel stencil operations (including the new Poisson solver, gradient dot product, and correlation kernel).
- **Chunk-based processing** for efficient handling of large, sparse worlds.
- **Stable-structure memory corpus** with persistence, multi-scale scanning, probabilistic recall, compositional injection, and lineage tracking.
- **S-functional caching** to avoid redundant computation between steps.
- Five-band voxel sectorization (void, air, soil, stone, bedrock) for richer terrain structure.
- Per-step S-functional computation (ΔC, ΔI, κ, S) connecting the simulation to URP theory.
- Goal-driven multi-agent inhabitants with peer sensing and optional field influence.
- **Perception-action interface** for external AI agent control via structured perception dicts and action queues.
- **S-compass bridge** for computing recommended actions from local S-functional gradients.
- **Headless server mode** with auto-save, graceful shutdown, and command-line configuration.
- **WebSocket API** for remote monitoring, chunk inspection, agent perception, and action dispatch.
- **Three.js web viewer** with live voxel rendering, S-functional charting, and play/pause controls.
- **Matplotlib visualization** — 3-D voxel scatter plots, field cross-section heat maps, and S-functional time-series charts (via `--visualize` CLI flag or programmatic API).
- A modular Python package with clean separation of concerns.
- Structured artifact export so contributors can inspect runs without graphics dependencies.
- An installable console entry point for repeatable sandbox runs.
- A validation layer covering repeatability, persistence, sensitivity, artifacts, CLI flows, agent behavior, chunk management, WebSocket serialization, S-compass consistency, coherence potential correctness, integration functional output, and visualization output.
- **Performance benchmarks** for measuring steps-per-second.

## What Comes Next

Recommended next steps for expansion:

1. ~~Implement a richer coherence potential V(x,t) satisfying ∇²V = ρ, replacing the simple G·φ damping with G·∇V·∇φ from the full URP field equation.~~ ✅ Implemented
2. ~~Add the nonlocal integration functional I[φ] using correlation kernels K(x,x')φ(x)φ(x').~~ ✅ Implemented
3. ~~Introduce agent-agent interaction — multiple agents that can sense each other and cooperate.~~ ✅ Implemented
4. ~~Add agent goal-seeking behavior driven by the S-functional (agents that maximize local S).~~ ✅ Implemented
5. ~~Richer visualization — matplotlib or VTK-based 3D voxel rendering.~~ ✅ Implemented (matplotlib)
6. Evaluate whether the simulation loop is compelling enough to justify networking and avatars.
7. Add higher-order field dynamics — second-order time derivatives (∂²φ/∂t²) for wave-like behavior.
8. Extend the Poisson solver to support anisotropic or spatially varying ρ (e.g., agent-driven source terms).
9. Explore emergent gauge sectorization — identify conditions under which the field spontaneously partitions into distinct coherent domains.
10. Implement inter-agent communication protocols — agents that negotiate and share structured messages beyond simple signal sharing.

## Theory Reference

The foundational theory document remains in:

- `Docs/The Universal Recursion Principle (URP) _260312_170343.txt`

That document describes the broader URP framing this sandbox is intended to explore in executable form.
