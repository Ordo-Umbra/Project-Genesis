# Project Genesis: An Executable Testbench for the Universal Recursion Principle

Project Genesis turns the **Universal Recursion Principle (URP)** from prose and equations into **running, falsifiable code** — a simulation laboratory where the theory's central claims can be built, measured, and given honest verdicts.

## What this project is (and is not)

The URP proposes that sufficiently expressive recursive systems — physical, biological, cognitive — evolve by climbing a single scalar functional

```
S = ΔC + κΔI
```

balancing the growth of **distinction** (ΔC — making differences, articulating structure) against **integration** (ΔI — binding those differences into a coherent whole), under a finite **capacity** (κ — the resources available to sustain integration). From that one principle the theory derives a striking range of claims: that the field spontaneously partitions into exactly three sectors (the seed of colour **SU(3)**), that capacity-driven phase transitions break symmetries, that stable structures act as reusable "seeds" planted across scales.

**The goal of this repository is not to prove the theory.** It is to make the theory *executable* — to build the smallest honest simulation that implements the URP field dynamics and the S-functional, and then a set of **instruments** that measure what actually emerges, so each claim becomes a question with a real answer instead of a quotation. Following the framing of the theory's own companion essay *The Range*, the project holds itself to one test: it must be able to produce a **verdict** — *this is supported, that is not, here the model can't decide* — rather than only appreciation. Where the simulation reproduces a prediction, it says so; where it falls short, it says that too, and names the missing physics.

This is, in that essay's terms, a *map and an instrument* — not, by itself, evidence about the physical world. The theory documents live in [`Docs/`](Docs/); this code is the bench they are tested on.

### How it works, in practice

Each capability below arrived as one turn of the same loop: **state a URP claim → build an instrument to measure it → run it → report the verdict, caveats included.** That loop has already produced concrete results — for example, that wall cost scales linearly with β exactly as the boundary term predicts (`a(β) ≈ 2.6·β`), that a single scalar field provably *cannot* form the theory's three-way SU(3) junctions (a three-component field can), and that sector selection toward `N⋆ = 3` appears to be a **capacity-scarcity phenomenon** — invisible with abundant κ, emerging only when a dynamical capacity budget binds. None of these were assumed; all were measured, and all are reproducible from the experiments and tests in this repo.

## Current Scope

The simulation laboratory currently provides:

- a configurable 3D scalar field evolved with the prototype URP-inspired update rule,
- **optional full URP coherence potential** V(x,t) satisfying ∇²V = ρ, replacing simple gravity damping with the proper G·∇V·∇φ coherence advection term,
- **optional nonlocal integration functional** I[φ] using exponential-decay correlation kernels,
- deterministic seeding for repeatable terrain runs,
- five-band voxel sectorization: **void**, **air**, **soil**, **stone**, and **bedrock**,
- **S-functional tracking** — per-step computation of ΔC (distinction), ΔI (integration), κ (capacity), and S = ΔC + κΔI,
- **multi-agent terrain-sensing inhabitants** with configurable density-seeking, exploration, or S-functional-driven policies,
- agent-agent sensing, shared best-known signals, and optional field influence at visited cells,
- **stable-structure memory corpus** with multi-scale patch scanning, bounded corpus retention, probabilistic recall, compositional injection, and lineage tracking,
- **β-sectorisation / boundary-formation analysis** — domain-wall detection, periodic connected-component sector counting, per-sector distinction/integration statistics, and triple-junction counting, for empirically testing the URP `N⋆=3` (SU(3)) prediction,
- **dynamical capacity field κ(x,t)** — capacity consumed by distinction load, regenerating with slack, diffusing between regions, and gating the integration term in the dynamics, with multi-scale and per-sector capacity reporting,
- **κ-as-soil corpus coupling** — recalled "seeds" only take root where local capacity is sufficient (fertile soil) and consume it when they do, so structure can only re-grow where the field can support it,
- **two zero-dependency browser toys** and a Numba-accelerated 3-D engine, sharing the same dynamics so the visual intuition and the measured physics stay in step,
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
  multiphase.py        Three-component Ψ∈ℂ³ sector field with 120° Y-junctions
  network_server.py    WebSocket server for remote monitoring and control
  numba_kernels.py     Numba JIT-accelerated field evolution kernels
  render.py            Text-based slice rendering for terrain inspection
  s_compass_bridge.py  S-compass connector bridge for AI agent integration
  sectorisation.py     β-sectorisation / boundary-formation domain analysis
  visualize.py         Matplotlib-based 3-D voxel and S-functional visualization
Docs/
  The Universal Recursion Principle (URP) _260312_170343.txt
tests/                 145 checks across the engine, instruments, and physics
  test_genesis_engine.py
  test_corpus_kappa.py
  test_dynamic_kappa.py
  test_memory_corpus.py
  test_multiphase.py
  test_new_subsystems.py
  test_sectorisation.py
  test_urp_extensions.py
experiments/
  beta_sectorisation.py β-sweep measuring emergent sector counts
  n_star_fit.py         Fits the F(N) free-energy coefficients from run data
web_toy/
  index.html           Standalone in-browser URP toy (scalar field)
  su3.html             Three-component SU(3) sector toy with Y-junctions
web_viewer/
  index.html           Three.js live voxel viewer
  client.js            WebSocket client for the viewer
benchmarks/
  bench_field_step.py   Steps-per-second benchmark
genesis_engine.py       CLI entry point
run_server.py           Headless simulation server entry point
.github/workflows/ci.yml CI: runs the full test suite on every push
requirements.txt
```

## Findings so far

A running tally of what the instruments have actually measured — the verdicts, with their caveats. Each links to the section that explains how it was obtained.

| URP claim | Verdict | How |
|-----------|---------|-----|
| The S-functional `S = ΔC + κΔI` governs the dynamics | **Implemented** and faithful to the reduced field equation | [S-Functional](#s-functional) |
| Boundary (wall) cost scales with β | **Supported** — measured `a(β) ≈ 2.6·β` | [N⋆ experiment](#fitting-fn-from-simulation--the-n-experiment) |
| The β-nonlinearity alone makes the field sectorise | **Not supported** — the reduced `β\|∇φ\|²` term smooths to a single sector; a wall-tension term is required | [β-Sectorisation](#β-sectorisation--boundary-formation) |
| Three mutually-adjacent sectors with 120° Y-junctions (colour SU(3)) | **Not from a scalar field** (structurally impossible) — **achieved** with the three-component Ψ∈ℂ³ model | [Three-Component Sector Field](#three-component-sector-field--genuine-su3-y-junctions) |
| Capacity κ drives selection toward `N⋆ = 3` | **Conditional** — invisible with abundant κ; under a binding capacity budget, `k = 3` becomes S-optimal (robust across seeds, one β/size so far) | [Dynamical κ](#dynamical-κ--capacity-as-a-field) |

The honest through-line: the *machinery* of URP sectorisation is reproducible and the boundary-cost half of its free-energy argument is measured; the *selection* of three sectors is looking like a scarcity phenomenon that needs dynamical capacity, and mapping its `(consumption, recovery, β)` phase diagram is the open frontier.

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
- `--corpus-kappa-threshold` and `--corpus-kappa-cost` (the κ-as-soil coupling; see [κ as soil](#κ-as-soil--coupling-capacity-to-the-memory-corpus))

### β-Sectorisation & Boundary Formation

The URP gauge derivation predicts that the β-nonlinearity drives a continuous medium to partition into a small number of stable domains separated by domain walls, with **N⋆ = 3** the dominant attractor at the QCD-derived β ≈ 0.09 — the seed of color SU(3). Separately, the framing of *The Range* holds that a "being" is not a thing inside a boundary but **is** the boundary: the work of maintaining a coherent domain against its surroundings, through the balance of distinction and integration.

The `sectorisation` module makes both measurable. Given an evolved field it:

- computes the periodic gradient magnitude |∇φ| (reusing the simulation's own kernel),
- marks **domain walls** where |∇φ| is high (the boundary set, the locus of boundary-work),
- labels the low-gradient **interiors** into connected components — the *sectors* (candidate "beings"), with optional periodic merging,
- reports per-sector **distinction** (wall gradient energy, β|∇φ|²) and **integration** (internal coherence, 1/(1+mean|∇φ|²)),
- counts **triple junctions** — the discrete analogue of the 120° Y-junctions tied to the three-sector attractor.

Measure it on a run with `--analyze-sectors` (writes `sectorisation.json`), via the engine API `engine.analyze_sectorisation()`, or sweep β with the experiment script:

```bash
python experiments/beta_sectorisation.py --size 32 --steps 200 --seed 7 --trials 3
```

**Empirical finding (baseline dynamics):** with only the reduced `β|∇φ|²` term, the field collapses to a single sector (N = 1) at *every* β, including 0.09 — it does **not** spontaneously sectorise. This is an honest negative result, not a tooling bug (the analyzer correctly recovers N = 3 on fields that genuinely contain three domains; see `tests/test_sectorisation.py`). The cause is structural: the overdamped reduction drops the theory's `−(β/4)(∇φ)⁴` wall-tension term, leaving nothing to hold a domain wall against diffusion.

#### Sectorisation potential (wall tension)

Enabling `--sector-potential` adds the missing wall tension as a periodic multi-well potential `V(φ) = −cos(2π·k·φ)`, whose `k` minima (`--sector-count`, default 3) give the field domains to settle into:

```
∂_t φ = ∇²φ + β|∇φ|² − G·φ − w · sin(2π·k·φ)
```

(The sine drift is a *pinning stand-in* for the Lagrangian's true `−(β/4)(∇φ)⁴` wall-tension term, which is numerically stiff under explicit time-stepping; the stand-in supplies the same qualitative ingredient — an energetic preference for discrete field levels separated by walls — in a form the existing integrator handles stably.)

With this term the behavior **flips from N = 1 to genuine multi-domain phase separation** — boundary formation now occurs and is measurable:

| dynamics | β = 0.09 sector count |
|----------|----------------------|
| baseline (`β\|∇φ\|²` only) | **1** (no walls) |
| `--sector-potential --sector-count 3` | **many** domains, then coarsening |

Two effects visible in the sweep match the theory's distinction/integration tension: sector count **grows with β** (more distinction → more walls → finer fragmentation), and **falls with evolution time** as Allen–Cahn-style coarsening absorbs small domains. Settling reproducibly onto the predicted `N⋆ = 3` attractor is a genuine coarsening/tuning study (sensitive to initialization, well count, and potential strength) — the analyzer and the [browser toy](web_toy/) are the instruments built to explore it, rather than assuming the answer.

#### Dynamical κ — capacity as a field

In the theory, κ is not a diagnostic but the protagonist: a local, dynamic constraint on how much integration the system can sustain, "computed as a function of the system's current load … and the accumulated stress of past integrations" (capacity assessment, Phase 2 of the update cycle). `--dynamic-kappa` promotes κ from the recorded proxy `1/(1+mean|∇φ|²)` to a co-evolving field that feeds back into the dynamics:

```
∂_t φ = κ(x)·∇²φ + β|∇φ|² − G·φ          κ gates the integrating term
∂_t κ = D_κ·∇²κ + r·(κ₀ − κ) − c·|∇φ|²·κ  diffusion + recovery − load consumption
```

The S-functional's `κΔI` gating now acts *inside* the evolution rather than only in the metrics: where distinction load is high (domain walls), capacity drains and integration stalls; where there is slack, capacity regenerates and smoothing resumes. Tune with `--kappa-baseline`, `--kappa-recovery`, `--kappa-consumption`, `--kappa-diffusion`. The field persists through snapshots, and S uses the real mean κ when the field is active.

Because **different scales see capacity differently**, the instrumentation reports κ at multiple resolutions: `kappa_by_scale` gives block-averaged capacity statistics per scale (fine scales expose the depleted texture around walls; coarse blocks average it away), snapshots carry `kappa_field_mean/min/std`, and the sectorisation report splits capacity into `wall_kappa_mean` vs `interior_kappa_mean` plus a per-sector `mean_kappa` — each sector's capacity budget, the resources that "being" has available to keep integrating its interior.

Two measured behaviors worth knowing:

- **Capacity texture is real**: walls run measurably depleted relative to interiors (e.g. 0.78 vs 0.91 at default rates), and the multi-scale report shows the depletion visible at scale 4 vanishing by scale 16.
- **Starved capacity freezes structure**: with κ depleted at walls, integration stalls and fragmentation persists (coarsening slows dramatically) — the field-theory analogue of the theory's "stalled integration" phenomenology, verified in `tests/test_dynamic_kappa.py`.

#### κ as soil — coupling capacity to the memory corpus

The theory's lineage essay frames stable structures as *seeds*, and is explicit that "a seed requires soil … A seed planted in barren ground remains a seed. It does not die, but it does not unfold." When both `--enable-memory-corpus` and `--dynamic-kappa` are active, the capacity field plays exactly that role of soil:

- a recalled seed only **roots** where the local mean capacity meets `--corpus-kappa-threshold` (fertile ground); below it, the seed does not unfold and the attempt is tallied as *barren*,
- rooting then **consumes** capacity (`--corpus-kappa-cost`), drawing down the soil it grew in so the same patch cannot be replanted until capacity regenerates — pushing recall to spread into fresh, coherent regions.

Snapshots and run summaries then carry `corpus_seeds_rooted` / `corpus_seeds_barren`. The effect is real and tunable: in a fertile run nearly every recall roots; in a capacity-scarce run (high consumption, low recovery, high threshold) the soil gates **100% of recall attempts as barren** — structure can only re-grow where the field can currently support it. Verified in `tests/test_corpus_kappa.py`.

#### Fitting F(N) from simulation — the N⋆ experiment

The gauge paper's selection argument rests on `F(N) = a·N^(2/3) − b·N` with `N⋆ = (2a/3b)³ = 3` at β ≈ 0.09. Rather than quoting that formula, `experiments/n_star_fit.py` measures its ingredients from runs across a (β, k) grid, where k is the number of wells made available:

```bash
python experiments/n_star_fit.py --size 24 --steps 400 --trials 2
python experiments/n_star_fit.py --betas 0.09 --gravity 0   # degenerate-well control
```

Per cell it records the realized domain count N, the time-averaged S-functional over a trailing window (the theory's selection criterion), and the total wall energy `E_wall` (β|∇φ|² summed over wall voxels), then fits the boundary coefficient `a(β)` by least squares on `E_wall ≈ a·N^(2/3)` and inverts the stationarity condition for the implied `b` per cell.

**Findings (24³, 400 steps, 2 trials — honest, mixed):**

- **The boundary-cost half of F(N) is supported.** `a(β)` is cleanly measurable and scales linearly: `a ≈ 2.6·β` across β ∈ {0.03, 0.09, 0.2}. Wall cost proportional to β is exactly what the theory's boundary term predicts.
- **The selection half is not (yet).** Time-averaged S generally *grows* with wall density, picking k = 5–6 rather than an interior optimum at 3; and the implied `b` varies ~45% across k at fixed β, inconsistent with the k-independent `b` the F(N) form requires.
- **Gravity is a real confound.** The `−G·φ` damping tilts the multi-well potential toward φ = 0, breaking well degeneracy and driving collapse toward one sector. In a gravity-free control at β = 0.09, k = 3 *does* maximize S by a wide margin — suggestive, but reported as anecdote: the domain count degenerates (the wall network percolates at this size), one β, two trials.
- **Dynamical κ changes the verdict — under scarcity.** With `--dynamic-kappa` at default rates, capacity texture forms (walls depleted) and coarsening slows, but selection stays ΔC-dominated (k = 6 still wins; in quasi-steady state ΔI ≈ 0, so S reduces to wall energy). In the **capacity-scarce regime** (`--kappa-consumption 50 --kappa-recovery 0.02`), however, the selection flips: **k = 3 maximizes time-averaged S, robustly across 4 seeds, with gravity on, by ~17% over the runner-up** — and the b-consistency of the F(N) fit improves (spread 0.45 → 0.33). Read with care: one β, one lattice size, one (c, r) pair, and the k = 3 domains are near-collapsed (N ≈ 1) — but the direction matches the theory's own structure, where `b = b(β, κ)` only has teeth when κ is finite. **Selection appears to be a scarcity phenomenon: with abundant capacity nothing penalizes fragmentation; under a binding capacity budget, three wells become S-optimal.** Mapping the (c, r, β) phase diagram of this selection is the clear next experiment.

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
- β-sectorisation analysis: gradient magnitude, sector labelling (including periodic merging and size filtering), per-sector distinction/integration statistics, triple-junction detection, and the engine integration hook,
- three-component (Ψ∈ℂ³) sector model: vector Allen–Cahn evolution, argmax sector labelling, interface detection, triple-junction counting (2-D and 3-D), S₃ permutation invariance, and report serialization,
- dynamical κ: depletion under load, recovery with slack, boundedness, determinism, κ-gated integration feedback (starved capacity preserves walls), multi-scale capacity reporting, per-sector κ budgets, and snapshot persistence,
- κ-as-soil corpus coupling: barren-soil rejection, fertile rooting with capacity consumption, replant gating after depletion, and rooting-statistics reporting,
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

### Three-Component Sector Field — genuine SU(3) Y-junctions

*The Ψ∈ℂ³ sector-membership layer of the gauge derivation.*

The scalar field above has a structural limit: stacked wells give only **layered** domains — a region in well `n` borders wells `n±1`, so three phases never meet and **no 120° Y-junctions can form**. This is an honest property of a single scalar, and it points at the next layer of the gauge derivation (§4.3.3): a **sector-membership field** `Ψ(x) = (R, G, B)` whose three components compete on equal footing.

`project_genesis.multiphase` implements this as a vector Allen–Cahn field — the multi-phase generalisation of the URP update:

```
∂_t η_a = D·∇²η_a − [ η_a³ − η_a + 2γ·η_a·(Σ_b η_b² − η_a²) ]
```

The triple-well free energy is **S₃-symmetric** (relabelling R/G/B is a symmetry — the discrete remnant of the global symmetry surviving deep inside sectors), and with all three phases mutually adjacent, **genuine three-way domains with 120° triple junctions form and coarsen** exactly as in grain-growth / soap-foam physics:

```python
import numpy as np
from project_genesis.multiphase import step_multiphase, analyze_multiphase

fields = np.random.default_rng(7).random((3, 96, 96)) * 0.1
for _ in range(600):
    fields = step_multiphase(fields, diffusion=1.0, gamma=1.5, dt=0.1)
print(analyze_multiphase(fields))   # -> n_phases=3, triple_junctions>0, ...
```

Triple-junction counts fall as the structure coarsens (e.g. ~360 → ~35 over 1500 steps), and are invariant under permuting R/G/B (the S₃ check in `tests/test_multiphase.py`). The model is dimension-agnostic — the same code runs in 2-D (the browser toy) and 3-D (matching the engine).

## Browser Toy (zero dependencies)

`web_toy/index.html` is a self-contained, single-file demonstration of the core URP ideas — **just open it in a browser**, no server, no build step, no CDN. It runs a 2-D version of the URP field equation live on a canvas:

```
∂_t φ = D·∇²φ + β·|∇φ|² − w·sin(2π·k·φ)
```

and shows:

- the field evolving by gradient-ascent, coloured by **sector** (nearest of `k` wells — red/green/blue for `k=3`) or as a raw field heat map,
- **domain walls** (high-|∇φ| boundaries) overlaid in real time,
- the live **S-functional** (`ΔC`, `κ`, `ΔI`, `S`) with an S-over-time graph,
- the measured **sector count `N`**, coarsening from a fine mosaic toward a few large domains.

Interactive sliders make the distinction/integration trade-off tangible: **raise β** and the field fragments into more sectors (distinction wins); **lower it** and domains coarsen toward a few (integration wins). It is the conceptual companion to the full 3-D Python engine.

A second page, `web_toy/su3.html`, runs the **three-component SU(3) sector model** (`multiphase` above) live: three R/G/B colours competing, forming domains with genuine **120° Y-junctions** that coarsen over time, with a live junction count and S-functional. The two pages cross-link so you can directly compare the scalar (layered, no junctions) and three-component (true triple junctions) models — the comparison *is* the lesson.

```bash
# any static file server, or literally just double-click the files:
python -m http.server -d web_toy 8000   # then open http://localhost:8000 (and /su3.html)
```

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

The frontier questions, roughly in priority order:

1. **Map the `N⋆ = 3` selection phase diagram.** The headline open result is that three-sector selection appears only under binding capacity. Sweep the `(kappa_consumption, kappa_recovery, β)` space with `experiments/n_star_fit.py` across several lattice sizes to find *where* `k = 3` wins — and whether the boundary tracks β ≈ 0.09. This turns a single suggestive data point into a real map.
2. **Couple a gauge connection `A_μ` to the Ψ∈ℂ³ sector field** to recover the Yang–Mills boundary modes (gluons) the derivation describes — the next theoretical layer above the three-component domains.
3. **Promote the F(N) fit to two free coefficients** by measuring an independent information-gain proxy for `b(β, κ)`, rather than inverting stationarity — the missing half of a non-circular free-energy test.
4. Higher-order field dynamics — second-order time derivatives (∂²φ/∂t²) for the wave-like behavior in the full Lagrangian (the current model is the overdamped limit).
5. Replace the sine pinning potential with the true `−(β/4)(∇φ)⁴` gradient-quartic via an implicit/stabilized integrator.

Earlier roadmap items now implemented: ~~coherence potential V(x,t)~~, ~~nonlocal integration functional I[φ]~~, ~~agent-agent interaction~~, ~~S-functional-driven agents~~, ~~matplotlib visualization~~, ~~emergent gauge sectorisation (measurement + wall tension + Ψ∈ℂ³ Y-junctions)~~, ~~dynamical capacity field κ~~.

## Theory Reference

The foundational theory document remains in:

- `Docs/The Universal Recursion Principle (URP) _260312_170343.txt`

That document describes the broader URP framing this sandbox is intended to explore in executable form.
