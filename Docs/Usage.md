# Running the sandbox

*Operational documentation: setup, the CLI, agents, the headless server and
WebSocket API, the browser toys, and the acceleration and visualisation layers.
Moved out of the README when that file was cut from 2137 lines to a landing
page — nothing here is removed, and the commands are unchanged.*

*For the theory see [`The_Principle.md`](The_Principle.md); for the measurement
record see [`Experiment_Log.md`](Experiment_Log.md).*

---

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
- κ-coupled multi-phase model: capacity-gated integration, depletion at walls, scarcity arresting coarsening, periodic domain counting, and the multi-phase S-functional,
- junction-resolving (volume-conserving) dynamics: phase-fraction conservation, persistent triple junctions, determinism, the full-palette neutrality measure, and the topological S-functional's interior optimum at three in both 2-D and 3-D,
- lattice gauge connection: U(N)/SU(N) group elements (unitary, unit determinant), gauge-invariance of covariant coherence, non-invariance of the naive coherence, coherence restoration by the pure-gauge connection, and zero curvature of flat connections,
- Yang–Mills gradient-flow dynamics: traceless-antihermitian projection, staple/matter-current identities, S-ascent, Yang–Mills residual → 0, SU(N)-preserving link updates, pure-gauge curvature relaxation, and covariant matter relaxation,
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

A third page, `web_toy/generations3d.html`, takes the same sector model into **three dimensions** and classifies cells by codimension: where *m* sectors meet is a codimension-(m−1) object, so m=2 is a wall, m=3 a triple *line*, and m=4 a quadruple *point*. That is why it defaults to a **four-colour** palette — a genuine 0-cell in 3-D needs four sectors, which is the same `d+1` counting the generation argument uses. Set the palette to 3 and the point class is empty by construction; the page says so rather than reporting zeros.

It is deliberately honest about its own limits, because they are real and were measured rather than assumed:

- The **γ slider is the load-bearing knob.** Only `γ > 1` makes the single-sector corners the minima of `Σ(¼η⁴ − ½η²) + γ·Σ_{a<b}η_a²η_b²`, which is what creates walls. At `γ = 0.5` the potential collapses *exactly* to `¼|η|⁴ − ½|η|²` — fully O(n)-symmetric, ground state the whole sphere, no distinct phases at all. Slide it there and the structure dies; that identity is the difference between a multiphase field and a Heisenberg one.
- **3-D has no steady state at browser-sized lattices.** Measured at N = 24–32: any bath below ≈0.35 coarsens to a single domain, anything above ≈0.5 is lattice-scale mush, and there is no window in between. So the page runs a finite-lifetime coarsening run and reseeds, instead of pretending to persistence it does not have.
- **Run-to-run spread is large and set by the initial draw, not the bath.** At fixed noise, four seeds gave domain scales from 1.2 to full monopoly, and the same pattern held at noise 0.005 and 0.05. Why the initial condition selects the branch so strongly is *not* currently understood — watch several runs, not one.
- The curvature colour scale is normalised by a smoothed 95th percentile, not the maximum: `κ = |∇²s|/|∇s|` has a vanishing denominator exactly *on* the wall it is colouring, and the max runs 30–80× the median, which is enough for one outlier to wash the whole field flat.

`window.__probe()` exposes the field state so the page can be checked against the NumPy kernel in `project_genesis/multiphase.py` rather than eyeballed.

A fourth page, `web_toy/observer_trial.html`, is not a toy but an instrument: it runs the randomised sham-controlled trials that `experiments/n3_observer.py` analyses. Fixed step budget per trial, computer-assigned ATTEND/LOOK-AWAY cue, a quarter of trials replaying pre-recorded runs as a negative control, running totals hidden so the stopping rule cannot be broken, and a JSON log at the end. Physics runs on a wall-clock timer and rendering on frames, deliberately separated — see the note in `generations3d.html` for why that separation is load-bearing.

```bash
# any static file server, or literally just double-click the files:
python -m http.server -d web_toy 8000   # then open http://localhost:8000 (also /su3.html, /generations3d.html)
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

---

## Repository Layout

```text
project_genesis/
  __init__.py          Package exports
  agent.py             Terrain-sensing agent with perception and action queue
  chunk_manager.py     Chunk-based world partitioning for active-region tracking
  config.py            Engine configuration and defaults
  engine.py            Field evolution, voxel quantization, agent orchestration, save/load
  gauge.py             Lattice gauge connection on the Ψ∈ℂ³ sectors (U(1)/SU(2)/SU(3))
  gauge_mc.py          Monte-Carlo sampling of exp(−β_g·S_W + g_m·Re[ψ†Uψ]): SU(N) heat-bath, overrelaxation, Wilson/Polyakov loops
  gauge_mc_kernels.py  Numba JIT kernels for the MC layer (~50–100× the reference path)
  annealed_matter.py   Joint (ψ, U) ensemble: thermal sector matter with gauge back-reaction
  io.py                Snapshot serialization helpers
  metrics.py           URP terrain summary metrics and S-functional computation
  memory_corpus.py     Stable-object corpus, composition, serialization, lineage
  sector_seeds.py      κ-as-soil rooting of stored structure into the thermal sector field
  soil_clusters.py     Percolation-style connected-component analysis of the fertile-soil mask
  topological_charge.py Geometric (Berg–Lüscher) topological charge of the ψ∈ℂ^N (CP^(N-1)) field
  gauge_topology.py    4-D SU(N) clover topological charge, gauge cooling, susceptibility
  instanton_scales.py  Peak-height instanton sizes (BPST/CP), CP gradient flow with measured D, the matched-scale (s=λ/ρ̄) comparison
  sector_field_4d.py   4-D CP² sector field: composite U(1) f_{μν}, second-Chern charge (c₁∪c₁ exact on fluxes), d-generic Metropolis + gradient flow
  hopfield_substrate.py  Second substrate for the criticality law: thermal Hopfield network, ΔC/ΔI/κ readings, S-compass trajectory taxonomy
  kuramoto_substrate.py  Third substrate for the criticality law: mean-field Kuramoto oscillators (synchronisation transition), ΔC/ΔI/activity readings — the capacity law, S-functional, and taxonomy imported verbatim from hopfield_substrate
  continual_learning.py  κ-as-soil for weights: numpy MLP, capacity-gated SGD (per-parameter regenerating plasticity), task generators
  capacity_gravity.py  κ as gravity: load masses, relaxed capacity wells, the free energy F[κ], screening-length instruments
  capacity_dynamics.py Self-gravitating masses in the κ-field: overdamped/inertial/cosmological (FLRW) evolution, stress-energy, Friedmann-from-action
  capacity_waves.py    Finite-speed κ (telegrapher form): causal cone at c_κ=√(D/τ), massive dispersion ω²=(Dk²+r+cρ)/τ−1/4τ², retarded inertial gravity (Lyapunov energy), the exclusion contact term (b/2)∫ρ², parabolic control
  stable_forms.py      The spectrum of stable κ-forms: structural mass, binding, form interactions
  multiphase.py        Three-component Ψ∈ℂ³ sector field with 120° Y-junctions
  dimensional_forms.py CW-census of the sector tessellation (any dimension): 0D/1D/2D(/3D) cells, Euler V−E+F(−C), junction valence, the Plateau structure, and the flavour (sector-composition) multiplets
  chiral_field.py      The chiral (parity-breaking) spin term: complex Ginzburg–Landau with coupling λ, intrinsic precession Ω=−λ, vorticity (spin density)
  two_field.py         The derived two-field coupling: the chiral ψ co-evolves with the telegrapher κ — κ wells detune ψ (matter holes the field), ψ's phase current presses on the masses (radial, odd in λ), and the molecule's spin is slaved to ψ's own measured precession (zero couplings = bitwise the retarded baseline)
  vortex_chiral.py     The vorticity-bearing chiral field: a vortex pinned per form (integer winding), quantised field angular momentum L∝q, and a phase-current torque (same-sign vortices co-rotate, vortex–antivortex translate) — the molecule spins from its own circulation with no imposed drive. Also the self-sustained mode (winding_number, evolve_seeded_field, reimprint=False, chiral_lambda): seed once and co-evolve the CGL — the winding is a dynamically-conserved topological charge and the κ wells pin the core; with the field's own precession (λ≠0) the circulation regenerates and the emergent molecule spins at the pinned strength (handedness from the charge, strength from λ)
  vortex_chiral_3d.py  3D spin: the vortex is a LINE and its angular momentum an axial VECTOR aligned with the line (vortex_line, line_angular_momentum, line_winding, evolve_seeded_line) — quantised by winding, conserved with its axis under the self-sustained CGL; an integer/spin-1-like (bosonic) angular momentum (the half-integer spinor is the frontier). Also the 3D molecule (line_phase_force, evolve_line_molecule): a real vector spin-torque the overdamped bound pair cannot turn
  nematic_spinor.py    Half-integer spin: a nematic director field (n̂≡−n̂, RP¹) whose defects are ±½ disclinations — disclination_strength (s=q/2), director_holonomy (the 4π double cover: a 2π loop flips the director by −1, 4π restores it), plaquette_winding (the tracker-free defect map) — the topological realisation of a spinor (the SU(2) double cover of SO(3))
  spin_statistics.py   The exchange sign of the ½-disclination: braid_positions / exchange_holonomy (braid two disclinations, read the centre-winding and the exchange sign (−1)^k) and self_rotation_sign — the statistics half of spin–statistics and its connection to spin ((−1)^2s on both sides). Also the dynamical braid: gaussian_wells, transport_defect (adiabatic well-transport, with a speed limit), dynamical_braid (co-evolved κ-pinned braid, no re-imprint; phase_pin adds the phase-anchoring cure)
  race.py              The imposed-lead intervention — seed_with_lead (tilts one sector's amplitude, field stays on the unit sphere), run_race (follows the top-two share gap and records deaths), growth_rate (fits log/linear/sqrt side by side so 'it compounds' can actually be refuted — and was)
  nematic_rp2.py       Nematic defects in 3-D — a real three-component director (not a complex phase), Lebwohl-Lasher relaxation with confine_plane to recover RP1, loop_z2_class (the pi_1(RP2) = Z/2 invariant: does the director come back flipped?), loop_strength (the RP1 integer grading, for comparison), tilted_core (seeds the escape off the symmetric saddle), core_escape
  emergence.py         Causal emergence — effective_information (EI in bits, with its determinism/degeneracy decomposition), equilibrated_backgrounds, intervention_tpm (performs the do-operation rather than harvesting observations, which is the whole point since EI is defined over interventions), emergence_profile across coarse-graining scales
  expansion.py         Expansion schedules — scale_factor (flat / radiation / matter / de Sitter) and diffusion_schedule, implementing D(t)=1/a(t)^2. For a first-order order parameter comoving coordinates give expansion as a diluting Laplacian and nothing else; there is no Hubble friction term, which belongs to second-order wave equations
  survival.py          Survival analysis for "sometimes it doesn't end" — batched ensemble evolution (step_batch is the same rule as multiphase.step_multiphase, pinned elementwise), kaplan_meier / split_hazard (is the death rate memoryless?), held_out_auc + permutation_p (is the ending foreseeable?), max_feature_permutation (look-elsewhere correction), confirm_feature (pre-specified test on fresh data)
  multiphase.py        (+ texture guard) domain_scale, majority_filter and resolved_palette_junction_density — full_palette_junction_density counts lattice-scale texture as binding once domains shrink to the grid (measured: pure 4-sector noise scores 0.997), so the guarded form majority-filters the labels and, more importantly, flags the domain scale. The `resolved` flag is the real guard; the filter alone only takes 0.997 to 0.641. On genuine domains it is a no-op to four decimals
  selection.py         The selection sweep: run the alternatives on one instrument — evolve_palette (any sector count P), measure_window (distinction / integration / churn / sectors-alive), selection_score (integration × churn: binds AND still moving)
  boundary_gravity.py  Does the boundary know the interior? A Gauss law for κ-gravity: disc_indicator, enclosed_flux (exact via the lattice divergence theorem), enclosed_mass, gauss_ratio_profile (flat = a Gauss law), flatness
  area_law.py          The scaling dimension of the generative gap: nested-region distinction/integration content, the Gaussian-MI cross-boundary estimator (ensemble-averaged, floor-subtracted), and gaussian_control_field — a synthetic field with a KNOWN area law for calibrating the instrument before trusting it
  gauged_vortex.py     The self-consistent gauge field (abelian Higgs / Ginzburg–Landau): seed_vortices, plaquette_flux / local_flux, covariant_laplacian, energy, relax (gradient-flow ψ and the U(1) links; gauge_on=False = global-vortex control), gauge_transform — the vortex as a gauged particle (quantised flux, London screening, gauge invariance); wilson_loop / ab_phase — the Aharonov–Bohm holonomy and the flux–charge composite's statistical phase
  condensation.py      The condensation instruments: grow_defect_gas (spin defects from noise), amp_reaction_force (the detuning's third-law force on a mass — raw clumps, envelope-normalised is selective to the topological core), condensation_run (masses + a co-evolving defect gas, with a λ transport current and opt-in ψ snapshots to read a captured core's spinor signature), sourced_gas_step (a Langevin gas source — a bath that replenishes the coarsening gas above a threshold), inject_defect_pair (a cold source — a clean winding-±1 = s=±½ pair injected dilutely)
  network_server.py    WebSocket server for remote monitoring and control
  numba_kernels.py     Numba JIT-accelerated field evolution kernels
  render.py            Text-based slice rendering for terrain inspection
  s_compass_bridge.py  S-compass connector bridge for AI agent integration
  sectorisation.py     β-sectorisation / boundary-formation domain analysis
  visualize.py         Matplotlib-based 3-D voxel and S-functional visualization
Docs/
  The Universal Recursion Principle (URP) _260312_170343.txt
  Thermal_Sector_Program.md  Synthesis: the full thermodynamic N⋆=3 program and its verdicts
  The_Generative_Gap.md      Capstone: the distinction–integration gap, and the ordinal→functor→instanton bridge
  The_Measured_Bridge.md     Closing synthesis: ordinals→functors→instantons as one chain, κ≈0.22 in the 4-D vacuum
  Capacity_As_Gravity.md     κ as the framework's gravity: a universal, mass-sourced, √(D/r)-screened attraction
  The_Emergent_Cosmos.md     Capstone (Act II): κ→gravity→matter→structure→cosmos, with toolkit map and frontiers
  The_Principle.md           **Start here** — the theory from zero: principle, consequences, measured/framework/declined tiers, the selection argument, and what would refute it
  The_Complete_Arc.md        Top-level synthesis: the whole program — both acts, the one-κ frontier, the honest boundaries
  Deriving_The_Exclusion_Coefficient.md  The pair-binding derivation (Parts I–V): no-cloning as a parameter-free degeneracy pressure
  The_Woven_Forms.md         Capstone (Act III): the field's matter — dimensional forms (2:3:1), natural pairs, finite-speed gravity, chiral spin; measured vs. vision
  The_Gauged_Fermion.md      Capstone (the fermion arc): spin → composites → exchange sign → dynamics → gauge field → Aharonov–Bohm — one (−1)^2s three ways, and the second-quantisation boundary
tests/                 633 checks across the engine, instruments, and physics
  test_genesis_engine.py
  test_annealed_matter.py
  test_corpus_kappa.py
  test_dynamic_kappa.py
  test_gauge.py
  test_gauge_mc.py
  test_gauge_mc_confinement.py
  test_gauge_mc_matter.py
  test_gauge_mc_numba.py
  test_kappa_annealed.py
  test_s_landscape.py
  test_sector_seeds.py
  test_recall_recovery.py
  test_recall_finite_size.py
  test_soil_clusters.py
  test_memory_competition.py
  test_memory_clusters_3d.py
  test_recovery_rescue_3d.py
  test_form_selection.py
  test_capacity_separation.py
  test_topological_charge.py
  test_functor_bridge.py
  test_continual_learning.py
  test_gauge_topology.py
  test_hopfield_substrate.py
  test_kuramoto_substrate.py
  test_scarcity_power.py
  test_spin_statistics.py
  test_dynamical_braid.py
  test_phase_pinned_braid.py
  test_gauged_vortex.py
  test_ab_statistics.py
  test_area_law.py
  test_boundary_gravity.py
  test_selection.py
  test_instanton_scales.py
  test_sector_field_4d.py
  test_memory_corpus.py
  test_multiphase.py
  test_multiphase_kappa.py
  test_n3_phase_boundary.py
  test_n3_thermal_selection.py
  test_new_subsystems.py
  test_s_criticality.py
  test_sectorisation.py
  test_potts_transition.py
  test_potts_universality.py
  test_scaling_ladder.py
  test_sigma_scan.py
  test_topological_selection.py
  test_urp_extensions.py
experiments/
  beta_sectorisation.py β-sweep measuring emergent sector counts
  n_star_fit.py         Fits the F(N) free-energy coefficients from run data
  phase_diagram.py      Maps the (consumption, recovery, β) N⋆=3 selection map
  multiphase_kappa.py   κ-coupled Ψ∈ℂ³ run: emergent N vs S-maximizing P
  standing_integration.py  Tests standing coherence for an interior N⋆
  topological_selection.py Conserved dynamics + neutrality: S-optimum at three
  gauge_coherence.py    Gauge connection as coherence restoration (U(1)/SU(2)/SU(3))
  yang_mills_flow.py    Gradient ascent on S: YM residual → 0, gluons on walls
  confinement_sigma_scan.py σ(β_g) area-law scan: Creutz ratios with jackknife errors
  n3_thermal_selection.py   N⋆=3 selection under a fluctuating SU(P) gauge ensemble
  n3_annealed_matter.py     Junction-network melting in the joint (ψ, U) ensemble (2-D and 3-D)
  n3_phase_boundary.py      T_melt(g_m) boundary map with susceptibility crossover check
  n3_scaling_ladder.py      Finite-size-scaling ladder: chi_max(L) exponent fit
  n3_potts_transition.py    Unpinned S_P order-disorder transition vs 3-state Potts
  n3_potts_nu.py            Binder data collapse: the second Potts exponent (nu)
  n3_potts_3d.py            3-D first-order hunt: hysteresis + Binder minima
  n3_latent_heat.py         3-D energy histograms: the latent-heat verdict
  n3_s_criticality.py       The S-functional measured across the Potts transition
  n3_kappa_criticality.py   Dynamical capacity co-evolving at the transition
  n3_s_landscape.py         S-landscape (c, w) phase diagram: the optimum-relocation level crossing
  n3_seed_rooting.py        Memory recall at criticality: κ-as-soil seed rooting across the melt
  n3_recall_recovery.py     Does capacity regeneration rescue recall? recovery-rate rescue scan
  n3_recall_finite_size.py  Does "recall outlives order" survive L→∞? finite-size ladder + extrapolation
  n3_memory_clusters.py     Spatial structure of surviving memory: fertile-soil percolation across the melt
  n3_memory_competition.py  Competing memories: two seeds, one κ budget — the persistence↔plasticity dial
  n3_memory_clusters_3d.py  Memory connectivity in 3-D: surface walls vs 2-D line walls
  n3_recovery_rescue_3d.py  Can faster recovery rescue 3-D de-percolation? reconnection-rate scan (2-D vs 3-D)
  n3_form_selection.py      Platonic forms selected by S: capacity decides which form the universe manifests
  n3_capacity_separation.py The distinction–integration gap, measured: the structural cliff past P=3
  n3_instanton_content.py   The instanton content of the sector field (CP²): χ_top, cooling, the topological fraction
  n3_functor_bridge.py      The functor logic→vacuum, measured: integration ladder ↦ topology, path-independent
  n3_su3_topology.py        4-D SU(3) topological charge (Stage 1): clover Q, cooling, χ_top vs coupling
  n3_su3_gradient_flow.py   4-D SU(3) gradient flow (Stage 2): t₀ scale, Q→integer, self-dual fraction vs κ
  n3_su3_continuum.py       4-D SU(3) continuum trend: f_SD(t₀) volume-converged but cutoff-dependent
  n3_kappa_gravity.py       Capacity as gravity: κ mediates a √(D/r)-screened, mass-sourced attraction
  n3_stable_forms.py        The corpus of stable forms: discrete mass spectrum, m_inertial = m_gravitational
  n3_self_gravity.py        Self-gravitating forms: two-body infall + N-body accretion under κ-gravity
  n3_orbital_gravity.py     Inertial κ-gravity: Kepler-like orbits, conserved energy, precession, virialization
  n3_cosmic_structure.py    κ-gravity vs Hubble expansion: turnaround, and structure suppressed by expansion
  n3_expanding_universe.py  FLRW background: scale factor a(t), Hubble drag, dark-energy freeze-out of structure
  n3_self_contained_cosmos.py  Closed loop: dark energy from κ's self-maintenance drives the emergent expansion
  n3_matter_from_forms.py   Matter source from the form spectrum: ρ_m0 ∝ Σ|Q| (Bogomolny) + a^(−dim) from topology
  n3_form_equation_of_state.py  Equation of state of the matter: cold forms are dust (w=0), the capacity vacuum is Λ (w=−1)
  n3_stress_energy_closure.py  Relativistic closure: T^μ_ν from the field, expansion as a consequence of ∇·T=0
  n3_friedmann_from_action.py  Variational closure: Friedmann H²=ρ as the Hamiltonian constraint / a first integral of an action
  n3_gravity_from_capacity.py  Gravity from the field: −a ȧ² as the capacity scalar's kinetic free energy, expansion as κ_s=ln a rolling
  n3_one_kappa_frontier.py  The one-κ frontier: κ̂=Σ|q|/Σe as one operator across Act I (SU3) and Act II (CP²) — same concept, not (yet) one number
  n3_kappa_obstruction.py   The one-κ obstruction: the sharper invariant ⟨Q²⟩/⟨S⟩ fails too — mechanism is the instantons' different RG fate
  n3_scale_matched_bridge.py  The scale-matched bridge: κ̂ compared at equal smoothing-per-instanton-size (s = λ/ρ̄) — the obstruction's own condition, imposed
  n3_4d_sector_bridge.py    The like-for-like bridge: a 4-D sector field, so both κ̂'s share operator, dimension, and flow clock — and where they cross
  n3_kappa_deflation.py     The deflation test: sweep the t²E = c reading convention and watch the 0.22 — is Act I's constant a number or a convention?
  n3_criticality_transplant.py  The criticality transplant: "scarcity pushes S to criticality" tested on a Hopfield network — the condition toggled, not assumed
  n3_kuramoto_transplant.py  The Kuramoto transplant: the same law on a third, structurally-independent substrate — mean-field oscillators (no lattice, no memories, continuous order parameter) — 3/3, the level crossing returns under the noise-repair toggle and the S-compass `diverging` band again brackets the scarce optimum
  n3_continual_learning.py  The capacity law meets external ground truth: the persistence↔plasticity dial on real learning, with controls and a fair baseline
  n3_curriculum_order.py    Curriculum order under the capacity law: foundations-first vs composite-first on compositional tasks — and the protection↔composability dial
  n3_constructive_kappa.py  Constructive-load κ: the per-parameter building/breaking distinction, tested — a registered negative with its mechanism (function-space is next)
  n3_functional_kappa.py    Function-space κ: damage measured on prior function, consolidation in the law — protection AND composability, first variant to hold both
  n3_combined_benchmark.py  The combined benchmark: composability + protection in one sequence, vs plain SGD, standard κ, and rehearsal at equal information
  n3_scarcity_benchmark.py  The scarcity-scaled benchmark: trunk width as the capacity dial — the margin trends right but stays within noise (unpaired)
  n3_scarcity_power.py  The scarcity power test: the SAME paired runs, analysed paired and powered — the unpaired error bar was the artifact; functional κ's advantage over plain SGD is real, significant, and scarcity-graded, with the rehearsal boundary named
  n3_growth_factor.py       The growth factor: perturbations in the κ cosmology — scale-dependent growth (the theory's GR departure) and Λ freeze-out, measured
  n3_growth_spectrum.py     The growth spectrum: S(λ) is band-passed (footprint UV wall, screening IR wall) — the knee needs bigger boxes, quantified
  n3_screening_knee.py      The screening knee: the field's own dial moves the knee into the window — 3/3, and matter screens gravity (the loaded Debye law, measured)
  n3_local_screening.py     Local screening: one field, two gravities — 3/3, the range is set by the local environment (chameleon-style), not the box mean
  n3_environment_growth.py  Environmental growth: the static field predicts the dynamics — 3/3, the dense band grows slower (near-sightedness beats extra mass)
  n3_kappa_lightcone.py     The κ light cone: finite update rate τ gives the field a causal cone at √(D/τ) — 2/3, with the damping-envelope wall measured
  n3_retarded_gravity.py    Retarded κ-gravity: drag law from statics, supersonic silence, and the binary inspiral — 3/3, the adiabatic control conserving
  n3_quadrupole_line.py     The quadrupole line: even-harmonics-only selection rule (the no-dipole analogue), the dipole control, the medium's complex k — 3/3
  n3_plunge_ringdown.py     The plunge and the ringdown: no long inspiral in this gravity — the trench mechanism, the sweep clock, the healing afterglow — 3/3
  n3_exclusion_core.py      The exclusion core: no-cloning as degeneracy pressure — the collapse gets a floor, and the ringdown returns at the well's own pitch — 3/3
  n3_exclusion_derived.py   Deriving b, Part I: the homogeneous gap 2F(ρ)−F(2ρ) — 2/3; refuses clones only in the dilute window, inverts into a merger subsidy at the operating point
  n3_exclusion_gradient.py  Deriving b, Part II: the full-functional gap on the duplicated component — 3/3; the parameter-free floor (s*=8), the ringdown at the statics' pitch
  n3_exclusion_labelled.py  Deriving b, Part III: labelled loads — 3/3; exclusion prices only true clones (φ=0 is bitwise the baseline), barrier monotone in shared fraction
  n3_exclusion_ncopy.py     Deriving b, Part IV: the n-copy sector — 3/3; pairwise = n-copy at O(c²), trimer floor, the saturation split measured
  n3_exclusion_dilute.py    Deriving b, Part V: the dilute operating point — 0/3 recorded as-is; joint-limit convergence, the min-construction pinned by a 2×2, the repelled binary
  n3_identity_generation.py Identity generation: sameness measured from internal patterns (φ = Σmin), not assigned — clones floor, independents don't, and drift sets an individuality threshold
  n3_identity_invariance.py Identity invariance: aligned sameness (max over rigid motions) — the stranger gap survives, the floor follows identity not pose; the resolution trade-off measured
  n3_kappa_molecule.py      The κ-molecule: derived-exclusion floor + spin — 2/3; the first persistent bound object, overdamped rotation (Q<½), but the quasi-static drag rate is refuted 8×; sings at the statics' libration pitch
  n3_quark_generations.py   The dimensional-form hierarchy: 0D/1D/2D forms as CW-cells — 3/3; the confined ratio is the trivalent 2:3:1 (N⋆=3), the Euler defect is a deconfinement order parameter
  n3_chiral_spin.py         Chiral spin: the parity-breaking term gives the field intrinsic spin Ω=−λ — 3/3; λ=0 restores parity, and the spin lives on the 0D 'light' forms
  n3_spinning_molecule.py   The spinning molecule: a chiral drive lets the bound κ-pair hold Ω where the achiral molecule drained (Z2) — the first bound object that turns
  n3_form_abundances.py     Form abundances: three generations because space is 2-D (families=d+1, palette-independent) — 3/3; topologically protected, heavy rarest; no numerical quark-match claimed
  n3_3d_generations.py      Four generations in 3-D: the census's sharpest prediction — 3/3; families=min(P,d+1), the Plateau foam valences 4/3/2/1, Euler V−E+F−C=0 on T³
  n3_flavour_structure.py   Flavour structure: the forms' second quantum number (sector-composition) — 3/3; the multiplet sizes are Pascal's C(P,d+1−ℓ), democratic, conserved; no CKM-match claimed
  n3_two_field_chiral.py    The derived two-field coupling: ψ co-evolves with κ — 3/3; the field carries its spin while holed by the wells (Ω=−λ measured), the phase current presses the bond (radial, odd in λ, torque-free), and the molecule spins slaved to the field's own precession — no Ω_bg
  n3_vortex_chiral.py       The vorticity-bearing chiral field — 3/3; a vortex pinned per form carries quantised angular momentum (L∝q), the phase-current force becomes a torque (same-sign vortices co-rotate, vortex–antivortex translate), and the molecule spins from its own circulation with no imposed rotation — the rigid-rotation flow-profile ansatz retired
  n3_emergent_vortex.py     The self-sustained vortex — 2/3 (an honest boundary); seed once, co-evolve the CGL with no re-imprint: the winding is a dynamically-conserved topological charge (noise-robust), the κ wells pin the core and select like-survive/unlike-annihilate — but the strong orbital spin does NOT hold without the re-imprinting (the circulation drains)
  n3_driven_vortex.py       The self-sustained strong drive — 3/3; closes the emergent E3 negative: the field's own precession (λ≠0) regenerates the circulation so the seeded-once, co-evolved molecule spins at the pinned strength — handedness set by the vortex charge, strength by |λ|, and both ingredients necessary (no vortex → zero phase force, no precession → weak/draining)
  n3_vortex_3d.py           3D spin is an axial vector — 3/3 (field-level); the defect is a vortex LINE and its angular momentum L is a 3-vector aligned with the line at any orientation, quantised by winding; conserved with its axis under the self-sustained CGL (noise-robust); like-survive/unlike-annihilate — an integer/bosonic spin (the half-integer spinor, the fermion, is the open frontier)
  n3_molecule_3d.py         The 3D molecule — 2/3 (an honest negative); the bound pair carries a real VECTOR spin-torque (tangential, sign-locked, axis = the line direction) and holds its winding/L on the exclusion floor, but it is OVERDAMPED and cannot turn — the κ-molecule's Q<½ failure returning in the thicker 3D medium (the 2D vortex drive beat that drag; here it can't)
  n3_spinor.py              Half-integer spin — 3/3; a nematic ±½ disclination (the director winds by π, s=q/2 — impossible for a vector), the 4π double cover (a 2π loop flips the oriented director to −1, 4π restores it — the SU(2) double cover, the signature of spin-½), conserved & fused (½+½=1) & bound to matter. The TOPOLOGICAL realisation of a spinor (not yet the quantum Dirac field — spin-statistics is the open frontier)
  n3_hadron_spin.py         Hadron-like composites — 3/3; the pieces together: n half-integer constituents at derived-floor spacing carry total spin s=n/2, statistics alternating by count (1,3 → fermionic/quark-baryon-like; 2,4 → bosonic/meson-like), the ½'s confined inside; and the meson-analog is a real molecule — bound by κ-gravity + the derived floor, spun by the self-sustained driven field, integer & no-flip from outside
  n3_junction_fermion.py    The junction is the fermion — 3/3; N⋆=3 selects the spin-½ constituent count: a defect's 2π winding must cross EVERY phase sector, so in emergent (noise-grown) fields every spin defect IS a trivalent three-sector junction (bijective, singlet 1:1:1 composition, every one s=±½ with the −1 holonomy; two-sector walls spinless) — and the defect needs valence P while the matter tessellation caps at Plateau 3, so the structures coincide only at P=3
  n3_exchange_statistics.py The exchange sign — 3/3; the STATISTICS half of spin-statistics: braiding two identical ½ disclinations reads a genuine −1 (antisymmetric), the sign is (−1)^2s across charges (½→−1, 1→+1, 3/2→−1), and it EQUALS the 2π self-rotation holonomy — the spin-statistics connection measured on the field's own disclinations (double braid / far field / no-swap / ½,−½ all +1). Topological (Finkelstein–Rubinstein), not yet Fock-space anticommutation
  n3_dynamical_braid.py     The exchange sign under genuine dynamics — 3/3; D1 a κ-pinned ½ is transported by a moving well with its winding conserved ADIABATICALLY, with a measured speed limit (κ pins amplitude, not phase); D2 a static-pinned ½ keeps its winding + −1 holonomy under long co-evolution (no re-imprint); D3 the two-body braid realises the fermionic sign PARTWAY (sign −0.77, cores surviving 92%) — the amplitude-vs-phase pinning boundary, cure = phase-anchoring (the bridge to a dynamical fermion)
  n3_phase_pinned_braid.py  Phase-aware pinning closes the D3 boundary — 3/3; P1 a weak, local, gauge-like phase-anchor completes the two-body braid (fermionic −1 clean from co-evolution, sign −0.99, 100% survival) where amplitude-only reached −0.57; P2 a modest threshold η⋆≈0.2 (a finite-rate force, not a per-step reset); P3 the honest control — integer→+1, double braid→+1 under the SAME pin, so the pin transports winding and the sign is the braid's geometry, not the pin's. Still classical, not Fock-space anticommutation
  n3_gauge_field.py         The self-consistent gauge field — 3/3; the phase-template caveat closed with the real field (abelian Higgs): G1 flux quantization (the U(1) gauge field self-consistently carries Φ=2πq per winding, q=1,2,3; zero gauge-off), G2 London screening (the global vortex's log-divergent energy screened into a finite-energy soliton — gauged saturates, global grows), G3 gauge invariance (energy + flux invariant under a local gauge transform to machine precision). The vortex is a gauged particle; Fock-space anticommutation + the AB/Chern–Simons statistical phase remain the frontier to a quantum fermion
  n3_race.py                The race — 2/3, and Q2's failure corrects the story. n3_survival found maxfrac at step 60 predicts survival (AUC 0.801), but that was a CORRELATION; this IMPOSES the lead by tilting one sector's amplitude before normalising, turning it into a dose-response. Q1 HELD on a fresh seed and the shape is the point: survival is a CLIFF, 0.44 -> 0.00 across a 1.6 percentage-point change in initial share (0.260 -> 0.276). So the lead is causal, not a marker; the unbiased arm survives because no sector leads SYSTEMATICALLY, not because none leads. Q2 FAILED and it matters: the gap grows LINEARLY (drift, R2 0.994) not exponentially (compounding, R2 0.949), so it is NOT a race — it is first passage, the gap opening at constant rate until it crosses threshold. Q3 HELD: survival rises monotonically with palette (P=2 0.09 -> P=6 0.52), a THIRD axis preferring large palettes, which sharpens the selection sweep's interior optimum rather than softening it
  n3_rp2_defects.py         RP2 defects — 2/3, and it audits the dimension of the fermion arc. Every spinor/statistics module represents the director as theta = 1/2 arg psi, and a complex phase is a circle, so they all describe RP1 (pi_1 = Z, integer-graded, +/-1/2 opposite charges). A director free in 3-D lives on RP2, where pi_1 = Z/2 — two classes only. Q1 HELD: the Z/2 invariant reads exactly (2s) mod 2, and +1/2 and -1/2 land in the SAME class. Q2 FAILED as registered because the flat texture is a symmetric saddle where zero-temperature descent has no out-of-plane gradient — an optimiser fact, kept as failed. Q3 HELD: the 1/2 line cannot escape even given the third dimension. The repaired instrument (seeded off the saddle, labelled post-hoc) separates the classes cleanly: integer core |n_z| 0.66 -> 0.98 (escapes, unprotected), half-integer 0.66 -> 0.02 (protected). Conclusion: the 3-D spinor SURVIVES (Z/2 gives exactly the two classes an exchange sign needs; the 4-pi cover was always pi_1(SO(3))), but the integer winding does NOT, so the 'winding crosses every sector' constituent-count argument is planar and does not transfer
  n3_causal_emergence.py    Causal emergence — 1/3, and the failures are the informative part. Hoel/Albantakis/Tononi effective information with GENUINE interventions (the block is overwritten with a chosen sector on an equilibrated background, then run forward — a real do(s), not an observational transition matrix). Same macro variable at every block scale, so the log2(P) ceiling is identical and no normalisation is needed. Q1 HELD: emergence +1.58 bits, the full ceiling — a single-site intervention is essentially entirely undone by its neighbourhood while a block intervention sticks. Q2 FAILED because it was saturated: at ZERO noise the micro already scores 0.001 bits, so deterministic coupling alone fully degrades the micro and stochasticity is not needed. Q3 failed as registered (decoupled at the highest noise shows +1.32 bits, i.e. EI does partly reward trivial averaging — a real limitation) but the paired noise sweep separates it cleanly: at zero noise coupled +1.584 vs decoupled +0.000, with no randomness anywhere for averaging to act on
  n3_capacity_gating.py     Capacity gating — 2/3, and it CORRECTS a claim in Docs/The_Principle.md. The discriminating test between 'URP is doing work' and 'this is Plateau's law': does capacity GATE the P=3 selection, or is the selection pure geometry? Pre-registered AGAINST the framework's own section 3. Q1 HELD: P=3 peaks at EVERY capacity level, including the arm with no capacity field at all and the arm where capacity is free. Q2 failed on the letter (margin spread 205x vs 3x tolerance) but the direction is backwards from what section 3 claims — scarcity DESTROYS the selection, and restricted to arms the texture guard calls valid the spread is 2.17x, inside tolerance, matching n3_capacity_separation's capacity-invariance. Q3 manipulation confirmed (mean kappa 1.000 -> 0.025, distinction 0.058 -> 0.694), so the nulls are informative not vacuous. Conclusion: the P=3 selection is GEOMETRIC. It cannot be cited as evidence URP does work a generic coarsening model would not — the substrate transplants carry that, and they are not reducible to coarsening because two of the three substrates have no geometry
  n3_expansion.py           Expansion — 1/3 pre-registered. Does heat death belong to the law or to the box? Expansion enters a relaxational field only as a diluting Laplacian, D(t)=1/a(t)^2, so it is one scalar schedule on the same kernel. Q1 HELD and it is the headline: the flat control keeps its palette in 41% of runs, matter-like expansion in 98% — the terminal state was a property of the BOUNDARY, not the dynamics. Q2 failed on the letter (registered domain scale < 1.5, measured 1.68) though de Sitter's scale is 5.5x smaller than every power-law arm, the decoupling signature. Q3 failed: the optimum sits at the fastest arm, a boundary maximum. Most useful outcome is a NEGATIVE about the instrument — the framework's own integration x churn score is maximised by the MOST fragmented arm, because full-palette junction density counts lattice-scale texture as binding once domains shrink to the grid. The score does not protect against decoupling; gating it on a resolved domain scale is a new prediction, not a re-scoring
  n3_observer.py            Observer protocol — a randomised, sham-controlled design for "does attention change the outcome?", plus the power table that makes it answerable. Names the confounds and closes them: the OLD viewer had a real mechanical observer effect (physics ran inside requestAnimationFrame, so a foreground tab computed 4.7x as many steps as a background one — now 1.0x); fixed step budget removes the sampling channel; randomised assignment removes reverse causation; code-scored outcomes remove judgement. Sham arm replays pre-recorded runs, so an effect appearing there is in the observing, not the field. Key number: detecting even a +30 point shift in survival needs 84 trials — twenty informal trials are no evidence in EITHER direction. `--demo-null` shows what the pipeline reports when nothing is there
  n3_survival.py            Survival of the 3-D sector field — 1/3 pre-registered, confirmation PASSED. Answers "most runs fade out, some don't — fact or feeling?" without watching. 96 runs, death defined in code, censored runs kept. Q1 HELD: the hazard FALLS 2.5x (ratio 0.405, CI [0.227, 0.691]) — last death at step 720, then 29,640 run-steps with zero deaths, so survivors are a different population, not an exponential tail. Q2 PARTIAL: held-out AUC 0.666 (p=0.032) beats the null but misses the 0.70 bar. Q3 FAILED: the carrier is `maxfrac` (largest sector share at step 60), not the predicted `imbalance` — then CONFIRMED on a fresh ensemble at AUC 0.801, p=0.0005. The explore-then-confirm split is the point: the winning feature is re-tested as a pre-specified statistic on data it was not chosen from. `--when` sweeps the capture step instead of fixing it, and corrects the natural reading of Q2: there is NO decision moment — the bias is already present at step 5 (AUC 0.617, before any structure exists) and climbs monotonically through the whole mortality window (0.745 at step 60, 0.905 at 200, 0.990 at 400) rather than crossing a threshold. Quoted against the field's own relaxation time tau, since the simulation has no seconds: median lifetime is 0.78 tau and the last death 2.01 tau, so half of all deaths happen before the field has finished climbing into its wells — those runs are not dying of old age, they are failing to form
  n3_selection_sweep.py     The selection sweep — 3/3; the answer to "you're showing one slice of possibility": run the also-rans on one instrument. Q1 distinction rises monotonically with P (it picks the LARGEST palette, not ours), Q2 full-palette integration is a strict monopoly at P=3 (P=2 cannot form a trivalent junction; P>=4 makes more junctions but binds none), Q3 the joint criterion has a unique INTERIOR optimum at P=3, winning 88x while being best at no single axis. Selected, not chosen. `--ndim 3` is the harder test: the competing hypothesis P=d+1 (which would predict FOUR colours in 3-D space) is ruled out — P=3 still wins — but the monopoly fails and the margin falls 88x -> 4x, barely clearing its own falsifier. REGRESSION-CHECKED against the texture artifact found in n3_expansion: every palette is resolved in both dimensions, the guard changes no ranking, it STRENGTHENS 2-D (joint margin 88x -> 272x, since P=4's integration was itself partly texture), and in 3-D the P=3:P=4 integration ratio moves 5.6x -> 6.4x, still under the 10x monopoly bar — so the 3-D weakening is REAL, not an artifact. Weigh the 3-D number
  n3_boundary_gravity.py    Does the boundary know the interior? — 1/3, B1 and B2 REFUTED; κ-gravity has NO Gauss law while screened (flux/M spans 10^5 across surface radii, decaying as e^(-R/xi), and varies 10x with the arrangement of the same enclosed mass) — but the Gauss law IS restored as r→0 (spread 1.4x at xi=45). Boundary-encoded gravity needs a massless mediator; the screening term that blinds it is the same r(k0-k) the programme derives dark energy from
  n3_area_law.py            The gap has a scaling dimension — 2/3, A3 REFUTED; A1 the estimator is calibrated against a control with a known area law (a single realisation reads 1.77 vs truth 1.0 — the correlation noise floor fakes a volume term), A2 distinction is volume-law (n_C=2.00) while integration is area-law (n_I=0.96) so the gap is EXACTLY ONE DIMENSION, A3 the registered hypothesis that scarcity forces the area law is refuted — n_I moves by 0.04 across a 12x change in coherence length: the separation is capacity-INVARIANT
  n3_ab_statistics.py       Statistics as a gauge holonomy (Aharonov–Bohm) — 3/3; AB1 the Wilson loop of the self-consistent gauge field = the flux Φ=2πq (a unit charge → +1 Dirac, a ½ charge → (−1)^q); AB2 statistical transmutation — the flux-charge composite exchanges with phase Φ/2=πq, so one flux quantum makes a FERMION (Wilczek flux attachment); AB3 the three faces of (−1)^2s agree — AB gauge phase = topological braid = 2π spin rotation. Classical abelian statistical phase; Fock {ψ,ψ†} anticommutation is the remaining frontier
  n3_condensation_boundary.py  The condensation boundary — 3/3, headline a NEGATIVE; matter cannot yet GATHER its fermions from a noise-grown gas: passive traps are frozen (no transport, K1), the naive third-law force clumps matter (every well is an amplitude hole, K2), but the envelope-normalised force IS selective — pulls on a topological core, ignores an empty well (K3). The handle exists; the missing piece is a transport current (the λ-precession candidate)
  n3_condensation_transport.py  The transport current — 3/3, the boundary LIFTED; the λ≠0 precession that regenerated the driven spin IS the transport current: a relaxational λ=0 field settles to rest, a λ>0 field sustains a persistent spiral current (field velocity 0.0002→0.05→0.10 for λ=0/0.5/1, L1); with it the κ wells now GATHER fermions from a noise-grown gas (occupancy 4/12 at λ=1 vs 0/12 flat-κ chance and 1/12 frozen, L2); and what condenses reads s=±½ with the −1 holonomy, singly (L3). PARTIAL self-assembly (~⅓ the wells, λ<1.5 stability) — K1's frozen boundary lifted
  n3_self_assembly.py       Self-assembly — 3/3; the whole chain runs itself: the mass-side selective force (the boundary's K3) and the field-side transport current in ONE co-evolution, and a molecule assembles from noise — two masses released beyond the floor BIND at the derived exclusion spacing (S1, 6/6), a mass GATHERS a ±½ out of the noise-grown gas and holds it while bound (S2, 6/6; λ=0 frozen control 4/6), the catch reads s=±½ with the −1 holonomy (S3, 6/6). TWO honest boundaries: the hold is intermittent (~60% of the run, not a locked bond) and transport is not cleanly isolated (the selective force's capture-and-hold grabs born-near defects without it) — a locked ground state + a gas source are the next rungs
  n3_gas_source.py          The gas source — 2/3, transience LIFTED, a new boundary named; a Langevin bath (η·√dt noise in the CGL) replenishes the coarsening gas: above a threshold it holds a STEADY defect gas (G1, ~30 vs ~2 defects) so the κ well is occupied 0.85 of the time (vs 0.08 un-sourced) while empty space stays empty (ambient 0.03) — the self-assembly's intermittent hold made PERSISTENT and SELECTIVE (G2). But the dense bath also nucleates integer (s=±1) & clustered defects, so what's held reads a clean ±½ only ~47% of the time (G3 ✗) — the sparse gas was pure-but-transient, the sourced gas persistent-but-impure. A "cold" source (replenish ½'s without boiling to integer defects) is the next rung
  n3_cold_source.py         The cold source — 2/3, headline a NEGATIVE; inject the fundamental spin-½ quantum (clean winding-±1 pairs, inject_defect_pair) DILUTELY instead of the hot bath's indiscriminate boil: the source is PURE (C1, 82% ±½ vs the hot bath's 47%) and SELECTIVE/local (C2, occ 0.34 vs far ambient 0.10) — but NOT persistent (C3 ✗): the clean dilute supply self-annihilates before it feeds the traps, so the clean-½ occupancy (occ×purity) 0.28 does NOT beat the hot bath's 0.40. The persistence–purity tradeoff is FUNDAMENTAL for a memoryless source; the fix is a single-occupancy LOCK (a well that holds one ½ against annihilation & excludes a second — the derived exclusion floor as a trap). The cold-source & locked-ground-state rungs are ONE
  n3_chirality.py           The chirality cascade — 3/3; the chiral leaning λ SORTS MATTER BY HANDEDNESS. A same-charge vortex pair orbits under λ with a sense that IS a handedness: absent at λ=0 (parity holds), growing with |λ| (0°/9°/18° for λ=0/0.6/1), reversing under λ→−λ (CP-like, X1+X2) AND with the charge sign (+½ +18° vs −½ −18°) — so matter & antimatter are wound OPPOSITELY by the same leaning. And it cascades: a BOUND composite inherits the same handedness (winds many times faster, confined; X3). The homochirality root — λ's one sign winds all matter of a charge the same way. Honest scope: handedness, NOT baryogenesis (the torus keeps the count charge-balanced; the drift is charge-blind)
  n3_single_occupancy.py    The single-occupancy lock — 1/3, headline a NEGATIVE that closes the loop; the hot bath's impurity is 87% CLUSTERING (the well is a sink, not a single trap), so the fix is charge polarization: a source+sink (feed one sign into matter, drain its anti-particle) removes annihilation and lets like-charge (Pauli) repulsion exclude a second. It works PARTWAY (L1 ✓, clean-single-½ 0.07→0.24, ~3×) but does NOT complete (L3 ✗, caps ~¼) and even a singly-occupied well reads clean only ~65% (L2 ✗) — because a FREE gas ½ carries only soft ~1/r repulsion, not the hard DERIVED exclusion floor s⋆ that only BOUND structures carry (imposing it crudely backfires). Resolution: matter's fermions are single-occupancy (Pauli) because they are BOUND objects carrying the no-cloning floor — the self-assembly molecule, not a free defect. Closes the loop to n3_exclusion
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
