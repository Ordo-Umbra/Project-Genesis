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
- **Monte-Carlo lattice gauge sampling** — exact SU(2) heat-bath / Cabibbo–Marinari SU(N ≥ 3) / overrelaxation updates of the Wilson ensemble with Numba-JIT kernels, optional exact coupling to a quenched sector-matter field ψ, plus Wilson-loop, Creutz-ratio, and Polyakov-loop instruments for measuring confinement (σ > 0) and coherence retention with jackknife errors,
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
  gauge.py             Lattice gauge connection on the Ψ∈ℂ³ sectors (U(1)/SU(2)/SU(3))
  gauge_mc.py          Monte-Carlo sampling of exp(−β_g·S_W + g_m·Re[ψ†Uψ]): SU(N) heat-bath, overrelaxation, Wilson/Polyakov loops
  gauge_mc_kernels.py  Numba JIT kernels for the MC layer (~50–100× the reference path)
  annealed_matter.py   Joint (ψ, U) ensemble: thermal sector matter with gauge back-reaction
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
  Thermal_Sector_Program.md  Synthesis: the full thermodynamic N⋆=3 program and its verdicts
tests/                 331 checks across the engine, instruments, and physics
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
| Capacity κ drives selection toward `N⋆ = 3` | **Conditional / transient** — a 3-well S-optimal band appears *while the field is actively coarsening*, but in steady state selection runs to more sectors; traced to how ΔI is measured (below) | [Phase diagram](#the-phase-diagram-of-n3-selection) |
| A gauge connection is the minimal structure restoring coherence under local rotations (§2–3) | **Demonstrated** (U(1)/SU(2)/SU(3)) — covariant coherence is gauge-invariant to machine precision while naive coherence is scrambled; a pure-gauge connection has zero curvature; runs on the real Ψ∈ℂ³ sectors | [Gauge connection](#gauge-connection--coherence-under-local-rotations) |
| The Yang–Mills equations are the S-stationarity conditions (§3.2) | **Demonstrated** — gradient ascent on `S = coupling·coherence − stress` monotonically raises S and drives the lattice YM residual → 0 (SU(2)/SU(3)); curvature is enriched ~2.6× on the sector walls ("gluons as boundary modes") | [Yang–Mills dynamics](#yangmills-dynamics--gradient-ascent-on-s) |
| Confinement: Wilson loops obey an area law with σ > 0 (§4.A) | **Measured** — Monte-Carlo ensembles of exp(−β_g·S_W): the SU(2) 2-D instrument reproduces the *exact* analytic σ(β_g) within errors at every scanned coupling (calibration), and SU(3) in 3-D shows σ > 0 with sub-percent errors across β_g ∈ [1, 5], decreasing with β_g, with confined Polyakov loops throughout. Small lattices, no continuum limit — lattice-units signatures, not physical σ | [Monte-Carlo confinement](#monte-carlo-confinement--the-measured-area-law) |
| The three-sector junction network exists and survives as a *thermal* state — matter and gauge co-fluctuating | **Measured melting curve, in 2-D and 3-D** — in the joint annealed (ψ, U) ensemble, a colour-neutral junction network exists in equilibrium *only* for P=3 (P=2 structurally cannot form one; P≥4 cannot fit its palette on 3-fold junctions even when thermal — exactly zero in the annealed state in both dimensions), survives with slight thermal roughening up to a measured melting temperature T ≈ 0.2, and melts above it. In 3-D the network is junction *lines* and an order of magnitude denser (neutrality ≈ 0.55 vs ≈ 0.06), yet melts at the same T — the dimensional argument survives thermodynamics. Integration retention with full back-reaction is rank-ordered in P; curvature localization remains absent | [Annealed matter](#annealed-matter--the-junction-network-as-a-thermal-state) |
| With the sector basis dynamical (fractions unpinned), the S_P symmetry breaks in the 3-state Potts class | **Measured — both independent exponents agree** — removing the fraction pin lets the system *choose* its sector spontaneously; the order–disorder point at T_c ≈ 0.1 is a real transition: χ_max(L) ∝ L^b with **b = 1.60 ± 0.22** (0.6σ from the exact Potts γ/ν = 26/15, 7.2σ from the pinned crossover), the Binder collapse puts **Potts ν = 5/6 inside its band**, and equilibrium magnetisation histograms are unimodal (no weak-first-order contamination in 2-D). In 3-D — where Potts predicts *first order* — the energy-histogram instrument overturned the suggestive hysteresis reading: every pooled histogram is unimodal, Δe shrinks with L (no latent heat at ≲0.001 resolution), and the hysteresis is explained as kinetic coarsening — the 3-D transition is continuous or unresolvably weakly first order, a candidate genuine deviation from the discrete-Potts expectation. Dichotomy fully explained: pinned fractions (coexistence) ⇒ smooth dissolution; free fractions ⇒ spontaneous S_P breaking in the Potts class | [Unpinned S_P transition](#the-unpinned-s_p-transition--sector-choice-in-the-potts-class) |
| Dynamical capacity κ(x,t) at the transition: scarcity governs where S sends the system | **Measured** — with the engine's capacity field (consumed by load, regenerating with slack, diffusing) co-evolving in the joint (ψ, U, κ) system and gating the coherence coupling locally: **κ develops its trough exactly in the critical region at every consumption strength** (⟨κ⟩_min = 0.34 → 0.02 as c = 0.5 → 15) — capacity is consumed where distinction peaks; the ordered phase shows the κ-as-soil **wall deficit** (κ_wall = 0.22 vs κ_bulk = 0.39 at strong consumption); **scarcity shifts the transition down** (the m-drop moves from T ≈ 0.078 to ≈ 0.060); and the S-optimum **relocates from the deep-ordered phase to the critical point as capacity binds** (argmax_T S: T = 0.03 for c ≤ 2, T = 0.085 for c ≥ 5) — scarcity taxes integration but not distinction, so a capacity-bound S-climbing system is pushed toward criticality while abundance lets it rest in deep order | [κ at criticality](#dynamical-capacity-at-criticality) |
| The S-functional carries a critical signature — and can select criticality | **Measured** — on the unpinned ensembles across T_c: the distinction term **ΔC peaks exactly at the transition** (0.0342 at T = 0.115 vs 0.0288/0.0178 on the flanks, L = 48) while the standing coherence I falls through it order-parameter-like (0.48 → 0.334). The S = ΔC + κ·w·I optimum therefore **sweeps across T_c as the integration weight varies** — sitting just above T_c for w = 0.02, *at* T_c for w ≈ 0.05, and at the ordered end for w ≥ 0.1: there is a weight window in which the theory's own functional selects the critical neighbourhood | [S at criticality](#the-s-functional-at-criticality) |
| The melting boundary in the (g_m, T) plane: crossover or transition? | **Mapped — and the crossover is confirmed by finite-size scaling** — T_melt(g_m) rises monotonically with the matter–gauge coupling (0.093 → 0.132 for g_m = 1 → 8: the coupling *stabilises* the junction network). The four-size ladder L ∈ {24, 32, 48, 64} gives χ_max(L) ∝ L^b with **b = −0.09 ± 0.30** — consistent with zero at 0.3σ and **6.0σ away** from 2-D-Ising-like transition scaling (b = 1.75). Nothing diverges: the junction network dissolves smoothly, as expected where the sector basis is explicitly (not spontaneously) selected | [Melting boundary](#the-melting-boundary--tg_m-map-and-the-crossover-verdict) |
| The N⋆=3 selection survives a *fluctuating* (thermodynamic) gauge sector | **Measured crossover** — with the converged P-sector networks quench-coupled to SU(P) Wilson ensembles, the integration retention R(g_m) is rank-ordered (bigger palettes pay a higher gauge tax) and selection at three survives exactly where κ·w·neutrality·R exceeds the ΔC gap — washing out to P=4 below a sharp (w, g_m) threshold. **Negative sub-verdict**: curvature does *not* localize on walls or junctions in equilibrium (quenched sector matter never frustrates the gauge field — the per-link constraints are integrable); the deterministic 2.6× wall enrichment is a relaxation-dynamics effect, not a thermodynamic one | [Thermal N⋆=3 selection](#thermal-n3-selection--the-sector-field-in-a-fluctuating-gauge-ensemble) |
| The S-functional rewards an interior sector optimum | **Achieved in 2-D and 3-D** — with volume-conserving dynamics (persistent junctions) and a *topological* neutrality term (full-palette junctions, non-collinear with ΔC), `S = ΔC + κ·neutrality` is maximized at **exactly three sectors**, robust across seeds/weights. In 3-D three wins ~10× over four (triple *lines* vs sparse quadruple *points*) | [Topological selection](#topological-selection--an-interior-optimum-at-three) |

The honest through-line: the *machinery* of URP sectorisation is reproducible, the boundary-cost half of its free-energy argument is measured, and — after localizing why naive selection failed (ΔI vanishes at equilibrium; coherence magnitude is collinear with ΔC) — a junction-resolving dynamics plus a topological neutrality term reproduces an interior optimum at **three in both 2-D and 3-D**, echoing the gauge paper's §6. The count is no longer a free parameter; it falls out of the junction geometry.

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
- **Dynamical κ changes the verdict — under scarcity.** With `--dynamic-kappa` at default rates, capacity texture forms (walls depleted) and coarsening slows, but selection stays ΔC-dominated (k = 6 still wins; in quasi-steady state ΔI ≈ 0, so S reduces to wall energy). In the **capacity-scarce regime** (`--kappa-consumption 50 --kappa-recovery 0.02`), the selection flips to **k = 3** — and the full phase diagram below shows this is not a one-off.

### The phase diagram of N⋆=3 selection

`experiments/phase_diagram.py` sweeps the capacity-scarcity space — for each (β, consumption `c`, recovery `r`) it runs the wells sweep with dynamical κ and records which well count maximizes time-averaged S. The full run (3 β × 6 c × 5 r × 5 wells × 3 trials = 1350 runs, 24³, gravity on) gives a strikingly clean structure:

```
β = 0.09 — winning well count k (★ = k=3)        consumption c (scarcity →)
  c \ r |  0.01   0.03   0.05    0.1    0.2
    5.0 |   6      6      6      6      6      abundant κ  → over-fragmentation
   15.0 |   3★     3★     3★     3★     3★
   30.0 |   3★     3★     3★     3★     3★     ← the k=3 band
   50.0 |   3★     3★     3★     3★     3★
   80.0 |   4      4      4      4      4
  120.0 |   2      2      2      2      2      over-scarce → collapse to two
```

Three findings, each with its caveat:

1. **There is a robust k=3 band in consumption.** Three wells are S-optimal for `c ≈ 15–50`, with too-abundant capacity selecting more wells and over-scarcity collapsing to two. The band is **identical across all three β** (0.03, 0.09, 0.2) — only the abundant-capacity corner shifts (k = 5 at β = 0.03 vs k = 6 above). 45 of 90 cells select k = 3; mean win margin 0.23, but some boundary wins are razor-thin (min 0.03).
2. **Recovery rate does essentially nothing here.** Every row is uniform across `r` ∈ [0.01, 0.2] — selection is set by *consumption* (which fixes the standing capacity level), not by how fast capacity regenerates. Honest flag: recovery may only bite on longer timescales than the 400-step / 50-step-window measurement.
3. **The decisive caveat — k is the *imposed* well count, not the emergent N.** "k = 3 wins" means the field given a 3-well potential reaches higher quasi-steady S than with 2/4/5/6 wells. The *realized* domain count in those winning cells is **not** 3: at `c = 30–50` it is ≈ 1 (near-collapsed), at `c = 15` it is ≈ 39 (fragmented). So this measures a preference for *three available sector-types*, not a field that settles into three domains. Bridging "3 wells are S-optimal" to "the field forms 3 sectors" is the remaining work — likely needing the Ψ∈ℂ³ model (which sustains genuine three-domain structure) coupled to dynamical κ, rather than the scalar multi-well stand-in.

The honest headline: **N⋆=3 selection is real, robust, and a capacity-*consumption* phenomenon — a Goldilocks band, β- and recovery-independent — but it currently selects the imposed well count rather than a cleanly emergent three-domain field.** Reproduce with `python experiments/phase_diagram.py` (or `--quick`).

### Coupling κ to the three-component model — what it revealed about the S-functional

*Running the Ψ∈ℂ³ sector field with the dynamical capacity field.*

The natural next step was to bridge "3 wells are S-optimal" to "the field forms 3 domains" by running the *three-component* model — which sustains genuine three-way domains — coupled to dynamical κ (`step_multiphase_kappa`; experiment `experiments/multiphase_kappa.py`). Two robust results, one expected and one more important than the experiment itself:

- **κ scarcity genuinely arrests coarsening** (verified, `tests/test_multiphase_kappa.py`): capacity depletes at the dense walls, integration stalls there, and the domain network is pinned — a starved run holds many more domains than a fully-integrating one. The emergent domain count lands in the single-to-low-double digits, far closer to 3 than the scalar model's 1-or-39.
- **But P=3 is *not* robustly selected** — and the reason exposes a flaw in the S-functional itself. Sweeping candidate component counts P ∈ {2…6} against consumption, the S-maximizing P is **5–6, not 3**, with emergent N of 10–40. Tracing the components reveals why: in any quasi-steady state **ΔI → 0** (measured 0.000000–0.000004), because ΔI is defined as the *transient* reduction in field curvature — the rate of smoothing — which vanishes once coarsening stalls. So `S = ΔC + κΔI` collapses to `S ≈ ΔC` = total wall energy, which **monotonically rewards more sectors**. The whole capacity-weighted integration half of the functional is inert in steady state.

This relocates the open problem. The earlier "scarcity selects 3" result was real but **transient** — it caught the system mid-coarsening, where ΔI is briefly non-zero and contributes; it is also why the scalar phase diagram's margins were razor-thin and timescale-sensitive. The robust steady-state verdict is that S, as currently measured, is just distinction and always prefers fragmentation.

**The missing ingredient looked like a *standing* measure of integration** — coherence shared across domain boundaries that persists at equilibrium, rather than a one-step smoothing rate. So the next experiment (`experiments/standing_integration.py`, `multiphase.coherence_integration`) implemented the nonlocal coherence `I = Σ_a ⟨η_a(x)·η_a(x+δ)⟩ exp(−decay·|δ|)` — the multi-component form of `I[φ] = ∫∫ K(x,x')φ(x)φ(x')`. It half-worked, in an instructive way:

- **The standing measure does survive equilibrium** (verified) — a static coherent field keeps `I > 0` where the transient ΔI read zero. That part of the fix is real.
- **But `S = ΔC + κ·I` still has no interior optimum** — it flips monotonically from many sectors (w→0) to two (any w>0). The reason, measured directly: **ΔC and coherence are collinear**, `corr(ΔC, −I) = +1.00` at short range and `+0.998` at long range. Both just track wall density, so their weighted sum is monotonic and the optimum sits at a boundary.

This sharpens the frontier one more turn. An interior `N⋆` needs an integration term that is *not* collinear with wall area — and the gauge paper's own §6 says exactly what distinguishes three: **topology**, the 120° Y-junctions that let three (and only three) sectors tile into neutral composites. The integration measure has to be junction-/neutrality-aware, not coherence-magnitude. A second observation blocks the direct route for now: in the κ-pinned scarce regime the frozen domains produce **zero clean triple junctions**, so that topological structure would first have to be coaxed into forming. That — a junction-resolving dynamics plus a topological integration term — is the real next experiment.

### Topological selection — an interior optimum at three

That next experiment (`experiments/topological_selection.py`) was built, and it closes the loop. Two ingredients, each answering one half of the obstacle above:

- **Junction-resolving dynamics** (`step_multiphase_conserved`). Plain Allen–Cahn coarsens without bound, so junctions are transient. Making the dynamics **volume-conserving** — a global Lagrange multiplier that fixes each phase's total (subtract the spatial mean of the bulk drift per component) — prevents any phase from being eliminated. The field settles into a *stable* multi-domain tiling whose 120° triple junctions persist (≈ 40, holding, where the κ-pinned regime had zero).
- **A topological neutrality term** (`full_palette_junction_density`). Count the junctions whose neighbourhood carries the *complete* colour palette — the discrete form of §6's neutrality criterion. The junction geometry does the selecting: a 2-D junction is 3-fold, so it can show *all* the colours only when the palette is exactly three. Measured across palette sizes P, the quantity is non-zero **only at P=3** (`~0.008`) and exactly **zero at P=2,4,5,6**, robust across seeds. Crucially it is **not** collinear with ΔC (which is flat in P).

With it, `S = ΔC + κ·w·neutrality` is **maximized at exactly three sectors for every positive weight** — the first time the S-functional in this repo shows an interior sector optimum, and it lands on three:

```
 P |  Yjunc |      ΔC | neutrality   S(w=0.2)
 2 |      0 | 0.00348 |   0.00000    0.00348
 3 |     39 | 0.00403 |   0.00759    0.00555   ← maximum
 4 |     62 | 0.00398 |   0.00000    0.00398
 5 |    107 | 0.00423 |   0.00000    0.00423
 6 |    119 | 0.00425 |   0.00000    0.00425
```

A faithful in-silico echo of the gauge paper's §6: SU(3) is selected because three sectors, and only three, tile into colour-neutral composites.

**And it survives in 3-D** (`--dim 3`) — for a sharper reason than the 2-D case. The worry was that three rode on junctions being 3-fold, and that 3-D, where Plateau's laws make generic vertices 4-fold (tetrahedral), would select four. The opposite holds: the full-palette density is still sharply peaked at **P=3** (≈ 0.03) and an order of magnitude smaller at P=4 (≈ 0.003), robust across seeds. The reason is dimensional — the locus where *three* domains meet is a **line** (1-D, abundant), while *four* meet at a **point** (0-D, sparse), so a three-colour palette saturates the triple-line network while four lights up only rare vertices. Three wins by the dimensionality of the neutral locus; P=4 is now faintly non-zero (vs exactly zero in 2-D) precisely because those sparse vertices exist.

**The honest boundary** that remains: the neutrality measure *operationalizes* §6 rather than deriving its full gauge/anomaly content — what is emergent, not assumed, is that conserved P=3 dynamics produce stable full-palette junctions while P≥4 (almost) cannot. But the *count* — three, in both 2-D and 3-D, by a clean geometric mechanism — is no longer a free parameter. The full account is in [`Docs/Narrowing_the_N3_Question.md`](Docs/Narrowing_the_N3_Question.md).

### Gauge connection — coherence under local rotations

With genuine three-sector structure in hand, the next layer of the derivation
(§2–3) is the **gauge connection**: when the sector-membership field `ψ(x) ∈ ℂ³`
(its components are exactly the R/G/B sectors) is rotated *locally*, `ψ(x) → g(x)ψ(x)`,
the naive inter-site coherence is scrambled — path-dependent transport, the drop
in integration the theory describes. A connection `U_μ(x) ∈ U(N)` on the lattice
links is the minimal structure that repairs this. `project_genesis.gauge` +
`experiments/gauge_coherence.py` demonstrate it for U(1), SU(2), SU(3) and on the
real coarsened sector field:

- **Covariant coherence is gauge-invariant** — `Σ Re[ψ†(x) U_μ(x) ψ(x+μ̂)]` is unchanged (to ~10⁻¹⁴) under the joint transform `ψ→gψ, U_μ→g(x)U_μ g(x+μ̂)†`, while the naive `Σ Re[ψ†(x) ψ(x+μ̂)]` shifts by O(field size). The connection carries `ψ(x+μ̂)` back into `ψ(x)`'s colour frame before comparing.
- **It restores destroyed coherence** — a uniform field (coherence 392) scrambled by a local rotation (→ ~0) is brought back to 392 exactly by the pure-gauge connection built from the same `g`. That is the §2.2 claim made literal: the connection is the minimal structure preserving ΔI under local ΔC.
- **Curvature is coherence stress** — a flat (pure-gauge) connection has zero Wilson action `Σ Re Tr(1 − P_{μν})`; a connection whose parallel transport is path-dependent has positive action. This is the lattice `Tr(F_{μν}F^{μν})` the derivation reads as the cost of residual coherence stress.

### Yang–Mills dynamics — gradient ascent on S

The URP picture is gradient ascent on `S = coupling·(covariant coherence) − (curvature stress)`, and §3.2 derives the Yang–Mills equations as the *stationarity conditions* of that S. `project_genesis.gauge.flow_step` + `experiments/yang_mills_flow.py` run that flow (Wilson gradient flow: links move along `exp(ε·force)·U` with `force = −TA[U_μ Ω_μ]`, `Ω = coupling·ψ(x+μ̂)ψ(x)† + staple`; ψ optionally relaxes covariantly). Two results:

- **The flow ascends S and the YM residual → 0.** From a hot random start, `S` rises monotonically while the lattice equation-of-motion residual `‖TA[U_μ Ω_μ]‖` drives toward zero (SU(2): 1.63 → 0.03; SU(3): 1.65 → 0.02 over 160 steps) — the configuration converges onto a solution of the discrete Yang–Mills equations, with the matter current as source. Links stay in SU(N) throughout. A pure-gauge flow (no matter) relaxes the curvature to ~0 (the flat vacuum); a correctness check confirms the staple identity `Σ_μΣ_x Re Tr[U_μ A_μ] = 4(N·n_plaq − S_Wilson)`.
- **Curvature localizes on the sector walls** (§4.3.4, "gluons as boundary modes"). Fixing ψ to a real three-sector field and flowing only the connection, the resulting curvature density is **enriched ~2.6× on the domain walls** — 57% of the total curvature sits on the 33% of sites that are walls. The gauge field carries the colour-frame mismatch between adjacent sectors, exactly where the derivation places the gluonic excitations.

**Honest scope:** this is deterministic gradient flow — the gauge equations of motion and the boundary-mode localization. Thermodynamic confinement signatures (Wilson-loop area law, string tension) are properties of the *ensemble*, and are measured by the Monte-Carlo layer below.

### Monte-Carlo confinement — the measured area law

`project_genesis.gauge_mc` samples the Boltzmann ensemble `exp(−β_g·S_W)` of the Wilson action with an exact SU(2) heat-bath (Kennedy–Pendleton at strong effective coupling, Creutz at weak, so link updates never stall), Cabibbo–Marinari subgroup updates for SU(3), and exact microcanonical overrelaxation. `project_genesis.gauge_mc_kernels` provides Numba JIT versions of the sweeps and loop measurements (~50× SU(2), ~100× SU(3) over the pure-Python reference; the two paths share the same random-draw order and are held equal by `tests/test_gauge_mc_numba.py`). `experiments/confinement_sigma_scan.py` uses them to measure the string tension honestly, with binned-jackknife errors:

- **Calibration against an exact result.** 2-D SU(2) is exactly solvable — plaquettes decouple and `σ_exact(β_g) = −ln[I₂(2β_g)/I₁(2β_g)]`. On a 16² lattice (1600 measurements/point, 1 heat-bath + 2 overrelaxation sweeps per compound update) the measured Creutz ratio χ(2,2) tracks the exact curve at all seven couplings β_g ∈ [0.75, 4], worst pull 2.2σ. The instrument measures what it claims to measure.
- **SU(3) confinement in 3-D.** On an 8³ lattice, χ(2,2) = σ runs from 1.73(16) at β_g = 1 to 0.1290(3) at β_g = 5 — positive at >10σ at every coupling, monotonically decreasing (the lattice-units echo of asymptotic freedom), with `|⟨P⟩| ≲ 0.003` (confined, Z₃ unbroken) across the whole scan. This is the expected behaviour of 2+1-D Yang–Mills, which confines at all couplings — and it is now measured here, not quoted.
- **At 4× the volume the picture holds and sharpens** (`--size-su2 32 --size-su3 16 --r-max 4`): the 32² SU(2) calibration still tracks the exact curve (worst pull 2.7σ over 7 couplings), and the 2-D Creutz *plateau* — χ(3,3) and χ(4,4) agreeing with χ(2,2), as the exact pure-area law demands — is now resolved (e.g. β_g = 3: χ22 = 0.2711(15), χ33 = 0.2672(57); β_g = 4: χ22 = 0.2004(11), χ33 = 0.1992(33), χ44 = 0.209(14)). In 3-D SU(3), σ is finite-size stable between 8³ and 16³ at every coupling, and the larger loops now expose what 8³ could not: χ(3,3) sits systematically *below* χ(2,2) (β_g = 5: 0.0956(6) vs 0.1298(3)), the standard perimeter/short-distance contamination of small Creutz ratios — so the asymptotic string tension is somewhat below χ(2,2), approached from above as the loops grow.
- Reproduce with `python experiments/confinement_sigma_scan.py` (≈ 10 minutes; `--quick` for a smoke run). Artifacts: per-point Wilson loops, Creutz ratios with jackknife errors, Polyakov loops, a σ(β_g) figure, and a summary with explicit verdict lines.

**Honest scope:** these are lattice-units signatures on small lattices — no continuum limit, no scale setting, no physical σ in MeV/fm, and 2-D/3-D rather than 3+1-D. What the instrument establishes is that the URP-side gauge sector, sampled as a proper thermodynamic ensemble, exhibits the area law and confined order parameter the theory's §4.A points to — with a calibration line showing the estimator is trustworthy.

### Thermal N⋆=3 selection — the sector field in a fluctuating gauge ensemble

The topological-selection optimum at three was a *deterministic* result. `experiments/n3_thermal_selection.py` asks whether it survives when the gauge sector actually fluctuates: each converged P-sector network is embedded as a quenched matter field `ψ ∈ ℂ^P` and coupled to an SU(P) Wilson ensemble, `weight ∝ exp(−β_g·S_W + g_m·Σ Re[ψ†Uψ])`. The matter term enters each link's heat-bath weight *exactly* (as a staple addition, handled by the same quaternionic projection the sampler already uses; Cabibbo–Marinari now covers every SU(N ≥ 3), so all palette sizes get exact updates). Three measurements, with binned-jackknife errors:

- **Integration retention is coupling-controlled and rank-ordered.** `R = ⟨covariant coherence⟩/naive coherence` — the fraction of the deterministic integration the fluctuating connection actually delivers — depends almost entirely on g_m (β_g moves it by only a few percent) and *falls with the palette size at every coupling* (at g_m = 8, β_g = 3: R ≈ 0.90 / 0.80 / 0.71 / 0.62 for P = 2/3/4/5). Larger groups have more directions to wander, so bigger palettes pay a higher gauge tax on integration — a group-rank effect the deterministic picture cannot see.
- **The selection has a measured washout threshold.** Dressing the selection functional with the retention, `S = ΔC + κ·w·neutrality·R`, selection at P=3 holds in 28/48 (w, β_g, g_m) cells and fails in a coherent wedge: exactly where `κ·w·neutrality·R` falls below the ΔC gap to P=4, the argmax slips to four. The boundary tracks a level set of `w·R` (w=0.05 needs g_m ≥ 4; w=0.1 needs g_m ≥ 2; w=0.2 needs g_m ≥ 1; w=0.5 needs g_m ≥ 0.5). So three is not unconditional: it is selected whenever the matter–gauge coupling preserves enough integration to pay for the junctions — and the crossover is now a measured curve, not a hypothesis.
- **Negative verdict: no curvature localization in equilibrium.** Across every (P, β_g, g_m) cell the junction/bulk and wall/bulk curvature ratios are 1.00 within errors (max deviation 2.8%). The reason is structural: the per-link matter constraint is rank-1 (`U·ψ(x+μ̂) = ψ(x)` plus a free orthogonal completion), and composing those transports around any plaquette returns ψ to itself — the constraints are *integrable*, so a zero-curvature connection satisfies all of them simultaneously and quenched sector matter never frustrates the gauge field. The deterministic "curvature enriched ~2.6× on walls" is therefore a statement about *transient relaxation dynamics*, not about the equilibrium ensemble.

Reproduce with `python experiments/n3_thermal_selection.py` (≈ 15 minutes; `--quick` for a smoke run). Artifacts: per-cell retention and curvature decompositions with errors, the full selection table, a three-panel figure, and a summary with explicit verdict lines.

**Honest scope:** the matter field is quenched — the gauge ensemble does not back-react on the sector network, so this isolates one direction of the coupling. Gauge groups of different rank are compared at equal (β_g, g_m) with the per-group mean plaquette reported rather than equalised. 2-D, one matter configuration per palette size, and the same structural caveats as the deterministic topological-selection result it extends. The annealed-matter experiment below closes the loop.

### Annealed matter — the junction network as a thermal state

`project_genesis.annealed_matter` + `experiments/n3_annealed_matter.py` remove the quench: the sector field ψ(x) ∈ ℂ^P and the SU(P) links co-evolve in **one joint Gibbs measure** — `exp(−β_g·S_W + (1/T)·[g_m·Σ Re(ψ†Uψ′) + u·Σ|ψ_a|⁴ − λ_c·N·Σ_a(f_a − f⁰_a)²])` — with single-site Metropolis matter updates (Numba-JIT, held equal to a pure-Python reference at the bit level) interleaved with the exact matter-coupled heat-bath. The corner potential u makes sectors form; the soft fraction pinning λ_c is the thermodynamic analogue of the volume-conserving dynamics (without it the system trivially freezes into one sector); T is the matter temperature. Junction observables use a noise-robust measure (smoothed amplitudes, ordered-cell labels, wide junction neighbourhoods) calibrated so a converged cold network scores clearly nonzero while pure label noise scores exactly zero.

What the melting scan (P ∈ {2,3,4,5}, T ∈ [0.02, 1.6], each point an independent fresh-start ensemble, binned-jackknife errors) measures:

- **The colour-neutral junction network exists in equilibrium only for P = 3.** Across every temperature, the full-palette junction density is nonzero solely for the three-colour palette — P = 2 structurally cannot form junctions, and P ≥ 4 cannot fit its whole palette on generically 3-fold 2-D junctions even when the field is free to fluctuate. The deterministic structural selection is a property of the *thermal state*, not just of a particular relaxation path.
- **The network has a measured melting temperature.** Neutrality holds (with slight thermal roughening — it *rises* a little from T = 0.02 to 0.05) and then collapses: 0.063 → 0.068 → 0.038 → 0 across T = 0.02/0.05/0.1/0.2, giving T_melt ≈ 0.2 at these couplings (β_g = 3, g_m = 4, u = 1, λ_c = 20, 48²). Below melting, the annealed selection functional S = ΔC + κ·w·neutrality·R is maximised at P = 3 through the junction channel; above melting that channel is empty for *every* palette, so the P-comparison there rides on ΔC alone and says nothing about junctions — the meaningful selection statement is the sub-melting one.
- **Integration retention with full back-reaction is again rank-ordered.** R (measured against the cold network's coherence) falls smoothly with T and with P at every temperature (at T = 1.6: 0.64/0.43/0.32/0.25 for P = 2/3/4/5) — the group-rank tax persists when the matter fluctuates too.
- **Curvature localization stays absent** (max |ratio − 1| = 0.3% across all cells) — even annealed junction cores do not pin gauge curvature, extending the quenched-matter negative verdict.

Reproduce with `python experiments/n3_annealed_matter.py` (≈ 10 minutes; `--quick` for a smoke run).

**The 3-D version** (`--dim 3`, 16³): the deterministic dimensional argument — three-way junctions are *lines* in 3-D (abundant), four-way meetings are *points* (sparse) — survives the thermodynamics intact. The annealed P=3 ensemble carries a junction-line network an order of magnitude denser than in 2-D (neutrality ≈ 0.55 vs ≈ 0.06), melting at the same T ≈ 0.2 (0.546 → 0.534 → 0.464 → 0.001 across T = 0.02/0.05/0.1/0.2), while the annealed P=4 state's full-palette density is **exactly zero at every temperature** — even sharper than the deterministic snapshot comparison, where under-converged P=4 networks retain a faint transient signal. Retention stays rank-ordered and curvature localization stays absent (≤ 0.9%) in 3-D too.

**Honest scope:** the corner potential and pinned fractions explicitly break local SU(P) — deliberately, since in the URP picture the sectors define preferred frames and the gauge freedom is the local relabelling the connection repairs. One seed network per palette, groups compared at equal (β_g, g_m, T).

### The melting boundary — T(g_m) map and the crossover verdict

`experiments/n3_phase_boundary.py` maps where the junction network melts in the (g_m, T) plane (P = 3, β_g = 3, two lattice sizes, binned-jackknife errors) and asks whether the melt is a genuine phase transition:

- **The boundary is monotone**: T_half = 0.093 / 0.097 / 0.111 / 0.132 for g_m = 1/2/4/8 — the matter–gauge coupling *stabilises* the junction network against thermal fluctuations, quantifying the direction the washout-threshold result pointed.
- **The melt is a crossover, not a transition — confirmed by finite-size scaling.** The two-size comparison first suggested it (susceptibility bumps that don't grow with volume); `experiments/n3_scaling_ladder.py` settles it with a four-size ladder L ∈ {24, 32, 48, 64} at g_m = 4, parabolic peak fits, and a weighted log–log fit of the peak height: **χ_max(L) ∝ L^b with b = −0.09 ± 0.30** — consistent with zero at 0.3σ and 6.0σ away from 2-D-Ising-like transition scaling (b = γ/ν = 1.75). Over a 2.7× range of L (which would grow χ_max by ×5.7 at such a transition) the peak height simply does not move, and its location wanders rather than converging — there is no diverging peak. This is also what the model's structure predicts: the corner potential and pinned fractions select the sector basis *explicitly*, leaving no symmetry to break spontaneously at the melt (which is also why Binder-cumulant analysis is not meaningful here). The junction network dissolves smoothly.

Reproduce with `python experiments/n3_phase_boundary.py` and `python experiments/n3_scaling_ladder.py` (≈ 10 minutes each; `--quick` for smoke runs).

**Honest scope:** the ladder is one coupling point (g_m = 4) and one exponent channel (the order-parameter susceptibility); T_half is measured on the thermal junction density, whose absolute normalisation is measure-specific — the *shape* and monotonicity of the boundary are the robust content. Two of the four χ-maximum locations sit at the scan edge (the high-T tail drifts mildly upward from disorder fluctuations); the height-scaling verdict is insensitive to this, since the heights stay flat everywhere.

### The unpinned S_P transition — sector choice in the Potts class

The crossover result begged its own follow-up: the fraction pinning that stabilises the junction network also *forbids* the S_P permutation symmetry from breaking — all P sectors are always present by construction, so the melt can only be interface dissolution. `experiments/n3_potts_transition.py` removes the pin (`frac_penalty = 0`). Nothing in the model now prefers any sector — the corner potential is exactly S_P-symmetric — so at low temperature the system must **choose** one spontaneously: the sector basis has become dynamical in the meaningful sense, and the order–disorder point can be a genuine transition. For P = 3 in 2-D the natural universality class is the 3-state Potts model (γ/ν = 26/15 ≈ 1.733).

Measured on a cold-started ladder L ∈ {16, 24, 32, 48} with the gauge sector kept as in the melting studies (β_g = 3, g_m = 4), the Potts magnetisation `m = (P·n_max − 1)/(P − 1)`, its susceptibility, and the Binder cumulant, all with binned-jackknife errors:

- **Spontaneous sector order appears and melts at T_c ≈ 0.11**: m runs from ≈ 0.98 (cold) through a sharpening drop to the disorder plateau, and the susceptibility peaks are fully bracketed and interior (T_peak = 0.107–0.113 across all sizes), growing 3.6 → 4.7 → 8.5 → 20.0 from L = 16 to 48.
- **The scaling exponent lands on Potts**: χ_max(L) ∝ L^b with **b = 1.60 ± 0.22** — 7.2σ away from the pinned model's flat crossover and 0.6σ from the exact 3-state Potts value. Same model, one constraint removed, and the melt changes character exactly as the symmetry analysis predicts.

Two follow-ups tie the identification down (`n3_potts_nu.py`, `n3_potts_3d.py`; the full narrative lives in [`Docs/Thermal_Sector_Program.md`](Docs/Thermal_Sector_Program.md)):

- **The second exponent agrees.** Binder-cumulant data collapse — single-parameter scaling U₄(T, L) = f((T−T_c)·L^{1/ν}) — selects ν ≈ 1.0 with a 2×-residual band [0.53, 1.60]: the exact Potts ν = 5/6 sits inside. Getting this honestly required a thermalisation lesson: a shallower first pass produced spurious *negative* Binder values near T_c (slow melting straddling the measurement window mimics bimodality); magnetisation histograms at equilibrium are cleanly **unimodal**, which simultaneously cleans the collapse and rules out the weak-first-order alternative in 2-D.
- **The 3-D prediction is probed — and the sharper instrument overturned the first reading.** The Potts class predicts the 3-D transition is *first order*. The hysteresis/Binder scan (L ∈ {8…16}) found suggestive signatures: a persistent hot/cold window and Binder minima deepening with L. But the decisive observable is the **energy histogram** (`n3_latent_heat.py`): a first-order transition has latent heat, so the per-site energy distribution must go bimodal at the transition. It does not — every pooled hot+cold histogram is cleanly **unimodal** at every size, and the branch energy separation Δe ≤ 0.005 *shrinks* with L, even while the magnetisation branches still disagree. Same energy, different order: the "hysteresis" was **kinetic** (slow 3-D coarsening), not phase coexistence. Verdict: **no latent heat at Δe ≲ 0.001 resolution — the 3-D transition is continuous or unresolvably weakly first order at these sizes**, a candidate genuine deviation from the naive 3-D Potts expectation (the continuous-field, gauge-coupled realisation need not inherit the discrete model's order).

Reproduce with `python experiments/n3_potts_transition.py`, `n3_potts_nu.py`, `n3_potts_3d.py`, `n3_latent_heat.py` (≈ 25 minutes each; `--quick` for smoke runs).

### The S-functional at criticality

With a genuine critical system in hand, the program's last question comes home to the theory's central object: **what does `S = ΔC + κ·ΔI` do at a real phase transition?** `experiments/n3_s_criticality.py` measures the full functional — ΔC from the amplitude gradient energy, κ from the load, and the *standing* nonlocal coherence for the integration half — across temperature through T_c, on the same unpinned ensembles (deep thermalisation, jackknife errors, the Potts magnetisation measured alongside as the anchor):

- **The distinction term carries the critical signature.** ΔC(T) rises to a sharp peak *at the transition* (0.0342 at T = 0.115 vs 0.0288 at 0.085 and 0.0178 at 0.60, L = 48) — walls and fluctuations are densest exactly at criticality — while the standing coherence I falls through T_c order-parameter-like (0.48 → 0.334 disorder plateau).
- **The S optimum sweeps across T_c with the integration weight.** Because its two halves pull opposite ways and *cross at the transition*, `argmax_T S` sits just above T_c for w = 0.02, **at T_c for w ≈ 0.05**, and at the ordered end for w ≥ 0.1. There is a window of integration weights in which the theory's own functional selects the critical neighbourhood — the ordered-but-maximally-fluctuating regime — rather than deep order or disorder.

Reproduce with `python experiments/n3_s_criticality.py` (≈ 15 minutes; `--quick` for a smoke run).

**Honest scope:** κ here is the diagnostic proxy `1/(1+load)`, not the dynamical capacity field (the next section promotes it); the weight w is a free dial, so the sharp claim is the *structure* (ΔC peaks at T_c; the optimum crosses T_c through a finite w-window), not any particular w value. One coupling point, L ∈ {32, 48}.

### Dynamical capacity at criticality

`experiments/n3_kappa_criticality.py` promotes κ from the diagnostic proxy to the **dynamical capacity field** the URP engine actually uses — `∂_t κ = D_κ∇²κ + r(κ₀ − κ) − c·load·κ` — co-evolving with the joint (ψ, U) system and **gating the matter–gauge coherence coupling locally** (the link x → x+μ̂ couples with weight κ(x)·g_m, on both the matter and link updates; the same κ-gated-integration structure as the engine's field dynamics). The system is an adaptive steady state rather than a fixed-Hamiltonian ensemble — deliberately: URP capacity is a driven, consumed resource. Scanning temperature × consumption strength on the unpinned model:

- **Capacity troughs exactly at criticality.** ⟨κ⟩(T) dips in the critical region at every consumption strength (trough at T = 0.085–0.10 for c = 0.5…15, deepening from 0.34 to 0.02) — the load that consumes κ *is* the distinction term, and ΔC peaks at T_c. The theory's own feedback loop, closed and measured.
- **The κ-as-soil wall deficit appears in the thermal state.** In the ordered phase at strong consumption, κ_wall = 0.22 vs κ_bulk = 0.39 — capacity is depleted on the domain walls, the thermal counterpart of the engine's corpus-rooting picture.
- **Scarcity destabilises sector order.** The transition shifts down with consumption (m-drop from T ≈ 0.078 at weak c to ≈ 0.060 at c = 15): binding capacity gates the coherence that holds sectors together.
- **Scarcity relocates the S-optimum to the critical point.** With S assembled from the *measured* capacity (S = ΔC + ⟨κ⟩·w·I): at abundant capacity (c ≤ 2) the optimum sits in the deep-ordered phase; once the budget binds (c ≥ 5) it jumps to the ΔC peak at criticality. Scarcity taxes integration everywhere but leaves distinction untouched — so a capacity-bound S-climbing system is pushed toward the edge, while abundance lets it rest in order. This is the thermodynamic echo of the repo's earliest capacity verdict ("sector selection is a scarcity phenomenon — invisible with abundant κ"), now stated for criticality itself.

Reproduce with `python experiments/n3_kappa_criticality.py` (≈ 20 minutes; `--quick` for a smoke run).

**Honest scope:** the adaptive (ψ, U, κ) system is not a Gibbs measure — steady states, not equilibrium averages; κ gates only the coherence channel (the quartic term is untaxed by construction); one lattice size (32²), one w slice for the optimum statement (the JSON carries the components for any w); the transition-shift estimate uses the coarse m-drop locator, not a full FSS study.

**Honest scope:** the exponents are few-size fits at one (β_g, g_m) point — consistency with Potts from two independent exponents plus unimodal histograms, not a universality proof (that would want larger L and corrections to scaling). The Binder-crossing T_c estimate is unstable at this precision; peak positions and the collapse give the quoted T_c. The ν-collapse estimator carries interpolation bias on coarse T grids (quantified in `tests/test_potts_universality.py`).

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

1. **Monte-Carlo confinement.** Gradient-flow Yang–Mills dynamics are now in (`gauge.flow_step`: S-ascent → YM residual 0, gluon-like wall modes). The remaining lattice signatures — Wilson-loop **area law**, **string tension**, the **deconfinement temperature** the derivation quotes (~150–170 MeV) — are properties of the finite-temperature ensemble, so they need a Monte-Carlo sampler (heat-bath / Metropolis on the Wilson action) rather than deterministic flow. That is the natural next build on top of the existing `gauge.py` primitives.
3. **Promote the F(N) fit to two free coefficients** by measuring an independent information-gain proxy for `b(β, κ)`, rather than inverting stationarity — the missing half of a non-circular free-energy test.
4. Higher-order field dynamics — second-order time derivatives (∂²φ/∂t²) for the wave-like behavior in the full Lagrangian (the current model is the overdamped limit).
5. Replace the sine pinning potential with the true `−(β/4)(∇φ)⁴` gradient-quartic via an implicit/stabilized integrator.

Earlier roadmap items now implemented: ~~coherence potential V(x,t)~~, ~~nonlocal integration functional I[φ]~~, ~~agent-agent interaction~~, ~~S-functional-driven agents~~, ~~matplotlib visualization~~, ~~emergent gauge sectorisation (measurement + wall tension + Ψ∈ℂ³ Y-junctions)~~, ~~dynamical capacity field κ~~, ~~κ-as-soil corpus coupling~~, ~~`(c, r, β)` phase diagram~~, ~~κ × Ψ∈ℂ³ coupling~~, ~~standing nonlocal coherence~~, ~~junction-resolving dynamics + topological selection of three (2-D and 3-D)~~, ~~gauge connection on the Ψ∈ℂ³ sectors~~, ~~Yang–Mills gradient-flow dynamics (S-ascent → YM residual 0, boundary-mode curvature)~~.

## Theory Reference

The foundational theory document remains in:

- `Docs/The Universal Recursion Principle (URP) _260312_170343.txt`

That document describes the broader URP framing this sandbox is intended to explore in executable form.

A working note distilling this repo's own investigation — how the `N⋆=3`
question narrowed across six experiments into a sharp, buildable next target —
is in [`Docs/Narrowing_the_N3_Question.md`](Docs/Narrowing_the_N3_Question.md).
It is the standalone, pick-up-cold version of the *Findings so far* table.
