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
  sector_seeds.py      κ-as-soil rooting of stored structure into the thermal sector field
  soil_clusters.py     Percolation-style connected-component analysis of the fertile-soil mask
  topological_charge.py Geometric (Berg–Lüscher) topological charge of the ψ∈ℂ^N (CP^(N-1)) field
  gauge_topology.py    4-D SU(N) clover topological charge, gauge cooling, susceptibility
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
  The_Generative_Gap.md      Capstone: the distinction–integration gap, and the ordinal→functor→instanton bridge
  The_Measured_Bridge.md     Closing synthesis: ordinals→functors→instantons as one chain, κ≈0.22 in the 4-D vacuum
  Capacity_As_Gravity.md     κ as the framework's gravity: a universal, mass-sourced, √(D/r)-screened attraction
  The_Emergent_Cosmos.md     Capstone (Act II): κ→gravity→matter→structure→cosmos, with toolkit map and frontiers
tests/                 509 checks across the engine, instruments, and physics
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
  test_gauge_topology.py
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
| The scarcity-driven relocation of the S-optimum: a drift or a level crossing? | **Level crossing — first-order-like in the optimum** — mapping the (consumption, weight) plane, S(T) carries *two* competing local maxima at once (a deep-ordered one where coherence is highest, a critical one at the ΔC peak). The global optimum T\*(c) is a **step function**: it sits at T = 0.03 then jumps ~3× to T ≈ 0.08 with no intermediate value, exactly where the order parameter ΔS = S_ordered − S_critical crosses zero. The boundary c\*(w) sweeps monotonically (0.59 → 14.7 across w = 0.045 → 0.06): more integration weight makes the ordered phase more valuable, so more scarcity is needed to abandon it. Two S-maxima trading global rank, not a continuous drift | [S-landscape](#the-s-landscape--a-level-crossing-not-a-drift) |
| Memory recall (κ-as-soil corpus rooting) survives the melt — or fails at criticality | **Recall fails first, before order** — porting the scalar engine's exact κ-as-soil rooting rule (a seed roots only where local κ ≥ 0.3, and rooting consumes κ) to the thermal ψ∈ℂ³ ensemble: the recall capacity (fertile-soil fraction) collapses across the melt (0.98 → 0.00) and reaches ≈ 0 at T = 0.08 **while the order parameter is still m = 0.44** — the soil goes barren before the field disorders, so a capacity-bound system can *hold* structure but can no longer *regenerate* what it loses. Recall is nonzero only in a cold, low-consumption corner (heat and scarcity each starve it), and is self-limiting (rooting consumes soil: 0.99 → 0.22 over 140 rootings). The repo's two halves — memory corpus and thermal capacity — meet, and agree | [Memory recall](#memory-recall-at-criticality--where-can-stored-structure-re-root) |
| Can capacity *regeneration* rescue memory from the critical collapse? | **Yes — recovery is the dial, and recall can outlive order** — scanning the κ recovery rate r (steady state κ = r/(r + c·load)): the recall edge climbs monotonically with r (T = 0.057 → beyond the scan as r = 0.05 → 0.8), so the seed-rooting collapse was a property of the *default* rate, not a barrier. At r = 0.8 the recall capacity never falls below ½ while the field's order dies at T ≈ 0.097 — recall stays 0.6–0.87 out to T = 0.20 where m ≈ 0.03 (fully melted): the prerequisite for memory outlives order. And the recall curve is non-monotonic — it dips at T_c and recovers on both sides — because the distinction load that consumes κ peaks at criticality, making the transition itself the memory bottleneck | [κ-recovery rescue](#can-regeneration-rescue-memory--the-κ-recovery-dial) |
| Does "recall outlives order" survive the thermodynamic limit, or is it a small-lattice artifact? | **Survives — it is size-independent** — the overtaking claim was measured at one size (28²), so a four-size ladder L ∈ {16, 24, 32, 40} at r = 0.8 tests it. The decisive scalar is the recall capacity *at the melt* — the fertile-soil fraction at the temperature where order has fallen to m = ½. It is **flat at ≈ 0.71 across every size** (margin above ½ = +0.211, +0.199, +0.211, +0.206), and the weighted 1/L → 0 extrapolation gives margin → **+0.208 with a near-zero slope**: at the melt a ~71% majority of soil stays fertile even in the thermodynamic limit. The recall and order curves themselves lie on top of one another across sizes — both are already at their L → ∞ shapes. Recall outliving order is a property of the model, not of the lattice | [Finite-size recall](#does-recall-outlive-order-as-l--) |
| Is surviving memory globally *usable* (connected), or does it fragment into islands at criticality? | **Stays connected — but only just: it bends at T_c without breaking** — a recalled seed can *spread* only across connected fertile soil, so recoverability is a percolation question, not a fraction. Labelling the fertile mask (κ ≥ 0.3) into connected components on the torus: the percolation strength P∞ collapses from ≈ 1.0 to **0.41 at T ≈ 0.11** and the percolation susceptibility χ **spikes ~40× (0 → 43)** there — the classic near-threshold signature, in the same critical region where recall dips. Yet a system-spanning path survives at *every* temperature (spanning probability ≥ 0.97). Decisively, the fertile fraction dips **below the random site-percolation threshold** p_c ≈ 0.593 (crossing it at T = 0.104) while still spanning: the thermal field's spatial structure — barren soil confined to thin domain walls, fertile bulk staying contiguous — keeps memory connected at a density where randomly-placed soil would already be islands. Connectivity is the tighter constraint than fertility, and criticality nearly severs it, but the backbone holds by a thread and heals on both flanks | [Memory connectivity](#is-surviving-memory-connected--the-percolation-of-fertile-soil) |
| When two memories want the same ground, who wins — and what decides it? | **The first mover locks out the second, until capacity heals fast enough to let it in** — rooting *consumes* κ, so the first seed to write a region draws the soil down and can turn a rival away. Whether a later memory can overwrite an earlier one is set by the same recovery-rate dial that rescued recall. Writing seed A, letting the soil recover for a fixed delay, then attempting rival seed B at the same site: the overwrite probability P(B roots \| A rooted) climbs **0.00 → 0.06 → 0.62 → 1.00** as r = 0.1 → 0.2 → 0.4 → 0.8, a sharp **persistence→plasticity crossover at r ≈ 0.36** (exactly where the local ⟨κ⟩ heals back past the 0.3 rooting threshold). A single site accepts **0.3 → 5.0** consecutive memories as r rises, and the last-writer territory flips from a first-mover-locked mosaic (B claims 0%) to a fully plastic one (B overwrites 100%). Slow-healing capacity makes **write-once** memory; fast-healing capacity makes it **plastic** — one dial trades permanence against the ability to learn anew | [Competing memories](#competing-memories--persistence-vs-plasticity) |
| In 3-D, where domain walls are *surfaces*, is memory connectivity more robust or more fragile? | **More fragile — memory de-percolates in 3-D, and the load physics beats the geometry** — the naive geometric prediction favours 3-D: a surface partitions a volume less than a line partitions a plane, and the site-percolation threshold is far lower (p_c ≈ 0.312 simple-cubic vs 0.593 square). But running the same fertile-soil percolation scan in both dimensions overturns it. In 2-D the backbone bends but holds (P∞ dips to 0.45, spanning ≥ 0.98). In **3-D it genuinely de-percolates**: P∞ collapses to **0.02** and the spanning probability falls to **0.15** at criticality — the fertile soil shatters into disconnected islands. The cause is the κ dynamics, not the geometry: the fertile fraction craters to **0.17** in 3-D versus 0.57 in 2-D, plunging well below even the lower 3-D threshold, because the denser 3-D critical structure (surface walls, six neighbours, the order-of-magnitude-denser junction network) consumes far more capacity at the transition. The favourable geometry is overwhelmed by a deeper capacity trough — adding a dimension makes memory *harder* to keep, not easier | [3-D connectivity](#memory-connectivity-in-3-d--surface-walls-vs-line-walls) |
| Can faster recovery rescue the 3-D de-percolation, and how much more does 3-D need than 2-D? | **Yes — the recovery dial reconnects 3-D too, it just has to be turned up further** — since steady-state κ = r/(r + c·load), a high enough recovery must refill the capacity trough and reconnect the soil in any dimension. Scanning r and taking the *worst* case over the critical-temperature window: the backbone re-percolates (worst-case spanning climbs back through ½) at **r\* ≈ 0.67 in 2-D and r\* ≈ 0.85 in 3-D** (~1.3× more). The clearer signal is the fertile-fraction curve — at every intermediate rate the 3-D worst-case fertile fraction lags 2-D substantially (0.46 vs 0.86 at r = 1.0), the whole rescue curve shifted right, so 3-D reaches its (lower) percolation threshold later. The same dial that rescued 2-D recall (and set the persistence↔plasticity crossover) also rescues 3-D connectivity: distributed memory in more dimensions is not un-keepable, it simply demands a proportionately faster capacity refill | [3-D rescue](#can-recovery-rescue-3-d-connectivity) |
| Does the universe select its "Platonic forms" by S, and does capacity decide which one manifests? | **Yes — N⋆=3 is a *conditional* manifestation: the integrated three-fold form is what abundant capacity buys, and scarcity fragments** — treating the P-sector networks as a corpus of candidate forms, integration (the full-palette neutrality) is a strict **P = 3 monopoly** (0.0071 at P=3, ≈ 0 for all others) while distinction ΔC rises with P. Since `S = ΔC + κ·w·ΔI`, the three-fold form is selected only where the capacity-gated integration bonus clears the distinction gap to the maximally-fragmented form — a clean hyperbolic **κ·w island** in the (capacity, weight) plane. Below it the universe can afford only distinction and jumps to the fragmented form. A thermal spot-check confirms sustained ⟨κ⟩ falls with consumption (0.98 → 0.73), and from the κ-criticality result it craters to ≈ 0.02 at the melt — so scarcity drives the field out of the three-fold island. The N⋆=3 selection is not a bare constant but the manifestation abundant capacity permits; perfection (the neutral, unified form) appears where conditions allow, exactly as S = ΔC + κΔI demands | [Form selection](#platonic-forms-selected-by-s) |
| Is the distinction–integration gap (the field's echo of `I(F) < C(F)`) a gradual shortfall or something sharper? | **A structural cliff at the three-fold threshold — and capacity can't close it** — measuring distinction as *all* triple junctions the field represents and integration as the *full-palette neutral* ones it binds (so integration ⊆ distinction, `I ≤ C` exactly): the integrated fraction φ = I/C is **1 at P = 3** (every represented junction is integrated — "complete") and collapses to **0 for P ≥ 4** (junctions represented, none integrable — "incomplete"), with the raw gap C − I *widening* as expressivity P climbs. Sweeping the capacity field shows the separation is **structural, not a matter of effort**: capacity changes the *density* of distinction by >10× (scarcity fragments, abundance coarsens) but leaves φ pinned — 1 for the three-fold form, and for P = 4 only the *accidental* palette-completeness of a fragmented mess (φ ≈ 0.3) that *washes out* as capacity organizes the field. Capacity is a distinction dial, not an integration dial; past the three-fold expressivity threshold, integration cannot be bought — the field's echo of "no computation within F proves G_F; F must be extended" | [The generative gap](#the-generative-gap-measured) |
| Does the sector field have genuine instanton content — the physical side of the gap the bridge maps to? | **Yes — the ψ∈ℂ³ field is a CP² field, and its instantons switch on through the melt** — the normalized sector field is a CP^(N-1) field, the textbook 2-D stand-in for the QCD vacuum (asymptotic freedom, confinement, a mass gap, a θ-vacuum, integer-charge instantons). Built the topological-charge instrument the gauge sector lacked — a geometric (Berg–Lüscher) estimator that is **exactly integer, gauge-invariant, and reads Q = +1 on a constructed CP¹ winding** (a single instanton resolved directly). With UV dislocations removed by cooling, the physical topological susceptibility χ_top is **≈ 0 in the cold ordered vacuum and switches on right at the melt** (peaking ≈ 0.014 at T ≈ 0.17, where order m ≈ 0.02) — topological activity is the disordered phase's, organized by the same criticality as everything else. The topological fraction of the action is a small sub-dominant minority (κ_top ≈ 0.014): its *value* sits well below the framework's 4-D QCD κ ≈ 0.22 (expected — a different theory and dimension at arbitrary couplings), but its *role* — coherent topology as the κ ≪ 1 minority that does the integrating — is exactly the structural claim the bridge rests on | [Instanton content](#the-instanton-content-of-the-sector-field) |
| Is the logic↔physics correspondence a real *functor* (structure-preserving), or just a loose analogy? | **A genuine functor on objects — the vacuum's topology is a path-independent function of the integration level** — instantiating the bridge `F : logic → vacuum` on the field: a *reflection ladder* (gentle cooling = the integration dynamics, the field's `F_{n+1} = F_n + Con(F_n)`) climbs integration `I = ⟨\|z̄·z'\|²⟩` from a random "pure-representation" start, and the functor's image is the topological content `T`. As integration climbs **0.33 → 0.94**, the topology falls **0.22 → 0.02** — one ladder through two instruments, a monotone (contravariant) map. The decisive test: running the ladder at different cooling *rates*, `T` at matched `I` agrees to a **1.9% relative scatter** — the topological image depends only on the integration level reached, **not on the history** of reaching it. So `F` is well-defined on objects: a genuine functor, not merely a correlation. And the single action-descent gradient ∇S drives *both* ladders at once (naturality, `∇S : D ⇒ κ·Int`). **Honest limit:** the direction is contravariant because the thermal regime's topology is the disordered instanton *gas*, not the coherent *condensate* that integrates the QCD θ-vacuum — the covariant, condensate-side functor is what the 4-D build would test | [The functor](#the-functor-logic--vacuum-measured) |
| Can the genuine 4-D SU(3) instanton content be measured — the theory the framework's κ ≈ 0.22 actually lives in? | **Stage 1 built and validated: the gauge sector now has a topological-charge instrument** — the 4-D SU(3) Wilson ensemble runs (numba, dimension-agnostic) and `project_genesis/gauge_topology.py` adds the **clover** field-theoretic topological charge `Q = (1/32π²)Σ ε F F` with gauge cooling. Validation: pure-gauge configs read **Q = 0 exactly**, the clover and single-plaquette definitions agree, the mean plaquette climbs with β_g (0.475 → 0.682), and cooled Q is **quantized and Z-renormalized** — single-instanton configs read \|Q\| ≈ **0.84** (Z ≈ 0.84, the standard coarse-lattice suppression), 0 exact. The vacuum shows **topological freezing**: it tunnels freely across sectors at strong coupling (Q from −5 to +3, χ_top ≈ 0.0012) and sticks in one sector toward weak coupling — the known critical slowing of topology, seen directly. This is the instrument and a first susceptibility, on a small 8⁴ lattice in lattice units; the precise normalization (Z → 1 via gradient flow), scale-setting, and the instanton/perturbative condensate split that would test κ ≈ 0.22 are the next stage — now reachable, because the charge exists | [4-D SU(3) topology](#4-d-su3-topological-charge-stage-1) |
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

### The S-landscape — a level crossing, not a drift

The κ-at-criticality result found the S-optimum *relocating* from deep order to criticality as capacity binds. `experiments/n3_s_landscape.py` resolves the **character** of that relocation across the (consumption c, integration weight w) plane. The key observation is that S(T) is not single-peaked: it carries **two competing local maxima at once** — an *ordered* one at low T (where the standing coherence I is highest, so integration is cheap while capacity is abundant) and a *critical* one at the ΔC peak near T_c. The global optimum is whichever is taller.

- **The optimum jumps, it does not drift.** Splitting each S(T) profile at the magnetisation mid-drop and tracking the two branch maxima, the global optimum location T\*(c) is a **step function**: it sits at T = 0.03 (deep ordered) and then *teleports* to T ≈ 0.08 (critical), a ~3× jump, with **no intermediate value ever selected** — precisely where the level-crossing order parameter ΔS = S_ordered − S_critical passes through zero. This is a level crossing (two maxima trading global rank), the S-landscape analogue of a first-order transition, not a continuous migration of a single peak.
- **The boundary c\*(w) sweeps monotonically.** Because weights are pure post-processing (`--from-json` re-analyses a saved scan without re-running the Monte Carlo), the boundary is mapped finely: c\* = 0.59 / 1.56 / 8.76 / 14.7 for w = 0.045 / 0.05 / 0.055 / 0.06, flanked by the always-critical (w ≲ 0.04) and always-ordered (w ≳ 0.07) regimes. c\*(w) *increases* with w: a larger integration weight makes the high-coherence ordered phase more valuable, so more scarcity is needed to send the optimum to the edge — the two dials trade off exactly as the functional's structure predicts.

Reproduce with `python experiments/n3_s_landscape.py` (≈ 40 minutes), or re-weight a saved run instantly with `--from-json artifacts/n3_landscape/n3_s_landscape.json` (`--quick` for a smoke run).

**Honest scope:** "first-order-like" describes the *optimum-location* observable (a discontinuous argmax), not a thermodynamic transition of the ensemble — the underlying melt is still the continuous Potts crossover mapped earlier. κ and w are dials, not derived quantities; the jump discreteness is partly set by the temperature-grid spacing (the two peaks are genuinely separated, but T\*'s exact jump size inherits the grid), and the (c, w) boundary is at one lattice size and coupling point.

### Memory recall at criticality — where can stored structure re-root?

This experiment joins the two halves of the repository that had never met: the **memory corpus** (stable structure stored and re-seeded to keep the universe from collapsing) and the **thermal sector program** (the dynamical capacity field, measured to trough at criticality). The corpus re-roots seeds under a **κ-as-soil** rule (`engine._plant_seed`): a recalled seed unfolds only where local capacity is sufficient (κ ≥ `corpus_kappa_threshold` = 0.3), and rooting *consumes* that capacity. `project_genesis.sector_seeds` carries the *exact* rule — same threshold/cost/blend — to the thermal ψ∈ℂ³ ensemble, and `experiments/n3_seed_rooting.py` asks it of a fluctuating field. Because capacity collapses at the melt, the prediction is sharp: recall should fail at criticality, where the soil is barren.

- **Recall dies before order.** The recall capacity — the fertile-soil fraction, i.e. the fraction of sites where a seed would root under the engine's own criterion — collapses across the melt (0.98 below T_c → 0.00 above). Crucially it reaches ≈ 0 at T = 0.08 where the order parameter is still **m = 0.44**: the soil goes barren *before* the field disorders. A capacity-bound system can **hold** the structure it has but can no longer **regenerate** what it loses — memory fails first.
- **A memory phase diagram.** recall_capacity(T, c) is nonzero only in the cold, low-consumption corner: heat and scarcity each independently starve recall, and a binding capacity budget shrinks the recall-possible region even in the ordered phase (recall at the coldest T falls with rising consumption).
- **Recall is self-limiting.** Actually rooting seeds (via `plant_sector_seed`, which draws κ down on success) exhausts the soil: 400 recall attempts root 140 seeds and pull the fertile fraction 0.99 → 0.22, so recall spreads into fresh ground rather than piling onto the same spot — the consumption half of the κ-as-soil rule, demonstrated on the thermal field.

Reproduce with `python experiments/n3_seed_rooting.py` (≈ 35 minutes; `--quick` for a smoke run).

**Honest scope:** the κ-as-soil rule's physical content in this thermal model is the *gating* — below T_c any patch is coherent regardless of κ, above T_c no fertile soil exists, so "can a seed root here" (κ ≥ threshold) is exactly the meaningful, measurable question, and it is the engine's own criterion; a separate seed-*persistence* dynamics is confounded here (the ordering coherence force acts on a planted domain independent of κ) and is deliberately not claimed. One lattice size (28²), one coupling point, the scalar engine's default threshold/cost.

### Can regeneration rescue memory? — the κ-recovery dial

If capacity scarcity is what kills recall at the melt, then capacity's *recovery* term should be able to rescue it. The steady-state capacity balances consumption against regeneration, `κ = r / (r + c·load)`, so a faster recovery rate `r` keeps κ above the rooting threshold to higher temperatures. `experiments/n3_recall_recovery.py` scans `r` against temperature (fixed consumption) and measures both the recall capacity and the Potts order:

- **Recovery rescues recall.** The recall edge (the temperature where recall capacity falls below ½) climbs monotonically with regeneration rate — T = 0.057 → 0.066 → 0.072 → 0.080 → *beyond the scan window* as r = 0.05 → 0.8. Faster-regenerating soil sustains memory to higher temperature; the seed-rooting collapse was a property of the *default* recovery rate, not an absolute barrier.
- **Recall capacity outlives order.** The order edge also rises with r (κ gates coherence, so more capacity means a bit more order), but the recall edge climbs *faster* and overtakes it: at r = 0.8 the recall capacity never drops below ½ across the scan while the field's order dies at T ≈ 0.097 — recall stays at 0.6–0.87 out to T = 0.20 where the Potts magnetisation is m ≈ 0.03 (fully melted). The prerequisite for memory (fertile soil) survives into the disordered phase.
- **The bottleneck *is* criticality.** At high recovery the recall curve is non-monotonic: it dips to a minimum right at T_c and *recovers on both sides*. This is the ΔC-peaks-at-criticality result seen through capacity — the distinction load that consumes κ is maximal at the transition (dense walls and fluctuations), so the critical region is the single worst place for memory, flanked by the ordered phase (low load, high κ) and the deep-disordered phase (low load, κ restored by fast recovery).

Reproduce with `python experiments/n3_recall_recovery.py` (≈ 35 minutes; `--quick` for a smoke run).

**Honest scope:** "recall capacity" is the fertile-soil gating predicate (κ ≥ threshold, the engine's own rooting criterion), as in the seed-rooting section — not a separate persistence claim. Recovery raises the order edge as well as the recall edge; the result is that recall *overtakes* order, quantified by the two-edge comparison, at one consumption strength and lattice size.

### Does recall outlive order as L → ∞?

The overtaking above is the program's strongest claim, and it was measured at a single lattice size (28²). Both the recall edge and the order edge are finite-size proxies for sharp thresholds, so the honest worry is that the gap between them closes as the lattice grows — the Potts magnetisation has a fat finite-size tail above T_c, and if the order edge climbs toward the true T_c faster than the recall edge does, the overtaking could be an artifact. `experiments/n3_recall_finite_size.py` settles it with a four-size ladder L ∈ {16, 24, 32, 40} at fixed high recovery (r = 0.8).

In the overtaking regime the recall edge runs *beyond* the scan window (recall never falls below ½), so the raw edge gap is +∞ and cannot be extrapolated. The decisive, always-finite observable is the **recall margin at the melt** — the fertile-soil fraction evaluated at the very temperature where the field has half-disordered (m = ½), minus ½. It is exactly positive when recall outlives order, but as a bounded number it extrapolates to the thermodynamic limit.

- **The overtaking is size-independent.** The recall capacity at the melt is **flat at ≈ 0.71 across every size** — margin = +0.211 (L=16), +0.199 (L=24), +0.211 (L=32), +0.206 (L=40) — with no trend. At the temperature where order has fallen to ½, roughly 71% of the soil is still fertile, on every lattice.
- **It survives L → ∞.** The weighted least-squares extrapolation in 1/L gives **margin → +0.208 with a near-zero slope** (−0.04 in 1/L): the intercept at the thermodynamic limit is firmly positive. Recall outliving order is a property of the model, not of the lattice size.
- **Both curves are already at their L → ∞ shapes.** The recall(T) and order(T) curves lie on top of one another across all four sizes (the recall dip at T_c and its recovery on both flanks are size-stable), and the order edge converges quickly to T ≈ 0.090 — so nothing about the crossing is a small-volume accident.

Reproduce with `python experiments/n3_recall_finite_size.py` (≈ 35 minutes; `--quick` for a smoke run).

**Honest scope:** one recovery rate (r = 0.8) and one consumption/coupling point, four sizes up to 40²; the observable is the same fertile-soil gating predicate throughout. The order edge is the crude m = ½ crossing, not a Binder-cumulant T_c (its finite-size drift is small here, ~0.093 → 0.090, and does not change the sign of the margin). The extrapolation is a two-parameter fit to four points — a thermodynamic-limit consistency check, not a proof.

### Is surviving memory *connected*? — the percolation of fertile soil

The finite-size result is a scalar: at the melt ~71% of the soil stays fertile. But 71% *fertile* is not 71% *usable*. A recalled seed roots anywhere fertile, yet it can only **spread** — grow a remembered domain — across a *connected* patch of fertile ground. So whether memory is globally recoverable at criticality is a **percolation** question, not a fraction: is the surviving soil one spanning continent, or an archipelago of disconnected islands? `project_genesis/soil_clusters.py` labels the fertile mask (κ ≥ 0.3) into connected components on the torus, and `experiments/n3_memory_clusters.py` scans temperature at fixed high recovery (r = 0.8), measuring the percolation strength P∞ (largest-cluster fraction), the spanning probability, and the susceptibility χ (mean finite-cluster size). The reference number is the 2-D site-percolation threshold **p_c ≈ 0.593**: a *randomly* occupied mask percolates above it and fragments below it.

- **The backbone is driven to the edge at T_c.** P∞ collapses from ≈ 1.0 in the ordered phase to **0.41 at T ≈ 0.11**, and χ **spikes ~40× (0 → 43)** at the same temperature — the textbook near-threshold percolation signature. The critical region isn't just where recall dips; it's where the *connected structure* of surviving memory nearly comes apart. The cluster montage shows it directly: a solid continent below T_c, a tenuous spanning filament threaded through a field of disconnected coloured islands at T_c, a re-consolidated continent above.
- **But it never actually breaks.** The spanning probability stays ≥ 0.97 at every temperature — a system-crossing fertile path survives throughout. Memory bends at criticality; it does not fragment.
- **It percolates *below* the random threshold.** Decisively, the fertile fraction dips **below p_c = 0.593** in the critical window (crossing it at T = 0.104, reaching 0.55 at T = 0.11) *while still spanning*. A randomly occupied lattice at that density would already be islands — but the thermal field's fertile soil is not random: barren sites concentrate on the thin domain walls where κ is consumed, leaving the fertile bulk contiguous. Spatial structure buys memory its connectivity, keeping it globally recoverable at a density where chance alone would sever it.

The reading: **connectivity, not fertility, is the tighter constraint on recoverable memory** — and criticality attacks it hardest — but the geometry of where capacity is spent keeps the backbone intact by a thread.

Reproduce with `python experiments/n3_memory_clusters.py` (≈ 35 minutes; `--quick` for a smoke run).

**Honest scope:** one recovery/consumption/coupling point, one lattice size (40²); "spanning" is the standard occupies-every-row-or-column criterion on the torus, and the montage panels are single representative snapshots (the quantitative claims use the snapshot-averaged P∞, χ and spanning probability). p_c ≈ 0.593 is the *random* site-percolation threshold quoted for contrast — the fertile mask is spatially correlated, which is exactly the point of the comparison, not a claim that this system shares the random-percolation universality class.

### Competing memories — persistence vs plasticity

Every measurement so far followed a *single* recalled memory. But the κ-as-soil rule has an immediate consequence the moment a *second* memory wants the same ground: rooting **consumes** capacity (`κ *= 1 − cost`), so the first seed to root in a region draws the soil down and can **lock out** a rival — unless the capacity heals first. Whether a later memory can be written over an earlier one is therefore decided by the very same dial that rescued recall: the recovery rate `r`. `experiments/n3_memory_competition.py` isolates the write mechanism — a rival roots only where local κ has healed back above threshold (the engine's own gate; the subsequent coherence competition between domains is confounded by the ambient field and deliberately not claimed) — and scans `r`.

- **The persistence→plasticity crossover.** Plant seed A, let the soil recover for a fixed delay, then attempt rival seed B at the same site. The overwrite probability P(B roots | A rooted) climbs **0.00 → 0.00 → 0.06 → 0.62 → 1.00** as r = 0.05 → 0.1 → 0.2 → 0.4 → 0.8, a sharp crossover at **r ≈ 0.36**. The mechanism is explicit in the numbers: the mean local capacity just before B is attempted rises 0.05 → 0.09 → 0.13 → 0.21 → **0.33** → 0.45, and the overwrite switches on exactly as ⟨κ⟩ heals back past the 0.3 rooting threshold. Below the crossover the first memory owns the ground and a rival is turned away (**write-once**); above it the soil heals fast enough that a later memory writes over the earlier one (**plastic**).
- **Rewrite capacity.** How many memories a single site accepts in succession before the soil is exhausted rises from **0.3** consecutive roots at r = 0.02 (the ground can barely hold even the first) to **5.0** at r = 0.8 — faster healing lets one location carry a longer palimpsest of overwritten memories.
- **Territory.** A wave of A across the lattice, then a wave of B: at slow recovery (r = 0.1) the first mover roots everywhere and B overwrites **0%** — a frozen mosaic locked to whoever arrived first; at fast recovery (r = 0.8) B overwrites **100%** — a plastic mosaic owned by whoever wrote last. (At the very slowest r = 0.02 the soil is so starved that even the first wave only claims 47%, the rest barren — capacity scarcity caps memory before competition even begins.)

The synthesis: **the recovery rate is a single dial that trades permanence against plasticity.** Slow-healing capacity gives durable, write-once memory that resists being overwritten; fast-healing capacity gives plastic memory that can always learn something new over the old — and the crossover between the two regimes sits squarely inside the same `r` window that lets recall outlive order. Stability and adaptability are the two faces of one parameter.

Reproduce with `python experiments/n3_memory_competition.py` (≈ 15 minutes; `--quick` for a smoke run).

**Honest scope:** the capacity-gated *write* mechanism only (rooting under the κ budget), not the coherence competition that would decide long-run domain dominance — the same honest boundary as the seed-rooting section. One consumption/temperature/lattice point, one fixed inter-write delay; the crossover rate `r*` shifts with the delay and consumption (it is the healing *per delay* relative to the cost that matters), so r ≈ 0.36 is specific to these settings, not a universal constant.

### Memory connectivity in 3-D — surface walls vs line walls

Everything about memory so far — recall, percolation, competition — has lived on 2-D fields, where the barren soil concentrates on *line* domain walls. In 3-D those walls become *surfaces*, and the naive geometric prediction is that memory should get **more** robust: a 2-D surface partitions a 3-D volume far less than a 1-D line partitions a 2-D plane (you route a connected path around a surface through the extra dimension), and the site-percolation threshold is much lower — p_c ≈ 0.312 on the simple-cubic lattice versus 0.593 on the square. `experiments/n3_memory_clusters_3d.py` runs the *same* fertile-soil percolation scan (fixed r = 0.8, temperature across the melt) in both 2-D (L = 40², line walls) and 3-D (L = 16³, surface walls) and compares them head to head. **The prediction is wrong**, and the way it fails is the finding:

- **Memory de-percolates in 3-D.** In 2-D the backbone bends but holds — P∞ dips to 0.45 and the spanning probability stays ≥ 0.98. In 3-D the backbone **genuinely breaks**: P∞ collapses to **0.02** and the spanning probability falls to **0.15** at criticality. The cluster-slice montage shows it directly — a solid continent below T_c shattering into a field of disconnected islands at the transition, then re-consolidating above.
- **The cause is the capacity dynamics, not the geometry.** The fertile fraction craters to **0.17** in 3-D against 0.57 in 2-D — plunging well *below* even the lower 3-D percolation threshold, where a random mask would already be dust. The favourable geometry (route-around, lower p_c) is real but is overwhelmed by a far deeper capacity trough: the denser 3-D critical structure — surface walls instead of lines, six neighbours instead of four, the order-of-magnitude-denser junction network measured in the annealed-matter study — consumes much more capacity at the transition. Load beats geometry.

The synthesis: **adding a dimension makes memory harder to keep, not easier.** The instinct that more room means more ways to stay connected is correct in isolation, but it is the *wrong* effect to bet on — the critical structure that drains capacity grows faster with dimension than the connectivity bonus does, so the capacity trough deepens and the memory backbone that merely bent in 2-D snaps in 3-D. This is the first point in the whole program where surviving memory actually de-percolates, and it says the binding constraint on distributed memory is where capacity is *spent*, not how the space is shaped.

Reproduce with `python experiments/n3_memory_clusters_3d.py` (≈ 20 minutes; `--quick` for a smoke run).

**Honest scope:** the capacity-gated fertile mask (κ ≥ threshold) as throughout the memory work, one recovery/consumption/coupling point; 2-D at 40² and 3-D at 16³ are single sizes (no finite-size extrapolation), the susceptibility χ is not volume-normalised across dimensions, the montage panels are single-snapshot z-slices of the 3-D field (in-plane cross-sections of a tenuous spanning cluster can look more fragmented than the volume is), and the p_c values are the *random*-lattice references quoted for contrast, not a universality claim.

### Can recovery rescue 3-D connectivity?

If 3-D memory de-percolates because the capacity trough is deeper, then the fix is the dial that has recurred through all of this work: the recovery rate. Steady-state capacity is `κ = r/(r + c·load)`, so a high enough `r` must lift κ back above threshold everywhere and reconnect the soil — in any dimension. The quantitative questions are whether it does, and how much more recovery 3-D needs than 2-D. `experiments/n3_recovery_rescue_3d.py` scans `r` in both dimensions and, for each rate, takes the **worst case over a critical-temperature window** — the recovery must rescue the tightest bottleneck, not the average.

- **Recovery reconnects 3-D too.** The worst-case spanning probability climbs from 0 back to 1 as `r` rises in both dimensions; the backbone re-percolates (worst-case spanning ≥ ½) at **r\* ≈ 0.67 in 2-D** and **r\* ≈ 0.85 in 3-D**. The 3-D reconnection montage shows it directly: at r = 0.3 the z-slice is entirely barren, at r = 1.0 a blue backbone has reassembled through a field of islands, and at r = 2.4 it is a solid continent again.
- **3-D needs a faster refill.** The spanning threshold is ~1.3× higher in 3-D, but the clearer signal is the fertile-fraction curve: at every intermediate rate the 3-D worst-case fertile fraction lags 2-D substantially (**0.46 vs 0.86 at r = 1.0**) — the whole rescue curve is shifted right, so 3-D reaches even its lower percolation threshold later. (Spanning is a sharp all-or-nothing step, so its threshold ratio understates the gap the fertile curve makes plain.)

The synthesis closes the loop the whole memory arc has been tracing: **the recovery rate is the one dial behind all of it** — it rescues recall at the melt, sets the persistence↔plasticity crossover for competing memories, and now reconnects the 3-D backbone. Distributed memory in more dimensions is not un-keepable; it simply demands a proportionately faster capacity refill, because the denser critical structure that drains capacity must be out-healed. Where capacity is spent sets the problem; how fast it heals sets the answer.

Reproduce with `python experiments/n3_recovery_rescue_3d.py` (≈ 20 minutes; `--quick` for a smoke run).

**Honest scope:** worst-case over a fixed critical-T window, the same capacity-gated fertile mask; single sizes (2-D 32², 3-D 14³), one consumption/coupling point; the reconnection threshold r\* is interpolated on a coarse `r` grid and the spanning-based estimate is a step function (the fertile-fraction shift is the more robust comparison). r\* moves with consumption and the window, so the ~1.3× ratio is specific to these settings, not a universal constant.

### Platonic forms selected by S

The program's founding result is that a capacity-bounded recursive field *selects* a three-sector structure — `S = ΔC + κ·ΔI` is maximized at the three-fold colour-neutral Y-junction. `experiments/n3_form_selection.py` reframes that as the more general picture it is an instance of: a **corpus of Platonic forms** (the P-sector domain networks, P ∈ {2..6}, each realized as a stable configuration by the conserved multiphase dynamics), of which the universe *manifests* whichever climbs S highest under the prevailing conditions — and asks the question that framing makes unavoidable: **which form is selected depends on capacity.**

Each form carries two S-ingredients: **distinction ΔC** (wall density), which rises with P, and **integration ΔI** (the full-palette neutrality — a junction carrying the whole colour palette), which is a strict **P = 3 monopoly** (0.0071 at P = 3, essentially zero for every other P, because a 2-D junction is three-fold). The integration term is capacity-gated — `S_P = ΔC_P + κ·w·ΔI_P` — so the three-fold form is selected only where the capacity `κ` and integration weight `w` are large enough to lift its neutrality bonus above the distinction gap to the maximally-fragmented form.

- **A κ·w island of the three-fold form.** Mapping the selected form over the (capacity κ, weight w) plane gives a clean hyperbolic boundary: the integrated three-fold form fills the abundant-capacity / valued-integration corner (20/42 cells), and **fragmentation** — the maximally-distinct high-P form — fills the rest. At w = 0.4 the three-fold form is selected only once κ rises above ≈ 0.2; below that the κ·ΔI term cannot be paid and the S-optimum jumps to the fragmented form.
- **Capacity really moves.** A thermal spot-check on the annealed ψ∈ℂ³ field anchors where the universe sits in κ: sustained ⟨κ⟩ falls from 0.98 to 0.73 as consumption rises (0.5 → 32), and from the [κ-at-criticality result](#dynamical-capacity-at-criticality) it craters toward ≈ 0.02 at the melt. So scarcity drives the field leftward, out of the three-fold island and into fragmentation.

The reading tracks the vision this experiment set out to test: **N⋆ = 3 is not a bare constant but a *conditional* manifestation.** The integrated three-fold Platonic form is what abundant capacity buys; under scarcity the universe can afford only distinction, and fragments. Perfection — the neutral, unified form — is manifestable exactly where conditions permit, which is precisely what the `S = ΔC + κΔI` structure says: integration is the term capacity has to pay for, and where it can't, only distinction remains.

Reproduce with `python experiments/n3_form_selection.py` (≈ 3 minutes; `--quick` for a smoke run).

**Honest scope:** the forms are realized by 2-D conserved dynamics — a structural, three-fold-junction argument that need not transfer to 3-D, where vertices are four-fold (a stated limit of the whole N⋆ line). The selection phase diagram is drawn in physical capacity κ ∈ (0, 1); the thermal run and the κ-criticality result anchor *where* the field sits in κ under given conditions, but the diagram itself is the reduced S-comparison, not a full dynamical manifestation of each rival form in one shared field. Single coupling and weight ranges.

### The generative gap, measured

`Docs/The_Generative_Gap.md` reads the whole program as one asymmetry — a recursive field can *distinguish* more than it can *integrate* — and identifies it with the ordinal separation of formal systems (`I(F) < C(F) = ω₁^CK`, the gap that *is* Gödel incompleteness). `experiments/n3_capacity_separation.py` measures that gap directly, with two commensurable observables where one is a subset of the other: **distinction C** = the density of *all* triple junctions (every place the field draws a genuine distinction), and **integration I** = the density of *full-palette neutral* junctions (the distinctions it actually binds into a colour-neutral whole). Since neutral junctions ⊆ all junctions, `I ≤ C` identically; the integrated fraction is `φ = I/C`. We set out to watch integration climb toward distinction and never catch it — and found something sharper.

- **The separation is a structural cliff at the three-fold threshold.** A 2-D junction is three-fold, so a junction can carry the *whole* palette only at P = 3. Below it (P = 2) no junction forms; *at* P = 3 every represented junction is integrated (**φ = 1, "complete"**); *above* it the field represents ever more junctions (C rises with P) that carry **zero** integration (**φ = 0, "incomplete"**), the raw gap C − I widening with expressivity. Representation outruns integration exactly past three — the field's echo of the expressivity threshold above which a system is *necessarily* incomplete.
- **The gap is capacity-invariant — it can't be bought.** Sweeping the capacity field κ that gates the integration dynamics changes the *density* of distinction by more than 10× (scarcity fragments the field into many junctions, abundance coarsens it into few) — but leaves φ pinned: **1 for the three-fold form at every capacity**, and for P = 4 only the *accidental* palette-completeness of a fragmented mess (φ ≈ 0.3 at scarce capacity) that **falls toward zero as capacity organizes the field**, never becoming genuine. Capacity is a *distinction* dial, not an *integration* dial.

The reading closes the loop the capstone opened: the distinction–integration gap is not a gradual shortfall the field slowly makes up — past the three-fold threshold it is a **structural** separation, and no amount of capacity or effort closes it. To integrate what it represents, an over-expressive field must *change its structure* (drop to three) — the field's analogue of the one move Gödel leaves open: not computing longer within F, but extending F. Distinction is cheap and unbounded; integration is structural and threshold-bound; and that permanent shortfall is exactly the room the gap leaves — the generative gap that, in failing to close, is what selects the three-fold form in the first place.

Reproduce with `python experiments/n3_capacity_separation.py` (≈ 1 minute; `--quick` for a smoke run).

**Honest scope:** distinction and integration are junction-density proxies (all triple points vs full-palette ones), so φ comes out exactly binary in P by the 2-D three-fold geometry — a clean illustration of the expressivity-threshold *structure*, not a continuous measurement of a gap magnitude. The ordinal reading (`I(F) < C(F)`, incompleteness, meta-theoretic extension) is a structural analogy the testbench *illustrates*, not a theorem it proves. 2-D conserved dynamics, single coupling; the three-fold threshold is a 2-D fact and 3-D vertices differ.

### The instanton content of the sector field

The functorial-bridge paper reads logical *reflection* as physical *tunneling*: the instanton that binds degenerate vacuum sectors into a coherent θ-vacuum is the physical image of the inferential step that integrates represented distinctions. Testing that needed an instrument the gauge sector never had — a topological-charge estimator. It exists now, on a clean observation: the normalized sector field **ψ∈ℂ³ is a CP² field**, and the 2-D CP^(N-1) model is the textbook analogue of the QCD vacuum — asymptotically free, confining, with a mass gap, a θ-vacuum, and genuine **instantons** carrying integer topological charge. `project_genesis/topological_charge.py` implements the geometric (Berg–Lüscher) charge; `experiments/n3_instanton_content.py` measures the field's instanton content across the melt.

- **The instrument is exact and validated.** The geometric charge is integer to machine precision, invariant under the local CP phase (the gauge freedom), and a constructed CP¹ winding reads **Q = +1** — a single instanton resolved directly (its conjugate reads −1). Configurations are **cooled** (a leading-eigenvector relaxation that drives the action toward the Bogomolny bound 2π|Q|) to strip UV dislocations before the physical topology is read — the standard lattice route to genuine instanton content.
- **Instantons switch on through the melt.** The cooled topological susceptibility χ_top is **≈ 0 in the cold ordered phase** (a uniform field carries no winding) and **switches on right where order collapses** — climbing from 0 at m ≈ 0.99 to a peak ≈ 0.014 at T ≈ 0.17 where m ≈ 0.02, with the instanton density stepping up sharply at the transition. Topological activity belongs to the disordered phase, and it is organized by the same criticality as everything else in the program — now on the gauge/vacuum side of the bridge.
- **The field's own "κ" is a sub-dominant minority.** The bridge paper identifies the URP integration constant κ ≈ 0.22 with the *fraction of the vacuum action that is topological rather than perturbative*. The CP² field has an exact analogue via the Bogomolny bound, `κ_top = 2π⟨|Q|⟩ / ⟨S⟩`, and it peaks at **≈ 0.014** — a small coherent minority, the rest of the action smooth and perturbative. Its *magnitude* sits well below the framework's 0.22, which is expected and not a strike against the bridge: 0.22 is a 4-D SU(3) gluon-condensate number, not a universal constant a 2-D CP² model at arbitrary couplings should reproduce. What *does* carry over is the structural claim the bridge rests on — **coherent topology is the κ ≪ 1 minority that does the integrating**, exactly the role κ plays in `S = ΔC + κΔI`.

This is Movement 3 of the ordinal → functor → instanton bridge, and the point where the physical side of the distinction–integration gap acquires a measuring stick: the gauge/vacuum sector now has genuine, quantified topological content, switched on by the same criticality that troughs capacity and starves memory elsewhere in the program.

Reproduce with `python experiments/n3_instanton_content.py` (≈ 6 minutes; `--quick` for a smoke run).

**Honest scope:** this is 2-D CP² (the sector field), not 4-D SU(3); `κ_top` is the structural analogue of the framework's κ in a different theory and dimension, so only its *role* (a small coherent fraction), not its value, is meaningful here. The cooled χ_top is cooling-dependent (a standard lattice caveat — reported at a fixed cooling depth); one lattice size and coupling point; χ_top in lattice units. The instrument itself (integer, gauge-invariant, unit-charge on a constructed instanton) is exact and independently tested.

### The functor (logic → vacuum), measured

With the gap measured on the logic side (the [generative gap](#the-generative-gap-measured)) and topology measured on the physics side (the [instanton content](#the-instanton-content-of-the-sector-field)), the last movement asks whether the bridge between them is a real **functor** — a structure-preserving map `F : 𝒪 → 𝒬` carrying the logical hierarchy of reflective extensions onto the vacuum's topological structure — or just a suggestive analogy. A functor is testable, because structure preservation is. `experiments/n3_functor_bridge.py` instantiates it on the field.

The construction: start from a random ψ∈ℂ³ field — maximal distinction, no integration, the analogue of the perturbative vacuum with every winding sector populated. A **reflection step** is a gentle cooling sweep (moving each site toward the local action-minimizer): the integration dynamics that binds represented distinctions, the field's `F_{n+1} = F_n + Con(F_n)`. The integration observable `I = ⟨|z̄·z'|²⟩` climbs up the ladder; the functor's image at each rung is the topological content `T` (the instanton density).

- **Object correspondence — one ladder, two instruments.** As integration climbs `I = 0.33 → 0.94`, the topological image falls `T = 0.22 → 0.02`: a monotone map. It is **contravariant** (more integration, less topological gas) — the disordered instanton–anti-instanton gas is annihilated as the field binds.
- **Path-independence — the functor is well-defined on objects.** The decisive test: run the ladder at different reflection *rates* (fast and slow cooling schedules) and compare `T` at matched integration `I`. The curves **collapse onto one — a 1.9% relative scatter**. The topological content depends only on the *integration level reached*, not on the *history* of reaching it. That is exactly what makes `F` a functor on objects rather than a mere correlation: the physics side is a genuine, path-independent function of the logic side.
- **Naturality — one gradient drives both.** Every reflection step is action descent (∇S), and it moves *both* ladders at once: each step that raises integration lowers the topological gas, monotonically. The single S-gradient is the shared driver `∇S : D ⇒ κ·Int` — the natural transformation the bridge names.

The upshot for the program: the logic↔physics correspondence is **not a loose metaphor but a structure-preserving map** on the field — the vacuum's topology is a path-independent function of the integration level, driven by the same S-gradient that runs everything else. That is a real, measured basis for taking the bridge seriously — and the sharpest reason to attempt the genuine 4-D SU(3) build, where the covariant *condensate*-side functor (instantons that integrate rather than a gas that is annihilated) could be tested.

Reproduce with `python experiments/n3_functor_bridge.py` (≈ 10 seconds; deterministic cooling, no Monte Carlo).

**Honest scope:** the correspondence comes out **contravariant** because the thermal/cooling regime's topology is the disordered instanton *gas* — the opposite ordering from the coherent instanton *condensate* that integrates the QCD θ-vacuum (a stated limit; the covariant, condensate-side functor is exactly what the 4-D ensemble would test). The observables are scalar and the categorical claim is *illustrated* on the field, not proven; deterministic cooling from random starts, single lattice size. Path-independence is measured to ~2%, not exactly zero.

### 4-D SU(3) topological charge (Stage 1)

The functor result gave the reason to attempt the real thing: the framework's quantitative target — κ ≈ 0.22 as the instanton fraction of the QCD vacuum — lives in **4-D SU(3)**, the theory the 2-D CP² sector field only stands in for. This is Stage 1 of building the instrument there. The existing gauge Monte-Carlo is dimension-agnostic (numba, `ndim = links.shape[0]`), so a 4-D SU(3) Wilson ensemble runs directly; `project_genesis/gauge_topology.py` adds the missing piece — the **clover** field-theoretic topological charge `Q = (1/32π²) Σ_x ε_{μνρσ} Tr[F_{μν}F_{ρσ}]` with gauge cooling (`experiments/n3_su3_topology.py`).

- **The instrument is validated.** Pure-gauge configurations (a gauge transform of the identity) read **Q = 0 exactly**; the clover and single-plaquette field-strength definitions agree; and cooled configurations are **quantized and Z-renormalized** — single-instanton configs read |Q| ≈ **0.84** (Z ≈ 0.84, the standard coarse-lattice suppression of the field-theoretic charge, → 1 with gradient flow / finer lattices), with the trivial sector at exactly 0. The mean plaquette climbs with β_g (0.475 → 0.682), the standard Wilson-action behaviour.
- **Topological freezing, seen directly.** The vacuum tunnels freely across sectors at strong coupling (β_g = 1.8: cooled Q ranges −5 → +3, χ_top ≈ 0.0012) and **sticks in a single sector** toward weak coupling (β_g = 2.4: every config Q = 0) — the well-known critical slowing of topology in lattice gauge theory, reproduced by this instrument. The topological susceptibility χ_top is largest where the vacuum actually samples topology and thins toward weak coupling.

This is the pivot the whole bridge pointed to: the gauge/vacuum sector — the *physical* side of the distinction–integration gap — now has a genuine, validated topological-charge instrument, in the real 4-D SU(3) theory rather than its 2-D stand-in. It is Stage 1 (the instrument and a first susceptibility), not yet the number: the precise normalization and the condensate split that would test κ ≈ 0.22 are the next stage, but they are now *reachable* because the charge exists and is validated.

Reproduce with `python experiments/n3_su3_topology.py` (≈ 6 minutes; `--quick` for a smoke run).

**Honest scope:** Stage 1 on a small 8⁴ lattice in lattice units. The field-theoretic Q carries the multiplicative renormalization Z ≈ 0.84 (levels at ~Z·n, not exact integers, until gradient-flowed), and weak-coupling **topological freezing** limits the χ_top statistics (the Markov chain does not decorrelate the topological sector). Scale-setting (a physical lattice spacing), gradient flow (Z → 1), and the instanton/perturbative condensate split needed for the actual κ ≈ 0.22 comparison are explicitly deferred to the next stage — this PR establishes and validates the instrument they require.

### 4-D SU(3) gradient flow: the instanton fraction of the vacuum (Stage 2)

Stage 1 read the charge through crude *cooling* — an uncontrolled smoother with no clean scale, leaving Q multiplicatively renormalized (Z ≈ 0.84). Stage 2 replaces cooling with the **Wilson (gradient) flow**, the gradient descent of the Wilson action in a flow "time" `t`, which Lüscher showed is a genuine renormalization-group smoothing. `project_genesis/gauge_topology.py` gains a Lüscher third-order Runge–Kutta flow integrator, the clover action density `E(t)`, and the self-dual fraction; `experiments/n3_su3_gradient_flow.py` runs the ensemble.

- **The flow sets a scale.** The dimensionless clock `t² E(t)` climbs monotonically and crosses the reference `0.3` at a well-defined flow time `t₀` (t₀ = 0.39, 0.46, 0.95 at β_g = 1.7, 1.8, 1.9) — the standard Wilson-flow scale `√(8 t₀)`. Evaluating each ensemble at *its own* t₀ compares them at the same physical smoothing radius.
- **Z → 1.** Under the flow the topological charge sharpens off the Stage-1 renormalized levels (`|Q| ≈ 0.84·n`) toward genuine integers — the coarse-lattice suppression flowing away, the RG content of the flow made visible.
- **The instanton fraction of the vacuum.** The headline observable is the **self-dual fraction** `f_SD = Σ_x |q(x)| / Σ_x e(x) ∈ [0,1]`, the fraction of the field energy that saturates the Bogomolny bound `e(x) ≥ |q(x)|` — i.e. is carried by (anti-)self-dual, instanton-like structure rather than structureless UV field energy. This is the lattice proxy for the quantity the functorial-bridge paper attaches a number to: the **instanton fraction of the gluon condensate, κ ≈ 0.22**. Read at the RG-clean scale t₀, it **drifts through κ**: f_SD(t₀) = **0.187 → 0.221 → 0.352** at β_g = 1.7 → 1.8 → 1.9, landing essentially *on* κ = 0.22 at β_g = 1.8 (0.221 ± 0.004).

The number the whole bridge pointed at — κ ≈ 0.22 — appears, at the principled flow scale, as the self-dual fraction of the 4-D SU(3) vacuum. It brackets and crosses κ across a coupling window; it does not sit on it universally.

Reproduce with `python experiments/n3_su3_gradient_flow.py` (≈ 11 minutes; `--quick` for a smoke run).

**Honest scope:** a coarse 8⁴ lattice over a narrow window of strong couplings. `f_SD` is a *single-scale reading of a monotone-rising quantity* — it keeps climbing past t₀ toward 1 as the field is smoothed to a few classical lumps, so it depends on the coupling (t₀ in lattice units grows with β_g, so the field is smoothed further before the reading). That the instanton fraction is an O(0.2) number crossing κ in this window is the result; a coupling-*independent* determination of κ needs the continuum limit (where the reading stabilizes) and a scheme-matched operator-product-expansion condensate. This stage establishes the flow, the scale, and the observable. *(The continuum trend below tests that stabilization — and corrects the picture.)*

### 4-D SU(3) continuum trend: where the Stage-2 uncertainty lives

The Stage-2 caveat was a hypothesis with two names — was the β_g-drift of `f_SD(t₀)` a **finite-size** artifact (the physical box `L/√t₀` shrinks as β_g grows at fixed `L = 8`) or a genuine lattice-**cutoff** (finite spacing) dependence? — and a question: does the reading *stabilize* toward a coupling-independent number as `a → 0`? `experiments/n3_su3_continuum.py` answers all three, with `continuum_limit` added to `gauge_topology.py`.

- **It is volume-converged.** At fixed β_g, varying `L` barely moves `f_SD(t₀)`: **0.218 → 0.221 → 0.219** across `L = 6, 8, 10` at β_g = 1.8 (boxes 9 → 15), and **0.344 → 0.345** across `L = 8, 12` at β_g = 1.9 (boxes 8 → 12). Finite size is *not* the source of the Stage-2 drift.
- **The drift is a cutoff effect.** Scanning β_g at the volume-converged `L = 8`, `f_SD(t₀)` rises monotonically — **0.189, 0.200, 0.221, 0.264, 0.344** at β_g = 1.7, 1.75, 1.8, 1.85, 1.9 — as the lattice refines (t₀ grows, `a` shrinks). The uncertainty lives entirely in the lattice spacing.
- **The continuum limit overshoots κ.** A linear `O(a²)` extrapolation of `f_SD(t₀)` against the cutoff surrogate `1/t₀ ∝ a²` gives **f_SD → 0.435 as a → 0** — well *above* κ = 0.22. So the Stage-2 crossing of κ at β_g = 1.8 was a **coarse-lattice coincidence**, not a cutoff-stable determination. The self-dual fraction at the flow scale is a legitimate instanton-content observable, but it carries a real `O(a²)` cutoff dependence and is *not*, by itself, a scheme-free estimator of the framework's κ.

This is the honest correction the continuum push existed to make: it localizes the Stage-2 uncertainty precisely — **volume-converged, cutoff-dominated** — and shows the number `0.22` does not survive the naive `a → 0` limit of this observable. It sharpens, rather than confirms, the earlier reading.

Reproduce with `python experiments/n3_su3_continuum.py` (≈ 30 minutes; `--quick` for a smoke run).

**Honest scope:** a 3–5 point trend over a narrow, coarse strong-coupling window (the coarsest t₀ ≈ 0.39 is barely one lattice spacing, so its scale is the least trustworthy point), fit with a single linear `a²` ansatz. It does *not* deliver the continuum κ — that needs genuinely finer lattices (fighting topological freezing, which throttles the topology sampling exactly where the flow scale is cleanest) and the matched OPE condensate the self-dual fraction only stands in for. What it delivers is the correction: the Stage-2 agreement with 0.22 was cutoff-driven.

### Capacity as gravity: the attraction κ mediates

A different question about κ — not its *value* but its *character*. The capacity field obeys `∂_t κ = D∇²κ + r(κ₀−κ) − c·load·κ`, which is the gradient flow of the **capacity free energy** `F[κ] = ∫[(D/2)|∇κ|² + (r/2)(κ−κ₀)² + (c/2)·load·κ²]`. That one fact makes κ behave like gravity *in the general sense*: a concentration of `load` (distinction — the field's "mass") digs a well in κ, and because F is a genuine energy, two such masses lower it by drawing together. `project_genesis/capacity_gravity.py` + `experiments/n3_kappa_gravity.py` measure it, on the real κ dynamics.

- **κ mediates a screened attraction.** Two rigid Gaussian masses at separation r, relaxed to steady-state κ, lower the free energy — `V(r) < 0` throughout (−2.6 at r=4 deepening as they approach) — and the potential is **Yukawa-shaped**: a fit `V ∝ −e^{−r/ξ}/r` gives ξ ≈ 4.8. An attractive, short-ranged, mass-mediated force.
- **The range is √(D/r) — the recovery rate is the graviton mass.** Linearizing, the κ-well is screened with length `ξ_κ = √(D/r)`. Measured across **independently varied** D and r (so the test is √(D/r), not just 1/√r), the screening length tracks the prediction as **ξ_meas = 1.02·√(D/r), R² = 1.000** — including a three-way degeneracy where (D,r) = (0.5, 0.02), (1, 0.04), (2, 0.08) all give ξ ≈ 5.1. Fast recovery screens the force to short range; slow recovery lets it reach. The persistence↔plasticity dial from the memory work is secretly the massive↔massless-graviton dial.
- **It obeys an equivalence principle.** The interaction strength scales with the **product of the masses**, `V ∝ m₁·m₂` (linear in m² with R² = 0.97). κ-gravity couples to *how much* distinction is present, universally — not to what kind it is.

So κ is not just a coupling constant whose number we chase — it is, structurally, **the framework's gravity**: the weak, universal, mass-sourced binding field, sourced by structure and back-reacting on it, with a screened range set by the capacity recovery length.

Reproduce with `python experiments/n3_kappa_gravity.py` (≈ 4 minutes; `--quick` for a smoke run).

**Honest scope:** a classical, static, scalar mediation — κ is the binding *coupling*, not the metric tensor of general relativity, and the screened `√(D/r)` range is a genuine difference from Newtonian `1/r²` (κ-gravity is a *massive*-graviton analogue). Rigid load "masses" stand in for self-consistent structure; the measured range carries a small (~2%) lattice/finite-width offset absorbed into the constant α. What is established is the gravitational *role* — a universal attraction whose strength is κ and whose range is the recovery length — not an identification with GR.

### The spectrum of stable forms: mass is stable structure made manifest

Where the last section asked what κ *does* to masses, this asks what a *mass* is. The framework's answer: a particle is a **stable structure made manifest** — a localized configuration the dynamics cannot smoothly undo — and in the CP² sector field that is a **topological soliton** of integer charge `Q ∈ π₂(CP²) = ℤ`. `project_genesis/stable_forms.py` + `experiments/n3_stable_forms.py` build that corpus and weigh each form two ways.

- **A discrete corpus.** The admissible forms carry integer charge (Q = +1, +2, +3, +4) and their structural (inertial) masses — CP-action energies — lie on a **discrete ladder** `E = 6.84, 12.47, 18.68, 24.90`, rising by a near-constant 6.0 per unit charge: the Bogomolny spectrum `E ∝ |Q|`. Matter comes in quanta because topology does.
- **They are stable.** Under cooling a charge-Q form **holds its charge** (Q = 3 stays +3.0 → +3.0) and settles to an **energy floor** it cannot decay below (18.7 → 18.7), while a topologically trivial bump relaxes toward the vacuum (1.55 → 0.93). Only the protected forms persist — they are the *stable* forms.
- **Structural mass = gravitational mass.** Feeding each form's distinction density `d(x) = Σ_μ(1 − |ψ̄·ψ'|²)` into the capacity-gravity dynamics, the κ-well it sources — its **gravitational mass** — is **proportional to its structural mass** across the whole spectrum: `M_grav = 3.67·E, R² = 1.000`. The equivalence principle, *emergent*: inertial and gravitational mass coincide because both are the same underlying quantity — the form's distinction content. The framework **explains** `m_inertial = m_gravitational` rather than imposing it.
- **A measured nonlinearity.** At strong capacity coupling the κ-well **saturates** (bottoms out at 0), and gravitational mass grows *sub-linearly* — the strong/weak ratio falls 8.40 → 7.22 from lightest to heaviest form. So `M_grav` is a genuine field response, not a relabelling of the energy: the equivalence holds in the weak-well regime and bends where capacity runs out.

This closes a loop the whole program has been circling: the generative gap drives structure → where capacity permits it crystallizes into **stable, discrete forms** (matter) → those forms *are* concentrations of distinction (**mass**) → and that same distinction sources **κ-gravity**, with gravitational mass equal to inertial mass. Stable structure, made manifest, that gravitates in proportion to itself.

Reproduce with `python experiments/n3_stable_forms.py` (≈ 3 minutes; `--quick` for a smoke run).

**Honest scope:** 2-D CP² solitons on a periodic lattice, masses in lattice-action units; the constructed forms are lightly cooled to a common smoothness (the raw `w^Q` winding profile is Q-dependent), and the equivalence is measured in the weak-well regime. The structural=gravitational proportionality is *explained* by the shared distinction root, not an independent coincidence — but that shared root is precisely the framework's account of why the two masses are equal. This is the structural claim (matter as stable topological forms with `m_i = m_g`), not a derivation of the Standard-Model spectrum.

### Self-gravitating forms: infall and accretion

The masses of the last two sections were pinned. Unpin them — let each move in the κ-field they mutually source — and the framework grows its own bound structure. `project_genesis/capacity_dynamics.py` + `experiments/n3_self_gravity.py` run the adiabatic (Born–Oppenheimer) dynamics: κ relaxes to steady state for the instantaneous mass positions, and each mass drifts down the gradient of the capacity free energy. By the envelope theorem (at the relaxed κ, `δF/δκ = 0`) the force is the direct coupling gradient, `F_i = −c·Σ_x load_i·κ·∇κ` — each mass feeling the κ-gradient of the well the ensemble digs — integrated overdamped, `dR_i/dt = μ·F_i`.

- **Infall.** Two masses fall together and merge (separation 14 → contact in ~19 steps), and the fall **accelerates** — the closing rate grows from 0.49 to 0.68 per step as the screened force steepens near contact. Exactly as gravity should.
- **It is gravity, not drift.** With the capacity coupling switched off (`c = 0`) the same pair sources no κ-well and **stays put** (separation 14 → 14). The infall is the field, not a numerical bias.
- **Accretion.** Nine masses scattered at random, left to their own κ-gravity, **clump** — pairs merge, clumps merge, the count of bound objects falls monotonically to a **single bound object** (mass 9) while the total mass is exactly conserved. Structure forming out of the capacity field, from first principles.

This is the dynamical capstone of the gravity thread: the generative gap builds distinction → capacity crystallizes it into stable, discrete **forms** (matter, with `m_i = m_g`) → and those forms, moving in the κ-field they source, **fall together and accrete into bound structure**. Gap → matter → gravity → structure, each step measured.

Reproduce with `python experiments/n3_self_gravity.py` (≈ 3 minutes; `--quick` for a smoke run).

**Honest scope:** overdamped adiabatic dynamics — κ relaxed each step, rigid Gaussian masses (standing in for the stable forms, whose `m_g = m_i` was established separately) moved down the energy gradient — a *screened* force on a periodic lattice. It demonstrates the emergence of bound structure under κ-gravity; it is not a cosmological N-body simulation. *(The inertial version below adds momentum — orbits and virialization.)*

### Inertial κ-gravity: orbits, conserved energy, precession, virialization

The self-gravity above was overdamped — masses *fall*. Give them **inertia** (`M·d²R/dt² = F`, same envelope κ-force, symplectic velocity-Verlet) and the fall becomes a full gravitational dynamics. `project_genesis/capacity_dynamics.py` (`evolve_inertial`) + `experiments/n3_orbital_gravity.py`.

- **A Kepler-like family.** Tuning the initial tangential speed of two equal masses sweeps the whole family a central force should produce: **radial plunge** (v₀=0, straight through the centre), **bound elliptical orbit** (v₀=0.5, separation swings 5.8–12), **near-circular orbit** (v₀=0.8, separation nearly constant), and **unbound escape** (v₀=1.2).
- **Energy is conserved.** The symplectic integrator holds the total `T + F[κ]` to **0.12%** over the orbit while kinetic and potential energy trade back and forth — real conservative dynamics, not numerical drift.
- **Orbits precess.** Because the κ mediator is **screened** (Yukawa, not 1/r), the bound ellipse *does not close* — its perihelion advances **+141° per orbit**, tracing a rosette. Orbital precession is the direct dynamical signature of the massive-graviton range `√(D/r)` (and is the finite-range analogue of — not the same as — the relativistic perihelion advance of GR).
- **An N-body cloud virializes.** With a little dissipation, a cloud of 8 masses settles into a bound cluster obeying the virial relation `2⟨T⟩ + ⟨W⟩ → 0` (measured `2⟨T⟩/|⟨W⟩| → 1.19`), the damped oscillations of the virial ratio ringing down toward equilibrium.

So κ-gravity is a *complete* gravitational dynamics, not just infall: bound orbits, escape, conserved energy, precession, and virial equilibrium all emerge from the one screened force whose strength is the forms' mass.

Reproduce with `python experiments/n3_orbital_gravity.py` (≈ 4 minutes; `--quick` for a smoke run).

**Honest scope:** point masses (rigid load blobs standing in for the stable forms) under the adiabatic κ-force on a 2-D periodic lattice — a Newtonian-in-spirit dynamics of a *screened* force, not a relativistic or cosmological calculation. The precession is the Yukawa (finite-range) effect, distinct from GR's relativistic advance; the virial ratio settles to O(1) with the residual above 1 reflecting the finite dissipation/run length. *(The cosmological version below adds an expanding background.)*

### Cosmic structure: κ-gravity against an expanding background

The last step toward cosmology: put the inertial κ-gravity in an **expanding background** — every mass given the Hubble-law recession `v = H·(r − r_centre)` — and ask the question structure formation turns on: does gravity still assemble bound structure against the expansion? `project_genesis/capacity_dynamics.py` (`hubble_flow`, `fof_groups`) + `experiments/n3_cosmic_structure.py`.

- **Turnaround.** Two receding masses decelerate under gravity, reach a **maximum separation** (the turnaround radius), and **recollapse** into a bound pair — for expansion below a critical rate. The turnaround radius grows with `H` (**10.0 → 12.5 → 15.5 → 22.9** at H = 0, 0.10, 0.14, 0.18) and then gives way to **escape** (H = 0.26, the pair runs to the box edge). This is the spherical-collapse picture: a critical expansion rate separates recollapse from escape.
- **Structure vs expansion.** A cloud of 12 masses in Hubble flow **collapses into a single bound halo at low expansion** (100% of the mass in one friends-of-friends group at H = 0) but is progressively **dispersed to fragments as expansion rises** — largest bound fraction **100% → 75% → 33% → 25%** across H = 0, 0.08, 0.16, 0.26. Expansion suppresses structure, exactly as a faster-expanding universe forms less.

So κ-gravity builds structure against expansion up to a threshold and loses beyond it — the same **gap → matter → gravity** chain, now competing with a cosmological background. The essential ingredient of structure formation is present: a critical rate above which the universe expands faster than it can assemble.

Reproduce with `python experiments/n3_cosmic_structure.py` (≈ 6 minutes; `--quick` for a smoke run).

**Honest scope:** a **Newtonian, coasting-background** model — the expansion enters as the initial Hubble peculiar-velocity field and gravity is the screened κ-force; there is **no FLRW metric, no dark-energy term, no relativistic horizon**, and the "background" is carried by initial velocities rather than an evolving scale factor with its own Friedmann equation. It captures the essential competition (self-gravity vs expansion, turnaround, suppression of structure by expansion), not a quantitative cosmology. *(The FLRW version below adds the evolving scale factor and dark energy.)*

### An expanding universe: FLRW scale factor, Hubble drag, and dark energy

The coasting model carried the expansion in the initial velocities only. Here the background is a genuine, **evolving scale factor** `a(t)` obeying a Friedmann-like law with a matter component and a cosmological-constant / dark-energy component, `(ȧ/a)² = H₀²[Ω_m a^{−p} + Ω_Λ]`. Masses feel the peculiar κ-force plus the background `(ä/a)(r − c)` and start in the Hubble flow. `project_genesis/capacity_dynamics.py` (`friedmann_rates`, `evolve_cosmological`) + `experiments/n3_expanding_universe.py`.

- **Expansion histories.** `a(t)` **decelerates** when matter dominates and **accelerates** when Λ dominates — over the run `a` grows to 3.0, 5.6, 7.8, 10.4, 15.1 as Ω_Λ = 0 → 1, the qualitatively different fates of a matter vs a dark-energy universe.
- **Hubble drag.** A peculiar velocity **redshifts as 1/a** — the product `a·|v_pec|` holds constant to **0.2%** across the expansion. This momentum-redshift is the genuinely new FLRW effect the coasting model could not show.
- **Dark energy suppresses structure.** The *same* initial cloud collapses into a single bound halo in a matter universe (**100%** of the mass in one group at Ω_Λ = 0) but is progressively **frozen out** as the expansion accelerates — largest bound fraction **100% → 92% → 92% → 92% → 42%** across Ω_Λ = 0, 0.3, 0.5, 0.7, 1.0, with a sharp freeze-out at pure de Sitter. A faster-expanding, dark-energy universe assembles less structure — the defining signature of Λ in growth.

So the **gap → matter → gravity → structure** chain now runs in a genuine FLRW-like background: an evolving scale factor, momentum redshift, and the dark-energy freeze-out of structure growth. The mechanisms of cosmology are present.

Reproduce with `python experiments/n3_expanding_universe.py` (≈ 10 minutes; `--quick` for a smoke run).

**Honest scope:** a **Newtonian FLRW-analogue** — a real evolving scale factor and genuine Hubble drag, but the Friedmann law is **imposed** (`p = 3` matter dilution applied to the 2-D screened dynamics), not derived from the κ-field's own stress-energy, and there is no metric, horizon, or relativistic growth factor. It reproduces the *mechanisms* — Hubble drag and the dark-energy suppression of growth — not a quantitative ΛCDM. The κ screening length is a fixed physical scale (it does not itself redshift with the background). *(The next section closes the loop — deriving the dark energy from the κ-field itself.)*

### A self-contained cosmos: the expansion driven by the κ-field itself

The FLRW section still *dialled* Ω_Λ by hand. This closes the loop: the capacity field supplies the dark energy for free. Its recovery term `r·(κ₀ − κ)` continually heals the field back to baseline — an energy spent **maintaining itself** that does *not* dilute as space expands, exactly a cosmological constant. So `ρ_Λ = coeff·r·κ₀²` is a **property of the field, not an input**, while matter dilutes as `ρ_m(a) = ρ_m0·a^{−dim}`; the Friedmann equation `H² = ρ_m + ρ_Λ` makes the whole expansion history a prediction. `project_genesis/capacity_dynamics.py` (`capacity_vacuum_density`, `deceleration_parameter`, `acceleration_onset`, `integrate_scale_factor`) + `experiments/n3_self_contained_cosmos.py`.

- **An emergent cosmic history.** `a(t)` starts matter-dominated and **decelerating**, then — as matter dilutes below the capacity vacuum — turns over into **Λ-dominated acceleration**. The deceleration parameter `q` crosses zero and runs to the de Sitter limit `q → −1`: the same decel→accel history our universe has, out of one field.
- **Dark energy is the capacity field maintaining itself.** The acceleration onset `a_acc = (ρ_m0/2ρ_Λ)^{1/dim}` is **derived**, not dialled — and turning up the recovery rate `r` (more self-maintenance) raises ρ_Λ and brings the acceleration earlier, exactly on the predicted curve: `a_acc = 1.71, 1.36, 1.19, 1.00` at `r = 0.01, 0.02, 0.03, 0.05` (predicted = measured to the interpolation precision).
- **The energy budget hands over.** Matter dilutes as `a^{−3}` while the capacity vacuum stays constant, so the universe passes matter → dark-energy dominated (equality at `a_eq`, acceleration following at `a_acc`) — the standard cosmic energy-budget handoff, here a consequence of the field.

**This closes the cosmological loop.** The same capacity field `κ` is now *gravity* (its free energy), *matter* (stable forms of distinction), **and** *dark energy* (its self-maintenance). A cosmos — matter, gravity, expansion, and its late-time acceleration — out of one field.

Reproduce with `python experiments/n3_self_contained_cosmos.py` (≈ 20 seconds; background integration only). `--quick` for a smaller scan.

**Honest scope:** a Newtonian `8πG/3 = 1` Friedmann closure. The dark-energy density is genuinely *derived* from the recovery term (not dialled), but the identification `ρ_Λ = coeff·r·κ₀²` carries a modelling coefficient, the matter dilution law `a^{−dim}` is imposed, and there is still no metric, horizon, or relativistic stress-energy tensor. It establishes the *mechanism* — capacity self-maintenance as an emergent cosmological constant — not a first-principles ΛCDM. *(The next section removes the last dialled input — the matter density — deriving it from the form spectrum.)*

### Matter from forms: the Friedmann source read off the topological spectrum

The self-contained cosmos still *dialled* the matter density `ρ_m0` and *imposed* its `a^{−dim}` dilution. This traces the matter source all the way back to **Link 2** — matter as the stable topological forms of the sector field ψ∈ℂ³ — so the whole Friedmann source becomes the field's own content. `project_genesis/capacity_dynamics.py` (`matter_energy_density`) + `experiments/n3_matter_from_forms.py`.

- **The Bogomolny mass ladder.** The charge `Q = 1..4` forms have structural (rest) energies on a straight line `E ≈ 5.38·|Q| + 2.87` (R² = 0.986) — a Bogomolny energy floor per unit charge. So the matter density `ρ_m0 = ΣE/V ∝ Σ|Q|` is **read off the topological content** of the field, not chosen.
- **Topological protection.** Deforming a charge-2 form and cooling recovers **exactly Q = 2** for every realisation up to noise ≈ 0.3, while the raw geometric charge has already blown up on UV dislocations (⟨|Q|⟩ → 10). Only past a threshold (noise ≈ 0.4) does a kick comparable to the field itself carry it into a neighbouring sector — the rest-energy floor is genuinely protected.
- **The dilution law is topological.** Because that total rest energy is conserved, spreading it through a growing comoving volume `V₀·a^{dim}` gives `ρ_m(a)` with a log–log slope of exactly **−dim** — the `a^{−dim}` law derived from charge conservation + volume, not imposed.
- **The cosmos from its form content.** Feeding `ρ_m0` (from the forms) and `ρ_Λ` (from the recovery term) into the Friedmann integrator makes the acceleration onset `a_acc = (ρ_m0/2ρ_Λ)^{1/dim}` a function of **how many stable forms the universe holds** — `a_acc = 1.14 → 1.95` as `Σ|Q| = 10 → 50`, on the predicted curve. More matter, later acceleration.

**The last dialled density is gone.** The matter term of the cosmology is now fixed by the topological content of the field — `ρ_m0` from the Bogomolny spectrum, `a^{−dim}` from charge conservation — and combined with dark energy from κ's self-maintenance, the entire Friedmann source is the field's own content. Matter is stable form (Link 2), and the cosmos runs on it.

Reproduce with `python experiments/n3_matter_from_forms.py` (≈ 40 seconds; the cooling sweeps of the protection scan dominate). `--quick` for a smaller scan.

**Honest scope:** a Newtonian `8πG/3 = 1` closure. The forms are 2-D CP² solitons whose Bogomolny energies set the rest masses; they are then treated as point masses populating a `dim`-dimensional comoving volume, which is what gives the `a^{−dim}` dilution. The dark-energy identification still carries a modelling coefficient and there is no metric or relativistic stress-energy tensor. It fixes the matter *source* from topology — not a first-principles ΛCDM. *(The next section measures the **equation of state** of that source — the character a relativistic stress-energy tensor will need.)*

### The equation of state of the matter: what kind of stuff the forms are

The cosmology's Friedmann source is now field-sourced, but with an *assumed* character: matter as pressureless **dust** (`ρ_m ∝ a^{−dim}`) and the capacity vacuum as a **cosmological constant** (`ρ_Λ = const`). Those characters are equations of state `p = w·ρ`. This measures `w` for the forms — the last qualitative input, and the piece a relativistic stress-energy tensor `T_{μν}` will need. `project_genesis/capacity_dynamics.py` (`equation_of_state_from_dilution`, `gas_equation_of_state`) + `experiments/n3_form_equation_of_state.py`. Three independent readings agree:

- **Kinetic (the measurement).** A gas of forms with velocity dispersion `σ_v` has a directly computable `w = p/ρ = Σγm v²/dim ÷ Σγm`. At rest it is pressureless dust (`w → 0`); heated toward `c` it becomes radiation (`w → 1/dim`). The measured `w(σ_v)` rises smoothly from `0.002` (cold) toward `0.315` (`σ_v = 0.9`), heading for `1/3` — the **cold form gas the cosmology assumes really is dust**.
- **Mechanical (`p = −∂E/∂V`).** A form is a *localized* lump: its rest energy is independent of the comoving box (`∂E/∂V ≈ −7×10⁻⁵ → 0`), so its pressure vanishes — `w ≈ 0`, dust, with no reference to velocities. The capacity vacuum instead has `E_Λ = ρ_Λ·V`, so `∂E/∂V = ρ_Λ` and `p = −ρ_Λ` — `w = −1`, a cosmological constant.
- **Kinematic (from the dilution exponent).** Since `ρ ∝ a^{−dim(1+w)}`, the exponent *is* the equation of state: the `a^{−dim}` matter law (Link 8) reads back `w = 0`, the constant vacuum `w = −1`, and radiation (`a^{−(dim+1)}`) `w = 1/dim`.

The two components a covariant `T_{μν}` must carry are now fixed and mutually consistent: a pressureless dust of forms (`w = 0`) and a `w = −1` capacity vacuum, with the warm form gas bridging dust and radiation. The cosmology's "dust + Λ" assumption is **measured, not assumed**.

Reproduce with `python experiments/n3_form_equation_of_state.py` (≈ 10 seconds). `--quick` for a smaller scan.

**Honest scope:** still a Newtonian `8πG/3 = 1` background. This measures the *equation of state* the eventual `T_{μν}` needs — it does not yet build the tensor or a covariant field equation. The forms are point masses of the measured Bogomolny rest energies; `w` is intensive (box-independent), and the mechanical `∂E/∂V → 0` is read off the localized CP² lump. *(The next section assembles those pieces into a stress-energy tensor and makes the expansion its **consequence**.)*

### The relativistic closure: a stress-energy tensor, and expansion as its output

Everything the cosmology needs is now measured from the field — the matter density and its dilution (Links 7–8) and the equation of state of each component (Link 9). This assembles them into a **perfect-fluid stress-energy tensor** in a 3+1 FLRW background and lets the expansion *follow* from it, rather than imposing the Friedmann matter law by hand. `project_genesis/capacity_dynamics.py` (`stress_energy_tensor`, `covariant_conservation_rate`, `friedmann_acceleration`, `integrate_stress_energy`) + `experiments/n3_stress_energy_closure.py`.

`T^μ_ν = diag(−ρ, p, p, p)`, `p_i = w_i·ρ_i`. Its covariant conservation `∇_μ T^{μν} = 0` has one non-trivial FLRW component — the continuity equation `ρ̇_i + 3H(ρ_i + p_i) = 0` — and closing with `H² = ρ`, `ä/a = −½(ρ + 3p)` makes `a(t)` a consequence of the field's own stress-energy:

- **The tensor's equation of state evolves.** Built from the dust of forms (`w = 0`) and the capacity vacuum (`w = −1`), the effective `w_eff = p/ρ` runs from `−0.17` (matter-dominated) to `−1` (vacuum-dominated), crossing the acceleration threshold `w_eff = −1/3`.
- **Conservation *derives* the dilution laws.** Integrating `ρ̇ = −3H(ρ + p)` reproduces `ρ ∝ a^{−3(1+w)}` for each component to `~10⁻⁷` — dust `a^{−3}`, the vacuum constant, radiation `a^{−4}`. The `a^{−dim}` matter law we had **imposed** is now an *output* of covariant conservation.
- **Expansion as an output.** The deceleration parameter `q = ½(1 + 3w_eff)` crosses zero at `a_acc = 1.36`, so `a(t)` decelerates then accelerates — nothing about the dilution assumed, only the measured equations of state and conservation.
- **Consistency.** The `a(t)` from the coupled tensor coincides with the earlier imposed-`a^{−3}` cosmology (`integrate_scale_factor`) to `max rel |Δa|/a ≈ 5×10⁻³` — the new derivation *explains* the old input.

**This closes the Friedmann level.** A stress-energy tensor built from the field's own content (dust of forms + capacity vacuum), with measured equations of state, made to conserve covariantly, produces the whole expansion history — matter dilution, the decel→accel turnover, the de Sitter limit — as a consequence. The Friedmann equation is now an *output*.

Reproduce with `python experiments/n3_stress_energy_closure.py` (≈ 5 seconds; background integration only). `--quick` for a smaller scan.

**Honest scope:** a perfect-fluid `T^μ_ν = diag(−ρ, p, p, p)` in a homogeneous FLRW background with the standard Einstein/Friedmann relations (units `8πG/3 = 1`). It makes the *expansion law* a consequence of the field's stress-energy and its measured equations of state — it is **not** a derivation of the Einstein field equations from the κ-action, and it stays homogeneous (no perturbations, metric solved, or covariant field theory of κ itself). It closes the Friedmann level: `T_{μν}` from the field, expansion as its consequence. *(The next section removes the last hand-input — the Friedmann relation `H²=ρ` itself — deriving it from a variational principle.)*

### The variational closure: the Friedmann equation from an action, not by hand

The relativistic closure still *put in by hand* the Einstein/Friedmann relations `H² = ρ`, `ä/a = −½(ρ + 3p)` that turn `T^μ_ν` into `a(t)`. Those relations are the content of a **minisuperspace variational principle**. For flat FLRW with lapse `N` and scale factor `a` (units `8πG/3 = 1`), the action `S = ∫dt[−a ȧ²/N − N a³ρ(a)]` — the gravitational kinetic term `−a ȧ²` (the dynamical geometry's own free energy) plus the stress-energy content `ρ(a)` — has: the lapse/Hamiltonian constraint `∂S/∂N = 0 ⇒ H² = ρ` (Friedmann), and the `a`-Euler–Lagrange equation (with conservation) `⇒ ä/a = −½(ρ + 3p)`. `project_genesis/capacity_dynamics.py` (`minisuperspace_lagrangian`, `hamiltonian_constraint`, `integrate_friedmann_action`) + `experiments/n3_friedmann_from_action.py`.

- **One history, three derivations.** Evolving the `a`-Euler–Lagrange equation with conservation reproduces the same `a(t)` as the stress-energy route (`~10⁻⁸`) and the original imposed-`a^{−3}` cosmology (`~5×10⁻³`).
- **Friedmann is a first integral, not an input.** Along that acceleration equation the constraint `C = H² − ρ` stays below `~10⁻¹⁰` — `H² = ρ` is *preserved* by the dynamics though it is never substituted. Analytically `Ċ = −2HC`, so `C = 0` is conserved.
- **…and an attractor.** Start from a *wrong* expansion rate (constraint violated by ±30–60%) and `C(t) → 0`: the universe relaxes onto the Friedmann trajectory, because `Ċ = −2HC` with `H > 0`. Even the initial condition need not satisfy Friedmann; the dynamics enforces it.
- **The physics is intact.** The deceleration parameter crosses zero at the same `a_acc = 1.36`; the decel→accel history survives the derivation.

**The last hand-input at the Friedmann level is removed.** `H² = ρ` and `ä/a = −½(ρ + 3p)` are the Hamiltonian constraint and Euler–Lagrange equation of one action, and `H² = ρ` is a preserved, attracting first integral. The geometry (the scale factor's dynamics) and its source (the field's stress-energy) come from a single variational principle.

Reproduce with `python experiments/n3_friedmann_from_action.py` (≈ 5 seconds; background integration only). `--quick` for a smaller scan.

**Honest scope:** a *minisuperspace* variational principle — homogeneous flat FLRW with the single degree of freedom `a(t)`. The Friedmann equation is genuinely the Hamiltonian constraint and `H² = ρ` a preserved, attracting first integral, but the gravitational kinetic term `−a ȧ²` is the reduced Einstein–Hilbert form **posited** (read as the geometry's free energy), not derived from the κ-action microscopically, and there are no inhomogeneous field equations or a solved metric. It removes the last hand-input at the Friedmann level; it is not a derivation of general relativity. *(The next section shows that posited term **is** the capacity field's own kinetic free energy.)*

### Gravity from the capacity field: the gravitational action as capacity free energy

The variational closure still *posited* the gravitational kinetic term `−a ȧ²` (the reduced Einstein–Hilbert form). It **is** the capacity field's own kinetic free energy. Identify the scale factor with the exponential of the homogeneous capacity scalar — the zero-mode `κ_s` whose global value sets the overall integration scale — `a = e^{κ_s}` (so `κ_s = ln a`, `H = κ̇_s`). Its kinetic free energy on the FLRW volume measure `a³` is `a³ κ̇_s² = a ȧ²`, so `−a ȧ² = −a³ κ̇_s²`. `project_genesis/capacity_dynamics.py` (`scale_capacity`, `capacity_kinetic_energy`, `capacity_scalar_acceleration`, `integrate_capacity_scale`) + `experiments/n3_gravity_from_capacity.py`.

- **The gravitational term is capacity free energy.** Along the cosmic history the posited `−a ȧ²` and the capacity scalar's kinetic free energy `−a³ κ̇_s²` coincide identically (`~10⁻¹⁶`) — the geometry's kinetic energy *is* the field's.
- **Friedmann is an energy balance `κ̇_s² = ρ`.** The lapse constraint of the capacity action says the mean capacity's kinetic free-energy density equals the content; along the trajectory `|κ̇_s² − ρ|` stays below `~10⁻⁹`. Expansion is the capacity field rolling; Friedmann balances its kinetic free energy against matter + vacuum.
- **Expansion is the capacity scalar rolling.** Its field equation `κ̈_s = −(3/2)(κ̇_s² + p)` drives the roll — `κ̈_s → 0` in the vacuum limit (constant roll → de Sitter), `κ̈_s < 0` under dust. The balance is never imposed yet holds, and a wrong initial roll-rate relaxes onto it (`Ċ = −3 κ̇_s C`).
- **It reproduces the whole history.** The scale factor `a = e^{κ_s}` from the capacity-scalar dynamics coincides with the stress-energy (`~10⁻⁸`) and imposed-law (`~5×10⁻³`) cosmologies — gravity's expansion re-derived as the capacity field rolling.

**The last posit is gone.** The gravitational action's kinetic term is not an independent Einstein–Hilbert input but the capacity field's own kinetic free energy, and the cosmic expansion is that field — the mean capacity — rolling under it. Gravity, and not just its cosmology, is read off the URP field.

Reproduce with `python experiments/n3_gravity_from_capacity.py` (≈ 5 seconds; background integration only). `--quick` for a smaller scan.

**Honest scope:** the identification `a = e^{κ_s}` (the scale factor as the exp of the homogeneous capacity scalar / e-folding number) is a *reading* within the URP framework, and the kinetic free energy carries gravity's wrong-sign conformal mode, taken as given. It is still minisuperspace (homogeneous, one degree of freedom), with no inhomogeneous field equations or a solved metric. It traces the last hand-input at the Friedmann level to the field; it is **not** a derivation of general relativity.

### The one-κ frontier: one operator across the two acts, honestly mapped

Act I measured `κ` as the self-dual fraction of the SU(3) vacuum (`f_SD(t₀) ≈ 0.22`, the instanton fraction of the gluon condensate); Act II uses `κ` as the capacity field whose free energy is gravity. Whether these are the *same dimensionless number* is the question that would fuse the two acts. This experiment reports, honestly, what the computation says — a **frontier, not a coincidence**. `project_genesis/topological_charge.py` (`coherent_fraction`, `cp_action_density`, `cp_coherent_fraction`, `cp_metropolis_sweep`) + `experiments/n3_one_kappa_frontier.py`.

- **One operator, two sectors.** The Bogomolny coherent fraction `κ̂ = Σ|q(x)|/Σe(x) ∈ [0,1]` is a single function — literally what `self_dual_fraction` computes for SU(3) and `cp_coherent_fraction` for the CP sector (verified equal), with the bound `Σe ≥ Σ|q|` holding in both. The operator transfers cleanly.
- **Act I: it rises to 0.22.** Under the Wilson flow the SU(3) coherent fraction rises from the UV-noise floor to `κ̂(t₀) = 0.220 ± 0.005` at the RG-clean scale `t₀` (`t²E = 0.3`) — the established value, reproduced.
- **Act II substrate: it falls.** In a *thermal* CP² vacuum (a Metropolis ensemble), the same `κ̂` *falls* monotonically under the cooling flow (`0.15 → 0.01`) — the opposite behaviour, no plateau near 0.22.
- **The frontier.** At the deepest smoothing both flows share, `κ̂ ≈ 0.18` (gauge) vs `0.03` (CP), and they move apart. **The naive coherent fraction does not give a parameter-free `κ_I = κ_II`.** The two κ's are the same *concept* (the integration fraction) but not the same measured number in this estimator.

**What an identity would require** (stated plainly, not forced): a matched renormalisation-group condition relating the 4-D SU(3) and 2-D CP flows (their dimensionful flow clocks `t²E` vs `t·E` are not automatically comparable), or a different invariant — a ratio of topological susceptibilities `χ_top`, or the instanton *size* distribution rather than the action fraction. Those are concrete next tests.

Reproduce with `python experiments/n3_one_kappa_frontier.py` (≈ 1 minute; the SU(3) ensemble dominates). `--quick` for a smaller scan.

**Honest scope:** a deliberately honest **boundary** result. The operator is genuinely one function and Act I's 0.22 is solid; what is *not* established is a matched numerical identity `κ_I = κ_II`. The value is the clean map — what holds, what does not, and what a real bridge would need — kept in the record so the theory is built on what is true, not what is hoped.

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
