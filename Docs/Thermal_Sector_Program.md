# The Thermal Sector Program — from a hanging test suite to Potts universality

*A synthesis of the Monte-Carlo program that grew out of repairing the gauge
sampler: what was asked, what was measured, and what the verdicts add up to.
Every claim below is reproducible from a script in `experiments/` and guarded
by tests in `tests/`; the figures live in the artifacts each script writes.*

## The instrument chain

The program rests on a validated sampling stack, built in this order:

1. **An exact SU(N) heat-bath** (`project_genesis/gauge_mc.py`) — Kennedy–
   Pendleton at strong effective coupling, Creutz at weak, Cabibbo–Marinari
   subgroups for any N ≥ 3, exact microcanonical overrelaxation, and an
   optional quenched matter source that enters each link's weight *exactly*
   (as a staple addition, through the same quaternionic projection the
   sampler already needs). Validated against exact single-plaquette and
   single-link Bessel results, strong-coupling expansions, and an
   independent Metropolis chain.
2. **Numba JIT kernels** (`gauge_mc_kernels.py`) — the same updates, same
   random-draw order (Numba `Generator` streams are bit-identical to
   NumPy's), ~50–100× the pure-Python reference. This is what makes every
   experiment below affordable.
3. **The joint (ψ, U) ensemble** (`annealed_matter.py`) — sector matter and
   gauge links co-evolving in one Gibbs measure, with a corner potential
   for sector formation, optional fraction pinning, and a noise-robust
   junction density for thermal fields.

## The measurements, in the order the questions arose

### 1. Confinement (`confinement_sigma_scan.py`)

The Wilson ensemble confines and the instrument can prove it deserves
trust: 2-D SU(2) is exactly solvable, and the measured Creutz ratios track
`σ_exact(β_g) = −ln[I₂(2β_g)/I₁(2β_g)]` at every scanned coupling (32²:
worst pull 2.7σ over seven couplings, with the χ(3,3)/χ(4,4) plateau
resolved). On that calibrated footing, 3-D SU(3) shows σ > 0 at >10σ
everywhere in β_g ∈ [1, 5], finite-size stable between 8³ and 16³, with
larger loops approaching the asymptotic tension from above — the standard
lattice picture, in miniature.

### 2. Quenched selection (`n3_thermal_selection.py`)

Coupling the *fixed* converged P-sector networks to fluctuating SU(P)
ensembles measured two things:

- **A group-rank tax on integration.** The coherence retention
  R(g_m) falls with palette size at every coupling — bigger gauge groups
  wander in more directions, so bigger palettes keep less of their
  integration.
- **A washout threshold.** Selection at P = 3 survives exactly where
  `κ·w·neutrality·R` exceeds the ΔC gap to P = 4; the boundary tracks a
  level set of `w·R`.

And one sharp negative: **quenched sector matter never frustrates the
gauge field.** The per-link constraint is rank-1 and integrable — a
zero-curvature connection satisfies every link simultaneously — so
curvature does not localize on walls or junctions in equilibrium. The
deterministic "gluons on walls" enrichment is a statement about transient
relaxation, not about the thermal state.

### 3. Annealed melting (`n3_annealed_matter.py`, 2-D and 3-D)

With matter and gauge co-fluctuating, the colour-neutral junction network
exists in equilibrium **only for P = 3** — in 2-D and, an order of
magnitude more densely (junction *lines*), in 3-D — and melts at
T ≈ 0.2 with slight thermal roughening first. The annealed P = 4 state
carries exactly zero full-palette density at every temperature, in both
dimensions: equilibrium is a cleaner selector than relaxation snapshots.

### 4. The melting boundary and its character
(`n3_phase_boundary.py`, `n3_scaling_ladder.py`)

T_melt(g_m) rises monotonically — the matter–gauge coupling *stabilises*
the junction network. The melt itself is a **crossover**: the four-size
scaling ladder gives χ_max(L) ∝ L^b with b = −0.09 ± 0.30, consistent
with flat at 0.3σ and 6σ away from transition-like scaling.

### 5. Why a crossover — and the dynamical-basis resolution
(`n3_potts_transition.py`)

The crossover is not a disappointment; it is a clue. The fraction pinning
that stabilises the junction network *forbids* the S_P permutation
symmetry from breaking — all P sectors are always present, so the melt
can only be interface dissolution. Remove the pin and nothing in the
model prefers any sector: the system must **choose** one spontaneously.
The sector basis becomes dynamical in the meaningful sense, and the
order–disorder point becomes a genuine transition:

- T_c ≈ 0.11, fully bracketed susceptibility peaks growing
  3.6 → 20.0 across L ∈ {16, 24, 32, 48},
- **χ_max(L) ∝ L^b with b = 1.60 ± 0.22** — 7.2σ from the pinned
  crossover, 0.6σ from the exact 2-D 3-state Potts γ/ν = 26/15.

The dichotomy, stated once: **coexistence (pinned) ⇒ smooth network
dissolution; free choice (unpinned) ⇒ spontaneous S_P breaking in the
Potts class.** Same model, one constraint toggled.

### 6. Tying the universality down (`n3_potts_nu.py`, `n3_potts_3d.py`)

Two independent checks of the Potts identification:

- **The second exponent.** Binder-cumulant data collapse over (T_c, ν)
  measures ν independently of γ/ν. With deep thermalisation (a first,
  shallower pass produced spurious negative Binder values from slow
  melting near T_c — diagnosed by magnetisation histograms, which are
  cleanly unimodal at equilibrium), the collapse selects ν ≈ 1.0 with a
  2×-residual band of [0.53, 1.60]: **the Potts ν = 5/6 sits inside the
  band**. A consistency check, not a precision measurement — the
  estimator carries interpolation bias on coarse grids (quantified in
  `tests/test_potts_universality.py`) — but both independent exponents
  now agree with the 2-D 3-state Potts class, and the unimodal
  histograms rule out the weak-first-order alternative in 2-D.
- **The 3-D prediction — where the sharper instrument overturned the
  first reading.** The 3-state Potts class makes a falsifiable claim: in
  3-D the transition is *first order*. The hysteresis/Binder scan
  (L ∈ {8…16}) found suggestive signatures — a persistent hot/cold
  window, Binder minima deepening with L. The decisive observable is the
  **energy histogram** (`n3_latent_heat.py`): first order means latent
  heat means a bimodal energy distribution at the transition. It is not
  there. Every pooled hot+cold histogram is cleanly unimodal at every
  size, and the branch energy separation Δe ≤ 0.005 *shrinks* with L —
  even while the magnetisation branches still disagree. Same energy,
  different order: the "hysteresis" was **kinetic** (slow 3-D
  coarsening), not phase coexistence. Verdict: no latent heat at
  Δe ≲ 0.001 resolution; the 3-D transition is continuous or
  unresolvably weakly first order — a candidate genuine deviation from
  the discrete-Potts expectation, and a case study in why suggestive
  metastability evidence must be checked against the energy channel.

### 7. The S-functional at criticality (`n3_s_criticality.py`)

The program's last measurement brings it home to the theory's central
object. On the unpinned ensembles across T_c: the **distinction term ΔC
peaks exactly at the transition** (walls and fluctuations are densest at
criticality) while the standing coherence falls through it
order-parameter-like. Because the two halves of S pull opposite ways and
cross at T_c, the S optimum sweeps across the transition as the
integration weight varies — sitting *at* T_c for w ≈ 0.05. There is a
window of integration weights in which the theory's own functional
selects the critical neighbourhood — the ordered-but-maximally-
fluctuating regime — rather than deep order or deep disorder. Given the
URP's framing of S-climbing systems living at the edge between rigidity
and dissolution, this is the program's most theory-facing verdict.

### 8. Dynamical capacity at criticality (`n3_kappa_criticality.py`)

The proxy κ is then promoted to the real thing: the engine's capacity
field — consumed by load, regenerating with slack, diffusing — co-evolves
with (ψ, U) and gates the coherence coupling locally. Four measured
verdicts close the capacity loop:

- **κ troughs exactly at criticality**, at every consumption strength
  (⟨κ⟩_min from 0.34 down to 0.02 as c grows): the load that consumes
  capacity *is* the distinction term, and ΔC peaks at T_c.
- **The κ-as-soil wall deficit appears in the thermal state**
  (κ_wall = 0.22 vs κ_bulk = 0.39 in the ordered phase at strong
  consumption) — the engine's corpus-rooting picture, thermally.
- **Scarcity destabilises sector order**: the transition shifts down
  with consumption.
- **Scarcity relocates the S-optimum to the critical point.** With
  abundant capacity the optimum rests in deep order; once the budget
  binds it jumps to the ΔC peak at T_c — scarcity taxes integration but
  not distinction. The repo's earliest capacity verdict ("selection is a
  scarcity phenomenon") returns at the level of criticality itself: a
  capacity-bound S-climbing system is pushed toward the edge.

### 9. The S-landscape: a level crossing (`n3_s_landscape.py`)

The relocation's *character*, resolved across the (consumption, weight)
plane. S(T) is not single-peaked — it carries two competing maxima at
once, an ordered one (coherence-rich, low T) and a critical one (the ΔC
peak at T_c). The global optimum T\* is a **step function**: it sits at
the ordered peak, then jumps ~3× to the critical peak with no
intermediate value, exactly where ΔS = S_ordered − S_critical crosses
zero. This is a level crossing — two maxima trading global rank — the
S-landscape analogue of a first-order transition, not a continuous
drift. The boundary c\*(w) sweeps monotonically (a larger integration
weight makes the ordered phase more valuable, so more scarcity is needed
to abandon it), and because the weight is pure post-processing the whole
(c, w) map is extracted from one Monte Carlo scan. The verdict is about
the optimum-location observable, not a thermodynamic transition of the
ensemble — the underlying melt remains the continuous Potts crossover.

### 10. Memory recall at criticality (`n3_seed_rooting.py`)

The program's final measurement reaches back across the whole repository:
the **memory corpus** — the scalar engine's mechanism for storing stable
structure and re-seeding it to prevent collapse — had never met the
thermal capacity field. The corpus roots seeds under a κ-as-soil rule (a
seed unfolds only where local κ ≥ 0.3, and rooting consumes κ);
``project_genesis.sector_seeds`` ports that exact rule to the thermal
ψ∈ℂ³ ensemble. Since capacity troughs at criticality, the prediction is
that **recall fails there** — and it does, with a twist worth stating:
the recall capacity (fertile-soil fraction) collapses to zero *before*
the order parameter does (recall ≈ 0 at T where m ≈ 0.44). A
capacity-bound system can hold the structure it already carries but can
no longer regenerate what it loses — memory dies before order. Recall is
possible only in a cold, low-consumption corner (heat and scarcity each
starve it), and is self-limiting: rooting consumes soil, so recall
spreads into fresh ground rather than piling onto exhausted spots. The
two halves of the codebase, built for different purposes, turn out to
tell one story — capacity governs not just what structure forms, but what
structure can be *remembered*.

### 11. Can regeneration rescue memory? (`n3_recall_recovery.py`)

The recall collapse is not a fixed barrier — it is a property of how fast
the soil regenerates. Since steady-state capacity balances consumption
against recovery (``κ = r/(r + c·load)``), scanning the recovery rate ``r``
climbs the recall edge to higher temperature, and past a threshold rate
the recall capacity *outlives order*: at fast recovery it stays high
(0.6–0.87) deep into the disordered phase where the Potts magnetisation
has fallen to ≈ 0.03. The prerequisite for memory survives where the
field's long-range order does not. And the rescued recall curve is
non-monotonic — it dips at T_c and recovers on both sides — because the
distinction load that consumes κ peaks at criticality (section 7): the
transition itself is the memory bottleneck, flanked by an ordered phase
(low load, high κ) and a disordered phase (low load, κ restored by
regeneration). Whether a capacity-bound system can remember across a
critical transition is not fixed by the transition — it is set by the
one dial the theory already contains, the rate at which capacity heals.

## What the program adds up to

The URP claim under test was never "three is a magic number" but that a
capacity-bounded recursive field *selects* three-sector structure. The
thermodynamic program sharpened that into statements a lattice ensemble
can answer, and answered them:

- the gauge sector confines (calibrated, measured);
- the three-sector junction network is the *only* colour-neutral network
  that exists as a thermal state, in 2-D and 3-D;
- its stability is coupling-controlled, with measured boundaries;
- and the sector *choice* itself, once nothing external pins it, is a
  spontaneous symmetry breaking in exactly the universality class the
  three-fold symmetry dictates.

Honest limits, kept in one place: 2-D/3-D rather than 3+1-D; small
lattices, lattice units, no continuum limit; the sector potential is the
model's structure, not derived from the URP field equations; exponents
are consistency measurements at single coupling points, not universality
proofs. Each experiment's own docstring and the README sections carry the
finer-grained caveats.
