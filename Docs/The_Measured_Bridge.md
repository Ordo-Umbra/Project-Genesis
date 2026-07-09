# The Measured Bridge

### Ordinals → functors → instantons, as one chain of measurements

*A closing synthesis. `The_Generative_Gap.md` states the thread and points
the way; this document reports the arc as finished work — the single claim
of the functorial-bridge program, and the four measurements that carried it
from a statement about ordinals to a number in the 4-D SU(3) vacuum.*

---

## The claim, in one sentence

The functorial-bridge papers make a startling proposal: that **logical
incompleteness and the QCD vacuum are the same structure seen twice**. A
formal system can name more than it can prove; a gauge vacuum can distinguish
more field configurations than it can coherently bind. Both shortfalls,
the claim goes, are the *same gap* — and the mechanism that partially closes
each (reflection in logic, instanton tunnelling in physics) is one functor

    F : 𝒪  ⟶  𝒬

carrying theories to vacuum sectors, reflection to tunnelling, and — if it is
real — carrying a *number* across: the exchange rate `κ ≈ 0.22`, the
instanton fraction of the gluon condensate, as the physical image of the
distinction-to-integration ratio the URP functional `S = ΔC + κ·ΔI` is built
from.

That is a metaphysical claim, and the testbench cannot prove it. What the
testbench *can* do — and now has done — is walk the chain the claim asserts,
one measurable link at a time, and report where each link holds and where it
frays. This is that report.

---

## Link 1 — The gap is real, and it is a cliff

*`n3_capacity_separation.py` · [gap](../README.md#the-generative-gap-measured)*

The chain begins in logic: the **Capacity Separation Theorem** says a
recursive system's representational capacity strictly exceeds its inferential
capacity, `I(F) < C(F) = ω₁^CK`. The field analogue is commensurable by
construction — distinction as *every* triple junction the field represents,
integration as the *full-palette, colour-neutral* junctions it can actually
bind (so `I ≤ C` exactly, the same object counted two ways).

Measured, the shortfall is not gradual. The integrated fraction `φ = I/C` is
**1 at P = 3** — every represented junction integrated, "complete" — and
**collapses to 0 for P ≥ 4** — junctions represented, none integrable,
"incomplete." A structural cliff at the three-fold threshold. And it is
**capacity-invariant**: sweeping κ moves the *density* of distinction by more
than 10× and never moves the fraction. Past the threshold, integration
cannot be *bought*; it can only be *reached* by changing structure — dropping
to three. That is Gödel's own move (extend the system, don't compute longer)
and the theory's expressivity threshold, in one measurement.

**Holds.** The gap exists, it is sharp, and it sits exactly at N⋆ = 3.

---

## Link 2 — The correspondence is a functor, not an analogy

*`n3_functor_bridge.py` · [functor](../README.md#the-functor-logic--vacuum-measured)*

A correspondence between two categories is only a *functor* if it preserves
structure — if the image of a state depends on the state, not on the path
taken to it. The bridge is built as an actual such map: a **reflection
ladder** of field configurations (gentle cooling from a random,
maximally-representational start — the field's `F_{n+1} = F_n + Con(F_n)`),
with the functor's image at each rung the field's topological content.

As integration climbs the ladder (`I = 0.33 → 0.94`) the topological image
falls (`T = 0.22 → 0.02`): one ladder read through two instruments, a single
monotone map. The decisive test is **path-independence** — run the ladder at
different cooling *rates* and compare the image at *matched* integration. The
curves collapse to a **1.9% relative scatter**: the topology depends only on
the integration level reached, not the history of reaching it. That is the
functor axiom, measured. A single action-descent gradient `∇S` drives both
ladders at once — the naturality `∇S : D ⇒ κ·Int`.

**Holds, with a known sign.** The map is a genuine structure-preserving
functor on the field. Its direction comes out *contravariant* — the thermal
regime's topology is the disordered instanton *gas*, the opposite ordering
from the coherent *condensate* that integrates the true θ-vacuum. That sign
is the reason Link 3 had to be built in the real theory, not the 2-D
stand-in.

---

## Link 3 — The number appears in the 4-D SU(3) vacuum

*`gauge_topology.py`, `n3_su3_topology.py`, `n3_su3_gradient_flow.py` ·
[Stage 1](../README.md#4-d-su3-topological-charge-stage-1) ·
[Stage 2](../README.md#4-d-su3-gradient-flow-the-instanton-fraction-of-the-vacuum-stage-2)*

The deepest link needed the physical side of the gap to carry a real
topological-charge instrument — the thing the gauge sector had lacked all
along. It was built in two stages, in the genuine theory the whole program
had only ever approximated.

**Stage 1 — the instrument.** The dimension-agnostic gauge Monte-Carlo runs
a 4-D SU(3) Wilson ensemble directly; the added piece is the **clover**
field-theoretic topological charge `Q = (1/32π²) Σ ε_{μνρσ} Tr[F_{μν}F_{ρσ}]`
with cooling. It is validated against the one check that cannot be faked:
**pure-gauge configurations read Q = 0 exactly.** Cooled charges quantise at
evenly spaced, Z-renormalised levels (single instantons at `|Q| ≈ 0.84`, the
standard coarse-lattice suppression), and the vacuum reproduces **topological
freezing** — free tunnelling at strong coupling, sticking in one sector
toward weak coupling — the known critical slowing of topology, seen directly.

**Stage 2 — the number.** Cooling is replaced by the **Wilson gradient flow**
(a Lüscher RK3 integrator), a genuine renormalisation-group smoothing. It
buys two things Stage 1 could not. A *scale*: the clock `t² E(t)` crosses the
reference `0.3` at a definite flow time `t₀`, the standard Wilson-flow scale
`√(8 t₀)`, and under the flow `Q` sharpens off the renormalised levels toward
genuine integers (`Z → 1`). And the *observable* the whole bridge pointed
at: the **self-dual fraction**

    f_SD  =  Σ_x |q(x)|  /  Σ_x e(x)   ∈ [0, 1] ,

the fraction of the field energy that saturates the Bogomolny bound
`e(x) ≥ |q(x)|` — that is carried by (anti-)self-dual, instanton-like
structure rather than structureless UV field energy. It is the lattice proxy
for the instanton fraction of the gluon condensate. Read at the RG-clean
scale `t₀`, it **drifts through κ**:

| coupling `β_g` | flow scale `t₀` | self-dual fraction `f_SD(t₀)` |
|:---:|:---:|:---:|
| 1.7 | 0.39 | 0.187 ± 0.003 |
| **1.8** | **0.46** | **0.221 ± 0.004** |
| 1.9 | 0.95 | 0.352 ± 0.013 |

At `β_g = 1.8` the self-dual fraction of the 4-D SU(3) vacuum lands
essentially **on** `κ = 0.22` — the number the functor was claimed to carry,
appearing at the principled flow scale, in the real theory.

**Holds in the neighbourhood, not on the nose — and the continuum push says
where.** `f_SD` is a *single-scale reading of a monotone-rising quantity*, so
across the coupling window it *brackets and crosses* κ rather than sitting on
it. A dedicated continuum trend (`n3_su3_continuum.py`) then localises the
uncertainty exactly: the reading is **volume-converged** (flat in `L` at fixed
coupling — 0.218/0.221/0.219 over `L = 6/8/10` at β_g = 1.8) but strongly
**cutoff-dependent** (0.189 → 0.344 as `a` shrinks across β_g = 1.7 → 1.9),
and a linear `a²` extrapolation gives `f_SD → 0.44` as `a → 0` — *above* κ. So
the β_g = 1.8 agreement with 0.22 was a **coarse-lattice coincidence**, not a
cutoff-stable determination. What survives is real and weaker than the
headline: the physical side of the gap carries a genuine, validated,
RG-scaled instanton fraction of `O(0.2–0.4)` — the self-dual fraction is an
honest instanton-content observable, but not by itself a scheme-free estimator
of κ.

---

## The chain, and its edge

Read end to end, the arc is a single measured statement:

> A recursive field distinguishes more than it can integrate (**Link 1**, a
> cliff at three). The map from how-much-it-integrates to how-much-topology-it-
> carries is a genuine functor — path-independent, structure-preserving
> (**Link 2**). And the exchange rate that functor was claimed to carry — the
> instanton fraction `κ ≈ 0.22` — has a real, validated, RG-scaled image in
> the 4-D SU(3) vacuum: the self-dual fraction of the field energy, an
> `O(0.2–0.4)` number whose coarse-lattice reading *crosses* κ but whose
> continuum trend sits above it (**Link 3**).

The edge is bright and it is honest — and the last link's edge is now
*measured*, not merely flagged. The testbench establishes the *structure* of
the bridge — the gap's existence and sharpness, the map's functoriality, the
existence and magnitude of a genuine instanton fraction on the physical side.
It does not establish the metaphysical identity of logical incompleteness with
the QCD vacuum, and it does **not** pin κ: the continuum push that was the
obvious next step was taken, and it corrected rather than confirmed the
headline — the self-dual fraction is volume-converged but cutoff-dependent,
extrapolating *above* 0.22, so that observable alone is not a scheme-free
estimator of κ. Pinning the number would need genuinely finer lattices
(against topological freezing) and the matched operator-product-expansion
condensate the self-dual fraction only stands in for. Those are the frontiers
the arc leaves open, having built — and honestly stress-tested — the
instruments they require.

What the program has shown is smaller than the metaphysics and larger than a
demo: that the generative gap — the shortfall that, in failing to close,
builds structure — is real, is functorial, and carries, on the physical side,
a genuine instanton fraction in the neighbourhood the papers named — measured
carefully enough to know exactly how far that last claim can, and cannot, yet
be pushed. The bridge, as far as an honest measurement takes it, is measured.

---

*See `The_Generative_Gap.md` for the full `S = ΔC + κ·ΔI` program the three
links sit inside, and `Thermal_Sector_Program.md` for the individual
experiment records.*
