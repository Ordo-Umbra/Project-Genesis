# The Principle

### A short statement of the Universal Recursion Principle — what it claims, what has been measured, and what would refute it

*This document assumes nothing. It does not summarise the repository, and it is
not organised by the order in which things were discovered. It states the
principle, derives what follows from it, and marks every claim with its
standing: **[measured]** — a pre-registered experiment in this repo with a
verdict; **[framework]** — a coherent consequence of the principle that has not
been tested here; **[declined]** — a claim the programme explicitly refuses to
make. If you read one thing, read §1 and §6.*

---

## 1. The principle

Start with an asymmetry that is provable, not postulated.

Take any formal system strong enough to talk about itself. It has two different
capacities:

- **what it can represent** — the structures it can write down and distinguish;
- **what it can stabilise** — the structures it can actually prove well-founded,
  and so rely on.

These are not the same, and the gap between them is not an accident of poor
axiom choice. For any consistent, recursively axiomatised system,
**stabilisation always falls strictly short of representation**, and closing the
gap by adding what's missing simply opens a new one. This is Gödel's result read
structurally rather than as a limitation: *incompleteness is the room a system
has left to grow into.*

The Universal Recursion Principle proposes that this asymmetry is not peculiar
to logic. Any system that (a) makes distinctions and (b) must hold them together
under finite resources inherits the same shape. So it proposes a single quantity
such systems climb:

```
    S  =  ΔC  +  κ · ΔI
```

- **ΔC — distinction.** New structure: boundaries, gradients, differences,
  articulations. The making of *more kinds of thing*.
- **ΔI — integration.** The binding of those distinctions into something that
  coheres — that can be held, remembered, acted on as one.
- **κ — capacity.** The finite resource that decides *how much integration the
  system can afford*. It is consumed by load and regenerates with slack:
  `∂ₜκ = D∇²κ + r(κ₀−κ) − c·load·κ`.

That is the entire theory. Everything below is consequence.

### The one asymmetry that does the work

Distinction is **cheap**. Any structured system makes it for free — noise makes
gradients. Integration is **expensive**: it must be paid for in capacity, and
the bill comes due exactly where distinction is richest.

So a recursive system can always distinguish more than it can integrate, it
never catches up, and *in failing to catch up it builds structure*. That is the
generative gap, and it is the engine of everything that follows.

---

## 2. The asymmetry is real, and it cannot be bought

Two independent measurements, on unrelated instruments, find the same thing.

**The separation is a cliff, not a slope.** **[measured]** Measure what a field
represents (all junctions it forms) against what it integrates (junctions that
bind the *whole* palette). The integrated fraction is **1** at three sectors and
**collapses to 0** at four or more. Sweeping capacity by more than 10× changes
the *density* of distinction but never moves that fraction. Past the threshold,
**integration cannot be bought — only reached by changing structure.** That is
the field's echo of Gödel's own move: not computing longer inside the system,
but extending it.

**The separation has a dimension.** **[measured]** Look at nested regions of a
field. Distinction scales with the **volume** (`n = 2.00` in 2-D); integration
scales with the **surface** (`n = 1.00`). The gap is `1.03` — *exactly one
dimension*. And again it is capacity-invariant: across a 12× change in coherence
length the exponent moves by 0.04.

The shortfall is therefore not a resource problem. It is structural, in the same
way `I(F) < C(F)` is structural. A system does not fail to integrate everything
because it is underfunded; it never had the option.

*(Honest note: for a classical field an area law is expected. The content is the
exact one-dimension gap and its invariance, not the existence of an area law.)*

---

## 3. What follows: structure

If integration is the scarce term, then **the structures that persist are the
ones that can be integrated** — not the ones that are most articulate.

**Three is special.** **[measured]** Only a three-fold junction can carry a
complete palette at once. With two kinds you cannot form such a junction at all;
with four or more you form *more* junctions but essentially none that bind
everything. Three sectors is not a constant put in by hand.

*Correction — this was overclaimed here, and the correction matters.* An
earlier version of this section said the selection is "invisible when capacity
is abundant" and "appears only when a capacity budget binds." **That is
false**, it contradicted §2 of this same document, and it has now been measured
directly. **[measured]** Running the palette sweep across the capacity axis —
from the published arm that has *no capacity field at all*, through free
capacity, to a budget so tight that mean κ falls by 98% — the peak stays at
`P = 3` at **every** level, and among the arms where the measurement is valid
the margin does not depend on capacity. Where scarcity does change the margin,
it *destroys* the selection rather than revealing it, and most of that change
is the field fragmenting to lattice scale rather than binding responding to
scarcity.

So the honest statement is the one §2 already made: **the selection is
geometric, and capacity-invariant.** In two dimensions a generic vertex is
three-fold — Plateau's law — so a junction can carry the whole palette only at
`P = 3`. That is a real and sharp result, and it is *not* evidence that the
capacity principle is doing work here: a plain multiphase field with no κ
selects three just as sharply. The evidence for substrate-independent capacity
dynamics is §5, not this.

**Matter is the cells of a tessellation.** **[measured]** Held near its critical
point, the field partitions space into domains, walls, and junctions — a
countable inventory with exact invariants. Where walls meet in threes, Euler's
formula fixes the proportions at `2:3:1`. The number of families is `d+1` — three
in two dimensions, **four in three** — so *the number of generations is the
dimension of space plus one*. The abundances are topologically protected, not
energetically tuned.

**Binding is derived, not dialled.** **[measured]** "Adding the same distinction
twice does not expand the structure" — no-cloning — priced against the capacity
free energy gives a repulsion that is non-negative by a concavity theorem and
produces a **stable separation with no free parameter**. That floor is the
binding radius of a conjugate pair.

**Spin, statistics, and a fermion.** **[measured]** A headless (nematic) order
parameter admits **half-integer** defects: the ±½ disclination, whose oriented
director flips under a 2π rotation and returns only after 4π — the signature of
a spinor. Two identical such defects, exchanged, pick up **−1**. That sign is
the *same* as the single defect's 2π rotation sign, and — once the field is
given a real U(1) gauge field carrying quantised flux — the *same* as the
Aharonov–Bohm holonomy. **Spin, statistics, and gauge give one number, measured
three independent ways.** The constituent count of a composite is fixed at three
by the same `N⋆ = 3`: a spin defect's winding must cross every sector.

*Dimensional audit — half of this is 3-D and half is planar.* **[measured]**
Every module above represents the director as `θ = ½·arg ψ`, and a complex phase
is a point on a circle, so they describe a director confined to a plane: order
parameter space `RP¹`, where `π₁ = Z`. A director free to point anywhere lives
on `RP²`, where `π₁ = **Z/2**` — only two classes. Run on a real
three-component director, with the plane-confinement as the only difference
between arms: seeded identically off the symmetric saddle, the **integer** line
escapes along its own axis and ends non-singular (core `|n_z|` 0.66 → **0.98**),
while the **half-integer** line pushes the same tilt back out and stays singular
(0.66 → **0.02**). That asymmetry is `π₁(RP²) = Z/2`, measured rather than
cited.

So: **the ½ defect is a genuine, protected, singular object in three
dimensions** — there is a 3-D spinor, the exchange sign needs exactly two
classes and `Z/2` supplies exactly two, and the 4π double cover was always a
`π₁(SO(3)) = Z/2` fact independent of the order parameter's dimension. What does
*not* survive is the integer-graded ladder: in `RP²`, `+½` and `−½` are the
**same class**, a half-line is its own antiparticle, and there is no signed
winding to count. **The "winding must cross every sector" argument for the
constituent count is therefore a planar argument and does not transfer as
written.** It needs restating in `Z/2` terms or marking as 2-D.

---

## 4. What follows: gravity and cosmos

The capacity field descends a free energy
`F[κ] = ∫[(D/2)|∇κ|² + (r/2)(κ−κ₀)² + (c/2)·load·κ²]`. **A field whose free
energy responds to mass-like load is a theory of gravity.**

**[measured]** Load digs a screened well; two masses attract with `V ∝ m₁m₂`;
inertial and gravitational mass coincide because both are the same distinction
density. Give the field a finite update rate `τ` and it acquires a **causal cone
at `c = √(D/τ)`** — *the speed limit is the update rate* — with drag, radiation
carrying only even harmonics (the no-dipole rule), and mergers by plunge.

**[measured]** At cosmological scale: a genuine scale factor, Hubble drag, matter
whose density follows from the topological charge spectrum, an equation of state
(`w = 0` for cold forms, `w = −1` for the vacuum), a conserved stress-energy
tensor, and the Friedmann equation recovered as a variational constraint. **Dark
energy is derived, not inserted**: the capacity self-maintenance term `r(κ₀−κ)`
is an energy that does not dilute as space expands — exactly a cosmological
constant.

**[measured, a negative]** But this gravity is **not** boundary-encoded. A flux
integral does not know the mass it encloses: the reading decays like `e^{−R/ξ}`
and depends on how the mass is arranged. A Gauss law is recovered **only** in the
unscreened limit `r → 0`. Boundary-encoding is a property of a *massless*
mediator; this one has a mass.

This exposes a real tension inside the theory: **`r → 0` gives boundary-encoded
gravity; `r > 0` gives derived dark energy. They are the same parameter.** That a
very small `r` might give both a long screening length and a small Λ is
suggestive and **[framework]**, not a result.

---

## 5. What follows: the law is not about physics

If the principle is about recursion under finite capacity, it should not care
what the system is made of.

**[measured]** The programme's deepest result — *a capacity-bound system is
pushed to criticality exactly when maintaining order costs capacity* — has been
run on three structurally disjoint substrates: a lattice field, an attractor
network with learned couplings and no geometry, and a population of coupled
oscillators with no memories and a continuous order parameter. The capacity law
is imported as literally the same code object in each. All three show the same
level crossing under the same toggle: order that is *free* is scarcity-proof;
order that must be *maintained* is driven to the edge.

**[measured, and it narrows the claim]** An audit of the units
(`n3_crossing_prediction`) establishes what that agreement is and is not.

- The consumption `c` and the recovery rate `r` are **one parameter**, not two:
  the capacity law depends only on `u = c/r`. Every `c` the transplants quote is
  really `c/r`, and `r` is the axis's unit.
- The mechanism's condition needs no toggle. Eviction happens **iff** the
  capacity floor the load permits, `1/(1 + u·L_o)`, falls below the crossing
  capacity `κ_o⋆` fixed by the measured curves. The published four-condition
  result is a comparison of two numbers.
- Where the relocation is a single jump it **is** exactly the two-point
  competition the mechanism describes — integration funding decaying past the
  distinction gap — to **0.1%** on the Hopfield network.
- But it is not always a single jump. Under a flat load the optimum walks the
  upper convex hull of the `(ΔI, ΔC)` cloud; that hull has two vertices on the
  Hopfield network and **three** on the driven oscillators, which stop at an
  intermediate partially-synchronised state first.
- And the oscillators' load is **mostly the drive**. The injected phase noise
  contributes `σ√dt` per step to the measured activity, and the ordered phase's
  reading is `1.004×` that floor as `dt → 0`: `activity` never subtracts the
  perturbation, where the Hopfield query drive explicitly does. Between the two
  points that matter the load ratio is `ρ ≈ 1.0`, against `34` on the network.

So on the published reading the third substrate agrees on the **verdict** while
differing on the **mechanism**: a uniform tax that costs the integration-funded
optimum its lead because `ΔI_o > ΔI_⋆`, not a differential tax on order. `1/3`
registered predictions held.

**[measured, and it repairs half of that]** The last bullet named a specific
fix, and `n3_kuramoto_repair` carried it out. `simulate` now also reports
`repair_rate` — the spread of the *deterministic* phase velocity,
`std_i(v_i − ⟨v⟩)` with `v_i = ω_i + K·r·sin(ψ − θ_i)` — which contains none of
the injected noise and, being a rate rather than a per-step displacement,
carries no `dt`. `activity` is unchanged and bit-identical, so the three
published results remain reproducible as stated. On the corrected reading
(`2/3` held):

- It measures the system, not the apparatus. Across a 16× change in the
  integrator's step `repair_rate` moves **7.8%** where `activity` moves
  **4.4×** — and a deterministic locked rhythm still pays nothing, so the
  toggle the transplant design needs survives.
- The load profile is **not** flat after all: `ρ = 1.70`, against `1.04` for
  the contaminated reading. The differential tax the mechanism describes is
  genuinely present here; the noise floor was hiding it.
- So the **condition** does transplant to all three substrates. Order that must
  be maintained pays, and scarcity evicts it.

What did not repair is the **destination**. The route is set by the `(ΔI, ΔC)`
cloud, which no choice of load metric can move; correcting the load made the
oscillators' walk *longer* — four stops rather than three — because taxing the
ordered phase harder evicts it earlier, onto a state only part-way to the ΔC
peak.

**[measured — and it retracts the reading above]** That was written here as
*anatomy, not convention*, on the reasoning that a load metric cannot move the
cloud. The reasoning has a hole: the cloud is not raw data either. **ΔC is
itself a convention on both substrates** — a structured-overlap threshold on the
network (published `0.3`), a fixed frequency-bin width and entrainment tolerance
on the oscillators (published `0.4` / `0.3`) — and none of those numbers is
derived from anything. `n3_anatomy` sweeps each substrate's own ΔC convention
across a range a reader would accept without argument. `2/3` held:

- **The route is not a property of either substrate.** Both hulls flip between
  two and three vertices inside their own sweeps. "The network crosses once and
  the oscillators walk" was a statement about the dictionaries.
- **The eviction condition is untouched**, in `16/16` arms. The capacity floor
  sits orders of magnitude below `κ_o⋆` everywhere, so this was never a close
  call a convention could tip. Everything above about the *condition* stands.
- **The critical neighbourhood does not fully survive.** The ΔC peak slides
  monotonically with the network's threshold — `T⋆ = 1.20 → 0.70` across
  `0.15 → 0.50` — and leaves the published `±35%·T_c` window at the two most
  permissive settings. One of those, `0.20`, is `3.5σ` above chance overlap for
  `N = 300`: strict enough that its failure is not an artefact of an absurd
  setting.

So the claim survives at one resolution and not at the next two. **Does a
relocation happen?** Yes, and convention-independently. **Does it reach
criticality?** At the published conventions yes, but the ΔC peak's location is
a function of an underived threshold, so this is not established in general.
**Which state, by which route?** Not established at all.

That is a narrower result than §5 has claimed at any point, and it is the one
the measurements support. The next thing worth doing is not another substrate:
it is a principled ΔC — a distinction reading whose scale comes from the
substrate rather than from a chosen cut.

**[measured]** Where the law meets external ground truth — a learning system —
capacity-gated plasticity beats plain SGD on a compositional task sequence,
paired and powered (`p < 0.001`), and the advantage is scarcity-graded. Its
boundary is equally measured: given the same stored information, plain replay
beats it.

---

## 6. Why this rather than "one slice of possibility"

The standing objection to any framework like this is fatal if unanswered:
*reproducing known structure is under-determined.* Many frameworks can hit a
number. Hitting it shows only that yours is one option among many.

The answer is not a better fit. It is to **run the alternatives on the same
instrument and see which corner survives.** **[measured]**

| kinds | distinction | integration | still moving |
|---|---|---|---|
| 2 | 0.031 | **0.00000** | 0.005 |
| **3** | 0.046 | **0.00549** | 0.008 |
| 4 | 0.059 | 0.00005 | 0.011 |
| 5 | 0.069 | 0.00000 | 0.013 |
| 6 | 0.077 | 0.00000 | 0.013 |

Read it carefully, because the shape of the argument matters more than the
numbers:

- **Distinction does not select.** It rises monotonically — it prefers the
  *largest* palette. More kinds simply means more structure.
- **Integration selects, and it is a monopoly.** Only one palette size can bind
  what it distinguishes. Two kinds cannot form the junction at all; four or more
  make *more* junctions and bind none of them completely.
- **The joint criterion has a unique interior optimum.** Requiring both — bound
  *and* still generating, the theory's own "good seed" test — selects three by
  **88×**. And three is best at *nothing individually*: it carries less
  distinction and less churn than every larger palette.

**The possibility space is open.** Every alternative runs; every one makes
structure; the larger ones make more. What they cannot do is hold it together.
This corner is not chosen by fitting — it is what is left standing.

**This is the falsifiable core.** Another palette scoring within 3× on the joint
criterion sinks it.

*One audit, because this measure is load-bearing.* The integration axis is
full-palette junction density, and that measure has a known failure mode: once
domains shrink to the lattice, every neighbourhood contains every sector for
the trivial reason that neighbouring cells are uncorrelated, so a field that
binds **nothing** scores near-maximally. It was caught in `n3_expansion`, where
the most fragmented configuration in a sweep scored the *highest* integration.
The measure now carries a guard — a majority filter plus a resolved-domain-scale
flag — and the sweep above was re-run with it. **[measured]** Every palette is
resolved in both dimensions; the guard moves the numbers by less than 25% and
changes no ranking. In 2-D it *strengthens* the result (the joint margin goes
88× → 272×, because `P = 4`'s small integration was itself partly texture). In
3-D `P = 3` still wins, and the `P = 3`:`P = 4` integration ratio moves 5.6× →
6.4× — still short of the 10× the monopoly claim asks for. **So the 3-D
weakening recorded below is real, not an artifact of the measure.**

---

## 7. What this is not

The boundaries are load-bearing: they are what make the rest credible.

- **It is not general relativity.** κ-gravity is a screened, Newtonian-analogue
  force with no metric, no covariant action, and a minisuperspace cosmology.
- **It is not the Standard Model.** The fermion here is *classical and
  topological*: real spin-½, real exchange antisymmetry, a real gauge field with
  quantised flux — but no Fock space, no `{ψ, ψ†}`, no Pauli principle between
  identical quanta.
- **It does not derive the constants of nature.** **[declined]** Earlier work in
  this corpus claimed that two fixed numbers (`β ≈ 0.09`, `κ ≈ 0.22`) predict
  helium's ionisation energy, bond lengths, and fusion cross-sections. This
  programme's own measurement **retired `κ ≈ 0.22` as a constant**: it is a
  coordinate on a rising, plateau-free, cutoff-dependent curve, sliding with the
  reading convention. Quark masses, mixing angles, and generation abundances are
  **declined**, not pending.
- **The two halves are not yet numerically one.** The coherent-fraction operator
  transfers cleanly between the gauge and sector sectors; the *number* does not,
  and the reason is understood (their topologies renormalise oppositely).
- **Most results are 2-D, on one lattice, in one model family.** The selection
  argument selects within *that* family under *those* measures. It does not
  establish that our universe is the attractor of reality.

---

## 8. How to check it

Every claim above is a file you can run.

```
python experiments/n3_selection_sweep.py       # §6, the selection argument
python experiments/n3_capacity_separation.py   # §2, the cliff
python experiments/n3_capacity_gating.py       # §3, the correction (geometry, not scarcity)
python experiments/n3_area_law.py              # §2, the scaling dimension
python experiments/n3_exchange_statistics.py   # §3, the exchange sign
python experiments/n3_ab_statistics.py         # §3, spin = statistics = gauge
python experiments/n3_kappa_gravity.py         # §4, gravity
python experiments/n3_boundary_gravity.py      # §4, the negative
python experiments/n3_criticality_transplant.py  # §5, substrate independence
python experiments/n3_kuramoto_transplant.py     # §5, the third substrate
python experiments/n3_crossing_prediction.py     # §5, the units audit of both
python experiments/n3_kuramoto_repair.py         # §5, the drive-free load
python experiments/n3_anatomy.py                 # §5, is the destination real?
```

Each prints its own pre-registered predictions, its verdict, and its honest
scope — including when the prediction failed. 872 checks in `tests/`
lock the central claims so they cannot drift silently.

**What would refute the whole thing:** a palette other than three scoring
comparably on the joint criterion (§6); a capacity sweep that moves the
integrated fraction (§2 — run, and it does not); or an integration measure that scales with volume
rather than surface (§2). Those are the load-bearing measurements. The rest is
consequence.

---

*One principle: a system that can distinguish more than it can integrate never
closes the gap, and builds in the shortfall. What persists is what can be held
together — which is why there is structure at all, why it comes in the
proportions it does, and why the alternatives, which are perfectly possible, are
not what we find.*
