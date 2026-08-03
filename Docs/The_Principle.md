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

**[measured — the only load-bearing claim here that came out of its convention
sweep with nothing added to it]** Both of §2's measurements have now been swept
for hidden conventions,
because the rest of this document is a list of readings that turned out to
depend on constants nobody derived. The gap has two such constants: the
**fitting window** (four nested regions spanning a factor of two) and the
**noise floor** subtracted from the correlation kernel. `n3_gap_conventions`
moves both.

| convention swept | `n_C` | `n_I` | gap |
|---|---|---|---|
| fitting window — 5 ladders, incl. sub-ranges and shifted | `0.046` | `0.017` | `0.029` |
| noise-floor band — 4 defensible bands | `0.000` | `0.015` | `0.015` |
| noise-floor magnitude — `0×` to `2×`, `0×` = no floor at all | `0.000` | `0.096` | `0.096` |

Every entry is a full spread, not an error bar. The largest motion under any
convention swept is `0.096` — **38% of the tolerance the original experiment
set for itself**, and it is the case where the noise-floor correction is
dropped entirely, which no reader would ask for. At the published conventions:
`n_C = 1.99`, `n_I = 0.95` after a calibrated instrument bias of `+0.12`, gap
`1.04`. The one-dimension gap is a fact about the field, full stop.

**And the audit found something it was not looking for.** The prediction being
tested was the split that had held in five previous audits — *locations move
under a convention change, differences of co-measured quantities do not,
because the compared readings share the distortion and it cancels*. The window
was chosen as the common-mode case (both exponents fitted on the same regions)
and the floor as the differential one (`n_C` is gradient energy and never
touches the kernel). Neither test ran. `n_I` barely responds to the window, and
`n_C` cannot respond to the floor **at all** — so in the floor sweeps the gap's
spread equals `n_I`'s to fourteen decimal places, which is not a result but a
subtraction. *A difference whose parts do not both move is not protected by
being a difference; it inherits the fragile part exactly.*

That is a real limit on a rule this programme has been leaning on, and the
sharpened version is: **prefer differences over quantities that share the
convention you are worried about — and check that both of them actually respond
to it.** The second clause was implicit and is doing more work than the first.
It also indicts the bar: "the difference moved less than the tolerance" passes
both when cancellation happens and when nothing moved, and five audits scored
that bar without distinguishing them. `cancellation()` now separates the cases,
and applied here it reports **not exercised** on all three sweeps rather than a
pass on all three.

**[measured — the bar re-applied to the earlier audits, and it acquits most of
them]** If the bar was blind, the six claims scored against it needed
re-checking, so `n3_robustness_retrospective` re-scores the published sweeps
with instruments that ask *how close did this come to breaking, in units of how
much the sweep moved things*. Three claim-shapes, three diagnostics
(`project_genesis.robustness`). The result was largely **not** what was
predicted — `1/3`:

| claim | shape | headroom | verdict |
|---|---|---|---|
| §2 gap, fitting window | difference | ratio `1.77` | not exercised |
| §2 gap, noise floor (both sweeps) | difference | `n_C` pinned | not exercised |
| §2/§3 `P = 3`, 2-D | ranking | `2.79×` | **exercised — held** |
| §2/§3 `P = 3`, 3-D | ranking | `0.68×` | **exercised — held** |
| §5 eviction, ordered point ahead | boolean | `0.24×` | **exercised — held** |
| §5 eviction, order costs capacity | boolean | `27–50×` | categorical, see §5 |

So the blind spot is real but **local to §2's gap**. The `P = 3` cliff was
predicted to be *structural* — rivals identically zero, hence unflippable, hence
a categorical claim filed as a robustness one. That is wrong. The rivals are
live: as the probe widens, `P = 4` closes on `P = 3` from `45.5×` to `16.6×` in
the plane, and harder in space. The ranking survived a convention that genuinely
threatened it, which is the strong version of the result and was already the
honest reading. Same for the eviction condition on the oscillator substrate.

*One methodological correction, found by checking rather than by reasoning.*
The first run of that experiment measured leads **additively** and reported the
wrong rival. On densities spanning orders of magnitude the smallest *difference*
sits at the tightest probe — where the rivals are exactly zero and the winner is
merely small — so it reads as a near miss when the winner is in fact infinitely
ahead. Scored as a ratio, the rival that is actually closing is a different one.
The published analysis quotes a margin for exactly this reason; the diagnostic
now matches it.

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

**[measured — and this one got *stronger* under audit]** The selection has one
free choice: the neighbourhood the junction test looks at, hardcoded to the
immediate `3^d` ring. Unlike every other constant this programme has swept, it
has an argument behind it — on a lattice it is the smallest window a junction
can appear in. `n3_junction_scale` widened it anyway, on fields the texture
guard confirms are tilings (domain scale `4.07–9.79`):

| neighbourhood | P=2 | P=3 | P=4 | P=5 | P=6 | margin |
|---|---|---|---|---|---|---|
| radius 1 (published, 3×3) | 0 | **0.0069** | 0 | 0 | 0 | ∞ |
| radius 2 (5×5) | 0 | **0.0293** | 0.0006 | 0 | 0 | 45.5× |
| radius 3 (7×7) | 0 | **0.0684** | 0.0041 | 0.0007 | 0 | 16.6× |

The registered prediction was that the margin would *collapse* once the probe
stopped being vertex-sized — a 5×5 window can geometrically hold four colours.
**It does not.** At radius 2 every rival is still exactly zero, and at radius 3
— a window wide enough to be measuring a region rather than a vertex — `P = 3`
still leads by `16.6×`. The field simply never places four colours within a
domain width of one another. That is a stronger result than the vertex-scale
mechanism proposed for it, and it is the **only** claim in this programme to
come out of a convention sweep wider than it went in.

And the ranking is what carries: `argmax = P = 3` at *every* radius while the
margin moves across `∞ → 45.5 → 16.6`. The magnitude is a location and it
drifts; the ranking is a comparison and it does not — the same split found in
every audit of §5, now on the claim the document rests on. (This case meets the
precondition §2 added afterwards, and has since been measured against it:
widening the probe moves the winning density *and* the runner-up, and the
ranking survived with a headroom of `2.79×` in the plane and `0.68×` in space —
`P = 4` genuinely closing, not a rival pinned at zero. The survival is a
measurement, not an identity. Not every case is.)

**Matter is the cells of a tessellation.** **[measured]** Held near its critical
point, the field partitions space into domains, walls, and junctions — a
countable inventory with exact invariants. Where walls meet in threes, Euler's
formula fixes the proportions at `2:3:1`. The number of families is `d+1` — three
in two dimensions, **four in three** — so *the number of generations is the
dimension of space plus one*. The abundances are topologically protected, not
energetically tuned.

*`d+1` was asserted until now, and the instrument could not have shown it.*
**[measured]** `n3_junction_scale` ran the selection measure in 3-D for the
first time and it picks `P = 3`, not `4`. The cause is a constant, not the
claim: `full_palette_junction_density` hardcodes its junction test to
`distinct >= 3`, which is the **2-D** vertex number. By codimension counting
`m` sectors meeting form a codimension-`(m−1)` object, so three sectors meet at
a *point* in the plane but along a *line* in space, and a point junction in `d`
dimensions needs `d+1` of them. Left at `3`, the measure asks in 3-D whether an
**edge** carries the whole palette — which a 3-palette does trivially.

Reading it at the geometrically correct `distinct >= d+1`, on fields the
texture guard confirms are genuine tilings:

| junction test | P=2 | P=3 | P=4 | P=5 | P=6 | winner |
|---|---|---|---|---|---|---|
| `>= 3` (published) | 0 | **0.0290** | 0.0021 | 0.0002 | 0 | `P=3` |
| `>= 4` (`d+1`) | 0 | **0** | **0.0021** | 0.0002 | 0 | **`P=4`** |

`P = 3` falls to *identically* zero, because a three-colour palette cannot form
a four-fold vertex at all. So `d+1` is right and is now **measured rather than
asserted** — but the published instrument is 2-D-specific and cannot express it
outside the plane. That is a bug in the measure, not in the geometry, and it
had never been caught because the measure had never been run in 3-D.

**[measured — `d+1` now holds in four dimensions]** The same test in `d = 1`
selects `P = 2` and in `d = 4` selects `P = 5`, rivals identically zero in both.
The `d = 4` arm also shows what a four-dimensional tiling is *made of*: all four
codimension tiers are occupied — 3-D walls at `0.314`, 2-D surfaces at `0.096`,
1-D lines at `0.0115`, point vertices at `0.00028` — with the falloff per tier
**accelerating** (`1.8× → 3.3× → 8.3× → 41×`).

Raw vertex density drops `330×` from `d = 1` to `d = 4`, but almost all of that
is geometry rather than instability: points are a vanishing fraction of a
`d`-volume, and dividing out the expected `(3/L)^d` leaves a residual decline of
about `3×` that is not even monotone. **Higher dimensions support this structure
perfectly well; they simply hold less of it per unit volume.**

The `d = 4` arm is the weakest measured here — the box holds ~1.5 domains per
axis against 3.7 in the plane — and `d = 5` is out of reach by this route, since
its point tier would sit near `1e-6`. Full treatment, standing free of this
framework, in `Docs/Junction_Selection.md`.

*A second dimension-blind constant surfaced in the same measure while reaching
`d = 4`: the texture guard's floor of `2.5`, tuned to a 2-D domain. The wall
ring grows as `3^d`, so the same raw score means a smaller domain in higher
dimensions — 2-D at `5.14` and 4-D at `2.38` hold domains of `19.5` and `15.7`
lattice units. `domain_diameter` reports the width instead, and noise then reads
`1.0` in every dimension. A measure written in one dimension will assume that
dimension twice over before anyone notices.*

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
  **[amended, then resolved — the ceiling is gone]** `evicted` was
  `isfinite(c_evict)`, with `c_evict` found on a search ladder running to
  `c = 1e5` — `2000×` past the `c_max = 50` the same run declares as its
  scarcity range. Two ceilings, nothing in the model choosing between them, and
  *the answer differed across the range*: against the ladder all `16` arms
  passed, against the declared budget the Hopfield arm at `threshold = 0.50`
  did not, needing `c = 59.7`.

  **The resolution is not to pick one, because `c` is the wrong variable.**
  §5's own reduction is that the load scale is arbitrary and cancels in the
  capacity form; the verdict simply had not been carried through it. Eviction
  requires `floor < κ_o⋆` with `floor = 1/(1 + u·L_o)`, and as `u` grows the
  floor falls to zero *for any non-zero load*. Taking that limit removes the
  ceiling from the statement:

  > **evictable ⟺ the ordered point starts ahead, and holding it costs
  > capacity** — `crossing_exists(w)` and `L_o > 0`.

  No consumption bound appears anywhere in it. This is the sentence §5 has
  been making all along — *scarcity evicts order exactly when maintaining
  order costs capacity* — with the last free constant taken out.

  **[measured]** It reproduces the published result and keeps the control:
  `16/16` driven arms evictable, `0/16` undriven, agreeing with the ladder
  verdict on all `16`. The Hopfield "failure" was an artefact of comparing
  `c_evict` against `c_max`; the structural claim was never about `c`. What
  the boolean *was* hiding is that the condition has two halves behaving
  differently — **the ordered point starting ahead is a genuine measurement
  that nearly fails** (headroom `0.24×` on the oscillators, where the ΔC
  convention moves the margin by four times its distance to zero), while
  **holding order costing capacity is categorical** (driven and undriven loads
  `7–9` decades apart — a difference in kind, not a near miss). `16/16`
  averaged those together and reported neither.

  *One constant remains and it is a different animal.* `L_o > 0` is the right
  condition and the wrong test: an undriven oscillator scan measures
  `L_o = 1.24e-15`, machine noise around a physically zero load, and a literal
  `> 0` passes it — breaking the control on `9/16` arms, which is how the guard
  came to exist. The comparison is made on the dimensionless ratio `L_o/L_⋆`
  against a numerical zero. Measured loads separate into `0` or `1e-15` when
  free and `1e-2 … 5.8e-1` when driven: **thirteen orders of empty space**, and
  every threshold from `1e-12` to `1e-3` gives the same verdict on every arm.
  A constant with ten orders of slack that no result depends on is not the
  same object as one with three orders that decides the outcome.

  The margins are now stored beside the verdict, which they were not: the
  original kept the conclusion and discarded the distance to it, so the claim
  could not be re-checked even in principle.
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

**[measured, a negative — and it may be a constraint rather than a gap]**
`causal_state.py` and `n3_causal_delta_c` attempt exactly that, via the
causal-state construction: two states are the same state when the dynamics
cannot tell their futures apart, with "cannot tell apart" calibrated against
the spread of a *single* state's own replicates. Nothing is thresholded; the
resolution comes from the system's own stochasticity.

It is a correct instrument. Against ground truth — `k` genuinely distinct
attractors — it recovers `k` exactly for `k = 1, 2, 3, 5`, and **the count does
not move with the noise amplitude** (`0.1 → 2.0`), which is precisely the
failure mode that contaminated the oscillator load reading.

It is also **worse than what it replaces**, on the same sweep design:

| across the free choices | threshold ΔC | causal ΔC |
|---|---|---|
| hull sizes | `{2, 3}` | `{2, 3, 4, 5, 6, 7}` |
| arms with the ΔC peak in the critical window | 5/7 | 5/31 |

And the reason is structural, not a tuning failure. In 17 of 31 Hopfield arms
the causal ΔC peaks at the **cold end**: in the ordered phase the dynamics are
near-deterministic, so distinct microstates stay distinct and every sample
resolves as its own causal state. **Causal distinguishability is maximal
exactly where URP needs ΔC to be minimal.**

Nor is there a horizon that escapes it. There are three regimes and none of
them is usable:

| horizon | what the reading does | scan at ceiling |
|---|---|---|
| short (`τ = 1` relaxation) | no shape at all | **100%** |
| intermediate (`τ = 5`) | "peak" is a tie-break among ceiling points | **62–69%** |
| long (`τ = 20–40`) | peaks at the cold end, or the shape inverts | — |

The intermediate case is the one most likely to be mistaken for success. On the
oscillators at `τ = 5` all nine arms report the peak inside the critical
window, which reads as a clean pass. But the measured curve is
`ΔC = 0.355, 0.595, 1.000, 1.000, 1.000, 1.000, 1.000` across `γ` — every one
of the twenty sampled states resolving as a distinct causal state from
`γ = 0.87` up — so the argmax is picking whichever tied ceiling point comes
first. `n3_causal_delta_c` now reports the saturated fraction per arm and marks
these `TIE` instead of counting them as landing in the window. With the
diagnostic in place **all 18 completed oscillator arms are `TIE`**: not one of
them carries a usable peak, where before the diagnostic nine of them looked
like successes.

Any construction of the form *"states differ if their futures differ"* inherits
this, because determinism is what makes futures differ reliably. Recovering the
quantity URP actually wants — diversity of *structured* states — would require
counting distinguishable **macro**states, which needs a coarse-graining, which
is the chosen cut this was meant to eliminate. The loop closes.

So the honest reading is stronger than "this attempt failed": distinction
requires committing to *which differences count as differences*, and the
dynamics alone will not supply that commitment. On present evidence the
arbitrariness in ΔC looks **irreducible** — a constraint on the principle as
formulated, not a defect in three experiments. What is unaffected is the
eviction *condition*, which survived every convention sweep in `16/16` arms;
what is blocked is anything depending on ΔC's shape.

**[measured — the gap has a coordinate]** If the choice cannot be removed, the
next question is whether it is *structured*: do the admissible ΔC conventions
move the reading along one axis, or in many independent directions?
`n3_convention_manifold` tests this, and it is testable because the convention
changes **only** ΔC — ΔI and the load come off the same trajectories — so a
family of conventions gives a family of curves over one fixed scan with
everything else identical. `2/3` held:

- **On the oscillators the consequence is invariant.** Across all nine
  combinations of two unrelated knobs — entrainment tolerance and frequency bin
  width — the ΔC peak spans `0.192`, exactly one scan grid step. Two independent
  conventions do not move it at all.
- **On the network it is monotone in the cut.** The peak slides
  `T⋆ = 1.10 → 0.70` as the threshold tightens, and the leading curve-shape axis
  is perfectly monotone in the knob (`Spearman = +1.000`).
- **The registered bar was `r² ≥ 0.80` and the network gave `0.756`**, so Q1 is
  recorded as failed. But the second axis adds `+0.014`, and by rank rather than
  linear fit the shape axis and the knob are the *same* predictor of the peak —
  `Spearman = −0.906` for both. `r²` is a linear statistic and `argmax` is not a
  linear functional of a grid-quantised curve, so what failed is the estimator.
  Registered as a failure regardless.

So the freedom in ΔC is not an open set. It is **one interpretable knob** —
resolution — with two independent conventions on one substrate collapsing onto
it and the second direction adding nothing on either. That does not repair
anything above: ΔC still has no derived value and the retraction stands. What
it changes is the *kind* of object the gap is. A choice with a coordinate can be
stated, compared between observers, and quotiented; a choice without one cannot.

**[measured — the same audit, aimed at the one agent-facing instrument]**
`trajectory_label` sorts a step in `(ΔC, ΔI)` into `expanding` /
`consolidating` / `diverging` / `contracting` / `steady`, and the identical
taxonomy is read on LLM session traces in the sibling repo. Both transplants use
it to report a `diverging` band — "the compass's hallucination trajectory" — and
place the capacity-starved optimum inside it. It takes a `deadband`, published
at `0.01`, which nothing derives. `n3_label_stability` sweeps it. `2/3` held:

- **The labels move.** Only `27%` (network) and `17%` (oscillators) of steps
  keep the same label across the sweep, so the constant is a parameter of the
  reading, not a formality.
- **But monotonically.** Widening the deadband only ever converts `diverging`
  steps into `steady` ones — never the reverse — so it behaves as a sensitivity
  dial. A flag raised at a wide deadband is a fortiori raised at a narrow one.
- **And `0.01` is knife-edge on one substrate.** It survives `20×` downward on
  both, but only `5×` up on the network and **`1.0×` up on the oscillators** —
  the very next setting moves the band.

The useful part is that **presence and extent come apart**. The `diverging`
regime is present at *every* deadband tested on both substrates, while its
**onset** takes three distinct values on the oscillators. So the two questions
one would actually ask of a running agent have different answers: *is this
trajectory diverging?* is robust; *when did it start?* is not. Reporting the
flag is defensible; reporting where it began requires reporting the deadband
with it.

This is the third instrument in a row where a confident-looking reading turned
out to rest on an underived constant, and the second where the failure was the
*statistic* rather than the phenomenon. The transferable discipline — sweep the
convention, and separate the claim that survives from the one that does not — is
at this point better established than any particular substrate result.

**[measured — the complete oscillator grid, and it is worse than "it never
works"]** All `36` oscillator arms are now in. With `γ_c ≈ 0.94` and a critical
window of `[0.61, 1.26]`:

| `τ` (relaxation times) | ΔC peak across the 9 arms | scan at ceiling | usable |
|---|---|---|---|
| 1 | `0.10` | 100% | 0/9 |
| 5 | `0.87, 1.06` | 62–69% | 0/9 |
| 20 | `1.44, 1.63, 1.82, 2.02, 2.21, 2.40` | 0–46% | 0/9 |
| 40 | `0.67, 0.87` | **0%** | **9/9** |

*An earlier revision of this section, written from the first three blocks,
claimed the peak "slides monotonically from the ordered end to the disordered
end as the horizon grows". **That is false** — `τ = 40` brings it back to
`0.67–0.87`. The dependence is non-monotone: out to the disordered end and
back.*

At `τ = 40` the construction **works**: the reading is unsaturated, and it puts
the peak inside the critical window on all nine arms, robustly across replicate
count and merge confidence. Taken alone that would read as a success.

It is not one, because of the row above it. At `τ = 20` with `R = 8` the
reading is **also** completely unsaturated — `0%` at ceiling, no ties, no
internal sign of trouble — and it puts the peak at `2.02–2.40`, far outside the
window. Two horizons, both giving clean, internally consistent, confident
readings; one right and one wrong; **and nothing available inside the
measurement distinguishes them.** The saturation diagnostic catches `τ = 1` and
`τ = 5` and is silent on exactly the case that matters.

So the honest statement is not "a dictionary-free ΔC cannot find criticality" —
at a long enough horizon it does. It is that **the horizon at which it is right
cannot be identified without already knowing the answer**, which is the same
thing as not having an instrument. `9` of `36` arms are usable and only in
hindsight. (Q3 is unaffected either way: the hull still ranges over `{3, 4, 5}`
at `τ = 40`, so the route stays indeterminate.)

This is the fourth instance in this arc of the same failure mode and the most
subtle: not a statistic reporting on data that cannot support it, but a clean
measurement that is simply wrong with no flag available.

*Scope.* The Hopfield sweep covers `31` of `36` arms and every value of each
free choice; the missing five are its most expensive corner. The oscillator
sweep is complete at `36/36`.

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
python experiments/n3_junction_scale.py        # §2/§3, at what scale is 3 special?
python experiments/n3_junction_scale.py --with-4d   # adds d=4 (~60 min)
#   -> Docs/Junction_Selection.md is this result standing alone, framework-free
python experiments/n3_capacity_gating.py       # §3, the correction (geometry, not scarcity)
python experiments/n3_area_law.py              # §2, the scaling dimension
python experiments/n3_gap_conventions.py       # §2, which conventions the gap survives
python experiments/n3_exchange_statistics.py   # §3, the exchange sign
python experiments/n3_ab_statistics.py         # §3, spin = statistics = gauge
python experiments/n3_kappa_gravity.py         # §4, gravity
python experiments/n3_boundary_gravity.py      # §4, the negative
python experiments/n3_criticality_transplant.py  # §5, substrate independence
python experiments/n3_kuramoto_transplant.py     # §5, the third substrate
python experiments/n3_crossing_prediction.py     # §5, the units audit of both
python experiments/n3_kuramoto_repair.py         # §5, the drive-free load
python experiments/n3_anatomy.py                 # §5, is the destination real?
python experiments/n3_causal_delta_c.py          # §5, a DC with no cut (negative)
python experiments/n3_convention_manifold.py     # §5, does the gap have a coordinate?
python experiments/n3_label_stability.py         # §5, the compass's own deadband
python experiments/n3_robustness_retrospective.py --from RESULTS  # all, was any of it tested?
```

Each prints its own pre-registered predictions, its verdict, and its honest
scope — including when the prediction failed. 1255 test functions in `tests/`
lock the central claims so they cannot drift silently. (That number was `872`
from the day this file was written until it was counted again — a stale figure
in a document about auditing stale figures.)

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
