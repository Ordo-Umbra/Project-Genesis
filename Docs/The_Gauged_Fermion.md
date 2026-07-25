# The Gauged Fermion

### Spin, statistics, and the gauge field — one sign, built rung by rung

*A closing synthesis of the fermion arc. `The_Woven_Forms.md` reports Act III
as a whole — the field's matter, its dimensional families, its pairs, its
gravity, its spin. This document lifts out one thread of it and reports it as
finished work: the eight measurements that carried a fermion from a
topological squiggle in a director field to a gauged particle whose exchange
antisymmetry is an Aharonov–Bohm holonomy — and states, precisely, the one
step that remains.*

---

## The question, in one sentence

A quark is not a lump. It is a **half-integer spinor**: rotate it by `2π` and it
comes back multiplied by `−1`; only `4π` returns it to itself. It is
**antisymmetric**: exchange two identical ones and the configuration picks up
`−1`. And those two facts are the *same* fact — the spin–statistics connection,
`exchange = (−1)^{2s} = rotation`.

The programme's matter sector had integer spin: a `U(1)` complex order
parameter, whose defects are vortices, whose angular momentum is an integer
ladder. Every object it made was, in the technical sense, a **boson**. The
question this arc asks is whether a capacity-driven recursive field can make
the other kind of thing — and whether "the other kind of thing" can be built
out of measurements rather than asserted.

The answer is yes, at the classical field-theoretic level, in eight rungs. The
arc's shape is its own best summary: **every rung's honest boundary named the
next rung.**

---

## Movement I — The spin: a defect that remembers `4π`

*`nematic_spinor.py`, `n3_spinor.py` (3/3) · the ½-disclination*

Half-integer spin is not a value you dial; it is a property of the
*order-parameter space*. A vector field cannot wind by half a turn. A
**nematic** — a headless director `n̂ ≡ −n̂`, order-parameter space `RP¹` — can:
its elementary defects are **±½ disclinations**, around which the director
winds by `π`.

Carried by `ψ = e^{2iθ}` (the doubled angle), a `ψ`-vortex of charge `q` is a
disclination of strength `s = q/2`, so the whole vortex machinery applies while
the *observable* object is double-valued. Three things measured:

- **The elementary defect is half-integer.** `s(q) = q/2` exactly; `q = ±1` is a
  `±½` disclination — the director winds by `±π`, impossible for a vector.
- **The double cover is explicit.** Transport the *oriented* director once
  (`2π`) around a `½` disclination and it **flips** (`n̂·n̂ = −1` — the spinor's
  minus sign); a second loop (`4π`) restores it (`+1`). An integer defect never
  flips.
- **It is a real charge on matter.** Pinned to a `κ` well and co-evolved with
  **no re-imprinting**, the `½` keeps its strength *and* its `−1` holonomy; two
  `½`'s **fuse** to an integer, a `½` and a `−½` **annihilate** and the field
  heals.

**Holds.** The `SU(2)` double cover of `SO(3)` — the topological content of
"spin-½" — is realised on the field, and matter can carry it.

---

## Movement II — The composites, and why the count is three

*`n3_hadron_spin.py` (3/3), `n3_junction_fermion.py` (3/3) · hadrons and `N⋆`*

Spin adds. Put `n` half-integer constituents at the derived exclusion floor,
bind them with `κ`-gravity, spin them with the driven chiral field, and the
composite carries total `s = n/2` — with the far-field double-cover class
**alternating by count**: `n = 1, 3` are **fermionic** (quark-like,
baryon-like); `n = 2, 4` are **bosonic** (meson-like). Hadron statistics from
nothing but additivity, and the half-integer content visible only *inside* (the
radial profile jumps to `n/2` across the constituent shell and is constant on
every larger loop). The 2-constituent composite exists as a *real* molecule —
bound, spinning, integer and no-flip from outside.

That result named its own gap: **nothing selected the constituent count.** The
answer is exact, and it is the programme's own `N⋆ = 3`. A spin defect's `2π`
phase winding must pass through **every** phase sector once — so with three
sectors the elementary spin-½ defect *is* a trivalent, three-sector junction:
the colour singlet. Measured on **emergent** fields (grown from noise, never
imprinted): every defect sits on a junction where all three sectors meet and
every junction on a defect, bijectively, each carrying the singlet `1:1:1`
composition and reading `s = ±½` with the `−1` holonomy — while two-sector
*walls* carry no spin. And three is selected *uniquely*: the defect forces
sector-valence `= P` (measured `P = 3, 4, 5`), while the matter tessellation
caps its generic junctions at the Plateau valence `d + 1 = 3` regardless
(`3.00 / 3.01 / 3.02`). **The two structures coincide only at `P = 3`.**

**Holds.** The baryon's "3" is the sector count — a fundamental bleeding through
into a higher manifestation. And the named gap moves on: the composites' spin
was topological, with *"no anticommutation, no Pauli principle between identical
composites; that frontier stands."*

---

## Movement III — The statistics: the exchange sign

*`spin_statistics.py`, `n3_exchange_statistics.py` (3/3) · the braid*

That frontier is the **statistics** half of spin–statistics, and it is a
different measurement from the spin: not how a defect behaves when *rotated*,
but what happens when two identical ones are **exchanged**. For classical
order-parameter defects the answer is the **Finkelstein–Rubinstein**
construction — exchange is homotopic to a `2π` rotation of the pair's frame, so
the two carry the same holonomy.

The instrument is a **braid**: two defects rotated rigidly about their midpoint,
a half-turn swapping them. Because a half-turn advances each defect's relative
angle by `π`, the phase at the braid **centre** winds by `q·π` per exchange —
so the oriented director's exchange sign is `(−1)^q = (−1)^{2s}`.

- **E1 — The exchange sign is `(−1)^{2s}`.** `q = 1` (a `½`) reads **`−1`** —
  *antisymmetric, fermionic*; `q = 2` (`s = 1`) reads `+1` — bosonic; `q = 3`
  (`s = 3/2`) reads `−1`. Independent of braid direction.
- **E2 — Exchange = rotation.** That sign *equals* the single-defect `2π`
  self-rotation holonomy for every charge. **The connection, measured on one
  field, not assumed.**
- **E3 — It is the *exchange* that carries it.** A **double** braid reads `+1`
  (two exchanges = a boson); the **far field** winds `0` (it sees only total
  charge — that is *fusion*, `½+½=1`, not exchange); a **no-swap** loop reads
  `+1` (motion is not exchange); a non-identical `(½, −½)` pair reads `+1` (the
  `−1` needs two *identical* `½`'s).

**Holds.** Two identical topological ½-spinors are antisymmetric under exchange,
and that sign is their rotation sign. Boundary named: the braid is **kinematic**
— the field re-imprinted at every step, the winding *imposed*.

---

## Movement IV — From kinematics to dynamics, and the pinning limit

*`n3_dynamical_braid.py` (3/3), `n3_phase_pinned_braid.py` (3/3) · co-evolution*

Is the exchange sign an artefact of imprinting, or does a defect that
**co-evolves** under the `κ`-detuned CGL carry its winding through a braid?
Three measurements, and the middle one is a genuine piece of physics:

- **D2 — Static pinning is robust.** A `½` pinned to a well and co-evolved 2000
  steps with no re-imprint keeps its winding (`−1 → −1`) *and* its double-cover
  holonomy (`−1 → −1`). The topological charge is dynamically conserved.
- **D1 — Transport is adiabatic, with a speed limit.** Dragged by a *moving*
  well, the winding follows at low speed (`|w| = 1` at `v ≈ 0.03`) and is
  **lost** above a threshold (`|w| → 0` by `v ≈ 0.1`). The mechanism: **a `κ`
  well pins *amplitude*, not phase.** The amplitude hole follows instantly; the
  winding must migrate diffusively, and if it cannot keep up the vacated core
  heals.
- **D3 — The two-body braid, partway.** Sign `−0.77`, cores surviving `92%` of
  the braid — the fermionic sign emerging from co-evolution, but not cleanly to
  completion.

D3's boundary *is* the amplitude-vs-phase gap, and it named its own cure:
**phase-aware pinning**, a term that anchors the winding rather than the
amplitude. Added as a weak, local, gauge-like restoring force (the field still
co-evolving between applications — a finite-rate force, not a per-step reset),
it closes the boundary: the braid **completes**, sign `−0.99` at **100%**
survival, above a modest threshold `η⋆ ≈ 0.2`.

And the control is what makes that honest rather than circular: under the
*same* pin, an **integer** pair braids to `+1` and a **double** `½` braid to
`+1`. **The pin transports whatever winding is present; the sign still comes
from the braid's geometry, not the pin.**

**Holds.** The exchange antisymmetry is carried by co-evolution. Boundary named:
the phase anchor is a *background template*, not a gauge field solved
self-consistently.

---

## Movement V — The gauge field: the vortex becomes a particle

*`gauged_vortex.py`, `n3_gauge_field.py` (3/3) · abelian Higgs*

A background template is a stand-in for the thing a real charged particle
carries. The real thing is a **`U(1)` gauge field** — the textbook **abelian
Higgs / Ginzburg–Landau** model, a lattice superconductor, whose vortex is the
Abrikosov / Nielsen–Olesen flux tube. Scalar `ψ` on sites, link phases
`θ_μ` as the connection, and

    E = Σ_μ |ψ − U_μ ψ(+μ)|²  +  (β/2) Σ B²  +  (λ/4) Σ (|ψ|² − 1)²

gradient-flowed in **both** fields to the self-consistent solution.

- **G1 — Flux quantization.** The winding forces the gauge field to carry a
  **quantised** magnetic flux `Φ = 2π·q` (`q = 1, 2, 3` → `−1.00, −1.99, −2.99`
  quanta) — *solved by the dynamics, not imposed* — and **zero** with the gauge
  field frozen off. The gauge field is the real anchor of the winding.
- **G2 — Finite energy.** The global vortex's logarithmically-divergent pair
  energy (`1 → 9 → 79 → 97` with separation) is **screened** into a
  finite-energy soliton (saturating at `≈ 23.8`) — the London length made
  visible.
- **G3 — Gauge invariance.** Energy and flux are invariant under a random
  *local* gauge transformation to **machine precision** (`ΔΦ ≈ 4×10⁻¹⁵`) — the
  signature of a genuine gauge theory, which a fixed template is not.

**Holds.** The vortex is a **gauged particle**: quantised flux, finite energy,
gauge-invariant observables.

---

## Movement VI — Statistics as a gauge holonomy

*`n3_ab_statistics.py` (3/3) · Aharonov–Bohm*

With a real gauge field in hand, the exchange sign can be asked a third,
deeper question: is it also a **dynamical gauge phase**? This is **Wilczek's
statistical transmutation** — a composite of charge and flux exchanges with a
phase set by the flux it drags, and one flux quantum turns a boson into a
**fermion**.

- **AB1 — The Aharonov–Bohm phase is the enclosed flux.** The Wilson-loop
  holonomy of the *self-consistent* field is `Φ = 2π·q`. A **unit** test charge
  encircling it gets `e^{iΦ} = +1` — **Dirac**: an integer flux quantum is
  invisible to an integer charge. A **half** charge gets `e^{iΦ/2} = (−1)^q` —
  the quantum is visible to a fractional charge *as a sign*.
- **AB2 — The flux–charge composite is a fermion.** Exchange is half a braid, so
  each charge sees half its partner's flux: the statistical phase is
  `θ = Φ/2 = π·q`, exchange sign `(−1)^q`. **One flux quantum makes a fermion**
  (`q = 1 → −1`); two make a boson (`q = 2 → +1`).
- **AB3 — The three faces agree.** For every `q`, the Aharonov–Bohm **gauge**
  phase, the **topological** braid exchange sign, and the **geometric** `2π`
  self-rotation holonomy read one and the same value.

**Holds.** The `−1` is a dynamical gauge phase, not only a topological
invariant.

---

## The one sign, three ways

The arc's result is not any single rung; it is their **convergence**. The same
number is computed three independent ways, from three different kinds of
mathematics, on the same field:

| | what is done to the object | the machinery | `q=1` | `q=2` | `q=3` |
|---|---|---|---|---|---|
| **spin** | rotate it by `2π` | director holonomy on `RP¹` | `−1` | `+1` | `−1` |
| **statistics** | exchange two of them | braid winding at the centre | `−1` | `+1` | `−1` |
| **gauge** | carry a charge around it | Wilson loop of the solved `A_μ` | `−1` | `+1` | `−1` |

Rotation, exchange, and holonomy are three faces of one `(−1)^{2s}`. That is
the spin–statistics connection — not quoted, but measured, on a field the
programme built for other reasons entirely.

And it is not the only place the fermion's character shows up as a *derived*
fact. The programme's exclusion floor — the parameter-free binding radius that
falls out of no-cloning against the capacity free energy — turns out to be what
makes **Pauli single-occupancy** possible at all: a *free* gas `½` carries only
soft `~1/r` repulsion and cannot be excluded from a well, while a **bound**
structure carries the hard, derived floor `s⋆` (`n3_single_occupancy`, 1/3, a
negative that closes a loop). *A fermion is single-occupancy matter when it is
bound and carries the no-cloning floor.* The statistics and the exclusion arrive
from opposite ends of the programme and meet.

---

## The honest boundary — stated precisely

What this arc has built is a **classical field-theoretic fermion**: a
topological spinor with the `4π` double cover, additive into hadron-like
composites whose count is selected by `N⋆ = 3`, antisymmetric under exchange,
carrying that antisymmetry through genuine co-evolution, gauged by a
self-consistent `U(1)` field with quantised flux, and exhibiting its statistics
as an Aharonov–Bohm holonomy.

What it is **not** is a *quantum* fermion. Precisely:

- **No Fock space, no `{ψ, ψ†}`.** There is no operator algebra and no
  many-body Pauli principle *between identical quanta*. The exclusion result
  above is a statement about bound classical structures, not about the
  antisymmetry of a many-body state vector.
- **No Dirac equation, no dynamical fermion field.** The spinor here is an
  order-parameter texture, not a spinor field with a first-order relativistic
  equation of motion.
- **The flux–charge binding is read off, not enforced.** The statistical phase
  is computed from the solved gauge field; no **Chern–Simons** term dynamically
  binds flux to charge.
- **2-D, relaxational, one operating point.** Gradient-flow dynamics, one
  lattice, stated `(λ, β)`; the 3-D rung is partial — a 3-D molecule carries a
  real *vector* spin-torque but is **overdamped** and cannot turn
  (`n3_molecule_3d`, 2/3, the standing subsidiary boundary).

**Second quantisation is the frontier**, and it is a genuine change of
framework rather than another rung on this ladder: an operator algebra is a
different kind of mathematics from a lattice field one can gradient-flow. The
arc stops here not because it ran out of ideas but because the next step is a
different instrument.

---

## What the arc's shape shows

Read as a sequence, the eight rungs have one recurring form: **each rung's
honest boundary is the next rung's specification.**

    the composites' spin is topological, nothing selects the count
        → N⋆ = 3 selects it (the defect IS the three-sector junction)
    the exchange sign is measured, but the braid is kinematic
        → co-evolve it: transport is adiabatic, with a speed limit
    the two-body braid completes only partway (κ pins amplitude, not phase)
        → phase-aware pinning completes it — and a control proves it honest
    the phase anchor is a background template, not a solved field
        → the self-consistent U(1) gauge field: quantised flux, finite energy
    a gauge field should make statistics dynamical
        → the Aharonov–Bohm phase: one flux quantum makes a fermion
    the statistics are a classical holonomy, not an operator algebra
        → second quantisation (the standing frontier)

Nothing here was reached by widening a claim until it fit. Each boundary was
measured, stated in the experiment's own output, and then attacked. The
negatives did as much work as the positives: D3's partial braid produced the
amplitude-vs-phase diagnosis; the single-occupancy failure produced the
bound-vs-free resolution; the honest controls (integer → `+1`, double braid →
`+1`, no-swap → `+1`) are what make the `−1` mean anything at all.

---

## The map

**Instruments** (`project_genesis/`).

| file | what it adds |
|---|---|
| `nematic_spinor.py` | the nematic director and its `±½` disclinations — `disclination_strength`, `director_holonomy` (the `4π` double cover), `plaquette_winding` |
| `spin_statistics.py` | the exchange sign — `braid_positions`, `exchange_holonomy`, `self_rotation_sign`; and the dynamical braid — `gaussian_wells`, `transport_defect`, `dynamical_braid` (with `phase_pin`) |
| `gauged_vortex.py` | the self-consistent `U(1)` gauge field (abelian Higgs) — `relax` (ψ **and** links), `plaquette_flux` / `local_flux`, `gauge_transform`; and `wilson_loop` / `ab_phase` |
| `vortex_chiral.py`, `vortex_chiral_3d.py` | the vortex machinery the spinor is lifted from (integer spin, 2-D and 3-D) |

**Experiments** (`experiments/`).

| file | verdict | result |
|---|---|---|
| `n3_spinor.py` | 3/3 | half-integer spin: `s = q/2`, the `4π` double cover, conserved & fused & bound to matter |
| `n3_hadron_spin.py` | 3/3 | composites: `s = n/2`, statistics alternating by count; the meson-analog a real bound molecule |
| `n3_junction_fermion.py` | 3/3 | `N⋆ = 3` selects the count — the elementary spin-½ defect **is** the trivalent three-sector junction |
| `n3_exchange_statistics.py` | 3/3 | the exchange sign `(−1)^{2s}`, equal to the `2π` rotation; controls isolate the single exchange of identicals |
| `n3_dynamical_braid.py` | 3/3 | co-evolved: static pinning robust, transport adiabatic **with a speed limit**, the two-body braid partway |
| `n3_phase_pinned_braid.py` | 3/3 | phase-anchoring completes the braid (`−0.99`, 100%); the control proves the pin transports, not fabricates |
| `n3_gauge_field.py` | 3/3 | the self-consistent gauge field: `Φ = 2π·q`, London screening, gauge invariance to machine precision |
| `n3_ab_statistics.py` | 3/3 | statistics as an Aharonov–Bohm holonomy; one flux quantum → a fermion; the three faces agree |

**Related.** `Deriving_The_Exclusion_Coefficient.md` (the parameter-free floor
the composites are spaced on), `n3_single_occupancy.py` (why Pauli
single-occupancy belongs to *bound* fermions), `n3_vortex_3d.py` /
`n3_molecule_3d.py` (spin as an axial vector in 3-D, and the overdamped
molecule).

**Tests.** `test_spinor.py`, `test_spin_statistics.py`,
`test_dynamical_braid.py`, `test_phase_pinned_braid.py`,
`test_gauged_vortex.py`, `test_ab_statistics.py` — every instrument and every
central claim, so the arc is a standing, checkable record.

---

*A director that cannot decide which way it points; a defect that needs `4π` to
come home; two of them that refuse to be swapped without a minus sign; a gauge
field that answers the winding with exactly one flux quantum — and a charge
carried around it that finds the same minus sign again. Rotation, exchange, and
holonomy: one number, three ways, measured. What is missing is the operator
algebra that would make it quantum — and that boundary, like every other in this
arc, is stated rather than blurred.*
