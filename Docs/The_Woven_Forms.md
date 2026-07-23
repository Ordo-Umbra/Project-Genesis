# The Woven Forms

### Act III — the field's own matter: dimensional forms, natural pairs, spin, and their gravity

*The third movement of the program.  `The_Measured_Bridge.md` reports Act I
(the generative gap carries the instanton number `κ`); `The_Emergent_Cosmos.md`
and `The_Complete_Arc.md` report Act II (the same `κ` is gravity, matter, and
cosmos).  This document sits above the work built after them: it states how a
single capacity/sector field, driven by the one functional, weaves a whole
**taxonomy of matter** — forms sorted by dimension, bound in pairs, given spin
by a chiral term, and made to gravitate at a finite speed — and it holds the
line between what is now **measured** and what remains **vision**.  Everything
marked "measured" is reproduced by a pre-registered experiment in this repo;
everything marked "frontier" is intuition the foundations now make testable but
that has not yet been given a verdict.*

---

## 1. The one thesis, continued

The program's single functional is

    S  =  ΔC  +  κ · ΔI

— a system grows by making **distinctions** (`ΔC`) and **integrating** them
(`ΔI`), traded at the capacity exchange rate `κ`.  Acts I and II followed that
`κ` from a number in the QCD vacuum to gravity and the cosmos.  This act asks a
different question of the same field: **what matter does it make?**  Not "does
the field gravitate" but "what are the *things* that gravitate — what forms does
the field settle into, how do they combine, and what quantum numbers do they
carry?"

The answer that has emerged is that the field, held near criticality, does not
make a featureless soup.  It **tessellates** — partitions space into domains
separated by walls meeting at junctions — and that tessellation is a complete,
countable inventory of forms, with its own conservation laws, its own pairing
rule, and (once one term is added) its own spin.  The working intuition that
drove this, stated plainly: *working backwards from the way quarks come in
generations and pairs, a recursive field near its critical point should produce
quark-like forms in dimensional families, pair them, and spin them — and the
same `S` that builds structure should fix their proportions.*  Act III is the
first stretch of that intuition turned into measured verdicts.

---

## 2. The forms are the cells of a tessellation — and their proportion is fixed by topology

*Measured — `experiments/n3_quark_generations.py` (3/3), `project_genesis/dimensional_forms.py`.*

The collab explorations classified the field's emergent structures into
**dimensional families**: 0D points, 1D lines, 2D blobs — a "quark generation
hierarchy" whose densities shifted with phase.  The rigorous reading is that
these families are the **cells of the sector tessellation's CW-complex**.  A
site's codimension is set by how many sectors meet in its neighbourhood:

    1 sector   → domain interior  → a 2-cell   (2D, "heavy")
    2 sectors  → a domain wall    → a 1-cell   (1D, "medium")
    ≥ 3 sectors → a junction      → a 0-cell   (0D, "light")

Count the connected components — `V` junctions, `E` wall arcs, `F` domains — and
two exact facts follow that mere morphology-counting never had:

- **Euler on the torus.**  A clean CW-decomposition obeys `V − E + F = 0`
  (verified exactly on synthetic tilings in `tests/test_dimensional_forms.py`).
- **The trivalent (N⋆=3) signature.**  Where walls meet in threes — the 120°
  Y-junctions the sector program selects (`Thermal_Sector_Program.md`) — every
  vertex has degree 3, so `2E = 3V`, and Euler fixes the hierarchy at

        V : E : F  =  2 : 3 : 1 .

  **The generation hierarchy is not a free density; it is the junction valence.**
  Measured in the confined phase: mean valence `3.00`, ratio `1.90 : 2.54 : 1`.

The same instrument makes deconfinement a topological statement.  Heating the
field shatters the clean tessellation into speckle, and the **normalised Euler
defect** `|V − E + F| / (V + E + F)` jumps from `0.066` (confined) to `0.71`
(hot): the census's own consistency is a deconfinement order parameter, and the
break is sharp — the clean tessellation exists only at the cold point.  This is
the same confined→deconfined axis Act II's Wilson-loop work walks
(`Monte_Carlo_Confinement.md`), now read off the geometry of the forms
themselves.

**And why *three*.**  *(Measured — `experiments/n3_form_abundances.py`, 3/3.)*
The abundance question — how many families, and which is rarest — has a
topological answer that declines the numerology.  Across sector counts
`P = 3, 4, 5, 6` exactly **three** families stay populated with valence ≈ 3 and
ratio near `2:3:1`: three domains generically meet at a point whatever the
palette, so the family count is `d + 1 = 3` in two dimensions.  *Three
generations because space is 2-D* — cells of dimension 0, 1, 2 — not because of
any tuning.  And the sharpest form of the claim is now **measured**
(`experiments/n3_3d_generations.py`, 3/3): a 3-D field carries **four**
families (vertices, triple-lines, faces, volumes) with the tetrahedral
**Plateau** valences `4/3/2/1` and Euler `V − E + F − C = 0` on the 3-torus —
and the count is `min(P, d + 1)`, the fourth (vertex) generation appearing
exactly when a fourth sector is available to meet at a point (P = 3 in 3-D is
one sector short, and carries only three).  *The number of generations is the
dimension of space plus one.*  The abundances are **topologically
protected** (the ratio's coefficient of variation is `0.057` as the field is run
from fine to coarse — fixed by the junctions, not by energy) and the **2D "heavy"
family is the rarest** at every point — the qualitative shape of ordinary matter,
where the heaviest generation is fewest.  What is *not* claimed, and is marked so
in the experiment itself, is any numerical match to real quark masses or
abundances; the result is the **structure** of the hierarchy, not its numbers.

**What this establishes:** the "three generations" are a *topological necessity*
of a trivalently-junctioned field, keyed to the same `N⋆ = 3` that runs through
the whole program.  The light/medium/heavy forms and the three colour sectors
are one fact seen twice — and *three* is the dimension of space plus one.

---

## 3. The forms come in natural pairs — and the pair has a derived binding radius

*Measured — the exclusion series (`Docs/Deriving_The_Exclusion_Coefficient.md`,
Parts I–V), `experiments/n3_exclusion_*.py`.*

The collab charge histograms carried the next piece: at the ordered point the
topological charge is **bimodal**, quantised to `±` values and avoiding zero.
Charges come in conjugate pairs and never as an isolated neutral — confinement,
stated in one histogram, and the seed of "natural pairs."

The program's own no-cloning principle gives the pair a binding law with **no
free parameter**.  The URP exclusion idea — *adding the same distinction does
not expand the structure* — becomes, against the capacity field's full free
energy, the gap

    E_x  =  2·E(ρ_dup)  −  E(2·ρ_dup) ,      ρ_dup = min(ρ₁, ρ₂) ,

the price of pricing a *cloned* component at the extensive cost its content
deserves rather than the concave bargain the field offers.  This gap is
**non-negative by a concavity theorem**, repulsive at every separation, and at
the operating point it buys an interior **floor** — a stable separation `s⋆`
where two overlapping forms sit, neither merging nor escaping.  A `±` pair held
at that floor is a **meson-like bound state**: the exclusion floor *is* the
confinement radius, derived from `r` and `c` alone (Part II, 3/3).

Two refinements complete the pairing story.  **Identity is selective**
(Part III): a labelled load prices only *true* clones, so gravity binds
everything while exclusion refuses the sharing discount only for same-type
overlap — and (`n3_identity_generation`, `n3_identity_invariance`) identity can
be **measured from a form's internal structure** rather than assigned, with a
pose-invariant sameness `φ` that routes the exclusion.  And the pairing extends:
the `n`-copy sector (Part IV) prices `n`-fold stacks by `nE(ρ) − E(nρ)`,
reducing to the pair form at `O(c²)`, with a trimer floor of its own.

**What this establishes:** the "natural pairs" are the generic bound state of the
matter sector — a charge-conjugate pair on a parameter-free exclusion floor —
and the framework now *derives* the binding rather than dialing it.

**And the forms carry flavour.**  *(Measured — `experiments/n3_flavour_structure.py`,
3/3.)*  Beyond its generation (§2), a form carries a second quantum number: its
**flavour**, the identity of the sectors that compose it — a domain is one
sector (a bare colour), a wall two (a colour–anticolour pair, meson-like), a
junction ``d+1`` (a colour-singlet triple, baryon-like).  The multiplet sizes
are **Pascal's triangle**: with ``P`` sectors, generation ℓ realises exactly
``C(P, d+1−ℓ)`` flavours (in 2-D: ``C(P,1)``, ``C(P,2)``, ``C(P,3)`` for
domains, walls, junctions — measured all realised, `P = 4`: 4/6/4, `P = 5`:
5/10/10).  Under the sector symmetry the multiplets are **democratic** (uniform,
normalised entropy > 0.93), and the flavour distribution is a **conserved**
label (stable as the field runs further).  So each form carries *two orthogonal
quantum numbers at once* — generation × flavour, the structural echo of a
quark's generation and its colour content.  A flavour *hierarchy* from breaking
the sector symmetry stays open (the coarsening dynamics is winner-take-all, so a
global bias dominates rather than grades), and the numerology — CKM, masses —
is declined, as everywhere in this act: the **structure** is what is measured.

---

## 4. The forms gravitate — at a finite speed, with a merger of their own

*Measured — the finite-speed arc, `project_genesis/capacity_waves.py`,
`experiments/n3_kappa_lightcone / _retarded_gravity / _quadrupole_line /
_plunge_ringdown / _kappa_molecule.py`.*

Act II's `κ`-gravity acted instantaneously (the field was relaxed adiabatically).
Giving `κ` a finite update latency `τ` — the telegrapher form
`τ·∂²ₜκ + ∂ₜκ = D∇²κ + …` — turns it into a genuine dynamical gravity with the
shape of the real thing:

- **A causal cone** at `c_κ = √(D/τ)` — *the field's update rate is its speed
  limit* (`n3_kappa_lightcone`, 2/3), and the same matter term that screens
  static `κ`-gravity gives the waves a **mass** `m² = r + c·ρ`, measured a fourth
  independent way in the dispersion.
- **Gravitational drag and inspiral** (`n3_retarded_gravity`, 3/3): a moving mass
  radiates, predictable from the static well alone; a bound pair loses energy and
  spirals — the analogue of gravitational-wave decay, with the adiabatic control
  conserving.
- **A quadrupole-led spectrum** (`n3_quadrupole_line`, 3/3): an equal-mass binary
  broadcasts only *even* harmonics of its orbital frequency — the analogue of
  general relativity's "no dipole radiation," an exact symmetry statement.
- **A merger that is a plunge, not an orbit** (`n3_plunge_ringdown`, 3/3): this
  gravity has *no long inspiral* — a moving well digs faster than the vacuum
  heals, carving a trench that squeezes the pair to contact on a geometric clock.

And where the last three sections meet — a `±` pair, held open by the derived
exclusion floor, made to gravitate at finite speed — the program's **first
persistent bound object** appears: a **κ-molecule** (`n3_kappa_molecule`, 2/3).
Its one failure is a finding: the achiral molecule **cannot spin** (rotation is
overdamped, `Q < ½`) — the medium is too viscous, and the scalar field carries
no angular momentum to hold.  Which is exactly the gap the next section fills.

---

## 5. The forms spin — a single parity-breaking term, carried by the 0D forms

*Measured — `project_genesis/chiral_field.py`, `experiments/n3_chiral_spin.py`
(3/3).*

The collab's "vorticity (spin analog)" histograms were **symmetric**: with no
parity-breaking term, left- and right-handed textures form in equal measure, and
there is no net spin.  The κ-molecule's failure to hold angular momentum was the
same absence seen dynamically.  Spin, in this program, *starts as a chiral term.*

Added minimally through the complex Ginzburg–Landau field,

    ∂_t ψ  =  ψ  +  (1 + iλ)·∇²ψ  −  (1 + iλ)·|ψ|²·ψ ,

`λ = 0` is real Ginzburg–Landau — parity-symmetric, the collab's case.  The
observable must be chosen against a topological trap: the field-mean vorticity
`⟨ω⟩` is **pinned to zero** on the torus (vortices and antivortices pair), so it
cannot see chirality.  The observable that can is the field's **intrinsic
precession** `Ω` — the rate its order parameter rotates — and the result is exact
and clean:

    Ω  =  −λ .

`λ = 0` gives `Ω = 0` (parity restored, the collab baseline); turning `λ` on
gives the field an intrinsic angular frequency whose sign is the handedness —
**spin as a term**, the internal rotation the scalar molecule lacked.  And the
census closes the loop: the spin density concentrates **83–107×** on the **0D
junctions** of the tessellation while the trivalent structure survives (valence
`3.00`).  *The point-like "light" forms carry the spin.*

**What this establishes:** the three ingredients of a particle — a place in a
dimensional family, a conjugate partner, and a spin — are now all present in one
field, and the spin lands precisely on the 0D forms the census counts.

**And the loop closes.**  *(Measured — `experiments/n3_spinning_molecule.py`,
3/3.)*  The κ-molecule of §4 could not spin (its `Z2` failure); the chiral term
is what it was missing.  Embedding the molecule's forms in the chirally-
precessing background — so the medium drags each toward co-rotation at
`Ω_bg = −λ` — gives the bound pair a **steady, persistent spin** where the achiral
molecule drained to rest (`Ω → 0`), with the pair still bound on its derived
exclusion floor (`sep ≈ s⋆`) and the spin's sign following the coupling's.  The
first bound object in the program **turns**: derived-exclusion binding +
finite-speed gravity + a chiral term assemble a persistent, spinning bound state
— a form with a place in the census, a conjugate partner on a parameter-free
floor, and a spin it can hold.

**And the drive becomes the field.**  *(Measured — `experiments/n3_two_field_chiral.py`,
3/3, `project_genesis/two_field.py`.)*  The spinning molecule's one caveat was
that its drive was imposed — a rigid rotation at a free parameter `Ω_bg`.  The
two-field instrument retires it: the complex chiral field ``ψ`` co-evolves on
the same lattice as the telegrapher ``κ``, coupled both ways.  The same ``κ``
wells that bind the pair **hole the chiral field** (detuning
``g = γ(1 − κ/κ₀)``; ``|ψ|`` drops to ``0.55`` at the matter while the bulk
stays ordered), and the field answers with a force of its own: the detuning
sources a static phase dip at each well, the chiral term shears it into a
phase current ``j = Im(ψ*∇ψ)``, and the pair feels a **bond-axis force exactly
odd in ``λ``** (``F_r = ∓6.3×10⁻²`` at ``λ = ±0.2``, the ``λ = 0`` residual
``85,000×`` smaller, the tangential component ``10⁴×`` below the radial) —
*the chiral field presses on the bond but cannot itself turn it*, because a
uniformly precessing bulk carries no mechanical current.  And the molecule's
spin is now slaved to the field's **own measured precession**: the circulation
drag runs at ``Ω_field`` read off the co-evolving ``ψ`` each step — no
``Ω_bg`` anywhere, ``λ`` the only chiral input, ``λ = 0`` the achiral molecule
with no special casing.  The pair turns at the rate and handedness it reads
off the medium it swims in.  (The rigid-rotation *flow profile* remained the
one modelling ansatz — converting internal precession into mechanical
circulation needs a vorticity-bearing field; the next rung, below.)

**And the circulation becomes real.**  *(Measured — `experiments/n3_vortex_chiral.py`,
3/3, `project_genesis/vortex_chiral.py`.)*  The two-field molecule named its own
last ansatz — the rigid-rotation *flow profile* — and named its cure: a uniformly
precessing bulk carries no mechanical current (`j = 0` there, which is exactly
why the two-field force was radial), so a real torque needs a field that carries
mechanical **angular momentum, i.e. vorticity**.  The vortex instrument supplies
it: each form carries a **vortex** pinned at its core,
``ψ = A(x)·exp(i·Σ q_k arg(x − x_k))``, an integer winding ``q_k`` with the
amplitude holed at the cores and by the ``κ`` wells (the same detuning).  Three
things now hold that the precessing bulk could not.  The field carries real
angular momentum, **quantised by the winding**: ``L = Σ (x − x_cm) × j`` is zero
at ``q = 0``, sign-locked to ``q``, antisymmetric, and climbs a monotone ladder
in ``|q|`` — an integer winding, not a tunable dial.  The phase-current force is
now a **torque**: two same-sign vortices give a force *tangential* to the bond
(``|F_t| ≥ 5|F_r|``, a net torque signed by ``q``), while a vortex–antivortex
pair *translates* with the torque collapsing — the point-vortex law, the
mechanical current the precessing bulk lacked.  And so the molecule **spins from
its own circulation with no imposed rotation at all**: bound by ``κ``-gravity and
the derived exclusion floor, torqued only by its own vortices, each nonzero ``q``
holds a persistent signed spin (``sign Ω = sign q``, same-sign vortices
co-rotate) while ``q = 0`` is the achiral molecule (Z2) that drains.  (The
remaining idealisation: the vortex is *pinned* to its form — imprinted each step,
its amplitude co-evolving with the wells — rather than self-sustained by the bare
CGL dynamics; a topological binding of a defect to matter that the ``κ`` wells
physically anchor.  A fully emergent, CGL-sustained vortex is the deeper version,
below.)

**And the vortex sustains itself — but the strong spin does not follow.**
*(Measured — `experiments/n3_vortex_chiral.py`… the emergent test
`experiments/n3_emergent_vortex.py`, 2/3, an honest boundary.)*  Remove the
re-imprinting entirely: seed the vortices once and let the field co-evolve under
the ``κ``-detuned CGL alone (`evolve_seeded_field`, or
`evolve_vortex_molecule(reimprint=False)`).  Two of the three claims survive, and
they are the deep ones.  The winding is a **dynamically-conserved topological
charge**: with no re-imprinting the integer enclosing each well is held for 2000
steps and stays integer from a *noisy* seed — quantisation the dynamics enforces,
not a value imposed each step (`winding_number` counts it tracker-free).  And the
sign structure is a real **selection rule**: like charges *survive and stay
pinned to their matter* (``|w| = 1`` held, the cores on the wells, and — released
as a free molecule — the winding around each *moving* mass stays ``±1``, the
self-sustained vortex tracking its form), while a vortex–antivortex pair
*annihilates* — unwinds to ``w = 0``, the amplitude healing, ``L → 0``.  What
does **not** survive is the *strong* spin: with the field self-sustained the
molecule's torque is sign-locked to the winding but an order of magnitude weaker
than the re-imprinted drive, and the field's ``L`` drains — so the pinned mode's
strong spin was, in part, the imprinting doing work.  A topological defect can
bind to matter and keep its quantised charge under its own dynamics — the honest
core of a *quantised, matter-bound* spin-analog; turning that bound charge into a
*strong orbital* torque without re-sharpening is the open problem — closed next.

**And the precession regenerates the spin — strongly.**  *(Measured —
`experiments/n3_driven_vortex.py`, 3/3.)*  That open problem had a clear
diagnosis: at ``λ = 0`` the self-sustained circulation *drains* for want of the
re-sharpening the re-imprinting supplied.  The cure is the field's **own
precession**: the CGL twist ``λ ≠ 0`` (precession ``Ω = −λ``) turns each vortex
into a spiral source that continuously renews its azimuthal current.  Keep
everything self-sustained — seed once, co-evolve, no re-imprinting, no imposed
rotation (`evolve_vortex_molecule(reimprint=False, chiral_lambda=λ)`) — and the
molecule now holds the **strong spin at the pinned strength and beyond**,
growing with ``λ`` (``|Ω| = 0.010, 0.019, 0.022`` at ``λ = 0.1, 0.2, 0.3``,
against the re-imprinted ``0.010`` and the ``λ = 0`` self-sustained ``0.002``),
the winding preserved and the pair bound.  And the handedness and the strength
**separate into two knobs**: ``+q`` and ``−q`` spin *oppositely* at the same
drive (the sign is the topological charge's, independent of the sign of ``λ``),
while the magnitude is the precession's (``|Ω| ∝ |λ|``).  Both ingredients are
necessary — a vortex without the precession is the weak, draining ``λ = 0``
spin; a precession without a vortex carries *no* torque at all (a winding-0
field's phase-current force is exactly zero, the two-field ``C2`` result of a
uniformly precessing bulk).  Only the two together spin it strongly.  So a
self-sustained field *does* hold a strong, quantised-handed spin: the vortex is
the spin the matter carries, the precession the drive that keeps it turning.
(The strong-spin state lives in a chirality-compatibility window — a twist of
the wrong sign can unwind the vortex; and the CGL is a driven, non-equilibrium
medium, so ``λ`` inputs the energy a persistent spin costs.)

**And in 3-D, spin acquires a direction.**  *(Measured —
`experiments/n3_vortex_3d.py`, 3/3, `project_genesis/vortex_chiral_3d.py`;
field-level.)*  Everything so far was 2-D, where spin lived on a *point* vortex
and its angular momentum was a scalar — a sign.  A complex field ``ψ: ℝ³ → ℂ``
vanishes generically on **lines** (codimension 2), so in 3-D the defect is a
vortex **line**, and the field's angular momentum ``L = Σ (x − c) × j`` is a
genuine **3-vector aligned with the line**: measured over a centred sphere it
points along the line for *every* orientation — the axes, the face- and
body-diagonals, a skew direction — to align ``1.000`` with a direction-
independent magnitude, and along a fixed axis it is a sign-locked, quantised
ladder in the winding (``L_z = 0, ±23000, ±46000`` at ``q = 0, ±1, ±2``).  Spin
now has a *direction* — rotate the line and its spin vector rotates with it.
Seeded once and co-evolved under the ``κ``-detuned CGL with no re-imprinting, the
line keeps both its integer winding *and* its axis (robust to a noisy seed, and
— because a line ∥ an axis wraps a torus cycle — more stable than the 2-D point
vortex, whose ``L`` drained); and matter enforces the same sign rule as in 2-D:
two like lines survive on their wells while a line–antiline pair annihilates and
the field heals.  Honest scope, and the reason this is a *rung* not the summit:
this ``U(1)`` field carries an **integer / axial-vector** angular momentum — a
spin-1-like, *bosonic* object.  A genuine **half-integer spinor** — the
``SU(2)`` double cover, ``4π`` periodicity, the fermion the quark actually is —
is a different field structure, and reaching it is the open frontier toward
*fermionic* matter, not what a complex order parameter gives.

**But the 3-D molecule cannot turn.**  *(Measured — `experiments/n3_molecule_3d.py`,
2/3, an honest negative.)*  Binding two such lines into a molecule (3-D
``κ``-gravity + the derived exclusion floor, each mass threaded by a line) asks
whether the bound object can *spin* now that its spin has a direction.  The
torque is genuinely there and genuinely a **vector**: for a floor-bound pair the
phase-current force is tangential to the bond, sign-locked to the winding, its
axis the line's direction (lines ∥ ``ẑ`` give ``τ ∥ ẑ``, lines ∥ ``x̂`` give
``τ ∥ x̂``, magnitude ``0.17``), the antivortex control null — the very mechanism
that spun the 2-D molecule.  And yet, released, the pair **twists to a static
few-degree angle and stops** (``|Ω| ≈ 0`` for both windings, an order below the
2-D driven bar): it is **overdamped** — the ``κ``-drag that binds it dissipates
the rotation faster than the drive builds it, the original κ-molecule's
``Q < ½`` failure returning now that the medium is 3-D and the line threads its
whole column.  The pair stays on its floor and keeps its winding and ``L``
vector throughout, so the form *carries* a spin vector — it simply cannot turn
it.  What overcame this drag in 2-D was a thinner medium; a 3-D molecule that
genuinely rotates needs a higher ``Q`` or a different binding, the open rung.

**And spin becomes half-integer — a spinor.**  *(Measured —
`experiments/n3_spinor.py`, 3/3, `project_genesis/nematic_spinor.py`.)*  Every
spin so far was **integer** — a ``U(1)`` complex order parameter, a boson.  A
quark is a **half-integer spinor**: a ``2π`` rotation gives ``−1``, only ``4π``
returns it — the ``SU(2)`` double cover of ``SO(3)``.  That is a *different*
order parameter: a **nematic**, a headless director ``n̂ ≡ −n̂`` (space
``RP¹``), whose defects are **±½ disclinations** — the director winds by only
``π``, which no vector can do.  Carried by ``ψ = e^{2iθ}`` (the doubled angle),
the director is ``θ = ½·arg ψ`` and a ``ψ``-vortex of charge ``q`` is a
disclination of strength ``s = q/2`` — so the whole vortex machinery applies but
the observable object is the double-valued director.  Three things, each
measured.  The elementary defect is **half-integer**: ``s(q) = q/2`` exactly, so
``q = ±1`` is a ``±½`` disclination (the director winds by ``±π``), impossible
for a vector.  The **double cover** is explicit: transport the oriented director
once (``2π``) around a ``½`` disclination and it **flips** (``n̂·n̂ = −1``, the
spinor's minus sign), a second loop (``4π``) restores it (``+1``) — while an
integer defect never flips.  And it is a real, conserved, additive charge on
matter: pinned to a ``κ`` well and co-evolved under the CGL with no
re-imprinting, the ``½`` keeps its strength *and* its ``−1`` holonomy; two
``½``'s **fuse** to an integer, a ``½`` and a ``−½`` **annihilate** and the field
heals.  Honest scope, and why this is a foundation not the finish: this is the
**order-parameter (topological)** realisation of half-integer spin — a genuine
``Z₂/RP¹`` double cover with ``π``-winding disclinations, the topological essence
of a spinor.  It is **not** the full quantum Dirac field — no spin–statistics
(anticommutation), no Dirac equation, no dynamical fermion — but the half-integer
winding, the ``4π`` double cover, and the fusion rules are the measured content
of "spin-½", and matter can now carry it.

**And the pieces assemble into hadron-like composites.**  *(Measured —
`experiments/n3_hadron_spin.py`, 3/3.)*  Put everything together — the derived
exclusion floor to space the constituents, ``κ``-gravity to bind them, the
driven chiral field to spin them, and the ``½`` disclination as their spin — and
ask what composite objects come out.  The answer is the **meson/baryon
statistics split, from nothing but additivity**: topological spin adds, so ``n``
half-integer constituents carry total ``s = n/2``, and the far-field
double-cover class alternates with the count — ``n = 1`` and ``3`` are
**fermionic** composites (quark-like, baryon-like: the ``2π`` loop flips the
director), ``n = 2`` and ``4`` are **bosonic** (meson-like: no flip) — exactly
the statistics real hadrons get from counting their quarks.  The half-integer
content is visible only *inside*: each constituent carries its own local ``½``
(and its ``−1``) at the seed, and after co-evolution the total charge stays
confined and conserved — the radial profile jumps to ``n/2`` across the
constituent shell and is constant on every larger loop.  And the meson-analog is
a *real dynamical object*: assembled as a molecule — bound by ``κ``-gravity on
the derived floor, spun by the self-sustained driven field, no re-imprinting —
it stays bound, keeps both constituent ``½``'s, holds a persistent spin, and
reads **integer, no-flip** from outside: a bound, spinning boson made of two
topological fermions.  (Honest scope: the statistics are topological, not
quantum — no anticommutation, no Pauli principle between identical composites;
and the constituent count is put in by hand — no dynamical confinement selects
2 or 3, though the ``N⋆ = 3`` sector story is the standing suggestion — taken
up next.)

**And ``N⋆ = 3`` is why the count is three.**  *(Measured —
`experiments/n3_junction_fermion.py`, 3/3.)*  The hadron experiment's named gap
— nothing *selected* the constituent count — closes with a mechanism that is
exact: **a spin defect's ``2π`` phase winding must pass through every phase
sector once**, so with three sectors the elementary spin-½ defect *is* a
trivalent, three-sector junction — the colour-singlet triple.  Everything
measured is **emergent** (fields grown from noise, defects formed by the
dynamics, never imprinted): every defect sits on a junction where all three
``N⋆ = 3`` phase sectors meet and every junction sits on a defect (within a
site, across seeds, defects in ``±`` pairs), each isolated defect's
neighbourhood carrying the singlet ``1:1:1`` sector composition; read as a
nematic, **every such junction is a spin-½** (``s = ±½``, the ``−1``/``+1``
double cover) while two-sector *walls* carry none — the elementary fermion and
the three-sector singlet are one object, so the baryon's "3" *is* ``N⋆``.  And
three is selected uniquely: the defect forces all ``P`` phase sectors to meet
(sector-valence ``= P``, measured for ``P = 3, 4, 5``), while the *matter*
tessellation caps its generic junctions at the Plateau valence ``d + 1 = 3``
regardless of ``P`` (``3.00/3.01/3.02`` measured) — the two structures coincide
**only at ``P = 3``**: the three-sector world is the unique one whose own
junctions can carry the elementary spin-½.  (Caught in the wild, too: tight
``±½`` dipoles annihilating mid-flight — the ``½/−½`` meson channel of the
spinor experiment, happening on its own.  Honest scope: the phase binning
inherits ``N⋆ = 3`` from the program's sector results rather than deriving it
afresh, and the hadron composites are still assembled at wells; hadrons
*condensing* at the junctions of a living tessellation is the open rung.)

**And here is where matter cannot yet gather its fermions.**  *(Measured —
`experiments/n3_condensation_boundary.py`, 3/3 — a boundary, the headline a
negative.)*  The last "assembled by hand" gap is **condensation**: can matter
*collect* its spin-½ defects out of a noise-grown field, rather than have them
imprinted?  Not yet — and the experiment maps exactly why, in three clean facts.
Passive κ traps do **not** condense: with ψ grown from noise over the wells, the
well occupancy stays at chance across every trap depth and coarsening time, with
no survival enhancement — the dilute defect gas is **frozen**, nothing carries a
defect across the lattice to a well (K1).  The obvious force fails the obvious
way: the detuning energy (wells suppress ``|ψ|²``) implies a third-law reaction
pulling a mass toward amplitude holes — but *every mass's own well is an
amplitude hole*, so the raw force just **clumps** matter, collapsing a floor-
bound pair through its own exclusion floor (K2).  What *does* work is
**selectivity**: divide the wells' reversible envelope out (react to
``|ψ|²/(1−g)``) and the force responds only to the *topological* core — a strong
pull toward a fermion, essentially zero (``~10⁻¹⁸``) toward an empty well, where
the raw force is fooled by both (K3).  So the handle exists; what is missing is a
**current** to carry fermions to matter — defect transport, the frozen gas of
K1.  The named candidate is the same ``λ ≠ 0`` precession that regenerated the
driven spin (spiral-wave advection mobilises vortex cores).

**And that candidate closes the boundary** (`n3_condensation_transport`, 3/3).  A
relaxational (``λ = 0``) detuned CGL settles to rest — its late-time field
velocity ``⟨|ψ_{t+1} − ψ_t|⟩`` decays to ``~0`` — while a ``λ > 0`` field sustains
a persistent spiral **current** that grows monotonically with ``λ`` (L1: velocity
``0.0002 → 0.05 → 0.10`` for ``λ = 0/0.5/1``).  With the current on, the κ wells
now **gather** fermions from the noise-grown gas above chance (L2: ``4/12`` wells
occupied at ``λ = 1`` versus a flat-κ chance of ``0/12`` and the frozen gas's
``1/12``), and what condenses is a genuine ``±½`` disclination with the ``−1``
holonomy, singly per well (L3).  It is **partial** — about a third of the wells, a
transport window bounded above by numerical stability (``λ ≳ 1.5`` overflows at
``dt = 0.1``) — but K1's frozen boundary is lifted: matter gathers its fermions,
and the whole chain — sector → fermion → composite — runs itself from noise.

**And the two dynamical halves assemble a molecule** (`n3_self_assembly`, 3/3).
Put the mass-side *selective* reaction force (the boundary's K3 — a mass holds a
topological core while ignoring a fellow mass) and the field-side *transport*
current into a single co-evolution, and a molecule assembles itself from noise:
two masses released *beyond* the floor **bind** at the derived exclusion spacing
(S1: ``6/6`` seeds, ``s⋆ ≈ 8.5``, a little tighter when a core is caught between
them); a mass **gathers** a ``±½`` out of the noise-grown gas and holds it while
bound (S2: ``6/6``); and the catch reads ``s = ±½`` with the ``−1`` holonomy (S3:
``6/6``) — gas grown, binding derived, nothing imprinted.  Two honest boundaries
remain: the hold is *imperfect* (a mass holds its fermion for ``~60%`` of the run,
the defect wandering in and out — an intermittent bond, not a locked ground
state), and transport is *not cleanly isolated* (the frozen ``λ = 0`` control
still captures ``4/6``, because capture-and-hold grabs born-*near* defects without
transport — so transport's gather-from-afar is a modest edge at this density).  A
locked decorated ground state and a replenished gas source are the next rungs; but
the object of Act III — a bound composite carrying a spin-½, assembled from noise
— now exists.

---

## 6. Where criticality sits in all of this

The collab work read its clearest patterns near a critical point and used
`β ≈ 0.22` as the value where "the most obvious basic patterns" appeared.  The
program's own criticality results say why this should be so.  The `S`-functional
is maximised in the **critical neighbourhood** once capacity binds (the
criticality-transplant result, `The_Complete_Arc.md` §3 and `Docs`): distinction
peaks at the transition while integration is still available, so a
capacity-bound `S`-climber is pushed to the edge of order — exactly where the
tessellation is richest in forms and the junctions are most sharply trivalent.
The confined phase (clean CW-complex, `Euler ≈ 0`, `2:3:1`) is the ordered side;
deconfinement (Euler defect large) is the disordered side; **criticality is the
band between them where the forms are most articulate.**

The number `0.22` deserves a careful word.  In Act I it is the self-dual
instanton fraction `κ̂` of the 4-D SU(3) vacuum, read at the RG-clean flow scale
— and Act I's honest edge is that it is a coarse-lattice reading of an
`O(0.2–0.4)` quantity, not a scheme-free constant (the deflation test retired
"κ ≈ 0.22" as a bare number in favour of the *function* `κ̂(scale)`).  The
collab's pattern-clarity `β` and Act I's `κ̂` are **not shown to be the same
number** — that they resonate is suggestive, not established.  Marking that
honestly is part of the record.

---

## 7. The line between measured and vision

The foundations of Act III are measured; the destination is not yet.  Held
apart plainly:

**Measured (pre-registered verdicts in this repo).**
- The dimensional forms are CW-cells; the confined ratio is the trivalent
  `2:3:1` fixed by `N⋆ = 3`; the Euler defect is a deconfinement order parameter.
- **The generation count is the dimension of space plus one**: three families
  in 2-D, **four in 3-D** (vertices/edges/faces/volumes, with the Plateau `4/3/2/1`
  valences and Euler `V−E+F−C = 0` on `T³`), the count `min(P, d+1)` — the
  abundances are topologically protected across energy, the heavy family rarest —
  the *structure* of the collab's abundance hierarchy, no numerical quark-match
  claimed.
- The natural pair has a parameter-free binding floor derived from the field's
  own free energy; identity can be measured, not just assigned.
- **The forms carry flavour** — their sector-composition — with multiplet sizes
  fixed by Pascal's triangle (`C(P, d+1−ℓ)` per generation), democratic under the
  sector symmetry, and conserved: a second quantum number orthogonal to the
  generation.
- The forms gravitate at finite speed, drag, radiate quadrupole-led, and merge by
  plunge; a `±` pair on the exclusion floor is a persistent bound object.
- Spin is a chiral term giving intrinsic precession `Ω = −λ`, carried by the 0D
  forms; `λ = 0` restores the collab's parity-symmetric baseline.
- **The molecule spins**: with the chiral term, a bound `±` pair holds a
  persistent, bounded rotation where the achiral molecule drained (its `Z2`
  failure) — the loop from §5 back to §4 closed.
- **The spin is slaved to the living field**: the chiral `ψ` co-evolves with
  the telegrapher `κ` — the `κ` wells hole `ψ`, `ψ`'s phase current presses
  the bond (radial, odd in `λ`, torque-free), and the molecule turns at the
  field's own measured precession `Ω_field` — rate and handedness derived,
  no `Ω_bg` parameter; `λ = 0` is the achiral molecule automatically.
- **The circulation is real, and quantised**: with a vortex pinned at each form,
  the field carries genuine mechanical angular momentum `L ∝` the integer winding
  (sign-locked, a quantised ladder), the phase-current force becomes a genuine
  *torque* (same-sign vortices co-rotate, a vortex–antivortex pair translates),
  and the molecule spins from its own circulation with **no imposed rotation at
  all** — the rigid-rotation flow-profile ansatz retired; `q = 0` is the achiral
  molecule that drains.
- **The self-sustained vortex keeps its topology and pins to matter** (2/3, an
  honest boundary): seeded once and co-evolved under the CGL with no re-imprinting,
  the integer winding is a dynamically-conserved topological charge (noise-robust),
  and the `κ` wells pin the core with a real selection rule (like charges survive
  and track their moving mass, a vortex–antivortex pair annihilates and heals) —
  **but** the *strong* orbital spin did not follow at `λ = 0`: without the
  re-imprinting the circulation drained and the torque was an order weaker.
- **The field's own precession regenerates the strong spin** (3/3): give the
  self-sustained field its CGL twist (`λ ≠ 0`) and the seeded-once, co-evolved
  molecule spins at the pinned strength and beyond (`|Ω|` growing with `λ`), the
  winding preserved — with the **handedness set by the vortex charge and the
  strength by `|λ|`** (two independent knobs).  Both ingredients are necessary: a
  vortex without the precession is weak and draining, a precession without a
  vortex is torque-free.  The self-sustained *strong* spin — matter-bound charge,
  field-driven — closes the emergent boundary.
- **In 3-D, spin is an axial vector** (3/3, field-level): the defect is a vortex
  *line* and its angular momentum ``L`` is a 3-vector *aligned with the line* for
  every orientation (magnitude direction-independent), a sign-locked quantised
  ladder in the winding — spin has a *direction*.  Seeded once and co-evolved
  under the CGL with no re-imprinting, the line keeps its winding and its axis
  (noise-robust, and more stable than the 2-D point vortex because it wraps a
  torus cycle), and matter enforces the same like-survive / unlike-annihilate
  sign rule.  This is an integer / spin-1-like (bosonic) angular momentum.
- **The 3-D molecule carries a vector spin-torque but cannot turn** (2/3, an
  honest negative): binding two lines into a molecule, the torque on the bond is
  real and *vectorial* (tangential, sign-locked, its axis the line's direction,
  the antivortex control null) and the pair holds its winding and ``L`` on the
  floor — yet it is **overdamped**, twisting to a static angle instead of
  spinning (the κ-molecule's ``Q < ½`` returning in 3-D).  The 2-D vortex drive
  beat that drag; the thicker 3-D medium wins.
- **Spin becomes half-integer — a spinor** (3/3): a nematic director field
  (``n̂ ≡ −n̂``, ``RP¹``) has **±½ disclinations** — the director winds by ``π``,
  ``s = q/2`` — and the *double cover* is explicit: a ``2π`` loop around a ``½``
  disclination flips the oriented director (``−1``), ``4π`` restores it (an
  integer defect never flips).  The ``½`` is a conserved, additive charge on
  matter (fusing ``½ + ½ = 1``, annihilating ``½ − ½ → 0``).  This is the
  **topological** realisation of spin-½ (the ``SU(2)`` double cover), not yet the
  quantum Dirac field.
- **The pieces make hadron-like composites** (3/3): ``n`` half-integer
  constituents at derived-floor spacing carry total spin ``s = n/2`` with the
  double-cover class alternating by count — **2 constituents make a boson
  (meson-like), 3 make a fermion (baryon-like)** — the half-integer content
  confined inside while only the composite spin shows outside; and the
  2-constituent composite exists as a *real molecule*, bound by ``κ``-gravity on
  the floor and spun by the self-sustained driven field, integer and no-flip
  from outside.  Hadron statistics from counting, with the program's own binding.
- **``N⋆ = 3`` selects the count** (3/3): a spin defect's ``2π`` winding must
  cross *every* phase sector, so the elementary spin-½ defect **is** a
  trivalent three-sector junction (emergent, bijective, singlet ``1:1:1``
  composition; walls carry no spin) — and the matter tessellation's Plateau cap
  (``d + 1 = 3``, measured for ``P = 3, 4, 5``) means the two structures
  coincide *only* at ``P = 3``.  The baryon's "3" is the sector count — a
  fundamental (``N⋆``) bleeding through into a higher manifestation (the
  constituent count).
- **The condensation boundary is mapped** (3/3, headline a negative): matter
  cannot *yet* gather its fermions from a noise-grown gas to build a composite —
  passive traps are frozen (no transport), the naive third-law force clumps
  matter (every well is an amplitude hole), but the envelope-normalised force is
  *selective* (it responds to the topological core, not an empty well).  The
  handle exists; the missing piece is a transport current.
- **The transport current lifts the boundary** (3/3): the same ``λ ≠ 0``
  precession that regenerated the driven spin *is* that current — a relaxational
  ``λ = 0`` field settles to rest, a ``λ > 0`` field sustains a persistent spiral
  current growing with ``λ`` (L1), and with it the κ wells **gather** fermions
  from the noise-grown gas above chance (L2: ``4/12`` at ``λ = 1`` vs ``0/12``
  chance), each a genuine ``±½`` with the ``−1`` holonomy, singly (L3).  Partial
  (about a third of the wells, ``λ < 1.5`` for stability) — but matter now
  gathers its fermions, and the chain sector → fermion → composite runs itself.
- **A molecule assembles itself from noise** (3/3): the mass-side selective force
  and the field-side transport current in one co-evolution — two masses bind at
  the derived floor (S1, ``6/6``), a mass gathers and holds a ``±½`` from the
  noise-grown gas (S2, ``6/6``; frozen control ``4/6``), the catch reads
  ``s = ±½`` with the ``−1`` holonomy (S3, ``6/6``), nothing imprinted.  Two
  honest boundaries: the hold is intermittent (``~60%`` of the run, not a locked
  bond), and transport is not cleanly isolated (capture-and-hold grabs born-near
  defects without it) — a locked ground state and a gas source are the next rungs.

**Frontier (intuition the foundations now make testable, without a verdict).**
- **The abundance *numbers*.**  The count and ordering of the families are now
  measured as *structure*; the collab's stronger claim — that the densities at
  different energies reproduce the *numerical proportions* of the real
  generations — is deliberately **not** made, and is a numerological stretch the
  program declines rather than a pending verdict.
- **The flavour hierarchy and the CKM numbers.**  The flavour *structure* is now
  measured (Pascal multiplets, democracy, conservation), but a flavour
  *hierarchy* from breaking the sector symmetry is not — the coarsening dynamics
  is winner-take-all, so a global bias dominates rather than grades; and no
  numerical match to CKM mixing or quark masses is claimed (declined, not
  pending).  The `N⋆ = 3` colour tie and the multiplet structure are measured;
  the mass/mixing numbers are the numerology the program does not chase.
- **The quantum Dirac fermion — spin–statistics.**  Spin-½ is now realised
  *topologically* — the nematic ``±½`` disclination, its ``4π`` double cover, on
  matter (the spinor result, 3/3) — and spin as a full axial *vector* in 3-D (the
  vortex-line result, 3/3).  What is *not* yet done is the full quantum fermion:
  the **spin–statistics** connection (anticommutation, the Pauli exclusion of
  identical half-integer excitations), a Dirac equation, a dynamical fermion
  field.  The program already has a *derived* exclusion floor (the no-cloning
  binding) and now a *topological* half-integer spin — tying those two into
  genuine Fermi statistics is the open frontier toward truly fermionic matter.
  (A subsidiary open rung: a 3-D molecule that actually *turns* — the bound pair
  carries a real vector spin-torque but is overdamped, `n3_molecule_3d`, 2/3.)
- **The deeper meaning of the compatibility window.**  The self-sustained strong
  spin lives in a window where the field's chirality and the defect charge agree;
  whether that selection has a deeper structural meaning is an open thread.
- **One number for criticality.**  Whether the collab's pattern-clarity `β` and
  Act I's `κ̂(scale)` are the same object, or one is a shadow of the other.

---

## 8. The map — where Act III lives

**Instruments (`project_genesis/`).**

| File | What it adds |
|---|---|
| `dimensional_forms.py` | CW-census (any dimension): cells, Euler, valence, the Plateau structure, the flavour multiplets |
| `chiral_field.py` | The chiral spin term (CGL), precession `Ω = −λ`, vorticity |
| `two_field.py` | The two-field coupling: `ψ` co-evolves with `κ` — well detuning, the phase-current force, the self-consistent circulation |
| `vortex_chiral.py` | The vorticity-bearing field: a vortex pinned per form, quantised angular momentum, the phase-current *torque* — spin from real circulation; and the self-sustained mode (`winding_number`, `evolve_seeded_field`, `reimprint=False`, `chiral_lambda`): seed once and co-evolve the CGL, the winding a conserved topological charge, the field's precession regenerating a strong spin |
| `vortex_chiral_3d.py` | 3-D spin: the vortex is a *line*, its angular momentum an axial *vector* aligned with the line (`vortex_line`, `line_angular_momentum`, `line_winding`, `evolve_seeded_line`) — quantised, conserved with its axis, integer/bosonic; and the 3-D molecule (`line_phase_force`, `evolve_line_molecule`): a real vector torque the overdamped pair cannot turn |
| `nematic_spinor.py` | Half-integer spin: a nematic director (`n̂ ≡ −n̂`, `RP¹`) and its ``±½`` disclinations — `disclination_strength` (`s = q/2`), `director_holonomy` (the ``4π`` double cover: ``2π → −1``), `plaquette_winding` (the tracker-free defect map) — the topological realisation of a spinor |
| `condensation.py` | The condensation instruments: `grow_defect_gas` (defects from noise), `amp_reaction_force` (the detuning's third-law force — raw clumps, envelope-normalised is selective), `condensation_run` (masses + a co-evolving gas) |
| `capacity_waves.py` | Finite-speed `κ` (telegrapher), retarded gravity, the exclusion contact terms |

**Experiments (`experiments/`).**

| File | Verdict | Result |
|---|---|---|
| `n3_quark_generations.py` | 3/3 | forms as CW-cells; confined `2:3:1`; Euler-deconfinement |
| `n3_form_abundances.py` | 3/3 | three families because space is 2-D (`d+1`); protected; heavy rarest |
| `n3_3d_generations.py` | 3/3 | **four** families in 3-D (`min(P,d+1)`); Plateau `4/3/2/1`; Euler on `T³` |
| `n3_flavour_structure.py` | 3/3 | flavour = sector-composition; Pascal multiplets `C(P,d+1−ℓ)`; democratic, conserved |
| `n3_chiral_spin.py` | 3/3 | spin `Ω = −λ`, on the 0D forms; parity at `λ = 0` |
| `n3_spinning_molecule.py` | 3/3 | a chiral drive lets the bound pair hold `Ω` (§5→§4 closed) |
| `n3_two_field_chiral.py` | 3/3 | `ψ` co-evolves with `κ`; the field presses the bond (radial, λ-odd); the spin is slaved to the field's own `Ω` |
| `n3_vortex_chiral.py` | 3/3 | a vortex per form: quantised angular momentum; the phase-current *torque*; the molecule spins from its own circulation, no imposed drive |
| `n3_emergent_vortex.py` | 2/3 | self-sustained (no re-imprint): the winding a dynamically-conserved charge, the wells pin the core (like-survive/unlike-annihilate) — but the strong spin drains away, an honest boundary |
| `n3_driven_vortex.py` | 3/3 | the field's precession (`λ≠0`) regenerates the circulation: the self-sustained molecule spins at the pinned strength — handedness from the charge, strength from `|λ|`, both ingredients necessary — closing the emergent boundary |
| `n3_vortex_3d.py` | 3/3 | 3-D spin is an axial *vector*: the defect is a line, `L` aligns with it at any orientation, quantised by winding; conserved with its axis under the self-sustained CGL; like-survive/unlike-annihilate (integer/bosonic; the spinor is the frontier) |
| `n3_molecule_3d.py` | 2/3 | the 3-D molecule carries a real *vector* spin-torque (tangential, sign-locked, axis = the line) and holds its spin bound — but it is overdamped and cannot turn (the κ-molecule's `Q<½` back in 3-D), an honest negative |
| `n3_spinor.py` | 3/3 | **half-integer spin**: a nematic ``±½`` disclination (`s = q/2`), the ``4π`` double cover (``2π`` flips the director, ``4π`` restores), conserved & fused (``½+½=1``) & bound to matter — the topological realisation of a spinor (not yet the Dirac field) |
| `n3_hadron_spin.py` | 3/3 | **hadron-like composites**: total spin `s = n/2`, statistics alternate by count (1, 3 fermionic — quark/baryon-like; 2, 4 bosonic — meson-like), the ``½``'s confined inside; the meson-analog a real bound, spinning molecule, integer & no-flip outside |
| `n3_junction_fermion.py` | 3/3 | **`N⋆ = 3` selects the count**: emergent spin defects *are* trivalent three-sector junctions (bijective, singlet `1:1:1`, every one a `±½` fermion; walls spinless); defect valence `= P` vs the matter tessellation's Plateau `3` — they coincide only at `P = 3` |
| `n3_condensation_boundary.py` | 3/3 | the **condensation boundary** (headline a negative): matter cannot yet *gather* its fermions — passive traps frozen (K1), the naive force clumps (K2), but the envelope-normalised force is *selective* (K3); the missing piece is defect transport |
| `n3_condensation_transport.py` | 3/3 | the **transport current** (the boundary lifted): the `λ≠0` precession *is* the current — `λ=0` relaxes to rest, `λ>0` sustains a persistent spiral current (L1); the κ wells now **gather** fermions from a noise-grown gas (`4/12` at `λ=1` vs `0/12` chance, L2), each a `±½` with the `−1` holonomy, singly (L3); partial (~⅓ the wells, `λ<1.5`) |
| `n3_self_assembly.py` | 3/3 | **self-assembly**: the selective force + the transport current in one co-evolution — masses bind at the derived floor (S1, `6/6`), a mass gathers & holds a `±½` from the noise-grown gas (S2, `6/6`; frozen control `4/6`), the catch reads `s=±½` with the `−1` holonomy (S3, `6/6`), nothing imprinted; two honest boundaries — the hold is intermittent (`~60%`) and transport isn't cleanly isolated (capture-and-hold grabs born-near defects) |
| `n3_exclusion_*` (five) | see doc | the derived, parameter-free pair-binding floor |
| `n3_identity_generation / _invariance.py` | 2/3, 2/3 | identity measured from structure, pose-invariant |
| `n3_kappa_lightcone.py` | 2/3 | causal cone `c_κ = √(D/τ)`; the propagating mass |
| `n3_retarded_gravity.py` | 3/3 | drag, inspiral, supersonic silence |
| `n3_quadrupole_line.py` | 3/3 | even-harmonic (no-dipole) selection rule |
| `n3_plunge_ringdown.py` | 3/3 | no long inspiral — the trench plunge |
| `n3_kappa_molecule.py` | 2/3 | the first persistent bound object; achiral, cannot spin |

**Documents.**  `Deriving_The_Exclusion_Coefficient.md` (the pair-binding
derivation, Parts I–V); this file (the Act III synthesis); `The_Complete_Arc.md`
(Acts I–II above it).

---

*Act III's one sentence: held near criticality, a single capacity field
tessellates into a countable inventory of forms — three families because space
is two-dimensional, their proportion fixed by topology and the heavy one rarest —
binds them in conjugate pairs on a derived floor, gravitates them at a finite
speed, and, with one parity-breaking term, spins the point-like ones and lets a
bound pair hold that spin — each form carrying two orthogonal quantum numbers,
its generation and its Pascal-sized flavour; the matter that intuition read
backwards from quarks is, in its structure, now measured, and what stays
declined is only its numerology.*
