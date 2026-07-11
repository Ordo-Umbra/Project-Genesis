# The Emergent Cosmos

### From a binding constant to a self-assembling universe

*The capstone of the second act of Project-Genesis.  `The_Measured_Bridge.md`
closed the first act — the ordinal gap, the functor, and the instanton number
`κ ≈ 0.22`.  This document reports the second: that the same capacity field
`κ`, given nothing but the free energy it already carries, becomes **gravity**,
and drives a chain that runs from a binding constant all the way to structure
formation in an expanding universe.  It is written as a reference — the thesis,
the chain link by link, the toolkit, the honest boundaries, and the open
frontiers — so the arc can be built on cleanly.*

---

## 1. The thesis

The Universal Recursion Principle functional is `S = ΔC + κ·ΔI`: distinction
`ΔC` (structure represented) plus capacity `κ` times integration `ΔI`
(structure bound into a coherent whole).  In the first act, `κ` is the
**exchange rate** of the generative gap — how much integration a
distinction-rich system can afford — and its physical image is the instanton
fraction of the QCD vacuum, `κ ≈ 0.22`.

The second act begins with a single observation about the *dynamical* capacity
field `κ(x, t)`:

> Its equation of motion is the **gradient flow of a free energy**.

From that one fact, with no further assumptions, the following chain is forced
and then *measured*:

> **κ is gravity → matter is stable form → forms gravitate → structure
> assembles → the whole thing runs in an expanding universe.**

The universe the framework describes does not merely *contain* mass and
gravity.  It **assembles itself** out of them — and freezes out when the
expansion runs away.

---

## 2. The pivot: κ has a free energy, so κ is gravity

The capacity field obeys (from the multiphase dynamics,
`multiphase.step_multiphase_kappa`)

    ∂_t κ = D·∇²κ + r·(κ₀ − κ) − c·load·κ ,

which is exactly the gradient flow `∂_t κ = −δF/δκ` of the **capacity free
energy**

    F[κ] = ∫ [ (D/2)|∇κ|² + (r/2)(κ − κ₀)² + (c/2)·load·κ² ] .

Once `κ` carries a genuine energy `F`, everything gravitational is definitional:

| gravitational notion | in the capacity field |
|---|---|
| mass | `load` — the local density of distinction `Σ|∇η|²` |
| the field of a mass | the **well** a load digs in `κ` |
| range | `ξ_κ = √(D/r)` (linearised: `∇²δκ = (r/D)δκ`, a screened/Yukawa field) |
| attraction | two wells overlap → `F` drops → masses draw together |
| equivalence principle | the source is `load·κ²`, bilinear in load → couples to *how much* structure, not what kind |

So `κ` is not merely the number `0.22`.  It is, structurally, **the framework's
gravity**: the weak, universal, mass-sourced binding field, sourced by
structure and back-reacting on it.  The **recovery rate `r` is the graviton
mass** — the persistence↔plasticity dial of the memory experiments is, secretly,
the massive↔massless-graviton dial.

---

## 3. The chain, link by link

Each link is a claim, the number that settles it, the code that measures it,
and the boundary it does not cross.

### Link 1 — κ mediates a screened, mass-sourced attraction
*`capacity_gravity.py` · `n3_kappa_gravity.py`*

Rigid Gaussian masses, `κ` relaxed to steady state, `F` read off:
- The interaction is an attractive **Yukawa** `V ∝ −e^{−r/ξ}/r`.
- The range is **`ξ = 1.02·√(D/r)`, R² = 1.000** across independently varied
  `D` and `r` (a three-way degeneracy: `(D,r) = (0.5,0.02), (1,0.04),
  (2,0.08)` all give `ξ ≈ 5.1`).
- **Equivalence principle**: `V ∝ m₁·m₂`, R² = 0.97.

*Edge:* a classical, static, scalar mediation — the Newton's-`G` analogue of
the integration term, not the metric tensor of GR; screened, not `1/r²`.

### Link 2 — matter is stable form, with m_inertial = m_gravitational
*`stable_forms.py` · `n3_stable_forms.py`*

A particle is a **topological soliton** of the CP² sector field — a winding of
integer charge `Q`, a *stable structure made manifest*:
- A **discrete corpus**: `E ∝ |Q|` (the Bogomolny ladder `E = 6.84, 12.47,
  18.68, 24.90` for `Q = 1..4`).  Matter is quantised because topology is.
- **Stable**: a charge-`Q` form preserves `Q` under cooling and holds an energy
  floor; a topologically trivial bump decays to the vacuum.
- **Structural mass = gravitational mass**: each form's distinction density,
  fed into the κ dynamics, sources a well with `M_grav = 3.67·E, R² = 1.000`.
  Inertial and gravitational mass coincide because both are the form's
  distinction content — the equivalence principle *explained*, not imposed.  (At
  strong coupling the well saturates and `M_grav` bends sub-linear — a real
  field response, so it is not merely a relabelling of the energy.)

*Edge:* 2-D CP² solitons in lattice-action units; the proportionality is
explained by the shared distinction root, and is the framework's account of why
`m_i = m_g` — not a derivation of the Standard-Model spectrum.

### Link 3 — the forms assemble structure
*`capacity_dynamics.py` (`capacity_force`, `evolve`) · `n3_self_gravity.py`*

Let the masses move in the κ-field they mutually source.  By the envelope
theorem (at relaxed `κ`, `δF/δκ = 0`) the force is the direct coupling term
`F_i = −c·Σ_x load_i·κ·∇κ`; overdamped, `dR_i/dt = μ·F_i`:
- **Two masses fall together and merge**, the fall *accelerating* — and with
  the coupling off (`c = 0`) they stay put, so it is the field, not drift.
- **Nine masses accrete** into a single bound object (mass conserved) — bound
  structure from first principles.

### Link 4 — a complete gravitational dynamics
*`capacity_dynamics.py` (`evolve_inertial`) · `n3_orbital_gravity.py`*

Give the forms inertia (`M·d²R/dt² = F`, symplectic velocity-Verlet):
- **A Kepler-like family**: radial plunge → bound ellipse → near-circular →
  unbound escape, tuned by the tangential speed.
- **Energy conserved** to **0.12%** over an orbit (`T + F[κ]`).
- **Precession**: the screened potential makes the bound ellipse rosette —
  perihelion advance **+141°/orbit** — the direct dynamical fingerprint of the
  finite range `√(D/r)` (the finite-range analogue of, not identical to, GR's
  relativistic advance).
- **Virialisation**: a dissipative cloud rings down to `2⟨T⟩/|⟨W⟩| → 1.19`.

### Link 5 — structure against expansion
*`capacity_dynamics.py` (`hubble_flow`, `fof_groups`) · `n3_cosmic_structure.py`*

Put the dynamics in an expanding background (`v = H·(r − c)`):
- **Turnaround**: a receding pair decelerates, reaches a maximum separation
  (turnaround radius `10.0 → 12.5 → 15.5 → 22.9` as `H` rises), and recollapses
  — below a **critical rate**, above which it escapes.  The spherical-collapse
  picture.
- **Structure vs expansion**: a cloud collapses to one halo at low `H` (100%)
  but is dispersed to fragments as `H` rises (largest bound fraction `100% →
  75% → 33% → 25%`).  A faster-expanding universe forms less.

### Link 6 — an FLRW universe: scale factor, Hubble drag, dark energy
*`capacity_dynamics.py` (`friedmann_rates`, `evolve_cosmological`) · `n3_expanding_universe.py`*

Replace the coasting background with an evolving scale factor obeying a
Friedmann-like law `(ȧ/a)² = H₀²[Ω_m a^{−p} + Ω_Λ]`:
- **Expansion histories**: `a(t)` decelerates under matter, accelerates under
  Λ (`a → 3.0 … 15.1` as `Ω_Λ: 0 → 1`).
- **Hubble drag**: a peculiar velocity redshifts as `1/a` (`a·|v_pec|` constant
  to **0.2%**) — momentum redshift, which the coasting model could not show.
- **Dark-energy freeze-out**: the same cloud collapses to one halo in a matter
  universe (**100%**) but is suppressed as `Ω_Λ` rises (`100% → 92% → 92% →
  92% → 42%`), sharply at pure de Sitter — the defining signature of Λ in
  structure growth.

### Link 7 — the loop closes: the κ-field drives its own expansion
*`capacity_dynamics.py` (`capacity_vacuum_density`, `deceleration_parameter`, `acceleration_onset`, `integrate_scale_factor`) · `n3_self_contained_cosmos.py`*

Link 6 still *dialled* Ω_Λ.  But the capacity field supplies the dark energy
itself: the recovery term `r·(κ₀ − κ)` heals the field back to baseline — an
energy spent **maintaining itself** that does not dilute as space expands,
exactly a cosmological constant.  So `ρ_Λ = coeff·r·κ₀²` is a *property of the
field*, and with matter diluting as `ρ_m0 a^{−dim}` the Friedmann equation
`H² = ρ_m + ρ_Λ` makes the history a **prediction**:
- **Emergent decel→accel.**  `a(t)` decelerates under matter, then turns over
  into Λ-dominated acceleration (`q` crosses zero and runs to the de Sitter
  limit `q → −1`).
- **Dark energy = self-maintenance.**  The acceleration onset
  `a_acc = (ρ_m0/2ρ_Λ)^{1/dim}` is *derived*; more recovery `r` (more
  self-maintenance) brings acceleration earlier (`a_acc = 1.71, 1.36, 1.19,
  1.00` at `r = 0.01, 0.02, 0.03, 0.05`, predicted = measured).
- **Energy-budget handoff.**  Matter dilutes while the capacity vacuum stays
  constant → the universe passes matter- to Λ-dominated, acceleration
  following.

The same `κ` is now **gravity** (its free energy), **matter** (stable forms),
*and* **dark energy** (its self-maintenance).  The cosmological loop is closed:
a cosmos out of one field.

---

## 4. The toolkit

For building on the arc, the modules and experiments that make it up:

**Modules** (`project_genesis/`)
| file | provides |
|---|---|
| `multiphase.py` | the κ dynamics `∂_t κ = D∇²κ + r(κ₀−κ) − c·load·κ` (its origin) |
| `topological_charge.py` | CP² geometric (Berg–Lüscher) charge, cooling, action |
| `capacity_gravity.py` | `screening_length`, `gaussian_load`, `relax_capacity`, `capacity_free_energy`, `interaction_potential`, `fit_yukawa_range`, `well_range` |
| `stable_forms.py` | `winding_form`, `trivial_bump`, `distinction_density`, `structural_mass`, `gravitational_mass`, `form_charge` |
| `capacity_dynamics.py` | `capacity_force`, `evolve` (overdamped), `evolve_inertial`, `hubble_flow`, `fof_groups`, `friedmann_rates`, `evolve_cosmological`, `capacity_vacuum_density`, `deceleration_parameter`, `acceleration_onset`, `integrate_scale_factor` |

**Experiments** (`experiments/`) — each writes a figure + verdict to `artifacts/`
| file | the link |
|---|---|
| `n3_kappa_gravity.py` | κ is gravity (Yukawa, `√(D/r)`, equivalence) |
| `n3_stable_forms.py` | matter is stable form (`E∝|Q|`, `m_i = m_g`) |
| `n3_self_gravity.py` | infall and accretion |
| `n3_orbital_gravity.py` | orbits, energy, precession, virial |
| `n3_cosmic_structure.py` | turnaround, structure vs expansion |
| `n3_expanding_universe.py` | FLRW: scale factor, Hubble drag, dark energy |
| `n3_self_contained_cosmos.py` | the closed loop: dark energy from κ's self-maintenance drives the expansion |

**Tests**: `test_capacity_gravity.py`, `test_stable_forms.py`,
`test_capacity_dynamics.py`, `test_capacity_inertial.py`,
`test_cosmic_structure.py`, `test_capacity_cosmology.py`.

---

## 5. The honest boundaries, collected

The arc is real and its boundaries are bright.  What it is **not**:

- **Not general relativity.** `κ` is a scalar binding *coupling* (the Newton's-
  `G` analogue), not the spacetime metric.  There is no metric, no horizon, no
  light-bending, no relativistic growth factor.
- **Screened, not `1/r²`.** The force has finite range `√(D/r)` (a massive-
  graviton analogue).  Whether a genuinely long-range regime exists is open.
- **Adiabatic and 2-D (dynamics).** The κ-field is relaxed to steady state each
  step (Born–Oppenheimer), and the dynamics runs in 2-D lattice units; masses
  are rigid Gaussian blobs standing in for the stable forms (whose `m_g = m_i`
  is established separately).
- **The Friedmann closure is partial.** The **dark-energy density is now
  derived** — it is the capacity field's self-maintenance, `ρ_Λ = coeff·r·κ₀²`
  (Link 7) — but the *matter* dilution law `a^{−dim}` is still imposed, the
  identification carries a modelling coefficient, and there is no metric or
  relativistic stress-energy tensor.  The κ screening length is a fixed
  physical scale (it does not redshift with the background).

The arc reproduces the **mechanisms** — a screened universal attraction, an
emergent equivalence principle, orbits and precession, spherical collapse, and
the dark-energy suppression of growth — not a quantitative ΛCDM.

---

## 6. Open frontiers

Ranked by how much they would deepen the arc.  *(Frontier 1 — closing the
cosmological loop — is now partly done: Link 7 derives the dark energy from
the κ-field's self-maintenance.  What remains is to derive the **matter**
sector's dilution and the full stress-energy from the field too.)*

1. **Complete the stress-energy closure.** Link 7 derives `ρ_Λ` from the
   capacity recovery; the next step is to derive the *matter* density and its
   `a^{−dim}` dilution from the stable forms' own energy, so the entire
   Friedmann source — not just its vacuum piece — comes from the κ-field.
2. **The unscreened regime.** Is there a limit (`r → 0`, or a different
   coupling) in which κ-gravity becomes genuinely long-range `1/r²`?  The whole
   massive/massless-graviton question lives here.
3. **Self-consistent forms.** Let the stable forms move as full CP² fields
   (not rigid blobs), so matter and gravity co-evolve without the adiabatic
   split.
4. **Three dimensions.** Lift the dynamics from 2-D to 3-D (the κ machinery is
   already dimension-agnostic; the cost is compute).
5. **The one-κ identity.** The first act's `κ ≈ 0.22` (integration exchange
   rate) and the second act's `κ` (gravity) are, in the framework, *the same
   field*.  Making that identity quantitative — showing the gravitational
   coupling and the integration constant are one number — would fuse the two
   acts.
6. **A metric formulation.** Recast κ-gravity so the well acts as an effective
   metric / index of refraction, the bridge from "binding coupling" toward a
   geometric theory.

---

## 7. Where it sits in the whole program

Project-Genesis now has two measured arcs meeting at `κ`:

- **Act I — the generative gap** (`The_Generative_Gap.md`,
  `The_Measured_Bridge.md`): a recursive field distinguishes more than it can
  integrate; the shortfall is capacity `κ`; the ordinal gap maps by a genuine
  functor to the QCD vacuum, where `κ` is the instanton fraction `≈ 0.22`.
- **Act II — the emergent cosmos** (this document): the *same* capacity field,
  because it carries a free energy, is gravity — and drives matter, structure,
  orbits, and an expanding universe.

`κ` is the hinge.  In Act I it is the exchange rate that binds distinction into
integration; in Act II it is the field that binds matter into structure.  The
framework's wager is that these are **one κ** — that the constant which decides
how much a system can integrate is the same one that decides how strongly
matter gravitates.  Frontier 5 above is where that wager becomes a measurement.

> The generative gap, in failing to close, builds structure.  Give that
> structure a capacity field with an energy, and the structure gravitates,
> assembles, orbits, and — in an expanding universe — grows until the expansion
> outruns it.  A cosmos, emergent from a gap that cannot finish closing.

---

*See `The_Generative_Gap.md` and `The_Measured_Bridge.md` for Act I;
`Capacity_As_Gravity.md` for the running derivation of Act II; and the README
"Capacity as gravity" through "expanding universe" sections for figures and
reproduction commands.*
