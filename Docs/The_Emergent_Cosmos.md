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

### Link 8 — the matter source read off the form spectrum
*`capacity_dynamics.py` (`matter_energy_density`) · `n3_matter_from_forms.py`*

Link 7 closed the loop but still *dialled* the matter density `ρ_m0` and
*imposed* its `a^{−dim}` dilution.  Link 8 removes them, tracing the matter
source back to Link 2 (matter = stable form).  Two facts about the forms do all
the work — their rest energies are quantised by charge, and that charge is
topologically protected:
- **The Bogomolny mass ladder.**  The `Q = 1..4` forms sit on a straight line
  `E ≈ 5.38·|Q| + 2.87` (R² = 0.986) — an energy floor per unit charge.  So
  `ρ_m0 = ΣE/V ∝ Σ|Q|` is *read off the topological content*, not chosen.
- **Topological protection.**  Deform a charge-2 form and cool: the charge
  returns to **exactly 2** for every realisation up to noise ≈ 0.3 (the raw
  geometric charge has already blown up on UV dislocations, `⟨|Q|⟩ → 10`);
  only past a threshold does a kick comparable to the field itself tunnel it to
  a neighbouring sector.  The rest-energy floor rides with the protected charge.
- **The dilution law is topological.**  Because that total rest energy is
  *conserved*, spreading it through a growing comoving volume `V₀·a^{dim}` gives
  `ρ_m(a)` with a log–log slope of exactly `−dim` — the `a^{−dim}` law derived,
  not assumed.
- **The cosmos from its form content.**  Feeding `ρ_m0` (forms) and `ρ_Λ`
  (recovery) into the Friedmann integrator makes the acceleration onset a
  function of *how many forms the universe holds*: `a_acc = 1.14 → 1.95` as
  `Σ|Q| = 10 → 50`, on the predicted curve.

The last dialled density is gone.  The whole Friedmann source — matter *and*
dark energy — is now the field's own content: `ρ_m0` from the Bogomolny
spectrum, `a^{−dim}` from charge conservation, `ρ_Λ` from self-maintenance.

### Link 9 — the equation of state: what kind of stuff the forms are
*`capacity_dynamics.py` (`equation_of_state_from_dilution`, `gas_equation_of_state`) · `n3_form_equation_of_state.py`*

The Friedmann source is now field-sourced, but with an *assumed character*:
matter as pressureless **dust**, the vacuum as a **cosmological constant**.
Those characters are equations of state `p = w·ρ` — the last qualitative input,
and exactly the piece a relativistic `T_{μν}` will need.  Link 9 measures `w`,
three independent ways that all agree:
- **Kinetic (the measurement).**  A gas of forms with dispersion `σ_v` has
  `w = p/ρ = Σγm v²/dim ÷ Σγm`, rising from `w ≈ 0` (cold — dust) toward
  `w = 1/dim` (hot — radiation).  The *cold* form gas the cosmology assumes
  really is dust.
- **Mechanical (`p = −∂E/∂V`).**  A form is a localized lump: its rest energy is
  independent of the box (`∂E/∂V → 0`), so `p = 0`, `w = 0` — dust with no
  reference to velocity.  The vacuum has `E_Λ = ρ_Λ·V`, so `p = −ρ_Λ`, `w = −1`.
- **Kinematic (from the dilution exponent).**  Since `ρ ∝ a^{−dim(1+w)}`, the
  `a^{−dim}` matter law of Link 8 reads back `w = 0`, the constant vacuum
  `w = −1`, radiation `w = 1/dim`.

So the two components a covariant `T_{μν}` must carry are fixed and mutually
consistent: a pressureless dust of forms (`w = 0`) and a `w = −1` capacity
vacuum — the "dust + Λ" of the cosmology *measured, not assumed*, with the warm
form gas bridging dust and radiation.

### Link 10 — the relativistic closure: expansion from a stress-energy tensor
*`capacity_dynamics.py` (`stress_energy_tensor`, `covariant_conservation_rate`, `friedmann_acceleration`, `integrate_stress_energy`) · `n3_stress_energy_closure.py`*

The pieces are all measured now — the densities (Links 7–8) and their equations
of state (Link 9).  Link 10 assembles them into a perfect-fluid stress-energy
tensor in a 3+1 FLRW background, `T^μ_ν = diag(−ρ, p, p, p)` with `p_i = w_iρ_i`,
and lets the expansion *follow* from it instead of imposing the Friedmann matter
law:
- **Conservation derives the dilution.**  The one non-trivial FLRW component of
  `∇_μT^{μν}=0` is the continuity equation `ρ̇_i + 3H(ρ_i + p_i) = 0`; integrating
  it reproduces `ρ_i ∝ a^{−3(1+w_i)}` to `~10⁻⁷` — dust `a^{−3}`, vacuum constant,
  radiation `a^{−4}`.  The `a^{−dim}` law we had *imposed* is now an *output*.
- **Expansion as an output.**  Closing with `H² = ρ` and `ä/a = −½(ρ + 3p)`, the
  effective equation of state runs `w_eff : −0.17 → −1` (matter → Λ) and
  `q = ½(1+3w_eff)` crosses zero at `a_acc = 1.36` — decel→accel with nothing
  about the dilution assumed.
- **Consistency.**  The coupled-tensor `a(t)` coincides with the earlier
  imposed-`a^{−3}` cosmology to `max rel |Δa|/a ≈ 5×10⁻³`; the new derivation
  *explains* the old input rather than replacing it.

The Friedmann level is closed: a stress-energy tensor built from the field's own
content, made to conserve covariantly, produces the expansion history as a
consequence.  The Friedmann equation is now an output.

### Link 11 — the variational closure: the Friedmann equation from an action
*`capacity_dynamics.py` (`minisuperspace_lagrangian`, `hamiltonian_constraint`, `integrate_friedmann_action`) · `n3_friedmann_from_action.py`*

Link 10 still *put in by hand* the Einstein/Friedmann relations that turn the
stress-energy tensor into expansion.  Those relations are the content of a
minisuperspace variational principle.  For flat FLRW with lapse `N` and scale
factor `a` (units `8πG/3 = 1`), the action

    S = ∫ dt [ −a ȧ²/N − N a³ ρ(a) ]

— gravitational kinetic term `−a ȧ²` (the geometry's own free energy) plus the
stress-energy content `ρ(a)` — has the lapse/Hamiltonian constraint
`∂S/∂N = 0 ⇒ H² = ρ` (Friedmann) and the `a`-Euler–Lagrange equation (with
conservation) `⇒ ä/a = −½(ρ + 3p)`:
- **One history, three derivations.**  The `a`-Euler–Lagrange evolution
  reproduces the stress-energy route (`~10⁻⁸`) and the imposed-`a^{−3}`
  cosmology (`~5×10⁻³`).
- **Friedmann is a first integral, not an input.**  Along that acceleration
  equation the constraint `C = H² − ρ` stays `~10⁻¹⁰` though it is never
  substituted; analytically `Ċ = −2HC`, so `C = 0` is preserved — and, since
  `H > 0`, an **attractor**: a wrong initial expansion rate relaxes onto `H²=ρ`.
- **The physics is intact.**  `q` still crosses zero at `a_acc = 1.36`.

The last hand-input at the Friedmann level is gone: `H² = ρ` and `ä/a =
−½(ρ + 3p)` are the Hamiltonian constraint and Euler–Lagrange equation of one
action, and `H² = ρ` a preserved, attracting first integral.  The scale factor's
dynamics and its stress-energy source come from a single variational principle.

### Link 12 — gravity from the capacity field: the action as capacity free energy
*`capacity_dynamics.py` (`scale_capacity`, `capacity_kinetic_energy`, `capacity_scalar_acceleration`, `integrate_capacity_scale`) · `n3_gravity_from_capacity.py`*

Link 11 still *posited* the gravitational kinetic term `−a ȧ²`.  It **is** the
capacity field's own kinetic free energy.  Identify the scale factor with the
exponential of the homogeneous capacity scalar — the zero-mode `κ_s` whose global
value sets the overall integration scale — `a = e^{κ_s}` (`κ_s = ln a`,
`H = κ̇_s`).  Its kinetic free energy on the FLRW measure `a³` is `a³ κ̇_s² = a ȧ²`,
so `−a ȧ² = −a³ κ̇_s²`:
- **The gravitational term is capacity free energy.**  Along the history the
  posited `−a ȧ²` and `−a³ κ̇_s²` coincide to `~10⁻¹⁶` — the geometry's kinetic
  energy is the field's.
- **Friedmann is an energy balance `κ̇_s² = ρ`.**  The mean capacity's kinetic
  free-energy density equals the content; `|κ̇_s² − ρ|` stays `~10⁻⁹`.
- **Expansion is the capacity scalar rolling.**  Its field equation
  `κ̈_s = −(3/2)(κ̇_s² + p)` gives `κ̈_s → 0` in the vacuum limit (de Sitter) and
  reproduces the whole `a(t)` (`~10⁻⁸`); the balance is an attractor.

The last posit is gone: the gravitational action's kinetic term is the capacity
field's own kinetic free energy, and the cosmic expansion is that field — the
mean capacity — rolling under it.  Gravity, not just its cosmology, is read off
the URP field.

---

## 4. The toolkit

For building on the arc, the modules and experiments that make it up:

**Modules** (`project_genesis/`)
| file | provides |
|---|---|
| `multiphase.py` | the κ dynamics `∂_t κ = D∇²κ + r(κ₀−κ) − c·load·κ` (its origin) |
| `topological_charge.py` | CP² geometric (Berg–Lüscher) charge, cooling, action; `coherent_fraction`, `cp_action_density`, `cp_coherent_fraction`, `cp_metropolis_sweep`, `charge_variance_per_action` (the one-κ operator + thermal CP sampler + dimensionless susceptibility) |
| `capacity_gravity.py` | `screening_length`, `gaussian_load`, `relax_capacity`, `capacity_free_energy`, `interaction_potential`, `fit_yukawa_range`, `well_range` |
| `stable_forms.py` | `winding_form`, `trivial_bump`, `distinction_density`, `structural_mass`, `gravitational_mass`, `form_charge` |
| `capacity_dynamics.py` | `capacity_force`, `evolve` (overdamped), `evolve_inertial`, `hubble_flow`, `fof_groups`, `friedmann_rates`, `evolve_cosmological`, `capacity_vacuum_density`, `deceleration_parameter`, `acceleration_onset`, `integrate_scale_factor`, `matter_energy_density`, `equation_of_state_from_dilution`, `gas_equation_of_state`, `stress_energy_tensor`, `covariant_conservation_rate`, `friedmann_acceleration`, `integrate_stress_energy`, `minisuperspace_lagrangian`, `hamiltonian_constraint`, `integrate_friedmann_action`, `scale_capacity`, `capacity_kinetic_energy`, `capacity_scalar_acceleration`, `integrate_capacity_scale` |

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
| `n3_matter_from_forms.py` | the matter source from the form spectrum: `ρ_m0 ∝ Σ|Q|`, `a^{−dim}` from topology |
| `n3_form_equation_of_state.py` | the equation of state: cold forms are dust (`w=0`), the capacity vacuum is Λ (`w=−1`) |
| `n3_stress_energy_closure.py` | the relativistic closure: `T^μ_ν` from the field, expansion as a consequence of `∇·T=0` |
| `n3_friedmann_from_action.py` | the variational closure: `H²=ρ` as the Hamiltonian constraint / a first integral of an action |
| `n3_gravity_from_capacity.py` | gravity from the field: `−a ȧ²` as the capacity scalar's kinetic free energy, expansion as `κ_s=ln a` rolling |
| `n3_one_kappa_frontier.py` | the one-κ frontier: `κ̂=Σ\|q\|/Σe` as one operator across SU(3) (Act I) and CP² (Act II) — same concept, not (yet) one number |
| `n3_kappa_obstruction.py` | the one-κ obstruction: the sharper invariant `⟨Q²⟩/⟨S⟩` fails too — mechanism is the instantons' different RG fate |

**Tests**: `test_capacity_gravity.py`, `test_stable_forms.py`,
`test_capacity_dynamics.py`, `test_capacity_inertial.py`,
`test_cosmic_structure.py`, `test_capacity_cosmology.py`,
`test_matter_from_forms.py`, `test_form_equation_of_state.py`,
`test_stress_energy_closure.py`, `test_friedmann_from_action.py`,
`test_gravity_from_capacity.py`, `test_one_kappa_frontier.py`,
`test_kappa_obstruction.py`.

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
- **The Friedmann closure is now sourced by the field, but stays Newtonian.**
  Both densities are derived: the **dark energy** is the capacity field's
  self-maintenance `ρ_Λ = coeff·r·κ₀²` (Link 7), and the **matter** density
  `ρ_m0 ∝ Σ|Q|` with its `a^{−dim}` dilution comes from the Bogomolny form
  spectrum and topological charge conservation (Link 8).  What remains: the
  vacuum identification carries a modelling coefficient, the forms are treated
  as point masses in a `dim`-volume (which is what supplies `a^{−dim}`), and
  there is still no metric or relativistic stress-energy tensor.  The κ
  screening length is a fixed physical scale (it does not redshift).  The
  *equation of state* of both components is measured (Link 9: forms are dust
  `w = 0`, vacuum is `w = −1`) and assembled into a perfect-fluid stress-energy
  tensor whose covariant conservation *derives* the dilution and makes the
  expansion an output (Link 10).  Even the Friedmann relations are no longer
  hand-inputs: they are the Hamiltonian constraint and Euler–Lagrange equation
  of a minisuperspace action, with `H² = ρ` a preserved, attracting first
  integral (Link 11), and that gravitational term `−a ȧ²` is identified with the
  capacity scalar's own kinetic free energy `−a³ κ̇_s²` under `a = e^{κ_s}`
  (Link 12).  What remains: that identification (`a = e^{κ_s}`, the wrong-sign
  conformal mode) is a *reading*, the action stays *minisuperspace* (one degree
  of freedom, homogeneous), and there are no inhomogeneous field equations or a
  solved metric.

The arc reproduces the **mechanisms** — a screened universal attraction, an
emergent equivalence principle, orbits and precession, spherical collapse, and
the dark-energy suppression of growth — not a quantitative ΛCDM.

---

## 6. Open frontiers

Ranked by how much they would deepen the arc.  *(Frontier 1 — sourcing the
Friedmann level from the field — is now done end to end: Links 7–8 derive the
dark energy and the matter density/dilution, Link 9 measures the equations of
state, Link 10 assembles the conserved `T^μ_ν`, Link 11 derives the Friedmann
relations as the constraint + Euler–Lagrange equation of an action, and Link 12
identifies that action's gravitational term with the capacity scalar's kinetic
free energy.  What remains is to lift the homogeneous minisuperspace to a full
inhomogeneous field theory.)*

1. **Inhomogeneity: perturbations and a growth factor.** The whole gravitational
   sector is now the capacity scalar `κ_s = ln a` rolling (Links 11–12), but only
   its homogeneous zero-mode — one degree of freedom.  The next step is to let
   `κ_s` vary in space: cosmological perturbations `δκ_s(x, t)`, a real growth
   factor `D(a)`, and a solved (perturbed) metric — connecting back to the N-body
   structure work on a relativistic footing, and giving the wrong-sign conformal
   mode a proper field-theoretic treatment.
2. **The unscreened regime.** Is there a limit (`r → 0`, or a different
   coupling) in which κ-gravity becomes genuinely long-range `1/r²`?  The whole
   massive/massless-graviton question lives here.
3. **Self-consistent forms.** Let the stable forms move as full CP² fields
   (not rigid blobs), so matter and gravity co-evolve without the adiabatic
   split.
4. **Three dimensions.** Lift the dynamics from 2-D to 3-D (the κ machinery is
   already dimension-agnostic; the cost is compute).
5. **The one-κ identity (probed — an open frontier).** The first act's
   `κ ≈ 0.22` (integration exchange rate) and the second act's `κ` (gravity)
   are, in the framework, *the same field*.  `n3_one_kappa_frontier.py`
   built the shared operator — the Bogomolny coherent fraction
   `κ̂ = Σ|q|/Σe`, one function that is both `self_dual_fraction` (SU(3)) and
   `cp_coherent_fraction` (CP) — and measured it in both sectors.  The honest
   result is a *boundary*: `κ̂` **rises** to 0.22 under the SU(3) Wilson flow but
   **falls** under the CP² cooling flow, so the naive coherent fraction does not
   give a parameter-free `κ_I = κ_II`.  The two κ's are the same *concept*
   (the integration fraction) but not the same measured number in this estimator.
   A genuine identity would need a matched RG condition across the 4-D and 2-D
   flows, or a different invariant (a `χ_top` ratio, or the instanton-size
   distribution).  `n3_kappa_obstruction.py` then tested the sharper `χ_top`
   invariant `⟨Q²⟩/⟨S⟩` — it *also* diverges (`~10⁻⁴` gauge vs `~10⁻²` CP) — and
   diagnosed the **mechanism**: under the flow 4-D SU(3) instantons are
   scale-invariant and *survive* (the coherent fraction rises), while 2-D CP
   instantons are scale-relevant and *annihilate* (the instanton density
   collapses).  Same flow, opposite fate — so the two sectors' topology lives at
   different, moving scales, and no `q(x),e(x)` estimator can coincide.  A real
   bridge must match the physical instanton scales first (a renormalisation
   condition) or be a framework-level definition.  The frontier is now a
   *precise obstruction* with a stated mechanism, not a hand-waved hope.
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
