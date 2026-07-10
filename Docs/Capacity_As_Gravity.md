# Capacity as Gravity

### κ as the framework's universal, mass-sourced binding field

*A short synthesis of the claim that the URP capacity field κ plays the role
of gravity — and the measurement (`capacity_gravity.py`,
`n3_kappa_gravity.py`) that puts numbers under it.*

---

## The claim

Across the `S = ΔC + κ·ΔI` program, κ has been the **capacity** — the
dynamical field that decides how much integration the system can afford,
consumed by load and regenerating with slack:

    ∂_t κ = D·∇²κ + r·(κ₀ − κ) − c·load·κ .

The observation that turns κ into *gravity* is that this evolution is a
**gradient flow**, `∂_t κ = −δF/δκ`, of the capacity free energy

    F[κ] = ∫ [ (D/2)|∇κ|² + (r/2)(κ − κ₀)² + (c/2)·load·κ² ] .

Once κ has an energy `F`, everything gravitational follows by definition:

- **Mass.** `load` — the local density of distinction (`Σ|∇η|²`), the sheer
  amount of structure present — is the source. A concentration of load is a
  mass.
- **The field of a mass.** A mass depletes κ around it: it digs a **well** in
  the capacity field. Linearising the deficit `δκ = κ₀ − κ` gives a screened
  (Yukawa) equation `∇²δκ = (r/D)·δκ`, so the well has a range

      ξ_κ = √(D / r) .

- **Attraction.** Because `F` is a real energy, two masses lower it by
  overlapping their wells — they **attract**. The interaction energy `V(r)` is
  the separation-dependence of `F`.
- **The equivalence principle.** `F`'s source term is `load·κ²`; the
  interaction between two masses is bilinear in their loads, so the force
  couples to *how much* structure is present, not *what kind* — a universal
  coupling.

κ is therefore not merely *a* coupling constant (the `κ ≈ 0.22` of the
functorial bridge) — it is, structurally, **the framework's gravity**: the
weak, universal, mass-sourced binding field, sourced by structure and
back-reacting on it.

---

## The measurement

`experiments/n3_kappa_gravity.py` places rigid Gaussian masses in a 3-D box,
relaxes κ to steady state, and reads `F`.

1. **κ mediates a screened attraction.** Two masses give `V(r) < 0`
   throughout, deepening as they approach, with a clean **Yukawa** shape
   `V ∝ −e^{−r/ξ}/r` (ξ ≈ 4.8 at the reference coupling).
2. **The range is √(D/r).** Varying `D` and `r` *independently*, the measured
   screening length tracks the prediction as **ξ_meas = 1.02·√(D/r)** with
   **R² = 1.000** — including a three-way degeneracy where
   `(D,r) = (0.5, 0.02), (1, 0.04), (2, 0.08)` all give ξ ≈ 5.1. The **recovery
   rate is the graviton mass**: fast recovery screens the force to short
   range, slow recovery lets it reach. (The persistence↔plasticity dial of the
   memory work is, secretly, the massive↔massless-graviton dial.)
3. **The equivalence principle holds.** The interaction strength scales with
   the **product of the masses**, `V ∝ m₁·m₂` (R² = 0.97).

---

## The honest edge

This is a **classical, static, scalar** mediation. κ is the binding
*coupling* — the Newton's-`G` analogue of the integration term — not the
**metric tensor** of general relativity. Two differences are real and worth
keeping bright:

- **It is screened.** The `√(D/r)` range makes κ-gravity a *massive-graviton*
  analogue, short-ranged, unlike Newtonian `1/r²` (which is the `r → 0`,
  massless limit — no recovery, capacity never heals locally). Whether a
  genuinely long-range (unscreened) regime exists in the framework is an open
  question.
- **The masses are rigid.** We impose static load blobs; a full treatment
  would let the structure that *sources* κ also *move* in the κ-field it
  creates — self-gravitating distinction. That two-way, dynamical version is
  the natural next step.

What the measurement establishes is not an identification with GR but the
gravitational **role**, precisely: a universal attraction, sourced by and
back-reacting on structure, whose strength is κ and whose range is the
capacity recovery length. In the general sense — the thing that binds,
weakly and universally, wherever there is mass — κ *is* the framework's
gravity.

---

---

## What gravitates: the stable forms

The masses above were rigid, imposed blobs. The companion experiment
(`stable_forms.py`, `n3_stable_forms.py`) asks what a mass actually *is* in
the framework and lets the answer source the κ-well itself. A particle is a
**topological soliton** of the CP² sector field — a winding of integer charge
`Q`, a *stable structure made manifest*. Three results tie it back to gravity:

- The admissible forms make a **discrete corpus**: integer Q, with structural
  (inertial) masses on a Bogomolny ladder `E ∝ |Q|`. Matter is quantised
  because topology is.
- They are **stable**: a charge-Q form preserves Q and holds an energy floor;
  a topologically trivial bump decays to the vacuum.
- Their **gravitational mass equals their structural mass**: each form's
  distinction density, fed into the κ dynamics above, sources a well with
  `M_grav ∝ E` (R² = 1.000). Inertial and gravitational mass coincide because
  both are the form's distinction content — the equivalence principle,
  *explained* by the shared root rather than imposed. (At strong coupling the
  well saturates and `M_grav` bends sub-linear — a real field response, so the
  gravitational mass is not merely a relabelling of the energy.)

Together the two experiments close the loop: the generative gap builds
structure; where capacity permits it sets into stable, discrete **forms**
(matter = mass = concentrated distinction); and that same distinction sources
**κ-gravity**, pulling the forms together with a strength equal to their own
mass.

## Structure that grows itself

The forms above were held still.  Released — allowed to move in the κ-field
they mutually source — they complete the loop dynamically
(`capacity_dynamics.py`, `n3_self_gravity.py`).  In the adiabatic
(Born–Oppenheimer) regime κ relaxes to steady state for the instantaneous
positions and each mass drifts down the resulting energy gradient; by the
envelope theorem the force is the direct coupling term
``F_i = −c·Σ_x load_i·κ·∇κ``, integrated overdamped ``dR_i/dt = μ·F_i``.

- **Two masses fall together and merge**, the fall *accelerating* as the
  screened force steepens — and with the coupling off (``c = 0``) they stay
  put, so the infall is the field, not drift.
- **Many masses accrete**: a random scatter clumps, pairs then clumps merging,
  the count of bound objects falling to one while total mass is conserved —
  bound structure forming out of the capacity field from first principles.

So the whole arc is dynamical, not just static: the generative gap makes
distinction; capacity crystallises it into stable, discrete forms (matter,
with ``m_i = m_g``); and those forms, gravitating through the very field whose
depletion is gravity, **fall together and grow into structure**.  The universe
the framework describes does not merely *contain* mass and gravity — it
*assembles* itself out of them.

## A complete gravitational dynamics

Overdamped forms only fall.  Give them **inertia** — ``M·d²R/dt² = F`` with
the same envelope κ-force, integrated symplectically
(`evolve_inertial`, `n3_orbital_gravity.py`) — and κ-gravity becomes a full
gravitational dynamics:

- **A Kepler-like family**: tuning the tangential speed sweeps from a radial
  plunge through bound elliptical and near-circular orbits to unbound escape.
- **Conserved energy**: the symplectic integrator holds ``T + F[κ]`` to ~0.1%
  over an orbit — real conservative motion.
- **Precession**: because the mediator is *screened*, the bound ellipse does
  not close — its perihelion advances (~+140° per orbit here), a rosette.
  Orbital precession is the direct dynamical fingerprint of the finite range
  ``√(D/r)`` — the finite-range analogue of (not identical to) GR's advance.
- **Virialisation**: a dissipative N-body cloud rings down to the virial
  relation ``2⟨T⟩ + ⟨W⟩ → 0``.

Bound orbits, escape, conserved energy, precession, and virial equilibrium all
emerge from the one screened force whose strength is the forms' own mass — the
capacity field is not a metaphor for gravity but a working, if screened,
gravitational dynamics.

## Structure against expansion

The last step toward cosmology (`n3_cosmic_structure.py`, `hubble_flow`,
`fof_groups`): put that dynamics in an **expanding background** — every mass
given the Hubble recession ``v = H·(r − r_centre)`` — and ask whether gravity
still assembles structure against the outflow.

- **Turnaround.**  Two receding masses decelerate, reach a maximum separation
  (turnaround radius, growing with ``H``: 10 → 23 lattice units here) and
  recollapse — below a critical expansion rate; above it they escape.  The
  spherical-collapse picture.
- **Suppression of structure.**  A cloud in Hubble flow collapses into a
  single bound halo at low ``H`` (100% of the mass in one group) but fragments
  and disperses as ``H`` rises (down to ~25%).  A faster-expanding background
  forms less structure — the defining feature of gravitational structure
  formation.

So the whole chain runs against a cosmological background too: the generative
gap makes distinction; capacity crystallises it into stable forms (matter);
those forms gravitate through the capacity field; and they **assemble into
bound structure — so long as gravity outpaces the expansion**.  It is a
*Newtonian, coasting-background* model (expansion as an initial velocity
field, no FLRW metric or dark energy), so it captures the competition and the
critical rate, not a quantitative cosmology — but the essential ingredient of
a universe that grows structure is there.

---

*See `The_Generative_Gap.md` for the `S = ΔC + κ·ΔI` program κ lives in, and
the README "Capacity as gravity" / "spectrum of stable forms" sections for the
figures and reproduction.*
