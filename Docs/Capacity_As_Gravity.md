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

---

*See `The_Generative_Gap.md` for the `S = ΔC + κ·ΔI` program κ lives in, and
the README "Capacity as gravity" / "spectrum of stable forms" sections for the
figures and reproduction.*
