# Deriving The Exclusion Coefficient

*Why there is an exclusion term in the capacity framework at all, what
its coefficient is, and which parts of the story remain open.  This
document is the derivation companion to the exclusion experiments
(`experiments/n2_exclusion_gap.py`,
`experiments/n3_exclusion_contact.py`); the measured numbers quoted
below are those experiments' outputs at the commit this document
ships with.  It is written in the project's pre-registration culture:
distinguish what is derived, what is measured, and what is still
missing — and record failures with their mechanisms.*

---

# Part I — the homogeneous derivation

### The exclusion term is the extensivity gap of the capacity free energy — and its coefficient is the degeneracy stiffness

*Experiment: `experiments/n2_exclusion_gap.py`; implementation:
`project_genesis/capacity_waves.py` (`exclusion_energy_density`,
`contact_derived`); tests: `tests/test_exclusion_derived.py`.
Verdict: **3/3 — the derived term reproduces the calibrated
operating point, keeps the static floor, and keeps the stall.*

---

## What was left open

The preceding exclusion work ("quantization from classical capacity
fields") established the *phenomenology*: two solitons that would
otherwise merge into the lower-energy shared state are kept apart by
a short-range repulsion with a flat floor, and the culture named the
obvious candidate mechanism — **exclusion**: duplicating a
distinction in place costs capacity, so identical structures
selectively refuse to stack.  But the term used there,
`E_x = (b/2)∫ρ²`, had its coefficient **calibrated by hand**
(b = 64) to keep two operating-point solitons apart.  Left open:

1. **Is the exclusion principle true in this framework at all** —
   does stacking identical distinctions actually cost capacity, as
   opposed to merely being penalized by a term we added?
2. **What is the coefficient** — can `b` be derived from `(D, r, c)`
   rather than tuned?
3. **What is the term's functional form** — is `(b/2)ρ²` the right
   density dependence, or just the dilute limit?

## Why stacking clones must cost capacity: the counting argument

The framework's one dynamical resource is capacity: each site's
budget `κ` is spent maintaining the distinctions present at that
site, at unit cost `c` per distinction (`∂_t κ` has `−c·load·κ`;
the free energy has `(c/2)·load·κ²`).  Consider a site holding `n`
*fully identical* copies of the same distinction — same pattern,
same phase, same everything:

- **If copies share a budget** (`cost ∝ 1` regardless of `n`): the
  site is over-counted — `n` distinctions, one budget line.  Copying
  is free, distinctions multiply without bound, and the census that
  defines "distinction" diverges.  A framework that prices
  distinctions by capacity cannot allow this: unpriced copies
  dissolve the pricing.
- **If each copy pays full fare** (`cost ∝ n`): then a stack of `n`
  clones costs the same as `n` genuinely different distinctions
  spread over `n` sites — *but with none of the entropy*: the `n`
  clones occupy one site's worth of configuration space, not `n`.
  Per unit of distinguishable structure gained, the clone stack is
  strictly more expensive.  A least-action field relaxes toward the
  spread configuration — **clone overlap is refused**.

So the counting argument forces: full fare per copy (linearity), and
therefore a *disadvantage* for clones relative to the concave
sharing discount the field would otherwise give them.  The remaining
question is whether the framework's own free energy already contains
exactly that disadvantage — it does, and it is the *only* term in it
that distinguishes clones from non-clones.

## The derivation: exclusion energy = 2F(ρ) − F(2ρ)

The homogeneous steady-state capacity free energy density at uniform
load `ρ` (baseline `κ₀ = 1`, gradients neglected) is

    F(ρ) = (r/2)(κ̄ − 1)² + (c/2)ρκ̄² = r·c·ρ / (2(r + cρ)) ,
    κ̄(ρ) = r/(r + cρ) ,

which is **concave** in ρ: `F(2ρ) < 2F(ρ)` for all `ρ > 0`.  That
concavity *is* the sharing discount — it is exactly why two
solitons merge (one doubly-loaded site is cheaper than two singly
loaded ones; see the κ-gravity arc, where the same concavity is the
binding mechanism).

Now apply the exclusion principle as the framework states it:
*duplicating a distinction in place costs capacity*.  That is, a
stack of two identical copies is priced at the **extensive** cost
`2F(ρ)` — each copy pays full fare, per the counting argument —
**not** at the concave cost `F(2ρ)` that the field would charge two
*independent* loads sharing one site.  The **exclusion energy** is
the gap between what the principle charges and what the field alone
would charge:

    E_x(ρ) = 2F(ρ) − F(2ρ)
           = c²rρ² / ((r + cρ)(r + 2cρ))  per site .

Three readings of the same formula:

- **As a no-cloning tax.**  The gap is what the concavity would
  have refunded for stacking; exclusion claws back exactly the
  refund, for identical stacks only.  Non-identical distinctions
  legitimately keep the discount — that discount is binding.
- **As the only clone-sensitive term.**  Every other term in the
  framework's energy (gradient cost, recovery, consumption) is a
  functional of the *total* load and cannot tell `ρ + ρ` (two
  clones) from `ρ₁ + ρ₂` with `ρ₁ + ρ₂ = 2ρ` (two non-clones).
  The extensivity gap is the unique object in the theory that
  vanishes for separated structure and is positive exactly where
  a distinction is *duplicated* in place.
- **As a degeneracy pressure.**  The dense limit `ρ → ∞` saturates
  the per-site exclusion energy at `r/2` — the recovery supply
  rate — which is precisely the flat, contact-independent core the
  soliton scattering measurements showed.

## The coefficient: b = 2c²/r, and its density dependence

In the dilute limit `cρ ≪ r` the gap closes to the quadratic form
the earlier experiments used,

    E_x(ρ) ≈ (c²/r)·ρ²  =  (b/2)·ρ² ,   b = 2c²/r ,

so the hand-calibrated `b` is derived: **the exclusion coefficient
is the degeneracy stiffness** `2c²/r` — the square of the
per-distinction capacity price over the recovery rate that
 replenishes it.  Note `c²/r = 1/(2ℓ²)` with `ℓ² = D/(2r)` …
equivalently, using the loaded screening length `ℓ²(ρ) = D/(r + cρ)`
of the κ-field (the gravity arc's Debye term),

    b(ρ) = 2c²ℓ²(ρ)/D ,

i.e. *the degeneracy stiffness is set by the local capacity range*:
the farther the field reaches (`ℓ`), the more a site's budget is
mortgaged to its neighborhood, and the stiffer the refusal to clone.
The full density dependence `e(ρ) = 2F(ρ) − F(2ρ)` (not its dilute
limit) is the derived term; `e'(ρ)` — the force density — peaks
inside the refusal window and fades as `3r²/(4cρ²)` in the saturated
regime, which is why the net force on a dense blob is skirt-weighted
(and why the constant-b force *inverts* to attraction inside dense
cores — see Part II of the original note, reproduced below, for that
measured inversion and its mechanism).

## What the derived term predicts at the operating point

The exclusion arc's operating point (`r = 0.02, c = 0.8, width 2.5,
mass 0.6`) gives `b = 64` — *the calibrated value falls out of the
derivation with no tuning*.  With `contact_derived` the
density-dependent derived term drives the inertial dynamics with
exact-gradient forces and exact Lyapunov bookkeeping
(`energy = T + F[κ] + ∫(τ/2)κ̇² + ∫e(ρ)`,
`dE/dt = −∫κ̇² ≤ 0`).  Measured against the hand-term:

- **N1 (floor):** interior minimum at separation s* = 8, barrier
  0.6828 vs the hand-term's 0.6825 — the derived term keeps the
  static floor, with a +0.01% softer repulsion at the floor
  separation (the local `b(ρ) = 2E_x/ρ²` runs 64 → ~63 there; the
  densities in the overlap skirt sit just inside the refusal
  window).
- **N2 (window):** the clone-refusal window `ρ < r/(2c)` maps to
  overlap separations `s ≲ 7.4` at the operating width — i.e. the
  *whole contact side* of the floor.  A Gaussian blob that must
  shed peak density to merge pays to flatten first; the window is
  the mechanism of the core.
- **N3 (stall):** released at d₀ = 12 with the calibrated circular
  speed, the pair stalls at late separation 8.56 vs the hand-term's
  8.54 (floor 8) — the same orbit-and-hold, and the same merger
  barrier story at higher energies is expected (not re-measured).

## What remains open (registered follow-ups)

1. **The gradient terms.**  The derivation drops `(D/2)|∇κ|²` — the
   homogeneous limit.  Inside real soliton cores the gradient energy
   is *not* small; a fully derived `e(ρ, ∇ρ)` should re-derive the
   term from the local free-energy functional `F[κ, ρ]` rather than
   its homogeneous part.  This is the honest next derivation.
2. **The κ-wave sector.**  The exclusion force here is *adiabatic*
   (instantaneous), while gravity in the wave arc is retarded.  A
   retarded exclusion sector (the gap as a field, not a bookkeeping
   term) is unbuilt.
3. **Identity generation.**  "Identical distinction" is assumed
   recognizable; what makes two loads the *same* distinction (a
   pattern-matching dynamics) is outside the current framework.

---

# Part II — the gradient terms

### The full-functional gap is a screened self-interaction — and the homogeneous derivation underestimates the repulsion

*Experiment: `experiments/n3_exclusion_full.py`; implementation:
`project_genesis/capacity_waves.py` (`contact_full`,
`exclusion_gap_full`, `screened_green_function`,
`linear_response_exclusion_gap`, `_solve_relaxed`); tests:
`tests/test_exclusion_gradient.py`.  Verdict: **3/3 — the gradient
terms strengthen exclusion (gap ratio 1.59 at the operating
amplitude), the full instrument keeps the static floor (s* = 8,
barrier 0.9384 vs the derived term's 0.6828), and the released pair
stalls on the floor while the no-exclusion control plunges through
it.  The earlier report of a sign flip at deep overlap was a
relaxer-convergence artifact and is retracted.*

---

## What was left open

Part I derived the exclusion term from the *homogeneous* free
energy, dropping `(D/2)|∇κ|²`, and registered the gradient terms as
follow-up #1: inside real soliton cores the gradient energy is not
small, so a fully derived term should come from the local functional
`F[κ, ρ]` itself.  This part closes that follow-up.

## The linear-response prediction: a screened self-interaction

With gradients kept, the relaxed minimum of the capacity free energy
at fixed load ρ is, in linear response (`cρκ₀ ≪ r`),

    E(ρ) = (cκ₀²/2)·∫ρ  −  (c²κ₀²/2)·(ρ, G_r ρ)  +  O(c³) ,

where `(r − D∇²)G_r = δ` is the screened (Yukawa) kernel of range
`ξ = √(D/r)` — *the same kernel that mediates κ-gravity*.  The
full-overlap gap of two identical copies is therefore

    G(ρ₁) := 2E(ρ₁) − E(2ρ₁) = c²κ₀²·(ρ₁, G_r ρ₁)  >  0 ,

a positive, **nonlocal** self-interaction: the exclusion energy of a
cloned blob is its self-interaction through the screened kernel.
Exactly (beyond linear response), `E(A) = min_κ F[κ; A·ρ₁]` is a
minimum over functions *affine* in `A`, hence **concave** in `A` —
so the gap `G(A) = 2E(A) − E(2A)` is non-negative at every
amplitude: *no sign flip of the total gap is possible*.  The
homogeneous `∫e(ρ)` of Part I is the local (contact)
approximation: the Dirac kernel replacing the screened one.

## The binary instrument

The force law applies the gap to the **duplicated fraction** of the
pair — `ρ_dup = min(ρ₁, ρ₂)`, which for identical copies is exactly
the cloned component:

    E_x = 2E(ρ_dup) − E(2ρ_dup) ,

computed with the repo's own instruments (`relax_capacity` +
`capacity_free_energy`) for statics, and in the dynamics with the
two auxiliary relaxed fields solved directly (conjugate gradient on
the relaxer's linear fixed point — same fixed point, ~50× faster).
The force is the exact gradient by the envelope theorem:
`δE/δρ = (c/2)κ̄²` at the relaxed field, so

    F_i = Σ_x c·(κ̄[ρ_dup]² − κ̄[2ρ_dup]²)·(∂ρ_dup/∂ρ_i)·∇ρ_i ,

with analytic load gradients and a *smoothed* min (ε = 1e-4 — the
hard `min + indicator` rule mis-splits the symmetry tie-plane that
lands on lattice sites for an equal binary, breaking Newton's third
law by up to ~35%; the smoothed form conserves momentum to machine
precision and matches finite-difference E_x to 1 part in 1e4).
`E_x` is booked in the Lyapunov energy through the functional the
relaxer actually descends (`_relax_functional`).

## Measured (M1–M3)

- **M1 (gap curve):** the full gap exceeds the linear-response value
  at the operating amplitude — ratio **1.5886** at A = 0.6
  (1.000 at A = 0.001): the core's response is *stiffer* than
  quadratic.  And it exceeds the homogeneous Part-I term everywhere
  — ratio **1.12 at s = 6** on the split pair's duplicated fraction:
  the gradient energy of the overlapped core is priced in, i.e.
  *the homogeneous derivation underestimates the repulsion*.
- **M2 (relaxer validation):** the CG fixed point matches
  `relax_capacity` to 3.4e-10 max|Δκ| at the operating amplitude;
  the measured G(A)/G_LR ≈ 1 at A = 0.001 (0.5% — the instrument is
  self-consistent in the linear regime).
- **M3 (statics):** with the full term the pair's interior minimum
  stays at **s* = 8**, barrier **0.9384** vs the derived term's
  0.6825/0.6828: same floor position, ~38% stronger barrier — the
  gradient correction deepens the well without moving it.

## X1/X2 (force and stall)

The force on the mirrored pair at s = 6 is repulsive and matches
`−dE_x/ds` to 1 part in 1e4; the equal binary's total momentum
stays at zero to machine precision.  Released at d₀ = 12 with the
calibrated circular speed, the pair **stalls at late separation
8.37** (floor 8; the derived term gave 8.56) while the no-exclusion
control plunges to 1.91 — late-separation ratio 4.38.  Energy is a
Lyapunov function throughout (increments ≤ 1e-6).

## The retracted sign flip

An earlier draft of this experiment reported that the gap *flips
sign* at deep overlap (s ≤ 3), which would have made exclusion
attractive inside cores — a cliffhanger.  Reproduction found the
"flip" was a **relaxer-convergence artifact**: at deep overlap the
relaxed fields are sharply peaked, the relaxer's tolerance was met
in name only, and the gap (a small difference of large energies)
inherited the error.  At proper tolerance the gap stays positive
everywhere — as the concavity argument above *requires*: E is
concave in amplitude, so G ≥ 0 at every amplitude.  The barrier
toward contact (0.9384 at the floor) is real, finite, and *not*
infinite: deep enough overlap still wins against it, which is why
high-energy mergers still happen.  Failure recorded with its
mechanism; the artifact, not the physics, was the story.

## What remains open (registered follow-ups)

1. **The κ-wave / retarded exclusion sector.**  The exclusion force
   here is adiabatic (the two auxiliary fields are *solved*, not
   co-evolved), while gravity in the wave arc is retarded.  A
   retarded exclusion sector — the gap as a field degree of freedom
   with its own dynamics — is unbuilt.
2. **Beyond the binary.**  `ρ_dup = min(ρ₁, ρ₂)` is defined for
   pairs; the n-fold clone stack's gap `nE(ρ) − E(nρ)` generalizes
   it, but the *force* for n > 2 (which component repels which)
   needs the same envelope treatment per pair, and the n-body
   experiment is undone.
3. **Identity generation.**  Unchanged from Part I: "identical" is
   assumed recognizable by the min construction; a dynamics that
   *decides* sameness is outside the framework.

---

# Part III — the labelled load

### The load that can tell same from different: exclusion prices only shared distinctions

*Follow-up #2, registered by Parts I and II.  Experiment:
`experiments/n3_exclusion_labelled.py`; implementation:
`project_genesis/capacity_waves.py` (`shared_fraction_labels`,
`shared_duplicated_components`, `exclusion_gap_labelled`, the
`contact_full_share` / `contact_full_labels` options); tests:
`tests/test_exclusion_labelled.py`.  Verdict: **3/3 — identical
labels stall (8.53 vs the unlabelled 8.37), orthogonal labels plunge
through the floor (2.40 vs the control's 1.91), and a 50% hybrid
stalls between (4.14), in the registered ordering.*

---

## What was left open

`min(ρ₁, ρ₂)` prices **all** overlap as duplication: the load field
cannot tell same-distinction from different-distinction stacking.
But the gap `2F(ρ) − F(2ρ)` exactly cancels the concavity (sharing)
discount — *for identical overlap only*.  Different distinctions
legitimately keep the discount: that discount **is** binding (the
κ-gravity arc's whole mechanism).  Parts I–III register the gap:
exclusion should apply per *shared* distinction type, and the
no-cloning story needs a load that carries identity.

## The labelled load

Each mass carries a **label vector** `w_i` over distinction types
(weights ≥ 0, sum 1 — a distribution).  The capacity field responds
to the **total** load as before — gravity stays identity-blind —
while exclusion applies per **shared type** only:

    ρ_dup^(t) = min(w₁ₜ·ρ₁, w₂ₜ·ρ₂) ,
    E_x = Σ_t [2E(ρ_dup^(t)) − E(2ρ_dup^(t))] .

The scalar special case is the **shared fraction** φ: each blob
splits `φ·ρ_i` common-type + `(1−φ)·ρ_i` private-type.  The limits
are exact: **φ = 1** is the unlabelled `contact_full` (bitwise the
same path — one shared type of weight 1); **φ = 0** is the
no-exclusion control (no shared type — E_x = 0 and zero exclusion
force, identically).  The force stays the exact gradient of the
recorded E_x per shared type — same envelope pair, same smoothed
min, same Lyapunov bookkeeping — each shared type contributes its
own two auxiliary relaxed fields.

## Measured (L1–L3, pre-registered in the experiment's docstring)

- **L1 (statics):** the floor depth scales with the shared fraction
  and the floor does not move: barrier 0.9384 (φ = 1) → 0.5968
  (φ = 0.5) → 0 (φ = 0, monotone attraction) at s* = 8 throughout;
  the E_x curve is exactly the φ = 1 curve times φ² (the cloned
  component of each blob is φ·ρ_i, so the quadratic gap scales
  φ² — measured max deviation 1e-12).
- **L2 (dynamics):** released at d₀ = 12 with the calibrated
  circular speed: identical labels stall at **8.53** (the
  unlabelled full term: 8.37), orthogonal labels plunge to **2.40**
  (the no-exclusion control: 1.91), the 50% hybrid stalls at
  **4.14** — and the ordering lands in the registered sequence
  (identical > hybrid > orthogonal, with orthogonal ≈ control).
- **L3 (selectivity):** same-type pairs feel the full exclusion,
  different-type pairs feel none, and a general label pair's force
  is the exact gradient of its booked E_x (finite-difference check
  to 5%, the sub-lattice translation artifact accounted for).

## What the hybrid's shallow stall means

The 50% hybrid stalls at 4.14 — *inside* the shared core: it keeps
half the sharing discount, so its effective floor sits at contact
side of the full floor and its barrier (0.60 of the full 0.94) is
spent earlier in the plunge.  This is the framework's first
**graded identity**: same / different / 50%-same are three
macroscopically distinct binding outcomes from one knob.  Read
honestly: the labels are *assigned*, not derived — the framework
still cannot generate identity (follow-up #3 below), but it can now
*price* it selectively, which was the point of this part.

## What remains open (registered follow-ups)

1. **Identity generation.**  The labels are assigned by hand.  A
   load that carries its own distinction structure — a
   pattern-matching dynamics that *decides* sameness — is the
   framework's honest next candidate and remains outside it.
2. **The n-copy sector with labels.**  Part II's n-fold follow-up,
   one level up: for a triple same-label stack, does the pairwise
   `min` overcharge (three pairs for one triple clone)?  The
   n-copy gap `nE(ρ) − E(nρ)` is the same question again.
3. **A retarded exclusion sector.**  Unchanged from Part II: the
   exclusion force is adiabatic while gravity is retarded.

## Honest edges (Part III)

- The label vector is a *bookkeeping* device: it routes the
  exclusion term; it is not (yet) a dynamical degree of freedom —
  masses do not exchange or evolve labels.
- The φ² scaling of the statics is exact for the *gap*, but the
  dynamics at intermediate φ are not a simple interpolation of the
  two limits (the hybrid's 4.14 is not the mean of 8.53 and 2.40):
  the barrier spends against the plunge energy nonlinearly.
- The same sub-lattice translation artifact from Part II appears in
  the force/energy check for unequal label weights (mirror symmetry
  broken): the residual pair force equals `−dE_x/d(common shift)`
  to 0.2%, i.e. it is the lattice's, not the term's.


# Part IV — the n-copy sector

### Three-and-more identical stacks: the pairwise sum and the true n-copy gap are ONE at O(c²) — and their split prices the core's saturation

*Follow-up #3, registered by Parts I–III.  Experiment:
`experiments/n3_exclusion_ncopy.py`; implementation:
`project_genesis/capacity_waves.py` (`group_duplicated_component`,
`exclusion_gap_group`, `_smoothed_min_group`, the `contact_full`
path's n-mass generalization with `contact_full_ncopy`); tests:
`tests/test_exclusion_ncopy.py`.  Verdict: **3/3 — the O(c²) theorem
holds at the dilute check (pairwise vs n-copy within 5%), the trimer
has a statics floor (s\*₃ = 8, barrier 2.06), and the released trimer
stalls on it while the no-exclusion baseline plunges to three-way
contact.  Recorded as-is.*

---

## What was left open

Every part so far probed the 2-copy sector only, and each registered
the same gap: for a triple same-label stack the pairwise-min
construction prices each shared PAIR separately, and whether that
double-counts or under-prices the thrice-cloned component was open.
The general form, flagged since Part I, is the n-copy gap
`nE(ρ) − E(nρ)`.  This part builds it and measures the trio.

## The theorem: pairwise = n-copy at O(c²)

In linear response the relaxed minimum of the capacity free energy at
fixed load is, relative to the vacuum constant,
`E(A·m̂) = −k·A² + O(c³)` with `k = (c²κ₀²/2)(m̂, G_r m̂)` (Part II).
The vacuum constant and the linear term cancel in any gap combination,
so both candidate exclusions are pure `k`:

- **pairwise sum**: `C(n,2)` pairs, each
  `2E(m) − E(2m) = −2k + 4k = 2k` → total `2k·n(n−1)/2 = k·n(n−1)`;
- **n-copy gap**: `nE(m) − E(nm) = −nk + n²k = k·n(n−1)`.

Equal — one line each.  So the pairwise-summed exclusion and the true
n-copy gap are the SAME term at O(c²); any difference at finite
amplitude is a pure core-saturation effect.  This also settles the
Part-I copy-vs-stack density convention note: the min construction
already applies the gap at the copy density.

One more exact statement, with the same concavity that made the gap
non-negative (Part II: `E` is a minimum over functions affine in the
amplitude, hence concave, `E(0) = 0`): the chords give
`E(A) ≥ E(3A)/3` and `E(2A) ≥ 2E(3A)/3`, so for the triple

    pw − nc = 3E(A) − 3E(2A) + E(3A) ≥ 0 ,

i.e. **the pairwise sum is the concavity upper bound on the n-copy
gap**, with equality exactly where `E` is quadratic.  The split is a
signed measure of how far the core is from linear response.

## The instrument

The `contact_full` path groups the masses by SHARED distinction type
(every type carried by ≥ 2 masses with positive weight; singletons
price nothing — Part III's routing with n > 2 present).  Within a
group of n, the cloned component is the min over the group's
shared-type loads and

    E_x = n·E(m) − E(n·m) ,   δE_x/δm = (n·c/2)(κ̄[m]² − κ̄[n·m]²) ,

the same envelope theorem, the same CG-solved auxiliary relaxed fields
(two per group), the same `_relax_functional` Lyapunov bookkeeping.
The pairwise-sum form is kept as an option
(`contact_full_ncopy=False`, two fields per pair) so the experiment
compares the forms on identical configurations.  **For n = 2 both
forms reduce to the base branch's pair form bitwise** — the
regression is pinned to the base branch's own digits (statics
0.73159256758897584, booked E_x 0.73120365796494458, force kicks
reproduced to the last bit) in `tests/test_exclusion_ncopy.py`.

Design decision, measured and recorded: the group's min needs an
n-ary smoothing.  The choice is the **symmetrized iteration** of the
pair's smooth-abs min (`m = (1/n)Σ_i smin(m_{¬i}, s_i)`, recursively):
permutation-symmetric (the equilateral trimer's exact tie point
splits 1/3-1/3-1/3), error ≤ (n−1)ε below the hard min, weights the
exact partials summing to 1 — and at n = 2 it IS the pair instrument's
smoothed min, digit for digit, which log-sum-exp is not.  That
bitwise reduction is why it was chosen; any other smoothing gives the
same physics to O((n−1)ε).  Its cost grows like n!/2 leaf
evaluations — built for small groups; the pairwise option prices
C(n, 2) pairs instead when n is large.

## N1 — the theorem: PASS

The fully-stacked triple (three identical copies at one centre) at
the arc's own instruments (relax_capacity, exact min, 96²):

    A = 0.01 (dilute):    pairwise 0.0381542 , n-copy 0.0365265 ,
                          ratio 1.0446   (bar: within 5%)
    A = 0.6  (operating): pairwise 6.21113   , n-copy 4.41116   ,
                          ratio 1.4080   (recorded as-is)

The dilute ratio lands inside the 5% bar.  Against the O(c²) value
`3c²κ₀²(ρ, G_r ρ) = 0.0434952`: pw/LR = 0.877, nc/LR = 0.840 — both
forms sit ~12–16% under the quadratic prediction at A = 0.01, the
higher-order remainder at that amplitude (at A = 0.001 the ratio is
1.005 and pw/LR = 0.996 — the theorem's limit, measured during
development).  At the operating amplitude the split is the registered
expectation: the pairwise form prices the thrice-cloned core three
times over, 1.41× the n-copy gap, and both sit at 2.8–4.0% of the
linear-response value (156.583) — Part II's deeply saturated core
again.  The divergence from 1.0446 to 1.4080 between the two
amplitudes IS the saturation, measured with no free parameters.

## N2 — trimer statics floor: PASS

Three same-label masses on the equilateral triangle of side s,
E(s) zeroed at the far side, both forms plus the no-exclusion control
(the identity-blind field curve alone):

    n-copy   E(s) = 1:+0.54 2:+0.03 3:-0.43 4:-0.83 5:-1.17 6:-1.42
                    8:-1.52 10:-1.20 12:-0.75 16:+0.00
    pairwise E(s) = 1:+2.33 2:+1.80 3:+1.30 4:+0.80 5:+0.32 6:-0.13
                    8:-0.79 10:-0.95 12:-0.70 16:+0.00
    control  E(s) = 1:-3.27 2:-3.16 3:-3.00 4:-2.78 5:-2.54 6:-2.29
                    8:-1.75 10:-1.23 12:-0.75 16:+0.00

The n-copy form has its interior minimum at **s\*₃ = 8** with barrier
**2.06** toward contact (bars: s\*₃ ∈ (2, 12), ≥ 0.2); the control is
monotone attractive with no interior minimum.  The pairwise form's
floor (comparison, no bar): **s\*₃ = 10, barrier 3.29**.  Two readings,
both recorded as-is: the n-copy trimer floor sits at the SAME side as
the binary floor (s\* = 8) with ~3× the barrier (0.6828 → 2.06) —
three bodies bind harder, the floor does not move; the pairwise form,
over-pricing the triple overlap per the concavity bound, holds the
trimer farther out and harder still.

## N3 — trimer dynamics: PASS

Released from rest at side d₀ = 12 (no angular momentum — the
breathing channel).  The baseline was measured first: its three-way
collapse completes at t ≈ 8.0, far inside the arc's t_max = 250 (late
window t > 150) — the t_max choice is an instrument necessity,
recorded, not a tuned result.  Late mean pairwise separation:

    n-copy    6.78   vs s*₃ = 8 (ratio 0.85, bar a factor of 2)
                      vs baseline 1.87 (ratio 3.63, bar ≥ 1.5)
    pairwise  9.51   (recorded, no bar; vs its own s*₃ = 10: 0.95)
    baseline  1.87   (three-way contact, as registered)

The same-label trimer stalls on its static floor while the
no-exclusion baseline plunges.  The saturation split is visible in
the DYNAMICS too: the two forms hold the same trimer at 6.78 vs
9.51 — a 40% difference in stall separation, the core saturation made
measurable in positions.

**EXPLORATORY (NOT pre-registered — three-body spectra are not the
binary instrument's):** the n-copy run's breathing line rings at
ω = 0.636 in the separation channel vs the static breathing pitch
`√(E″(s*₃)/m)` = 0.422 (ratio 1.5 — recorded as-is, no bar; the
binary's statics-predicts-ringdown agreement is NOT claimed for the
trimer); the probe waveforms carry a line at ω = 0.315 with whitened
contrast 34.3.

## What the split measures

With the O(c²) theorem as the zero point, the pairwise/n-copy split
is a clean dial of the core's nonlinearity: 1.04 in the dilute stack,
1.41 in the operating stack's energy, 8 vs 10 in the floor position,
6.78 vs 9.51 in the stall separation.  The n-copy gap is the honest
pricing of an n-fold clone — the exclusion principle prices the stack
at the extensive cost of its content (n copies of m cost nE(m)), the
field's concavity charges only E(nm), and the gap is the degeneracy
debt; the pairwise sum is its quadratic upper bound, exact when the
core is linear.  No new parameters anywhere.

## What this does to the follow-up list

Part III's registered follow-up — the n-copy sector for ≥ 3
same-label stacks, where the pairwise-min construction might
double-count a triple stack — is now CLOSED: the general form exists,
reduces to the pair form bitwise at n = 2, answers the double-count
question by theorem at O(c²) (the pairwise sum does NOT over-count in
linear response — it equals the true gap; it over-prices only through
the measured saturation), and the trio works like the pair: floor,
stall, and a saturation dial.  The exclusion story is now complete
through the trio: gravity (the sharing discount) binds everything,
exclusion selectively refuses the discount for true clones, and n
clones pay the n-fold gap.

## Registered follow-ups

1. **The dilute operating point (follow-up #4, already specced).**
   Where the core stays linear the pairwise/n-copy split should
   vanish AND the homogeneous (Part I) and full-functional (Part II)
   derivations should agree — N1's dilute remainder (both forms
   ~12–16% under the O(c²) value at A = 0.01, 0.4–0.9% at A = 0.001)
   marks the edge of that regime.
2. **Identity generation.**  Unchanged from Part III: the labels are
   assigned, not derived; a load that carries its own distinction
   structure is the honest next candidate.
3. **A retarded exclusion sector.**  Unchanged from Part II: the
   exclusion force is adiabatic while gravity is retarded.
4. **Rotating trimers and n ≥ 4 stacks.**  The dynamics here are
   from rest — no angular momentum, the collapse is the breathing
   channel; and the symmetrized min's n!/2 cost makes large groups
   the pairwise option's territory.

## Honest edges (Part IV)

- **Three bodies only.**  The n-copy form is implemented and tested
  for general n, but the record (N1–N3) is the trimer; n ≥ 4 stacks
  are untested dynamics territory.
- **The min-over-group construction is the maximal-identicality
  convention again** — exact for identical copies, the maximal common
  component otherwise; non-identical stacks are conservatively
  charged the clone price.
- **Dynamics from rest only** — no angular momentum; the rotating
  trimer is follow-up territory, and the breathing-mode exploratory
  line (ω = 0.636 vs the static pitch 0.422) is recorded without a
  bar, not as a ringdown prediction.
- **The n-ary smoothed min is A convention.**  The symmetrized
  iteration was chosen because it is bitwise the pair instrument at
  n = 2 (the regression guard), not because nature picked it; any
  smoothing gives the same physics to O((n−1)ε = 2e-4 here).
- **Momentum bookkeeping** is the pair's, one level up: the mirror
  channel of the isosceles trimer conserves momentum to machine
  precision (the symmetrized min splits the tie exactly), and the
  common-translation channel equals −dE_x/d(shift) to 0.4% — the
  lattice artifact of Part III, unchanged.  A development bug made
  this visible: pair terms must write to their global mass indices
  (masked whenever the pair is masses (0, 1)); the (b, a, a)
  label arrangement guards it in the tests.
- One operating point (size 96, width 2.5, mass 0.6, r = 0.02,
  c = 0.8, τ = 0.1); the statics on relax_capacity with the exact
  min, the dynamics on the symmetrized smoothed min (ε = 1e-4) with
  conjugate-gradient solves, as the base branch.
