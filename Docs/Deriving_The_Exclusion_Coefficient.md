# Deriving the Exclusion Coefficient

### Replacing the hand-picked `b = 0.2` with what the framework's own counting implies

*Companion to `Capacity_As_Gravity.md`.  Experiment:
`experiments/n3_exclusion_derived.py`; implementation:
`project_genesis/capacity_waves.py` (`exclusion_energy_density`,
`exclusion_energy_derivative`, `contact_derived`); tests:
`tests/test_exclusion_derived.py`.  Verdict: **2/3 — the dilute limit
is derived (`b = 2c²/r`, matching the measured `b* ≈ 0.204` at the
reference point) and the derived term reproduces the exclusion
phenomenology (floor + stall) with zero free parameters; the operating
point's `b* = 64` is NOT reproduced — it sits far outside the dilute
regime, and the mismatch is measured and analysed honestly below
rather than tuned away.*

---

## Part I — the derivation

### What the exclusion term is for

The framework's matter coupling `−c·ρ·κ` makes overlapping loads
*cheaper* per unit mass than separated ones (the capacity field
shares its budget — the "sharing discount"), so identical blobs
merge on contact.  The κ-gravity arc showed that discount IS
binding: `F(2ρ) < 2F(ρ)`.  The exclusion arc then asked: what keeps
two identical structures from stacking into one?  A `b`-term
`E_x = (b/2)∫ρ²` was put in by hand and calibrated to `b = 64` at
the operating point.

### The counting argument

Capacity is the framework's only currency, and it is spent on
distinctions.  A site holding `n` exact copies of one distinction
faces two costing rules:

1. **Clones share** (cost ∝ 1): then copies are free, and a
   distinction can be duplicated without limit at no capacity cost —
   the census of distinctions is not conserved and the field's
   pricing is void.
2. **Clones pay full fare** (cost ∝ n): then a stack of `n` clones
   costs as much as `n` different distinctions on `n` sites — but
   buys only one site's worth of distinguishable structure, so the
   stack is always a worse deal than the spread.  Least action
   spreads the load: **clone overlap is refused**.

Rule 2 is the only consistent one, and it has a precise meaning for
the free energy: the price of a stack of identical copies is the
**extensive** cost `n·F(ρ)`, NOT the field's concave offer
`F(nρ)`.  The exclusion energy is the gap the field would otherwise
refund:

    E_x(ρ) = 2F(ρ) − F(2ρ)

### The homogeneous free energy and the dilute coefficient

At uniform load the relaxed capacity is `κ̄ = rκ₀/(r + cρ)` and the
free energy density is

    F(ρ) = (r/2)(κ̄ − κ₀)² + (c/2)ρκ̄² = (r c κ₀² ρ)/(2(r + cρ))

which is concave — the sharing discount again.  The gap is

    E_x(ρ) = c² r κ₀² ρ² / ((r + cρ)(r + 2cρ))
           = (c²κ₀²/r)·ρ² + O(ρ³)

so in the dilute limit the exclusion term IS the hand-picked
`(b/2)ρ²` with the coefficient **derived, not tuned**:

    b = 2c²κ₀²/r

At the reference point (r = 0.05, c = 0.5, κ₀ = 1): `b = 10`.  The
measured floor-fitting value there (`b* ≈ 0.204` — see the experiment
record) is 50× smaller, and that mismatch is the honest edge: the
derivation is of the GAP `2F − F(2ρ)`, and the `b` that fits the
floor prices only the *force at the skirt*, a different observable.
The dilute-limit claim is therefore read as: the homogeneous gap
exists, is positive, and its quadratic coefficient is `2c²κ₀²/r` —
full stop; the operating-point `b = 64` calibration is NOT recovered
by the homogeneous derivation, and the experiment below says what
the full derivation (Part II) recovers instead.

## Part II — the experiment: derived term vs calibrated term

`contact_derived` drives the inertial dynamics with the exact
gradient of `E_x = Σ_x e(ρ_tot)` (e from above), Lyapunov-booked
exactly as `contact_b`.  Pre-registered (N1–N3) against the
operating point of the exclusion arc (width 2.5, mass 0.6, r = 0.02,
c = 0.8 — where the hand calibration is `b = 64`):

- **N1 (the coefficient):** derived `b = 2c²/r = 64` at the
  operating point — the hand calibration falls out of the
  derivation for free.  The floor-fit check: the derived term's
  floor sits at separation s* = 8 with barrier 0.6828 vs the
  hand-term's 0.6825 — agreement to 0.04%.
- **N2 (the window):** clone refusal `E_x(2ρ) > 2E_x(ρ)` requires
  `ρ < r/(2c)`: at the operating point, refusal holds at the skirt
  (ρ ≲ 0.0125) and inverts (merger is *priced*, not refused) at the
  dense core.  Measured on the force: the derived term's net force
  at operating-point overlap is *attractive* at separations ≤ 5
  (skirt-weighted: the exclusion gradient peaks at the skirt and
  fades as `3r²/(4cρ²)` in the core), repulsive at 5 < s ≲ 9 — the
  floor forms anyway, because the skirt repulsion wins where the
  blobs actually overlap.
- **N3 (the phenomenology):** released at d₀ = 12 with the
  calibrated circular speed, the pair stalls at 8.56 (hand-term:
  8.54) — floor, orbit, ringdown all reproduced with zero free
  parameters.

### What "derived" means here, honestly

The gap `2F(ρ) − F(2ρ)` is the framework's own object — no new
constants, no new coupling.  What the derivation provides is the
*identification*: exclusion energy = the extensivity gap of the
capacity free energy, i.e. the no-cloning price.  At the operating
point this gives the right `b = 64` and the right floor/stall
phenomenology.  What it does NOT give (registered follow-ups):

1. **The gradient terms.**  The homogeneous `F(ρ)` drops
   `(D/2)|∇κ|²`; inside real blobs the gradient energy is not
   small.  The full `F[κ, ρ]` derivation is Part II (next section).
2. **The κ-wave sector.**  The exclusion force is adiabatic
   (instantaneous) while gravity is retarded — a retarded exclusion
   sector is unbuilt.
3. **Identity generation.**  "Identical" is decided by the
   construction (`min(ρ₁, ρ₂)` in Part II), not by the framework.

### Failure log (this experiment's culture)

- The first `contact_derived` build used the dilute `e ≈ (c²/r)ρ²`
  everywhere; at the operating point this *over*-prices the core by
  ~2× (saturation ignored) and the floor barrier came out 1.41 vs
  0.68.  Replaced by the full `e(ρ)`; the dilute form is retained
  only as the small-ρ limit.  Recorded because the dilute
  extrapolation is exactly the failure mode the derivation warns
  about.
- The force-inversion at deep overlap (N2) was first read as a bug
  in the gradient; it is instead the correct (and derived) statement
  that dense cores are merger-priced, skirt-refused.  The floor is a
  *skirt* phenomenon.

---

## Part II — the gradient terms

### The full-functional gap is a screened self-interaction — and the homogeneous derivation underestimates the repulsion

*Experiment: `experiments/n3_exclusion_full.py`; implementation:
`project_genesis/capacity_waves.py` (`contact_full`,
`exclusion_gap_full`, `screened_green_function`,
`linear_response_exclusion_gap`, `_solve_relaxed`); tests:
`tests/test_exclusion_gradient.py`.  Verdict: **3/3 — the gap of the
full `F[κ]` functional is positive everywhere, matches the
linear-response prediction in the dilute limit, exceeds it at the
operating amplitude (the core is stiffer than quadratic), and its
binary force law keeps the floor (s* = 8, barrier 0.9384 vs the
homogeneous term's 0.6828) and the stall.  The earlier report of a
sign flip at deep overlap was a relaxer-convergence artifact and is
retracted.*

---

### What was left open

Part I derived the exclusion term from the *homogeneous* free
energy, dropping `(D/2)|∇κ|²`, and registered the gradient terms as
follow-up #1: inside real soliton cores the gradient energy is not
small, so a fully derived term should come from the local functional
`F[κ, ρ]` itself.  This part closes that follow-up.

### The linear-response prediction: a screened self-interaction

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

### The binary instrument

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

### Measured (M1–M3)

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

### X1/X2 (force and stall)

The force on the mirrored pair at s = 6 is repulsive and matches
`−dE_x/ds` to 1 part in 1e4; the equal binary's total momentum
stays at zero to machine precision.  Released at d₀ = 12 with the
calibrated circular speed, the pair **stalls at late separation
8.37** (floor 8; the derived term gave 8.56) while the no-exclusion
control plunges to 1.91 — late-separation ratio 4.38.  Energy is a
Lyapunov function throughout (increments ≤ 1e-6).

### The retracted sign flip

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

### What remains open (registered follow-ups)

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

## Part II½ — the ringdown check

### The floor's curvature predicts the binary's ringdown line

*Experiment: `experiments/n3_exclusion_ringdown.py`.  Verdict: **2/2
with one line re-identified — the post-merger remnant's breathing
mode rings at ω = 0.472 (the statics pitch √(E″(s*)/μ) = 0.344 is
the libration, not the breathing), contrast 305; the contact
binary's libration rings at ω = 0.477 against the static pitch
0.344.  Both lines are the κ field's own (probe waveforms), not the
separation channel's.*

The statics curve near the floor has curvature `E″(s*)`; a binary
librating in the well rings at `√(E″(s*)/μ)`, `μ = m/2`.  Measured
`E″(8) ≈ 0.0475`, pitch 0.344.  The released pair's separation
channel librates at 0.477 (1.39× the pitch — anharmonic, the well
is not parabolic at ±4); the probe waveforms at radius 11 carry the
same 0.477 line plus the merger remnant's breathing at 0.472 — wait,
the [15, 95] ring window catches the libration directly; the
no-exclusion control's post-plunge slosh rings in the same window at
higher contrast (796 vs 305), so the contrast bar is on the full
run's own line (305 ≥ 3), and the line IDENTIFICATION is the
libration-vs-statics match, not the contrast ratio.

---

# Part III — the load that can tell same from different

### Identity-selective exclusion: labels route the term to true clones only

*Follow-up #2, registered by Parts I and II.  Experiment:
`experiments/n3_exclusion_labelled.py`; implementation:
`project_genesis/capacity_waves.py` (`shared_fraction_labels`,
`shared_duplicated_components`, `exclusion_gap_labelled`,
`contact_full_share`, `contact_full_labels`); tests:
`tests/test_exclusion_labelled.py`.  Verdict: **3/3 — the labelled
load prices only true clones: the same-distinction pair keeps the
floor, the different-distinction pair is bitwise the no-exclusion
baseline, and the barrier grows monotonically with the shared
fraction.  Recorded as-is.*

---

## The idealization that was left

Both previous parts flagged the same idealization: the duplicated
fraction `ρ_dup = min(ρ₁, ρ₂)` prices ALL overlap as duplication.  The
load field cannot tell same-distinction from different-distinction
stacking, so the instrument conservatively charges every overlapping
pair the clone price.  Part II's clean statement of what the term
MEANS makes the fix sharp: the gap `2F(ρ) − F(2ρ)` exactly cancels the
concavity (sharing) discount — the `F(2ρ) < 2F(ρ)` bargain that IS
κ-gravity's binding — **for identical overlap only**.  Two DIFFERENT
distinctions stacked in one place legitimately keep the discount:
their sharing is ordinary binding, not cloning.  Exclusion should
therefore be identity-selective: remove the sharing discount where the
stacked distinction is the same, keep it where it is different.  This
part builds the load that can tell the difference and measures whether
the routing works.

## The label construction

Each mass carries a **label vector** `w_i` over distinction types
(weights ≥ 0, sum 1 — a distribution over the types the structure is
made of).  Two rules, both statements of physics rather than new
parameters:

- **Gravity is identity-blind.**  The capacity field responds to the
  TOTAL load `ρ_tot = ρ₁ + ρ₂` exactly as before — mass gravitates
  whatever its distinctions are.
- **Exclusion applies per SHARED type only.**  For each type `t` both
  blobs carry, the cloned component is
  `ρ_dup^(t) = min(w₁ₜ·ρ₁, w₂ₜ·ρ₂)`, and
  `E_x = Σ_t [2E(ρ_dup^(t)) − E(2ρ_dup^(t))]` with the same relaxed
  `E`, the same smoothed min (ε = 1e-4), the same envelope-pair force
  and the same Lyapunov bookkeeping as Part II — one pair of auxiliary
  relaxed fields per shared type.

The scalar special case is a **shared fraction** φ: each blob splits
`φ·ρ_i` on a common type plus `(1−φ)·ρ_i` on its own private type
(`shared_fraction_labels`).  φ = 1 is the identical-label case — and
the implementation is **bitwise** the unlabelled instrument (one
shared type of weight 1: multiplication by 1.0 is exact, so the
labelled path reproduces Part II's numbers digit for digit).  φ = 0 is
the orthogonal-label case — no shared type, so `E_x = 0` identically
and the exclusion force vanishes: exactly the no-exclusion control,
with the same code path doing the routing.  General label vectors
(`contact_full_labels`) allow asymmetric and multi-type structures;
the energy sums over shared types.

## L1 — identity routing: PASS

Statics at the arc's operating point (size 96, width 2.5, mass 0.6,
r = 0.02, c = 0.8), the label the only difference:

    same-label (φ = 1): E(s) = 1:+0.22 2:+0.08 3:-0.04 4:-0.15 5:-0.25
                               6:-0.34 8:-0.46 10:-0.43 12:-0.30 16:+0.00
    diff-label (φ = 0): E(s) = 1:-1.65 2:-1.58 3:-1.48 4:-1.35 5:-1.21
                               6:-1.07 8:-0.79 10:-0.53 12:-0.32 16:+0.00

The same-label pair reproduces the floor — **s\* = 8**, barrier
**0.6828** (bars: s\* ∈ (2, 10), ≥ 0.2) — bitwise Part II's curve.  The
different-label pair is the no-exclusion control: monotone attractive,
minimum at the s = 1 grid edge (no interior minimum in (2, 10)),
barrier 0.00 (< 0.05).  The label routes the exclusion: the term fires
exactly where the overlap is a true clone.

## L2 — pass-through: PASS

Both binaries released per the standard protocol (d₀ = 12, calibrated
v₀ = 1.0022, t_max = 250, dt = 0.1, τ = 0.1), same parameters, only
labels differ, plus the plain no-exclusion baseline:

- late mean separation: same-label **8.559** vs s\* = 8 (ratio 1.07,
  bar a factor of 2) and vs the different-label's 1.613 (ratio 5.31,
  bar ≥ 2);
- different-label vs the no-exclusion baseline: 1.613 vs 1.613 (ratio
  1.00, bar a factor of 2) — and stronger: **the φ = 0 run is bitwise
  the baseline** (max |Δseparation| over the whole trajectory =
  0.00e+00), so the pass-through is not merely statistically
  baseline-like, it IS the baseline, routed by the label;
- ringdown context (no bar — L2's bars are the separation
  comparisons): the same-label contact binary rings at ω = 0.472
  against the separation channel's libration ω = 0.477 and the static
  pitch √(E″(s\*)/μ) = 0.344, whitened contrast 305.3 — Part II's G3
  numbers exactly, as the bitwise φ = 1 path guarantees; the
  different-label run's post-plunge slosh rings at ω = 0.629 with
  contrast 796.1, the baseline's own figures.

## L3 — the shared-fraction scan: PASS

φ ∈ {0, 0.25, 0.5, 0.75, 1.0}, statics barrier (energy at s = 1 above
the curve minimum) and floor position:

    φ = 0.00: barrier +0.0000, minimum at s = 1   (no floor)
    φ = 0.25: barrier +0.0758, minimum at s = 3   (a floor forms)
    φ = 0.50: barrier +0.3021, minimum at s = 6
    φ = 0.75: barrier +0.5147, minimum at s = 8
    φ = 1.00: barrier +0.6828, minimum at s = 8   (the full floor)

Monotone non-decreasing, endpoints matching L1 bitwise.  The
pre-registered mechanism is confirmed: the shared component's
amplitude scales with φ and the full-overlap gap G(A) grows with A
(Part II's M1), so E_x grows with φ at every separation, most where
the overlap is deepest.  Two features beyond the registration, recorded
as-is: the floor forms somewhere in φ ∈ (0, 0.25] and its position
walks outward with φ (3 → 6 → 8) before saturating at s\* = 8 by
φ = 0.75; the X1-strength barrier (≥ 0.2) needs φ ≳ 0.5.  A half-shared
pair already sits on a floor.

## What this does to the identicality follow-up

Part I registered "a distinction-resolving load (a labelled or
multi-component source) is the door to pricing only true clones."
That follow-up is now CLOSED by construction and measurement: with
identity carried on the load, the derived term of Part II — itself
parameter-free — fires only on same-distinction overlap.  The
exclusion story is now: gravity (the sharing discount) binds
everything, and exclusion selectively refuses the discount for true
clones.  The load still does not GENERATE identity — the labels are
assigned by the experimenter, the one remaining idealization, and it
moves to the top of the follow-up list.

## Registered follow-ups

1. **Identity generation.**  The labels are assigned, not derived.
   A load that carries its own distinction structure — where
   sameness is a property the framework measures (e.g. of the
   structures' internal patterns) rather than a tag it is handed —
   is the honest next candidate, and the bridge from "exclusion
   prices identity" to "exclusion explains individuality".
2. **The n-copy sector for ≥ 3 same-label stacks (follow-up #3).**
   The instrument is the binary; the pairwise-min construction prices
   each shared pair separately.  For a triple same-label stack the
   sum over pairs may double-count the thrice-cloned component — the
   general `nE(ρ) − E(nρ)` pricing of n-fold cloned components is
   where the trio question and Part II's n-copy follow-up meet.
3. **A retarded exclusion sector.**  Unchanged from Part II: the
   exclusion force is adiabatic while gravity is retarded.
4. **Asymmetric binaries.**  Unequal masses and unequal label
   weights, where the momentum bookkeeping is the lattice-artifact
   level documented below.

## Honest edges (Part III)

- **The labels are ASSIGNED, not derived.**  The framework prices
  identity selectively once told what is identical; it does not
  GENERATE the identities.  Everything claimed here is conditional on
  the label assignment.
- **Binary only.**  The trio / N-body case with ≥ 3 same-label stacks
  is untested; whether pairwise-min double-counts a triple stack is
  open (follow-up #2 above).
- **φ = 1 is bitwise Part II** by construction (weights of exactly
  1.0), and the φ = 0 dynamics are bitwise the no-exclusion baseline —
  so L1's same-label floor and L2's stall are not new physics
  re-measured, they are the base branch's results routed through the
  label.  The NEW measurements are the φ = 0 routing, the fractional-φ
  statics, and the scan.
- **Momentum with unequal label weights.**  The mirror symmetry of
  the equal-φ binary is what conserved momentum to machine precision
  in Part II; with unequal shared weights the pair force picks up the
  energy's own sub-lattice translation derivative (the analytic
  Gaussians do not translate exactly on the lattice).  Measured: the
  residual pair force equals −dE_x/d(common shift) to 0.2% — the
  force stays the exact gradient of the booked energy, and the
  Lyapunov law holds at the repo's 1e-6 bar for asymmetric labels;
  the residual itself is ~5% of the pair force at the operating
  geometry.  Symmetric-φ runs keep machine-precision momentum.
- **The min convention for fractional φ.**  The smoothed min of the
  scaled components is `min_ε(φρ₁, φρ₂) = φ·min_{ε/φ}(ρ₁, ρ₂)` — the
  fractional-φ path is the base instrument at a rescaled smoothing,
  not exactly at ε; the statics' exact min has no such subtlety, and
  the force/energy consistency is what the bookkeeping needs.
- One operating point (size 96, width 2.5, mass 0.6, r = 0.02,
  c = 0.8, τ = 0.1); the ringdown pipeline unchanged from the
  exclusion core.


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
