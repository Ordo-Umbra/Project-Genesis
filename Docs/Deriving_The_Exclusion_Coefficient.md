# Deriving the Exclusion Coefficient

### Replacing the hand-picked `b = 0.2` with what the framework's own counting implies

*Companion to `Capacity_As_Gravity.md`.  Experiment:
`experiments/n3_exclusion_derived.py`; implementation:
`project_genesis/capacity_waves.py` (`exclusion_energy_density`,
`exclusion_energy_derivative`, `contact_derived`); tests:
`tests/test_exclusion_derived.py`.  Verdict: **2/3 — the derivation is
real, and at the exclusion core's operating point it does NOT buy the
floor.  Recorded as-is.*

---

## The problem: the one free parameter

The exclusion core (`experiments/n3_exclusion_core.py`) landed its three
predictions — the static barrier, the contact-binary floor, the ringdown
at the statics' pitch — with a contact term

    E_x = (b/2)·∫ ρ_tot² ,      b = 0.2 chosen in static calibration.

That `b` was the last free parameter in the no-cloning story: a number
picked to put an interior minimum away from both walls, not a quantity
the framework produced.  This note derives the exclusion term from the
framework's own information counting and tests whether the floor and the
ringdown survive what the counting actually implies.  They do not — and
the reason is itself a result.

## The concavity problem

At fixed load ρ, the homogeneous steady state of the capacity field
(κ₀ = 1, gradient terms neglected) is `κ̄ = r/(r + cρ)`, and the capacity
free energy density there is

    F(ρ) = r·c·ρ / (2·(r + cρ)) .

`F` is **concave** in ρ.  Concavity is precisely why merging is cheap in
this framework: `F(2ρ) < 2F(ρ)`, so stacking two identical copies of a
structure costs less field energy than keeping them apart.  The binding
that becomes κ-gravity *is* this concavity.  It is also why the naive
no-cloning fix (cap the load) fails — capping ρ only makes merging
cheaper still.

## The extensivity argument

The URP exclusion principle says: *in a structure, adding the same
distinction does not expand the structure.*  Read as accounting, energy
should be extensive in **content**: two identical stacked copies carry
the distinction-content of one, so the honest price of the stack is the
extensive cost `2F(ρ)` — not the concave bargain `F(2ρ)` the field
equation offers.  The exclusion energy is the gap between what the
structure should cost and what the field charges:

    e(ρ) = 2F(ρ) − F(2ρ) = c²·r·ρ² / ((r + cρ)·(r + 2cρ)) .

No free parameters: `e` is built from `r` and `c` alone.  This is the
2-copy sector of the general n-copy gap `nF(ρ) − F(nρ)`.

## The derived form's four properties

All four are exact (sympy-verified; guarded numerically by
`tests/test_exclusion_derived.py` and the experiment's D1):

1. **Dilute limit.** `e(ρ) ≈ (b/2)ρ²` with `b = 2c²/r`.  The quadratic
   contact term is recovered — but with a coefficient the framework
   sets.  At the exclusion core's operating point (r = 0.02, c = 0.8)
   this is **b = 64**: the calibrated 0.2 is *not* in the derived
   family at any density.
2. **Clone refusal, and its inversion.** `E_x(2ρ) > 2E_x(ρ)` — a stack
   of two costs more than two singles — holds exactly for
   `ρ < r/(2c)` (= 0.0125 at the operating point), because
   `E_x(2ρ) − 2E_x(ρ) = 2c²rρ²(r − 2cρ)/((r+cρ)(r+2cρ)(r+4cρ))`.
   **Above `r/(2c)` the inequality inverts**: the derived term starts
   *rewarding* the merger it was built to refuse.  Measured in the
   experiment (D1): a single sign change at ρ = 0.012477 against the
   theoretical 0.0125.
3. **Saturation.** `e(ρ) → r/2` per site as `ρ → ∞` — the degeneracy
   debt is bounded, because capacity regenerates.  Deep overlap is a
   fixed-price neighborhood, not an ever-steepening wall.
4. **Repulsive per site, always.** `e'(ρ) = c²r²ρ(3cρ + 2r) /
   ((r+cρ)²(r+2cρ)²) > 0`, so the force `F_i = ∫ e'(ρ_tot)·∇load_i`
   pushes every blob away from load-weighted overlap — but `e'(ρ)`
   peaks at ρ ≈ 0.0088 (inside the refusal window) and falls as
   `3r²/(4cρ²)` in the saturated regime, so the *weighting* of that
   push is strongest on the dilute skirts, not on the dense core.

The physical reading ties the coefficient to the loaded screening
length `ℓ²(ρ) = D/(r + cρ)`:

    b(ρ) = 2c²ℓ²(ρ)/D

— degeneracy stiffness set by the *local* capacity range: stiff where
capacity reaches far (dilute), soft where the load has already shrunk
the reach (dense).  `b_eff(ρ) = 2e(ρ)/ρ²` runs from 64 in the dilute
limit through 1.42 at ρ = 0.1, exactly **0.2 at ρ ≈ 0.30** — the skirt
of the exclusion core's blob — to 0.05 at the blob peak ρ = 0.6 and
0.014 at the contact-overlap density ρ ≈ 1.2.

## The experiment

`experiments/n3_exclusion_derived.py` mirrors the exclusion core's
protocol exactly (size 96, width 2.5, mass 0.6, r = 0.02, c = 0.8,
τ = 0.1, d₀ = 12, t_max = 250, dt = 0.1, same detrend + Hann + whitening
pipeline), with the contact term replaced by the derived form's exact
gradient (`contact_derived=True`).  Three pre-registered predictions;

- **D1 — the derivation's self-check: PASS.**  On a ρ grid,
  `E_x(2ρ) > 2E_x(ρ)` below `r/(2c)` and inverts above, with a single
  sign change at ρ = 0.012477 (theory 0.0125).  The implementation
  carries the exact convexity window.
- **D2 — statics, the X1 bars: FAIL (recorded as-is).**  The derived
  two-body energy is *more attractive than the no-exclusion control at
  every separation*:

      derived: E(s) = 1:-2.37 2:-2.26 3:-2.09 4:-1.88 5:-1.66 6:-1.43
                      8:-1.00 10:-0.63 12:-0.34 16:+0.00
      control: E(s) = 1:-1.65 2:-1.58 3:-1.48 4:-1.35 5:-1.21 6:-1.07
                      8:-0.79 10:-0.53 12:-0.32 16:+0.00

  No interior minimum (s\* sits at the s = 1 grid edge), barrier 0.00,
  control monotone attractive.  The mechanism is property 2 above: the
  blob's working densities (peak load 0.6, overlap to ≈ 1.2) sit 50–100×
  above the refusal window, deep in the inverted regime.  Measured
  directly, the derived overlap force on a blob is *attractive* at every
  separation probed (s = 1…8): `e'` peaks on the dilute skirts, so the
  far side of each blob is pulled harder than the near side is pushed.
  The exclusion energy of the merged configuration is ≈ 0.72 *lower*
  than the separated one — the derived term pays the merger.
- **D3 — dynamics, no-floor branch: PASS.**  With no static floor, the
  registered expectation was a plunge-to-contact, baseline-like binary.
  Recorded: plunge at t ≈ 8–9 (control: 9); late mean separation 1.95
  vs the baseline's 1.61 (ratio 1.21, within the factor-of-2 bar); the
  post-plunge slosh rings in the [15, 95] window at ω = 0.786 against
  the separation channel's 0.795, whitened contrast 1028 — and the
  no-exclusion control rings in the same window at ω = 0.629 with
  contrast 796 (ratio 1.29, within the factor-of-2 bar).  The derived
  dynamics are the baseline's.

**Score: 2/3** — the self-check and the (conditional) dynamics land;
the statics fail, and the failure is the finding.

## What this means for the calibrated b = 0.2

The calibrated coefficient is **refuted as a derivation and clarified
as a surrogate**.  The framework's homogeneous counting does not
produce a constant `b`; it produces `b(ρ) = 2c²ℓ²(ρ)/D`, which equals
0.2 only at ρ ≈ 0.30 — the skirt of the blob, where overlap *begins*.
The exclusion core's constant `b = 0.2` is therefore best read as a
phenomenological surrogate that prices the skirt correctly and the core
of the blob far too generously (by 4–14×), and — decisively — keeps the
repulsive sign everywhere, while the derived form's net overlap force
inverts at the densities where the blob actually lives.  The floor and
the ringdown of the exclusion core are real *given that term*, but that
term is not (yet) the framework's own.

## Registered follow-ups

1. **Gradient terms.**  The derivation dropped them; at the operating
   point ξ = √(D/r) = 7.07 against a blob width of 2.5, so the
   homogeneous `F(ρ)` is not the whole free energy of a real blob.  A
   derivation from `F[κ]` with gradients — the actual functional the
   field descends — is the honest next candidate, and the inversion
   may move or vanish.
2. **Identicality.**  `e(ρ_tot)` prices *all* overlap as duplication;
   the load field cannot tell same-distinction from
   different-distinction stacking.  A distinction-resolving load (a
   labelled or multi-component source) is the door to pricing only
   true clones.
3. **The n-copy sector.**  Only `2F(ρ) − F(2ρ)` was probed; the general
   `nF(ρ) − F(nρ)` prices n-fold stacks and may stiffen where the
   2-copy sector saturates.
4. **A dilute operating point.**  Inside the refusal window
   (ρ < r/(2c)) the derived term *is* the calibrated term with
   b = 2c²/r, and the tests confirm repulsion there.  A binary of
   dilute, broad blobs would test the floor where the derivation is
   self-consistent.

## Honest edges

- The derivation uses the homogeneous `F(ρ)` — gradient energy
  neglected, and at the operating point ξ/width ≈ 2.8 says that is not a
  small omission.  The result bounds the *homogeneous-sector* counting,
  not the full field functional.
- Maximal-identicality: all overlap is priced as duplication —
  conservative by construction.
- 2-copy sector only.
- One operating point (size 96, width 2.5, mass 0.6, r = 0.02, c = 0.8,
  τ = 0.1).
- Instrument note, measured: with no floor the plunge lands at
  t ≈ 8–9 and the [15, 95] ring window is dominated by the post-plunge
  slosh — the no-exclusion *control* rings there with a contrast of the
  same order (796 vs 1028), so the no-floor D3 comparison is made
  against the control's own contrast, not against an absolute silence
  bar.  The silent-blob figure quoted in the exclusion core belongs to
  its late, settled window, not to this one.
- The D2 failure is a property of the operating point, not of the
  algebra: D1 confirms the implementation carries the exact window the
  symbolic derivation guarantees.  What the framework's counting gives
  at these densities is a merger subsidy, and that is what the
  experiment recorded.

---

# Part II — the gradient terms

### The exclusion energy of the FULL functional: the inversion is gone, and the framework's own term buys the floor

*Follow-up #1, registered by Part I.  Experiment:
`experiments/n3_exclusion_gradient.py`; implementation:
`project_genesis/capacity_waves.py` (`screened_green_function`,
`apply_screened_kernel`, `linear_response_exclusion_gap`,
`duplicated_load`, `exclusion_gap_full`, `contact_full`); tests:
`tests/test_exclusion_gradient.py`.  Verdict: **3/3 — with the gradient
energy kept, the derived term is positive at the operating point, buys
an exclusion floor with no free parameters, and the contact binary
rings at the statics' pitch.  Recorded as-is.*

---

## What Part I dropped

Part I derived the exclusion term from the HOMOGENEOUS capacity free
energy density `F(ρ) = rcρ/(2(r+cρ))` — the gradient energy
`(D/2)|∇κ|²` was neglected, and at the operating point
`ξ = √(D/r) = 7.07` against a blob width of 2.5 that is not a small
omission.  The derived term inverted into a merger subsidy there (D2
FAIL): the blob's working densities sit 50–100× above the refusal
window `ρ < r/(2c) = 0.0125`, deep in the regime where
`E_x(2ρ) < 2E_x(ρ)` and the `e′`-weighted overlap force points the
wrong way.  This part rederives the exclusion energy against the FULL
functional the field actually descends,

    F[κ] = ∫ [ (D/2)|∇κ|² + (r/2)(κ − κ₀)² + (c/2)·ρ·κ² ] ,

and asks how much of the linear-response exclusion survives the
nonlinear core.  All of the sign, ~4% of the magnitude — and that is
enough for the floor.

## The derivation, made exact

At fixed load ρ, the relaxed minimum of `F[κ]` obeys
`(r + cρ − D∇²)κ̄ = rκ₀`.  In linear response (`cρκ₀ ≪ r`), writing
`κ = κ₀ − u` and minimising the quadratic form gives

    E(ρ) = (cκ₀²/2)·∫ρ  −  (c²κ₀²/2)·(ρ, G_r ρ) + O(c³) ,
    (r − D∇²)G_r = δ ,

the screened/Yukawa kernel of range `ξ = √(D/r)` — the same kernel
that mediates κ-gravity.  The full-overlap gap of two identical
copies is then a POSITIVE, nonlocal self-interaction through that
kernel:

    G(ρ₁) := 2E(ρ₁) − E(2ρ₁) = c²κ₀²·(ρ₁, G_r ρ₁) > 0 .

Two exact refinements of the sketch this started from:

1. **The total gap can never flip sign.**  `E(A) = min_κ F[κ; Aρ₁]`
   is a minimum over functions AFFINE in A, hence concave in A; with
   `E(0) = 0` concavity gives `2E(A) ≥ E(2A)` at every amplitude.
   G(A) ≥ 0 is a theorem, not a hope — the M1 scan confirms it
   numerically at all eight amplitudes probed.
2. **The homogeneous part is not the gap's opponent.**  In the
   homogeneous limit the gap reduces to `∫e(ρ) > 0` — positive too.
   What inverted at the operating point in Part I was never the gap's
   sign: it was (a) SUPERADDITIVITY, `E_x(2ρ) < 2E_x(ρ)` above the
   refusal window — the term priced the merged stack cheaper than two
   singles; and (b) the net overlap FORCE, skirt-weighted into
   attraction.  The question the gradient terms answer is therefore
   not "is the gap positive" (it always is) but "is the
   separation-dependent exclusion energy large and repulsive where the
   clone actually lives".

## M1 — the exact full-overlap gap

`G(A) = 2E(A) − E(2A)` for one Gaussian blob (width 2.5, size 96),
decomposed into the three F components (gradient / recovery /
consumption), measured with `relax_capacity` + `capacity_free_energy`:

    A = 0.01: G = +0.012718  (grad −0.00620, rec −0.00460, cons +0.02353)
    A = 0.10: G = +0.499145  (grad −0.04698, rec −0.04737, cons +0.59350)
    A = 0.15: G = +0.776826  (grad +0.01179, rec −0.01366, cons +0.77869)
    A = 0.20: G = +1.015215  (grad +0.09233, rec +0.03843, cons +0.88445)
    A = 0.25: G = +1.218014  (grad +0.17799, rec +0.09716, cons +0.94287)
    A = 0.30: G = +1.391618  (grad +0.26128, rec +0.15682, cons +0.97352)
    A = 0.60: G = +2.070377  (grad +0.63851, rec +0.45901, cons +0.97286)
    A = 1.20: G = +2.715471  (grad +1.00069, rec +0.82099, cons +0.89380)

The total is positive everywhere (the theorem above).  The
decomposition carries the interesting sign structure: in linear
response the GRADIENT and RECOVERY components of the gap are NEGATIVE
(at the minimum the quadratic pieces of F equal half the binding with
opposite sign) and only the consumption component is positive; the
gradient component turns positive at **A\* ≈ 0.14** (between 0.10 and
0.15), the recovery component at ≈ 0.17.  This is the precise,
measured sense in which the gradient terms switch from binding to
excluding exactly where the κ-core saturates — ten times above the
homogeneous refusal window `r/(2c) = 0.0125`.

## M2 — the linear-response benchmark

`G_LR(A) = c²κ₀²(ρ₁, G_r ρ₁)` computed with the lattice screened
kernel (`Ĝ(k) = 1/(r + D|k|²)` with the 5-point symbol
`|k|² = 4Σ sin²(k_ax/2)`, so the kernel inverts exactly the operator
the relaxer descends; `Ĝ(0) = 1/r` — the k = 0 mode is finite because
r > 0).  The ratio `G(A)/G_LR(A)`:

    A = 0.001: 0.9956     A = 0.002: 0.9813     A = 0.005: 0.9401
    A = 0.010: 0.8772     A = 0.100: 0.3443     A = 0.300: 0.1066
    A = 0.600: 0.0397     A = 1.200: 0.0130

Linear response holds at small amplitude (within 0.5% at A = 0.001;
the residual at A = 0.01 is the O(c³) term, order `cAκ₀/r`).  At the
operating amplitude the kernel benchmark overshoots the exact gap by
**25×** — the saturated core eats the quadratic growth.  The sign,
not the magnitude, is what survives at 0.6.

## M3 — the gap as a function of separation

The naive interpolant `2E(ρ₁) − E(ρ₁+ρ₂)` does not vanish at infinity.
The correct one prices only the DUPLICATED fraction
`ρ_dup(x; s) = min(ρ₁, ρ₂)(x)` — the cloned component of two identical
copies:

    E_x(s) = 2E(ρ_dup(s)) − E(2ρ_dup(s)) :

    s =  1.0: +1.870058    s =  4.0: +1.203032    s =  8.0: +0.332032
    s =  2.0: +1.658551    s =  5.0: +0.965264    s = 10.0: +0.099613
    s =  3.0: +1.435694    s =  6.0: +0.731545    s = 12.0: +0.018647
                          s = 16.0: +0.000186

Checks close: `E_x(∞) = 3.6e-10 ≈ 0` (nothing cloned, nothing priced)
and `E_x(0) = 2.070377 = G(0.6)` (everything cloned, the full gap).
The curve is positive and monotone decreasing — so the exclusion force
`−dE_x/ds` is REPULSIVE AT EVERY SEPARATION.  That is the structural
difference from Part I: `e(ρ_tot)` prices the total load and inverts,
while `min(ρ₁, ρ₂)` can only shrink as the blobs separate — the cloned
component is monotone by construction, and the gradient terms put real
energy (not saturated `r/2`-per-site energy) at the densities where
the clone lives.

## The corrected instrument: `contact_full`

The binary exclusion force from the gradient-corrected functional:
with `E_x = 2E(ρ_dup) − E(2ρ_dup)`, the envelope theorem gives
`δE/δρ(x) = (c/2)κ̄(x)²` at the relaxed field, so

    F_i = Σ_x w·(∂ρ_dup/∂ρ_i)·∇ρ_i ,   w = c·(κ̄[ρ_dup]² − κ̄[2ρ_dup]²) ,

repulsive wherever anything is cloned (`κ̄[ρ_dup] > κ̄[2ρ_dup]`).
Implementation choices, all measured before adopting:

- **The min is smoothed** (`(a+b)/2 − ½√((a−b)² + 4ε²)`, `ε = 1e-4`).
  The hard `min + indicator` rule mis-splits the symmetry tie-plane
  `ρ₁ = ρ₂`, which LANDS ON LATTICE SITES for an equal binary: the
  measured third-law violation was up to ~35% of the force, and the
  force matched finite-difference E_x only to ~25%.  The smoothed min
  assigns the symmetric 50/50 subgradient at ties, conserves momentum
  to machine precision, and matches finite-difference E_x to 1 part in
  1e4 — because force AND recorded energy come from the same smoothed
  functional.  The statics (M3, G2) use the exact min; the two differ
  by ≤ 0.1% in E_x.
- **The two auxiliary relaxed fields are solved directly.**  The
  relaxer's steady state is the solution of the LINEAR system
  `(r + cρ − D∇²)κ = rκ₀`, so conjugate gradient on that SPD operator
  replaces the explicit flow in the force path (~50× faster; validated
  to the relaxer's own tolerance in the tests).  The recorded
  measurements (M1–M3, G2) still use `relax_capacity` itself.
- **Bookkeeping**: E_x in the dynamics' `energy` is evaluated with the
  5-point functional the relaxer actually descends (`_relax_functional`
  — integration by parts against `periodic_laplacian`), the functional
  whose minimum the relaxed field EXACTLY is; `capacity_free_energy`
  (central-difference gradient) differs by ≤ 0.5% but leaves a ~1%
  force/energy mismatch.  The Lyapunov test passes at the repo's
  1e-6 bar.
- **The exclusion sector is adiabatic** (the auxiliary fields are the
  relaxed minima of the CURRENT positions), while the gravity sector
  stays retarded — the same instantaneous-matter-sector split as
  `contact_b`, now with a field-derived price.

## The experiment

`experiments/n3_exclusion_gradient.py` mirrors Part I's protocol
(size 96, width 2.5, mass 0.6, r = 0.02, c = 0.8, τ = 0.1, d₀ = 12,
t_max = 250, dt = 0.1, same detrend + Hann + whitening pipeline),
with `contact_full=True` — no free parameters anywhere.

- **G1 — the sign the homogeneous derivation missed: PASS.**
  G(0.6) = **+2.0704 > 0** (M1), and at small amplitude the exact gap
  matches the kernel benchmark to 0.5% (A = 0.001, well inside the
  factor-of-2 bar).  No sign-flip amplitude of the total gap exists —
  by theorem, and none was found in the scan; the gradient COMPONENT's
  flip sits at A\* ≈ 0.14.
- **G2 — statics, the X1 bars: PASS.**  Interior minimum at
  **s\* = 8**, barrier **0.68** toward s = 1, no-exclusion control
  monotone attractive:

      full:    E(s) = 1:+0.22 2:+0.08 3:-0.04 4:-0.15 5:-0.25 6:-0.34
                      8:-0.46 10:-0.43 12:-0.30 16:+0.00
      derived: E(s) = 1:-2.37 2:-2.26 3:-2.09 4:-1.88 5:-1.66 6:-1.43
                      8:-1.00 10:-0.63 12:-0.34 16:+0.00   (Part I)
      b = 0.2: E(s) = 1:-0.29 2:-0.37 3:-0.49 4:-0.61 5:-0.69 6:-0.74
                      8:-0.68 10:-0.51 12:-0.31 16:+0.00
      control: E(s) = 1:-1.65 2:-1.58 3:-1.48 4:-1.35 5:-1.21 6:-1.07
                      8:-0.79 10:-0.53 12:-0.32 16:+0.00

  The nonlinear core does NOT eat the gradient correction: the gap at
  full overlap (2.07) is larger than the field's merger gain (1.65),
  and the floor appears.  The registered doubt is resolved by
  measurement.
- **G3 — dynamics, the floor branch: PASS.**  The released binary
  stalls on the floor: late mean separation **8.56** vs s\* = 8
  (ratio 1.07) and vs the no-exclusion baseline's 1.61 (≥ 2× bar).
  The contact binary rings: waveform line at ω = **0.472** vs the
  separation channel's libration ω = **0.477** (0.010 apart, bar 25%)
  vs the static pitch `√(E″(s*)/μ)` = **0.344** (ratio 1.39, bar a
  factor of 2); whitened contrast **305** (bar ≥ 3; the no-exclusion
  control's post-plunge slosh rings at ω = 0.629 with contrast 796 —
  reported for context; the G3 bars are on the full run's own line).
  The statics predict the ringdown's pitch, as in the exclusion core.

**Score: 3/3.**

## What this does to the inversion, and to b = 0.2

The Part-I inversion is a property of the homogeneous truncation, not
of the framework's counting.  With the gradient energy kept, the same
extensivity argument — price the clone at the extensive cost of its
content — produces a term that is positive at every amplitude,
repulsive at every separation, and strong enough at the operating
point to buy the floor with no free parameters.  The calibrated
b = 0.2 is thereby SUPERSEDED as a surrogate: the framework's own
term lands the same three predictions the exclusion core landed with
the hand-picked constant (a floor — s\* = 8, barrier 0.68 against the
calibrated s\* ≈ 6, barrier 0.45 — a stalled contact binary, and a
ringdown at the statics' pitch), and Part I's failure of the
homogeneous form is now understood as the missing gradient physics,
with the mechanism (the gradient component of the gap flips sign at
A\* ≈ 0.14, where the κ-core saturates) recorded.

## Registered follow-ups

1. **Identicality (the bridge to follow-up #2).**  `ρ_dup = min(ρ₁, ρ₂)`
   is exact for IDENTICAL copies only; for non-identical loads it is
   the maximal common component.  A distinction-resolving load (a
   labelled or multi-component source) is the door to pricing only
   true clones — and `min` generalises to labelled overlaps.
2. **A retarded exclusion sector.**  The exclusion force here is
   adiabatic while gravity is retarded.  Co-evolving the auxiliary
   fields (or deriving the instantaneous response of the relaxed
   minimum) would make the exclusion sector causal too.
3. **The n-copy sector of the full functional.**  `nE(ρ) − E(nρ)` for
   n > 2, against n-fold stacks of the cloned component.
4. **The dilute window.**  Below A\* ≈ 0.14 the gradient component of
   the gap is negative — a binary of dilute broad blobs would probe
   whether the total gap's positivity is the whole story there.

## Honest edges (Part II)

- Linear response is exact only to O(c²): at the operating amplitude
  the exact gap is ~4% of the kernel benchmark.  The sign survives,
  the magnitude does not — all conclusions that matter (the floor, the
  ringdown) use the EXACT functional, not the benchmark.
- The concavity theorem (`G(A) ≥ 0` always) assumes the exact relaxed
  minimum; the relaxer and the CG solver are validated against each
  other to the relaxer's tolerance, and the M1 scan is the numerical
  confirmation.
- The duplicated fraction is exact for identical copies only
  (maximal-identicality convention, as Part I).
- The smoothed min (ε = 1e-4) vs the exact min: ≤ 0.1% in E_x; the
  statics use exact, the dynamics smoothed (force/energy consistency
  is what the Lyapunov bookkeeping needs).
- The 5-point functional (`_relax_functional`, dynamics bookkeeping)
  vs `capacity_free_energy` (statics, repo convention): ≤ 0.5%
  everywhere; the difference is the repo's own pre-existing
  flow-vs-recorded-functional stencil mismatch, unchanged here.
- One operating point (size 96, width 2.5, mass 0.6, r = 0.02,
  c = 0.8, τ = 0.1).
- Instrument note, measured: with the floor at s\* = 8 and the release
  at d₀ = 12 the pair stalls almost immediately after the plunge, and
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

---

# Part IV — the n-copy sector

*Experiment: `experiments/n3_exclusion_ncopy.py`; implementation:
`project_genesis/capacity_waves.py` (`exclusion_gap_group`, the
n-ary symmetrized smoothed min, the `n_copy` option on the
`contact_full` path); tests: `tests/test_exclusion_ncopy.py`.
Verdict: **3/3 — the pairwise sum and the true n-copy gap are one
theorem at O(c²), the trimer has a floor, and the split between the
two forms at operating amplitude is itself the saturation
measurement.*

---

## The question Part III left open

The labelled instrument prices each shared pair separately. For
three or more same-label masses stacked on the same site, does the
pairwise sum price the stack correctly — or does it double-count?
The general form the extensivity argument implies is the n-copy
gap: n identical copies of content `m` should pay the extensive
price `n·E(m)`, not the field's concave offer `E(n·m)`:

    E_x = n·E(m) − E(n·m) ,      m = min over the group's loads .

## The theorem: pairwise = n-copy at O(c²)

In linear response the relaxed energy is quadratic in amplitude,
`E(A·m̂) = −k·A²` up to the vacuum constant
(`k = (c²κ₀²/2)(m̂, G_r m̂)`). Then for n fully-stacked copies:

    n-copy:   n·E(m) − E(n·m)       = k·(n² − n)
    pairwise: Σ_pairs [2E(m)−E(2m)] = C(n,2)·2k = k·n(n−1)

— identically equal. The pairwise instrument does NOT over-count
in linear response; any split at finite amplitude is the saturated
core departing from the quadratic form. The n-copy form is the
general one; the pairwise sum is its linear-response shadow.

## The experiment — 3/3 pre-registered

Geometry: three same-label masses on an equilateral triangle
(side s), the arc's operating point. Independently reproduced
before merging this record.

- **N1 (the theorem): PASS.** Fully-stacked triple, dilute
  (A = 0.01): pairwise 0.0381542 vs n-copy 0.0365265 — ratio
  1.0446, within the 5% bar; against the O(c²) value
  3c²κ₀²(ρ, Gρ) = 0.0434952 both sit at 0.877/0.840 (the known
  dilute-remainder of the lattice instruments). At the operating
  amplitude (A = 0.6) the ratio is **1.4080** (pairwise 6.21113,
  n-copy 4.41116): the pairwise form prices the thrice-cloned
  core three times, the n-copy gap prices it once — the split is
  the core saturation, measured.
- **N2 (trimer statics): PASS.** Interior minimum at side
  **s*₃ = 8 with barrier 2.06** toward contact (bars:
  s*₃ ∈ (2, 12), ≥ 0.2); the no-exclusion control is monotone
  attractive. The pairwise form (comparison, no bar) holds the
  trimer farther out and harder: s*₃ = 10, barrier 3.29 — the
  saturation-split made visible as a floor dial.
- **N3 (trimer dynamics from rest): PASS.** Released at side
  d₀ = 12 (the baseline's three-way collapse completes at
  t ≈ 8 — measured first; t_max chosen against it, an instrument
  necessity, recorded). Late mean pairwise separation: **6.83**
  vs s*₃ = 8 (ratio 0.85, bar a factor of 2) and vs the
  baseline's **2.55** (ratio 2.68, bar ≥ 1.5): the same-label
  trimer stalls on its static floor; the baseline plunges to
  three-way contact. Pairwise form (recorded, no bar): 9.52.
  Exploratory, NOT pre-registered: a breathing-mode line
  ω = 0.636 in the separation channel against the static
  breathing pitch √(E″(s*₃)/m) = 0.422; probe waveform line
  ω = 0.315, whitened contrast 34.3. Recorded without a bar —
  three-body spectra are not the binary instrument's.

## What this closes

Part III's registered follow-up is answered by theorem and
measurement: the general n-copy form exists, reduces to the pair
form **bitwise** at n = 2 (test-guarded, both statics and
dynamics), answers the double-count question at O(c²) (the
pairwise sum does not over-count in linear response), and the
trio works like the pair — floor, stall, and a saturation dial.
The exclusion story is now complete through the trio: gravity
(the sharing discount) binds everything; exclusion selectively
refuses the discount for true clones; and n clones pay the
n-fold gap.

## Registered follow-ups

1. **The dilute operating point (follow-up #4, already
   specced).** Where the core stays linear, the pairwise/n-copy
   split should vanish AND the homogeneous (Part I) and
   full-functional (Part II) derivations should agree — N1's
   dilute remainder marks the edge of that regime.
2. **Identity generation.** Unchanged from Part III: labels are
   assigned, not derived; a load that carries its own distinction
   structure is the honest next candidate.
3. **A retarded exclusion sector.** Unchanged from Part II: the
   exclusion force is adiabatic while gravity is retarded.
4. **Rotating trimers and n ≥ 4 stacks.** The dynamics here are
   from rest (no angular momentum; the collapse is the breathing
   channel), and the symmetrized min's ~n!/2 cost makes large
   groups the pairwise option's territory.

## Honest edges (Part IV)

- **Three bodies only.** The n-copy form is implemented and
  tested for general n, but the record (N1–N3) is the trimer;
  n ≥ 4 dynamics are untested territory.
- **The min-over-group construction is the maximal-identicality
  convention again** — exact for identical copies, the maximal
  common component otherwise; non-identical stacks are
  conservatively charged the clone price.
- **Dynamics from rest only** — no angular momentum; the
  breathing-mode line is exploratory, recorded without a bar.
- **The n-ary smoothed min is a convention.** The symmetrized
  iteration was chosen because it is bitwise the pair instrument
  at n = 2 (the regression guard), not because nature picked it;
  any smoothing gives the same physics to O((n−1)ε) = 2e-4 here.
- **Momentum bookkeeping is the pair's, one level up**: the
  mirror channel of the symmetric trimer conserves momentum to
  machine precision; the common-translation channel equals
  −dE_x/d(shift) to 0.4% — the lattice artifact of Part III,
  unchanged. A development bug made this visible: pair terms must
  write to their global mass indices; the (b, a, a) label
  arrangement guards it in the tests.
- One operating point (size 96, width 2.5, mass 0.6, r = 0.02,
  c = 0.8, τ = 0.1); statics on relax_capacity with the exact
  min, dynamics on the symmetrized smoothed min (ε = 1e-4) with
  conjugate-gradient solves, as the base branch.
