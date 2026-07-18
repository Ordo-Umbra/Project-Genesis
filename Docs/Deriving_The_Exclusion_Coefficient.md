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
