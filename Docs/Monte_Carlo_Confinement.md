# Monte Carlo Confinement — Theory to Code Bridge

This note explains how `project_genesis/gauge_mc.py` maps the physics of
lattice gauge confinement onto the URP gauge derivation, what each observable
means, and how to interpret the experiment output.

## Why Monte Carlo?

The Yang–Mills gradient-flow dynamics (`gauge.flow_step`) find a **stationary
point** of the gauge S-functional

```
S = coupling · covariant_coherence − wilson_action
```

by deterministic gradient ascent.  That finds the configuration that locally
maximises S — the lattice equations of motion — and confirms that the YM
residual → 0 and that curvature localises on the sector walls ("gluons as
boundary modes").

But **confinement** is a *thermodynamic* property of the ensemble, not of a
single configuration.  The string tension σ, the area law, and the
deconfinement transition are defined through expectation values

```
⟨W(R,T)⟩ = Z⁻¹ ∫ DU  W(R,T)[U]  exp(−β_g · S_W[U])
```

where the path integral is over all SU(N) link configurations.  Monte Carlo
samples this integral by constructing a Markov chain whose stationary
distribution is `exp(−β_g · S_W)`.

## The Wilson Action

The pure-gauge Wilson action is the direct lattice implementation of the URP
coherence-stress term:

```
S_W = Σ_{x,μ<ν} Re Tr(1 − P_{μν}(x))
```

where `P_{μν}(x) = U_μ(x) U_ν(x+μ̂) U_μ(x+ν̂)† U_ν(x)†` is the Wilson
plaquette — a closed loop of links around an elementary square.  `S_W = 0`
on the identity (no curvature); it grows with the curvature of the gauge
field, i.e. the residual coherence stress the derivation identifies with
`Tr(F_{μν}F^{μν})`.

## Update Algorithms

### Metropolis

Propose `U' = U · R` where `R` is a small near-identity SU(N) matrix.  The
local action change is computed **exactly from the staple sum**:

```
ΔS = Re Tr[(U_old − U_prop) · A†]
```

where `A = Σ_ν (forward_staple_ν + backward_staple_ν)` sums the links
surrounding the proposed update.  Accept with probability `min(1, exp(−β_g·ΔS))`.

The step scale controls acceptance rate; tuning to ~50 % is standard.

### Kennedy–Pendleton (SU(2) heat-bath)

For SU(2), the link distribution conditioned on its neighbours is exactly
integrable.  The Kennedy–Pendleton algorithm (1985) samples directly from:

```
P(U) ∝ exp(β_g/2 · Re Tr[U · A†])
```

This is exact — no accept/reject needed — and thermalises faster than
Metropolis.  The implementation factorises the staple as `A = k · V` (k ≥ 0,
V ∈ SU(2)) and samples `a₀` from `p(a₀) ∝ √(1−a₀²) exp(2·α·a₀)` where
`α = β_g · k / 2`, then draws a random unit vector on S² for the su(2)
component.

### Cabibbo–Marinari (SU(3) pseudo-heat-bath)

For SU(3), exact heat-bath is not available in closed form.  The standard
workaround cycles through the three SU(2) subgroups `{(0,1), (0,2), (1,2)}`
and applies an exact SU(2) heat-bath to each, embedded back into SU(3).  One
full cycle over all three subgroups constitutes one Cabibbo–Marinari step.

### Overrelaxation

For a single link, the overrelaxed update is the **reflection** through the
action minimum:

```
U_new = A† · (U · A†)†
```

This leaves `Re Tr[U · A†]` unchanged (microcanonical) but strongly
decorrelates the link orientation.  Standard practice: `n_hb` heat-bath
sweeps + `n_or` overrelaxation sweeps in each MC step.

## Observables

### Wilson Loops W(R,T)

A rectangular loop of spatial extent R and temporal extent T:

```
W(R,T) = (1/N) Re Tr[ Π_{R steps} U_μ · Π_{T steps} U_ν · Π_{R steps} U_μ† · Π_{T steps} U_ν† ]
```

averaged over all origins (translational invariance).

In the **confined phase**: `⟨W(R,T)⟩ ~ exp(−σ·R·T − c·(R+T))`  (area law)

In the **deconfined phase**: `⟨W(R,T)⟩ ~ exp(−c·(R+T))`  (perimeter law)

The area-law coefficient σ is the **string tension** — the energy cost per
unit length of a flux tube between colour sources.

### Creutz Ratio χ(R,T)

```
χ(R,T) = −log[ W(R,T) · W(R-1,T-1) / W(R,T-1) · W(R-1,T) ]
```

This cancels the perimeter contribution to leading order, giving a cleaner
estimate of σ.  `χ → σ` as `R,T → ∞`.

### Polyakov Loop ⟨P⟩

The Polyakov loop is a closed loop winding around the temporal boundary of the
lattice.  Its expectation value is:

- `⟨|P|⟩ ≈ 0` in the **confined phase** (Z_N centre symmetry unbroken)
- `⟨|P|⟩ > 0` in the **deconfined phase** (Z_N broken)

The susceptibility `⟨|P|²⟩ − ⟨|P|⟩²` peaks at the deconfinement transition.

## Running the Experiment

```bash
# Quick 2-D SU(2) Metropolis run
python experiments/monte_carlo_confinement.py --size 6 --n 2 --beta 1.8

# SU(3) heat-bath on a 5³ lattice
python experiments/monte_carlo_confinement.py \
    --size 5 --n 3 --ndim 3 --beta 5.0 \
    --updater heatbath --n-therm 200 --n-meas 50

# Deconfinement β-scan (SU(2), Metropolis)
python experiments/monte_carlo_confinement.py \
    --size 6 --n 2 --beta-scan \
    --beta-min 0.5 --beta-max 3.5 --beta-steps 7 \
    --updater heatbath
```

## Interpreting the Output

| Quantity | Confined | Deconfined | Unclear |
|---|---|---|---|
| σ (string tension) | > 0 | ≈ 0 or negative | near 0 |
| ⟨P⟩ | ≈ 0 | > 0 | near 0 |
| W(R,T) decay | Area law | Perimeter law | Mixed |
| χ(R,T) | ≈ σ > 0 | ≈ 0 | mixed |

The 2-D theory (strong-coupling expansion) confines for all β_g > 0.  In 3-D
and 4-D there is a deconfinement transition at a critical β_g.

## What Is and Is Not Demonstrated Here

**Demonstrated by this module:**
- The pure-gauge ensemble is correctly sampled (Metropolis, heat-bath, overrelax)
- Wilson loops and their area-law decay are measurable from the ensemble
- The Polyakov loop order parameter is measured
- String tension can be extracted from small lattices

**Not demonstrated (stated next steps):**
- Continuum-limit extrapolation and matching to physical units (MeV/fm)
- Dynamical fermion contributions
- The precise deconfinement temperature quoted in the URP gauge derivation
  (~150–170 MeV) — that requires renormalisation-group input from the
  continuum theory
- The full thermodynamic-limit string tension requires larger lattices
  (typically 16³ or larger with ≥10⁴ measurements)

Those are the explicitly stated next steps in the README's `What Comes Next`
section.
