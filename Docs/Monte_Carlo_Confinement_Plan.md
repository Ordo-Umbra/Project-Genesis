# Monte Carlo Confinement — Theory-to-Code Bridge

This document explains what the Monte Carlo confinement layer does, why
it is built the way it is, and how to interpret its output in the context
of the Universal Recursion Principle (URP).

---

## 1. The gap this fills

The deterministic layer (`project_genesis/gauge.py`) implements:

- SU(N) link fields and gauge transformations
- The Wilson plaquette `P_{μν}(x)` as a measure of curvature / coherence stress
- The Wilson action `S_W = Σ Re Tr(1 − P)` and its gradient (link forces)
- Yang–Mills gradient flow: `U_μ → exp(ε·F_μ) U_μ` toward a stationary point

Gradient flow finds a *local minimum of the action* — a single, deterministic
field configuration.  It cannot access:

- **Confinement** — whether the average Wilson loop decays as an *area law*
  (confined phase) or a perimeter law (deconfined / free phase)
- **String tension** σ — the coefficient of the area decay, which in the URP
  context quantifies the coherence cost of separating two charged sectors
- **Polyakov loop** ⟨|P|⟩ — the order parameter for the centre symmetry
  (Z_N) that distinguishes the two phases
- **Thermodynamic fluctuations** — entropy-driven configurations that only
  appear in the Boltzmann ensemble

The Monte Carlo layer (`project_genesis/gauge_mc.py`) samples the Boltzmann
weight `exp(−β_g · S_W)` and measures these ensemble averages.

---

## 2. The Wilson action and its lattice meaning

The Wilson action for a pure gauge theory on a lattice with spacing `a` is:

```
S_W = (β_g / N) Σ_{x, μ<ν}  Re Tr(1 − P_{μν}(x))
```

where the sum runs over all sites `x` and all oriented plaquette planes
`(μ,ν)`.  In the code, `β_g` absorbs the `1/N` factor so that the
single-plaquette contribution is `Re Tr(1 − P)` ∈ `[0, 2N]`.

Physical meaning in URP terms: `S_W` measures the total **coherence stress**
across the lattice.  A flat (pure-gauge) connection has `S_W = 0` — parallel
transport around any loop is trivial.  Any curvature increases `S_W`.

The inverse coupling `β_g` controls the phase:
- **Small `β_g`** (strong coupling, high temperature): configurations are
  disordered; the area law holds.
- **Large `β_g`** (weak coupling, low temperature): configurations are
  ordered near the identity; the perimeter law takes over.

In 2+1D SU(2) the crossover happens near `β_g ≈ 2.2`; in 3+1D SU(3)
(physical QCD) near `β_g ≈ 5.7`.

---

## 3. Confinement observables

### 3.1 Wilson loop and area law

A rectangular Wilson loop of spatial extent `R` and temporal extent `T` is:

```
W(R,T) = (1/N) ⟨Re Tr [ U_μ(1)...U_μ(R) · U_ν(1)...U_ν(T)
                         · U_μ†(R)...U_μ†(1) · U_ν†(T)...U_ν†(1) ]⟩
```

(averaged over all origins in the lattice).

In the **confined phase**: `W(R,T) ≈ exp(−σ·R·T − c·(R+T))`
- The `σ·R·T` *area term* comes from the string of flux between the two
  separated colour sources.
- The `c·(R+T)` *perimeter term* comes from self-energy effects.
- **σ > 0** means separation is energetically costly — the hallmark of
  confinement.

In the **deconfined phase**: `W(R,T) ≈ exp(−c·(R+T))` — the area term
vanishes (σ ≈ 0).

### 3.2 Creutz ratio

The Creutz ratio cancels the perimeter contribution to leading order:

```
χ(R,T) = −log[ W(R,T)·W(R−1,T−1) / W(R,T−1)·W(R−1,T) ]
```

In the confined phase `χ(R,T) → σ` for large `R,T`.  It is a cleaner
estimator of the string tension than a raw log of `W(R,T)` because it
automatically subtracts the perimeter self-energy.

### 3.3 Polyakov loop

The Polyakov loop winds once around the temporal direction:

```
P(x) = Tr[ Π_{t=0}^{N_t−1} U_t(x,t) ]
```

- **⟨|P|⟩ ≈ 0** — centre symmetry (Z_N) is unbroken; confined phase.
- **⟨|P|⟩ > 0** — centre symmetry is broken; deconfined phase.

The susceptibility `χ_P = N_s^3 (⟨|P|²⟩ − ⟨|P|⟩²)` peaks at the transition.

---

## 4. Update algorithms

### 4.1 Metropolis (default)

For each link `U_μ(x)`:
1. Propose `U' = U · R` where `R` is a random near-identity SU(N) matrix.
2. Compute the *exact local* `ΔS = Re Tr[(U − U') · A†]` using the staple
   sum `A` — only O(ndim) matrix products; no full-action recompute.
3. Accept with probability `min(1, exp(−β_g·ΔS))`.

**Tune `--step-scale`** so the acceptance rate ≈ 50–70%.  A scale of 0.18
works well for `β_g ∈ [1.5, 4.0]` on SU(2) in 2D.

### 4.2 SU(2) Kennedy–Pendleton heat-bath

For SU(2) the full conditional distribution for a single link (given its
staple) can be sampled exactly without accept/reject.  This converges to
the equilibrium distribution faster per sweep than Metropolis at strong
coupling.

For SU(3) the **Cabibbo–Marinari** algorithm cycles through the three
SU(2) subgroups `{(0,1),(0,2),(1,2)}` and applies an exact SU(2) heat-bath
to each in the background of the current link.

### 4.3 Overrelaxation (microcanonical)

For each link, the update `U_new = A† · (U · A†)†` is the reflection of
`U` through the SU(N) manifold defined by the local minimum `∝ A†`.  It
leaves the local action `Re Tr[U·A†]` unchanged (microcanonical) while
strongly decorrelating the link orientation.

**Recommended production schedule**: `1 heat-bath sweep + 3–4 overrelaxation
sweeps`, then measure.  This combination minimises autocorrelation time.

---

## 5. Module API

```python
from project_genesis.gauge_mc import (
    metropolis_sweep,       # (links, beta_g, rng, *, n_sweeps, step_scale)
    heatbath_sweep,         # (links, beta_g, rng, *, n_sweeps)
    overrelaxation_sweep,   # (links, rng, *, n_sweeps, omega)
    wilson_loop,            # (links, extents, plane) -> float
    polyakov_loop,          # (links, temporal_axis) -> float
    creutz_ratio,           # (w_ij, w_im1_jm1, w_i_jm1, w_im1_j) -> float
    fit_area_law,           # (loop_matrix, r_values, t_values) -> dict
    thermalize_and_measure_pure_gauge,  # high-level driver
    deconfinement_scan,     # beta-scan helper
)
```

### Quick start

```python
import numpy as np
from project_genesis.gauge_mc import thermalize_and_measure_pure_gauge

rng = np.random.default_rng(42)
summary, links = thermalize_and_measure_pure_gauge(
    size=8,           # 8^2 lattice
    n=2,              # SU(2)
    beta_g=2.5,       # moderately strong coupling
    rng=rng,
    ndim=2,
    n_therm=300,
    n_meas=100,
    n_skip=5,
    updater="heatbath",
    loop_sizes=[(1,1),(2,2),(3,3),(2,3)],
)

fit = summary["area_law_fit"]
print(f"String tension sigma = {fit['sigma']:.4f}")
print(f"Polyakov loop <|P|>  = {summary['polyakov_mean']:.4f}")
print(f"Creutz ratios: {fit['creutz_ratios']}")
```

### Driver return value

The `summary` dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `beta_g` | float | Inverse gauge coupling used |
| `ndim`, `size`, `n` | int | Lattice parameters |
| `updater` | str | Algorithm used |
| `loop_averages` | dict | `{"W_R_T": float, ...}` ensemble means |
| `area_law_fit` | dict | `sigma`, `perimeter_coeff`, `fit_residual`, `creutz_ratios`, `raw_loops` |
| `polyakov_mean` | float | Ensemble mean of the Polyakov loop |
| `polyakov_susceptibility` | float | Variance of Polyakov loop (peaks at transition) |
| `final_wilson_action` | float | Wilson action of the last configuration |

---

## 6. Running the experiment script

```bash
# Single beta point — moderate run
python experiments/mc_confinement.py \\
    --beta 2.5 --size 8 --n 2 \\
    --updater heatbath \\
    --n-therm 300 --n-meas 100 --n-skip 5 \\
    --loop-sizes 1x1,2x2,3x3,2x3,3x2

# Beta scan — observe the deconfinement crossover in SU(2)
python experiments/mc_confinement.py --scan \\
    --beta-min 1.0 --beta-max 5.0 --beta-steps 9 \\
    --size 6 --n 2 --updater heatbath \\
    --n-therm 200 --n-meas 40

# Quick smoke test (runs in < 30 s)
python experiments/mc_confinement.py --quick
```

---

## 7. Interpreting the output in URP terms

The URP gauge derivation establishes that gauge connections emerge from
requiring **covariant coherence** `Σ Re[ψ†(x) U_μ(x) ψ(x+μ̂)]` to be
locally gauge-invariant (§3.2).  The Wilson action `S_W` is the curvature
penalty — the "coherence stress" — that arises from non-trivial holonomy.

The Monte Carlo confinement measurement translates directly:

- **σ > 0 (area law)** — In the URP picture, this means the *energetic cost*
  of separating two sectors grows linearly with separation.  The gauge field
  forms a "flux tube" of coherence stress between them — a string.  This is
  the lattice realisation of what the URP describes as long-range coherence
  correlation enforced by gauge invariance.

- **σ ≈ 0 (perimeter law)** — The flux tube dissolves; separation is only
  penalised at the endpoints (self-energy).  This is the weak-coupling,
  high-symmetry phase where local coherence is sufficient without long-range
  structure.

- **Polyakov loop ≈ 0 → non-zero** at the crossover — The Z_N centre
  symmetry of the gauge group breaks, indicating that the thermal ensemble
  has developed a preferred global orientation.  In URP terms, the lattice
  acquires a "direction" in colour space that the confined phase suppresses.

### What this does NOT yet include

- **Dynamical matter / fermions** — The current implementation is pure-gauge
  only.  Adding fermionic matter fields (quenched or dynamical) is the next
  step after validating the pure-gauge string tension.
- **Continuum limit** — Measuring the physical string tension in MeV/fm
  requires a renormalisation-group analysis of how σ(β_g) scales with the
  lattice spacing `a(β_g)`.  This needs multi-beta runs on larger lattices.
- **Asymptotic freedom** — Demonstrating that the effective coupling runs to
  zero at short distances (high β_g) requires computing the running coupling
  from the plaquette and comparing to the two-loop beta function.

All of these are identified in the README as subsequent milestones after the
pure-gauge confinement baseline is established.

---

## 8. File map

| File | Role |
|------|------|
| `project_genesis/gauge.py` | SU(N) group elements, plaquette, Wilson action, Yang–Mills gradient flow |
| `project_genesis/gauge_mc.py` | Monte Carlo sampler, heat-bath, overrelaxation, Wilson loop, area-law fit |
| `experiments/mc_confinement.py` | CLI experiment: single-beta and beta-scan modes, formatted output |
| `tests/test_gauge_mc.py` | Unit and physics correctness tests for the MC layer |
| `Docs/Monte_Carlo_Confinement_Plan.md` | This document |

---

*Last updated: June 2026 — feature/monte-carlo-confinement branch*
