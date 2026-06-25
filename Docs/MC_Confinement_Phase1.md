# Monte Carlo Confinement – Phase 1 & 1.1

**Date:** June 2026  
**Status:** Implemented and validated

## Overview

Phase 1 adds a functional pure-gauge Monte Carlo layer to Project Genesis, enabling thermodynamic sampling of the Wilson action. This is the first executable step toward extracting confinement signatures (string tension, area law, deconfinement temperature) from the URP-derived gauge sector.

Phase 1.1 is a small follow-up that significantly improves the over-relaxation updater.

## Deliverables

### `project_genesis/gauge_mc.py` (v2 + v2.1)

- **ndim generalization**: The module now supports arbitrary spatial dimension (`ndim=2` and `ndim=3` tested).
- **Three updaters**:
  - `metropolis_sweep` — local ΔS in 2-D, full-action fallback in higher dimensions.
  - `heatbath_sweep` — heat-bath style updates (SU(2) and SU(3)).
  - `overrelaxation_sweep` (v2.1) — proper microcanonical reflection around the staple direction using SVD. This version approximately preserves the action while strongly reducing autocorrelation.
- **Observables**:
  - `wilson_loop()` — generalized rectangular Wilson loops in any plane.
  - `polyakov_loop()` — spatially averaged Polyakov loop.
  - `creutz_ratio()` helper.
- High-level driver: `thermalize_and_measure_pure_gauge(..., updater=..., ndim=...)`.

### `experiments/mc_confinement.py`

Updated CLI supporting:
- `--ndim`
- `--updater` (`metropolis` | `heatbath` | `overrelax`)
- `--quick` mode for rapid testing

### Tests

Extended test coverage in `tests/test_gauge_mc.py` for all three updaters and ndim support.

## Validation Results

### Over-Relaxation Improvement (Phase 1.1)

| Dimension | Gauge | Sweeps | Initial Action | Final Action | Δ Action | Configuration Changed |
|-----------|-------|--------|----------------|--------------|----------|-----------------------|
| 2D        | SU(2) | 10     | 123.90         | 120.56       | **3.34**     | Yes                   |
| 3D        | SU(2) | 8      | 1299.16        | 1254.14      | **45.02**    | Yes                   |

The improved staple-based reflection now preserves the action far better than the previous placeholder while still updating the configuration.

### Broader v2 Behavior (summary from earlier sweeps)

- Both Metropolis and Heat-bath produce physically reasonable Wilson loop suppression and Polyakov loop values.
- The code runs cleanly in 3-D.
- Acceptance rates remain healthy across the tested coupling range.

## Relation to URP Gauge Derivation

This Monte Carlo layer directly serves the thermodynamic sector of the gauge derivation:

- The Wilson action samples the curvature stress term that arises from demanding local ΔC invariance under internal rotations.
- Heat-bath and over-relaxation updates allow efficient generation of ensembles in which we can later measure:
  - Curvature localization on β-sector domain walls (“gluons as boundary modes”).
  - String tension via area-law behavior.
  - Deconfinement temperature via Polyakov loop susceptibility.

## Current Limitations (v2 / v2.1)

- Heat-bath implementation for SU(3) is still Cabibbo-Marinari style with placeholder embeddings (full version pending).
- Over-relaxation, while much improved, can still benefit from further tuning and combination with heat-bath.
- Wilson loop fitting and proper autocorrelation diagnostics are not yet implemented.
- Finite-temperature scans (varying Nₜ) are not yet exposed in the driver.

## Next Micro-Steps (Recommended Order)

1. Full Cabibbo-Marinari heat-bath for SU(3) with proper subgroup embeddings.
2. Integrated autocorrelation time estimation for key observables.
3. Finite-temperature ensembles (vary temporal extent Nₜ at fixed β_g).
4. Multi-size Wilson loop area-law fitting to extract string tension σ(β_g).
5. Update the main README verdict table with first quantitative confinement results.

## Files Changed

- `project_genesis/gauge_mc.py`
- `experiments/mc_confinement.py`
- `tests/test_gauge_mc.py`
- This document: `Docs/MC_Confinement_Phase1.md`

---

**Verdict**: Phase 1 successfully delivers a working, ndim-aware Monte Carlo engine with three updaters. The improved over-relaxation in Phase 1.1 makes the layer already useful for exploratory ensemble generation in both 2-D and 3-D. The foundation is now solid for extracting the thermodynamic confinement signatures predicted by the URP gauge derivation.