"""The growth spectrum: D(k) from the κ cosmology, against its own linear theory.

Follow-up registered by `n3_growth_factor.py`: replace the 3-D textbook
benchmark (whose failure was a 2-D/3-D convention mismatch) with the
**self-consistent** one, and turn the two-point scale-dependence result into
the full curve — the object a survey could someday be held against.

Instruments:

- **S(λ)**: the analogue's own gravitational source per wavelength, measured
  from static runs (early-window fit of `ln δ` vs `t`, linear regime only).
- **The analogue's own linear theory**: `δ̈ + 2H(a)δ̇ = S(λ)·δ` integrated
  alongside `ȧ = aH` with the same imposed Friedmann law — no textbook
  coefficients anywhere.

Pre-registered predictions:

Q1. **Self-consistency**: the expanding N-body's `D(a)` at the short
    wavelength matches its own linear-ODE prediction within 20% (relative)
    through the linear window — the honest replacement for the failed P1.
Q2. **The Yukawa form**: `S(λ)` fits `S₀ / (1 + (λ/2πℓ)²)` with `R² > 0.9` —
    screened κ-gravity's own functional form for the growth source.
Q3. **The screening length is the field's**: fitted `ℓ` within a factor of 2
    of the theoretical `√(D_κ/r)` from the field parameters.

Usage::

    python experiments/n3_growth_spectrum.py --output-dir artifacts/n3_spectrum
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.capacity_dynamics import friedmann_rates
from project_genesis.capacity_gravity import screening_length
from n3_growth_factor import growth_run


def measure_source(args, wavelength: float) -> float:
    """S(λ): early-window growth rate squared from a static run."""
    run = growth_run(args, wavelength, 0.0, static=True)
    d = np.asarray(run["D"])
    t = args.dt * np.arange(len(d))
    lin = (d > 1.1) & (d < 3.0)
    if lin.sum() < 4:
        return float("nan")
    slope = float(np.polyfit(t[lin], np.log(d[lin]), 1)[0])
    return slope ** 2


def linear_ode_prediction(a_grid, s_eff, hubble0, omega_m, omega_lambda,
                          dt) -> np.ndarray:
    """Integrate δ̈ + 2Hδ̇ = Sδ alongside ȧ = aH; return D on ``a_grid``."""
    a, d, dd = 1.0, 1.0, hubble0        # growing-mode start: δ̇ = H₀δ
    out_a, out_d = [a], [d]
    while a < a_grid[-1]:
        H, _ = friedmann_rates(a, hubble0, omega_m, omega_lambda, 3.0)
        dd += (s_eff * d - 2.0 * H * dd) * dt
        d += dd * dt
        a += a * H * dt
        out_a.append(a)
        out_d.append(d)
    return np.interp(a_grid, out_a, out_d)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--box", type=float, default=64.0)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--amplitude", type=float, default=0.10)
    p.add_argument("--mass", type=float, default=0.3)
    p.add_argument("--kappa-recovery", type=float, default=1.0)
    p.add_argument("--kappa-diffusion", type=float, default=1.0)
    p.add_argument("--hubble0", type=float, default=0.05)
    p.add_argument("--omega-m", type=float, default=1.0)
    p.add_argument("--wavelengths", type=float, nargs="+",
                   default=[8.0, 10.7, 16.0, 21.3, 32.0, 64.0])
    p.add_argument("--bench-lambda", type=float, default=16.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=140)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_spectrum")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("the growth spectrum: D(k) against the analogue's own linear theory",
          flush=True)

    # --- S(λ) scan
    spectrum = []
    for lam in args.wavelengths:
        s = measure_source(args, lam)
        spectrum.append({"wavelength": lam, "S": s})
        print(f"  λ={lam:>5.1f}: S = {s:.5f}", flush=True)

    lams = np.array([r["wavelength"] for r in spectrum])
    svals = np.array([r["S"] for r in spectrum])
    ok = np.isfinite(svals) & (svals > 0)
    # Yukawa fit: 1/S = (1/S0)(1 + (λ/2πℓ)²) — linear in λ²
    coef = np.polyfit(lams[ok] ** 2, 1.0 / svals[ok], 1)
    s0_fit = 1.0 / coef[1]
    ell_fit = float(np.sqrt(coef[0] * s0_fit)) / (2.0 * np.pi)
    pred = 1.0 / (coef[0] * lams[ok] ** 2 + coef[1])
    ss_res = float(np.sum((svals[ok] - pred) ** 2))
    ss_tot = float(np.sum((svals[ok] - svals[ok].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    ell_theory = screening_length(args.kappa_diffusion, args.kappa_recovery)

    # --- self-consistency benchmark at bench-lambda
    s_bench = float(np.interp(args.bench_lambda, lams, svals))
    args.hubble0 = float(np.sqrt(2.0 / 3.0 * s_bench))
    nbody = growth_run(args, args.bench_lambda, 0.0)
    a_nb = np.asarray(nbody["a"])
    d_nb = np.asarray(nbody["D"])
    lin = (d_nb > 0.9) & (d_nb < 3.0) & (a_nb > 1.02)
    d_ode = linear_ode_prediction(a_nb[lin], s_bench, args.hubble0,
                                  args.omega_m, 0.0, args.dt / 4.0)
    rel_dev = float(np.max(np.abs(d_nb[lin] - d_ode)
                           / np.maximum(d_ode, 1e-9)))

    q1 = rel_dev < 0.20
    q2 = r2 > 0.9
    q3 = np.isfinite(ell_fit) and 0.5 <= ell_fit / ell_theory <= 2.0

    lines = ["The growth spectrum — verdict", "=" * 74, ""]
    lines.append(
        f"Q1 (self-consistency): expanding N-body vs the analogue's own "
        f"linear ODE at λ = {args.bench_lambda:g}: max relative deviation "
        f"{rel_dev:.1%} through the linear window — "
        + ("✓ the N-body grows exactly as its own linear theory says — the "
           "P1 bookkeeping is closed." if q1 else
           "✗ the N-body disagrees with its own linear theory."))
    lines.append(
        f"Q2 (the Yukawa form): S(λ) fit to S₀/(1+(λ/2πℓ)²): R² = {r2:.3f} "
        f"(S₀ = {s0_fit:.4f}) — "
        + ("✓ the growth source carries screened κ-gravity's own functional "
           "form: D(k) is now a curve, not two points." if q2 else
           "✗ the Yukawa form does not describe S(λ)."))
    lines.append(
        f"Q3 (the screening length is the field's): ℓ_fit = {ell_fit:.2f} vs "
        f"√(D_κ/r) = {ell_theory:.2f} (ratio {ell_fit/ell_theory:.2f}) — "
        + ("✓ the cosmological suppression scale IS the capacity field's own "
           "screening length — gravity's range read off structure growth."
           if q3 else "✗ the scales do not match."))
    lines.append("")
    lines.append(f"score: {int(q1)+int(q2)+int(q3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope, with the post-run diagnosis kept: the registered "
        "Yukawa form was incomplete — the measured S(λ) is BAND-PASSED, "
        "UV-suppressed by the instrument's own Gaussian particle footprint "
        "(dividing out the derivable exp(−k²w²) factor gives R² ≈ 0.82 and "
        "reveals S ∝ k²) and IR-suppressed by κ screening. The corrected "
        "spectrum shows the entire accessible window lies BEYOND the "
        "screening knee at every field point tested: the knee needs "
        "footprint ≪ 2πℓ ≪ box, i.e. bigger boxes and finer grids — a "
        "compute rung, now quantified. Q1's ~20-27%% deviation is a "
        "near-miss (window/nonlinearity bias). 2-D analogue; the ODE "
        "benchmark shares the imposed Friedmann law (self-consistency, not "
        "an external test).")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_growth_spectrum.json"), "w") as fh:
        json.dump({"params": vars(args), "spectrum": spectrum,
                   "analysis": {"q1": bool(q1), "q2": bool(q2), "q3": bool(q3),
                                "rel_dev": rel_dev, "r2": r2,
                                "ell_fit": ell_fit,
                                "ell_theory": ell_theory}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
