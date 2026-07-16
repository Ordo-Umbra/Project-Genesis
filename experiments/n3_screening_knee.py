"""The screening knee: gravity's range moved into the window by the field's own dial.

The growth spectrum's 0/3 verdict came with a measured mechanism: the S(λ)
instrument is band-passed — a UV wall at the particle footprint, an IR wall at
κ screening — and the whole accessible window sat *beyond* the screening knee,
whose resolution was recorded as needing ``footprint ≪ 2πℓ ≪ box``: a compute
rung.  But the condition names ℓ, not the box — and the screening length is
the capacity field's **own dial**.  Turning the recovery rate ``r`` down moves
the knee *into* the existing window, and scanning the dial makes a sharper
test than any single point: the knee must **track** the field's law.

Calibration taught the law's correct form — and it is not the vacuum one.
Fitted ranges at heavy mass stalled far short of ``√(D_κ/r)``; linearising
the capacity equation about the **loaded** homogeneous steady state (uniform
matter density ⟨ρ⟩) instead gives a Debye-like screened mode

    (D_κ∇² − (r + c·⟨ρ⟩))·δκ = c·κ̄·δρ   ⇒   ℓ_eff = √(D_κ / (r + c·⟨ρ⟩)) ,

i.e. **matter consumes capacity, and consumed capacity screens gravity** —
κ-gravity is shorter-ranged inside matter, the analogue's own plasma-style
screening.  Four independent calibration points (two masses × two recovery
rates) match this loaded form to ~10% where the vacuum form is off by ×2.
The registered scan therefore tests the loaded law, and its intercept makes
the matter term itself measurable from growth data.

Instruments:

- **S(λ) ladder at each recovery rate** — static runs; the parent's
  estimator fitted ``ln δ`` on every point with ``1.1 < δ < 3.0``, but after
  shell-crossing δ re-enters that band and poisons the slope (measured: S at
  λ = 10.7 collapses 400× between 60- and 140-step runs), so the estimator
  here uses only the **first contiguous** window crossing.
- **The band-pass model, fitted per r**: ``S(k) = S₀·e^{−k²w²}·(kℓ)²/(1+(kℓ)²)``
  — the derivable footprint factor times screened κ-gravity's own form; two
  parameters (S₀, ℓ), fitted on ``ln S``.  The ladder stops at the 5th
  harmonic: λ = 8 is the particle grid's Nyquist mode (biased projection),
  and n ≥ 6 (< 3 particles per wavelength) measurably inflates S by
  aliasing, pulling the fitted range down.
- **A saturation gauge** — min/mean of the relaxed κ under the initial
  particle grid, recorded per row.  The design probe found this wall first:
  at the parent's mass 0.3 the κ-wells floor at slow recovery (the source
  collapses ~500× at r = 0.05), which is why the registered mass is light.

Pre-registered predictions:

K1. **The knee resolves**: at the slowest recovery in the scan the band-pass
    fit achieves ``R² > 0.9`` (on ln S) *and* the fitted knee ``2πℓ`` lies
    strictly inside the sampled ladder — the parent's walls measured from
    both sides at last, in the same box.
K2. **The knee is the field's — in the loaded vacuum**: ``ℓ_fit`` within a
    factor of 2 of ``√(D_κ/(r + c⟨ρ⟩))`` at every r in the scan.
K3. **The matter term is real and measurable**: regressing ``1/ℓ_fit²`` on
    ``r`` recovers both field constants — ``D_κ`` from the slope and the
    matter screening ``μ = c⟨ρ⟩`` from the intercept, each within a factor
    of 2 — gravity's range inside matter, read off structure growth.

Usage::

    python experiments/n3_screening_knee.py --output-dir artifacts/n3_knee
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.capacity_gravity import (
    gaussian_load,
    relax_capacity,
    screening_length,
)
from n3_growth_factor import growth_run


def measure_source(args, wavelength: float) -> float:
    """S(λ): early-window growth rate squared from a static run.

    The parent's estimator masked on ``1.1 < δ < 3.0`` over the *whole* run —
    but after shell-crossing δ oscillates and re-enters that band, poisoning
    the slope (measured: S at λ = 10.7 collapses 400× between 60- and
    140-step runs).  Here the fit is restricted to the **first contiguous**
    crossing of the linear window.
    """
    run = growth_run(args, wavelength, 0.0, static=True)
    d = np.asarray(run["D"])
    t = args.dt * np.arange(len(d))
    above = np.nonzero(d > 1.1)[0]
    if len(above) == 0:
        return float("nan")
    i0 = int(above[0])
    past = np.nonzero(d[i0:] >= 3.0)[0]
    i1 = i0 + (int(past[0]) if len(past) else len(d) - i0)
    sel = slice(i0, i1)
    if i1 - i0 < 6 or np.any(d[sel] <= 0):
        return float("nan")
    slope = float(np.polyfit(t[sel], np.log(d[sel]), 1)[0])
    return slope ** 2


def band_pass_fit(ks, svals, width: float) -> dict:
    """Fit ``ln S = ln S₀ − k²w² + ln((kℓ)²/(1+(kℓ)²))``; ℓ by grid scan."""
    ks, svals = np.asarray(ks, float), np.asarray(svals, float)
    ok = np.isfinite(svals) & (svals > 0)
    k, y = ks[ok], np.log(svals[ok])
    base = -(k ** 2) * width ** 2
    best = (np.inf, 0.0, np.nan)
    for ell in np.geomspace(0.25, 20.0, 600):
        screen = np.log(k ** 2 * ell ** 2 / (1.0 + k ** 2 * ell ** 2))
        resid = y - base - screen
        ln_s0 = float(resid.mean())
        sse = float(np.sum((resid - ln_s0) ** 2))
        if sse < best[0]:
            best = (sse, ln_s0, ell)
    sse, ln_s0, ell = best
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"S0": float(np.exp(ln_s0)), "ell": float(ell),
            "knee": float(2.0 * np.pi * ell),
            "r2": 1.0 - sse / ss_tot, "n_points": int(ok.sum())}


def grid_load(args) -> np.ndarray:
    """The exact load field of the initial uniform particle grid."""
    L, n = args.box, args.grid
    q = np.linspace(0, L, n, endpoint=False) + L / (2 * n)
    qx, qy = np.meshgrid(q, q, indexing="ij")
    return gaussian_load((int(L), int(L)),
                         list(zip(qx.ravel(), qy.ravel())),
                         args.width, args.mass)


def saturation_gauge(args) -> tuple[float, float]:
    """(min κ, mean κ) of the relaxed field under the initial uniform grid."""
    kappa = relax_capacity(grid_load(args),
                           kappa_recovery=args.kappa_recovery)
    return float(kappa.min()), float(kappa.mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--box", type=float, default=64.0)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--amplitude", type=float, default=0.10)
    p.add_argument("--mass", type=float, default=0.015)
    p.add_argument("--recoveries", type=float, nargs="+",
                   default=[1.0, 0.4, 0.2, 0.1, 0.05])
    p.add_argument("--harmonics", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5])
    p.add_argument("--hubble0", type=float, default=0.05)
    p.add_argument("--omega-m", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--base-steps", type=int, default=160)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_knee")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.recoveries = [1.0, 0.1]
        args.base_steps = 80

    os.makedirs(args.output_dir, exist_ok=True)
    print("the screening knee: the field's own dial moves gravity's range",
          flush=True)
    wavelengths = [args.box / n for n in args.harmonics]
    # c is relax_capacity's kappa_consumption default (2.0) — the constant
    # actually used inside the runs; it is deliberately not a CLI flag here
    # because forwarding it would also rescale the force prefactor.
    mu_theory = 2.0 * float(grid_load(args).mean())
    print(f"  matter screening term: mu = c*<rho> = {mu_theory:.4f}",
          flush=True)

    scan = []
    for r in sorted(args.recoveries, reverse=True):
        args.kappa_recovery = r
        args.steps = int(round(args.base_steps / np.sqrt(r)))
        k_min, k_mean = saturation_gauge(args)
        row = {"recovery": r, "steps": args.steps,
               "ell_vacuum": screening_length(1.0, r),
               "ell_loaded": float(np.sqrt(1.0 / (r + mu_theory))),
               "kappa_min": k_min, "kappa_mean": k_mean, "spectrum": []}
        for lam in wavelengths:
            s = measure_source(args, lam)
            row["spectrum"].append({"wavelength": lam,
                                    "k": 2.0 * np.pi / lam, "S": s})
        row["fit"] = band_pass_fit([e["k"] for e in row["spectrum"]],
                                   [e["S"] for e in row["spectrum"]],
                                   args.width)
        print(f"  r={r:<5}: ℓ_fit={row['fit']['ell']:.2f} "
              f"(loaded {row['ell_loaded']:.2f}, vacuum "
              f"{row['ell_vacuum']:.2f}), knee 2πℓ={row['fit']['knee']:.1f}, "
              f"R²={row['fit']['r2']:.3f}, κ_min={k_min:.2f}", flush=True)
        scan.append(row)

    slow = scan[-1]
    lam_lo, lam_hi = min(wavelengths), max(wavelengths)
    k1 = (slow["fit"]["r2"] > 0.9
          and lam_lo < slow["fit"]["knee"] < lam_hi)
    ratios = [row["fit"]["ell"] / row["ell_loaded"] for row in scan]
    k2 = all(np.isfinite(q) and 0.5 <= q <= 2.0 for q in ratios)
    # Debye regression: 1/ell^2 = r/D + mu/D — both field constants at once
    rvals = np.array([row["recovery"] for row in scan])
    inv_ell2 = np.array([row["fit"]["ell"] ** -2 for row in scan])
    coef = np.polyfit(rvals, inv_ell2, 1)
    d_fit = 1.0 / coef[0] if coef[0] > 0 else float("nan")
    mu_fit = coef[1] * d_fit if np.isfinite(d_fit) else float("nan")
    k3 = (np.isfinite(d_fit) and 0.5 <= d_fit <= 2.0
          and np.isfinite(mu_fit) and 0.5 <= mu_fit / mu_theory <= 2.0)

    lines = ["The screening knee — verdict", "=" * 74, ""]
    lines.append(
        f"K1 (the knee resolves): slowest recovery r = {slow['recovery']:g}: "
        f"R² = {slow['fit']['r2']:.3f}, knee 2πℓ = {slow['fit']['knee']:.1f} "
        f"vs ladder [{lam_lo:g}, {lam_hi:g}] — "
        + ("✓ the parent's walls are now measured from both sides — same "
           "box, no compute rung." if k1 else
           "✗ the knee still escapes the window."))
    lines.append(
        "K2 (the knee is the field's, in the loaded vacuum): "
        "ℓ_fit/√(D_κ/(r+c⟨ρ⟩)) = "
        + ", ".join(f"{q:.2f}" for q in ratios) + " — "
        + ("✓ within a factor of 2 at every recovery rate." if k2 else
           "✗ the fitted range departs from the loaded screening length."))
    lines.append(
        f"K3 (the matter term is measurable): 1/ℓ² vs r regression gives "
        f"D_κ = {d_fit:.2f} (true 1.0) and μ = {mu_fit:.4f} "
        f"(c⟨ρ⟩ = {mu_theory:.4f}) — "
        + ("✓ both field constants read off structure growth — including "
           "the matter screening itself." if k3 else
           "✗ the regression does not recover the field constants."))
    lines.append("")
    lines.append(f"score: {int(k1)+int(k2)+int(k3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: 2-D analogue, one box/grid/amplitude, as the "
        "parents; the dial is r with D_κ fixed (the relax integrator's "
        "stability caps D_κ, so recovery is the accessible half of the "
        "ratio). The loaded form of ℓ was derived during calibration after "
        "the vacuum form failed at heavy mass (four points, masses 0.1 and "
        "0.3); the operating mass was then fixed by a two-point corner "
        "check at r = 1 and 0.05, so the scan's middle rows are untouched "
        "predictions. "
        "⟨ρ⟩ is the mean-field reading of a corrugated background; the "
        "footprint factor exp(−k²w²) is imposed from the parent's "
        "diagnosis, not refit; the saturation gauge (min κ) is recorded "
        "per row — the parent's mass 0.3 floors the soil at slow recovery, "
        "which is why the registered mass is 20× lighter.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_screening_knee.json"),
              "w") as fh:
        json.dump({"params": vars(args), "scan": scan,
                   "analysis": {"k1": bool(k1), "k2": bool(k2),
                                "k3": bool(k3), "ell_ratios": ratios,
                                "mu_theory": mu_theory,
                                "mu_fit": mu_fit, "d_fit": d_fit}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
