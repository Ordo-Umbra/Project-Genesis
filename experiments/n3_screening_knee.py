"""The screening knee: gravity's range moved into the window by the field's own dial.

The growth spectrum's 0/3 verdict came with a measured mechanism: the S(λ)
instrument is band-passed — a UV wall at the particle footprint, an IR wall at
κ screening — and the whole accessible window sat *beyond* the screening knee,
whose resolution was recorded as needing ``footprint ≪ 2πℓ ≪ box``: a compute
rung.  But the condition names ℓ, not the box — and ``ℓ = √(D_κ/r)`` is the
capacity field's **own dial**.  Turning the recovery rate ``r`` down moves the
knee *into* the existing window, and scanning the dial makes a sharper test
than any single point: the knee must **track** ``√(D_κ/r)``.

Instruments:

- **S(λ) ladder at each recovery rate** — static runs, the parent's exact
  early-window estimator (``ln δ`` vs ``t`` over the linear window), with the
  step count scaled as ``1/√r`` so the slow-recovery window is still reached.
- **The band-pass model, fitted per r**: ``S(k) = S₀·e^{−k²w²}·(kℓ)²/(1+(kℓ)²)``
  — the derivable footprint factor times screened κ-gravity's own form; two
  parameters (S₀, ℓ), fitted on ``ln S``.
- **A saturation gauge** — min/mean of the relaxed κ under the initial
  particle grid, recorded per r.  The design probe found the walls of this
  instrument too: at ``r = 0.05`` the soil floors (κ-wells clip at 0), the
  gradients flatten, and the source collapses ~500× — so the scan stops at
  ``r = 0.1``, with the gauge kept in the record rather than discovered
  post hoc.

Pre-registered predictions:

K1. **The knee resolves**: at the slowest recovery in the scan the band-pass
    fit achieves ``R² > 0.9`` (on ln S) *and* the fitted knee ``2πℓ`` lies
    strictly inside the sampled ladder — the parent's walls measured from
    both sides at last.
K2. **The knee is the field's**: ``ℓ_fit`` within a factor of 2 of
    ``√(D_κ/r)`` at every r in the scan.
K3. **The knee moves as the law says**: the slope of ``ln ℓ_fit`` vs
    ``ln r`` is ``−0.5 ± 0.15`` — gravity's range doesn't just match one
    theory number, it tracks the dial.

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


def saturation_gauge(args) -> tuple[float, float]:
    """(min κ, mean κ) of the relaxed field under the initial uniform grid."""
    L, n = args.box, args.grid
    q = np.linspace(0, L, n, endpoint=False) + L / (2 * n)
    qx, qy = np.meshgrid(q, q, indexing="ij")
    load = gaussian_load((int(L), int(L)),
                         list(zip(qx.ravel(), qy.ravel())),
                         args.width, args.mass)
    kappa = relax_capacity(load, kappa_recovery=args.kappa_recovery)
    return float(kappa.min()), float(kappa.mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--box", type=float, default=64.0)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--amplitude", type=float, default=0.10)
    p.add_argument("--mass", type=float, default=0.3)
    p.add_argument("--recoveries", type=float, nargs="+",
                   default=[1.0, 0.4, 0.2, 0.1])
    p.add_argument("--harmonics", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--hubble0", type=float, default=0.05)
    p.add_argument("--omega-m", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--base-steps", type=int, default=100)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_knee")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.recoveries = [1.0, 0.2]
        args.harmonics = [1, 2, 4, 6, 8]
        args.base_steps = 60

    os.makedirs(args.output_dir, exist_ok=True)
    print("the screening knee: the field's own dial moves gravity's range",
          flush=True)
    wavelengths = [args.box / n for n in args.harmonics]

    scan = []
    for r in sorted(args.recoveries, reverse=True):
        args.kappa_recovery = r
        args.steps = int(round(args.base_steps / np.sqrt(r)))
        k_min, k_mean = saturation_gauge(args)
        row = {"recovery": r, "steps": args.steps,
               "ell_theory": screening_length(1.0, r),
               "kappa_min": k_min, "kappa_mean": k_mean, "spectrum": []}
        for lam in wavelengths:
            s = measure_source(args, lam)
            row["spectrum"].append({"wavelength": lam,
                                    "k": 2.0 * np.pi / lam, "S": s})
        row["fit"] = band_pass_fit([e["k"] for e in row["spectrum"]],
                                   [e["S"] for e in row["spectrum"]],
                                   args.width)
        print(f"  r={r:<5}: ℓ_fit={row['fit']['ell']:.2f} "
              f"(theory {row['ell_theory']:.2f}), knee 2πℓ="
              f"{row['fit']['knee']:.1f}, R²={row['fit']['r2']:.3f}, "
              f"κ_min={k_min:.2f}", flush=True)
        scan.append(row)

    slow = scan[-1]
    lam_lo, lam_hi = min(wavelengths), max(wavelengths)
    k1 = (slow["fit"]["r2"] > 0.9
          and lam_lo < slow["fit"]["knee"] < lam_hi)
    ratios = [row["fit"]["ell"] / row["ell_theory"] for row in scan]
    k2 = all(np.isfinite(q) and 0.5 <= q <= 2.0 for q in ratios)
    ln_r = np.log([row["recovery"] for row in scan])
    ln_ell = np.log([row["fit"]["ell"] for row in scan])
    slope = float(np.polyfit(ln_r, ln_ell, 1)[0])
    k3 = abs(slope - (-0.5)) <= 0.15

    lines = ["The screening knee — verdict", "=" * 74, ""]
    lines.append(
        f"K1 (the knee resolves): slowest recovery r = {slow['recovery']:g}: "
        f"R² = {slow['fit']['r2']:.3f}, knee 2πℓ = {slow['fit']['knee']:.1f} "
        f"vs ladder [{lam_lo:g}, {lam_hi:g}] — "
        + ("✓ the parent's walls are now measured from both sides — no "
           "bigger box required." if k1 else
           "✗ the knee still escapes the window."))
    lines.append(
        "K2 (the knee is the field's): ℓ_fit/√(D_κ/r) = "
        + ", ".join(f"{q:.2f}" for q in ratios) + " — "
        + ("✓ within a factor of 2 at every recovery rate." if k2 else
           "✗ the fitted range departs from the field's screening length."))
    lines.append(
        f"K3 (the knee moves as the law says): d ln ℓ / d ln r = {slope:.2f} "
        f"vs −0.5 — "
        + ("✓ gravity's range tracks the dial with the law's own exponent."
           if k3 else "✗ the scaling is off the √(D_κ/r) law."))
    lines.append("")
    lines.append(f"score: {int(k1)+int(k2)+int(k3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: 2-D analogue, one box/grid/mass, as the parents; the "
        "dial is r with D_κ fixed (the relax integrator's stability caps "
        "D_κ, so recovery is the accessible half of ℓ = √(D_κ/r)); the "
        "footprint factor exp(−k²w²) is imposed from the parent's diagnosis, "
        "not refit. Slower recovery deepens the κ-wells toward the floor — "
        "the saturation gauge is recorded per row, and r = 0.05 measured "
        "unusable at this mass. S₀ varies with r (the force normalisation "
        "is recovery-dependent), which the per-row fit absorbs.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_screening_knee.json"),
              "w") as fh:
        json.dump({"params": vars(args), "scan": scan,
                   "analysis": {"k1": bool(k1), "k2": bool(k2),
                                "k3": bool(k3), "ell_ratios": ratios,
                                "slope": slope}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
