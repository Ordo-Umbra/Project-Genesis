"""Extracting ν: Binder-cumulant data collapse of the unpinned transition.

The unpinned S_P transition's susceptibility exponent landed on the exact
2-D 3-state Potts value (γ/ν = 26/15).  This experiment double-locks the
universality identification through the *second* independent exponent: at
a continuous transition the Binder cumulant obeys single-parameter
scaling,

    U₄(T, L) = f( (T − T_c) · L^{1/ν} ),

so plotting U₄ against the scaled variable collapses every lattice size
onto one curve — for exactly one (T_c, ν) pair.  A grid search over
(T_c, ν) minimising the collapse residual measures both at once.
2-D 3-state Potts: ν = 5/6 ≈ 0.833.

Method:
- dense U₄(T, L) scan in a bracket around T_c, ladder of sizes,
- for each candidate (T_c, ν): map every point to x = (T−T_c)·L^{1/ν},
  linearly interpolate each size's curve, and take the weighted mean
  squared mismatch between sizes over the overlapping x-range,
- report the minimising (T_c, ν) and the ν interval where the residual is
  within 2× its minimum (a pragmatic confidence band, honest about being
  a consistency measurement rather than a precision one).

Usage::

    python experiments/n3_potts_nu.py --output-dir artifacts/n3_potts_nu
    python experiments/n3_potts_nu.py --quick     # smoke run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from n3_potts_transition import run_point  # noqa: E402

POTTS_NU_2D = 5.0 / 6.0  # 2-D 3-state Potts, exact


def collapse_residual(
    results: dict[int, list[dict]],
    sizes: list[int],
    t_c: float,
    nu: float,
) -> float:
    """Weighted mean squared mismatch of U₄ curves in the scaled variable."""
    curves = []
    for L in sizes:
        pts = results[L]
        x = np.array([(p["temperature"] - t_c) * L ** (1.0 / nu)
                      for p in pts])
        y = np.array([p["binder"] for p in pts])
        w = np.array([1.0 / max(p["binder_err"], 1e-3) ** 2 for p in pts])
        order = np.argsort(x)
        curves.append((x[order], y[order], w[order]))

    total, count = 0.0, 0
    for i in range(len(curves)):
        xi, yi, wi = curves[i]
        for j in range(len(curves)):
            if i == j:
                continue
            xj, yj, wj = curves[j]
            lo, hi = max(xi[0], xj[0]), min(xi[-1], xj[-1])
            mask = (xi >= lo) & (xi <= hi)
            if not mask.any():
                continue
            y_interp = np.interp(xi[mask], xj, yj)
            diff = (yi[mask] - y_interp) ** 2 * wi[mask]
            total += float(diff.sum())
            count += int(mask.sum())
    return total / count if count else float("inf")


def fit_collapse(
    results: dict[int, list[dict]],
    sizes: list[int],
    t_c_grid: np.ndarray,
    nu_grid: np.ndarray,
) -> dict:
    """Grid-search (T_c, ν) minimising the collapse residual."""
    res = np.empty((len(t_c_grid), len(nu_grid)))
    for i, t_c in enumerate(t_c_grid):
        for j, nu in enumerate(nu_grid):
            res[i, j] = collapse_residual(results, sizes, t_c, nu)
    i0, j0 = np.unravel_index(np.argmin(res), res.shape)
    r_min = float(res[i0, j0])
    # ν band where some T_c keeps the residual within 2× the minimum
    nu_ok = [nu_grid[j] for j in range(len(nu_grid))
             if res[:, j].min() <= 2.0 * r_min]
    return {
        "t_c": float(t_c_grid[i0]),
        "nu": float(nu_grid[j0]),
        "residual_min": r_min,
        "nu_band": [float(min(nu_ok)), float(max(nu_ok))] if nu_ok else None,
        "residual_grid": res.tolist(),
        "t_c_grid": t_c_grid.tolist(),
        "nu_grid": nu_grid.tolist(),
    }


def make_figure(results, sizes, fit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RAMP = ["#b8d4ea", "#7fb2d9", "#4a90c4", "#0072B2", "#004b78"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax in (ax1, ax2, ax3):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for li, L in enumerate(sizes):
        pts = results[L]
        ts = [p["temperature"] for p in pts]
        ax1.errorbar(ts, [p["binder"] for p in pts],
                     yerr=[p["binder_err"] for p in pts],
                     color=RAMP[li % len(RAMP)], marker="o", ms=4, lw=1.4,
                     capsize=2, label=f"L={L}", zorder=3)
        x = [(p["temperature"] - fit["t_c"]) * L ** (1.0 / fit["nu"])
             for p in pts]
        ax2.errorbar(x, [p["binder"] for p in pts],
                     yerr=[p["binder_err"] for p in pts],
                     color=RAMP[li % len(RAMP)], marker="o", ms=4, lw=1.4,
                     capsize=2, label=f"L={L}", zorder=3)
    ax1.set_xlabel("matter temperature T")
    ax1.set_ylabel("Binder U₄")
    ax1.set_title("Raw Binder curves")
    ax1.legend(frameon=False, fontsize=8)
    ax2.set_xlabel(f"(T − {fit['t_c']:.3f})·L^(1/{fit['nu']:.2f})")
    ax2.set_ylabel("Binder U₄")
    ax2.set_title("Best-fit data collapse")
    ax2.legend(frameon=False, fontsize=8)

    res = np.array(fit["residual_grid"])
    nu_grid = np.array(fit["nu_grid"])
    prof = res.min(axis=0)
    ax3.plot(nu_grid, prof / fit["residual_min"], color="#0072B2", lw=1.6,
             zorder=3)
    ax3.axhline(2.0, color="#999999", lw=0.9, ls=":",
                label="2× minimum band", zorder=2)
    ax3.axvline(POTTS_NU_2D, color="#E69F00", lw=1.2, ls="--",
                label="Potts ν = 5/6", zorder=2)
    ax3.axvline(fit["nu"], color="#0072B2", lw=1.2,
                label=f"best fit ν = {fit['nu']:.2f}", zorder=2)
    ax3.set_xlabel("ν")
    ax3.set_ylabel("collapse residual / minimum")
    ax3.set_yscale("log")
    ax3.set_title("Residual profile over ν")
    ax3.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-phases", type=int, default=3)
    p.add_argument("--sizes", type=str, default="16,24,32,48")
    p.add_argument("--temperatures", type=str,
                   default="0.07,0.085,0.10,0.115,0.13,0.15,0.17")
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--n-therm", type=int, default=400)
    p.add_argument("--n-meas", type=int, default=400)
    p.add_argument("--n-skip", type=int, default=2)
    p.add_argument("--bin-size", type=int, default=10)
    p.add_argument("--step-scale", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_potts_nu")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.sizes = "12,16"
        args.temperatures = "0.08,0.11,0.15"
        args.n_therm, args.n_meas = 80, 60

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    results: dict[int, list[dict]] = {}
    for L in sizes:
        pts = []
        for ti, t in enumerate(temps):
            pt = run_point(
                n_palette=args.n_phases, size=L, temperature=t,
                beta_g=args.beta_g, matter_coupling=args.matter_coupling,
                quartic_u=args.quartic_u, n_therm=args.n_therm,
                n_meas=args.n_meas, n_skip=args.n_skip,
                bin_size=args.bin_size, seed=args.seed + 10000 * L + ti,
                step_scale=args.step_scale)
            pts.append(pt)
            print(f"  L={L} T={t:5.3f}: U4={pt['binder']:.4f}"
                  f"±{pt['binder_err']:.4f}  m={pt['magnetisation']:.4f}  "
                  f"acc={pt['acceptance']:.2f}", flush=True)
        results[L] = pts

    fit = fit_collapse(
        results, sizes,
        t_c_grid=np.linspace(0.08, 0.14, 25),
        nu_grid=np.linspace(0.4, 1.6, 49))

    lines = ["Binder data collapse — ν extraction", "=" * 42, ""]
    lines.append(f"best collapse: T_c = {fit['t_c']:.4f}, ν = {fit['nu']:.3f}")
    if fit["nu_band"]:
        lines.append(f"ν band (residual ≤ 2× min): "
                     f"[{fit['nu_band'][0]:.2f}, {fit['nu_band'][1]:.2f}]")
    lines.append(f"reference: 2-D 3-state Potts ν = 5/6 ≈ {POTTS_NU_2D:.3f}")
    in_band = (fit["nu_band"] is not None
               and fit["nu_band"][0] <= POTTS_NU_2D <= fit["nu_band"][1])
    lines.append(
        "verdict: Potts ν "
        + ("INSIDE the collapse band — both independent exponents (γ/ν and "
           "ν) now agree with the 2-D 3-state Potts class"
           if in_band else
           "outside the collapse band — inspect the residual profile")
    )
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": vars(args),
               "results": {str(L): results[L] for L in sizes},
               "fit": {k: v for k, v in fit.items()
                       if k != "residual_grid"},
               "residual_grid": fit["residual_grid"]}
    with open(os.path.join(args.output_dir, "n3_potts_nu.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_potts_nu.png")
        make_figure(results, sizes, fit, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
