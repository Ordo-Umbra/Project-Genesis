"""Finite-size-scaling ladder: is the junction-network melt really a crossover?

The two-size comparison in the phase-boundary experiment suggested the
melt is a crossover (susceptibility bumps that do not grow with volume).
This experiment settles it properly: a ladder of lattice sizes, a fine
temperature grid around the melt, and the standard FSS discriminator —

    χ_max(L) ∝ L^(γ/ν)   at a continuous transition (2-D Ising: γ/ν = 1.75)
    χ_max(L) → const      at a crossover.

The peak height and location per size are extracted with a parabola fit in
ln T through the maximum and its neighbours (grid-discretisation-robust);
the exponent is a weighted least-squares slope of ln χ_max vs ln L.

Physics expectation, stated up front: the corner potential and the pinned
fractions select the sector basis *explicitly*, so there is no symmetry
left to break spontaneously at the melt — a genuine transition is not
expected a priori (and Binder-cumulant analysis is not meaningful here).
The ladder's job is to close the loophole the two-size comparison left,
by measuring the effective exponent with error bars.

Usage::

    python experiments/n3_scaling_ladder.py --output-dir artifacts/n3_ladder
    python experiments/n3_scaling_ladder.py --quick     # smoke run
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

from n3_annealed_matter import run_point  # noqa: E402
from n3_phase_boundary import half_value_temperature  # noqa: E402
from n3_thermal_selection import build_sector_network  # noqa: E402


def parabolic_peak(
    xs: list[float], ys: list[float], errs: list[float]
) -> dict:
    """Peak location/height via a parabola in ln x through the grid maximum.

    Returns ``{"x_peak", "y_peak", "y_peak_err", "at_edge"}``.  When the
    maximum sits on the grid edge the raw values are returned with
    ``at_edge=True`` (the parabola would extrapolate, not interpolate).
    """
    idx = int(np.nanargmax(ys))
    if idx in (0, len(ys) - 1):
        return {"x_peak": xs[idx], "y_peak": ys[idx],
                "y_peak_err": errs[idx], "at_edge": True}
    lx = [math.log(xs[idx - 1]), math.log(xs[idx]), math.log(xs[idx + 1])]
    ly = [ys[idx - 1], ys[idx], ys[idx + 1]]
    a, b, c = np.polyfit(lx, ly, 2)
    if a >= 0:  # not concave: fall back to the grid point
        return {"x_peak": xs[idx], "y_peak": ys[idx],
                "y_peak_err": errs[idx], "at_edge": False}
    x_star = -b / (2 * a)
    y_star = float(np.polyval([a, b, c], x_star))
    return {"x_peak": float(math.exp(x_star)), "y_peak": y_star,
            "y_peak_err": errs[idx], "at_edge": False}


def weighted_loglog_slope(
    ls: list[float], ys: list[float], errs: list[float]
) -> tuple[float, float]:
    """Weighted LSQ slope of ln y vs ln L, with its standard error."""
    lx = np.log(ls)
    ly = np.log(ys)
    w = (np.array(ys) / np.maximum(errs, 1e-12)) ** 2  # var(ln y) ≈ (err/y)²
    sw = w.sum()
    xbar = (w * lx).sum() / sw
    ybar = (w * ly).sum() / sw
    sxx = (w * (lx - xbar) ** 2).sum()
    slope = float((w * (lx - xbar) * (ly - ybar)).sum() / sxx)
    slope_err = float(math.sqrt(1.0 / sxx))
    return slope, slope_err


def make_figure(results, sizes, temps, fit, path):
    """χ(T) per size, the χ_max(L) scaling fit, and melt locations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RAMP = ["#b8d4ea", "#7fb2d9", "#4a90c4", "#0072B2", "#004b78"]
    ORANGE, GRAY = "#E69F00", "#999999"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax in (ax1, ax2, ax3):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for li, L in enumerate(sizes):
        pts = results[L]
        ts = [p["temperature"] for p in pts]
        ax1.errorbar(ts, [p["order_susceptibility"] for p in pts],
                     yerr=[p["order_susceptibility_err"] for p in pts],
                     color=RAMP[li % len(RAMP)], marker="o", ms=4, lw=1.6,
                     capsize=2, label=f"L={L}", zorder=3)
    ax1.set_xlabel("matter temperature T")
    ax1.set_xscale("log")
    ax1.set_ylabel("order susceptibility χ = N·Var(order)")
    ax1.set_title("Susceptibility across the melt, per size")
    ax1.legend(frameon=False)

    peaks = [fit["peaks"][str(L)] for L in sizes]
    ax2.errorbar(sizes, [p["y_peak"] for p in peaks],
                 yerr=[p["y_peak_err"] for p in peaks], color="#0072B2",
                 marker="o", ms=6, lw=0, capsize=3, label="measured χ_max",
                 zorder=3)
    lr = np.array([min(sizes) * 0.9, max(sizes) * 1.1])
    y0 = peaks[0]["y_peak"]
    ax2.plot(lr, y0 * (lr / sizes[0]) ** fit["slope"], color="#0072B2",
             lw=1.4, label=f"fit: slope {fit['slope']:.2f}±{fit['slope_err']:.2f}",
             zorder=2)
    ax2.plot(lr, y0 * (lr / sizes[0]) ** 1.75, color=ORANGE, ls="--", lw=1.2,
             label="transition-like (γ/ν = 1.75)", zorder=2)
    ax2.plot(lr, [y0, y0], color=GRAY, ls=":", lw=1.2,
             label="crossover (slope 0)", zorder=2)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("lattice size L")
    ax2.set_ylabel("χ_max")
    ax2.set_title("Peak-height scaling — the verdict panel")
    ax2.legend(frameon=False)

    ax3.plot(sizes, [fit["t_half"][str(L)] for L in sizes], color="#0072B2",
             marker="o", ms=6, lw=1.6, label="T_half (neutrality)", zorder=3)
    ax3.plot(sizes, [p["x_peak"] for p in peaks], color=ORANGE, marker="s",
             ms=6, lw=1.6, label="T at χ_max", zorder=3)
    ax3.set_xlabel("lattice size L")
    ax3.set_ylabel("temperature")
    ax3.set_title("Melt location vs size")
    ax3.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sizes", type=str, default="24,32,48,64")
    p.add_argument("--n-phases", type=int, default=3)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--diffusion", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.09)
    p.add_argument("--temperatures", type=str,
                   default="0.06,0.08,0.10,0.125,0.155,0.19,0.24,0.30")
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--frac-penalty", type=float, default=20.0)
    p.add_argument("--n-therm", type=int, default=300)
    p.add_argument("--n-meas", type=int, default=200)
    p.add_argument("--n-skip", type=int, default=2)
    p.add_argument("--bin-size", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_ladder")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.sizes = "16,24"
        args.steps = 200
        args.temperatures = "0.08,0.125,0.19,0.3"
        args.n_therm, args.n_meas = 50, 40

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    results: dict[int, list[dict]] = {}
    for L in sizes:
        network = build_sector_network(
            args.n_phases, size=L, steps=args.steps, gamma=args.gamma,
            diffusion=args.diffusion, dt=args.dt, seed=args.seed,
            beta=args.beta)
        pts = []
        for ti, t in enumerate(temps):
            pt = run_point(
                network, temperature=t, beta_g=args.beta_g,
                matter_coupling=args.matter_coupling,
                quartic_u=args.quartic_u, frac_penalty=args.frac_penalty,
                n_therm=args.n_therm, n_meas=args.n_meas,
                n_skip=args.n_skip, bin_size=args.bin_size,
                seed=args.seed + 10000 * L + ti, beta=args.beta)
            pts.append(pt)
            print(f"  L={L} T={t:5.3f}: "
                  f"chi_ord={pt['order_susceptibility']:8.4f}"
                  f"±{pt['order_susceptibility_err']:.4f}  "
                  f"neut={pt['neutrality']:.5f}  acc={pt['acceptance']:.2f}",
                  flush=True)
        results[L] = pts

    peaks = {}
    t_half = {}
    for L in sizes:
        pts = results[L]
        peaks[str(L)] = parabolic_peak(
            temps,
            [p["order_susceptibility"] for p in pts],
            [p["order_susceptibility_err"] for p in pts])
        t_half[str(L)] = half_value_temperature(
            temps, [p["neutrality"] for p in pts])

    slope, slope_err = weighted_loglog_slope(
        sizes,
        [peaks[str(L)]["y_peak"] for L in sizes],
        [peaks[str(L)]["y_peak_err"] for L in sizes])
    fit = {"peaks": peaks, "t_half": t_half,
           "slope": slope, "slope_err": slope_err}

    lines = ["finite-size-scaling ladder — verdict", "=" * 42, ""]
    for L in sizes:
        pk = peaks[str(L)]
        edge = " [at grid edge]" if pk["at_edge"] else ""
        lines.append(
            f"L={L}: chi_max={pk['y_peak']:.4f}±{pk['y_peak_err']:.4f} "
            f"at T={pk['x_peak']:.3f}{edge}   T_half={t_half[str(L)]:.3f}")
    lines.append("")
    lines.append(
        f"chi_max(L) ~ L^b fit: b = {slope:.2f} ± {slope_err:.2f}"
    )
    lines.append(
        "  reference: b = 0 (crossover), b = 1.75 (2-D Ising-like transition)"
    )
    sigmas_from_zero = abs(slope) / slope_err if slope_err else float("nan")
    sigmas_from_ising = abs(slope - 1.75) / slope_err if slope_err else float("nan")
    if sigmas_from_ising > 3 and sigmas_from_zero < 3:
        verdict = ("CROSSOVER confirmed: the peak height does not grow with "
                   f"volume (b consistent with 0 at {sigmas_from_zero:.1f}σ, "
                   f"{sigmas_from_ising:.1f}σ away from transition-like "
                   "scaling)")
    elif sigmas_from_zero > 3 and sigmas_from_ising < 3:
        verdict = ("TRANSITION-like scaling: chi_max grows with volume "
                   f"(b {sigmas_from_zero:.1f}σ from 0)")
    else:
        verdict = (f"INCONCLUSIVE at this precision (b = {slope:.2f}"
                   f"±{slope_err:.2f}; {sigmas_from_zero:.1f}σ from 0, "
                   f"{sigmas_from_ising:.1f}σ from 1.75)")
    lines.append("verdict: " + verdict)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {
        "params": vars(args),
        "results": {str(L): results[L] for L in sizes},
        "fit": fit,
    }
    with open(os.path.join(args.output_dir, "n3_ladder.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_ladder.png")
        make_figure(results, sizes, temps, fit, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
