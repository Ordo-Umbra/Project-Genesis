"""Mapping the junction-network melting boundary in the (g_m, T) plane.

The annealed-matter experiment measured a single melting point
(T_melt ≈ 0.2 at g_m = 4).  This experiment maps the boundary: for each
matter–gauge coupling g_m it scans the matter temperature T, measures the
junction-network observables of the joint (ψ, U) ensemble, and locates the
melt by two independent criteria:

- **half-value**: the (interpolated) T where the full-palette junction
  density falls to half its coldest value;
- **susceptibility peak**: the T maximising the order-parameter
  susceptibility χ = N·Var(⟨max_a|ψ_a|²⟩).

Crossover or transition?  A genuine phase transition sharpens with volume
(the χ peak grows ∝ N and narrows); a crossover's peak stays put.  The
scan runs at two lattice sizes and reports the peak-height ratio as the
discriminator — with the honest caveat that two sizes bound the question
rather than settle it (proper finite-size scaling needs a size ladder).

Usage::

    python experiments/n3_phase_boundary.py --output-dir artifacts/n3_boundary
    python experiments/n3_phase_boundary.py --quick    # smoke run
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
from n3_thermal_selection import build_sector_network  # noqa: E402


def half_value_temperature(temps: list[float], values: list[float]) -> float:
    """Linearly interpolated T where ``values`` first drops below half of
    its coldest (first) entry; NaN if it never does, inf if always below."""
    ref = values[0]
    if not ref > 0:
        return float("nan")
    target = 0.5 * ref
    if values[0] < target:
        return float("inf")
    for i in range(1, len(values)):
        if values[i] < target:
            t0, t1 = temps[i - 1], temps[i]
            v0, v1 = values[i - 1], values[i]
            if v0 == v1:
                return t1
            return t0 + (t1 - t0) * (v0 - target) / (v0 - v1)
    return float("inf")


def peak_temperature(
    temps: list[float], chi: list[float]
) -> tuple[float, float, bool]:
    """(T at the susceptibility maximum, the peak height, at-grid-edge?).

    A maximum on the first or last grid point means the peak is not
    resolved by the scan window — flagged so the summary can say so.
    """
    idx = int(np.nanargmax(chi))
    at_edge = idx in (0, len(temps) - 1)
    return temps[idx], float(chi[idx]), at_edge


def make_figure(scan, sizes, couplings, boundary, path):
    """Melting curves, susceptibility peaks, and the (g_m, T) boundary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # sequential single-hue ramp over g_m (magnitude), CVD-safe blues
    RAMP = ["#b8d4ea", "#7fb2d9", "#4a90c4", "#0072B2"]
    ORANGE = "#E69F00"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax in (ax1, ax2):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("matter temperature T")
        ax.set_xscale("log")
    ax3.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
    ax3.set_axisbelow(True)
    for side in ("top", "right"):
        ax3.spines[side].set_visible(False)

    big = max(sizes)
    for gi, g_m in enumerate(couplings):
        pts = scan[(big, g_m)]
        ts = [p["temperature"] for p in pts]
        ax1.errorbar(ts, [p["neutrality"] for p in pts],
                     yerr=[p["neutrality_err"] for p in pts],
                     color=RAMP[gi % len(RAMP)], marker="o", ms=4, lw=1.6,
                     capsize=2, label=f"g_m={g_m:g}", zorder=3)
        ax2.errorbar(ts, [p["order_susceptibility"] for p in pts],
                     yerr=[p["order_susceptibility_err"] for p in pts],
                     color=RAMP[gi % len(RAMP)], marker="o", ms=4, lw=1.6,
                     capsize=2, label=f"g_m={g_m:g}", zorder=3)
    ax1.set_ylabel("neutrality (full-palette junction density)")
    ax1.set_title(f"Junction-network melting ({big}²)")
    ax1.legend(frameon=False)
    ax2.set_ylabel("order susceptibility χ = N·Var(order)")
    ax2.set_title(f"Susceptibility across the melt ({big}²)")
    ax2.legend(frameon=False)

    for crit, color, marker, label in (
        ("t_half", "#0072B2", "o", "half-value of neutrality"),
        ("t_peak", ORANGE, "s", "χ-peak"),
    ):
        gs = [g for g in couplings
              if not math.isnan(boundary[(big, g)][crit])
              and not math.isinf(boundary[(big, g)][crit])]
        ax3.plot(gs, [boundary[(big, g)][crit] for g in gs], color=color,
                 marker=marker, ms=6, lw=1.6, label=label, zorder=3)
    ax3.set_xlabel("matter coupling g_m")
    ax3.set_xscale("log", base=2)
    ax3.set_ylabel("melting temperature")
    ax3.set_title("The (g_m, T) melting boundary")
    ax3.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-phases", type=int, default=3)
    p.add_argument("--sizes", type=str, default="32,48")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--diffusion", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.09)
    p.add_argument("--temperatures", type=str,
                   default="0.05,0.08,0.12,0.18,0.27,0.4,0.6,0.9")
    p.add_argument("--matter-couplings", type=str, default="1.0,2.0,4.0,8.0")
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--frac-penalty", type=float, default=20.0)
    p.add_argument("--n-therm", type=int, default=250)
    p.add_argument("--n-meas", type=int, default=120)
    p.add_argument("--n-skip", type=int, default=2)
    p.add_argument("--bin-size", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_boundary")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.sizes = "16,24"
        args.steps = 200
        args.temperatures = "0.05,0.15,0.4"
        args.matter_couplings = "2.0,8.0"
        args.n_therm, args.n_meas = 50, 30

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    couplings = [float(g) for g in args.matter_couplings.split(",") if g.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    networks = {}
    for size in sizes:
        networks[size] = build_sector_network(
            args.n_phases, size=size, steps=args.steps, gamma=args.gamma,
            diffusion=args.diffusion, dt=args.dt, seed=args.seed,
            beta=args.beta)
        print(f"network {size}²: cold neutrality="
              f"{networks[size]['neutrality']:.5f}", flush=True)

    scan: dict[tuple[int, float], list[dict]] = {}
    for size in sizes:
        for g_m in couplings:
            pts = []
            for ti, t in enumerate(temps):
                pt = run_point(
                    networks[size], temperature=t, beta_g=args.beta_g,
                    matter_coupling=g_m, quartic_u=args.quartic_u,
                    frac_penalty=args.frac_penalty,
                    n_therm=args.n_therm, n_meas=args.n_meas,
                    n_skip=args.n_skip, bin_size=args.bin_size,
                    seed=args.seed + 100000 * size + 100 * int(g_m * 4) + ti,
                    beta=args.beta)
                pts.append(pt)
                print(f"  L={size} g_m={g_m:4.1f} T={t:5.2f}: "
                      f"neut={pt['neutrality']:.5f}±{pt['neutrality_err']:.5f}"
                      f"  chi_ord={pt['order_susceptibility']:8.2f}"
                      f"±{pt['order_susceptibility_err']:.2f}"
                      f"  acc={pt['acceptance']:.2f}", flush=True)
            scan[(size, g_m)] = pts

    boundary = {}
    for size in sizes:
        for g_m in couplings:
            pts = scan[(size, g_m)]
            neut = [p["neutrality"] for p in pts]
            chi = [p["order_susceptibility"] for p in pts]
            t_half = half_value_temperature(temps, neut)
            t_peak, chi_peak, at_edge = peak_temperature(temps, chi)
            boundary[(size, g_m)] = {
                "t_half": t_half, "t_peak": t_peak, "chi_peak": chi_peak,
                "peak_at_grid_edge": at_edge,
            }

    lines = ["(g_m, T) melting boundary — verdicts", "=" * 44, ""]
    big, small = max(sizes), min(sizes)
    for g_m in couplings:
        b_big = boundary[(big, g_m)]
        b_small = boundary[(small, g_m)]
        ratio = (b_big["chi_peak"] / b_small["chi_peak"]
                 if b_small["chi_peak"] > 0 else float("nan"))
        vol_ratio = (big / small) ** 2
        edge = " [peak at grid edge — unresolved]" \
            if b_big["peak_at_grid_edge"] else ""
        lines.append(
            f"g_m={g_m:g}: T_half={b_big['t_half']:.3f}  "
            f"T_chi-peak={b_big['t_peak']:.3f}{edge}  "
            f"chi_peak {small}²→{big}²: {b_small['chi_peak']:.2f}→"
            f"{b_big['chi_peak']:.2f} (×{ratio:.2f}; ×{vol_ratio:.2f} "
            "would be transition-like)"
        )
    lines.append("")
    halves = [boundary[(big, g)]["t_half"] for g in couplings]
    finite = [(g, t) for g, t in zip(couplings, halves)
              if not math.isnan(t) and not math.isinf(t)]
    if len(finite) >= 2 and all(t1 <= t2 for (_, t1), (_, t2)
                                in zip(finite, finite[1:])):
        lines.append("boundary is monotone: stronger matter–gauge coupling "
                     "melts at higher temperature (coupling stabilises the "
                     "junction network)")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {
        "params": vars(args),
        "scan": {f"{s}_{g:g}": scan[(s, g)] for s in sizes
                 for g in couplings},
        "boundary": {f"{s}_{g:g}": boundary[(s, g)] for s in sizes
                     for g in couplings},
    }
    with open(os.path.join(args.output_dir, "n3_boundary.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_boundary.png")
        make_figure(scan, sizes, couplings, boundary, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
