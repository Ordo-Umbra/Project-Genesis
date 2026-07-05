"""The S-landscape phase diagram: is the optimum relocation a level crossing?

The dynamical-κ experiment found that as the capacity budget binds, the
S-optimum *relocates* from the deep-ordered phase to the critical point.
This experiment resolves the character of that relocation across the
(consumption, weight) plane.

Two competing local maxima of S(T) exist at once:

- the **ordered** optimum at low T, where the standing coherence I is
  highest (integration is cheap when capacity is abundant), and
- the **critical** optimum near T_c, where ΔC peaks (distinction is
  richest at the edge).

As consumption c rises, κ is consumed most where the load is highest, so
the integration channel of the ordered branch is taxed away and the
critical peak wins.  The question is whether the global optimum T*(c)
*jumps discontinuously* between the two (a level crossing — first-order-
like in the optimum location) or *drifts continuously* between them.

Method: reuse the adaptive (ψ, U, κ) ``run_point`` on a consumption × T
grid; then, for each (c, w), split the S(T) profile at the magnetisation
mid-drop into an ordered branch and a critical branch, take the max of
each, and track:

- ΔS(c, w) = S_ordered_max − S_critical_max   (the level-crossing order
  parameter; zero-crossing = boundary c*(w)),
- T*(c, w) = argmax location, and the jump ΔT* across the boundary.

Usage::

    python experiments/n3_s_landscape.py --output-dir artifacts/n3_landscape
    python experiments/n3_s_landscape.py --quick    # smoke run
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

from n3_kappa_criticality import run_point, s_of  # noqa: E402


def split_index(mags: list[float], threshold: float = 0.5) -> int:
    """First index where the magnetisation drops below ``threshold``.

    Temperatures before it are the ordered branch, from it on the critical/
    disordered branch.  Assumes the T grid is ascending (m descending).
    """
    for i, m in enumerate(mags):
        if m < threshold:
            return max(1, i)
    return len(mags) - 1  # never disorders in-window: all ordered


def analyse_curve(pts: list[dict], w: float, m_threshold: float) -> dict:
    """Split S(T) into ordered/critical branches and locate both maxima."""
    temps = [p["temperature"] for p in pts]
    mags = [p["magnetisation"] for p in pts]
    s_vals = [s_of(p, w) for p in pts]
    isplit = split_index(mags, m_threshold)

    ord_idx = range(0, isplit)
    crit_idx = range(isplit, len(pts))
    i_ord = max(ord_idx, key=lambda i: s_vals[i]) if isplit > 0 else 0
    i_crit = max(crit_idx, key=lambda i: s_vals[i]) if isplit < len(pts) else len(pts) - 1

    s_ord, s_crit = s_vals[i_ord], s_vals[i_crit]
    i_star = i_ord if s_ord >= s_crit else i_crit
    return {
        "weight": w,
        "t_ordered": temps[i_ord],
        "t_critical": temps[i_crit],
        "s_ordered": s_ord,
        "s_critical": s_crit,
        "delta_s": s_ord - s_crit,       # >0: ordered optimum wins
        "t_star": temps[i_star],
        "regime": "ordered" if s_ord >= s_crit else "critical",
    }


def zero_crossing(xs: list[float], ys: list[float]) -> float:
    """Interpolated x where y crosses from >0 to <0 (NaN if it doesn't)."""
    for i in range(1, len(xs)):
        if ys[i - 1] > 0.0 >= ys[i]:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            return float(x0 + (x1 - x0) * y0 / (y0 - y1))
    return float("nan")


def make_figure(results, consumptions, temps, weights, w_ref, t_c, path):
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

    # panel 1: S(T) profiles at w_ref for each consumption — the two peaks
    for ci, c in enumerate(consumptions):
        pts = results[c]
        ts = [p["temperature"] for p in pts]
        s_vals = [s_of(p, w_ref) for p in pts]
        ax1.plot(ts, s_vals, color=RAMP[ci % len(RAMP)], marker="o", ms=3,
                 lw=1.5, label=f"c={c:g}", zorder=3)
        a = analyse_curve(pts, w_ref, 0.5)
        i_star = ts.index(a["t_star"])
        ax1.plot([ts[i_star]], [s_vals[i_star]], marker="*", ms=13,
                 color=RAMP[ci % len(RAMP)], zorder=4)
    ax1.set_xscale("log")
    ax1.axvline(t_c, color=GRAY, lw=0.9, ls=":", zorder=1)
    ax1.set_xlabel("matter temperature T")
    ax1.set_ylabel(f"S = ΔC + ⟨κ⟩·w·I  (w={w_ref:g})")
    ax1.set_title("Two competing optima (★ = global)")
    ax1.legend(frameon=False, fontsize=8)

    # panel 2: ΔS(c) for each weight — the level-crossing order parameter
    for wi, w in enumerate(weights):
        ds = [analyse_curve(results[c], w, 0.5)["delta_s"]
              for c in consumptions]
        ax2.plot(consumptions, ds, color=RAMP[wi % len(RAMP)], marker="o",
                 ms=4, lw=1.6, label=f"w={w:g}", zorder=3)
    ax2.axhline(0.0, color=ORANGE, lw=1.0, ls="--",
                label="ordered ↔ critical", zorder=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("consumption c")
    ax2.set_ylabel("ΔS = S_ordered − S_critical")
    ax2.set_title("Level-crossing order parameter")
    ax2.legend(frameon=False, fontsize=8)

    # panel 3: T*(c) for each weight — the optimum location, showing the jump
    for wi, w in enumerate(weights):
        tstar = [analyse_curve(results[c], w, 0.5)["t_star"]
                 for c in consumptions]
        ax3.plot(consumptions, tstar, color=RAMP[wi % len(RAMP)], marker="o",
                 ms=5, lw=1.6, label=f"w={w:g}", zorder=3)
    ax3.axhline(t_c, color=GRAY, lw=0.9, ls=":", label="T_c", zorder=1)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("consumption c")
    ax3.set_ylabel("T* (S-optimum location)")
    ax3.set_title("The optimum jumps, not drifts")
    ax3.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-phases", type=int, default=3)
    p.add_argument("--size", type=int, default=28)
    p.add_argument("--temperatures", type=str,
                   default="0.03,0.05,0.065,0.08,0.095,0.11,0.13,0.16,0.22,0.35")
    p.add_argument("--consumptions", type=str,
                   default="0.5,1.0,2.0,3.5,6.0,10.0,18.0")
    p.add_argument("--kappa-recovery", type=float, default=0.1)
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.09)
    p.add_argument("--kernel-radius", type=int, default=3)
    p.add_argument("--kernel-decay", type=float, default=1.0)
    p.add_argument("--weights", type=str, default="0.04,0.045,0.05,0.055,0.06,0.07")
    p.add_argument("--m-threshold", type=float, default=0.5)
    p.add_argument("--from-json", type=str, default="",
                   help="re-analyse a saved n3_s_landscape.json (weights are "
                        "pure post-processing) instead of re-running the MC")
    p.add_argument("--n-therm", type=int, default=700)
    p.add_argument("--n-meas", type=int, default=180)
    p.add_argument("--n-skip", type=int, default=3)
    p.add_argument("--bin-size", type=int, default=10)
    p.add_argument("--step-scale", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--t-c", type=float, default=0.09)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_landscape")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size = 16
        args.temperatures = "0.04,0.09,0.2"
        args.consumptions = "0.5,18.0"
        args.n_therm, args.n_meas = 150, 60

    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    consumptions = [float(c) for c in args.consumptions.split(",") if c.strip()]
    weights = [float(w) for w in args.weights.split(",") if w.strip()]
    w_ref = weights[len(weights) // 2]
    os.makedirs(args.output_dir, exist_ok=True)

    if args.from_json:
        payload_in = json.load(open(args.from_json))
        results = {float(k): v for k, v in payload_in["results"].items()}
        consumptions = sorted(results)
        temps = [p["temperature"] for p in results[consumptions[0]]]
        print(f"re-analysing {args.from_json}: {len(consumptions)} "
              f"consumptions × {len(temps)} temperatures, weights {weights}")
    else:
        results = {}
        for c in consumptions:
            pts = []
            for ti, t in enumerate(temps):
                pt = run_point(
                    n_palette=args.n_phases, size=args.size, temperature=t,
                    beta_g=args.beta_g, matter_coupling=args.matter_coupling,
                    quartic_u=args.quartic_u, consumption=c,
                    kappa_recovery=args.kappa_recovery, n_therm=args.n_therm,
                    n_meas=args.n_meas, n_skip=args.n_skip,
                    bin_size=args.bin_size,
                    seed=args.seed + 1000 * int(c * 10) + ti,
                    step_scale=args.step_scale, beta=args.beta,
                    kernel_radius=args.kernel_radius,
                    kernel_decay=args.kernel_decay)
                pts.append(pt)
                print(f"  c={c:5.1f} T={t:5.3f}: ⟨κ⟩={pt['kappa_mean']:.4f}  "
                      f"m={pt['magnetisation']:.3f}  ΔC={pt['delta_c']:.5f}  "
                      f"I={pt['coherence']:.4f}", flush=True)
            results[c] = pts

    # boundary c*(w) and jump size per weight
    lines = ["S-landscape phase diagram — verdict", "=" * 44, ""]
    boundary = {}
    jump_sizes = []
    for w in weights:
        analyses = [analyse_curve(results[c], w, args.m_threshold)
                    for c in consumptions]
        ds = [a["delta_s"] for a in analyses]
        c_star = zero_crossing(consumptions, ds)
        regimes = [a["regime"] for a in analyses]
        tstars = [a["t_star"] for a in analyses]
        boundary[w] = {"c_star": c_star, "regimes": regimes,
                       "t_stars": tstars, "delta_s": ds}
        # T* jump across the boundary (largest adjacent change in log-space)
        jumps = [abs(math.log(tstars[i + 1]) - math.log(tstars[i]))
                 for i in range(len(tstars) - 1)]
        max_jump = max(jumps) if jumps else 0.0
        i_j = jumps.index(max_jump) if jumps else 0
        jump_ratio = (tstars[i_j + 1] / tstars[i_j]) if jumps else 1.0
        jump_sizes.append(max_jump)
        cs = "→".join(regimes)
        lines.append(
            f"w={w:g}: c* ≈ {c_star:.2f}  |  T* jump ×{jump_ratio:.2f} "
            f"between c={consumptions[i_j]:g} and {consumptions[i_j + 1]:g}  "
            f"|  regimes: {cs}")
    lines.append("")
    # verdict: is the jump discontinuous (level crossing) or a drift?
    typical_gap = np.median([
        abs(math.log(a["t_critical"]) - math.log(a["t_ordered"]))
        for w in weights for a in
        [analyse_curve(results[c], w, args.m_threshold) for c in consumptions]
        if a["t_critical"] != a["t_ordered"]])
    biggest_jump = max(jump_sizes) if jump_sizes else 0.0
    finite_boundary = any(not math.isnan(boundary[w]["c_star"])
                          for w in weights)
    if finite_boundary and biggest_jump > 0.5 * typical_gap:
        verdict = (
            "LEVEL CROSSING: the optimum location jumps discontinuously "
            f"across c*(w) (T* jump ≈ {math.exp(biggest_jump):.1f}× vs the "
            f"ordered–critical gap of {math.exp(typical_gap):.1f}×) — the "
            "relocation is first-order-like in the optimum, two competing "
            "S-maxima trading global rank, not a continuous drift")
    elif finite_boundary:
        verdict = (
            "continuous drift: the optimum moves smoothly between the "
            "ordered and critical branches as c rises")
    else:
        verdict = ("no boundary crossed in the scanned (c, w) window — "
                   "widen the consumption or weight range")
    lines.append("verdict: " + verdict)
    lines.append(
        f"boundary c*(w): " + ", ".join(
            f"w={w:g}→{boundary[w]['c_star']:.2f}" for w in weights))
    finite_cw = [(w, boundary[w]["c_star"]) for w in weights
                 if not math.isnan(boundary[w]["c_star"])]
    monotone = all(c1 <= c2 for (_, c1), (_, c2)
                   in zip(finite_cw, finite_cw[1:]))
    if len(finite_cw) >= 2 and monotone:
        lines.append("c*(w) increases with w: a larger integration weight "
                     "makes the high-coherence ordered phase more valuable, "
                     "so more scarcity is needed to send the optimum to the "
                     "edge (the two dials trade off, as expected)")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": vars(args),
               "results": {f"{c:g}": results[c] for c in consumptions},
               "boundary": {f"{w:g}": boundary[w] for w in weights}}
    with open(os.path.join(args.output_dir, "n3_s_landscape.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_s_landscape.png")
        make_figure(results, consumptions, temps, weights, w_ref, args.t_c,
                    fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
