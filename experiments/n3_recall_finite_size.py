"""Does "recall outlives order" survive a bigger lattice?

The κ-recovery experiment (``n3_recall_recovery.py``) reported the
program's strongest claim so far: at a high capacity-recovery rate the
**recall edge** (fertile soil fraction drops below ½) climbs *above* the
**order edge** (Potts magnetisation drops below ½) — remembered structure
can re-root at temperatures where the field itself has already melted.
Memory outlives order.

That result was measured at a single lattice size (L = 28).  Both edges
are finite-size proxies for sharp thresholds, so the honest worry is that
the gap between them is a small-lattice artifact that closes as L grows:
the Potts magnetisation has a fat finite-size tail above T_c that makes the
order edge sit *low* on a small lattice, and if it climbs toward the true
T_c faster than the recall edge does, the overtaking could vanish.

This experiment settles it.  At fixed high recovery ``r`` and fixed
consumption, it scans temperature across a **ladder of lattice sizes** and
tracks the single decisive observable —

    gap(L) = recall_edge(L) − order_edge(L)

If the overtaking is real, gap(L) stays positive as L increases; if it is
a finite-size illusion, gap(L) trends to zero.  Both per-size edges are
extracted with the same interpolated-crossing helper the recovery
experiment used.

In the overtaking regime the recall edge often runs *beyond* the scan
window (recall never falls below ½), so the raw gap is ``+∞`` and cannot
be extrapolated.  The decisive, always-finite observable is therefore the
**recall margin at the melt** —

    margin(L) = recall_capacity( order_edge(L) ) − ½

i.e. how much fertile soil survives at the very temperature where the
field has half-disordered.  ``margin > 0`` is exactly equivalent to
``recall_edge > order_edge`` (recall outlives order), but as a bounded
number it can be extrapolated: a weighted least-squares fit of margin vs
1/L reads the thermodynamic-limit value straight off the 1/L → 0
intercept.  A positive intercept means the overtaking survives L → ∞.

Usage::

    python experiments/n3_recall_finite_size.py --output-dir artifacts/n3_finite
    python experiments/n3_recall_finite_size.py --quick    # smoke run
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from n3_recall_recovery import crossing_below  # noqa: E402
from n3_seed_rooting import recall_capacity_point  # noqa: E402


def interp_at(temps: list[float], values: list[float], t: float) -> float:
    """Linear interpolation of ``values`` at temperature ``t`` (ascending T).

    Clamps to the endpoints outside the grid; returns ``nan`` if ``t`` is
    not finite (e.g. the order edge ran off the window).
    """
    if not math.isfinite(t):
        return float("nan")
    if t <= temps[0]:
        return float(values[0])
    if t >= temps[-1]:
        return float(values[-1])
    for i in range(1, len(temps)):
        if temps[i - 1] <= t <= temps[i]:
            t0, t1 = temps[i - 1], temps[i]
            v0, v1 = values[i - 1], values[i]
            if t1 == t0:
                return float(v1)
            return float(v0 + (v1 - v0) * (t - t0) / (t1 - t0))
    return float(values[-1])


def edge_gap(recall_edge: float, order_edge: float) -> float:
    """Signed distance the recall edge reaches past the order edge.

    Positive ⇒ recall outlives order.  ``+inf`` when recall never falls
    below the level within the scan (the strongest overtaking); ``-inf``
    when order survives but recall does not (recall undershoots order).
    ``nan`` when neither edge is finite (both survive the whole window),
    since the comparison is then undefined on this grid.
    """
    if math.isinf(recall_edge) and math.isinf(order_edge):
        return float("nan")
    return recall_edge - order_edge


def weighted_slope(xs: list[float], ys: list[float],
                   errs: list[float]) -> dict:
    """Weighted least-squares line ``y = a + b·x``; return slope/intercept.

    Only finite points contribute (an infinite edge-gap carries no finite
    ordinate).  Weights are ``1/err²``.  With fewer than two finite points
    the fit is undefined and slope/intercept come back ``nan``.
    """
    pts = [(x, y, e) for x, y, e in zip(xs, ys, errs)
           if math.isfinite(x) and math.isfinite(y)]
    if len(pts) < 2:
        return {"slope": float("nan"), "intercept": float("nan"),
                "n": len(pts)}
    w = np.array([1.0 / max(e, 1e-9) ** 2 for _, _, e in pts])
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    sw = w.sum()
    mx = (w * x).sum() / sw
    my = (w * y).sum() / sw
    sxx = (w * (x - mx) ** 2).sum()
    sxy = (w * (x - mx) * (y - my)).sum()
    if sxx <= 0:
        return {"slope": float("nan"), "intercept": my, "n": len(pts)}
    b = sxy / sxx
    a = my - b * mx
    return {"slope": float(b), "intercept": float(a), "n": len(pts)}


def scan_size(size: int, temps: list[float], args, recovery: float) -> dict:
    """Recall/order curves and their two edges for one lattice size."""
    sub = copy.copy(args)
    sub.size = size
    pts = []
    for ti, t in enumerate(temps):
        pt = recall_capacity_point(temperature=t, consumption=args.consumption,
                                   args=sub, seed=args.seed + 100 * ti,
                                   recovery=recovery)
        pts.append(pt)
        print(f"  L={size} T={t:.3f}: recall={pt['recall_capacity']:.3f}"
              f"±{pt['recall_capacity_err']:.3f}  m={pt['magnetisation']:.3f}"
              f"  ⟨κ⟩={pt['kappa_mean']:.3f}", flush=True)
    recall_vals = [p["recall_capacity"] for p in pts]
    recall_errs = [p["recall_capacity_err"] for p in pts]
    order_vals = [p["magnetisation"] for p in pts]
    recall_edge = crossing_below(temps, recall_vals, 0.5)
    order_edge = crossing_below(temps, order_vals, 0.5)
    # the always-finite decisive scalar: recall capacity at the melt (m=½)
    recall_at_melt = interp_at(temps, recall_vals, order_edge)
    margin = (recall_at_melt - 0.5 if math.isfinite(recall_at_melt)
              else float("nan"))
    margin_err = interp_at(temps, recall_errs, order_edge)
    return {"size": size, "points": pts, "recall_edge": recall_edge,
            "order_edge": order_edge, "gap": edge_gap(recall_edge, order_edge),
            "recall_at_melt": recall_at_melt, "margin": margin,
            "margin_err": margin_err}


def make_figure(by_size, sizes, temps, fit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RAMP = ["#cfe3f2", "#7fb2d8", "#2f7ab5", "#004b78"]
    GREEN, GRAY, ORANGE = "#009E73", "#999999", "#E69F00"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax in (ax1, ax2, ax3):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # panel 1: recall(T) and m(T) for every size — the two crossings per L
    for si, L in enumerate(sizes):
        pts = by_size[L]["points"]
        ts = [p["temperature"] for p in pts]
        col = RAMP[si % len(RAMP)]
        ax1.errorbar(ts, [p["recall_capacity"] for p in pts],
                     yerr=[p["recall_capacity_err"] for p in pts],
                     color=col, marker="o", ms=4, lw=1.7, capsize=2,
                     label=f"recall L={L}", zorder=3)
        ax1.plot(ts, [p["magnetisation"] for p in pts], color=col, marker="s",
                 ms=3, lw=1.2, ls="--", zorder=2)
    ax1.axhline(0.5, color=GRAY, lw=0.9, ls=":", zorder=1)
    ax1.set_xscale("log")
    ax1.set_xlabel("matter temperature T")
    ax1.set_ylabel("recall capacity (○) / order m (□,dashed)")
    ax1.set_title("Recall vs order across sizes")
    ax1.legend(frameon=False, fontsize=7)

    # panel 2: the two edges vs L — do they converge, and which stays higher?
    ax2.plot(sizes, [by_size[L]["recall_edge"] if math.isfinite(by_size[L]["recall_edge"])
                     else np.nan for L in sizes],
             color="#0072B2", marker="o", ms=6, lw=1.7,
             label="recall edge", zorder=3)
    ax2.plot(sizes, [by_size[L]["order_edge"] for L in sizes],
             color=GREEN, marker="s", ms=6, lw=1.7, label="order edge",
             zorder=3)
    # mark sizes where the recall edge ran off the top of the window
    for L in sizes:
        if math.isinf(by_size[L]["recall_edge"]):
            ax2.annotate("↑∞", (L, max(temps)), color="#0072B2",
                         fontsize=9, ha="center", va="bottom")
    ax2.set_xlabel("lattice size L")
    ax2.set_ylabel("edge temperature")
    ax2.set_title("Edges vs lattice size")
    ax2.legend(frameon=False, fontsize=8)

    # panel 3: the decisive observable — recall margin at the melt vs 1/L,
    # with the weighted L→∞ extrapolation (intercept > 0 ⇒ recall outlives
    # order in the thermodynamic limit)
    inv = [1.0 / L for L in sizes]
    margins = [by_size[L]["margin"] for L in sizes]
    errs = [by_size[L]["margin_err"] if math.isfinite(by_size[L]["margin_err"])
            else 0.02 for L in sizes]
    finite = [(x, g, e, L) for x, g, e, L in zip(inv, margins, errs, sizes)
              if math.isfinite(g)]
    if finite:
        ax3.errorbar([f[0] for f in finite], [f[1] for f in finite],
                     yerr=[f[2] for f in finite], color=ORANGE, marker="o",
                     ms=7, lw=0, elinewidth=1.3, capsize=3, zorder=3)
        for x, g, e, L in finite:
            ax3.annotate(f"L={L}", (x, g), fontsize=7, ha="left",
                         va="bottom", xytext=(3, 4), textcoords="offset points")
    if math.isfinite(fit["slope"]):
        xr = np.array([0.0, max(inv) * 1.05])
        ax3.plot(xr, fit["intercept"] + fit["slope"] * xr, color="#004b78",
                 lw=1.4, ls="--",
                 label=f"L→∞ margin ≈ {fit['intercept']:+.3f}", zorder=2)
        ax3.plot([0.0], [fit["intercept"]], color="#004b78", marker="*",
                 ms=13, zorder=4)
        ax3.legend(frameon=False, fontsize=8)
    ax3.axhline(0.0, color=GRAY, lw=1.0, ls="-", zorder=1)
    ax3.set_xlabel("1 / L   (→ 0 is the thermodynamic limit)")
    ax3.set_ylabel("recall margin at melt = recall(m=½) − ½")
    ax3.set_title("Does recall outlive order as L → ∞?")
    ax3.set_xlim(left=-0.005)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sizes", type=str, default="16,24,32,40")
    p.add_argument("--temperatures", type=str,
                   default="0.05,0.065,0.08,0.095,0.11,0.13,0.16,0.20")
    p.add_argument("--recovery", type=float, default=0.8,
                   help="fixed high capacity-recovery rate (the overtaking regime)")
    p.add_argument("--consumption", type=float, default=5.0)
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--kappa-recovery", type=float, default=0.8)
    p.add_argument("--n-therm", type=int, default=600)
    p.add_argument("--n-meas", type=int, default=80)
    p.add_argument("--n-skip", type=int, default=3)
    p.add_argument("--bin-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_finite")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.sizes = "12,16"
        args.temperatures = "0.06,0.10,0.16"
        args.n_therm, args.n_meas = 200, 40

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    # keep the fixed recovery rate consistent for both the scan and any
    # default lookups inside recall_capacity_point
    args.kappa_recovery = args.recovery
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Finite-size ladder at r={args.recovery:g}, c={args.consumption:g}",
          flush=True)
    by_size: dict[int, dict] = {}
    for L in sizes:
        print(f"L = {L}", flush=True)
        by_size[L] = scan_size(L, temps, args, args.recovery)
        e = by_size[L]
        print(f"  → recall edge T={e['recall_edge']:.3f}  order edge "
              f"T={e['order_edge']:.3f}  gap={e['gap']:.3f}", flush=True)

    inv = [1.0 / L for L in sizes]
    margins = [by_size[L]["margin"] for L in sizes]
    margin_errs = [by_size[L]["margin_err"] if math.isfinite(by_size[L]["margin_err"])
                   else 0.02 for L in sizes]
    fit = weighted_slope(inv, margins, margin_errs)

    lines = ["does recall outlive order as L grows? — verdict", "=" * 48, ""]
    for L in sizes:
        e = by_size[L]
        re = "∞" if math.isinf(e["recall_edge"]) else f"{e['recall_edge']:.3f}"
        mg = "nan" if math.isnan(e["margin"]) else f"{e['margin']:+.3f}"
        lines.append(f"L={L:>3}: recall edge T={re:>6}  order edge "
                     f"T={e['order_edge']:.3f}  recall@melt="
                     f"{e['recall_at_melt']:.3f}  margin={mg}")
    lines.append("")

    n_positive = sum(1 for L in sizes if by_size[L]["margin"] > 0)
    if n_positive == len(sizes):
        lines.append(
            f"recall outlives order at EVERY size ({n_positive}/{len(sizes)}): "
            "at the melt (m=½) more than half the soil is still fertile on "
            "every lattice — the overtaking is not a single-lattice artifact")
    elif n_positive > 0:
        lines.append(
            f"recall outlives order at {n_positive}/{len(sizes)} sizes: the "
            "effect is present but not uniform across the ladder")
    else:
        lines.append(
            "recall does not outlive order at any size: at the melt the soil "
            "is already below half fertile — the earlier overtaking does not "
            "reproduce on this ladder")

    if math.isfinite(fit["slope"]):
        trend = ("widens" if fit["slope"] < -1e-6 else
                 "narrows" if fit["slope"] > 1e-6 else "holds flat")
        lines.append(
            f"extrapolating the recall margin at the melt to the thermodynamic "
            f"limit (1/L → 0): margin → {fit['intercept']:+.3f} (slope "
            f"{fit['slope']:+.3f} in 1/L; as L grows the margin {trend}).")
        if fit["intercept"] > 0:
            lines.append(
                "the L → ∞ intercept is POSITIVE: recall outliving order "
                "survives the finite-size extrapolation — at the melt, a "
                "majority of soil stays fertile even in the thermodynamic "
                "limit. It is a property of the model, not of the lattice.")
        else:
            lines.append(
                "the L → ∞ intercept is not positive: on this ladder the "
                "overtaking is consistent with closing in the thermodynamic "
                "limit — the finite-lattice result should be read with caution.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": vars(args),
               "by_size": {str(L): by_size[L] for L in sizes},
               "fit": fit}
    with open(os.path.join(args.output_dir, "n3_recall_finite_size.json"),
              "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_recall_finite_size.png")
        make_figure(by_size, sizes, temps, fit, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
