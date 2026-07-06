"""Can faster soil regeneration rescue memory recall at criticality?

The seed-rooting experiment found that at the engine's default capacity
recovery rate, memory recall collapses at the melt — the soil goes barren
before the field disorders.  But κ has a *recovery* term as well as a
consumption term (``∂_t κ = D_κ∇²κ + r·(κ₀ − κ) − c·load·κ``), and the
steady-state capacity balances the two: ``κ = r / (r + c·load)``.  Since
the load (distinction) peaks at criticality, faster regeneration ``r``
should keep κ above the rooting threshold to higher temperatures — a dial
that decides whether memory survives the transition.

This experiment scans the recovery rate ``r`` against temperature at fixed
consumption, and asks two things:

1. **Does recovery rescue recall?**  recall_capacity(T; r) — does the
   recall-collapse temperature climb with r?
2. **Does recall outlive order, or does recovery just push order up too?**
   Because κ gates the coherence coupling, faster recovery also
   strengthens integration and could shift the Potts transition itself.
   So the Potts magnetisation m(T; r) is measured alongside, and the two
   edges — the recall edge (recall < ½) and the order edge (m < ½) — are
   compared.  If the recall edge crosses *above* the order edge at large r,
   memory genuinely survives into the disordered phase: recall outliving
   order.

Usage::

    python experiments/n3_recall_recovery.py --output-dir artifacts/n3_recovery
    python experiments/n3_recall_recovery.py --quick    # smoke run
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

from n3_seed_rooting import recall_capacity_point  # noqa: E402


def crossing_below(temps: list[float], values: list[float],
                   level: float = 0.5) -> float:
    """Interpolated T where ``values`` first drops below ``level`` (ascending T)."""
    for i in range(1, len(temps)):
        if values[i - 1] >= level > values[i]:
            t0, t1 = temps[i - 1], temps[i]
            v0, v1 = values[i - 1], values[i]
            if v0 == v1:
                return t1
            return float(t0 + (t1 - t0) * (v0 - level) / (v0 - v1))
    if values and values[0] < level:
        return float(temps[0])
    return float("inf")


def make_figure(by_r, recoveries, temps, edges, order_edge_ref, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RAMP = ["#cfe3f2", "#9ac4e2", "#4a90c4", "#1f6aa5", "#004b78"]
    GREEN, GRAY, ORANGE = "#009E73", "#999999", "#E69F00"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax in (ax1, ax2):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # panel 1: recall_capacity(T) for each recovery rate + m overlay
    for ri, r in enumerate(recoveries):
        pts = by_r[r]
        ts = [p["temperature"] for p in pts]
        ax1.errorbar(ts, [p["recall_capacity"] for p in pts],
                     yerr=[p["recall_capacity_err"] for p in pts],
                     color=RAMP[ri % len(RAMP)], marker="o", ms=4, lw=1.7,
                     capsize=2, label=f"recall (r={r:g})", zorder=3)
    pts = by_r[recoveries[len(recoveries) // 2]]
    ax1.plot([p["temperature"] for p in pts],
             [p["magnetisation"] for p in pts], color=GREEN, marker="s",
             ms=4, lw=1.4, ls="--", label=f"order m (r={recoveries[len(recoveries)//2]:g})",
             zorder=2)
    ax1.set_xscale("log")
    ax1.set_xlabel("matter temperature T")
    ax1.set_ylabel("recall capacity / order")
    ax1.set_title("Recovery rescues recall to higher T")
    ax1.legend(frameon=False, fontsize=7)

    # panel 2: the two edges vs recovery rate — recall edge climbing past order
    ax2.plot(recoveries, [edges[r]["recall_edge"] for r in recoveries],
             color="#0072B2", marker="o", ms=6, lw=1.7,
             label="recall edge (recall < ½)", zorder=3)
    ax2.plot(recoveries, [edges[r]["order_edge"] for r in recoveries],
             color=GREEN, marker="s", ms=6, lw=1.7,
             label="order edge (m < ½)", zorder=3)
    ax2.set_xscale("log")
    ax2.set_xlabel("capacity recovery rate r")
    ax2.set_ylabel("temperature")
    ax2.set_title("Does recall outlive order?")
    ax2.legend(frameon=False, fontsize=8)

    # panel 3: recall_capacity(T, r) heatmap — the memory-vs-recovery phase map
    grid = np.array([[next(p["recall_capacity"] for p in by_r[r]
                           if p["temperature"] == t) for t in temps]
                     for r in recoveries])
    im = ax3.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                    vmin=0.0, vmax=1.0)
    ax3.set_xticks(range(len(temps)))
    ax3.set_xticklabels([f"{t:g}" for t in temps], fontsize=7, rotation=45)
    ax3.set_yticks(range(len(recoveries)))
    ax3.set_yticklabels([f"{r:g}" for r in recoveries], fontsize=8)
    ax3.set_xlabel("matter temperature T")
    ax3.set_ylabel("recovery rate r")
    ax3.set_title("Memory phase map: recall(T, r)")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=28)
    p.add_argument("--temperatures", type=str,
                   default="0.05,0.065,0.08,0.095,0.11,0.13,0.16,0.20")
    p.add_argument("--recoveries", type=str, default="0.05,0.1,0.2,0.4,0.8")
    p.add_argument("--consumption", type=float, default=5.0)
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--n-therm", type=int, default=500)
    p.add_argument("--n-meas", type=int, default=80)
    p.add_argument("--n-skip", type=int, default=3)
    p.add_argument("--bin-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_recovery")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size = 16
        args.temperatures = "0.06,0.10,0.16"
        args.recoveries = "0.1,0.8"
        args.n_therm, args.n_meas = 200, 40

    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    recoveries = [float(r) for r in args.recoveries.split(",") if r.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    by_r: dict[float, list[dict]] = {}
    for r in recoveries:
        pts = []
        for ti, t in enumerate(temps):
            pt = recall_capacity_point(temperature=t, consumption=args.consumption,
                                       args=args, seed=args.seed + 100 * ti,
                                       recovery=r)
            pts.append(pt)
            print(f"  r={r:g} T={t:.3f}: recall={pt['recall_capacity']:.3f}"
                  f"±{pt['recall_capacity_err']:.3f}  m={pt['magnetisation']:.3f}"
                  f"  ⟨κ⟩={pt['kappa_mean']:.3f}", flush=True)
        by_r[r] = pts

    edges = {}
    for r in recoveries:
        pts = by_r[r]
        edges[r] = {
            "recall_edge": crossing_below(
                temps, [p["recall_capacity"] for p in pts], 0.5),
            "order_edge": crossing_below(
                temps, [p["magnetisation"] for p in pts], 0.5),
        }

    lines = ["capacity recovery rescues recall — verdict", "=" * 46, ""]
    for r in recoveries:
        e = edges[r]
        lines.append(f"r={r:g}: recall edge T={e['recall_edge']:.3f}  |  "
                     f"order edge T={e['order_edge']:.3f}")
    lines.append("")
    r_lo, r_hi = recoveries[0], recoveries[-1]
    re_lo, re_hi = edges[r_lo]["recall_edge"], edges[r_hi]["recall_edge"]
    if re_hi > re_lo + 1e-6:
        hi_str = ("beyond the scan window" if math.isinf(re_hi)
                  else f"T={re_hi:.3f}")
        lines.append(
            f"recovery rescues recall: the recall edge climbs from "
            f"T={re_lo:.3f} (r={r_lo:g}) to {hi_str} (r={r_hi:g}) — "
            "faster soil regeneration sustains memory to higher temperature")
    # does recall outlive order?  recall edge at or beyond a *finite* order
    # edge means memory survives where the field itself has disordered
    # (recall_edge = inf is the strongest such case).
    outlives = [r for r in recoveries
                if math.isfinite(edges[r]["order_edge"])
                and edges[r]["recall_edge"] >= edges[r]["order_edge"] - 1e-6]
    if outlives:
        r0 = min(outlives)
        lines.append(
            f"recall OUTLIVES order for r ≥ {r0:g}: the recall edge reaches "
            "at or beyond the order edge — with fast enough regeneration, "
            "remembered structure can re-root even where the field itself "
            "has lost its long-range order. Memory outlives order.")
    else:
        lines.append(
            "recall stays below the order edge at every scanned r: recovery "
            "rescues recall toward T_c but not past it in this window")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": vars(args),
               "recall_by_recovery": {f"{r:g}": by_r[r] for r in recoveries},
               "edges": {f"{r:g}": edges[r] for r in recoveries}}
    with open(os.path.join(args.output_dir, "n3_recall_recovery.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_recall_recovery.png")
        make_figure(by_r, recoveries, temps, edges,
                    edges[recoveries[len(recoveries) // 2]]["order_edge"],
                    fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
