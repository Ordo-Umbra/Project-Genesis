"""The 3-D unpinned transition: does it turn first order, as Potts predicts?

The 2-D unpinned S_P transition scaled like the 2-D 3-state Potts model —
a *continuous* transition.  The same universality argument makes a sharp,
falsifiable prediction in 3-D: the 3-state Potts transition is **first
order** there.  If the sector model really is in the Potts class, its 3-D
order–disorder point should change character: latent heat, phase
coexistence, hysteresis.

Signatures measured on a small ladder (weakly-first-order transitions are
subtle at small L — verdicts are stated with that caveat):

- **Hot/cold hysteresis**: at each (L, T) the ensemble is run twice, once
  from an ordered start and once from a disordered one.  At a continuous
  transition both converge; across a first-order region they stay on
  their respective branches, giving a magnetisation disagreement window
  that *widens or persists* with L (at a crossover/continuous point it
  shrinks).
- **Binder minimum**: at a first-order point U₄ dips to a *negative*
  minimum that deepens with volume (phase coexistence makes the m
  distribution bimodal); at a continuous transition U₄ stays between the
  trivial limits and the dip vanishes with L.

Usage::

    python experiments/n3_potts_3d.py --output-dir artifacts/n3_potts_3d
    python experiments/n3_potts_3d.py --quick     # smoke run
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


def scan(sizes, temps, args) -> dict:
    """Hot- and cold-started scans over (L, T)."""
    results: dict[str, dict[int, list[dict]]] = {"cold": {}, "hot": {}}
    for L in sizes:
        for start in ("cold", "hot"):
            pts = []
            for ti, t in enumerate(temps):
                pt = run_point(
                    n_palette=args.n_phases, size=L, temperature=t,
                    beta_g=args.beta_g,
                    matter_coupling=args.matter_coupling,
                    quartic_u=args.quartic_u, n_therm=args.n_therm,
                    n_meas=args.n_meas, n_skip=args.n_skip,
                    bin_size=args.bin_size,
                    seed=args.seed + 10000 * L + ti, step_scale=args.step_scale,
                    dim=3, start=start)
                pts.append(pt)
                print(f"  L={L} {start:4s} T={t:5.3f}: "
                      f"m={pt['magnetisation']:.4f}±{pt['magnetisation_err']:.4f}  "
                      f"U4={pt['binder']:.3f}±{pt['binder_err']:.3f}  "
                      f"acc={pt['acceptance']:.2f}", flush=True)
            results[start][L] = pts
    return results


def hysteresis_summary(results, sizes, temps) -> dict:
    """Per-size hysteresis window: T-range where hot and cold disagree."""
    out = {}
    for L in sizes:
        gaps = []
        for pc, ph in zip(results["cold"][L], results["hot"][L]):
            err = math.hypot(pc["magnetisation_err"], ph["magnetisation_err"])
            gap = pc["magnetisation"] - ph["magnetisation"]
            gaps.append({"temperature": pc["temperature"],
                         "gap": gap, "err": err,
                         "significant": bool(gap > 4 * err and gap > 0.1)})
        window = [g["temperature"] for g in gaps if g["significant"]]
        out[str(L)] = {
            "gaps": gaps,
            "window": [min(window), max(window)] if window else None,
            "n_significant": len(window),
        }
    return out


def binder_minima(results, sizes) -> dict:
    """Deepest cold-start Binder value per size (bimodality marker)."""
    out = {}
    for L in sizes:
        pts = results["cold"][L]
        vals = [(p["binder"], p["binder_err"], p["temperature"])
                for p in pts if np.isfinite(p["binder"])]
        u_min, u_err, t_at = min(vals, key=lambda v: v[0])
        out[str(L)] = {"binder_min": u_min, "err": u_err,
                       "temperature": t_at}
    return out


def make_figure(results, sizes, temps, hyst, minima, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RAMP = ["#7fb2d9", "#4a90c4", "#0072B2", "#004b78"]
    ORANGE = "#E69F00"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax in (ax1, ax2, ax3):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for li, L in enumerate(sizes):
        c = RAMP[li % len(RAMP)]
        for start, ls, mk in (("cold", "-", "o"), ("hot", "--", "^")):
            pts = results[start][L]
            ax1.errorbar([p["temperature"] for p in pts],
                         [p["magnetisation"] for p in pts],
                         yerr=[p["magnetisation_err"] for p in pts],
                         color=c, ls=ls, marker=mk, ms=4, lw=1.4, capsize=2,
                         label=f"L={L} {start}", zorder=3)
        gaps = hyst[str(L)]["gaps"]
        ax2.errorbar([g["temperature"] for g in gaps],
                     [g["gap"] for g in gaps],
                     yerr=[g["err"] for g in gaps], color=c, marker="o",
                     ms=4, lw=1.4, capsize=2, label=f"L={L}", zorder=3)
        pts = results["cold"][L]
        ax3.errorbar([p["temperature"] for p in pts],
                     [p["binder"] for p in pts],
                     yerr=[p["binder_err"] for p in pts], color=c,
                     marker="o", ms=4, lw=1.4, capsize=2, label=f"L={L}",
                     zorder=3)
    ax1.set_xlabel("matter temperature T")
    ax1.set_ylabel("Potts magnetisation m")
    ax1.set_title("3-D: hot vs cold branches")
    ax1.legend(frameon=False, fontsize=7)
    ax2.axhline(0.0, color="#999999", lw=0.8)
    ax2.set_xlabel("matter temperature T")
    ax2.set_ylabel("m(cold) − m(hot)")
    ax2.set_title("Hysteresis window")
    ax2.legend(frameon=False, fontsize=8)
    ax3.axhline(0.0, color=ORANGE, lw=1.0, ls="--",
                label="U₄ < 0 ⇒ bimodal (first order)")
    ax3.set_xlabel("matter temperature T")
    ax3.set_ylabel("Binder U₄ (cold start)")
    ax3.set_title("Binder minima")
    ax3.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-phases", type=int, default=3)
    p.add_argument("--sizes", type=str, default="8,10,12,16")
    p.add_argument("--temperatures", type=str,
                   default="0.10,0.14,0.18,0.22,0.27,0.33,0.40")
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--n-therm", type=int, default=400)
    p.add_argument("--n-meas", type=int, default=250)
    p.add_argument("--n-skip", type=int, default=2)
    p.add_argument("--bin-size", type=int, default=10)
    p.add_argument("--step-scale", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_potts_3d")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.sizes = "6,8"
        args.temperatures = "0.12,0.22,0.35"
        args.n_therm, args.n_meas = 80, 50

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    results = scan(sizes, temps, args)
    hyst = hysteresis_summary(results, sizes, temps)
    minima = binder_minima(results, sizes)

    lines = ["3-D unpinned transition — first-order signatures", "=" * 50, ""]
    for L in sizes:
        h = hyst[str(L)]
        mn = minima[str(L)]
        win = (f"T ∈ [{h['window'][0]:g}, {h['window'][1]:g}] "
               f"({h['n_significant']} points)" if h["window"] else "none")
        lines.append(f"L={L}: hysteresis window {win};  "
                     f"Binder min = {mn['binder_min']:.3f}±{mn['err']:.3f} "
                     f"at T={mn['temperature']:g}")
    lines.append("")
    growing_hyst = all(
        (hyst[str(l2)]["n_significant"] >= hyst[str(l1)]["n_significant"])
        for l1, l2 in zip(sizes, sizes[1:]))
    deepening = all(
        minima[str(l2)]["binder_min"] <= minima[str(l1)]["binder_min"]
        + minima[str(l1)]["err"] + minima[str(l2)]["err"]
        for l1, l2 in zip(sizes, sizes[1:]))
    negative_min = any(minima[str(L)]["binder_min"]
                       < -3 * minima[str(L)]["err"] for L in sizes)
    if negative_min and growing_hyst:
        verdict = ("FIRST-ORDER signatures present: negative Binder minima "
                   "and a persistent hot/cold hysteresis window — the 3-D "
                   "transition changes character exactly as the Potts class "
                   "predicts")
    elif negative_min or (growing_hyst and any(
            hyst[str(L)]["window"] for L in sizes)):
        verdict = ("PARTIAL first-order signatures (see table) — consistent "
                   "with the weakly-first-order 3-D Potts transition, but "
                   "not conclusive at these sizes")
    else:
        verdict = ("no first-order signature resolved at these sizes — "
                   "either the transition is continuous here (universality "
                   "deviation) or the latent heat is too small for L ≤ "
                   f"{max(sizes)}")
    lines.append("verdict: " + verdict)
    lines.append(f"(Binder minima deepening with L: {deepening})")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": vars(args),
               "results": {s: {str(L): results[s][L] for L in sizes}
                           for s in ("cold", "hot")},
               "hysteresis": hyst, "binder_minima": minima}
    with open(os.path.join(args.output_dir, "n3_potts_3d.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_potts_3d.png")
        make_figure(results, sizes, temps, hyst, minima, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
