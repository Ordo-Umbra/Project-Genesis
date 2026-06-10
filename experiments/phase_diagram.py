#!/usr/bin/env python3
"""Map the (consumption, recovery, β) phase diagram of N⋆=3 selection.

The headline open result of the N⋆ experiment is that three-sector selection
appears to be a *capacity-scarcity* phenomenon: with abundant κ nothing
penalizes fragmentation, but under a binding capacity budget k=3 becomes
S-optimal. This experiment turns that single data point into a map.

For each (β, consumption c, recovery r) cell it runs the wells sweep with the
dynamical capacity field active, time-averages the S-functional over a trailing
window per well count, and records which k maximizes S — and by how much k=3
beats its best rival (the "k=3 margin"). Positive margin = three sectors win.

Outputs:
- an ASCII grid per β (rows = consumption, cols = recovery, cell = winning k),
- a JSON dump of every cell,
- a matplotlib figure (winning-k and k=3-margin heatmaps per β).

Usage::

    python experiments/phase_diagram.py                       # default sweep
    python experiments/phase_diagram.py --betas 0.09 --trials 4
    python experiments/phase_diagram.py --quick               # fast, coarse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))

import numpy as np  # noqa: E402

from n_star_fit import run_cell  # noqa: E402


def evaluate_cell(
    beta: float,
    consumption: float,
    recovery: float,
    *,
    wells_list: list[int],
    size: int,
    steps: int,
    dt: float,
    weight: float,
    window: int,
    trials: int,
    seed: int | None,
    gravity: float,
) -> dict:
    """Run the wells sweep for one (β, c, r) cell; return winner + margin."""
    s_by_k: dict[int, float] = {}
    n_by_k: dict[int, float] = {}
    for k in wells_list:
        s_vals, n_vals = [], []
        for trial in range(trials):
            cell_seed = None if seed is None else seed + trial
            cell = run_cell(
                beta,
                k,
                size=size,
                steps=steps,
                dt=dt,
                seed=cell_seed,
                weight=weight,
                window=window,
                gravity=gravity,
                dynamic_kappa=True,
                kappa_consumption=consumption,
                kappa_recovery=recovery,
            )
            s_vals.append(cell["mean_s"])
            n_vals.append(cell["n_sectors"])
        s_by_k[k] = float(np.mean(s_vals))
        n_by_k[k] = float(np.mean(n_vals))

    best_k = max(s_by_k, key=lambda kk: s_by_k[kk])
    if 3 in s_by_k:
        rivals = [v for kk, v in s_by_k.items() if kk != 3]
        best_rival = max(rivals) if rivals else 0.0
        scale = max(abs(s_by_k[3]), best_rival, 1e-12)
        margin = (s_by_k[3] - best_rival) / scale  # normalized: >0 means k=3 wins
    else:
        margin = float("nan")

    return {
        "beta": beta,
        "consumption": consumption,
        "recovery": recovery,
        "best_k": best_k,
        "k3_margin": margin,
        "s_by_k": s_by_k,
        "n_by_k": n_by_k,
        "n_at_3": n_by_k.get(3, float("nan")),
    }


def print_ascii_grid(beta: float, consumptions, recoveries, results) -> None:
    print(f"\nβ = {beta}  —  winning well count k (★ = k=3 selected)")
    header = "  c \\ r |" + "".join(f"{r:>8}" for r in recoveries)
    print(header)
    print("-" * len(header))
    for c in consumptions:
        cells = []
        for r in recoveries:
            res = results[(beta, c, r)]
            mark = "★" if res["best_k"] == 3 else " "
            cells.append(f"{res['best_k']}{mark}".rjust(8))
        print(f"{c:>7} |" + "".join(cells))


def make_plot(betas, consumptions, recoveries, results, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        return False

    nrows = len(betas)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4.2 * nrows), squeeze=False)
    c_arr = np.array(consumptions, dtype=float)
    r_arr = np.array(recoveries, dtype=float)

    for row, beta in enumerate(betas):
        best = np.array([[results[(beta, c, r)]["best_k"] for r in recoveries] for c in consumptions])
        margin = np.array([[results[(beta, c, r)]["k3_margin"] for r in recoveries] for c in consumptions])

        ax = axes[row][0]
        im = ax.imshow(best, origin="lower", aspect="auto", cmap="viridis", vmin=2, vmax=6)
        ax.set_title(f"β = {beta} — winning k")
        ax.set_xticks(range(len(recoveries)), [f"{r:g}" for r in recoveries])
        ax.set_yticks(range(len(consumptions)), [f"{c:g}" for c in consumptions])
        ax.set_xlabel("recovery r"); ax.set_ylabel("consumption c")
        for i in range(len(consumptions)):
            for j in range(len(recoveries)):
                k = int(best[i, j])
                ax.text(j, i, f"{k}{'★' if k == 3 else ''}", ha="center", va="center",
                        color="white" if k < 4 else "black", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label="k")

        ax2 = axes[row][1]
        finite = margin[np.isfinite(margin)]
        vmax = max(0.05, float(np.nanmax(np.abs(finite)))) if finite.size else 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im2 = ax2.imshow(margin, origin="lower", aspect="auto", cmap="RdBu_r", norm=norm)
        ax2.set_title(f"β = {beta} — k=3 margin (red = 3 wins)")
        ax2.set_xticks(range(len(recoveries)), [f"{r:g}" for r in recoveries])
        ax2.set_yticks(range(len(consumptions)), [f"{c:g}" for c in consumptions])
        ax2.set_xlabel("recovery r"); ax2.set_ylabel("consumption c")
        fig.colorbar(im2, ax=ax2, shrink=0.8, label="normalized S margin")

    fig.suptitle("N⋆=3 selection phase diagram (capacity scarcity ↑ as c↑, r↓)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--betas", type=str, default="0.03,0.09,0.2")
    p.add_argument("--consumptions", type=str, default="5,15,30,50,80,120")
    p.add_argument("--recoveries", type=str, default="0.01,0.03,0.05,0.1,0.2")
    p.add_argument("--wells", type=str, default="2,3,4,5,6")
    p.add_argument("--size", type=int, default=24)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--weight", type=float, default=0.5)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--gravity", type=float, default=0.22)
    p.add_argument("--quick", action="store_true", help="Coarse, fast sweep for a smoke test.")
    p.add_argument("--output", default="artifacts/phase_diagram.json")
    p.add_argument("--plot", default="artifacts/phase_diagram.png")
    args = p.parse_args()

    if args.quick:
        args.betas, args.consumptions, args.recoveries = "0.09", "5,50,120", "0.02,0.1"
        args.trials, args.size, args.steps = 1, 20, 200

    betas = [float(b) for b in args.betas.split(",") if b.strip()]
    consumptions = [float(c) for c in args.consumptions.split(",") if c.strip()]
    recoveries = [float(r) for r in args.recoveries.split(",") if r.strip()]
    wells_list = [int(k) for k in args.wells.split(",") if k.strip()]

    total = len(betas) * len(consumptions) * len(recoveries)
    runs = total * len(wells_list) * args.trials
    print(
        f"Phase-diagram sweep | {len(betas)}β × {len(consumptions)}c × {len(recoveries)}r "
        f"= {total} cells | {len(wells_list)} wells × {args.trials} trials = {runs} runs\n"
        f"world={args.size}³, {args.steps} steps (S over last {args.window}), gravity={args.gravity}"
    )

    results: dict[tuple, dict] = {}
    done = 0
    for beta in betas:
        for c in consumptions:
            for r in recoveries:
                results[(beta, c, r)] = evaluate_cell(
                    beta, c, r,
                    wells_list=wells_list, size=args.size, steps=args.steps, dt=args.dt,
                    weight=args.weight, window=args.window, trials=args.trials,
                    seed=args.seed, gravity=args.gravity,
                )
                done += 1
        print_ascii_grid(beta, consumptions, recoveries, results)

    # k=3 win-rate summary per beta.
    print("\nk=3 selection rate (fraction of (c, r) cells where k=3 wins):")
    for beta in betas:
        wins = sum(1 for c in consumptions for r in recoveries if results[(beta, c, r)]["best_k"] == 3)
        print(f"  β = {beta:<5}: {wins}/{total // len(betas)} cells")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {**v, "s_by_k": {str(k): val for k, val in v["s_by_k"].items()},
         "n_by_k": {str(k): val for k, val in v["n_by_k"].items()}}
        for v in results.values()
    ]
    out.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults written to {out}")

    if args.plot:
        if make_plot(betas, consumptions, recoveries, results, Path(args.plot)):
            print(f"Phase-diagram figure written to {args.plot}")
        else:
            print("matplotlib unavailable; skipped figure.")


if __name__ == "__main__":
    main()
