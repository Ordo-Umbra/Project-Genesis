#!/usr/bin/env python3
"""Fit the F(N) free-energy coefficients from simulation data.

The URP gauge paper postulates a boundary–information tradeoff

    F(N) = a·N^(2/3) − b·N,      N⋆ = (2a / 3b)³,

and asserts N⋆ = 3 at the QCD-derived β ≈ 0.09. Until now the repo only
*quoted* this formula; this experiment measures its ingredients from actual
runs instead of assuming them.

Two complementary measurements per (β, k) cell, where k is the number of
wells made available to the field via the sectorisation potential:

Part A — direct S-maximization. The theory's selection principle is that
reality settles into the sector count that maximizes S. So: run each k to
quasi-steady state, time-average the S-functional over a trailing window,
and ask which k wins at fixed β.

Part B — F(N) consistency fit. For each run, measure the realized domain
count N and total wall energy E_wall (β·|∇φ|² summed over wall voxels).
Fit a(β) by least squares on E_wall ≈ a·N^(2/3) across the k sweep — the
boundary-cost half of F(N) is directly measurable. The information-gain
coefficient b is not directly observable, so instead of inventing a proxy we
invert the stationarity condition: b_implied = 2a / (3·N^(1/3)) for each
run. If the F(N) form describes these dynamics, b_implied should be roughly
constant across k at fixed β; large scatter is evidence against the form
(for this reduced model).

Honesty notes baked into the output: N keeps falling as coarsening proceeds
(domains merge indefinitely under Allen–Cahn-type dynamics), so all numbers
are reported at a fixed evolution time, stated in the header.

Usage::

    python experiments/n_star_fit.py --size 24 --steps 400 --trials 2
    python experiments/n_star_fit.py --betas 0.03,0.09,0.2 --wells 2,3,4,5,6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from project_genesis import EngineConfig, GenesisEngine  # noqa: E402
from project_genesis.metrics import compute_s_functional  # noqa: E402
from project_genesis.sectorisation import analyze_sectorisation  # noqa: E402


def run_cell(
    beta: float,
    wells: int,
    *,
    size: int,
    steps: int,
    dt: float,
    seed: int | None,
    weight: float,
    window: int,
    gravity: float = 0.22,
) -> dict[str, float]:
    """Run one (β, k) cell and return its measurements."""
    config = EngineConfig(
        chunk_size=size,
        beta=beta,
        gravity=gravity,
        seed=seed,
        default_dt=dt,
        use_sector_potential=True,
        sector_count=wells,
        sector_potential_weight=weight,
    )
    engine = GenesisEngine(config=config)

    warmup = max(0, steps - window)
    for _ in range(warmup):
        engine.step(dt)

    s_values: list[float] = []
    for _ in range(min(window, steps)):
        engine.step(dt)
        s = compute_s_functional(engine.field, engine.prev_field, beta, engine.G)
        s_values.append(s["s_increment"])

    report = analyze_sectorisation(engine.field, beta, include_sectors=False)
    n = int(report["n_sectors"])
    volume = size**3
    wall_voxels = float(report["boundary_fraction"]) * volume
    # Total distinction energy locked in walls: mean wall β|∇φ|² × wall volume.
    e_wall = float(report["wall_distinction"]) * wall_voxels

    return {
        "beta": beta,
        "wells": wells,
        "n_sectors": n,
        "mean_s": float(np.mean(s_values)) if s_values else 0.0,
        "e_wall": e_wall,
        "boundary_fraction": float(report["boundary_fraction"]),
    }


def fit_a(n_values: np.ndarray, e_values: np.ndarray) -> float:
    """Least-squares fit of E_wall ≈ a·N^(2/3) (single coefficient)."""
    x = n_values.astype(float) ** (2.0 / 3.0)
    denom = float(np.sum(x * x))
    if denom == 0.0:
        return 0.0
    return float(np.sum(e_values * x) / denom)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=24, help="Cubic world edge length.")
    p.add_argument("--steps", type=int, default=400, help="Evolution steps per run.")
    p.add_argument("--dt", type=float, default=0.01, help="Timestep.")
    p.add_argument("--seed", type=int, default=7, help="Base RNG seed.")
    p.add_argument("--trials", type=int, default=2, help="Trials per (β, k) cell.")
    p.add_argument("--betas", type=str, default="0.03,0.09,0.2", help="Comma-separated β sweep.")
    p.add_argument("--wells", type=str, default="2,3,4,5,6", help="Comma-separated well counts k.")
    p.add_argument("--weight", type=float, default=0.5, help="Sector potential strength w.")
    p.add_argument(
        "--gravity",
        type=float,
        default=0.22,
        help="Coherence damping G. Set 0 for a degenerate-well control: the "
        "−G·φ term tilts the multi-well potential toward φ=0, breaking well "
        "degeneracy and accelerating collapse to a single sector.",
    )
    p.add_argument("--window", type=int, default=50, help="Trailing steps averaged for S.")
    p.add_argument("--output", default=None, help="Optional path for a JSON results dump.")
    args = p.parse_args()

    betas = [float(b) for b in args.betas.split(",") if b.strip()]
    wells_list = [int(k) for k in args.wells.split(",") if k.strip()]

    print(
        f"N⋆ experiment | world={args.size}³ | steps={args.steps} (S averaged over last "
        f"{args.window}) | trials={args.trials} | w={args.weight}\n"
        f"All values at fixed evolution time; N continues to fall under coarsening.\n"
    )

    all_cells: list[dict[str, float]] = []
    summary: dict[str, dict] = {}

    for beta in betas:
        rows = []
        for wells in wells_list:
            cells = []
            for trial in range(args.trials):
                seed = None if args.seed is None else args.seed + trial
                cell = run_cell(
                    beta,
                    wells,
                    size=args.size,
                    steps=args.steps,
                    dt=args.dt,
                    seed=seed,
                    weight=args.weight,
                    window=args.window,
                    gravity=args.gravity,
                )
                cells.append(cell)
                all_cells.append(cell)
            rows.append(
                {
                    "wells": wells,
                    "n_mean": float(np.mean([c["n_sectors"] for c in cells])),
                    "s_mean": float(np.mean([c["mean_s"] for c in cells])),
                    "e_wall": float(np.mean([c["e_wall"] for c in cells])),
                }
            )

        # Part B: fit a(β) over cells with N >= 1, then implied b per row.
        usable = [r for r in rows if r["n_mean"] >= 1.0]
        n_arr = np.array([r["n_mean"] for r in usable])
        e_arr = np.array([r["e_wall"] for r in usable])
        a_fit = fit_a(n_arr, e_arr) if len(usable) >= 2 else float("nan")

        best = max(rows, key=lambda r: r["s_mean"])
        b_implied = []
        print(f"β = {beta}")
        print(f"{'k wells':>8} | {'N':>6} | {'mean S':>10} | {'E_wall':>8} | {'b_implied':>9}")
        print("-" * 55)
        for r in rows:
            if np.isfinite(a_fit) and r["n_mean"] >= 1.0:
                b_i = 2.0 * a_fit / (3.0 * r["n_mean"] ** (1.0 / 3.0))
                b_implied.append(b_i)
                b_str = f"{b_i:9.4f}"
            else:
                b_str = "      n/a"
            marker = "  <- max S" if r is best else ""
            print(
                f"{r['wells']:>8} | {r['n_mean']:>6.1f} | {r['s_mean']:>10.6f} | "
                f"{r['e_wall']:>8.3f} | {b_str}{marker}"
            )
        if len(b_implied) >= 2:
            spread = float(np.std(b_implied) / np.mean(b_implied)) if np.mean(b_implied) else float("inf")
            consistency = (
                "consistent with the F(N) form"
                if spread < 0.25
                else "NOT consistent with the F(N) form (b should be k-independent)"
            )
            print(
                f"\n  a(β) fit = {a_fit:.4f} | b_implied spread (std/mean) = {spread:.2f} "
                f"-> {consistency}"
            )
            if np.mean(b_implied) > 0:
                n_star = (2.0 * a_fit / (3.0 * float(np.mean(b_implied)))) ** 3
                print(f"  N⋆ from fitted a and mean b_implied = {n_star:.2f} (circular by construction; reported for scale only)")
        print(f"  S-maximizing well count at β={beta}: k = {best['wells']}\n")
        summary[str(beta)] = {"a_fit": a_fit, "rows": rows, "best_k": best["wells"]}

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"cells": all_cells, "summary": summary}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Results written to {out}")


if __name__ == "__main__":
    main()
