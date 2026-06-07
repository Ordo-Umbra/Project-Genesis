#!/usr/bin/env python3
"""Empirical β-sectorisation experiment.

The URP gauge paper predicts that the β-nonlinearity drives the field to
partition into a small number of stable domains, with N⋆ = 3 the dominant
attractor at the QCD-derived β ≈ 0.09. This script *measures* the sector
count the URP field actually settles into across a sweep of β, so the
prediction can be checked rather than assumed.

It also reports the boundary-work decomposition from "The Range": how much of
the volume is wall (boundary), the distinction concentrated on those walls,
and the integration sustained inside the domains.

Usage::

    python experiments/beta_sectorisation.py --size 32 --steps 200 --seed 7
    python experiments/beta_sectorisation.py --betas 0.03,0.09,0.2,0.5 --trials 3

Caveat: the default URP dynamics (diffusion + coherence damping) are
smoothing; persistent domain walls are not guaranteed to survive. A flat
result (N≈1) is itself an informative measurement — it says the *current*
reduced field equation does not, on its own, sustain β-sectors, and points at
the wall-supporting terms (the −(β/4)(∇φ)⁴ tension) the theory implies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from project_genesis import EngineConfig, GenesisEngine  # noqa: E402
from project_genesis.sectorisation import (  # noqa: E402
    analyze_sectorisation,
    stationary_sector_count,
)


def run_one(
    beta: float,
    *,
    size: int,
    steps: int,
    dt: float,
    seed: int | None,
    threshold_k: float,
    coherence: bool,
) -> dict[str, float]:
    config = EngineConfig(
        chunk_size=size,
        beta=beta,
        seed=seed,
        default_dt=dt,
        use_coherence_potential=coherence,
    )
    engine = GenesisEngine(config=config)
    engine.evolve_field(steps=steps, dt=dt, record_every=max(1, steps // 5))
    report = analyze_sectorisation(engine.field, beta, k=threshold_k)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=32, help="Cubic world edge length.")
    p.add_argument("--steps", type=int, default=200, help="Evolution steps per run.")
    p.add_argument("--dt", type=float, default=0.01, help="Timestep.")
    p.add_argument("--seed", type=int, default=7, help="Base RNG seed.")
    p.add_argument("--trials", type=int, default=1, help="Trials per β (seeds offset).")
    p.add_argument(
        "--betas",
        type=str,
        default="0.03,0.09,0.2,0.5",
        help="Comma-separated β values to sweep.",
    )
    p.add_argument(
        "--threshold-k",
        type=float,
        default=1.0,
        help="Wall threshold factor (|∇φ| > mean + k·std).",
    )
    p.add_argument(
        "--coherence",
        action="store_true",
        default=False,
        help="Enable the coherence potential during evolution.",
    )
    args = p.parse_args()

    betas = [float(b) for b in args.betas.split(",") if b.strip()]

    print(
        f"β-sectorisation sweep | world={args.size}³ | steps={args.steps} | "
        f"trials={args.trials} | coherence={args.coherence}\n"
    )
    header = f"{'β':>6} | {'N (mean)':>9} | {'N (all)':>16} | {'wall%':>6} | {'distinction':>11} | {'integration':>11} | {'Yjunc':>6}"
    print(header)
    print("-" * len(header))

    for beta in betas:
        n_values: list[int] = []
        wall_fracs: list[float] = []
        distinctions: list[float] = []
        integrations: list[float] = []
        junctions: list[int] = []
        for trial in range(args.trials):
            seed = None if args.seed is None else args.seed + trial
            report = run_one(
                beta,
                size=args.size,
                steps=args.steps,
                dt=args.dt,
                seed=seed,
                threshold_k=args.threshold_k,
                coherence=args.coherence,
            )
            n_values.append(int(report["n_sectors"]))
            wall_fracs.append(float(report["boundary_fraction"]))
            distinctions.append(float(report["wall_distinction"]))
            integrations.append(float(report["interior_integration"]))
            junctions.append(int(report["triple_junctions"]))

        print(
            f"{beta:>6.3f} | "
            f"{np.mean(n_values):>9.2f} | "
            f"{str(n_values):>16} | "
            f"{100 * np.mean(wall_fracs):>5.1f}% | "
            f"{np.mean(distinctions):>11.5f} | "
            f"{np.mean(integrations):>11.5f} | "
            f"{np.mean(junctions):>6.1f}"
        )

    print(
        "\nNote: the theory predicts N⋆=3 at β≈0.09. The 'N (all)' column shows "
        "per-trial measured sector counts. For an illustrative tradeoff curve, "
        "stationary_sector_count(a, b) returns (2a/3b)³ once a(β), b(β,κ) are fit."
    )
    # Illustrative: what a,b would be needed to predict N⋆=3.
    print(
        f"(For reference, stationary_sector_count(a=3, b=2) = "
        f"{stationary_sector_count(3.0, 2.0):.2f}.)"
    )


if __name__ == "__main__":
    main()
