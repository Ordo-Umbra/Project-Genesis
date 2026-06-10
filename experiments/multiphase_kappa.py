#!/usr/bin/env python3
"""Does κ scarcity select THREE genuine domains in the Ψ∈ℂ³ model?

The scalar phase diagram (`experiments/phase_diagram.py`) found that a 3-well
potential is S-optimal in a capacity-consumption band — but it selects the
*imposed* well count, not an emergent three-domain field (realized N was ~1 or
~39, not 3). This experiment closes that gap by running the **three-component
sector model** (which sustains real, mutually-adjacent domains with 120°
junctions) coupled to the **dynamical capacity field**, and asking two things
at once for each candidate phase count P:

1. Which P maximizes the time-averaged S-functional? (the selection question)
2. What is the *emergent* domain count N at that P? (the bridge the scalar
   model could not cross)

If capacity scarcity makes P=3 both S-optimal AND yields N≈3 genuine domains,
that is the cleanest in-silico echo of the URP SU(3) prediction this repo can
currently produce. The honest null result — P=3 wins on S but N≠3, or some
other P wins — is equally reported.

Usage::

    python experiments/multiphase_kappa.py
    python experiments/multiphase_kappa.py --consumptions 20,40,80 --trials 4
    python experiments/multiphase_kappa.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from project_genesis.multiphase import (  # noqa: E402
    analyze_multiphase,
    count_domains,
    multiphase_s_functional,
    sector_labels,
    step_multiphase_kappa,
)


def run_cell(
    n_phases: int,
    consumption: float,
    *,
    size: int,
    steps: int,
    window: int,
    dt: float,
    gamma: float,
    diffusion: float,
    recovery: float,
    seed: int,
) -> dict:
    """Evolve one (P, consumption) cell; return mean S, emergent N, junctions."""
    rng = np.random.default_rng(seed)
    fields = rng.random((n_phases, size, size)) * 0.1
    kappa = np.ones((size, size))

    warmup = max(0, steps - window)
    prev = None
    for _ in range(warmup):
        prev = fields
        fields, kappa = step_multiphase_kappa(
            fields, kappa, diffusion=diffusion, gamma=gamma, dt=dt,
            kappa_consumption=consumption, kappa_recovery=recovery,
        )

    s_vals, n_vals, j_vals = [], [], []
    for _ in range(min(window, steps)):
        prev = fields
        fields, kappa = step_multiphase_kappa(
            fields, kappa, diffusion=diffusion, gamma=gamma, dt=dt,
            kappa_consumption=consumption, kappa_recovery=recovery,
        )
        s_vals.append(multiphase_s_functional(fields, prev, kappa_field=kappa)["s_increment"])

    labels = sector_labels(fields)
    return {
        "n_phases_avail": n_phases,
        "consumption": consumption,
        "mean_s": float(np.mean(s_vals)),
        "n_domains": count_domains(labels),
        "phases_present": int(len(np.unique(labels))),
        "kappa_mean": float(kappa.mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phases", type=str, default="2,3,4,5,6", help="Candidate component counts P.")
    p.add_argument("--consumptions", type=str, default="10,20,40,80,160")
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--diffusion", type=float, default=1.0)
    p.add_argument("--recovery", type=float, default=0.02)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="artifacts/multiphase_kappa.json")
    args = p.parse_args()

    phases = [int(x) for x in args.phases.split(",") if x.strip()]
    consumptions = [float(x) for x in args.consumptions.split(",") if x.strip()]

    runs = len(phases) * len(consumptions) * args.trials
    print(
        f"Ψ∈ℂ³ + dynamical-κ experiment | {len(phases)} P × {len(consumptions)} c "
        f"× {args.trials} trials = {runs} runs | {args.size}² × {args.steps} steps "
        f"(S over last {args.window})\n"
    )

    all_cells = []
    # Organize as consumption rows; within each, find S-maximizing P and its emergent N.
    print(f"{'consumption':>11} | " + " | ".join(f"P={p}" for p in phases) + " || best-P  emergent-N")
    print("-" * (13 + 7 * len(phases) + 22))

    for c in consumptions:
        per_p = {}
        for n_phases in phases:
            cells = [
                run_cell(
                    n_phases, c, size=args.size, steps=args.steps, window=args.window,
                    dt=args.dt, gamma=args.gamma, diffusion=args.diffusion,
                    recovery=args.recovery, seed=args.seed + t,
                )
                for t in range(args.trials)
            ]
            per_p[n_phases] = {
                "mean_s": float(np.mean([x["mean_s"] for x in cells])),
                "n_domains": float(np.mean([x["n_domains"] for x in cells])),
            }
            all_cells.extend(cells)

        best_p = max(per_p, key=lambda pp: per_p[pp]["mean_s"])
        s_cells = " | ".join(
            f"{per_p[pp]['mean_s']:.4f}{'*' if pp == best_p else ' '}" for pp in phases
        )
        mark = " ★" if best_p == 3 else "  "
        print(
            f"{c:>11} | {s_cells} || P={best_p}{mark}  N≈{per_p[best_p]['n_domains']:.1f} "
            f"(N@P=3≈{per_p.get(3, {}).get('n_domains', float('nan')):.1f})"
        )

    # Verdict summary.
    print("\nSelection summary:")
    for c in consumptions:
        row = [x for x in all_cells if x["consumption"] == c]
        by_p = {}
        for pp in phases:
            sub = [x["mean_s"] for x in row if x["n_phases_avail"] == pp]
            by_p[pp] = float(np.mean(sub))
        best = max(by_p, key=lambda pp: by_p[pp])
        n3 = np.mean([x["n_domains"] for x in row if x["n_phases_avail"] == 3])
        verdict = "3 wins on S" if best == 3 else f"{best} wins on S"
        bridge = "and emergent N≈3" if 2.5 <= n3 <= 3.5 else f"but emergent N≈{n3:.1f}"
        print(f"  c={c:>6}: {verdict}; {bridge}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_cells, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
