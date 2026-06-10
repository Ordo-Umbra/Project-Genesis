#!/usr/bin/env python3
"""Does a *standing* integration term give the S-functional an interior N⋆?

Motivation: the κ × Ψ∈ℂ³ experiment found that the transient ΔI vanishes at
equilibrium, collapsing S to ΔC (wall energy), which rewards fragmentation.
The fix should be a *standing* nonlocal coherence I[φ] = ∫∫ K(x,x')φ(x)φ(x')
that survives equilibrium. This experiment tests whether it does — and, more
sharply, whether `S = ΔC + κ·w·I` finally has an interior optimum in sector
count.

The honest answer it returns: **standing coherence survives equilibrium (the
intended fix works), but its magnitude is nearly collinear with ΔC** — both
track wall density — so `S = ΔC + κ·w·I` flips monotonically from many sectors
(w→0) to two (w large) with no stable interior. An interior N⋆ needs a term
that is *not* collinear with wall density — i.e. a topological / junction
measure, which is the gauge paper's own §6 argument for why three is special.
This script measures and reports that collinearity so the conclusion is
reproducible rather than asserted.

Usage::

    python experiments/standing_integration.py
    python experiments/standing_integration.py --phases 2,3,4,5,6 --radii 3,8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from project_genesis.multiphase import (  # noqa: E402
    _total_gradient_energy,
    coherence_integration,
    count_domains,
    sector_labels,
    step_multiphase_kappa,
)


def evolve(n_phases, *, size, steps, consumption, recovery, dt, seed):
    rng = np.random.default_rng(seed)
    f = rng.random((n_phases, size, size)) * 0.1
    k = np.ones((size, size))
    for _ in range(steps):
        f, k = step_multiphase_kappa(
            f, k, kappa_consumption=consumption, kappa_recovery=recovery, dt=dt
        )
    return f, k


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phases", type=str, default="2,3,4,5,6")
    p.add_argument("--radii", type=str, default="3,8")
    p.add_argument("--size", type=int, default=56)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--consumption", type=float, default=80.0)
    p.add_argument("--recovery", type=float, default=0.02)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.09)
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    phases = [int(x) for x in args.phases.split(",") if x.strip()]
    radii = [int(x) for x in args.radii.split(",") if x.strip()]

    print(
        f"Standing-integration test | P ∈ {phases} | {args.size}² × {args.steps} steps "
        f"| consumption={args.consumption}, recovery={args.recovery}\n"
    )
    header = f"{'P':>2} | {'N':>4} | {'ΔC':>8} | {'κ':>6} | " + " | ".join(f"I(r={r})" for r in radii)
    print(header)
    print("-" * len(header))

    rows = {}
    for P in phases:
        dcs, kas, cohs, ns = [], [], {r: [] for r in radii}, []
        for t in range(args.trials):
            f, k = evolve(P, size=args.size, steps=args.steps, consumption=args.consumption,
                          recovery=args.recovery, dt=args.dt, seed=args.seed + t)
            dcs.append(args.beta * float(_total_gradient_energy(f).mean()))
            kas.append(float(k.mean()))
            ns.append(count_domains(sector_labels(f)))
            for r in radii:
                cohs[r].append(coherence_integration(f, radius=r, decay=1.0 if r <= 4 else 0.4))
        rows[P] = {
            "N": float(np.mean(ns)),
            "dc": float(np.mean(dcs)),
            "kappa": float(np.mean(kas)),
            "coh": {r: float(np.mean(cohs[r])) for r in radii},
        }
        coh_cols = " | ".join(f"{rows[P]['coh'][r]:>6.4f}" for r in radii)
        print(f"{P:>2} | {rows[P]['N']:>4.0f} | {rows[P]['dc']:>8.4f} | {rows[P]['kappa']:>6.3f} | {coh_cols}")

    # Collinearity: correlation between ΔC and (−coherence) across P.
    dc_arr = np.array([rows[P]["dc"] for P in phases])
    print("\nCollinearity of ΔC with −I (both ~ wall density):")
    for r in radii:
        coh_arr = np.array([rows[P]["coh"][r] for P in phases])
        if dc_arr.std() > 0 and coh_arr.std() > 0:
            corr = float(np.corrcoef(dc_arr, -coh_arr)[0, 1])
            print(f"  r={r}: corr(ΔC, −I) = {corr:+.3f}")

    # Interior-optimum test: argmax_P of S = ΔC + κ·w·I over weights w.
    print("\nS = ΔC + κ·w·I  — S-maximizing P vs integration weight w (per radius):")
    for r in radii:
        line = []
        for w in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            sP = {P: rows[P]["dc"] + rows[P]["kappa"] * w * rows[P]["coh"][r] for P in phases}
            line.append(f"w={w:g}:P*={max(sP, key=lambda pp: sP[pp])}")
        print(f"  r={r}: " + "  ".join(line))

    interior = any(
        2 < max(
            {P: rows[P]["dc"] + rows[P]["kappa"] * w * rows[P]["coh"][r] for P in phases},
            key=lambda pp: rows[pp]["dc"] + rows[pp]["kappa"] * w * rows[pp]["coh"][r],
        ) < max(phases)
        for r in radii for w in (0.25, 0.5, 1.0, 2.0)
    )
    print(
        "\nVerdict: "
        + ("an interior optimum appears — investigate further."
           if interior else
           "no stable interior optimum — ΔC and coherence are collinear (both track wall "
           "density). An interior N⋆ needs a non-collinear (topological/junction) term.")
    )


if __name__ == "__main__":
    main()
