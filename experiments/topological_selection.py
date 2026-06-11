#!/usr/bin/env python3
"""Does a topological integration term finally give the S-functional N⋆=3?

This is the experiment the `Docs/Narrowing_the_N3_Question.md` corridor pointed
to. Two ingredients, both motivated by the earlier verdicts:

1. **Junction-resolving dynamics.** Plain Allen–Cahn coarsens without bound, so
   triple junctions are transient and the κ-pinned regime froze before any
   formed (zero junctions). Volume-conserving dynamics
   (`step_multiphase_conserved`) fix each phase's fraction, so a multi-domain
   tiling with persistent 120° junctions is *stable*.

2. **A topological (non-collinear) integration term.** Coherence magnitude was
   collinear with ΔC, so it could not create an interior optimum. The
   full-palette junction density — junctions that carry the *complete* colour
   palette (the §6 neutrality criterion) — is non-zero essentially only when
   the palette is three, because a 2-D junction is 3-fold. It is not collinear
   with wall density.

For each candidate palette size P this evolves the conserved field to a stable
junction network and measures ΔC, the neutrality term, and
`S = ΔC + κ·w·neutrality`. If S is maximized at P=3 robustly, that is the
cleanest in-silico echo of the SU(3) sector count the testbench can produce.

Caveat baked into the conclusion: this is a 2-D structural argument (3-fold
junction geometry). In 3-D, junctions are lines with different valence and the
clean selection need not transfer — a stated next step, not a closed case.

Usage::

    python experiments/topological_selection.py
    python experiments/topological_selection.py --phases 2,3,4,5,6 --trials 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from project_genesis.multiphase import (  # noqa: E402
    _total_gradient_energy,
    count_triple_junctions,
    full_palette_junction_density,
    sector_labels,
    step_multiphase_conserved,
)


def evolve_conserved(P, *, size, steps, gamma, diffusion, dt, seed):
    rng = np.random.default_rng(seed)
    f = rng.random((P, size, size)) * 0.1
    for _ in range(steps):
        f, _ = step_multiphase_conserved(f, diffusion=diffusion, gamma=gamma, dt=dt)
    return f


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phases", type=str, default="2,3,4,5,6")
    p.add_argument("--size", type=int, default=72)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--diffusion", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.09)
    p.add_argument("--weights", type=str, default="0,0.05,0.1,0.2,0.5")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="artifacts/topological_selection.json")
    args = p.parse_args()

    phases = [int(x) for x in args.phases.split(",") if x.strip()]
    weights = [float(w) for w in args.weights.split(",") if w.strip()]

    print(
        f"Topological selection | conserved dynamics | P ∈ {phases} | {args.size}² × "
        f"{args.steps} steps × {args.trials} trials\n"
    )
    print(f"{'P':>2} | {'Yjunc':>6} | {'ΔC':>8} | {'neutrality (full-palette Jdensity)':>34}")
    print("-" * 60)

    rows = {}
    for P in phases:
        dcs, neuts, juncs = [], [], []
        for t in range(args.trials):
            f = evolve_conserved(P, size=args.size, steps=args.steps, gamma=args.gamma,
                                 diffusion=args.diffusion, dt=args.dt, seed=args.seed + t)
            lab = sector_labels(f)
            dcs.append(args.beta * float(_total_gradient_energy(f).mean()))
            neuts.append(full_palette_junction_density(lab, P))
            juncs.append(count_triple_junctions(lab))
        rows[P] = {"dc": float(np.mean(dcs)), "neutrality": float(np.mean(neuts)),
                   "yjunc": float(np.mean(juncs))}
        print(f"{P:>2} | {rows[P]['yjunc']:>6.0f} | {rows[P]['dc']:>8.5f} | {rows[P]['neutrality']:>34.5f}")

    print("\nS = ΔC + w·neutrality — S-maximizing P vs integration weight w:")
    selected3 = 0
    for w in weights:
        sP = {P: rows[P]["dc"] + w * rows[P]["neutrality"] for P in phases}
        best = max(sP, key=lambda pp: sP[pp])
        if best == 3 and w > 0:
            selected3 += 1
        print(f"  w={w:>5}: P*={best}  (S: " + ", ".join(f"{P}:{sP[P]:.5f}" for P in phases) + ")")

    pos_weights = [w for w in weights if w > 0]
    verdict = (
        f"S is maximized at P=3 for {selected3}/{len(pos_weights)} positive weights "
        "— the topological term yields an interior optimum at THREE."
        if pos_weights and selected3 == len(pos_weights)
        else "S is not consistently maximized at P=3; inspect the table."
    )
    print(f"\nVerdict: {verdict}")
    print("Note: 2-D structural result (3-fold junctions). 3-D valence differs — next step.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({str(P): rows[P] for P in phases}, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
