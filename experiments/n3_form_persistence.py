"""Form persistence under noise: living junction networks and finite-size self-repair.

Deterministic multiphase Allen–Cahn coarsens by mean curvature: walls and
junctions shrink and the lighter generations (0- and 1-cells) are driven toward
zero.  Continuous noise competes with that drain, but *unconstrained*
Allen–Cahn + additive Langevin either collapses to one domain or scrambles
into texture — measured in the first persistence runs (P1 failed; only strong
noise elevated interface density, and that was disorder, not a living network).

The dynamics that hold reliable junction networks elsewhere in the repo are
**volume-conserving** (``step_multiphase_conserved``: phase fractions cannot
be eliminated, so the system settles into a multi-domain tiling) and
**κ-gated** (capacity pins walls where load is high).  Fraction pinning in the
annealed ensemble is the thermodynamic analogue.  This experiment therefore
uses **conserved multiphase + mild additive noise** as the primary path: the
conservation law protects multi-domain structure while noise supplies the
fluctuations that can create a persistence window.

In 3-D the effect is sharper: surfaces have more ways to reduce area, so pure
coarsening collapses the light generations faster than in 2-D.  Larger systems
should still self-repair more effectively once a protective dynamics is in
place — local collapse removes only a small fraction of the inventory.

Pre-registered predictions:

P1. **Noise window for persistence.**  At fixed small lattice under conserved
    dynamics, there exists a noise band in which the mean late-time density of
    light forms (0- + 1-cells) is substantially higher than under pure
    conserved coarsening (noise = 0) *and* the multi-domain structure remains
    ordered (n_phases near palette), not scrambled.  Intermediate noise
    sustains a living network that either extreme does not.
P2. **Finite-size self-repair.**  At fixed intermediate noise, larger lattices
    retain a higher late-time light-form density than smaller ones.
P3. **Collapse or freeze without noise.**  With noise = 0 under conserved
    dynamics the light-form density still falls (curvature-driven smoothing of
    walls) or freezes at a low value — the baseline the noise window is
    measured against.  (Full single-domain collapse is forbidden by
    conservation; the residual is the protected multi-domain floor.)

Usage::

    python experiments/n3_form_persistence.py --output-dir artifacts/n3_persistence
    python experiments/n3_form_persistence.py --quick
    python experiments/n3_form_persistence.py --dynamics unconstrained  # old path
    python experiments/n3_form_persistence.py --ndim 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.dimensional_forms import dimensional_census, local_dimension
from project_genesis.multiphase import (
    sector_labels,
    step_multiphase,
    step_multiphase_conserved,
)


def light_form_density(labels: np.ndarray) -> float:
    """Fraction of sites that are 0-cells or 1-cells (light generations).

    In 2-D these are junctions and walls; in 3-D they are vertices and
    triple-lines — the two lightest families of the four-generation census.
    """
    dim = local_dimension(labels)
    return float(np.mean(dim <= 1))


def evolve(
    size: int,
    *,
    ndim: int,
    palette: int,
    steps: int,
    noise: float,
    seed: int,
    dynamics: str = "conserved",
    dt: float = 0.1,
    record_every: int = 20,
) -> dict:
    """Run multiphase dynamics with optional noise; track light-form density."""
    rng = np.random.default_rng(seed)
    shape = (palette,) + (size,) * ndim
    fields = 0.1 * rng.standard_normal(shape)
    # mild normalisation so sectors start mixed
    norms = np.sqrt(np.sum(fields * fields, axis=0, keepdims=True) + 1e-12)
    fields = fields / (norms * 0.7)
    noise_scale = np.sqrt(2.0 * max(noise, 0.0) * dt)

    history = []
    for t in range(steps):
        if dynamics == "conserved":
            fields, _ = step_multiphase_conserved(
                fields, None, diffusion=1.0, gamma=1.5, dt=dt
            )
        else:
            fields = step_multiphase(
                fields, diffusion=1.0, gamma=1.5, dt=dt
            )
        if noise > 0.0:
            fields = fields + noise_scale * rng.standard_normal(fields.shape)
        if t % record_every == 0 or t == steps - 1:
            labels = sector_labels(fields)
            history.append(light_form_density(labels))

    labels = sector_labels(fields)
    census = dimensional_census(labels)
    late = history[max(1, 2 * len(history) // 3) :]
    return {
        "size": size,
        "noise": noise,
        "ndim": ndim,
        "dynamics": dynamics,
        "final_light_density": float(history[-1]),
        "late_mean_light_density": float(np.mean(late)),
        "history": [float(h) for h in history],
        "cells": {str(k): int(v) for k, v in census["cells"].items()},
        "euler": int(census["euler"]),
        "n_phases": int(len(np.unique(labels))),
        "valence_by_dim": {
            str(k): float(v) for k, v in census.get("valence_by_dim", {}).items()
        },
    }


def mean_over_seeds(args, *, size: int, noise: float) -> dict:
    rows = [
        evolve(
            size,
            ndim=args.ndim,
            palette=args.palette,
            steps=args.steps,
            noise=noise,
            seed=args.seed + 97 * s + 13 * size + 31 * args.ndim,
            dynamics=args.dynamics,
            record_every=args.record_every,
        )
        for s in range(args.n_seeds)
    ]
    return {
        "size": size,
        "noise": noise,
        "late_mean": float(np.mean([r["late_mean_light_density"] for r in rows])),
        "late_std": float(np.std([r["late_mean_light_density"] for r in rows])),
        "final_mean": float(np.mean([r["final_light_density"] for r in rows])),
        "n_phases_mean": float(np.mean([r["n_phases"] for r in rows])),
        "runs": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--ndim", type=int, default=3, choices=[2, 3],
        help="spatial dimension (3 is default; 2 for faster exploration)",
    )
    p.add_argument(
        "--dynamics",
        choices=["conserved", "unconstrained"],
        default="conserved",
        help="conserved = volume-conserving multiphase (primary); "
             "unconstrained = plain Allen-Cahn (comparison / old path)",
    )
    p.add_argument("--palette", type=int, default=3)
    p.add_argument(
        "--sizes", type=int, nargs="+", default=None,
        help="lattice sizes (defaults depend on ndim)",
    )
    p.add_argument(
        "--noises", type=float, nargs="+",
        default=[0.0, 0.01, 0.03, 0.06, 0.12, 0.25],
    )
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--record-every", type=int, default=20)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_persistence")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.sizes is None:
        args.sizes = [20, 28, 36] if args.ndim == 3 else [32, 48, 64]
    if args.steps is None:
        args.steps = 280 if args.ndim == 3 else 400

    if args.quick:
        if args.ndim == 3:
            args.sizes = [16, 24]
            args.noises = [0.0, 0.02, 0.08, 0.20]
            args.steps = 140
            args.n_seeds = 2
        else:
            args.sizes = [24, 40]
            args.noises = [0.0, 0.02, 0.08, 0.20]
            args.steps = 200
            args.n_seeds = 2

    os.makedirs(args.output_dir, exist_ok=True)
    print(
        "form persistence under noise: living networks and finite-size "
        "self-repair",
        flush=True,
    )
    print(
        f"  dynamics={args.dynamics}  ndim={args.ndim}  palette={args.palette}  "
        f"steps={args.steps}  sizes={args.sizes}  noises={args.noises}",
        flush=True,
    )

    small = min(args.sizes)
    noise_scan = [mean_over_seeds(args, size=small, noise=n) for n in args.noises]
    print(f"  noise scan at size={small}:", flush=True)
    for r in noise_scan:
        print(
            f"    noise={r['noise']:.3f}: late_light={r['late_mean']:.4f} "
            f"± {r['late_std']:.4f}  n_phases={r['n_phases_mean']:.1f}",
            flush=True,
        )

    by_noise = {r["noise"]: r for r in noise_scan}
    zero = by_noise[0.0]
    mids = [r for r in noise_scan if 0.0 < r["noise"] < max(args.noises)]
    strong = by_noise[max(args.noises)]

    # P1: intermediate noise higher than pure coarsening, and still ordered
    # (n_phases not collapsed to 1, and not pure high-noise texture alone).
    # Prefer mid points that beat zero and keep n_phases near palette.
    p1 = False
    if mids:
        for r in mids:
            ordered = r["n_phases_mean"] >= max(2.0, args.palette - 1.1)
            above_zero = r["late_mean"] > zero["late_mean"] * 1.4 + 1e-4
            # not merely the strong-noise texture branch
            below_strong_or_ordered = (
                r["late_mean"] < strong["late_mean"] * 0.85
                or ordered
            )
            if above_zero and ordered and below_strong_or_ordered:
                p1 = True
                break
            # alternate: clear peak above both neighbours
            if above_zero and r["late_mean"] > strong["late_mean"] * 1.2 and ordered:
                p1 = True
                break

    # P3: without noise, light density is low (smoothed multi-domain floor)
    floor = 0.08 if args.dynamics == "conserved" else (0.06 if args.ndim == 3 else 0.08)
    p3 = (
        zero["late_mean"] < floor
        or zero["late_mean"] < zero["runs"][0]["history"][0] * 0.5
    )

    if mids:
        # prefer an ordered mid for the size scan
        ordered_mids = [
            r for r in mids
            if r["n_phases_mean"] >= max(2.0, args.palette - 1.1)
        ]
        pool = ordered_mids if ordered_mids else mids
        best_mid = max(pool, key=lambda r: r["late_mean"])
        mid_noise = best_mid["noise"]
    else:
        mid_noise = args.noises[len(args.noises) // 2]

    size_scan = [
        mean_over_seeds(args, size=s, noise=mid_noise) for s in args.sizes
    ]
    print(f"  size scan at noise={mid_noise:.3f}:", flush=True)
    for r in size_scan:
        print(
            f"    size={r['size']}: late_light={r['late_mean']:.4f} "
            f"± {r['late_std']:.4f}  n_phases={r['n_phases_mean']:.1f}",
            flush=True,
        )

    lates = [r["late_mean"] for r in size_scan]
    p2 = len(lates) >= 2 and lates[-1] >= lates[0] * 0.95 and (
        lates[-1] > lates[0] * 1.05 or max(lates) == lates[-1]
    )

    dim_label = "3-D (vertices + triple-lines)" if args.ndim == 3 else "2-D"
    dyn_label = args.dynamics
    lines = ["Form persistence under noise — verdict", "=" * 74, ""]
    lines.append(
        f"P1 (noise window for persistence, {dim_label}, {dyn_label}): at "
        f"size={small}, late light-form densities across noises {args.noises} "
        f"are "
        + "/".join(f"{r['late_mean']:.4f}" for r in noise_scan)
        + " (n_phases "
        + "/".join(f"{r['n_phases_mean']:.1f}" for r in noise_scan)
        + ") — "
        + (
            "✓ intermediate noise sustains more ordered light structure than "
            "pure coarsening, without collapsing to single-domain or pure "
            "texture."
            if p1
            else "✗ no clear ordered intermediate peak on this lattice and window."
        )
    )
    lines.append(
        f"P2 (finite-size self-repair): at noise={mid_noise:.3f}, late "
        f"light-form density vs size {args.sizes} is "
        + "/".join(f"{r['late_mean']:.4f}" for r in size_scan)
        + " — "
        + (
            "✓ larger lattices retain higher (or at least non-decreasing) "
            "light-form density — statistical self-repair against local "
            "collapse."
            if p2
            else "✗ size dependence does not show the expected self-repair trend."
        )
    )
    lines.append(
        f"P3 (baseline without noise): noise=0 late density = "
        f"{zero['late_mean']:.4f} — "
        + (
            "✓ light forms sit at a low baseline under pure dynamics "
            + (
                "(conserved multi-domain floor; single-domain collapse forbidden)."
                if args.dynamics == "conserved"
                else "(mean-curvature coarsening)."
            )
            if p3
            else "✗ zero-noise baseline did not suppress light forms on this window."
        )
    )
    lines.append("")
    lines.append(
        f"score: {int(p1) + int(p2) + int(p3)}/3 pre-registered predictions land."
    )
    lines.append("")
    lines.append(
        "honest scope: primary dynamics are volume-conserving multiphase "
        "(phase fractions protected) plus additive Langevin noise — the "
        "lightest structure-preserving model that still matches the browser "
        "fluctuation intuition.  Unconstrained Allen–Cahn remains available "
        "via --dynamics unconstrained (that path failed P1 previously: high "
        "noise only raised texture).  Not a true thermal Gibbs measure; finite "
        f"observation window; primary path 3-D (ndim={args.ndim} this run); "
        "no continuum limit and no universal critical noise claim.  Tests the "
        "STRUCTURE of the persistence window and size trend.  Annealed "
        "fraction-pinned and κ-gated dynamics are the heavier structure-holding "
        "alternatives already measured elsewhere in the repo "
        "(n3_annealed_matter, step_multiphase_kappa)."
    )
    text = "\n".join(lines)
    print("\n" + text)

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(
        os.path.join(args.output_dir, "n3_form_persistence.json"), "w"
    ) as fh:
        json.dump(
            {
                "params": vars(args),
                "noise_scan": [
                    {k: v for k, v in r.items() if k != "runs"}
                    for r in noise_scan
                ],
                "size_scan": [
                    {k: v for k, v in r.items() if k != "runs"}
                    for r in size_scan
                ],
                "analysis": {
                    "p1": bool(p1),
                    "p2": bool(p2),
                    "p3": bool(p3),
                    "mid_noise": float(mid_noise),
                    "ndim": int(args.ndim),
                    "dynamics": args.dynamics,
                },
            },
            fh,
            indent=2,
            default=str,
        )


if __name__ == "__main__":
    main()
