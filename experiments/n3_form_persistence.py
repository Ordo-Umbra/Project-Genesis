"""Form persistence under noise: living junction networks and finite-size self-repair.

Deterministic multiphase Allen–Cahn coarsens by mean curvature: walls and
junctions shrink and the lighter generations (0- and 1-cells) are driven toward
zero.  Continuous additive noise competes with that drain by nucleating and
roughening interfaces.  When the two rates balance, the tessellation can
persist as a non-equilibrium fluctuating regime — a living junction network —
instead of collapsing to a single domain.

In 3-D the effect is sharper: surfaces have more ways to reduce area, so pure
coarsening collapses the light generations (vertices and triple-lines) faster
than in 2-D.  Browser runs on small lattices showed that intermediate noise can
still sustain long-lived fluctuating networks, and that the balance is delicate
and history-dependent.  The same size dependence already measured for structure
emergence (larger lattices form and hold clean tessellations more reliably)
suggests larger systems should also *self-repair* more effectively: local
collapse removes only a small fraction of the form inventory, and new walls can
nucleate elsewhere while one region dies.

This experiment turns those observations into pre-registered structural claims
in **3-D** (the dimension where the four-generation census lives).  2-D remains
available via ``--ndim 2`` for faster exploration.  It does **not** claim a
continuum critical noise value, a true thermal Gibbs measure, or numerical match
to any physical generation data.

Pre-registered predictions:

P1. **Noise window for persistence.**  At fixed small lattice, there exists a
    noise band in which the mean late-time density of light forms (0- + 1-cells:
    vertices + triple-lines in 3-D) is substantially higher than under pure
    deterministic coarsening (noise = 0) and substantially lower than under
    strong scrambling noise.  Intermediate noise sustains structure that either
    extreme destroys.
P2. **Finite-size self-repair.**  At fixed intermediate noise, larger lattices
    retain a higher late-time light-form density (or higher survival of
    multi-domain structure) than smaller ones — the statistical buffering that
    lets a large system offset local collapse by creation elsewhere.
P3. **Collapse under pure coarsening.**  With noise = 0 the light-form density
    falls toward zero on the observation window (the classical mean-curvature
    baseline; expected to be faster in 3-D than in 2-D).

Usage::

    python experiments/n3_form_persistence.py --output-dir artifacts/n3_persistence
    python experiments/n3_form_persistence.py --quick
    python experiments/n3_form_persistence.py --ndim 2   # faster 2-D scan
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.dimensional_forms import dimensional_census, local_dimension
from project_genesis.multiphase import sector_labels, step_multiphase


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
    dt: float = 0.1,
    record_every: int = 20,
) -> dict:
    """Run multiphase dynamics with additive noise; track light-form density."""
    rng = np.random.default_rng(seed)
    shape = (palette,) + (size,) * ndim
    fields = 0.1 * rng.standard_normal(shape)
    noise_scale = np.sqrt(2.0 * max(noise, 0.0) * dt)

    history = []
    for t in range(steps):
        fields = step_multiphase(fields, diffusion=1.0, gamma=1.5, dt=dt)
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

    # Dimension-aware defaults: 3-D coarsens faster and is costlier per site.
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
        f"  ndim={args.ndim}  palette={args.palette}  steps={args.steps}  "
        f"sizes={args.sizes}  noises={args.noises}",
        flush=True,
    )

    # --- noise scan at the smallest size (P1, P3) ---
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

    # P1: some intermediate noise higher than both extremes
    p1 = (
        any(
            r["late_mean"] > zero["late_mean"] * 1.5
            and r["late_mean"] > strong["late_mean"] * 1.3
            for r in mids
        )
        if mids
        else False
    )

    # P3: pure coarsening drives light forms down
    # 3-D is expected to collapse faster; slightly stricter absolute floor
    floor = 0.06 if args.ndim == 3 else 0.08
    p3 = (
        zero["late_mean"] < floor
        or zero["late_mean"] < zero["runs"][0]["history"][0] * 0.4
    )

    # --- size scan at intermediate noise (P2) ---
    if mids:
        best_mid = max(mids, key=lambda r: r["late_mean"])
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
    lines = ["Form persistence under noise — verdict", "=" * 74, ""]
    lines.append(
        f"P1 (noise window for persistence, {dim_label}): at size={small}, "
        f"late light-form densities across noises {args.noises} are "
        + "/".join(f"{r['late_mean']:.4f}" for r in noise_scan)
        + " — "
        + (
            "✓ intermediate noise sustains more light structure than both "
            "pure coarsening and strong scrambling."
            if p1
            else "✗ no clear intermediate peak above both extremes on this "
            "lattice and window."
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
            else "✗ size dependence does not show the expected self-repair "
            "trend."
        )
    )
    lines.append(
        f"P3 (collapse under pure coarsening): noise=0 late density "
        f"= {zero['late_mean']:.4f} — "
        + (
            "✓ light forms are driven down by mean-curvature coarsening, "
            "the baseline the noise window is measured against"
            + (" (faster drain expected in 3-D)." if args.ndim == 3 else ".")
            if p3
            else "✗ pure coarsening did not suppress light forms on this window."
        )
    )
    lines.append("")
    lines.append(
        f"score: {int(p1) + int(p2) + int(p3)}/3 pre-registered predictions land."
    )
    lines.append("")
    lines.append(
        "honest scope: additive Langevin noise on the multiphase Allen–Cahn "
        "field (not a true thermal bath of the free energy); finite observation "
        f"window; primary path is 3-D (ndim={args.ndim} this run) with sizes "
        "chosen for tractable volume; 2-D available via --ndim 2; one palette; "
        "no continuum limit and no claim of a universal critical noise value.  "
        "Tests the STRUCTURE of the persistence window and the size trend, not "
        "a precise threshold.  Browser observations on small lattices "
        "(intermittent long runs, history dependence, faster 3-D collapse) are "
        "the qualitative signal this formalises; the topological abundance "
        "results (n3_form_abundances, n3_3d_generations) remain the static "
        "backbone."
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
                },
            },
            fh,
            indent=2,
            default=str,
        )


if __name__ == "__main__":
    main()
