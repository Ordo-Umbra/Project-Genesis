"""Quark generations: the dimensional-form hierarchy, fixed by junction topology.

The collab explorations read the field's emergent structures as **dimensional
families** — 0D points, 1D lines, 2D blobs — a "quark generation hierarchy"
whose densities shifted with phase.  This experiment puts that reading on a
rigorous footing: the families are the **cells of the sector tessellation's
CW-complex** (`project_genesis/dimensional_forms.py`), and the census carries
two exact facts the collab morphology-counting never had —

- **Euler on the torus.**  A clean CW-decomposition obeys ``V − E + F = 0``.
- **The trivalent (N⋆=3) signature.**  Where walls meet in threes — the
  120° Y-junctions the repo's sector program selects — ``2E = 3V`` and Euler
  fix the hierarchy at ``V : E : F = 2 : 3 : 1``, no freedom.

The phase axis is temperature (the collab's β ↔ 1/T): a three-component
Allen–Cahn sector field (`multiphase.step_multiphase`) is coarsened from
noise at each T, and censused.  Cold coarsens to a clean tessellation of a
few large domains; heating shatters it into speckle — deconfinement.

Pre-registered predictions:

Q1. **Topology fixes the confined hierarchy**: in the coldest (confined)
    phase the mean junction valence is 3.0 ± 0.3 (Y-junctions) and the
    census ratio ``V : E : F`` sits within 0.6 of the trivalent ``2 : 3 : 1``
    — the generation hierarchy is the junction constraint, not a free
    density.
Q2. **The Euler defect is a deconfinement order parameter**: the normalised
    defect ``|V − E + F| / (V + E + F)`` is small (< 0.12) in the confined
    phase and rises by ≥ 5× across the sweep to the hot phase — the clean
    tessellation shatters, and the census's own topological consistency
    tracks it.
Q3. **The form densities track the phase**: the 2D-interior area fraction
    falls monotonically as T rises while the wall+junction (1D+0D) fraction
    rises — the different phases carry different densities of the forms, the
    collab's central reading, now measured on a validated instrument.

Usage::

    python experiments/n3_quark_generations.py --output-dir artifacts/n3_quarks
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


def coarsen(args, temperature: float, seed: int) -> np.ndarray:
    """Allen–Cahn coarsening from noise at fixed temperature; returns labels."""
    rng = np.random.default_rng(seed)
    fields = args.init_amp * rng.standard_normal((args.palette, args.size,
                                                  args.size))
    noise_scale = np.sqrt(2.0 * temperature * args.dt)
    for _ in range(args.steps):
        fields = step_multiphase(fields, diffusion=1.0, gamma=1.5, dt=args.dt)
        if temperature > 0.0:
            fields = fields + noise_scale * rng.standard_normal(fields.shape)
    return sector_labels(fields)


def census_at(args, temperature: float) -> dict:
    """Seed-averaged census + area fractions at one temperature."""
    rows = [dimensional_census(coarsen(args, temperature, args.seed + 91 * s))
            for s in range(args.n_seeds)]
    keys = ("V", "E", "F", "euler", "mean_valence")
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    tot = out["V"] + out["E"] + out["F"]
    out["euler_defect"] = abs(out["euler"]) / max(tot, 1.0)
    out["ratio_VF"] = out["V"] / max(out["F"], 1.0)
    out["ratio_EF"] = out["E"] / max(out["F"], 1.0)
    # area fractions (interior vs wall vs junction), one representative field
    dim = local_dimension(coarsen(args, temperature, args.seed))
    n2 = dim.size
    out["area_2D"] = float(np.mean(dim == 2))
    out["area_1D"] = float(np.mean(dim == 1))
    out["area_0D"] = float(np.mean(dim == 0))
    out["temperature"] = temperature
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--palette", type=int, default=3)
    p.add_argument("--temperatures", type=float, nargs="+",
                   default=[0.0, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--init-amp", type=float, default=0.1)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_quarks")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.temperatures = [0.0, 0.2]
        args.n_seeds = 2
        args.steps = 140

    os.makedirs(args.output_dir, exist_ok=True)
    print("quark generations: the dimensional-form hierarchy vs phase",
          flush=True)

    scan = []
    for T in args.temperatures:
        row = census_at(args, T)
        scan.append(row)
        print(f"  T={T:<5}: V={row['V']:6.1f} E={row['E']:6.1f} "
              f"F={row['F']:6.1f}  valence={row['mean_valence']:.2f}  "
              f"V:E:F={row['ratio_VF']:.2f}:{row['ratio_EF']:.2f}:1  "
              f"euler_defect={row['euler_defect']:.3f}  "
              f"area2D={row['area_2D']:.2f}", flush=True)

    cold, hot = scan[0], scan[-1]
    q1 = (abs(cold["mean_valence"] - 3.0) <= 0.3
          and abs(cold["ratio_VF"] - 2.0) <= 0.6
          and abs(cold["ratio_EF"] - 3.0) <= 0.6)
    q2 = (cold["euler_defect"] < 0.12
          and hot["euler_defect"] >= 5.0 * max(cold["euler_defect"], 1e-6))
    area2d = [r["area_2D"] for r in scan]
    wall = [r["area_1D"] + r["area_0D"] for r in scan]
    q3 = (all(b <= a + 1e-3 for a, b in zip(area2d, area2d[1:]))
          and all(b >= a - 1e-3 for a, b in zip(wall, wall[1:]))
          and area2d[0] - area2d[-1] > 0.05)

    lines = ["Quark generations — verdict", "=" * 74, ""]
    lines.append(
        f"Q1 (topology fixes the confined hierarchy): coldest phase valence "
        f"= {cold['mean_valence']:.2f} (Y-junctions → 3), census ratio "
        f"V:E:F = {cold['ratio_VF']:.2f}:{cold['ratio_EF']:.2f}:1 vs the "
        f"trivalent 2:3:1 — "
        + ("✓ the generation hierarchy is the junction constraint, not a "
           "free density: N⋆=3 Y-junctions force 2:3:1." if q1 else
           "✗ the confined ratio is not the trivalent 2:3:1."))
    lines.append(
        f"Q2 (Euler defect = deconfinement order parameter): "
        f"|V−E+F|/(V+E+F) rises {cold['euler_defect']:.3f} (confined) → "
        f"{hot['euler_defect']:.3f} (hot) — "
        + ("✓ the clean tessellation shatters into speckle; the census's "
           "own topological consistency is the order parameter." if q2 else
           "✗ the Euler defect does not track deconfinement."))
    lines.append(
        f"Q3 (form densities track the phase): 2D-interior area fraction "
        f"{area2d[0]:.2f} → {area2d[-1]:.2f} as T rises, wall+junction "
        f"{wall[0]:.2f} → {wall[-1]:.2f} — "
        + ("✓ the phases carry different densities of the forms — the "
           "collab's reading, on a validated census." if q3 else
           "✗ the densities do not shift monotonically with phase."))
    lines.append("")
    lines.append(f"score: {int(q1)+int(q2)+int(q3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: the census is exact on synthetic tilings (stripes, "
        "block, brick → Euler 0; tests/test_dimensional_forms.py) but "
        "carries a pixel-scale defect on smooth fields — short edges eaten "
        "by the junction-dilation cut inflate the confined-phase Euler a "
        "few % (recorded); the deconfined speckle is genuinely not a clean "
        "CW-complex (that IS Q2). Temperature is the phase dial (β ↔ 1/T), "
        "not the repo's gauge β; the 2D 'domain count' F is few-large in "
        "the confined phase (the collab counted blob morphology, a "
        "different instrument — the rigorous object here is the CW-cell "
        "census and its Euler/valence invariants); one lattice (96²), "
        "3-palette Allen–Cahn, seed-averaged; a spin (chiral) term is the "
        "registered next rung — it would split the 0D junctions by "
        "handedness.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_quark_generations.json"),
              "w") as fh:
        json.dump({"params": vars(args), "scan": scan,
                   "analysis": {"q1": bool(q1), "q2": bool(q2),
                                "q3": bool(q3)}}, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
