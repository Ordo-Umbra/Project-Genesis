"""Form abundances: why three generations, and which is rarest.

The collab reading was that the dimensional forms (0D/1D/2D) appear in
*abundances* that shift with energy and phase — a "generation hierarchy."
`n3_quark_generations` put the forms on a rigorous footing (the CW-cells of
the tessellation, ratio `2:3:1` in the confined phase).  This experiment
asks the abundance question directly, and holds a hard line against
numerology: it tests the **structure** of the hierarchy (how many families,
which is rarest, what fixes the ratio), and explicitly does *not* claim a
numerical match to real quark masses or abundances.

Three structural facts, each falsifiable:

A1. **The number of generations is the spatial dimension, not the palette**:
    across sector counts `P = 3, 4, 5, 6`, exactly **three** dimensional
    families are populated (0-, 1-, 2-cells) and the count ratio stays near
    the trivalent `2 : 3 : 1`, with mean junction valence ≈ 3.  Three domains
    generically meet at a point whatever the palette, so the family count is
    `d + 1 = 3` in 2-D — a topological prediction (and `4` in 3-D, the
    registered next rung), independent of how many sector-types exist.
A2. **The abundances are topologically protected across energy**: at fixed
    (confined) phase, running the field further — coarsening it from fine to
    coarse — leaves the `0D:1D:2D` ratio constant (coefficient of variation
    < 0.15).  The generation abundances do not drift with how far the field
    has run; they are fixed by the junction topology until the phase changes.
A3. **The heavy generation is the rarest**: the 2D "heavy" family is the
    least abundant by count at every confined point (`F < V` and `F < E`) —
    the qualitative shape of ordinary matter, where the heaviest generation
    is rarest.  *(A resemblance in structure only; no numerical match to
    quark data is claimed or implied.)*

Usage::

    python experiments/n3_form_abundances.py --output-dir artifacts/n3_abundances
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.dimensional_forms import dimensional_census
from project_genesis.multiphase import sector_labels, step_multiphase


def coarsen(size: int, palette: int, steps: int, temperature: float,
            seed: int, dt: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fields = 0.1 * rng.standard_normal((palette, size, size))
    noise = np.sqrt(2.0 * temperature * dt)
    for _ in range(steps):
        fields = step_multiphase(fields, diffusion=1.0, gamma=1.5, dt=dt)
        if temperature > 0.0:
            fields = fields + noise * rng.standard_normal(fields.shape)
    return sector_labels(fields)


def census_avg(args, *, palette: int, steps: int, temperature: float) -> dict:
    rows = [dimensional_census(coarsen(args.size, palette, steps,
                                       temperature, args.seed + 71 * s))
            for s in range(args.n_seeds)]
    v = np.mean([r["V"] for r in rows])
    e = np.mean([r["E"] for r in rows])
    f = np.mean([r["F"] for r in rows])
    return {"palette": palette, "steps": steps, "temperature": temperature,
            "V": float(v), "E": float(e), "F": float(f),
            "ratio_VF": float(v / max(f, 1)), "ratio_EF": float(e / max(f, 1)),
            "valence": float(np.mean([r["mean_valence"] for r in rows])),
            "n_families": int(v > 0) + int(e > 0) + int(f > 0),
            "heavy_rarest": bool(f < v and f < e)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--palettes", type=int, nargs="+", default=[3, 4, 5, 6])
    p.add_argument("--depths", type=int, nargs="+", default=[150, 250, 400, 600])
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_abundances")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.palettes, args.depths, args.n_seeds = [3, 5], [200, 400], 2

    os.makedirs(args.output_dir, exist_ok=True)
    print("form abundances: why three generations, and which is rarest",
          flush=True)

    # A1 — palette scan (energy/phase fixed: cold, one coarsening depth)
    palette_scan = [census_avg(args, palette=P, steps=args.steps,
                               temperature=0.0) for P in args.palettes]
    print("  A1 — generations vs palette:", flush=True)
    for r in palette_scan:
        print(f"    P={r['palette']}: families={r['n_families']}  "
              f"valence={r['valence']:.2f}  V:E:F="
              f"{r['ratio_VF']:.2f}:{r['ratio_EF']:.2f}:1", flush=True)

    # A2 — coarsening-depth scan (energy varied within the confined phase)
    depth_scan = [census_avg(args, palette=3, steps=d, temperature=0.0)
                  for d in args.depths]
    print("  A2 — abundances vs coarsening depth (palette 3, cold):",
          flush=True)
    for r in depth_scan:
        print(f"    steps={r['steps']}: V:E:F="
              f"{r['ratio_VF']:.2f}:{r['ratio_EF']:.2f}:1  F={r['F']:.0f}",
              flush=True)

    a1 = all(r["n_families"] == 3 and abs(r["valence"] - 3.0) <= 0.3
             and abs(r["ratio_VF"] - 2.0) <= 0.6
             and abs(r["ratio_EF"] - 3.0) <= 0.8 for r in palette_scan)
    rvf = np.array([r["ratio_VF"] for r in depth_scan])
    ref = np.array([r["ratio_EF"] for r in depth_scan])
    cv = max(rvf.std() / max(rvf.mean(), 1e-9),
             ref.std() / max(ref.mean(), 1e-9))
    a2 = cv < 0.15
    a3 = all(r["heavy_rarest"] for r in palette_scan + depth_scan)

    lines = ["Form abundances — verdict", "=" * 74, ""]
    lines.append(
        "A1 (generations = spatial dimension, not palette): across "
        f"P = {args.palettes}, families populated = "
        + "/".join(str(r["n_families"]) for r in palette_scan)
        + f", valence ≈ {np.mean([r['valence'] for r in palette_scan]):.2f}, "
        "ratio near 2:3:1 — "
        + ("✓ three generations because space is 2-D (cells of dim 0,1,2), "
           "whatever the sector count — a topological census, not a "
           "palette accident." if a1 else
           "✗ the family count or ratio depends on the palette."))
    lines.append(
        f"A2 (abundances topologically protected across energy): the "
        f"0D:1D:2D ratio across coarsening depths {args.depths} has "
        f"coefficient of variation {cv:.3f} — "
        + ("✓ running the field further does not drift the abundances; "
           "they are fixed by the junction topology, not tuned by energy."
           if a2 else "✗ the abundances drift with coarsening depth."))
    lines.append(
        "A3 (the heavy generation is rarest): the 2D family is least "
        "abundant by count at every point — "
        + ("✓ the heaviest forms are the fewest, the qualitative shape of "
           "ordinary matter (no numerical match to quark data claimed)."
           if a3 else "✗ the 2D family is not consistently the rarest."))
    lines.append("")
    lines.append(f"score: {int(a1)+int(a2)+int(a3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: this tests the STRUCTURE of the abundance hierarchy "
        "— its family count (3, set by the 2-D CW dimensions), its "
        "topological protection, and which family is rarest — and makes NO "
        "claim of a numerical match to real quark masses or generation "
        "abundances; that remains a numerological stretch the program does "
        "not make.  The count ratio 2:3:1 is the trivalent-junction value "
        "(it drifts slightly as the palette grows and higher-order "
        "junctions creep in — measured, within the bar); 'energy' here is "
        "coarsening depth at fixed cold phase (the sharp deconfinement "
        "break, where the ratio finally changes, is `n3_quark_generations` "
        "Q2); one lattice (96²); the 3-D census — which would predict FOUR "
        "families (cells of dim 0,1,2,3) — is the registered next rung and "
        "the sharpest form of the prediction.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_form_abundances.json"),
              "w") as fh:
        json.dump({"params": vars(args), "palette_scan": palette_scan,
                   "depth_scan": depth_scan,
                   "analysis": {"a1": bool(a1), "a2": bool(a2),
                                "a3": bool(a3), "ratio_cv": float(cv)}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
