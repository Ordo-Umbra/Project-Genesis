"""Flavour structure: the forms carry a second quantum number, and it is Pascal.

The dimensional census gave each form a **generation** — its cell dimension
(0D/1D/2D, three families because space is 2-D).  This experiment measures the
forms' *other* quantum number, the one the synthesis named as unmeasured: their
**flavour**, the identity of the sectors that compose them.  A domain is one
sector (a bare colour), a wall two (a colour–anticolour pair, meson-like), a
junction ``d + 1`` (a colour-singlet triple, baryon-like).  The claim held to a
hard line against numerology: it tests the **structure** — the multiplet sizes,
their symmetry, their conservation — and makes NO match to CKM angles or quark
masses.

Three structural facts, each falsifiable:

L1. **The flavour spectrum is Pascal's triangle**: with ``P`` sectors, the
    number of flavours realised at generation ℓ (cell dimension ℓ) is exactly
    ``C(P, d + 1 − ℓ)`` — a row of Pascal's triangle across the generations.
    In 2-D: domains ``C(P,1)``, walls ``C(P,2)``, junctions ``C(P,3)``.  Every
    flavour of the multiplet is realised (union over the ensemble), for
    P = 4 and P = 5.
L2. **Flavour is democratic under the sector symmetry**: with the sectors
    equivalent (symmetric Allen–Cahn), each multiplet is uniformly populated —
    normalised entropy ``> 0.85`` at every generation — an unbroken flavour
    symmetry.
L3. **Flavour is a conserved quantum number**: the flavour distribution is
    stable as the field is run further (coarsening depths spanning ×3) — the
    mean entropy stays ``> 0.85`` and drifts by ``< 0.15`` — flavour is a good
    label, not a transient of how far the field has coarsened.

The reading (not scored): generation × flavour are two **orthogonal** quantum
numbers a form carries at once — its dimensional family and its
sector-composition — the structural echo of a quark's generation and its
colour content.

Usage::

    python experiments/n3_flavour_structure.py --output-dir artifacts/n3_flavour
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.dimensional_forms import cell_flavours, flavour_entropy
from project_genesis.multiphase import sector_labels, step_multiphase


def coarsen(size: int, palette: int, steps: int, seed: int,
            dt: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fields = 0.1 * rng.standard_normal((palette, size, size))
    for _ in range(steps):
        fields = step_multiphase(fields, diffusion=1.0, gamma=1.5, dt=dt)
    return sector_labels(fields)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--palettes", type=int, nargs="+", default=[4, 5])
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--depths", type=int, nargs="+", default=[150, 250, 450])
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_flavour")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.palettes, args.n_seeds, args.depths = [4], 2, [150, 300]

    os.makedirs(args.output_dir, exist_ok=True)
    print("flavour structure: the forms' second quantum number", flush=True)
    d, gens = 2, (2, 1, 0)                    # 2-D: domains, walls, junctions
    names = {2: "domains", 1: "walls", 0: "juncs"}

    # L1 + L2 : the Pascal multiplet and its democracy, per palette
    spectrum = []
    for P in args.palettes:
        labs = [coarsen(args.size, P, args.steps, args.seed + 41 * s)
                for s in range(args.n_seeds)]
        row = {"palette": P, "gens": {}}
        for ell in gens:
            want = d + 1 - ell
            realised = set()
            entropies = []
            for lab in labs:
                fc = cell_flavours(lab, ell)
                realised |= set(fc.keys())
                entropies.append(flavour_entropy(fc))
            row["gens"][ell] = {
                "expected": comb(P, want), "realised": len(realised),
                "entropy": float(np.mean(entropies))}
        spectrum.append(row)
        line = "  ".join(
            f"{names[ell]}: {row['gens'][ell]['realised']}/"
            f"{row['gens'][ell]['expected']} H={row['gens'][ell]['entropy']:.2f}"
            for ell in gens)
        print(f"  P={P}: {line}", flush=True)

    # L3 : conservation across coarsening depth (palette 4)
    depth_H = []
    for depth in args.depths:
        Hs = []
        for s in range(args.n_seeds):
            lab = coarsen(args.size, 4, depth, args.seed + 41 * s)
            Hs += [flavour_entropy(cell_flavours(lab, ell)) for ell in gens]
        depth_H.append(float(np.mean(Hs)))
    print(f"  L3 — flavour entropy vs depth {args.depths}: "
          + "/".join(f"{h:.3f}" for h in depth_H), flush=True)

    l1 = all(row["gens"][ell]["realised"] == row["gens"][ell]["expected"]
             for row in spectrum for ell in gens)
    l2 = all(row["gens"][ell]["entropy"] > 0.85
             for row in spectrum for ell in gens)
    l3 = (min(depth_H) > 0.85 and max(depth_H) - min(depth_H) < 0.15)

    lines = ["Flavour structure — verdict", "=" * 74, ""]
    lines.append(
        "L1 (the flavour spectrum is Pascal's triangle): realised/expected "
        "flavours = "
        + "; ".join(
            f"P={row['palette']} ["
            + ",".join(f"{row['gens'][ell]['realised']}/"
                       f"{row['gens'][ell]['expected']}" for ell in gens)
            + "]" for row in spectrum)
        + " — "
        + ("✓ the multiplet sizes are a row of Pascal's triangle "
           "(C(P,1), C(P,2), C(P,3) across the generations), every flavour "
           "realised." if l1 else "✗ the multiplet sizes are not C(P,d+1−ℓ)."))
    lines.append(
        "L2 (flavour democracy under the sector symmetry): entropies "
        + "; ".join(f"P={row['palette']} "
                    + "/".join(f"{row['gens'][ell]['entropy']:.2f}"
                               for ell in gens) for row in spectrum)
        + " — "
        + ("✓ every multiplet is uniformly populated: an unbroken flavour "
           "symmetry." if l2 else "✗ some multiplet is not democratic."))
    lines.append(
        f"L3 (flavour is conserved): mean entropy across depths "
        f"{args.depths} = " + "/".join(f"{h:.3f}" for h in depth_H) + " — "
        + ("✓ the flavour distribution is stable as the field runs further — "
           "a good quantum number, not a coarsening transient." if l3 else
           "✗ the flavour distribution drifts with coarsening depth."))
    lines.append("")
    lines.append(f"score: {int(l1)+int(l2)+int(l3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: this measures the STRUCTURE of the flavour multiplet "
        "— its Pascal sizes, its symmetry, its conservation — and makes NO "
        "numerical match to CKM mixing or quark masses (declined, as with "
        "the generation abundances).  Flavour here is the sector-composition "
        "of a form (1 sector = a domain, 2 = a wall/meson-like pair, 3 = a "
        "junction/baryon-like triple); the generic composition size d+1−ℓ is "
        "used per generation (the occasional 4-sector junction is dropped as "
        "non-generic).  A flavour HIERARCHY from breaking the sector symmetry "
        "is NOT shown — the Allen–Cahn dynamics is winner-take-all, so a "
        "global sector bias drives total domination rather than a graded "
        "abundance; a symmetry-breaking that yields a genuine hierarchy is "
        "the open frontier.  One lattice (96²), seed-unioned for L1; the "
        "generation × flavour orthogonality is stated as a reading, not "
        "scored.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_flavour_structure.json"),
              "w") as fh:
        json.dump({"params": vars(args),
                   "spectrum": [{"palette": r["palette"],
                                 "gens": {str(k): v for k, v in
                                          r["gens"].items()}}
                                for r in spectrum],
                   "depth_entropy": depth_H,
                   "analysis": {"l1": bool(l1), "l2": bool(l2),
                                "l3": bool(l3)}}, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
