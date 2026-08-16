"""Does anything about ΔC / ΔI survive coarse-graining on its own?

§5's conclusion is that the terms of `S = ΔC + κ·ΔI` have no substrate-
independent definition: every construction tried needs a coarse-graining, and
the coarse-graining is a chosen cut. That is where the programme stalls, because
a score needs someone to have already picked the level of description.

The renormalization group is the precedent for getting past exactly this. It
does not score a description, it **flows** one — and the classification that
falls out (relevant / irrelevant) is not chosen, even though the cut is. Different
blocking schemes and different microscopic models land in the same universality
class. If any analogue of that holds here, the arbitrariness of the cut stops
mattering downstream, which is the property the whole framework needs and does
not have.

So: block the field repeatedly, under three schemes that differ in kind, and ask
which observables flow the same way regardless of scheme.

**Everything is measured against nulls at matched lattice size, by design.**
Yesterday's retraction happened because a measure was never shown anything it
could fail on. Blocking shrinks the lattice, and small lattices fluctuate more,
so *any* observable drifts under iteration for reasons that have nothing to do
with the field. The only way to read a flow is to compare it against noise and
against a dynamics-free tessellation at the same level.

Pre-registered predictions:

R1. **The flow distinguishes structure from noise.** A relaxed field and pure
    random labels flow differently at matched lattice size, by more than
    ``--gap-tol`` on at least one observable. **Falsifier:** they flow the same,
    in which case blocking measures the lattice rather than the field and the
    whole approach is dead in one run.

R2. **At least one observable is scheme-independent and at least one is not.**
    That split is what makes the classification worth having. **Falsifier:**
    all observables agree across schemes (the cut never mattered, so there is
    nothing to classify) or all disagree (nothing survives arbitrariness, and
    the RG analogy fails here).

R3. **The relaxed URP field flows the same as a plain Voronoi.** *This predicts
    the deflationary outcome deliberately.* After `n3_junction_null`, the base
    case is that the dynamics contribute nothing beyond producing some
    tessellation. **Falsifier:** they flow measurably differently — which would
    be the first evidence in this programme that the dynamics do something a
    generic tessellation does not.

R4. **The classification is the same in every dimension and at every palette
    size.** This is the one that decides whether any of it matters. A
    classification that changes with `d` or `P` has not removed the
    arbitrariness — it has moved it one level up, from "which cut" to "which
    setup", and bought nothing. **Falsifier:** an observable that survives the
    cut in one cell of the sweep and not in another.

Honest scope. This tests whether a *classification* exists and is invariant,
not whether it is the one the theory wants. Three observables, three schemes,
one block factor. Invariance across `d` and `P` is necessary for the RG analogy
to be doing any work; it is nowhere near sufficient to say the surviving
quantity is meaningful.

    python experiments/n3_rg_flow.py                    # 2-D only, quick
    python experiments/n3_rg_flow.py --ndims 2 3 --palettes 2 3 4 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from project_genesis.coarse_grain import SCHEMES, rg_flow      # noqa: E402
from project_genesis.multiphase import domain_diameter        # noqa: E402
from n3_junction_null import noise_labels, voronoi_labels      # noqa: E402
from n3_junction_scale import relaxed                          # noqa: E402

BANNER = "=" * 78
OBS = ("wall_density", "order_parameter", "largest_component")


def matched_voronoi(size, ndim, n_palette, target_width, rng, tries=4):
    """A Voronoi whose domains are the same width as the field it controls for.

    Without this the control is worthless and worse than nothing. Wall density
    is a direct function of domain width, so comparing a relaxed field against a
    Voronoi with different-sized cells reads a size mismatch as a difference in
    physics. Measured: R3 "failed" in exactly the five cells where the widths
    were mismatched by 1.7-2.4x and held in the three where they matched to
    within 8%. A fixed seed count cannot match, because the relaxed field's
    coarsening depends on both dimension and palette.

    Seeds are estimated from ``N = (S/L)^d`` and then corrected against the
    measured width, because ``domain_diameter`` is not exactly that geometric
    L. The achieved ratio is returned so the match is reported rather than
    assumed.
    """
    n = max(2, int(round((size / max(target_width, 1e-9)) ** ndim)))
    best = None
    for _ in range(tries):
        lab = voronoi_labels(size, ndim, n, n_palette, rng)
        w = domain_diameter(lab)
        ratio = w / target_width if target_width > 0 else float("inf")
        if best is None or abs(np.log(ratio)) < abs(np.log(best[1])):
            best = (lab, ratio, n)
        if 0.92 <= ratio <= 1.08:
            break
        # width scales like N^(-1/d): to change width by 1/ratio, scale N by
        # ratio^d
        n = max(2, int(round(n * ratio ** ndim)))
    return best


def trajectories(labels, n_palette, schemes, levels, b, rng) -> dict:
    return {s: rg_flow(labels, n_palette, s, levels=levels, b=b, rng=rng)
            for s in schemes}


def series(traj: list, key: str) -> list:
    return [lv[key] for lv in traj]


def gap(a: list, b: list) -> float:
    """Largest level-by-level difference between two trajectories."""
    n = min(len(a), len(b))
    return float(max(abs(a[i] - b[i]) for i in range(n))) if n else float("nan")


def flow_range(trajs: dict, key: str) -> float:
    """How far the observable moves across the flow, averaged over schemes.

    The scale against which a disagreement has to be judged. An observable that
    barely moves cannot have a meaningful spread: the schemes agree because
    there is nothing to disagree about.
    """
    rs = []
    for t in trajs.values():
        ser = series(t, key)
        if ser:
            rs.append(max(ser) - min(ser))
    return float(np.mean(rs)) if rs else float("nan")


def relative_spread(trajs: dict, key: str) -> float:
    """Scheme disagreement as a fraction of the observable's own motion.

    A *fixed* threshold on the raw spread is not usable here: measured spreads
    fall monotonically with palette size (0.30 -> 0.09 for the order parameter
    across P = 2..5), so a fixed cut classifies by P rather than by the field —
    the same defect as the dimension-blind texture floor.

    Dividing by :func:`flow_range` asks the scale-free question instead: does
    the observable's *flow* survive the choice of cut, relative to how much
    flowing it does? Returns ``inf`` when nothing moves, which correctly reads
    as "cannot tell" rather than as agreement.
    """
    rng_ = flow_range(trajs, key)
    if not np.isfinite(rng_) or rng_ <= 1e-12:
        return float("inf")
    return float(scheme_spread(trajs, key) / rng_)


def scheme_spread(trajs: dict, key: str) -> float:
    """Largest disagreement between schemes, level by level."""
    ser = [series(t, key) for t in trajs.values()]
    n = min(len(s) for s in ser)
    return float(max(max(s[i] for s in ser) - min(s[i] for s in ser)
                     for i in range(n))) if n else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ndims", type=int, nargs="+", default=[2])
    p.add_argument("--palettes", type=int, nargs="+", default=[3])
    p.add_argument("--size-2d", type=int, default=128)
    p.add_argument("--size-3d", type=int, default=64)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--block", type=int, default=2)
    p.add_argument("--seeds-2d", type=int, default=64)
    p.add_argument("--seeds-3d", type=int, default=125,
                   help="Voronoi seed points; chosen so domains-per-axis is "
                        "broadly comparable to the 2-D arm rather than equal, "
                        "which is why domain width is reported per cell")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--gap-tol", type=float, default=0.10,
                   help="trajectories differing by more than this, on a [0,1] "
                        "fraction, count as different")
    p.add_argument("--rel-tol", type=float, default=0.35,
                   help="scheme disagreement as a fraction of the observable's "
                        "own motion, above which its flow is a fact about the "
                        "cut rather than the field")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    print(BANNER)
    print("DOES ANYTHING SURVIVE COARSE-GRAINING ON ITS OWN?")
    print(BANNER)
    print(f"  dimensions {args.ndims}, palettes {args.palettes}, "
          f"block {args.block}, {args.trials} trials")
    print("  three schemes that differ in kind: majority smooths, decimate")
    print("  samples a fixed corner, random samples uniformly")
    print()
    print("  Pre-registered:")
    print(f"    R1  structure and noise flow differently (> {args.gap_tol:g})")
    print("    R2  some observable is scheme-independent, some is not")
    print("    R3  the relaxed field flows the SAME as a Voronoi")
    print("    R4  and the classification is identical in every cell of the")
    print("        sweep -- otherwise the arbitrariness just moved up a level")
    print()

    rng = np.random.default_rng(args.seed)
    cells, r1_ok, r3_ok = {}, [], []

    for ndim in args.ndims:
        size = args.size_2d if ndim == 2 else args.size_3d
        seeds = args.seeds_2d if ndim == 2 else args.seeds_3d
        steps = args.steps if ndim == 2 else max(args.steps, 800)
        for P in args.palettes:
            fields = {"relaxed": [], "voronoi": [], "noise": []}
            ratios = []
            for t in range(args.trials):
                rel = relaxed(P, size, ndim, steps, 1.5, 0.1,
                              args.seed + 31 * t)
                fields["relaxed"].append(rel)
                # the control must have the SAME domain width as the thing it
                # controls for, or a size mismatch reads as physics
                vor, ratio, _n = matched_voronoi(size, ndim, P,
                                                 domain_diameter(rel), rng)
                fields["voronoi"].append(vor)
                ratios.append(ratio)
                fields["noise"].append(noise_labels(size, ndim, P, rng))

            avg = {}
            for kind, labs in fields.items():
                per = [trajectories(l, P, SCHEMES, args.levels, args.block, rng)
                       for l in labs]
                avg[kind] = {}
                for sc in SCHEMES:
                    n = min(len(pt[sc]) for pt in per)
                    avg[kind][sc] = [
                        {k: float(np.mean([pt[sc][i][k] for pt in per]))
                         for k in OBS} for i in range(n)]

            spreads = {k: relative_spread(avg["relaxed"], k) for k in OBS}
            raw = {k: scheme_spread(avg["relaxed"], k) for k in OBS}
            moves = {k: flow_range(avg["relaxed"], k) for k in OBS}
            r1g = {k: max(gap(series(avg["relaxed"][sc], k),
                              series(avg["noise"][sc], k)) for sc in SCHEMES)
                   for k in OBS}
            r3g = {k: max(gap(series(avg["relaxed"][sc], k),
                              series(avg["voronoi"][sc], k)) for sc in SCHEMES)
                   for k in OBS}
            # an effect counts only if it clears the instrument's own
            # disagreement with itself on that same observable
            resolved = {k: bool(r3g[k] > max(args.gap_tol, spreads[k]))
                        for k in OBS}
            survives = {k: bool(spreads[k] <= args.rel_tol) for k in OBS}
            width = float(np.mean([domain_diameter(l)
                                   for l in fields["relaxed"]]))
            vwidth = float(np.mean([domain_diameter(l)
                                    for l in fields["voronoi"]]))

            key = f"{ndim}D_P{P}"
            cells[key] = {"width_match_ratio": float(np.mean(ratios)),
                          "spreads": spreads, "raw_spread": raw,
                          "flow_range": moves, "survives": survives,
                          "r1_gaps": r1g, "r3_gaps": r3g,
                          "r3_resolved": resolved,
                          "relaxed_width": width, "voronoi_width": vwidth,
                          "levels": len(avg["relaxed"]["majority"])}
            r1_ok.append(any(g > args.gap_tol for g in r1g.values()))
            r3_ok.append(not any(resolved.values()))

            surv = [k for k in OBS if survives[k]]
            print(f"  {key:<8} width {width:5.1f} (voronoi {vwidth:5.1f}, "
                  f"match {np.mean(ratios):.2f}x)  "
                  f"levels {cells[key]['levels']}  "
                  f"survives: {', '.join(surv) if surv else 'NOTHING'}")

    print()
    print(BANNER)
    print("THE CLASSIFICATION, CELL BY CELL")
    print(BANNER)
    print(f"  {'cell':<10}" + "".join(f"{k:>20}" for k in OBS))
    for key, c in cells.items():
        row = "".join(
            f"{c['spreads'][k]:>13.2f} {'OK ' if c['survives'][k] else 'cut'}"
            for k in OBS)
        print(f"  {key:<10}{row}")
    print()
    print("  Numbers are scheme disagreement / the observable's own motion.")
    print("  OK = the flow survives the choice of cut. cut = the disagreement")
    print("  is comparable to the flow, so the flow is a fact about the scheme.")

    # ------------------------------------------------------------------ R1-R3
    print()
    print(BANNER)
    print("R1  DOES THE FLOW SEE THE FIELD?      "
          f"{sum(r1_ok)}/{len(r1_ok)} cells")
    print("R3  DYNAMICS INDISTINGUISHABLE FROM VORONOI?  "
          f"{sum(r3_ok)}/{len(r3_ok)} cells")
    print(BANNER)
    r1 = all(r1_ok)
    r3 = all(r3_ok)
    print("  R1 " + ("held everywhere." if r1 else
                     "FAILED in some cell -- blocking read the lattice there."))
    print("  R3 " + ("held everywhere, deflationary as predicted."
                     if r3 else
                     "FAILED somewhere -- worth a hard look before believing."))

    # --------------------------------------------------------------------- R2
    print()
    print(BANNER)
    print("R2  IS THERE A SPLIT AT ALL?")
    print(BANNER)
    r2 = all(any(c["survives"].values()) and not all(c["survives"].values())
             for c in cells.values())
    print("  PREDICTION HELD in every cell -- each has survivors and casualties."
          if r2 else
          "  PREDICTION FAILED in at least one cell (all survive, or none do).")

    # --------------------------------------------------------------------- R4
    print()
    print(BANNER)
    print("R4  IS THE CLASSIFICATION THE SAME EVERYWHERE?")
    print(BANNER)
    sigs = {key: tuple(c["survives"][k] for k in OBS)
            for key, c in cells.items()}
    uniq = set(sigs.values())
    r4 = len(uniq) == 1
    for key, sig in sigs.items():
        print(f"  {key:<10} " + ", ".join(
            f"{k}={'OK' if v else 'cut'}" for k, v in zip(OBS, sig)))
    print()
    if r4 and len(cells) > 1:
        surv = [k for k, v in zip(OBS, next(iter(uniq))) if v]
        print("  PREDICTION HELD — one classification across every dimension")
        print(f"  and palette tested. Survivors: {', '.join(surv) or 'none'}.")
        print("  That is the property the RG analogy needs: the cut is a")
        print("  choice, the classification is not, and neither is it a")
        print("  function of the setup.")
    elif len(cells) == 1:
        print("  ONLY ONE CELL — nothing to compare. Run with --ndims 2 3 and")
        print("  several palettes for this to mean anything.")
        r4 = False
    else:
        print("  PREDICTION FAILED — the classification changes across the")
        print(f"  sweep ({len(uniq)} distinct signatures). Then it is a fact")
        print("  about the setup, not about the field, and the arbitrariness")
        print("  has only moved from 'which cut' to 'which dimension and")
        print("  palette'. That is a real negative and it costs the approach")
        print("  its main claim.")

    print()
    print(BANNER)
    n = int(r1) + int(r2) + int(r3) + int(r4)
    print(f"VERDICT: {n}/4 pre-registered predictions held.")
    print(BANNER)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        o = args.output_dir / "rg_flow.json"
        o.write_text(json.dumps(
            {"params": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
             "cells": cells, "score": n}, indent=2, default=float))
        print(f"\n  wrote {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
