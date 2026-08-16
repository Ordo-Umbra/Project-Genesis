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

Honest scope. This tests whether a *classification* exists, not whether it is
the one the theory wants. Three observables, three schemes, one block factor,
2-D. A negative here is cheap and informative; a positive would need the whole
apparatus repeated in 3-D and across palettes before it meant anything.

    python experiments/n3_rg_flow.py
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
from n3_junction_null import noise_labels, voronoi_labels      # noqa: E402
from n3_junction_scale import relaxed                          # noqa: E402

BANNER = "=" * 78
OBS = ("wall_density", "order_parameter", "largest_component")


def trajectories(labels, n_palette, schemes, levels, b, rng) -> dict:
    return {s: rg_flow(labels, n_palette, s, levels=levels, b=b, rng=rng)
            for s in schemes}


def series(traj: list, key: str) -> list:
    return [lv[key] for lv in traj]


def gap(a: list, b: list) -> float:
    """Largest level-by-level difference between two trajectories."""
    n = min(len(a), len(b))
    return float(max(abs(a[i] - b[i]) for i in range(n))) if n else float("nan")


def scheme_spread(trajs: dict, key: str) -> float:
    """Largest disagreement between schemes, level by level."""
    ser = [series(t, key) for t in trajs.values()]
    n = min(len(s) for s in ser)
    return float(max(max(s[i] for s in ser) - min(s[i] for s in ser)
                     for i in range(n))) if n else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--palette", type=int, default=3)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--block", type=int, default=2)
    p.add_argument("--seeds", type=int, default=64,
                   help="Voronoi seed points (matched to the relaxed field's "
                        "domain count as closely as one can)")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--gap-tol", type=float, default=0.10,
                   help="trajectories differing by more than this, on a [0,1] "
                        "fraction, count as different")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    print(BANNER)
    print("DOES ANYTHING SURVIVE COARSE-GRAINING ON ITS OWN?")
    print(BANNER)
    print(f"  {args.size}^2, P={args.palette}, block {args.block}, "
          f"{args.levels} levels, {args.trials} trials")
    print("  three schemes that differ in kind: majority smooths, decimate")
    print("  samples a fixed corner, random samples uniformly")
    print()
    print("  Pre-registered:")
    print(f"    R1  structure and noise flow differently (> {args.gap_tol:g})")
    print("    R2  some observable is scheme-independent, some is not")
    print("    R3  the relaxed field flows the SAME as a Voronoi")
    print("        (predicting the deflationary outcome on purpose)")
    print()

    rng = np.random.default_rng(args.seed)
    fields = {}
    for t in range(args.trials):
        fields.setdefault("relaxed", []).append(
            relaxed(args.palette, args.size, 2, args.steps, 1.5, 0.1,
                    args.seed + 31 * t))
        fields.setdefault("voronoi", []).append(
            voronoi_labels(args.size, 2, args.seeds, args.palette, rng))
        fields.setdefault("noise", []).append(
            noise_labels(args.size, 2, args.palette, rng))

    # average each trajectory over trials
    avg = {}
    for kind, labs in fields.items():
        per = [trajectories(l, args.palette, SCHEMES, args.levels, args.block,
                            rng) for l in labs]
        avg[kind] = {}
        for s in SCHEMES:
            n = min(len(pt[s]) for pt in per)
            avg[kind][s] = [
                {k: float(np.mean([pt[s][i][k] for pt in per])) for k in OBS}
                for i in range(n)]

    for kind in ("relaxed", "voronoi", "noise"):
        print(BANNER)
        print(f"{kind}")
        print(BANNER)
        for s in SCHEMES:
            print(f"  {s}")
            for key in OBS:
                vals = series(avg[kind][s], key)
                print(f"    {key:<18}" + "  ".join(f"{v:.4f}" for v in vals))
        print()

    # ------------------------------------------------------------------- R1
    print(BANNER)
    print("R1  DOES THE FLOW SEE THE FIELD, OR JUST THE LATTICE?")
    print(BANNER)
    r1_gaps = {}
    for key in OBS:
        g = max(gap(series(avg["relaxed"][s], key),
                    series(avg["noise"][s], key)) for s in SCHEMES)
        r1_gaps[key] = g
        print(f"  {key:<18} relaxed vs noise, worst-scheme gap: {g:.4f}")
    r1 = any(g > args.gap_tol for g in r1_gaps.values())
    print()
    print("  PREDICTION HELD — blocking distinguishes a structured field from"
          if r1 else
          "  PREDICTION FAILED — structure and noise flow the same, so")
    print("  noise, so the flow is reading the field." if r1 else
          "  blocking measures the lattice and this approach is dead.")

    # ------------------------------------------------------------------- R2
    print()
    print(BANNER)
    print("R2  IS ANYTHING INDEPENDENT OF WHICH CUT YOU CHOSE?")
    print(BANNER)
    spreads = {}
    for key in OBS:
        sp = scheme_spread(avg["relaxed"], key)
        spreads[key] = sp
        verdict = ("scheme-INDEPENDENT" if sp <= args.gap_tol
                   else "depends on the cut")
        print(f"  {key:<18} spread across schemes: {sp:.4f}   {verdict}")
    indep = [k for k, v in spreads.items() if v <= args.gap_tol]
    dep = [k for k, v in spreads.items() if v > args.gap_tol]
    r2 = bool(indep) and bool(dep)
    print()
    if r2:
        print(f"  PREDICTION HELD — {len(indep)} observable(s) survive the")
        print(f"  choice of cut and {len(dep)} do not. That split is the whole")
        print("  point: it means 'which cut' has an answer that is not itself")
        print("  a choice.")
    elif not dep:
        print("  PREDICTION FAILED — everything agrees across schemes. The cut")
        print("  never mattered here, so there is nothing to classify and no")
        print("  arbitrariness was resolved.")
    else:
        print("  PREDICTION FAILED — nothing survives the choice of cut. The")
        print("  RG analogy does not hold for these observables.")

    # ------------------------------------------------------------------- R3
    print()
    print(BANNER)
    print("R3  DO THE DYNAMICS DO ANYTHING A VORONOI DOES NOT?")
    print(BANNER)
    # An effect only counts if it exceeds the instrument's own disagreement
    # with itself. `scheme_spread` IS that disagreement, measured on the same
    # observable: if three defensible cuts differ by more than the
    # relaxed-vs-Voronoi gap, the gap is inside the noise and no comparison has
    # been made. Without this the first run of this experiment reported "the
    # dynamics do something" on a margin of 0.008, on the least reliable of the
    # three observables -- which is the `d+1` mistake wearing a new hat.
    r3_gaps, resolved = {}, {}
    print(f"  {'observable':<18}{'gap':>9}{'scheme spread':>16}"
          f"{'resolvable?':>14}")
    for key in OBS:
        g = max(gap(series(avg["relaxed"][s], key),
                    series(avg["voronoi"][s], key)) for s in SCHEMES)
        floor = max(args.gap_tol, spreads[key])
        r3_gaps[key] = g
        resolved[key] = bool(g > floor)
        print(f"  {key:<18}{g:>9.4f}{spreads[key]:>16.4f}"
              f"{('yes' if resolved[key] else 'no'):>14}")
    same = not any(resolved.values())
    print()
    if same:
        print("  PREDICTION HELD — and it is the deflationary one. The relaxed")
        print("  field flows exactly as a dynamics-free tessellation does, so")
        print("  the Allen-Cahn dynamics contribute nothing the blocking can")
        print("  see. Consistent with n3_junction_null: what this repository's")
        print("  physics produces is *a tessellation*, and the rest is geometry.")
    else:
        hits = [k for k, v in resolved.items() if v]
        print("  PREDICTION FAILED — the relaxed field flows differently from a")
        print(f"  Voronoi on: {', '.join(hits)}, by more than the schemes")
        print("  disagree among themselves. That would be the first evidence in")
        print("  this programme that the dynamics do something generic")
        print("  tessellation does not, and it deserves a much harder look")
        print("  before anyone believes it.")

    print()
    print(BANNER)
    n = int(r1) + int(r2) + int(r3 := same)
    print(f"VERDICT: {n}/3 pre-registered predictions held.")
    print(BANNER)
    if r1 and r2:
        print("  The useful reading: a coarse-graining flow CAN be defined here,")
        print("  and it does sort observables into ones that survive the choice")
        print("  of cut and ones that do not. That is the structural property")
        print("  §5 said was missing — the cut stays arbitrary, the")
        print("  classification does not. What it does NOT yet show is that the")
        print("  surviving quantities are the ones the theory wants.")
    else:
        print("  The approach does not clear its own first hurdle, which is")
        print("  worth knowing cheaply rather than after building on it.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        o = args.output_dir / "rg_flow.json"
        o.write_text(json.dumps(
            {"params": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
             "trajectories": avg, "r1_gaps": r1_gaps,
             "scheme_spreads": spreads, "r3_gaps": r3_gaps,
             "score": n}, indent=2, default=float))
        print(f"\n  wrote {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
