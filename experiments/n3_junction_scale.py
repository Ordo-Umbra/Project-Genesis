"""At what scale is three special — and does the ranking outlive the margin?

`The_Principle.md` §2 and §3 rest on one measurement: the full-palette junction
density is non-zero essentially only at `P = 3`, and stays so across a 10x
sweep in capacity. Matter as a tessellation, generations as `d+1`, the whole
selection argument — all downstream. It is also the **least audited** claim in
the document, while four preceding audits each found a confident reading
resting on an underived constant.

The constant here is different, and better. The measure has **no numeric
threshold**: it asks whether a neighbourhood carries every colour
(`bits == full`) and whether it is a junction rather than a wall
(`distinct >= 3`), both exact, with the `3` being the definition of a vertex.
The single free choice is the **neighbourhood**, hardcoded as the immediate
`3^d` ring — and unlike `structured_threshold = 0.3`, that has an argument
behind it: on a lattice it is the smallest window in which a junction can
appear at all, so it is the natural vertex-scale probe.

This experiment tests the argument instead of asserting it, and applies the
one structural lesson the earlier audits produced: **locations move under a
convention change, ordinal comparisons do not**, because the compared
quantities share whatever the convention distorts. So the margin and the
ranking are scored separately.

Pre-registered predictions:

Q1. **The selection is vertex-scale, and widening the probe degrades it.**
    At `radius = 1` the `P = 3` density exceeds every rival by more than `10x`.
    At `radius >= 2` — a 5x5 window, which can hold four or more colours even
    where every vertex is strictly three-fold — that margin falls below `3x`.
    **Falsifier:** the margin survives widening, which would mean the selection
    is not about vertex geometry at all and is a *stronger* claim than the
    Plateau argument §3 now rests it on.

Q2. **But the ranking survives what the margin does not.**  `argmax_P` stays at
    `P = 3` at every radius tested in 2-D, even where the margin has collapsed.
    **Falsifier:** the winner changes with the radius — in which case "three is
    special" is a statement about the probe and §2's cliff does not survive its
    own convention, which would be the most consequential finding in this repo.

Q3. **In 3-D the winner moves to `P = 4`.**  §3 claims families `= d + 1`, and
    `multiphase.full_palette_junction_density`'s own docstring flags that 3-D
    junction *lines* have a different valence so "the clean selection need not
    transfer" — but nothing here has measured it. Prediction: at `radius = 1`
    in 3-D, `argmax_P = 4`.  **Falsifier:** it stays at 3, or nothing is
    selected — either would mean `d + 1` is asserted rather than measured, and
    §3 should say so.

Honest scope. This sweeps the neighbourhood only; the field configurations are
the published relaxed multiphase states at published settings, so the capacity
axis is not re-opened (§2 already swept it). Radii above 2 are reported for
shape but are not defensible probes of a vertex — a 7x7 window on a lattice
this size is measuring a region — and the registered claims use `radius <= 2`.
Nothing here re-derives Plateau's law; it asks whether the measurement that is
said to express it is a measurement of the field or of the window.

    python experiments/n3_junction_scale.py
    python experiments/n3_junction_scale.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.junction_scale import (  # noqa: E402
    full_palette_density, selection_profile, valence_profile,
)
from project_genesis.multiphase import (  # noqa: E402
    domain_scale, step_multiphase_conserved,
)

BANNER = "=" * 78


def relaxed(n_palette, size, ndim, steps, gamma, dt, seed, diffusion=1.0):
    """One relaxed field, mirroring `n3_capacity_separation._relax` exactly.

    The published cliff is measured on **volume-conserving** dynamics — plain
    Allen-Cahn coarsens without bound and every junction is transient, so the
    measure would have nothing stable to read. Seeding and stepping match that
    experiment line for line; only the spatial dimension is parameterised, and
    the 2-D case is pinned against `_relax` in the tests.
    """
    rng = np.random.default_rng(seed)
    f = rng.random((n_palette, *([size] * ndim))) * 0.1
    for _ in range(steps):
        f, _k = step_multiphase_conserved(f, None, diffusion=diffusion,
                                          gamma=gamma, dt=dt)
    return np.argmax(f, axis=0)


def sweep(palettes, radii, *, size, ndim, steps, gamma, dt, seed, trials,
          diffusion=1.0):
    """Density at every (palette, radius), averaged over trials."""
    out = {}
    for P in palettes:
        labels = [relaxed(P, size, ndim, steps, gamma, dt,
                          seed + 100 * P + t, diffusion)
                  for t in range(trials)]
        for r in radii:
            out[(P, r)] = float(np.mean(
                [full_palette_density(lab, P, r) for lab in labels]))
        out[("valence", P)] = {r: valence_profile(labels[0], r) for r in radii}
        # the texture guard: a junction count on a field that never tessellated
        # is counting lattice noise. This caught an artefact once already.
        out[("scale", P)] = float(np.mean([domain_scale(l) for l in labels]))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--palettes", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--radii", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--size", type=int, default=72)
    p.add_argument("--size-3d", type=int, default=28)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--steps-3d", type=int, default=1500,
                   help="3-D needs longer to coarsen: at the published 600 the "
                        "texture guard reports domain scale 2.45 (P=4) and "
                        "2.34 (P=5), below the 2.5 tiling floor. 1500 gives "
                        "3.17/2.99/2.72. Measured, not tuned for an answer.")
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--diffusion", type=float, default=1.0)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--tight-margin", type=float, default=10.0,
                   help="Q1: margin the published radius must exceed.")
    p.add_argument("--loose-margin", type=float, default=3.0,
                   help="Q1: margin a widened probe must fall below.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size, args.size_3d, args.trials = 40, 24, 2
        args.steps, args.steps_3d = 250, 900

    print(BANNER)
    print("THE JUNCTION SCALE — is 'three is special' about the field or the window?")
    print(BANNER)
    print(f"  2-D {args.size}^2 x {args.steps} steps; "
          f"3-D {args.size_3d}^3 x {args.steps_3d} steps; "
          f"{args.trials} trials, seed {args.seed}")
    print("  (3-D runs longer because it coarsens slower — at the published 600")
    print("   steps the texture guard rejects P=4 and P=5 as untiled)")
    print("  radius 1 is the published probe and reproduces the published")
    print("  measure exactly (pinned in tests/test_junction_scale.py)")
    print()
    print("  Pre-registered:")
    print(f"    Q1  margin > {args.tight_margin:g}x at radius 1, "
          f"< {args.loose_margin:g}x at radius >= 2")
    print("    Q2  but argmax stays at P=3 at every radius (2-D)")
    print("    Q3  and in 3-D at radius 1 the winner is P=4")
    print()

    t0 = time.time()
    print("  relaxing 2-D fields...", flush=True)
    two = sweep(args.palettes, args.radii, size=args.size, ndim=2,
                steps=args.steps, gamma=args.gamma, dt=args.dt,
                seed=args.seed, trials=args.trials, diffusion=args.diffusion)
    print("  relaxing 3-D fields...", flush=True)
    three = sweep(args.palettes, args.radii, size=args.size_3d, ndim=3,
                  steps=args.steps_3d, gamma=args.gamma, dt=args.dt,
                  seed=args.seed, trials=args.trials, diffusion=args.diffusion)
    print(f"  [{time.time() - t0:.0f}s]")

    results = {}
    for name, data, ndim in (("2-D", two, 2), ("3-D", three, 3)):
        print()
        print(BANNER)
        print(f"{name} — full-palette junction density vs neighbourhood radius")
        print(BANNER)
        head = "  " + f"{'radius':>7}" + "".join(f"{'P=' + str(P):>11}"
                                                 for P in args.palettes)
        print(head + f"{'argmax':>8}{'margin':>10}")
        per_radius = {}
        for r in args.radii:
            dens = {P: data[(P, r)] for P in args.palettes}
            prof = selection_profile(dens)
            per_radius[r] = prof
            row = "  " + f"{r:>7}" + "".join(f"{dens[P]:>11.5f}"
                                             for P in args.palettes)
            m = prof["margin"]
            mtxt = "inf" if np.isinf(m) else f"{m:.2f}"
            print(row + f"{prof['argmax']!s:>8}{mtxt:>10}")
        print()
        print(f"  {'palette':>8} {'domain scale':>13}"
              "   (texture guard: ~1 means no tessellation, only noise)")
        for P in args.palettes:
            sc = data[("scale", P)]
            flag = "  <-- TEXTURE, not a tiling" if sc < 2.5 else ""
            print(f"  {P:>8} {sc:>13.2f}{flag}")
        resolved = all(data[("scale", P)] >= 2.5 for P in args.palettes)
        print(f"  every palette resolved into domains: "
              f"{'yes' if resolved else 'NO — readings below are suspect'}")
        print()
        print(f"  {'radius':>7} {'mean valence':>13} {'junction frac':>14}"
              "   (why a wider window changes the answer)")
        for r in args.radii:
            v = data[("valence", 3)][r]
            print(f"  {r:>7} {v['mean']:>13.2f} {v['junction_fraction']:>13.0%}")
        results[name] = {str(r): per_radius[r] for r in args.radii}
        results[name + "_scale"] = {f"P{P}": data[("scale", P)]
                                    for P in args.palettes}
        results[name + "_resolved"] = bool(resolved)
        results[name + "_density"] = {f"P{P}_r{r}": data[(P, r)]
                                      for P in args.palettes for r in args.radii}

    # ------------------------------------------------------------------- Q1
    tight = results["2-D"][str(args.radii[0])]
    wide = [results["2-D"][str(r)] for r in args.radii if r >= 2]
    print()
    print(BANNER)
    print("Q1  IS THE SELECTION VERTEX-SCALE?")
    print(BANNER)
    mt = tight["margin"]
    print(f"  radius 1 margin: {'inf' if np.isinf(mt) else f'{mt:.2f}'}x "
          f"[need > {args.tight_margin:g}]")
    for r, w in zip([r for r in args.radii if r >= 2], wide):
        mw = w["margin"]
        print(f"  radius {r} margin: {'inf' if np.isinf(mw) else f'{mw:.2f}'}x "
              f"[need < {args.loose_margin:g}]")
    q1 = bool(mt > args.tight_margin
              and all(w["margin"] < args.loose_margin for w in wide))
    print()
    if q1:
        print("  PREDICTION HELD — the selection lives at the vertex scale and")
        print("  dissolves when the probe stops being a vertex. So the honest")
        print("  statement is 'P=3 is selected relative to a vertex-scale")
        print("  probe', which is exactly the Plateau argument §3 rests on —")
        print("  and it is a narrower claim than an unqualified 'three is")
        print("  special'.")
    else:
        print("  PREDICTION FAILED — reported as measured. If the margin holds")
        print("  at a widened probe the selection is not vertex geometry, and")
        print("  the mechanism §3 gives for it is not the mechanism at work.")

    # ------------------------------------------------------------------- Q2
    print()
    print(BANNER)
    print("Q2  DOES THE RANKING OUTLIVE THE MARGIN?")
    print(BANNER)
    winners = {r: results["2-D"][str(r)]["argmax"] for r in args.radii}
    for r in args.radii:
        print(f"  radius {r}: argmax = P={winners[r]}")
    q2 = bool(all(w == 3 for w in winners.values()))
    print()
    if q2:
        print("  PREDICTION HELD — and it is the structural point. The margin")
        print("  is a magnitude and it collapses; the ranking is a comparison")
        print("  and it does not. Both readings share whatever the window")
        print("  distorts, so the distortion cancels in the comparison. That is")
        print("  the same split every audit in this arc has found, now on the")
        print("  claim the whole document rests on.")
    else:
        print("  PREDICTION FAILED — the winner moves with the probe:")
        print(f"    {winners}")
        print("  Then §2's cliff does not survive its own convention. That is")
        print("  the most consequential negative available in this repo and it")
        print("  should be recorded at the top of the document, not here.")

    # ------------------------------------------------------------------- Q3
    print()
    print(BANNER)
    print("Q3  DOES 3-D SELECT P=4, AS 'FAMILIES = d+1' REQUIRES?")
    print(BANNER)
    w3 = results["3-D"][str(args.radii[0])]
    print(f"  3-D radius 1: argmax = P={w3['argmax']}, "
          f"best density {w3['best_density']:.5f}, "
          f"runner-up {w3['runner_up']:.5f}")
    q3 = bool(w3["argmax"] == 4 and w3["selects"])
    print()
    if q3:
        print("  PREDICTION HELD — d+1 is measured, not asserted. The same")
        print("  vertex-scale probe that picks 3 in the plane picks 4 in space,")
        print("  which is what §3 claims and what nothing here had checked.")
    elif not w3["selects"]:
        print("  PREDICTION FAILED — nothing is selected in 3-D at all. The")
        print("  measure that carries §2 in the plane has no content in space,")
        print("  so 'generations = d+1' is a 2-D result extrapolated, and §3")
        print("  should say that.")
    else:
        print(f"  PREDICTION FAILED — 3-D selects P={w3['argmax']}, not 4.")
        print("  §3's 'families = d+1' does not follow from this measure, and")
        print("  the docstring's own warning that the selection 'need not")
        print("  transfer' to 3-D turns out to be the operative fact.")

    print()
    print(BANNER)
    n = int(q1) + int(q2) + int(q3)
    print(f"VERDICT: {n}/3 pre-registered predictions held.")
    print(BANNER)
    print("  What this decides: whether the foundation is a fact about the")
    print("  field or about the window. On the evidence above the load-bearing")
    print("  part is " + ("the RANKING, which is convention-independent, while\n"
                          "  the margin is not — so the base to build on is "
                          "ordinal."
                          if q2 else
                          "NOT established — the winner itself depends on the\n"
                          "  probe, and nothing downstream of §2 is safe."))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        o = args.output_dir / "junction_scale.json"
        o.write_text(json.dumps(
            {"params": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
             "results": results, "score": n}, indent=2, default=float))
        print(f"\n  wrote {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
