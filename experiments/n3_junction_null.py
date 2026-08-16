"""The control the junction result never had: does `d+1` need the field at all?

`n3_junction_scale` measures full-palette junction density on relaxed multiphase
fields and finds the argmax at `P = d+1` in every dimension. That result was
written up as a standalone note, made the README's hero image, and listed among
the programme's falsifiers.

An outside review supplied the control that was missing, and it is decisive.
Replace the entire apparatus — Allen-Cahn dynamics, capacity field, S-functional,
relaxation protocol — with random seed points and nearest-neighbour colouring.
A plain Voronoi diagram. Score it with the repo's own unmodified measure. The
argmax is still `d+1`, in every dimension.

This experiment reproduces that control, adds a second one that is sharper, and
states what is left.

**Why the answer is forced.** At `m = d+1` the measure asks for all `P` sectors
*and* at least `d+1` distinct ones inside one `3^d` window. So:

- every `P < d+1` is **identically zero by definition** — you cannot draw `d+1`
  distinct colours from a palette of fewer than `d+1`;
- every `P > d+1` must fit *all* `P` colours into one small window, which gets
  monotonically harder as `P` grows.

A function that is zero below `d+1` and decreasing above it has its maximum at
`d+1` before anything is measured. The relaxation cannot move it, and neither
can the theory.

**The two controls, and what each shows.**

``noise``
    Uniform random labels: no tessellation, no spatial structure whatsoever.
    Returns argmax `d+1` at densities *two orders of magnitude higher* than the
    relaxed field. This demonstrates the construction argument directly — but it
    **fails the texture guard** (`domain_diameter` ~ 1.0), so on its own it does
    not refute a properly guarded claim.

``voronoi``
    Random points, nearest-neighbour colouring. **Passes every check the repo
    applies** — the texture guard included — and still returns `d+1` with no
    dynamics of any kind. *This is the one that settles it.*

**What survives.** That the relaxed field tessellates space at all, which the
texture guard already reported and which was never in doubt. The dimension-blind
constants found along the way (`min_valence` hardcoded to the 2-D vertex number,
and the texture floor tuned to a 2-D domain) were real defects in the measure and
their fixes stand. What does not survive is `d+1` as a finding *about the field*.

    python experiments/n3_junction_null.py
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

from project_genesis.junction_scale import full_palette_density  # noqa: E402
from project_genesis.multiphase import domain_diameter           # noqa: E402
from n3_junction_scale import relaxed                            # noqa: E402

BANNER = "=" * 78


def voronoi_labels(n: int, ndim: int, n_seeds: int, n_palette: int,
                   rng) -> np.ndarray:
    """Nearest-seed labelling on a periodic ``[0,n)^ndim`` lattice.

    Faithful to the review's control. Every colour is forced to appear, so a
    zero reading is never an artefact of a colour simply being absent.
    """
    seeds = rng.uniform(0, n, size=(n_seeds, ndim))
    colours = rng.integers(0, n_palette, size=n_seeds)
    for c in range(n_palette):
        if not np.any(colours == c):
            colours[rng.integers(0, n_seeds)] = c
    grid = np.stack(np.meshgrid(*([np.arange(n)] * ndim), indexing="ij"), -1)
    flat = grid.reshape(-1, ndim).astype(float)
    best_d = np.full(flat.shape[0], np.inf)
    best_c = np.zeros(flat.shape[0], dtype=np.int64)
    for s, c in zip(seeds, colours):
        delta = np.abs(flat - s)
        delta = np.minimum(delta, n - delta)          # periodic
        d2 = (delta ** 2).sum(axis=1)
        hit = d2 < best_d
        best_d[hit] = d2[hit]
        best_c[hit] = c
    return best_c.reshape([n] * ndim)


def noise_labels(n: int, ndim: int, n_palette: int, rng) -> np.ndarray:
    return rng.integers(0, n_palette, (n,) * ndim)


def score(labels, n_palette: int, ndim: int) -> float:
    return full_palette_density(labels, n_palette, radius=1,
                                min_valence=ndim + 1)


def arm(kind: str, ndim: int, n: int, n_seeds: int, palettes, trials, rng,
        steps: int) -> dict:
    out, widths = {}, {}
    for P in palettes:
        dens, wid = [], []
        for _ in range(trials):
            if kind == "voronoi":
                lab = voronoi_labels(n, ndim, n_seeds, P, rng)
            elif kind == "noise":
                lab = noise_labels(n, ndim, P, rng)
            else:
                lab = relaxed(P, n, ndim, steps, 1.5, 0.1,
                              int(rng.integers(0, 2 ** 31)))
            dens.append(score(lab, P, ndim))
            wid.append(domain_diameter(lab))
        out[P] = float(np.mean(dens))
        widths[P] = float(np.mean(wid))
    best = max(out, key=lambda k: out[k])
    return {"density": out, "width": widths, "argmax": int(best),
            "matches_d_plus_1": bool(best == ndim + 1)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--palettes", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--guard-floor", type=float, default=5.0,
                   help="domain width below this is texture, not a tiling")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    print(BANNER)
    print("DOES `d+1` NEED THE FIELD AT ALL?")
    print(BANNER)
    print("  Scored with the repo's own unmodified full_palette_density,")
    print("  radius 1, min_valence = d+1 — the published measure, untouched.")
    print()
    print("  Registered before running: the argmax is d+1 for EVERY arm,")
    print("  including the ones with no dynamics, because the measure is zero")
    print("  below d+1 by definition and decreasing above it. If any control")
    print("  fails to select d+1, that reasoning is wrong and the field is")
    print("  doing something after all.")
    print()

    rng = np.random.default_rng(args.seed)
    geom = [(1, 4000, 180, 1500), (2, 72, 14, 600), (3, 28, 9, 1500)]
    results, verdicts = {}, []

    for ndim, n, n_seeds, steps in geom:
        print(BANNER)
        print(f"d = {ndim}   ({n}^{ndim}, m = {ndim + 1})")
        print(BANNER)
        head = f"  {'arm':<24}" + "".join(f"{'P=' + str(P):>10}"
                                          for P in args.palettes)
        print(head + f"{'argmax':>8}{'width':>8}{'guard':>7}")
        for kind, label in (("relaxed", "relaxed URP field"),
                            ("voronoi", "Voronoi (no dynamics)"),
                            ("noise", "pure noise")):
            r = arm(kind, ndim, n, n_seeds, args.palettes, args.trials, rng,
                    steps)
            results[f"{ndim}D_{kind}"] = r
            w = r["width"][r["argmax"]]
            ok = "pass" if w > args.guard_floor else "FAIL"
            print(f"  {label:<24}"
                  + "".join(f"{r['density'][P]:>10.5f}" for P in args.palettes)
                  + f"{r['argmax']:>8}{w:>8.1f}{ok:>7}")
            verdicts.append((ndim, kind, r["matches_d_plus_1"], ok))
        print()

    print(BANNER)
    print("VERDICT")
    print(BANNER)
    all_match = all(v[2] for v in verdicts)
    vor_ok = [v for v in verdicts if v[1] == "voronoi" and v[3] == "pass"]
    noise_fail = [v for v in verdicts if v[1] == "noise" and v[3] == "FAIL"]

    if all_match:
        print("  PREDICTION HELD — every arm selects d+1, in every dimension,")
        print("  with or without dynamics. The argmax is a property of the")
        print("  STATISTIC, not of any field.")
    else:
        print("  PREDICTION FAILED — some control did not select d+1, so the")
        print("  construction argument is incomplete and the field may be")
        print("  contributing something. That would be the interesting result.")
    print()
    print(f"  Voronoi arms passing the texture guard: {len(vor_ok)}/{len(geom)}")
    print(f"  noise arms failing it:                  "
          f"{len(noise_fail)}/{len(geom)}")
    print()
    print("  The noise arm scores highest of all and fails the guard, so the")
    print("  guard is doing real work — it is not the case that anything at all")
    print("  passes. But the Voronoi arm passes every check this repository")
    print("  applies, has no dynamics of any kind, and returns the same answer.")
    print("  That is the one that settles it.")
    print()
    print("  WHAT SURVIVES: that the relaxed field tessellates space, which the")
    print("  texture guard already reported. WHAT DOES NOT: `d+1` as a finding")
    print("  about the field, as a result of this programme, or as a falsifier")
    print("  of anything in it.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        o = args.output_dir / "junction_null.json"
        o.write_text(json.dumps({"results": results,
                                 "all_select_d_plus_1": all_match},
                                indent=2, default=float))
        print(f"\n  wrote {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
