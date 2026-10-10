"""Which part of the d+1 junction result is geometry, and which is dynamics?

`Docs/Junction_Selection.md` reports that the full-palette junction density at
`m = d + 1` is maximised at `P = d + 1` in d = 1..4. Its own scope section
concedes that the rivals below `d + 1` are zero *by counting*. This experiment
asks the matching question about the rivals **above** `d + 1`, and about the
dynamics, with three checks that need no relaxation run to be large:

N1. **Voronoi null.** Replace the relaxed field with a random Voronoi
    tessellation, colours assigned uniformly at random — no dynamics, no energy,
    no conservation. Prediction: argmax is still `d + 1`.
    Reading if it holds: the selection is a property of *generic* tessellations,
    so "geometric, not energetic" is confirmed — and the relaxation dynamics are
    shown to contribute nothing the null lacks.

N2. **Any point junction, regardless of palette.** Drop the full-palette
    condition and count cells whose neighbourhood holds `>= d + 1` sectors.
    Prediction: non-zero for every `P >= d + 1`, rising or flat with `P`.
    Reading: point junctions are not rare for big palettes; what is rare is a
    point junction that carries the *whole* palette. The claim is about
    palette-completeness at a vertex, not about vertices existing.

N3. **Degenerate start.** Seed a 2-D field as a square 4-colour block tiling,
    whose every vertex is four-fold (a non-generic junction), then relax it with
    the published dynamics. Prediction: the `P = 4`, `m = 3` full-palette
    density collapses toward zero as four-fold vertices split into triple ones.
    Reading if it holds: this is the one non-trivial thing the dynamics supply —
    they **remove** the degenerate junctions that would otherwise let `P > d+1`
    score. Falsifier: four-fold vertices persist, in which case the relaxed
    measurement's zeros above `d + 1` are an accident of random seeding.

    python experiments/n3_junction_null.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.junction_scale import (  # noqa: E402
    full_palette_density, neighbourhood_label_bits, selection_profile,
)
from project_genesis.multiphase import _popcount, step_multiphase_conserved  # noqa: E402


def voronoi_labels(size, ndim, n_seeds, n_palette, rng, chunk=200_000):
    """Periodic lattice Voronoi tessellation with random colours per cell."""
    seeds = rng.random((n_seeds, ndim)) * size
    colours = rng.integers(0, n_palette, n_seeds)
    # make sure every colour is present so the full palette is attainable
    colours[:n_palette] = np.arange(n_palette)
    grid = np.stack(np.meshgrid(*[np.arange(size)] * ndim, indexing="ij"),
                    axis=-1).reshape(-1, ndim).astype(np.float64)
    out = np.empty(len(grid), dtype=np.int64)
    for i in range(0, len(grid), chunk):
        g = grid[i:i + chunk, None, :] - seeds[None, :, :]
        g -= size * np.round(g / size)
        out[i:i + chunk] = colours[np.argmin((g * g).sum(-1), axis=1)]
    return out.reshape([size] * ndim)


def any_junction_density(labels, m):
    return float(np.mean(_popcount(neighbourhood_label_bits(labels)) >= m))


def null_sweep(ndim, size, n_seeds, palettes, trials, seed):
    m = ndim + 1
    rng = np.random.default_rng(seed)
    full = {P: 0.0 for P in palettes}
    anyj = {P: 0.0 for P in palettes}
    for P in palettes:
        for _ in range(trials):
            lab = voronoi_labels(size, ndim, n_seeds, P, rng)
            full[P] += full_palette_density(lab, P, min_valence=m) / trials
            anyj[P] += any_junction_density(lab, m) / trials
    return full, anyj


def block_tiling_2d(size, block):
    """Four colours on a checkerboard of blocks: every vertex is four-fold."""
    i, j = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return 2 * ((i // block) % 2) + ((j // block) % 2)


def degenerate_relax(size=72, block=18, steps=(0, 50, 200, 600, 1500), seed=0):
    lab0 = block_tiling_2d(size, block)
    rng = np.random.default_rng(seed)
    f = np.stack([(lab0 == a).astype(float) for a in range(4)])
    f = f * 0.9 + rng.random(f.shape) * 0.01  # tiny noise to break symmetry
    out, t = {}, 0
    for target in steps:
        while t < target:
            f, _ = step_multiphase_conserved(f, None, gamma=1.5, dt=0.1)
            t += 1
        lab = np.argmax(f, axis=0)
        out[target] = (full_palette_density(lab, 4, min_valence=4),
                       full_palette_density(lab, 4, min_valence=3))
    return out


def fmt(d):
    return "  ".join(f"P={k}:{v:.5f}" for k, v in d.items())


def main() -> int:
    configs = [
        # ndim, size, seeds — seed counts give domains ~15-20 lattice units
        (2, 160, 70, [2, 3, 4, 5, 6], 3),
        (3, 56, 40, [2, 3, 4, 5, 6], 3),
        (4, 26, 24, [3, 4, 5, 6, 7], 2),
    ]
    verdicts = []
    for ndim, size, n_seeds, palettes, trials in configs:
        full, anyj = null_sweep(ndim, size, n_seeds, palettes, trials, seed=ndim)
        prof = selection_profile(full)
        print(f"\nd={ndim}  Voronoi null  ({size}^{ndim}, {n_seeds} seeds, m={ndim+1})")
        print("  full-palette :", fmt(full))
        print("  any junction :", fmt(anyj))
        print(f"  argmax={prof['argmax']}  margin={prof['margin']:.1f}")
        verdicts.append(("N1", ndim, prof["argmax"] == ndim + 1))
        verdicts.append(("N2", ndim,
                         all(anyj[P] > 0 for P in palettes if P >= ndim + 1)))

    print("\nN3  degenerate 4-fold start, 2-D, relaxed with published dynamics")
    deg = degenerate_relax()
    for t, (d4, d3) in deg.items():
        print(f"  step {t:5d}: four-fold (m=4) {d4:.5f}   full-palette at m=3 {d3:.5f}")
    first, last = deg[min(deg)][0], deg[max(deg)][0]
    verdicts.append(("N3", 2, first > 0 and last < 0.1 * first))

    print("\nverdicts")
    for name, d, ok in verdicts:
        print(f"  {name} d={d}: {'HOLDS' if ok else 'FAILS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
