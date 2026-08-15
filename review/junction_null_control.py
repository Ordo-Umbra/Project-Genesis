"""Control: does `d+1` need the URP field dynamics at all?

Replace the relaxed multiphase Allen-Cahn field with a plain Voronoi
tessellation -- random seed points on a torus, each cell labelled by its
nearest seed's colour. No capacity, no S-functional, no Allen-Cahn, no
programme. Then score it with the repo's own measure, unmodified.

If argmax is still P = d+1, the result is a fact about generic tessellations,
not about anything this repository's dynamics do.
"""
import numpy as np
from project_genesis.junction_scale import full_palette_density

def voronoi_labels(n, ndim, n_seeds, n_palette, rng):
    """Nearest-seed labelling on a periodic [0,n)^ndim lattice."""
    seeds = rng.uniform(0, n, size=(n_seeds, ndim))
    colours = rng.integers(0, n_palette, size=n_seeds)
    # ensure every colour is actually used
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

def run(ndim, n, n_seeds, palettes, trials=3, seed=17):
    rng = np.random.default_rng(seed)
    out = {}
    for p in palettes:
        vals = [full_palette_density(voronoi_labels(n, ndim, n_seeds, p, rng),
                                     p, radius=1, min_valence=ndim + 1)
                for _ in range(trials)]
        out[p] = float(np.mean(vals))
    return out

if __name__ == "__main__":
    print("Voronoi control -- scored with the repo's own full_palette_density,")
    print("min_valence = d+1, radius = 1. No field dynamics involved.\n")
    for ndim, n, n_seeds in [(1, 4000, 180), (2, 72, 14), (3, 28, 9)]:
        res = run(ndim, n, n_seeds, [2, 3, 4, 5, 6])
        best = max(res, key=lambda k: res[k])
        cells = f"{n}^{ndim}"
        print(f"d = {ndim}  ({cells}, {n_seeds} seeds)  m = {ndim + 1}")
        print("   " + "  ".join(f"P={p}: {v:.5f}" for p, v in res.items()))
        print(f"   argmax = P = {best}   (d+1 = {ndim + 1})"
              f"  {'MATCH' if best == ndim + 1 else 'MISMATCH'}\n")
