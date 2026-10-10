"""Are relaxed multiphase fields generic? The assumption d+1 rests on.

The null in `n3_junction_null.py` showed that `P = d + 1` is selected by *any*
generic tessellation, so what the relaxed-field measurement really certifies is
that the relaxed fields are generic: no point carries more than `d + 1` sectors
and no edge more than its codimension allows. This tests that directly.

Instrument. ``excess(k)`` = among cells whose `3^d` neighbourhood holds at
least ``k`` sectors, the fraction that hold at least ``k + 1``. At ``k = d + 1``
this is the over-valent share of point junctions. On a lattice it is never
exactly zero — a window can straddle two nearby generic vertices — so it is read
against a **Voronoi baseline at matched domain width**, which is generic with
probability one and therefore measures the lattice artefact alone.

G1. **Relaxed fields are no less generic than Voronoi.** For every ``d`` and
    ``P > d + 1`` tested, relaxed ``excess(d+1)`` is at most 1.5x the Voronoi
    baseline at the nearest matched width. Falsifier: a clear relaxed excess,
    meaning the dynamics stabilise degenerate vertices and the zeros above
    ``d + 1`` in the published tables are partly luck of sampling.

G2. **Positive control.** A block checkerboard (degenerate by construction)
    scores an excess far above both. If it does not, the instrument is blind.

G3. **Degenerate 3-D start splits.** An 8-colour 2x2x2 cubic checkerboard has
    8-fold vertices and 4-fold edges. Relaxed with the published dynamics, cells
    of valence ``>= 5`` (impossible at a generic 3-D point) should fall to the
    Voronoi baseline. Falsifier: they persist.

G4. *(Added after G2 failed in 3-D -- see below.)* **Junction cells form
    isolated blobs, not networks.** Connected components of the
    ``valence >= d + 1`` mask: a generic vertex is a blob of a few to a few tens
    of cells; a degenerate *edge* (four colours along a line in 3-D) joins them
    into extended runs. Relaxed mean component size should sit at or below the
    Voronoi value at matched width.

Why G2 failed in 3-D, and why G4 exists. The 8-colour checkerboard's excess is
only ``0.10``: its four-fold *edges* produce valence-4 cells along whole lines,
which ``excess(4)`` reads as ordinary point junctions. So ``excess`` detects
over-valent points but is blind to over-valent edges -- and an over-valent edge
is exactly what would let a non-generic field inflate the ``P = d + 1`` count.
Component size is the shape test that catches it.

    python experiments/n3_junction_genericity.py          # d = 2, 3
    python experiments/n3_junction_genericity.py --with-4d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from n3_junction_null import voronoi_labels  # noqa: E402
from n3_junction_scale import relaxed  # noqa: E402
from project_genesis.junction_scale import (  # noqa: E402
    neighbourhood_label_bits, neighbourhood_offsets,
)  # noqa: E402
from project_genesis.multiphase import (  # noqa: E402
    _popcount, domain_diameter, step_multiphase_conserved,
)


def valence(labels):
    return _popcount(neighbourhood_label_bits(labels))


def excess(labels, k):
    v = valence(labels)
    base = np.count_nonzero(v >= k)
    return (np.count_nonzero(v >= k + 1) / base if base else float("nan")), int(base)


def components(mask):
    """Sizes of periodic connected components (``3^d`` connectivity).

    Min-label propagation; iterations scale with component diameter, which is
    small for every field this is used on.
    """
    mask = np.asarray(mask, dtype=bool)
    lab = np.where(mask, np.arange(mask.size).reshape(mask.shape), mask.size)
    offsets = neighbourhood_offsets(mask.ndim)
    while True:
        new = lab
        for off in offsets:
            shifted = lab
            for axis, delta in enumerate(off):
                if delta:
                    shifted = np.roll(shifted, delta, axis=axis)
            new = np.minimum(new, shifted)
        new = np.where(mask, new, mask.size)
        if np.array_equal(new, lab):
            break
        lab = new
    return np.unique(lab[mask], return_counts=True)[1]


def blob_size(labels_list, k):
    """Mean component size of the ``valence >= k`` mask, pooled over fields."""
    sizes = [components(valence(lab) >= k) for lab in labels_list]
    sizes = [s for s in sizes if len(s)]
    return float(np.concatenate(sizes).mean()) if sizes else float("nan")


def checkerboard(size, ndim, block):
    """2^ndim colours on a block checkerboard: every corner is 2^ndim-fold."""
    idx = np.meshgrid(*[np.arange(size)] * ndim, indexing="ij")
    lab = np.zeros([size] * ndim, dtype=np.int64)
    for axis, i in enumerate(idx):
        lab += ((i // block) % 2) << axis
    return lab


def voronoi_baseline(ndim, size, P, widths, rng, trials=2):
    """Voronoi excess at a few seed densities, keyed by measured width."""
    vol = size ** ndim
    out = []
    for w in widths:
        n = max(P, int(round(vol / w ** ndim)))
        ex, wd, cnt, labs = [], [], 0, []
        for _ in range(trials):
            lab = voronoi_labels(size, ndim, n, P, rng)
            e, c = excess(lab, ndim + 1)
            ex.append(e); wd.append(domain_diameter(lab)); cnt += c
            labs.append(lab)
        out.append((float(np.mean(wd)), float(np.nanmean(ex)), cnt,
                    blob_size(labs, ndim + 1)))
    return out


def nearest(baseline, width):
    return min(baseline, key=lambda b: abs(b[0] - width))


def relax_from(labels, P, steps, seed=0):
    rng = np.random.default_rng(seed)
    f = np.stack([(labels == a).astype(float) for a in range(P)]) * 0.9
    f += rng.random(f.shape) * 0.01
    for _ in range(steps):
        f, _ = step_multiphase_conserved(f, None, gamma=1.5, dt=0.1)
    return np.argmax(f, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-4d", action="store_true")
    args = ap.parse_args()

    arms = [(2, 72, 600, [3, 4, 5, 6], 3), (3, 28, 1500, [4, 5, 6], 2)]
    if args.with_4d:
        arms.append((4, 24, 1500, [5, 6], 1))

    verdicts = []
    rng = np.random.default_rng(7)
    for ndim, size, steps, palettes, trials in arms:
        k = ndim + 1
        print(f"\nd={ndim}  excess({k}) = P(valence >= {k+1} | valence >= {k})")
        for P in palettes:
            ex, wd, cnt, labs = [], [], 0, []
            for t in range(trials):
                lab = relaxed(P, size, ndim, steps, 1.5, 0.1, seed=100 * P + t)
                e, c = excess(lab, k)
                ex.append(e); wd.append(domain_diameter(lab)); cnt += c
                labs.append(lab)
            blob = blob_size(labs, k)
            w = float(np.mean(wd))
            r = float(np.nanmean(ex))
            vbig = 2 * size if ndim < 4 else size
            base = voronoi_baseline(ndim, vbig, P, [0.8 * w, w, 1.2 * w], rng)
            bw, be, bc, bb = nearest(base, w)
            print(f"  P={P}: relaxed {r:.4f} blob {blob:5.1f} (n={cnt}, L={w:.1f})"
                  f"   Voronoi {be:.4f} blob {bb:5.1f} (n={bc}, L={bw:.1f})")
            if P > k and cnt >= 20:
                verdicts.append((f"G1 d={ndim} P={P}", r <= 1.5 * be + 1e-12))
            if cnt >= 20:
                verdicts.append((f"G4 d={ndim} P={P}", blob <= 1.5 * bb))

        block = 9 if ndim < 4 else 6
        cb = checkerboard(size, ndim, block)
        e, _ = excess(cb, k)
        print(f"  checkerboard ({2**ndim} colours, block {block}): excess {e:.4f}"
              f"  blob {blob_size([cb], k):.1f}")
        verdicts.append((f"G2 d={ndim}", e > 0.5))

    print("\nG3  8-colour cubic checkerboard, 3-D, relaxed")
    lab0 = checkerboard(32, 3, 8)
    rng3 = np.random.default_rng(3)
    rows = []
    for steps in (0, 300, 800, 1500, 2500):
        lab = lab0 if steps == 0 else relax_from(lab0, 8, steps)
        v = valence(lab)
        rows.append((steps, float(np.mean(v >= 5)), float(np.mean(v >= 4)),
                     excess(lab, 4)[0], domain_diameter(lab), blob_size([lab], 4)))
        print(f"  step {steps:5d}: valence>=5 {rows[-1][1]:.5f}  valence>=4 "
              f"{rows[-1][2]:.5f}  excess(4) {rows[-1][3]:.4f}  "
              f"blob {rows[-1][5]:7.1f}  L={rows[-1][4]:.1f}")
    vb = voronoi_baseline(3, 64, 8, [rows[-1][4]], rng3)[0]
    print(f"  Voronoi baseline at L={vb[0]:.1f}: excess(4) {vb[1]:.4f}  blob {vb[3]:.1f}")
    verdicts.append(("G3 d=3", rows[-1][3] <= 1.5 * vb[1]))
    verdicts.append(("G4 d=3 checkerboard relaxed", rows[-1][5] <= 1.5 * vb[3]))

    print("\nverdicts")
    for name, ok in verdicts:
        print(f"  {name}: {'HOLDS' if ok else 'FAILS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
