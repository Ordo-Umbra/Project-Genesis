"""Dimensional census: the field's forms sorted by codimension (0D/1D/2D).

The collab explorations classified emergent structures into **dimensional
families** — 0D points, 1D lines, 2D blobs — and read their relative
densities as a "quark generation hierarchy" that shifted with phase.  This
module gives that classification a rigorous backbone: the families are the
**cells of the domain tessellation's CW-complex**, sorted by local
codimension.

For a sector-labelled field (each cell carries the id of its dominant
sector), the local dimension of a site is set by how many distinct sectors
meet in its neighbourhood:

    d = 1 sector   → interior of a domain      → a **2-cell** (2D "heavy")
    d = 2 sectors  → a domain wall             → a **1-cell** (1D "medium")
    d ≥ 3 sectors  → a junction where walls meet → a **0-cell** (0D "light")

The census counts the connected components of each class (periodic):
V = #junctions, E = #wall arcs, F = #domains.  Two exact facts make this
more than arbitrary morphology:

- **Euler's theorem on the torus.**  A genuine CW-decomposition of the
  2-torus satisfies ``V − E + F = 0``.  The census must respect it (up to
  the pixel-discretisation of thin necks) — a topological consistency the
  collab's morphology thresholds never had.
- **The trivalent (N⋆=3) signature.**  Where walls meet in threes — the
  120° Y-junctions the repo's sector work selects — every vertex has
  degree 3, so ``2E = 3V`` and Euler gives ``V : E : F = 2 : 3 : 1``.  The
  dimensional hierarchy of the confined phase is not free; it is fixed by
  the junction valence.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage as _ndi
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def local_dimension(labels: np.ndarray) -> np.ndarray:
    """Per-site codimension class from the number of sectors in its 4-nbhd.

    Returns an int array: ``2`` (domain interior / 2-cell), ``1`` (wall /
    1-cell), ``0`` (junction / 0-cell).  Neighbourhoods are periodic — the
    tessellation lives on the torus.
    """
    if labels.ndim != 2:
        raise ValueError("labels must be a 2-D sector-id array")
    lab = labels
    neigh = np.stack([
        lab,
        np.roll(lab, 1, 0), np.roll(lab, -1, 0),
        np.roll(lab, 1, 1), np.roll(lab, -1, 1),
    ], axis=0)
    # distinct-label count per site: sort along the stack and count changes
    s = np.sort(neigh, axis=0)
    distinct = 1 + np.sum(s[1:] != s[:-1], axis=0)
    dim = np.full(lab.shape, 2, dtype=int)      # interior by default
    dim[distinct == 2] = 1                       # wall
    dim[distinct >= 3] = 0                       # junction
    return dim


def _dilate(mask: np.ndarray) -> np.ndarray:
    """One-step 8-connected periodic dilation."""
    out = mask.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            out |= np.roll(np.roll(mask, dx, 0), dy, 1)
    return out


def _count_components(mask: np.ndarray, *, periodic: bool = True) -> int:
    """Connected components of a boolean mask (periodic, 8-connectivity)."""
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for the dimensional census")
    if not mask.any():
        return 0
    structure = np.ones((3, 3), dtype=int)
    labeled, n = _ndi.label(mask, structure=structure)
    if not periodic:
        return int(n)
    # merge components that wrap across opposite faces
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for axis in (0, 1):
        front = labeled.take(0, axis=axis)
        back = labeled.take(-1, axis=axis)
        for f, b in zip(front.ravel(), back.ravel()):
            if f and b:
                union(int(f), int(b))
    roots = {find(i) for i in range(1, n + 1)}
    return len(roots)


def dimensional_census(labels: np.ndarray, *, periodic: bool = True) -> dict:
    """Full 0D/1D/2D census of a sector-labelled 2-D field.

    Returns counts ``V`` (junctions / 0-cells), ``E`` (wall arcs / 1-cells),
    ``F`` (domains / 2-cells), the Euler number ``V − E + F`` (0 on the
    torus for a clean CW-decomposition), the mean junction valence, and the
    ratio vector normalised to F — the dimensional hierarchy.
    """
    dim = local_dimension(labels)
    v = _count_components(dim == 0, periodic=periodic)
    # edges are wall arcs CUT at the junctions: remove a 1-pixel-dilated
    # junction neighbourhood from the wall set so arcs meeting at a vertex
    # become separate 1-cells (else the whole wall graph is one component)
    junction_dil = _dilate(dim == 0)
    wall_arcs = (dim == 1) & ~junction_dil
    e = _count_components(wall_arcs, periodic=periodic)
    # F = number of domains = periodic connected components of equal label
    f = _count_domains(labels, periodic=periodic)
    euler = v - e + f
    # mean junction valence: distinct sectors touching each junction site,
    # averaged over junction sites (3 for a clean Y-junction phase)
    valence = _mean_junction_valence(labels, dim)
    total = max(v + e + f, 1)
    return {
        "V": v, "E": e, "F": f, "euler": euler,
        "mean_valence": valence,
        "fraction_0D": v / total, "fraction_1D": e / total,
        "fraction_2D": f / total,
        "ratio_to_F": (v / max(f, 1), e / max(f, 1), 1.0),
        "area_0D": int(np.sum(dim == 0)),
        "area_1D": int(np.sum(dim == 1)),
        "area_2D": int(np.sum(dim == 2)),
    }


def _count_domains(labels: np.ndarray, *, periodic: bool = True) -> int:
    """Number of domains: periodic connected components of equal-label sites."""
    total = 0
    for sector in np.unique(labels):
        total += _count_components(labels == sector, periodic=periodic)
    return total


def _mean_junction_valence(labels: np.ndarray, dim: np.ndarray) -> float:
    """Average number of distinct sectors meeting at a junction site."""
    js = dim == 0
    if not js.any():
        return 0.0
    neigh = np.stack([
        labels,
        np.roll(labels, 1, 0), np.roll(labels, -1, 0),
        np.roll(labels, 1, 1), np.roll(labels, -1, 1),
        np.roll(np.roll(labels, 1, 0), 1, 1),
        np.roll(np.roll(labels, 1, 0), -1, 1),
        np.roll(np.roll(labels, -1, 0), 1, 1),
        np.roll(np.roll(labels, -1, 0), -1, 1),
    ], axis=0)
    s = np.sort(neigh, axis=0)
    distinct = 1 + np.sum(s[1:] != s[:-1], axis=0)
    return float(distinct[js].mean())


def scalar_sector_labels(field: np.ndarray, n_levels: int = 4) -> np.ndarray:
    """Sector labels for a scalar field by quantile level-banding.

    Bands the field into ``n_levels`` amplitude sectors (the collab φ-field
    convention) — the level sets are the domain walls.
    """
    qs = np.quantile(field, np.linspace(0, 1, n_levels + 1)[1:-1])
    return np.digitize(field, qs).astype(int)
