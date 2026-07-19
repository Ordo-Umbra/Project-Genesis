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

import itertools as _itertools

import numpy as np

try:
    from scipy import ndimage as _ndi
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def _moore_distinct(labels: np.ndarray) -> np.ndarray:
    """Distinct-sector count per site over the full Moore neighbourhood."""
    stack = []
    for offs in _itertools.product((-1, 0, 1), repeat=labels.ndim):
        shifted = labels
        for ax, o in enumerate(offs):
            if o:
                shifted = np.roll(shifted, o, ax)
        stack.append(shifted)
    s = np.sort(np.stack(stack, axis=0), axis=0)
    return 1 + np.sum(s[1:] != s[:-1], axis=0)


def local_dimension(labels: np.ndarray) -> np.ndarray:
    """Per-site cell dimension from the number of sectors in its von-Neumann nbhd.

    A site where ``k`` distinct sectors meet has codimension ``k − 1``, so its
    cell dimension is ``d + 1 − k`` (clamped at 0), where ``d`` is the spatial
    dimension.  In 2-D: 1 sector → 2-cell (interior), 2 → 1-cell (wall), ≥3 →
    0-cell (junction).  In 3-D: 1 → 3-cell (volume), 2 → 2-cell (face), 3 →
    1-cell (triple line), ≥4 → 0-cell (vertex).  Neighbourhoods are periodic —
    the tessellation lives on the torus.
    """
    d = labels.ndim
    stack = [labels]
    for ax in range(d):
        stack.append(np.roll(labels, 1, ax))
        stack.append(np.roll(labels, -1, ax))
    s = np.sort(np.stack(stack, axis=0), axis=0)
    distinct = 1 + np.sum(s[1:] != s[:-1], axis=0)
    return np.maximum(d + 1 - distinct, 0).astype(int)


def _dilate(mask: np.ndarray) -> np.ndarray:
    """One-step Moore (fully-connected) periodic dilation, any dimension."""
    out = mask.copy()
    for offs in _itertools.product((-1, 0, 1), repeat=mask.ndim):
        if any(offs):
            shifted = mask
            for ax, o in enumerate(offs):
                if o:
                    shifted = np.roll(shifted, o, ax)
            out = out | shifted
    return out


def _count_components(mask: np.ndarray, *, periodic: bool = True) -> int:
    """Connected components of a boolean mask (periodic, Moore-connectivity)."""
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for the dimensional census")
    if not mask.any():
        return 0
    structure = np.ones((3,) * mask.ndim, dtype=int)
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

    for axis in range(mask.ndim):
        front = labeled.take(0, axis=axis)
        back = labeled.take(-1, axis=axis)
        for f, b in zip(front.ravel(), back.ravel()):
            if f and b:
                union(int(f), int(b))
    roots = {find(i) for i in range(1, n + 1)}
    return len(roots)


def dimensional_census(labels: np.ndarray, *, periodic: bool = True) -> dict:
    """Full CW-census of a sector-labelled field, in any spatial dimension.

    Counts the connected components of each cell dimension ``ℓ = 0..d``
    (``d = labels.ndim``): a ℓ-cell is a component of ``{dim == ℓ}`` **cut at
    the lower-dimensional cells** (the dilated ``{dim < ℓ}`` removed, so cells
    split at their boundaries); the top cells (ℓ = d) are the domains
    themselves.  Returns per-dimension counts ``cells`` (``cells[0]`` = 0-cells
    …), the named ``V``/``E``/``F``(/``C`` in 3-D), and the **Euler number**
    ``Σ (−1)^ℓ N_ℓ`` — which is 0 on the torus ``Tᵈ`` for a clean
    decomposition (``V − E + F`` in 2-D, ``V − E + F − C`` in 3-D).

    Also: ``mean_valence`` (distinct sectors at the 0-cells — 3 for 2-D
    Y-junctions, ~4 for 3-D tetrahedral vertices), ``valence_by_dim`` (the
    Plateau structure: how many sectors meet at each cell dimension), the
    ratio to the top cell, and per-dimension area fractions.
    """
    d = labels.ndim
    dim = local_dimension(labels)
    cells = {}
    for ell in range(d):                     # 0 .. d-1 : cut at lower cells
        mask = (dim == ell)
        if ell > 0:
            mask = mask & ~_dilate(dim < ell)
        cells[ell] = _count_components(mask, periodic=periodic)
    cells[d] = _count_domains(labels, periodic=periodic)   # top cells = domains
    euler = int(sum((-1) ** ell * cells[ell] for ell in range(d + 1)))
    total = max(sum(cells.values()), 1)
    top = max(cells[d], 1)
    out = {
        "cells": {ell: cells[ell] for ell in range(d + 1)},
        "euler": euler,
        "mean_valence": _mean_junction_valence(labels, dim),
        "valence_by_dim": _valence_by_dim(labels, dim),
        "ratio_to_top": tuple(cells[ell] / top for ell in range(d + 1)),
    }
    # named keys (V/E/F in 2-D — F is the domain count there; V/E/F/C in 3-D)
    names = ["V", "E", "F", "C"]
    for ell in range(d + 1):
        out[names[ell]] = cells[ell]
        out[f"area_{ell}D"] = int(np.sum(dim == ell))
        out[f"fraction_{ell}D"] = cells[ell] / total
    if d == 2:                               # back-compat with the 2-D callers
        out["ratio_to_F"] = out["ratio_to_top"]
    return out


def _count_domains(labels: np.ndarray, *, periodic: bool = True) -> int:
    """Number of domains: periodic connected components of equal-label sites."""
    total = 0
    for sector in np.unique(labels):
        total += _count_components(labels == sector, periodic=periodic)
    return total


def _mean_junction_valence(labels: np.ndarray, dim: np.ndarray) -> float:
    """Average number of distinct sectors meeting at a 0-cell (vertex) site."""
    js = dim == 0
    if not js.any():
        return 0.0
    return float(_moore_distinct(labels)[js].mean())


def _valence_by_dim(labels: np.ndarray, dim: np.ndarray) -> dict:
    """Mean distinct sectors meeting at each cell dimension (Plateau structure).

    A clean foam has ``d + 1 − ℓ`` sectors meeting at a ℓ-cell: in 3-D,
    4 at vertices (0-cells), 3 along triple lines (1-cells), 2 on faces
    (2-cells).
    """
    distinct = _moore_distinct(labels)
    out = {}
    for ell in range(labels.ndim + 1):
        mask = dim == ell
        out[ell] = float(distinct[mask].mean()) if mask.any() else 0.0
    return out


def cell_flavours(labels: np.ndarray, ell: int, *,
                  periodic: bool = True) -> dict:
    """Flavour distribution over the ℓ-cells: their sector-composition.

    A form's **flavour** is the set of sectors composing it — a domain (top
    cell) carries one sector, a wall (2 sectors, a colour–anticolour pair),
    a junction (``d+1`` sectors, a colour-singlet).  Returns a dict mapping
    ``frozenset`` of sectors → count of ℓ-cells with that composition,
    restricted to the canonical composition size ``d + 1 − ℓ`` (the generic
    flavour of a ℓ-cell).  With ``P`` sectors the multiplet size is
    ``C(P, d+1−ℓ)`` — the Pascal-triangle flavour spectrum.
    """
    d = labels.ndim
    dim = local_dimension(labels)
    want = d + 1 - ell
    if ell == d:                                     # domains: per-sector
        out = {}
        for sector in np.unique(labels):
            out[frozenset((int(sector),))] = _count_components(
                labels == sector, periodic=periodic)
        return out
    mask = (dim == ell) & ~_dilate(dim < ell)
    if not mask.any() or not _HAS_SCIPY:
        return {}
    labeled, n = _ndi.label(mask, structure=np.ones((3,) * d, dtype=int))
    counts: dict = {}
    for comp in range(1, n + 1):
        footprint = _dilate(labeled == comp)
        flav = frozenset(int(s) for s in np.unique(labels[footprint]))
        if len(flav) == want:
            counts[flav] = counts.get(flav, 0) + 1
    return counts


def flavour_entropy(counts: dict) -> float:
    """Normalised Shannon entropy of a flavour distribution (1 = democratic)."""
    v = np.array(list(counts.values()), dtype=float)
    if v.sum() <= 0 or len(v) < 2:
        return 1.0 if len(v) == 1 else 0.0
    p = v / v.sum()
    return float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(v)))


def scalar_sector_labels(field: np.ndarray, n_levels: int = 4) -> np.ndarray:
    """Sector labels for a scalar field by quantile level-banding.

    Bands the field into ``n_levels`` amplitude sectors (the collab φ-field
    convention) — the level sets are the domain walls.
    """
    qs = np.quantile(field, np.linspace(0, 1, n_levels + 1)[1:-1])
    return np.digitize(field, qs).astype(int)
