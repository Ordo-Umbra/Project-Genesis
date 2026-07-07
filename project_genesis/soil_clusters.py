"""Percolation-style cluster analysis of the fertile-soil mask.

A recalled seed roots only where local capacity κ ≥ threshold (the engine's
κ-as-soil rule, ported to the thermal field in ``sector_seeds``).  The set
of fertile sites is a subset of the lattice, and whether stored *memory* is
globally usable depends not just on *how much* soil is fertile but on how it
is **connected**: one spanning continent lets a recalled seed re-root and
spread anywhere, an archipelago of disconnected islands does not.

This module labels the fertile mask into connected components under
periodic (toroidal) 4-connectivity and reports the standard percolation
observables — the strength ``P∞`` (largest-cluster fraction), the mean
finite-cluster size ``χ`` (the percolation susceptibility, which peaks at
the threshold), and a system-spanning test.  The relevant reference number
is the 2-D square-lattice site-percolation threshold ``p_c ≈ 0.5927``: a
*randomly* occupied mask percolates above it and fragments below it, so
measuring the fertile mask against that baseline shows whether the thermal
field's spatial structure helps or hurts memory connectivity.
"""

from __future__ import annotations

import numpy as np

# 2-D square-lattice site-percolation threshold (4-connectivity), for
# reference against a spatially-uncorrelated mask of the same density.
SITE_PERCOLATION_PC = 0.5927


def label_periodic(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label the True cells of a 2-D boolean mask into connected components.

    4-connectivity with periodic wrap on both axes (a torus).  Returns
    ``(labels, n)`` where ``labels`` is an int array (0 = background,
    1..n = components in discovery order) and ``n`` is the component count.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("label_periodic expects a 2-D mask")
    h, w = mask.shape
    parent = np.arange(h * w)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            here = r * w + c
            if mask[(r + 1) % h, c]:
                union(here, ((r + 1) % h) * w + c)
            if mask[r, (c + 1) % w]:
                union(here, r * w + (c + 1) % w)

    labels = np.zeros((h, w), dtype=int)
    roots: dict[int, int] = {}
    n = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            root = find(r * w + c)
            lab = roots.get(root)
            if lab is None:
                n += 1
                lab = roots[root] = n
            labels[r, c] = lab
    return labels, n


def cluster_sizes(labels: np.ndarray, n: int) -> np.ndarray:
    """Site count of each component 1..n (index 0 is component 1)."""
    if n == 0:
        return np.zeros(0, dtype=int)
    return np.bincount(labels.ravel(), minlength=n + 1)[1:]


def largest_cluster_fraction(mask: np.ndarray) -> float:
    """Percolation strength ``P∞`` = largest fertile cluster / all sites."""
    labels, n = label_periodic(mask)
    if n == 0:
        return 0.0
    return float(cluster_sizes(labels, n).max()) / mask.size


def percolation_susceptibility(sizes: np.ndarray) -> float:
    """Mean finite-cluster size ``χ = Σ' s² / Σ' s`` (largest excluded).

    The size of the cluster a random *non-spanning* fertile site belongs
    to; peaks at the percolation threshold.  Zero when there are no
    finite clusters (empty, or a single all-consuming cluster).
    """
    sizes = np.asarray(sizes, dtype=float)
    if sizes.size <= 1:
        return 0.0
    finite = np.delete(sizes, sizes.argmax())
    tot = finite.sum()
    if tot <= 0:
        return 0.0
    return float((finite ** 2).sum() / tot)


def cluster_spans(labels: np.ndarray, label_id: int) -> bool:
    """Does one component reach across the whole system in some direction?

    True if the component occupies a cell in *every* row or in *every*
    column — a connected path that traverses the full height or width and,
    with the periodic wrap, spans the torus in that direction.
    """
    rows, cols = np.where(labels == label_id)
    if rows.size == 0:
        return False
    h, w = labels.shape
    return len(np.unique(rows)) == h or len(np.unique(cols)) == w


def system_spans(labels: np.ndarray, n: int) -> bool:
    """Does any component span the system (see ``cluster_spans``)?"""
    return any(cluster_spans(labels, lab) for lab in range(1, n + 1))


def cluster_report(mask: np.ndarray) -> dict:
    """All fertile-cluster observables for one boolean mask.

    Returns ``fertile_fraction``, ``p_inf`` (largest-cluster fraction),
    ``chi`` (mean finite-cluster size), ``n_clusters``, and ``spans``.
    """
    mask = np.asarray(mask, dtype=bool)
    labels, n = label_periodic(mask)
    sizes = cluster_sizes(labels, n)
    return {
        "fertile_fraction": float(mask.mean()),
        "p_inf": (float(sizes.max()) / mask.size) if n else 0.0,
        "chi": percolation_susceptibility(sizes),
        "n_clusters": int(n),
        "spans": bool(system_spans(labels, n)),
    }
