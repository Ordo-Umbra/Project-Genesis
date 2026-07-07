"""Percolation-style cluster analysis of the fertile-soil mask.

A recalled seed roots only where local capacity κ ≥ threshold (the engine's
κ-as-soil rule, ported to the thermal field in ``sector_seeds``).  The set
of fertile sites is a subset of the lattice, and whether stored *memory* is
globally usable depends not just on *how much* soil is fertile but on how it
is **connected**: one spanning continent lets a recalled seed re-root and
spread anywhere, an archipelago of disconnected islands does not.

This module labels the fertile mask into connected components under
periodic (toroidal) nearest-neighbour connectivity — 4-connectivity in
2-D, 6-connectivity in 3-D — and reports the standard percolation
observables — the strength ``P∞`` (largest-cluster fraction), the mean
finite-cluster size ``χ`` (the percolation susceptibility, which peaks at
the threshold), and a system-spanning test.  The relevant reference number
is the site-percolation threshold of the lattice — ``p_c ≈ 0.5927`` for the
2-D square lattice, ``p_c ≈ 0.3116`` for the 3-D simple-cubic lattice: a
*randomly* occupied mask percolates above it and fragments below it, so
measuring the fertile mask against that baseline shows whether the thermal
field's spatial structure helps or hurts memory connectivity.  The move to
3-D matters because domain walls, where capacity is spent, become
*surfaces* rather than *lines* — and a surface partitions a volume far less
effectively than a line partitions a plane.
"""

from __future__ import annotations

import numpy as np

# Site-percolation thresholds for reference against a spatially-uncorrelated
# mask of the same density: square lattice (4-connectivity, 2-D) and
# simple-cubic lattice (6-connectivity, 3-D).
SITE_PERCOLATION_PC = 0.5927
SITE_PERCOLATION_PC_3D = 0.3116


def label_periodic(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label the True cells of a 2-D boolean mask into connected components.

    Nearest-neighbour connectivity with periodic wrap on every axis (a
    torus): 4-connectivity in 2-D, 6-connectivity in 3-D.  Returns
    ``(labels, n)`` where ``labels`` is an int array (0 = background,
    1..n = components in discovery order) and ``n`` is the component count.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim not in (2, 3):
        raise ValueError("label_periodic supports 2-D or 3-D masks")
    shape = mask.shape
    ndim = mask.ndim
    # C-order strides so a coordinate tuple maps to a flat index by a dot
    # product (avoids a per-cell numpy call in the hot loop)
    strides = [1] * ndim
    for ax in range(ndim - 2, -1, -1):
        strides[ax] = strides[ax + 1] * shape[ax + 1]
    parent = list(range(mask.size))

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

    for idx in np.ndindex(*shape):
        if not mask[idx]:
            continue
        here = sum(i * s for i, s in zip(idx, strides))
        for ax in range(ndim):  # union with the +1 neighbour on each axis
            nb = list(idx)
            nb[ax] = (idx[ax] + 1) % shape[ax]
            if mask[tuple(nb)]:
                union(here, sum(i * s for i, s in zip(nb, strides)))

    labels = np.zeros(shape, dtype=int)
    roots: dict[int, int] = {}
    n = 0
    for idx in np.ndindex(*shape):
        if not mask[idx]:
            continue
        root = find(sum(i * s for i, s in zip(idx, strides)))
        lab = roots.get(root)
        if lab is None:
            n += 1
            lab = roots[root] = n
        labels[idx] = lab
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
    """Does one component reach across the whole system along some axis?

    True if the component occupies a cell at *every* coordinate along at
    least one axis (every row, or every column, or — in 3-D — every plane):
    a connected path that traverses the full extent of that axis and, with
    the periodic wrap, spans the torus in that direction.
    """
    coords = np.argwhere(labels == label_id)
    if coords.shape[0] == 0:
        return False
    return any(len(np.unique(coords[:, ax])) == labels.shape[ax]
               for ax in range(labels.ndim))


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
