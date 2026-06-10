"""β-sectorisation and boundary-formation analysis for the URP field.

This module turns the abstract claim at the heart of the URP gauge paper —
that the β-nonlinearity drives a continuous medium to partition into a small
number of stable domains separated by domain walls — into a concrete
*measurement instrument*.

It also operationalises the framing of "The Range": a *being* is not a thing
inside a boundary, it **is** the boundary, the work of maintaining a domain of
coherence against its surroundings. Each detected sector is therefore treated
as a candidate "being", and reported through the two moves every such system
runs — **distinction** (gradient energy concentrated on its walls) and
**integration** (internal coherence within the domain).

Pipeline
--------
1. Compute the periodic gradient magnitude |∇φ| (reusing the simulation's own
   Numba kernel so the analysis matches the dynamics' boundary conditions).
2. Mark **domain walls** as the voxels where |∇φ| is high — these are the
   boundaries, the locus of boundary-work.
3. Label the remaining low-gradient **interior** voxels into connected
   components (with optional periodic wrapping). Each component is a *sector*.
4. Report sector count N, per-sector statistics, boundary metrics, and
   triple-junction ("Y-junction") counts.

The β-sectorisation paper predicts N⋆ = 3 for β ≈ 0.09. This module does not
assume that answer; it provides the means to *measure* N for any evolved field
so the prediction can be tested empirically.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy import ndimage

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    ndimage = None  # type: ignore[assignment]
    _HAS_SCIPY = False

from .numba_kernels import gradient_squared_3d


# ----------------------------------------------------------------------
# Gradient magnitude
# ----------------------------------------------------------------------


def gradient_magnitude(field: np.ndarray) -> np.ndarray:
    """Return the periodic gradient magnitude |∇φ| of a 3-D scalar field.

    Uses the same central-difference, periodic-BC kernel as the field
    evolution so that the detected walls correspond to the dynamics that
    produced them.
    """
    f64 = np.ascontiguousarray(field, dtype=np.float64)
    gsq = np.empty_like(f64)
    gradient_squared_3d(f64, gsq)
    return np.sqrt(gsq)


def _wall_threshold(
    grad_mag: np.ndarray,
    *,
    method: str,
    k: float,
) -> float:
    """Compute the |∇φ| cut-off separating walls (high) from interiors (low)."""
    if method == "std":
        return float(grad_mag.mean() + k * grad_mag.std())
    if method == "quantile":
        if not 0.0 <= k <= 1.0:
            raise ValueError("for method='quantile', k must lie in [0, 1]")
        return float(np.quantile(grad_mag, k))
    if method == "absolute":
        return float(k)
    raise ValueError("method must be one of: 'std', 'quantile', 'absolute'")


# ----------------------------------------------------------------------
# Periodic connected-component labelling
# ----------------------------------------------------------------------


class _UnionFind:
    """Minimal union-find for merging labels across periodic boundaries."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression.
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)


def _merge_periodic_faces(labels: np.ndarray, n_labels: int) -> np.ndarray:
    """Merge interior labels that are connected across opposite periodic faces.

    Returns a relabelled array with contiguous labels ``1..K``.
    """
    uf = _UnionFind(n_labels + 1)
    for axis in range(labels.ndim):
        face_lo = np.take(labels, 0, axis=axis)
        face_hi = np.take(labels, labels.shape[axis] - 1, axis=axis)
        mask = (face_lo > 0) & (face_hi > 0)
        if not np.any(mask):
            continue
        pairs = np.unique(
            np.stack([face_lo[mask], face_hi[mask]], axis=1), axis=0
        )
        for a, b in pairs:
            uf.union(int(a), int(b))

    # Vectorised relabel: map each original label to its union-find root,
    # then compact the surviving roots to contiguous ids 1..K.
    roots = np.array([uf.find(lbl) for lbl in range(n_labels + 1)], dtype=labels.dtype)
    unique_roots = np.unique(roots[1:]) if n_labels > 0 else np.array([], dtype=labels.dtype)
    compact = np.zeros(int(roots.max()) + 1, dtype=labels.dtype)
    compact[unique_roots] = np.arange(1, len(unique_roots) + 1, dtype=labels.dtype)
    return compact[roots[labels]]


def label_sectors(
    field: np.ndarray,
    *,
    method: str = "std",
    k: float = 1.0,
    periodic: bool = True,
    min_size: int = 1,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Partition a field into sectors separated by high-gradient domain walls.

    Parameters
    ----------
    field:
        3-D scalar field.
    method, k:
        Wall threshold rule. ``'std'`` (default) marks walls where
        ``|∇φ| > mean + k·std``; ``'quantile'`` uses the ``k``-quantile of
        ``|∇φ|``; ``'absolute'`` uses ``k`` directly.
    periodic:
        If True, sectors that touch across opposite faces are merged
        (matching the simulation's periodic boundary conditions).
    min_size:
        Sectors smaller than this many voxels are discarded (relabelled to
        wall). Useful for ignoring speckle.

    Returns
    -------
    ``(labels, n_sectors, wall_mask)`` where ``labels`` is an integer array
    (0 = wall/boundary, ``1..N`` = sectors), ``n_sectors`` is ``N`` after
    size-filtering, and ``wall_mask`` is the boolean boundary set.
    """
    if not _HAS_SCIPY:
        raise RuntimeError(
            "scipy is required for sector labelling. Install it with: pip install scipy"
        )
    if field.ndim != 3:
        raise ValueError("field must be 3-dimensional")

    grad_mag = gradient_magnitude(field)
    threshold = _wall_threshold(grad_mag, method=method, k=k)
    wall_mask = grad_mag > threshold
    interior = ~wall_mask

    # 6-connectivity for the interior components.
    structure = ndimage.generate_binary_structure(3, 1)
    labels, n_labels = ndimage.label(interior, structure=structure)

    if periodic and n_labels > 1:
        labels = _merge_periodic_faces(labels, n_labels)
        n_labels = int(labels.max())

    if min_size > 1 and n_labels > 0:
        counts = np.bincount(labels.ravel(), minlength=n_labels + 1)
        # Relabel sectors below min_size to wall (0), compacting the rest.
        kept = [lbl for lbl in range(1, n_labels + 1) if counts[lbl] >= min_size]
        remap = np.zeros(n_labels + 1, dtype=labels.dtype)
        remap[kept] = np.arange(1, len(kept) + 1, dtype=labels.dtype)
        labels = remap[labels]
        n_labels = len(kept)

    return labels, int(n_labels), wall_mask


# ----------------------------------------------------------------------
# Per-sector statistics & boundary metrics
# ----------------------------------------------------------------------


def _sector_surface_area(labels: np.ndarray, sector_id: int) -> int:
    """Count voxel-faces where the sector touches a different label (periodic)."""
    mask = labels == sector_id
    area = 0
    for axis in range(labels.ndim):
        neighbour = np.roll(labels, shift=1, axis=axis)
        area += int(np.sum(mask & (neighbour != sector_id)))
        neighbour = np.roll(labels, shift=-1, axis=axis)
        area += int(np.sum(mask & (neighbour != sector_id)))
    return area


def sector_statistics(
    field: np.ndarray,
    labels: np.ndarray,
    beta: float,
) -> list[dict[str, float | int]]:
    """Per-sector report: volume, mean field, surface area, and a being's
    distinction/integration balance.

    Each sector is treated as a candidate "being". ``distinction`` is the
    internal gradient energy (β|∇φ|² averaged over the sector — the structural
    differentiation it sustains); ``integration`` is its internal coherence
    ``1/(1+mean|∇φ|²)`` (how tightly the domain holds together).
    """
    grad_sq = gradient_magnitude(field) ** 2
    n = int(labels.max())
    stats: list[dict[str, float | int]] = []
    for sid in range(1, n + 1):
        mask = labels == sid
        volume = int(np.sum(mask))
        if volume == 0:
            continue
        local_gsq = grad_sq[mask]
        grad_energy = float(local_gsq.mean())
        distinction = float(beta * grad_energy)
        integration = float(1.0 / (1.0 + grad_energy))
        stats.append(
            {
                "sector_id": sid,
                "volume": volume,
                "volume_fraction": volume / field.size,
                "mean_field": float(field[mask].mean()),
                "std_field": float(field[mask].std()),
                "surface_area": _sector_surface_area(labels, sid),
                "distinction": distinction,
                "integration": integration,
            }
        )
    return stats


def count_triple_junctions(labels: np.ndarray) -> int:
    """Count wall voxels where three or more distinct sectors meet.

    A triple junction is the discrete analogue of the 120° Y-junctions the
    β-sectorisation paper associates with the three-sector (SU(3)) attractor.
    """
    wall = labels == 0
    junctions = 0
    coords = np.argwhere(wall)
    nx, ny, nz = labels.shape
    for x, y, z in coords:
        neighbours = set()
        for axis, (a, b, c) in enumerate(
            [
                ((x + 1) % nx, y, z),
                ((x - 1) % nx, y, z),
                (x, (y + 1) % ny, z),
                (x, (y - 1) % ny, z),
                (x, y, (z + 1) % nz),
                (x, y, (z - 1) % nz),
            ]
        ):
            lbl = int(labels[a, b, c])
            if lbl > 0:
                neighbours.add(lbl)
        if len(neighbours) >= 3:
            junctions += 1
    return junctions


# ----------------------------------------------------------------------
# Free-energy / sector-count tradeoff (illustrative)
# ----------------------------------------------------------------------


def free_energy(n: int | np.ndarray, a: float, b: float) -> float | np.ndarray:
    """Boundary–information tradeoff ``F(N) = a·N^(2/3) − b·N``.

    From the β-sectorisation analysis: ``a > 0`` is the boundary (wall) cost,
    growing with total wall area ``∝ N^(2/3)``; ``b > 0`` is the information
    gain from having ``N`` distinct sectors. The stationary point sits at
    ``N⋆ = (2a / 3b)³`` (see :func:`stationary_sector_count`).

    Note: with these signs the stationary point is where the *gain–cost*
    balance turns over; whether it is the operative optimum depends on the
    additional topological/anomaly constraints discussed in the gauge paper.
    The robust empirical quantity is the *measured* sector count from
    :func:`label_sectors`; this helper is provided for comparison only.
    """
    n_arr = np.asarray(n, dtype=float)
    return a * n_arr ** (2.0 / 3.0) - b * n_arr


def stationary_sector_count(a: float, b: float) -> float:
    """Return the continuous stationary sector count ``N⋆ = (2a / 3b)³``."""
    if b <= 0:
        raise ValueError("b must be > 0")
    return (2.0 * a / (3.0 * b)) ** 3


# ----------------------------------------------------------------------
# Top-level report
# ----------------------------------------------------------------------


def analyze_sectorisation(
    field: np.ndarray,
    beta: float,
    *,
    method: str = "std",
    k: float = 1.0,
    periodic: bool = True,
    min_size: int = 1,
    include_sectors: bool = True,
) -> dict[str, Any]:
    """Full β-sectorisation / boundary-formation report for a field.

    Returns a JSON-friendly dict combining the sector count (the ``N⋆``
    question), the global boundary-work measures (The Range's "a being is its
    boundary"), and optional per-sector statistics.
    """
    labels, n_sectors, wall_mask = label_sectors(
        field, method=method, k=k, periodic=periodic, min_size=min_size
    )
    grad_mag = gradient_magnitude(field)

    boundary_fraction = float(wall_mask.mean())
    # Distinction lives on the walls; integration lives in the interiors.
    wall_distinction = float(beta * (grad_mag[wall_mask] ** 2).mean()) if wall_mask.any() else 0.0
    interior_mask = ~wall_mask
    interior_grad_energy = float((grad_mag[interior_mask] ** 2).mean()) if interior_mask.any() else 0.0
    interior_integration = float(1.0 / (1.0 + interior_grad_energy))

    report: dict[str, Any] = {
        "n_sectors": n_sectors,
        "boundary_fraction": boundary_fraction,
        "wall_distinction": wall_distinction,
        "interior_integration": interior_integration,
        "triple_junctions": count_triple_junctions(labels),
        "wall_threshold_method": method,
        "wall_threshold_k": k,
    }
    if include_sectors:
        report["sectors"] = sector_statistics(field, labels, beta)
    return report
