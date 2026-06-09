"""Three-component sector field — the Ψ∈ℂ³ layer of the URP gauge picture.

The scalar URP field (see :mod:`project_genesis.sectorisation`) can only form
*layered* domains: a region in well ``n`` borders wells ``n±1``, so three
mutually-adjacent phases — and the 120° Y-junctions the gauge derivation ties
to colour SU(3) — cannot arise from a single scalar.

This module implements the next layer the gauge paper describes (§4.3.3): a
*sector-membership field* ``Ψ(x) = (R, G, B)`` whose three components compete
on equal footing. Deep inside a sector one component dominates (Ψ ≈ a basis
vector); near boundaries the components mix. With all three phases mutually
adjacent, genuine three-way domains with 120° triple junctions form.

Dynamics are vector Allen–Cahn (overdamped gradient descent on a triple-well
free energy), the natural multi-phase generalisation of the scalar URP update:

    ∂_t η_a = D·∇²η_a − ∂f/∂η_a,
    f(η)    = Σ_a (¼η_a⁴ − ½η_a²) + γ·Σ_{a<b} η_a²·η_b²,
    ∂f/∂η_a = η_a³ − η_a + 2γ·η_a·(Σ_b η_b² − η_a²).

The free energy is S₃-symmetric (permuting R/G/B is a symmetry), the discrete
analogue of the global relabelling symmetry that survives deep inside sectors.
Implemented with vectorised periodic NumPy so it is dimension-agnostic (2-D for
the browser toy, 3-D to match the engine).
"""

from __future__ import annotations

import numpy as np


def periodic_laplacian(field: np.ndarray) -> np.ndarray:
    """Discrete Laplacian with periodic boundaries, for any dimensionality."""
    out = -2.0 * field.ndim * field
    for axis in range(field.ndim):
        out = out + np.roll(field, 1, axis=axis) + np.roll(field, -1, axis=axis)
    return out


def step_multiphase(
    fields: np.ndarray,
    *,
    diffusion: float = 1.0,
    gamma: float = 1.5,
    dt: float = 0.1,
) -> np.ndarray:
    """Advance a multi-phase sector field one Allen–Cahn step.

    Parameters
    ----------
    fields:
        Array of shape ``(P, *spatial)`` holding the ``P`` competing
        components (``P = 3`` for the R/G/B colour sectors).
    diffusion:
        Gradient (surface-tension) coefficient ``D``.
    gamma:
        Cross-coupling penalty. ``γ > 1`` makes the three unit-corner states
        ``(1,0,0)``, ``(0,1,0)``, ``(0,0,1)`` the degenerate minima, so each
        cell is driven toward a single dominant phase.
    dt:
        Explicit time step.

    Returns
    -------
    The updated ``(P, *spatial)`` field array.
    """
    if fields.ndim < 2:
        raise ValueError("fields must have shape (P, *spatial) with P >= 1 component axis")
    sum_sq = np.sum(fields * fields, axis=0)
    new = np.empty_like(fields)
    for a in range(fields.shape[0]):
        fa = fields[a]
        lap = periodic_laplacian(fa)
        df = fa**3 - fa + 2.0 * gamma * fa * (sum_sq - fa * fa)
        new[a] = fa + dt * (diffusion * lap - df)
    return new


def sector_labels(fields: np.ndarray) -> np.ndarray:
    """Return the dominant-component (argmax) sector label at each voxel."""
    return np.argmax(fields, axis=0)


def interface_mask(fields: np.ndarray, *, margin: float = 0.15) -> np.ndarray:
    """Boolean mask of boundary cells where the top two components are close.

    A cell is on a domain wall when no single phase clearly dominates — i.e.
    the gap between the largest and second-largest component is below
    ``margin``.
    """
    srt = np.sort(fields, axis=0)
    gap = srt[-1] - srt[-2]
    return gap < margin


def count_triple_junctions(labels: np.ndarray) -> int:
    """Count cells whose local neighbourhood contains three or more sectors.

    For a multi-phase field these are genuine triple points — the discrete
    analogue of the 120° Y-junctions where three colour domains meet. Works in
    2-D and 3-D via periodic neighbour shifts.
    """
    distinct = np.zeros(labels.shape, dtype=np.int64)
    # Bit-set of labels seen in the (3^d − 1) neighbourhood, via periodic rolls.
    seen_bits = np.zeros(labels.shape, dtype=np.int64)
    offsets = _neighbour_offsets(labels.ndim)
    own = 1 << labels
    for off in offsets:
        shifted = labels
        for axis, delta in enumerate(off):
            if delta:
                shifted = np.roll(shifted, -delta, axis=axis)
        seen_bits = seen_bits | (1 << shifted)
    seen_bits = seen_bits | own
    distinct = _popcount(seen_bits)
    return int(np.sum(distinct >= 3))


def _neighbour_offsets(ndim: int) -> list[tuple[int, ...]]:
    """All non-zero offsets in the 3^ndim neighbourhood."""
    from itertools import product

    offsets = [o for o in product((-1, 0, 1), repeat=ndim) if any(o)]
    return offsets


def _popcount(arr: np.ndarray) -> np.ndarray:
    """Vectorised population count for a small-int bit-set array."""
    out = np.zeros_like(arr)
    a = arr.copy()
    while np.any(a):
        out += a & 1
        a >>= 1
    return out


def analyze_multiphase(fields: np.ndarray, *, beta: float = 0.09) -> dict:
    """Summary report for a multi-phase sector field.

    Reports the number of distinct phases present, the triple-junction count
    (now genuinely non-zero, unlike the scalar model), the boundary fraction,
    and a distinction/integration decomposition in the spirit of the
    S-functional.
    """
    labels = sector_labels(fields)
    walls = interface_mask(fields)
    # Distinction: gradient energy summed across components.
    grad_energy = np.zeros(labels.shape, dtype=np.float64)
    for a in range(fields.shape[0]):
        for axis in range(fields[a].ndim):
            g = 0.5 * (np.roll(fields[a], -1, axis) - np.roll(fields[a], 1, axis))
            grad_energy += g * g
    mean_grad = float(grad_energy.mean())
    return {
        "n_phases": int(len(np.unique(labels))),
        "triple_junctions": count_triple_junctions(labels),
        "boundary_fraction": float(walls.mean()),
        "distinction": float(beta * mean_grad),
        "integration": float(1.0 / (1.0 + mean_grad)),
    }
