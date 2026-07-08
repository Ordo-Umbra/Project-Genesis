"""Geometric topological charge for the ψ∈ℂ^N (CP^(N-1)) sector field.

The normalised sector field ``ψ(x) ∈ ℂ^N`` (a unit vector, physics invariant
under an overall local phase ``ψ → e^{iθ(x)}ψ``) is a **CP^(N-1) field**.  The
2-D CP^(N-1) model is the textbook laboratory for the non-perturbative
structure the URP papers invoke for the QCD vacuum — it is asymptotically
free, confines, has a dynamical mass gap, a θ-vacuum, and genuine
**instantons** carrying an integer topological charge ``Q ∈ π₂(CP^(N-1)) =
ℤ``.  This module measures that charge, so the "instanton content" of the
sector field can be read off directly.

The estimator is the **geometric (Berg–Lüscher) construction**, which is
exactly integer on any configuration and invariant under the local phase
(the CP gauge freedom).  Each lattice square is split into two triangles;
the charge of a triangle with corners ``z₁, z₂, z₃`` is the signed area of
the geodesic triangle they span on CP^(N-1), divided by 2π:

    q(△) = (1/2π) · arg[ (z̄₁·z₂)(z̄₂·z₃)(z̄₃·z₁) ] ,   arg ∈ (−π, π]

Summing the two triangles gives the charge in a plaquette; summing all
plaquettes on the periodic lattice gives the total charge ``Q``, an integer.
The topological susceptibility ``χ_top = ⟨Q²⟩ / V`` measures the vacuum's
instanton activity.
"""

from __future__ import annotations

import numpy as np

_TWO_PI = 2.0 * np.pi


def _triangle_charge(z1: np.ndarray, z2: np.ndarray, z3: np.ndarray) -> np.ndarray:
    """Signed geodesic-triangle area / 2π for CP^(N-1) corners (vectorised).

    ``z1, z2, z3`` are ``(..., N)`` complex arrays of (not necessarily unit)
    vectors.  Returns the per-site triangle charge in ``(-½, ½]``.  Invariant
    under an independent phase on each corner (the CP gauge freedom) because
    the three overlaps' phases telescope.
    """
    o12 = np.sum(np.conj(z1) * z2, axis=-1)
    o23 = np.sum(np.conj(z2) * z3, axis=-1)
    o31 = np.sum(np.conj(z3) * z1, axis=-1)
    return np.angle(o12 * o23 * o31) / _TWO_PI


def topological_charge_density(psi: np.ndarray) -> np.ndarray:
    """Per-plaquette geometric charge of a 2-D CP^(N-1) field ``(H, W, N)``.

    Returns an ``(H, W)`` real array; the plaquette anchored at ``(y, x)`` is
    the unit square with corners ``(y,x), (y,x+1), (y+1,x+1), (y+1,x)`` on the
    periodic lattice, triangulated as ``[00,10,11] + [00,11,01]``.
    """
    if psi.ndim != 3:
        raise ValueError("topological_charge_density expects a 2-D field (H, W, N)")
    z00 = psi
    z10 = np.roll(psi, -1, axis=1)   # (y, x+1)
    z01 = np.roll(psi, -1, axis=0)   # (y+1, x)
    z11 = np.roll(np.roll(psi, -1, axis=0), -1, axis=1)  # (y+1, x+1)
    return (_triangle_charge(z00, z10, z11)
            + _triangle_charge(z00, z11, z01))


def topological_charge(psi: np.ndarray) -> float:
    """Total geometric topological charge ``Q`` (integer on a periodic lattice)."""
    return float(topological_charge_density(psi).sum())


def topological_susceptibility(charges, volume: float) -> float:
    """``χ_top = ⟨Q²⟩ / V`` from a sample of total charges over the ensemble."""
    q = np.asarray(charges, dtype=float)
    if q.size == 0:
        return 0.0
    return float(np.mean(q ** 2) / volume)


def instanton_density(psi: np.ndarray) -> float:
    """Mean absolute charge per site — the (anti-)instanton activity density."""
    return float(np.abs(topological_charge_density(psi)).mean())


def cp_action(psi: np.ndarray) -> float:
    """Gauge-invariant CP^(N-1) lattice action ``Σ_x Σ_μ (1 − |z̄·z'|²)``."""
    s = 0.0
    for axis in (0, 1):
        nb = np.roll(psi, -1, axis=axis)
        s += float((1.0 - np.abs(np.sum(np.conj(psi) * nb, axis=-1)) ** 2).sum())
    return s


def cool_step(psi: np.ndarray) -> np.ndarray:
    """One cooling sweep: set each ``z(x)`` to the local action-minimiser.

    Minimising the CP action at a site means maximising
    ``Σ_neighbours |z̄(x)·z_nb|²``, whose solution is the leading eigenvector
    of ``M(x) = Σ_neighbours z_nb z_nb†``.  Cooling removes short-wavelength
    (UV) lattice noise — the dislocations that inflate the raw geometric
    charge — while leaving well-separated instantons intact, the standard
    lattice route to *physical* topological content.
    """
    H, W, N = psi.shape
    M = np.zeros((H, W, N, N), dtype=np.complex128)
    for axis in (0, 1):
        for shift in (-1, 1):
            nb = np.roll(psi, shift, axis=axis)
            M += nb[..., :, None] * np.conj(nb[..., None, :])
    _, vecs = np.linalg.eigh(M)          # ascending eigenvalues
    z = vecs[..., :, -1]                  # leading eigenvector per site
    return z / np.linalg.norm(z, axis=-1, keepdims=True)


def cool(psi: np.ndarray, n_steps: int) -> np.ndarray:
    """Apply ``n_steps`` cooling sweeps (returns a fresh array)."""
    z = psi.copy()
    for _ in range(n_steps):
        z = cool_step(z)
    return z
