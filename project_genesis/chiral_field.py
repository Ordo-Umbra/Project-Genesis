"""Chirality: spin as a parity-breaking term, and the handedness it sets.

The dimensional census sorted the field's forms by codimension but left the
0D junctions handedness-blind, and the κ-molecule could not hold angular
momentum — both because the sector dynamics carry no **chiral term**.  The
collab's "vorticity (spin analog)" histograms were symmetric for the same
reason: with no parity-breaking coupling, left- and right-handed textures
form in equal measure.

This module adds the minimal chiral term via the **complex Ginzburg–Landau
(CGL) field** ``ψ = ψ_r + iψ_i``:

    ∂_t ψ = ψ + (1 + iλ)·∇²ψ − (1 + iλ)·|ψ|²·ψ ,

where ``λ`` is the chiral coupling.  ``λ = 0`` is real Ginzburg–Landau —
parity-symmetric, the collab's uncoupled case.  ``λ ≠ 0`` makes the
diffusion and reaction phase-rotating, so relaxing textures acquire a
definite twist: spiral waves and vortices of one handedness, set by
``sign(λ)``.  The **vorticity** (the spin analog) is the vortex-charge
density

    ω(x) = ∂_x ψ_r · ∂_y ψ_i − ∂_y ψ_r · ∂_x ψ_i ,

a parity-odd pseudoscalar whose field-mean ``⟨ω⟩`` is the net handedness:
zero when parity holds, ``∝ λ`` (same sign) when the chiral term breaks it.
"""

from __future__ import annotations

import numpy as np

from .multiphase import periodic_laplacian


def step_chiral(psi: np.ndarray, *, chiral: float = 0.0,
                dt: float = 0.05) -> np.ndarray:
    """One CGL step of the complex field ``psi`` with chiral coupling ``λ``.

    ``chiral = 0`` is real Ginzburg–Landau (parity-symmetric).  The Laplacian
    is the repo's periodic 5-point stencil, applied to real and imaginary
    parts.  Explicit Euler; keep ``dt`` small (the reaction is stiff).
    """
    lap = periodic_laplacian(psi.real) + 1j * periodic_laplacian(psi.imag)
    coeff = 1.0 + 1j * chiral
    dpsi = psi + coeff * lap - coeff * (np.abs(psi) ** 2) * psi
    return psi + dt * dpsi


def vorticity(psi: np.ndarray) -> np.ndarray:
    """The spin analog ``ω = ∂_xψ_r ∂_yψ_i − ∂_yψ_r ∂_xψ_i`` (vortex density)."""
    def dx(f):
        return 0.5 * (np.roll(f, -1, 0) - np.roll(f, 1, 0))

    def dy(f):
        return 0.5 * (np.roll(f, -1, 1) - np.roll(f, 1, 1))

    ar, ai = psi.real, psi.imag
    return dx(ar) * dy(ai) - dy(ar) * dx(ai)


def evolve_chiral(shape, *, chiral: float = 0.0, steps: int = 400,
                  dt: float = 0.05, seed: int = 0,
                  init_amp: float = 0.1) -> np.ndarray:
    """Relax a CGL field from complex noise; return the final ``ψ``."""
    rng = np.random.default_rng(seed)
    psi = init_amp * (rng.standard_normal(shape)
                      + 1j * rng.standard_normal(shape))
    for _ in range(steps):
        psi = step_chiral(psi, chiral=chiral, dt=dt)
    return psi


def net_vorticity(psi: np.ndarray) -> float:
    """Field-mean vorticity ``⟨ω⟩`` — pinned to ~0 on the torus (vortex pairs).

    Kept as a documented control: total vorticity is topologically neutral on
    the closed torus, so it *cannot* see chirality — the intrinsic precession
    ``Ω`` (below) is the observable that can.
    """
    return float(vorticity(psi).mean())


def precession_rate(psi: np.ndarray, *, chiral: float, dt: float = 0.05,
                    n_probe: int = 20, amp_floor: float = 0.5) -> float:
    """Intrinsic precession ``Ω`` — the field's own spin (rad per unit time).

    Advances a copy ``n_probe`` steps and reads the mean phase advance where
    ``|ψ| > amp_floor`` (the ordered bulk).  For the CGL field ``Ω = −λ``: the
    chiral coupling makes the order parameter rotate, an intrinsic angular
    frequency — spin.  ``λ = 0`` gives ``Ω = 0`` (parity holds).
    """
    theta0 = np.angle(psi)
    p = psi.copy()
    for _ in range(n_probe):
        p = step_chiral(p, chiral=chiral, dt=dt)
    dtheta = np.angle(np.exp(1j * (np.angle(p) - theta0)))
    mask = np.abs(psi) > amp_floor
    if not mask.any():
        return 0.0
    return float(dtheta[mask].mean() / (dt * n_probe))


def phase_sector_labels(psi: np.ndarray, n_sectors: int = 3) -> np.ndarray:
    """Sector labels by phase band — the phase circle cut into ``n_sectors``.

    The complex field's phase ``arg ψ`` partitioned into equal arcs; the
    domain walls are the phase-band boundaries, so the dimensional census
    applies and its 0D junctions are phase vortices whose sign is chiral.
    """
    theta = np.angle(psi) % (2.0 * np.pi)
    return np.floor(theta / (2.0 * np.pi / n_sectors)).astype(int)
