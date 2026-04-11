"""Numba-accelerated kernels for field evolution."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def laplacian_3d(field: np.ndarray, out: np.ndarray) -> None:
    """Compute the discrete Laplacian of a 3-D scalar field with periodic BCs."""
    nx, ny, nz = field.shape
    for i in prange(nx):
        ip = (i + 1) % nx
        im = (i - 1) % nx
        for j in range(ny):
            jp = (j + 1) % ny
            jm = (j - 1) % ny
            for k in range(nz):
                kp = (k + 1) % nz
                km = (k - 1) % nz
                out[i, j, k] = (
                    field[ip, j, k]
                    + field[im, j, k]
                    + field[i, jp, k]
                    + field[i, jm, k]
                    + field[i, j, kp]
                    + field[i, j, km]
                    - 6.0 * field[i, j, k]
                )


@njit(parallel=True, cache=True)
def gradient_squared_3d(field: np.ndarray, out: np.ndarray) -> None:
    """Compute |∇φ|² using central differences with periodic BCs."""
    nx, ny, nz = field.shape
    for i in prange(nx):
        ip = (i + 1) % nx
        im = (i - 1) % nx
        for j in range(ny):
            jp = (j + 1) % ny
            jm = (j - 1) % ny
            for k in range(nz):
                kp = (k + 1) % nz
                km = (k - 1) % nz
                gx = (field[ip, j, k] - field[im, j, k]) * 0.5
                gy = (field[i, jp, k] - field[i, jm, k]) * 0.5
                gz = (field[i, j, kp] - field[i, j, km]) * 0.5
                out[i, j, k] = gx * gx + gy * gy + gz * gz


@njit(parallel=True, cache=True)
def evolve_field_kernel(
    field: np.ndarray,
    laplacian: np.ndarray,
    grad_sq: np.ndarray,
    beta: float,
    gravity: float,
    dt: float,
    out: np.ndarray,
) -> None:
    """Apply the URP field update rule: φ += (∇²φ + β|∇φ|² - Gφ) · dt."""
    nx, ny, nz = field.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = field[i, j, k] + (
                    laplacian[i, j, k]
                    + beta * grad_sq[i, j, k]
                    - gravity * field[i, j, k]
                ) * dt


def jit_step(
    field: np.ndarray,
    beta: float,
    gravity: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform one JIT-accelerated field evolution step.

    Returns (new_field, laplacian, gradient_squared).
    """
    lap = np.empty_like(field)
    gsq = np.empty_like(field)
    out = np.empty_like(field)
    laplacian_3d(field, lap)
    gradient_squared_3d(field, gsq)
    evolve_field_kernel(field, lap, gsq, beta, gravity, dt, out)
    return out, lap, gsq
