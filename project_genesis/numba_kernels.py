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


# ------------------------------------------------------------------
# Coherence potential: Poisson solver & gradient dot product
# ------------------------------------------------------------------


@njit(parallel=True, cache=True)
def solve_poisson_jacobi(
    rho: np.ndarray,
    out: np.ndarray,
    iterations: int = 30,
) -> None:
    """Solve ∇²V = ρ via Jacobi iteration with periodic BCs.

    Parameters
    ----------
    rho : 3-D array
        Source term (proportional to the scalar field φ).
    out : 3-D array
        Output potential V.  Should be initialised to zeros by the caller.
    iterations : int
        Number of Jacobi sweeps.
    """
    nx, ny, nz = rho.shape
    tmp = np.zeros_like(out)

    # Alternate between *out* and *tmp* as read / write buffers.
    for iteration in range(iterations):
        if iteration % 2 == 0:
            src = out
            dst = tmp
        else:
            src = tmp
            dst = out

        for i in prange(nx):
            ip = (i + 1) % nx
            im = (i - 1) % nx
            for j in range(ny):
                jp = (j + 1) % ny
                jm = (j - 1) % ny
                for k in range(nz):
                    kp = (k + 1) % nz
                    km = (k - 1) % nz
                    dst[i, j, k] = (
                        src[ip, j, k]
                        + src[im, j, k]
                        + src[i, jp, k]
                        + src[i, jm, k]
                        + src[i, j, kp]
                        + src[i, j, km]
                        - rho[i, j, k]
                    ) / 6.0

    # If the last write went to *tmp*, copy back to *out*.
    if iterations > 0 and (iterations - 1) % 2 == 0:
        for i in prange(nx):
            for j in range(ny):
                for k in range(nz):
                    out[i, j, k] = tmp[i, j, k]


@njit(parallel=True, cache=True)
def gradient_dot_product_3d(
    field_a: np.ndarray,
    field_b: np.ndarray,
    out: np.ndarray,
) -> None:
    """Compute ∇A·∇B using central differences with periodic BCs."""
    nx, ny, nz = field_a.shape
    for i in prange(nx):
        ip = (i + 1) % nx
        im = (i - 1) % nx
        for j in range(ny):
            jp = (j + 1) % ny
            jm = (j - 1) % ny
            for k in range(nz):
                kp = (k + 1) % nz
                km = (k - 1) % nz
                ax = (field_a[ip, j, k] - field_a[im, j, k]) * 0.5
                ay = (field_a[i, jp, k] - field_a[i, jm, k]) * 0.5
                az = (field_a[i, j, kp] - field_a[i, j, km]) * 0.5
                bx = (field_b[ip, j, k] - field_b[im, j, k]) * 0.5
                by = (field_b[i, jp, k] - field_b[i, jm, k]) * 0.5
                bz = (field_b[i, j, kp] - field_b[i, j, km]) * 0.5
                out[i, j, k] = ax * bx + ay * by + az * bz


# ------------------------------------------------------------------
# Nonlocal integration functional: correlation kernel
# ------------------------------------------------------------------


@njit(parallel=True, cache=True)
def correlation_kernel_3d(
    field: np.ndarray,
    out: np.ndarray,
    radius: int = 2,
    decay: float = 1.0,
) -> None:
    """Compute the functional derivative δI/δφ of a nonlocal integration functional.

    For each voxel x the output is:

        out[x] = 2 · Σ_{x' in radius} exp(-decay·|x-x'|) · φ(x')

    which approximates δI/δφ for I[φ] = ∫∫ K(x,x') φ(x) φ(x') dx dx'.
    Uses periodic boundary conditions.
    """
    nx, ny, nz = field.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                total = 0.0
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        for dk in range(-radius, radius + 1):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            dist = np.sqrt(float(di * di + dj * dj + dk * dk))
                            if dist > radius:
                                continue
                            ni = (i + di) % nx
                            nj = (j + dj) % ny
                            nk = (k + dk) % nz
                            total += np.exp(-decay * dist) * field[ni, nj, nk]
                out[i, j, k] = 2.0 * total


# ------------------------------------------------------------------
# Full URP evolution kernel (v2)
# ------------------------------------------------------------------


@njit(parallel=True, cache=True)
def evolve_field_kernel_v2(
    field: np.ndarray,
    laplacian: np.ndarray,
    grad_sq: np.ndarray,
    coherence_advection: np.ndarray,
    integration_term: np.ndarray,
    beta: float,
    integration_weight: float,
    dt: float,
    out: np.ndarray,
) -> None:
    """Apply the full URP field update:

    φ += (∇²φ + β|∇φ|² + G·∇V·∇φ + w_I·δI/δφ) · dt

    ``coherence_advection`` should already include the gravity factor G.
    """
    nx, ny, nz = field.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = field[i, j, k] + (
                    laplacian[i, j, k]
                    + beta * grad_sq[i, j, k]
                    + coherence_advection[i, j, k]
                    + integration_weight * integration_term[i, j, k]
                ) * dt


def jit_step_v2(
    field: np.ndarray,
    beta: float,
    gravity: float,
    dt: float,
    *,
    poisson_iterations: int = 30,
    integration_radius: int = 2,
    integration_decay: float = 1.0,
    integration_weight: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform one full-URP evolution step with coherence potential & integration.

    Returns (new_field, laplacian, gradient_squared).
    """
    lap = np.empty_like(field)
    gsq = np.empty_like(field)
    out = np.empty_like(field)

    laplacian_3d(field, lap)
    gradient_squared_3d(field, gsq)

    # Coherence potential: solve ∇²V = φ  then compute G·∇V·∇φ
    potential = np.zeros_like(field)
    solve_poisson_jacobi(field, potential, poisson_iterations)

    grad_dot = np.empty_like(field)
    gradient_dot_product_3d(potential, field, grad_dot)
    # Multiply by gravity to form the coherence advection term G·∇V·∇φ
    coherence_adv = gravity * grad_dot

    # Integration functional derivative δI/δφ
    integ = np.empty_like(field)
    correlation_kernel_3d(field, integ, integration_radius, integration_decay)

    evolve_field_kernel_v2(
        field, lap, gsq, coherence_adv, integ,
        beta, integration_weight, dt, out,
    )
    return out, lap, gsq
