"""Lattice gauge connection on the Ψ∈ℂ³ sector field.

This module implements the *core principle* of the URP gauge derivation
(`Docs/URP_Gauge_Symmetries_Derivation`, §2–3): when a multi-component field
``ψ(x) ∈ ℂ^N`` carries a local internal orientation, a **gauge connection**
``U_μ(x) ∈ U(N)`` on the lattice links is the minimal structure that keeps the
inter-site coherence (the integration term ΔI) invariant under *local* internal
rotations ``ψ(x) → g(x)ψ(x)``. The curvature of that connection — measured by
the Wilson plaquette — is the residual "coherence stress" that the theory
identifies with ``Tr(F_{μν}F^{μν})``.

The N components are exactly the colour sectors R/G/B of
:mod:`project_genesis.multiphase` (a sector-membership field deep inside a
domain is a basis vector ``(1,0,0)`` etc.; near a wall it is a superposition).

What this demonstrates (and what it does not):

- **Does**: that the *covariant* coherence ``Σ Re[ψ†(x) U_μ(x) ψ(x+μ̂)]`` is
  exactly gauge-invariant while the naive coherence ``Σ Re[ψ†(x) ψ(x+μ̂)]`` is
  not; that a pure-gauge (flat) connection has zero curvature; and that
  curvature measures the failure of parallel transport.
- **Does not**: evolve the Yang–Mills dynamics, or derive confinement /
  asymptotic freedom from S-maximization. Those are a full lattice-gauge
  simulation — future work. This is the connection-as-coherence-restoration
  principle, made executable.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# U(N) / SU(N) group elements
# ----------------------------------------------------------------------


def random_unitary(
    rng: np.random.Generator,
    n: int,
    size: tuple[int, ...] = (),
    *,
    special: bool = True,
    scale: float = 1.0,
) -> np.ndarray:
    """Random U(N) (or SU(N)) matrices via ``exp(i·H)`` of a Hermitian ``H``.

    Parameters
    ----------
    n : int
        Matrix dimension (1 = U(1), 2 = SU(2), 3 = SU(3) colour).
    size : tuple
        Leading (lattice) shape; output is ``(*size, n, n)``.
    special : bool
        If True, make ``H`` traceless so ``det U = 1`` (the SU(N) case).
    scale : float
        Multiplies ``H`` — ``scale → 0`` gives matrices near the identity.
    """
    shape = (*size, n, n)
    a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    herm = 0.5 * (a + np.conjugate(np.swapaxes(a, -1, -2)))
    if special and n > 1:
        trace = np.trace(herm, axis1=-2, axis2=-1)
        herm = herm - (trace / n)[..., None, None] * np.eye(n)
    return _expm_iH(scale * herm)


def _expm_iH(herm: np.ndarray) -> np.ndarray:
    """Matrix exponential ``exp(i·H)`` for stacked Hermitian ``H`` via eigh."""
    w, v = np.linalg.eigh(herm)  # H = v diag(w) v†, batched over leading dims
    phase = np.exp(1j * w)
    return (v * phase[..., None, :]) @ np.conjugate(np.swapaxes(v, -1, -2))


def is_unitary(u: np.ndarray, tol: float = 1e-9) -> bool:
    """Check that every matrix in a stack is unitary."""
    n = u.shape[-1]
    prod = u @ np.conjugate(np.swapaxes(u, -1, -2))
    return bool(np.allclose(prod, np.eye(n), atol=tol))


# ----------------------------------------------------------------------
# Field initialisation from sector membership
# ----------------------------------------------------------------------


def sector_field_to_psi(fields: np.ndarray) -> np.ndarray:
    """Turn a multiphase sector field ``(N, *spatial)`` into ``ψ`` ``(*spatial, N)``.

    Each voxel's component vector is L2-normalised, so deep inside a domain ψ is
    (close to) a colour basis vector and near a wall it is a superposition — the
    sector-membership field Ψ the gauge derivation posits.
    """
    psi = np.moveaxis(fields, 0, -1).astype(np.complex128)
    norm = np.linalg.norm(psi, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return psi / norm


def random_psi(rng: np.random.Generator, n: int, spatial: tuple[int, ...]) -> np.ndarray:
    """A random normalised ``ℂ^N`` field of shape ``(*spatial, n)``."""
    psi = rng.standard_normal((*spatial, n)) + 1j * rng.standard_normal((*spatial, n))
    return psi / np.linalg.norm(psi, axis=-1, keepdims=True)


def identity_links(n: int, spatial: tuple[int, ...]) -> np.ndarray:
    """Trivial connection: ``U_μ(x) = 1`` on every link. Shape ``(ndim, *spatial, n, n)``."""
    ndim = len(spatial)
    eye = np.broadcast_to(np.eye(n, dtype=np.complex128), (*spatial, n, n))
    return np.broadcast_to(eye, (ndim, *spatial, n, n)).copy()


# ----------------------------------------------------------------------
# Gauge transformations
# ----------------------------------------------------------------------


def apply_gauge_transform(
    psi: np.ndarray,
    links: np.ndarray,
    g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a local gauge transform ``g(x)``:  ψ→gψ,  U_μ→g(x)U_μ g(x+μ̂)†.

    Returns the transformed ``(psi, links)``.
    """
    psi_new = np.einsum("...ij,...j->...i", g, psi)
    g_dag = np.conjugate(np.swapaxes(g, -1, -2))
    ndim = links.shape[0]
    links_new = np.empty_like(links)
    for mu in range(ndim):
        g_shift_dag = np.roll(g_dag, shift=-1, axis=mu)  # g(x+μ̂)†
        links_new[mu] = g @ links[mu] @ g_shift_dag
    return psi_new, links_new


def pure_gauge_links(g: np.ndarray, ndim: int) -> np.ndarray:
    """Build a *flat* connection ``U_μ(x) = g(x) g(x+μ̂)†`` (zero curvature)."""
    g_dag = np.conjugate(np.swapaxes(g, -1, -2))
    spatial = g.shape[:-2]
    n = g.shape[-1]
    links = np.empty((ndim, *spatial, n, n), dtype=np.complex128)
    for mu in range(ndim):
        links[mu] = g @ np.roll(g_dag, shift=-1, axis=mu)
    return links


# ----------------------------------------------------------------------
# Coherence (integration) measures
# ----------------------------------------------------------------------


def naive_coherence(psi: np.ndarray) -> float:
    """``Σ_{x,μ} Re[ψ†(x) ψ(x+μ̂)]`` — the inter-site coherence with no connection.

    This is *not* gauge-invariant: a local rotation scrambles the relative
    orientation of neighbouring sites.
    """
    ndim = psi.ndim - 1
    total = 0.0
    for mu in range(ndim):
        shifted = np.roll(psi, shift=-1, axis=mu)
        total += float(np.sum(np.real(np.conjugate(psi) * shifted)))
    return total


def covariant_coherence(psi: np.ndarray, links: np.ndarray) -> float:
    """``Σ_{x,μ} Re[ψ†(x) U_μ(x) ψ(x+μ̂)]`` — coherence parallel-transported by U.

    This *is* exactly gauge-invariant: ``U_μ`` carries ψ(x+μ̂) back into the
    frame of ψ(x) before they are compared. It is the integration term that the
    gauge connection restores.
    """
    ndim = links.shape[0]
    total = 0.0
    for mu in range(ndim):
        shifted = np.roll(psi, shift=-1, axis=mu)
        transported = np.einsum("...ij,...j->...i", links[mu], shifted)
        total += float(np.sum(np.real(np.conjugate(psi) * transported)))
    return total


# ----------------------------------------------------------------------
# Curvature = coherence stress
# ----------------------------------------------------------------------


def plaquette(links: np.ndarray, mu: int, nu: int) -> np.ndarray:
    """The Wilson plaquette ``U_μ(x) U_ν(x+μ̂) U_μ(x+ν̂)† U_ν(x)†`` at every site."""
    u_mu = links[mu]
    u_nu = links[nu]
    u_nu_shift_mu = np.roll(u_nu, shift=-1, axis=mu)      # U_ν(x+μ̂)
    u_mu_shift_nu = np.roll(u_mu, shift=-1, axis=nu)      # U_μ(x+ν̂)
    dag = lambda m: np.conjugate(np.swapaxes(m, -1, -2))
    return u_mu @ u_nu_shift_mu @ dag(u_mu_shift_nu) @ dag(u_nu)


def wilson_action(links: np.ndarray) -> float:
    """``Σ_{x,μ<ν} Re Tr(1 − P_{μν}(x))`` — total curvature / coherence stress.

    Zero for a flat (pure-gauge) connection; positive when parallel transport
    around a loop fails to return to the identity. This is the lattice form of
    the ``Tr(F_{μν}F^{μν})`` term the derivation reads as coherence stress.
    """
    ndim = links.shape[0]
    n = links.shape[-1]
    total = 0.0
    for mu in range(ndim):
        for nu in range(mu + 1, ndim):
            p = plaquette(links, mu, nu)
            tr = np.trace(p, axis1=-2, axis2=-1)
            total += float(np.sum(np.real(n - tr)))
    return total
