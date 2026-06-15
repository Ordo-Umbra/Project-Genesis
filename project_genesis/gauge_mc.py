"""Monte-Carlo lattice gauge theory: confinement via the Wilson-loop area law.

The Yang–Mills *dynamics* module (`gauge.flow_step`) is deterministic gradient
flow. Confinement, by contrast, is a property of the finite-coupling *ensemble*:
the static quark–antiquark potential rises linearly with separation, which on
the lattice shows up as an **area law** for Wilson loops,
``⟨W(R,T)⟩ ~ exp(−σ·R·T)`` with string tension ``σ``. Measuring it requires
sampling configurations with the Boltzmann weight ``exp(−S_W)``.

This module provides a checkerboard Metropolis sampler for the Wilson action

    S_W = β · Σ_plaq (1 − (1/N) Re Tr U_plaq),

planar Wilson-loop and plaquette observables, and Creutz ratios for extracting
the string tension. ``β`` here is the *lattice coupling* (``2N/g²``), not the URP
β.

Correctness is anchored on the exactly-solvable 2-D case: for 2-D SU(2) the
single-plaquette average is ``I_2(β)/I_1(β)`` (modified Bessel functions) and the
Wilson loop is exactly ``⟨P⟩^Area`` — so the sampler can be validated, not just
trusted.
"""

from __future__ import annotations

import numpy as np

from .gauge import _dagger, gauge_staple, plaquette, random_unitary


# ----------------------------------------------------------------------
# Observables
# ----------------------------------------------------------------------


def mean_plaquette(links: np.ndarray) -> float:
    """``⟨(1/N) Re Tr U_plaq⟩`` averaged over all plaquettes (1 at β→∞, 0 at β→0)."""
    ndim = links.shape[0]
    n = links.shape[-1]
    vals = []
    for mu in range(ndim):
        for nu in range(mu + 1, ndim):
            p = plaquette(links, mu, nu)
            vals.append(np.mean(np.real(np.trace(p, axis1=-2, axis2=-1))) / n)
    return float(np.mean(vals))


def wilson_loop(links: np.ndarray, r: int, t: int, mu: int = 0, nu: int = 1) -> float:
    """``⟨(1/N) Re Tr W⟩`` for a planar ``r×t`` rectangular loop in the (μ,ν) plane.

    The loop at site x runs: r links in +μ, t links in +ν, r back in −μ, t back
    in −ν, parallel-transported around the rectangle. Averaged over all sites.
    """
    n = links.shape[-1]
    spatial = links.shape[1:-2]
    eye = np.broadcast_to(np.eye(n, dtype=np.complex128), (*spatial, n, n))
    w = eye.copy()

    # bottom edge: r links along +μ
    pos_shift = [0] * len(spatial)
    for _ in range(r):
        u = _shifted(links[mu], pos_shift)
        w = w @ u
        pos_shift[mu] += 1
    # right edge: t links along +ν
    for _ in range(t):
        u = _shifted(links[nu], pos_shift)
        w = w @ u
        pos_shift[nu] += 1
    # top edge: r links along −μ (daggered)
    for _ in range(r):
        pos_shift[mu] -= 1
        u = _shifted(links[mu], pos_shift)
        w = w @ _dagger(u)
    # left edge: t links along −ν (daggered)
    for _ in range(t):
        pos_shift[nu] -= 1
        u = _shifted(links[nu], pos_shift)
        w = w @ _dagger(u)

    return float(np.mean(np.real(np.trace(w, axis1=-2, axis2=-1))) / n)


def _shifted(field: np.ndarray, shift: list[int]) -> np.ndarray:
    """Field with link at site x replaced by the link at x+shift (periodic)."""
    out = field
    for axis, s in enumerate(shift):
        if s:
            out = np.roll(out, -s, axis=axis)
    return out


def creutz_ratio(w: dict[tuple[int, int], float], r: int, t: int) -> float:
    """Creutz ratio ``χ(r,t) = −log[W(r,t)W(r−1,t−1) / (W(r,t−1)W(r−1,t))]``.

    Perimeter and corner contributions cancel, so ``χ → σ`` (the string tension)
    for large loops. ``w`` maps ``(r,t) → ⟨W⟩``.
    """
    num = w[(r, t)] * w[(r - 1, t - 1)]
    den = w[(r, t - 1)] * w[(r - 1, t)]
    if num <= 0 or den <= 0:
        return float("nan")
    return -float(np.log(num / den))


# ----------------------------------------------------------------------
# Checkerboard Metropolis
# ----------------------------------------------------------------------


def _parity_mask(spatial: tuple[int, ...], parity: int) -> np.ndarray:
    """Boolean lattice mask selecting sites whose coordinate sum has given parity."""
    grids = np.indices(spatial)
    return (grids.sum(axis=0) % 2) == parity


def metropolis_sweep(
    links: np.ndarray,
    beta: float,
    rng: np.random.Generator,
    *,
    n_hits: int = 2,
    step: float = 0.3,
) -> float:
    """One checkerboard Metropolis sweep over all links. Returns the accept rate.

    For each direction and site-parity (so updated links are conditionally
    independent given the rest), propose ``U' = R·U`` with ``R`` a random SU(N)
    near the identity, and accept with ``min(1, exp(−ΔS))``.
    """
    ndim = links.shape[0]
    n = links.shape[-1]
    spatial = links.shape[1:-2]
    accepts = 0
    trials = 0
    for mu in range(ndim):
        for parity in (0, 1):
            # Recompute the staple per parity: a parity-p link's staple depends on
            # opposite-parity links of the same direction, which the previous
            # sub-sweep just changed — using a stale staple breaks detailed balance.
            staple = gauge_staple(links, mu)
            mask = _parity_mask(spatial, parity)
            for _ in range(n_hits):
                u_old = links[mu]
                r = random_unitary(rng, n, spatial, special=(n > 1), scale=step)
                u_new = r @ u_old
                # ΔS = -(β/N) Re Tr[(U_new - U_old) A]
                d = (u_new - u_old) @ staple
                dS = -(beta / n) * np.real(np.trace(d, axis1=-2, axis2=-1))
                accept = (rng.random(spatial) < np.exp(-dS)) & mask
                links[mu] = np.where(accept[..., None, None], u_new, u_old)
                accepts += int(accept.sum())
                trials += int(mask.sum())
    return accepts / max(1, trials)


def hot_start(rng: np.random.Generator, n: int, spatial: tuple[int, ...]) -> np.ndarray:
    """Random (hot) link configuration of shape ``(ndim, *spatial, n, n)``."""
    ndim = len(spatial)
    return random_unitary(rng, n, (ndim, *spatial), special=(n > 1), scale=np.pi)


# ----------------------------------------------------------------------
# Exact 2-D checks
# ----------------------------------------------------------------------


def exact_plaquette_su2_2d(beta: float) -> float:
    """Exact 2-D SU(2) single-plaquette average ``⟨(1/2)Tr U_p⟩ = I_2(β)/I_1(β)``."""
    from scipy.special import iv

    return float(iv(2, beta) / iv(1, beta))
