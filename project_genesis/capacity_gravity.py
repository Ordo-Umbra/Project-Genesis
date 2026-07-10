"""Capacity as gravity: the screened attraction the κ field mediates.

The dynamical capacity field obeys (``multiphase.step_multiphase_kappa``)

    ∂_t κ = D·∇²κ + r·(κ₀ − κ) − c·load·κ ,

which is the gradient flow ``∂_t κ = −δF/δκ`` of the **capacity free energy**

    F[κ] = ∫ [ (D/2)|∇κ|² + (r/2)(κ − κ₀)² + (c/2)·load·κ² ] .

That single fact makes κ behave like gravity *in the general sense*: a
concentration of ``load`` (distinction — the field's "mass") digs a well in κ,
and because F is a genuine energy, two such masses lower it by drawing
together.  Linearising about κ₀, the deficit ``δκ = κ₀ − κ`` obeys a screened
(Yukawa) equation ``∇²δκ = (r/D)·δκ`` away from the source, so κ carries a
**screening length**

    ξ_κ = √(D / r) ,

set by the ratio of capacity diffusion ``D`` to recovery rate ``r``.  The
mediated force is therefore short-ranged — a *massive*-graviton analogue whose
range is the recovery length; slow recovery (small ``r``) is long-ranged,
fast recovery is heavily screened.  This module provides the pieces to measure
that: build a load "mass" distribution, relax κ to its steady state, evaluate
F, and read the range of the resulting attraction.
"""

from __future__ import annotations

import numpy as np

from .multiphase import periodic_laplacian


def screening_length(kappa_diffusion: float, kappa_recovery: float) -> float:
    """The capacity screening length ``ξ_κ = √(D/r)`` — the range of κ-gravity."""
    if kappa_recovery <= 0.0:
        return float("inf")
    return float(np.sqrt(kappa_diffusion / kappa_recovery))


def gaussian_load(shape, centers, width: float, amplitude: float) -> np.ndarray:
    """A ``load`` field of Gaussian "masses" at ``centers`` (periodic distance).

    ``centers`` is a single center tuple or a list of them; every blob has the
    same ``width`` and ``amplitude``.  The load is the field's distinction
    density — the source that digs κ-wells.
    """
    shape = tuple(int(s) for s in shape)
    if len(centers) and np.isscalar(centers[0]):
        centers = [centers]
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    out = np.zeros(shape, dtype=float)
    for center in centers:
        r2 = sum((((g - c + s / 2) % s) - s / 2) ** 2
                 for g, c, s in zip(grids, center, shape))
        out += amplitude * np.exp(-r2 / (2.0 * width ** 2))
    return out


def relax_capacity(load: np.ndarray, *, kappa_diffusion: float = 1.0,
                   kappa_recovery: float = 0.05, kappa_consumption: float = 2.0,
                   kappa_baseline: float = 1.0, dt: float = 0.1,
                   max_iters: int = 8000, tol: float = 1e-8) -> np.ndarray:
    """Steady-state capacity κ for a fixed ``load``, by relaxing the κ flow.

    Iterates the exact ``multiphase`` capacity update with the load held fixed
    (two rigid masses) until the field stops moving.  Returns the relaxed κ.
    """
    k0 = float(kappa_baseline)
    k = np.full(load.shape, k0, dtype=float)
    src = kappa_consumption * load
    for _ in range(max_iters):
        upd = dt * (kappa_diffusion * periodic_laplacian(k)
                    + kappa_recovery * (k0 - k) - src * k)
        k += upd
        np.clip(k, 0.0, k0, out=k)
        if np.max(np.abs(upd)) < tol:
            break
    return k


def capacity_free_energy(kappa: np.ndarray, load: np.ndarray, *,
                         kappa_diffusion: float = 1.0,
                         kappa_recovery: float = 0.05,
                         kappa_consumption: float = 2.0,
                         kappa_baseline: float = 1.0) -> float:
    """The capacity free energy ``F[κ]`` — the Lyapunov energy the κ flow descends.

    ``F = Σ_x [ (D/2)|∇κ|² + (r/2)(κ−κ₀)² + (c/2)·load·κ² ]``; the κ dynamics is
    exactly ``∂_t κ = −δF/δκ``, so ``F`` is the energy whose separation
    dependence *is* the interaction potential between load masses.
    """
    gsq = np.zeros_like(kappa)
    for ax in range(kappa.ndim):
        g = 0.5 * (np.roll(kappa, -1, ax) - np.roll(kappa, 1, ax))
        gsq += g * g
    dens = (0.5 * kappa_diffusion * gsq
            + 0.5 * kappa_recovery * (kappa - kappa_baseline) ** 2
            + 0.5 * kappa_consumption * load * kappa * kappa)
    return float(np.sum(dens))


def interaction_potential(separation: float, *, shape, width: float,
                          amplitude: float, axis: int = 0,
                          reference: float | None = None, **kappa_kw) -> float:
    """κ-mediated interaction energy ``V(r) = F(r) − F(reference)`` of two masses.

    Places two equal Gaussian masses ``separation`` apart along ``axis``,
    relaxes κ, and returns the capacity free energy relative to a widely
    separated reference (default: half the box along ``axis``).
    """
    shape = tuple(int(s) for s in shape)
    center = [s // 2 for s in shape]

    def _F(sep):
        c1, c2 = list(center), list(center)
        c1[axis] = center[axis] - sep / 2.0
        c2[axis] = center[axis] + sep / 2.0
        load = gaussian_load(shape, [tuple(c1), tuple(c2)], width, amplitude)
        kappa = relax_capacity(load, **kappa_kw)
        return capacity_free_energy(kappa, load, **{
            k: v for k, v in kappa_kw.items()
            if k in ("kappa_diffusion", "kappa_recovery",
                     "kappa_consumption", "kappa_baseline")})

    ref = shape[axis] / 2.0 if reference is None else reference
    return _F(separation) - _F(ref)


def fit_yukawa_range(radii, potentials) -> dict:
    """Extract the range ``ξ`` of a 3-D Yukawa ``V(r) = −A·e^{−r/ξ}/r``.

    Linearises as ``ln(|V|·r) = ln A − r/ξ`` and fits a line; the slope gives
    ``ξ = −1/slope``.  Uses only points with ``V < 0`` (the attractive well).
    Returns the range, amplitude, and the number of points used.
    """
    r = np.asarray(radii, dtype=float)
    v = np.asarray(potentials, dtype=float)
    mask = v < 0
    if mask.sum() < 2:
        raise ValueError("need at least two attractive (V<0) points")
    r, y = r[mask], np.log(np.abs(v[mask]) * r[mask])
    slope, intercept = np.polyfit(r, y, 1)
    return {"xi": float(-1.0 / slope), "amplitude": float(np.exp(intercept)),
            "n": int(mask.sum())}


def well_range(load_center_line) -> float:
    """Screening length read from a single κ-well radial deficit profile.

    ``load_center_line`` is the deficit ``κ₀ − κ`` sampled radially outward
    from a single mass; a 3-D screened field falls as ``δκ ∝ e^{−r/ξ}/r``, so
    ``ln(δκ·r)`` is linear with slope ``−1/ξ``.
    """
    d = np.asarray(load_center_line, dtype=float)
    dist = np.arange(d.size, dtype=float)
    # fit the exponential shoulder: past the finite-width source, before the
    # periodic image contaminates the far tail (roughly the inner half-box)
    upper = max(6.0, 0.55 * d.size)
    sel = (dist >= 3) & (dist <= upper) & (d > 1e-6)
    if sel.sum() < 2:
        raise ValueError("profile too short or flat to fit")
    slope = np.polyfit(dist[sel], np.log(d[sel] * np.maximum(dist[sel], 1e-9)), 1)[0]
    return float(-1.0 / slope)
