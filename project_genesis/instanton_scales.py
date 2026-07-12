"""The physical instanton scale, and the matched-scale bridge instrument.

The one-κ obstruction (`n3_kappa_obstruction.py`) closed the naive route to a
matched `κ_I = κ_II`: the two acts' topological sectors renormalise in opposite
directions under the flow, so any bare `q(x), e(x)` estimator reads scale-tied
numbers that drift apart.  It also named the concrete requirement for a genuine
bridge: **match the physical instanton scales first** — a renormalisation
condition — and only then compare the coherent fractions.  This module is that
instrument.

Three pieces:

1. **Instanton sizes from peak heights.**  Both sectors' classical instantons
   have exact, normalised charge-density profiles, so a lump's *size* can be
   read off its *peak density*:

   - 4-D SU(N) (BPST):  ``q(x) = (6/π²)·ρ⁴/(|x−x₀|²+ρ²)⁴``  ⇒  ``ρ = (6/(π²·q_peak))^{1/4}``
   - 2-D CP^(N−1):      ``q(x) = (1/π)·ρ²/(|x−x₀|²+ρ²)²``   ⇒  ``ρ = 1/√(π·q_peak)``

   The estimator finds local maxima of ``|q(x)|`` above a height threshold and
   converts each peak to a size — the standard peak-height method.  On a coarse
   lattice the clover/geometric densities carry a multiplicative ``Z ≲ 1``, so
   sizes are systematically over-read by ``Z^{-1/4}`` (4-D) / ``Z^{-1/2}``
   (2-D); after smoothing ``Z → 1`` and the estimate becomes clean.

2. **A genuine CP gradient flow.**  Cooling (`topological_charge.cool`) snaps
   to local minimisers with no controlled clock.  The projected gradient flow
   of the lattice CP action ``S = Σ(1−|z̄·z'|²)``,

       dz/dτ = P_z( M(x)·z ),   M(x) = Σ_{y~x} z(y) z(y)†,   P_z = 1 − z z†,

   is the 2-D analogue of the Wilson flow: it descends the action smoothly in a
   flow time τ with units of lattice-spacing².  Linearised about a smooth field
   it is the lattice heat equation with unit diffusion constant — and rather
   than assume that, :func:`cp_flow_diffusion_constant` *measures* it from the
   decay rate of a transverse plane wave.

3. **The matched coordinate and the comparison.**  A flow at time ``t`` smooths
   over the heat-kernel radius ``λ_d(t) = √(2·d·D·t)`` (Lüscher's ``√(8t)`` in
   4-D).  The renormalisation condition is dimensionless and sector-local:
   observe each theory at the depth where the smoothing radius is the same
   *fraction of its own mean instanton size*,

       s(t) := λ(t) / ρ̄(t)  matched across sectors.

   :func:`matched_comparison` interpolates both sectors' coherent fractions
   onto a common grid of ``s`` and reports the gap — and the crossing, if the
   curves meet.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "local_maxima",
    "instanton_sizes_cp",
    "instanton_sizes_su",
    "cp_flow_step",
    "cp_gradient_flow",
    "cp_flow_diffusion_constant",
    "smoothing_radius",
    "matched_comparison",
]


# --------------------------------------------------------------------------
# 1. Instanton sizes from charge-density peaks
# --------------------------------------------------------------------------

def local_maxima(density: np.ndarray, min_frac: float = 0.25,
                 min_height: float = 0.0) -> list[tuple[tuple[int, ...], float]]:
    """Strict local maxima of ``density`` on the periodic lattice, above threshold.

    A site is a peak when its value exceeds all ``2·d`` axis neighbours
    (periodic wrap) and is at least ``max(min_frac·max(density), min_height)``.
    Returns ``[(index_tuple, value), …]`` sorted by descending value.
    """
    d = np.asarray(density, dtype=float)
    if d.size == 0 or not np.isfinite(d).any():
        return []
    thresh = max(min_frac * float(np.nanmax(d)), min_height)
    mask = np.ones(d.shape, dtype=bool)
    for axis in range(d.ndim):
        for shift in (-1, 1):
            mask &= d > np.roll(d, shift, axis=axis)
    mask &= d >= thresh
    peaks = [(tuple(int(i) for i in idx), float(d[tuple(idx)]))
             for idx in np.argwhere(mask)]
    return sorted(peaks, key=lambda p: -p[1])


def instanton_sizes_cp(charge_density: np.ndarray, min_frac: float = 0.25,
                       min_height: float = 0.0) -> np.ndarray:
    """Instanton sizes ``ρ`` of a 2-D CP^(N−1) charge density, from peak heights.

    Each local maximum of ``|q(x)|`` above threshold is converted by the exact
    lump profile: ``q_peak = 1/(π ρ²)`` ⇒ ``ρ = (π·q_peak)^{−1/2}``.  Returns a
    (possibly empty) array of sizes in lattice units.
    """
    q = np.abs(np.asarray(charge_density, dtype=float))
    peaks = local_maxima(q, min_frac=min_frac, min_height=min_height)
    return np.array([1.0 / np.sqrt(np.pi * h) for _, h in peaks if h > 0.0])


def instanton_sizes_su(charge_density: np.ndarray, min_frac: float = 0.25,
                       min_height: float = 0.0) -> np.ndarray:
    """Instanton sizes ``ρ`` of a 4-D SU(N) clover charge density, from peak heights.

    Each local maximum of ``|q(x)|`` above threshold is converted by the BPST
    profile: ``q_peak = 6/(π² ρ⁴)`` ⇒ ``ρ = (6/(π²·q_peak))^{1/4}``.  Returns a
    (possibly empty) array of sizes in lattice units.
    """
    q = np.abs(np.asarray(charge_density, dtype=float))
    peaks = local_maxima(q, min_frac=min_frac, min_height=min_height)
    return np.array([(6.0 / (np.pi ** 2 * h)) ** 0.25 for _, h in peaks if h > 0.0])


# --------------------------------------------------------------------------
# 2. The CP gradient flow (the 2-D analogue of the Wilson flow)
# --------------------------------------------------------------------------

def cp_flow_step(psi: np.ndarray, dt: float) -> np.ndarray:
    """One projected-gradient-flow step of the CP^(N−1) field, step ``dt``.

    ``−∂S/∂z̄(x) = M(x)·z(x)`` with ``M(x) = Σ_{y~x} z(y)z(y)†`` (the four
    axis neighbours), so descending the action means ascending ``z†Mz``.  The
    tangent-space update ``z ← z + dt·(Mz − (z†Mz)z)`` stays normalised to
    ``O(dt²)``; the explicit renormalisation removes the remainder.  Stable
    for ``dt ≲ 0.25`` (the lattice heat-equation bound in 2-D).
    """
    g = np.zeros_like(psi)
    for axis in (0, 1):
        for shift in (-1, 1):
            nb = np.roll(psi, shift, axis=axis)
            g += nb * np.sum(np.conj(nb) * psi, axis=-1, keepdims=True)
    radial = np.sum(np.conj(psi) * g, axis=-1, keepdims=True)
    z = psi + dt * (g - radial * psi)
    return z / np.linalg.norm(z, axis=-1, keepdims=True)


def cp_gradient_flow(psi: np.ndarray, *, t_max: float, dt: float = 0.1,
                     measure_every: int = 5,
                     min_frac: float = 0.25) -> tuple[np.ndarray, list[dict]]:
    """Integrate the CP flow to ``t_max``; return flowed field + trajectory.

    The trajectory records, every ``measure_every`` steps: the flow time ``t``,
    the action ``S``, the topological charge ``Q``, the coherent fraction
    ``κ̂ = Σ|q|/Σe``, and the mean instanton size ``rho_mean`` (``nan`` when no
    lump clears the peak threshold) with the surviving lump count ``n_lumps``.
    """
    from .topological_charge import (
        cp_action,
        cp_action_density,
        coherent_fraction,
        topological_charge_density,
    )

    z = psi.copy()
    n_steps = max(1, int(round(t_max / dt)))
    traj: list[dict] = []

    def record(t: float) -> None:
        qd = topological_charge_density(z)
        sizes = instanton_sizes_cp(qd, min_frac=min_frac)
        traj.append({
            "t": float(t),
            "S": cp_action(z),
            "Q": float(qd.sum()),
            "kappa": coherent_fraction(cp_action_density(z), qd),
            "rho_mean": float(sizes.mean()) if sizes.size else float("nan"),
            "n_lumps": int(sizes.size),
        })

    record(0.0)
    for step in range(1, n_steps + 1):
        z = cp_flow_step(z, dt)
        if step % measure_every == 0 or step == n_steps:
            record(step * dt)
    return z, traj


def cp_flow_diffusion_constant(size: int = 32, n_colors: int = 3,
                               dt: float = 0.1, n_steps: int = 40) -> float:
    """Measure the CP flow's diffusion constant ``D`` — do not assume it.

    A transverse plane wave ``z = z₀ + a·cos(kx)·v`` (``v ⊥ z₀``,
    ``k = 2π/size``) decays under the linearised flow at rate
    ``γ = 2(1−cos k)·D`` (the lattice heat kernel).  Evolving with
    :func:`cp_flow_step` and fitting ``ln A(τ)`` gives ``D``; the linearised
    prediction is exactly ``D = 1``, and the smoothing radius
    :func:`smoothing_radius` uses whatever this measurement returns.
    """
    k = 2.0 * np.pi / size
    z0 = np.zeros(n_colors, dtype=np.complex128)
    z0[0] = 1.0
    v = np.zeros(n_colors, dtype=np.complex128)
    v[1] = 1.0
    x = np.arange(size)
    wave = 0.05 * np.cos(k * x)[None, :]          # varies along axis 1
    z = np.broadcast_to(z0, (size, size, n_colors)).copy()
    z = z + wave[..., None] * v
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)

    def amplitude(field: np.ndarray) -> float:
        comp = field @ np.conj(v)                  # (size, size) transverse part
        return float(np.abs(2.0 * np.mean(comp * np.cos(k * x)[None, :])))

    times, amps = [0.0], [amplitude(z)]
    for step in range(1, n_steps + 1):
        z = cp_flow_step(z, dt)
        times.append(step * dt)
        amps.append(amplitude(z))
    slope = np.polyfit(times, np.log(np.maximum(amps, 1e-300)), 1)[0]
    return float(-slope / (2.0 * (1.0 - np.cos(k))))


def smoothing_radius(t, dim: int, diffusion: float = 1.0):
    """Heat-kernel smoothing radius ``λ_d(t) = √(2·d·D·t)`` of a flow at time ``t``.

    The rms radius of the ``d``-dimensional heat kernel — Lüscher's ``√(8t)``
    for the 4-D Wilson flow (``d=4, D=1``), ``√(4Dt)`` for the 2-D CP flow.
    Accepts scalars or arrays.
    """
    return np.sqrt(2.0 * dim * diffusion * np.asarray(t, dtype=float))


# --------------------------------------------------------------------------
# 3. The matched-scale comparison
# --------------------------------------------------------------------------

def matched_comparison(s_a, kappa_a, s_b, kappa_b, n_grid: int = 60) -> dict:
    """Compare two sectors' coherent fractions at matched scale ``s = λ/ρ̄``.

    ``s_a, kappa_a`` (and ``s_b, kappa_b``) are same-length samples of the
    matched coordinate and the coherent fraction along each sector's flow;
    non-finite pairs are dropped and each curve is sorted in ``s``.  Both are
    interpolated onto a common grid over the overlap of their ``s`` ranges.

    Returns a dict: ``overlap`` (bool) and, when true, ``s_grid``, ``kappa_a``,
    ``kappa_b``, ``gap`` (a−b), ``crossing`` (bool), ``s_star`` / ``kappa_star``
    (the crossing, or the point of smallest ``|gap|``), and ``min_gap``.
    """
    def clean(s, k):
        s = np.asarray(s, dtype=float)
        k = np.asarray(k, dtype=float)
        good = np.isfinite(s) & np.isfinite(k)
        s, k = s[good], k[good]
        order = np.argsort(s)
        return s[order], k[order]

    sa, ka = clean(s_a, kappa_a)
    sb, kb = clean(s_b, kappa_b)
    if sa.size < 2 or sb.size < 2:
        return {"overlap": False, "reason": "fewer than two finite samples"}
    lo = max(sa.min(), sb.min())
    hi = min(sa.max(), sb.max())
    if not (hi > lo):
        return {"overlap": False, "reason": "matched-coordinate ranges do not overlap",
                "range_a": [float(sa.min()), float(sa.max())],
                "range_b": [float(sb.min()), float(sb.max())]}

    grid = np.linspace(lo, hi, n_grid)
    ia = np.interp(grid, sa, ka)
    ib = np.interp(grid, sb, kb)
    gap = ia - ib
    sign_change = np.nonzero(np.diff(np.sign(gap)) != 0)[0]
    if sign_change.size:
        i = int(sign_change[0])
        frac = gap[i] / (gap[i] - gap[i + 1])
        s_star = float(grid[i] + frac * (grid[i + 1] - grid[i]))
        kappa_star = float(ia[i] + frac * (ia[i + 1] - ia[i]))
        crossing = True
    else:
        i = int(np.argmin(np.abs(gap)))
        s_star, kappa_star, crossing = float(grid[i]), float(ia[i]), False
    return {"overlap": True, "s_grid": grid, "kappa_a": ia, "kappa_b": ib,
            "gap": gap, "crossing": crossing, "s_star": s_star,
            "kappa_star": kappa_star, "min_gap": float(np.min(np.abs(gap)))}
