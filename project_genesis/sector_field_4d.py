"""The 4-D sector field: the like-for-like substrate for the one-κ bridge.

The scale-matched bridge (`n3_scale_matched_bridge.py`) closed the last
lattice-topology route between the acts *across dimensions*: even at matched
smoothing-per-instanton-size, a 4-D gauge vacuum and a 2-D sigma-model
substrate carry robustly different coherent fractions.  The remaining
measurable route is **like-for-like**: read both κ's in the *same* dimension.
This module gives Act II's sector field its 4-D home.

The physics.  The normalised sector field ``ψ(x) ∈ ℂ^N`` (CP^(N-1)) carries a
**composite U(1) gauge field**: on the lattice, the link phase

    u_μ(x) = (ψ̄(x)·ψ(x+μ̂)) / |ψ̄(x)·ψ(x+μ̂)|

is the parallel transporter of the CP phase freedom, and the plaquette angle

    f_{μν}(x) = arg[ u_μ(x) u_ν(x+μ̂) u_μ(x+ν̂)* u_ν(x)* ]  ∈ (−π, π]

is its field strength — invariant under the local phase ``ψ → e^{iα(x)}ψ``.
In 2-D, ``Σ f/2π`` *is* the integer instanton number (each U(1) link phase
cancels around the periodic lattice, so the sum is exactly ``2π·ℤ``).  In 4-D
the same composite field carries the **second Chern form**:

    q(x) = (1/32π²) ε_{μνρσ} f_{μν} f_{ρσ}
         = (1/4π²) [ f₀₁f₂₃ − f₀₂f₁₃ + f₀₃f₁₂ ] ,

whose lattice total is exactly the intersection number ``c₁∪c₁ ∈ ℤ`` on
factorised fluxes — the sector field's own 4-D instanton number, the direct
analogue of the gauge sector's ``(1/32π²)εTr[FF]``.  And it comes with the same
**Bogomolny pair**: the field-strength action density

    e(x) = (1/8π²) Σ_{μ<ν} f_{μν}(x)²

obeys ``e(x) ≥ |q(x)|`` pointwise, with equality exactly on (anti-)self-dual
``f`` — so the shared coherent fraction ``κ̂ = Σ|q|/Σe ∈ [0,1]``
(`topological_charge.coherent_fraction`) applies unchanged, now with **both
sectors in four dimensions**: same operator, same dimension, same heat-kernel
smoothing ``λ = √(8Dt)``, same BPST peak-size convention.

The dynamics here is dimension-generic: the same lattice CP action
``S = Σ(1−|ψ̄·ψ'|²)``, a checkerboard Metropolis sampler, and the projected
gradient flow (the Wilson-flow analogue), all summing over however many axes
the field has — the 2-D specialisations in `topological_charge.py` /
`instanton_scales.py` are unchanged.
"""

from __future__ import annotations

import numpy as np

from .topological_charge import coherent_fraction

__all__ = [
    "link_phases",
    "plaquette_angles",
    "field_strength",
    "charge_density_from_f",
    "topological_charge_density_4d",
    "topological_charge_4d",
    "action_density_f",
    "coherent_fraction_4d",
    "sector_action",
    "sector_metropolis_sweep",
    "sector_flow_step",
    "sector_gradient_flow",
    "sector_flow_diffusion_constant",
]

_PLANES_4D = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))   # (μν, ρσ) with ε = +,−,+


def _ndim(psi: np.ndarray) -> int:
    return psi.ndim - 1


def link_phases(psi: np.ndarray) -> np.ndarray:
    """Composite U(1) link phases ``u_μ(x)`` of a CP^(N-1) field, all directions.

    Returns a complex array of shape ``(d, *lattice)`` with ``|u| = 1``; a
    vanishing overlap (measure-zero) is mapped to phase 1.
    """
    d = _ndim(psi)
    out = np.empty((d,) + psi.shape[:-1], dtype=np.complex128)
    for mu in range(d):
        overlap = np.sum(np.conj(psi) * np.roll(psi, -1, axis=mu), axis=-1)
        mag = np.abs(overlap)
        out[mu] = np.where(mag > 0.0, overlap / np.where(mag > 0.0, mag, 1.0), 1.0)
    return out


def plaquette_angles(u: np.ndarray, mu: int, nu: int) -> np.ndarray:
    """Plaquette angle ``f_{μν}(x) ∈ (−π, π]`` of U(1) link phases ``u``."""
    prod = (u[mu] * np.roll(u[nu], -1, axis=mu)
            * np.conj(np.roll(u[mu], -1, axis=nu)) * np.conj(u[nu]))
    return np.angle(prod)


def field_strength(psi: np.ndarray) -> np.ndarray:
    """All plaquette angles ``f_{μν}(x)`` of the composite field, ``μ < ν``.

    Returns shape ``(d(d−1)/2, *lattice)`` ordered ``(0,1),(0,2),…`` — for 4-D:
    ``f01, f02, f03, f12, f13, f23``.
    """
    u = link_phases(psi)
    d = u.shape[0]
    return np.stack([plaquette_angles(u, mu, nu)
                     for mu in range(d) for nu in range(mu + 1, d)])


def charge_density_from_f(f: np.ndarray) -> np.ndarray:
    """4-D second-Chern charge density from stacked ``(f01,f02,f03,f12,f13,f23)``.

    ``q(x) = (1/4π²)[f₀₁f₂₃ − f₀₂f₁₃ + f₀₃f₁₂]``; exactly the intersection
    number ``c₁∪c₁`` when summed over factorised fluxes.
    """
    if f.shape[0] != 6:
        raise ValueError("expected the six μ<ν components of a 4-D field strength")
    f01, f02, f03, f12, f13, f23 = f
    return (f01 * f23 - f02 * f13 + f03 * f12) / (4.0 * np.pi ** 2)


def topological_charge_density_4d(psi: np.ndarray) -> np.ndarray:
    """Per-site composite topological charge density ``q(x)`` of a 4-D ψ field."""
    if _ndim(psi) != 4:
        raise ValueError("topological_charge_density_4d expects a 4-D field (L,L,L,L,N)")
    return charge_density_from_f(field_strength(psi))


def topological_charge_4d(psi: np.ndarray) -> float:
    """Total composite charge ``Q = Σq`` (``c₁∪c₁ ∈ ℤ`` on factorised fluxes)."""
    return float(topological_charge_density_4d(psi).sum())


def action_density_f(psi: np.ndarray) -> np.ndarray:
    """Bogomolny partner ``e(x) = (1/8π²) Σ_{μ<ν} f_{μν}² ≥ |q(x)|`` (4-D ψ)."""
    if _ndim(psi) != 4:
        raise ValueError("action_density_f expects a 4-D field (L,L,L,L,N)")
    f = field_strength(psi)
    return np.sum(f ** 2, axis=0) / (8.0 * np.pi ** 2)


def coherent_fraction_4d(psi: np.ndarray) -> float:
    """The shared Bogomolny coherent fraction ``κ̂ = Σ|q|/Σe`` of the 4-D sector field.

    The same operator as `gauge_topology.self_dual_fraction` (4-D SU(N)) and
    `topological_charge.cp_coherent_fraction` (2-D CP) — here read on the
    composite U(1) field strength of the 4-D sector field, so the two theories
    of the one-κ comparison finally share dimension as well as operator.
    """
    f = field_strength(psi)
    return coherent_fraction(np.sum(f ** 2, axis=0) / (8.0 * np.pi ** 2),
                             charge_density_from_f(f))


# --------------------------------------------------------------------------
# Dimension-generic dynamics (2-D specialisations live in topological_charge /
# instanton_scales; these sum over every lattice axis the field has)
# --------------------------------------------------------------------------

def sector_action(psi: np.ndarray) -> float:
    """Lattice CP action ``Σ_x Σ_μ (1 − |ψ̄·ψ'|²)`` in any dimension."""
    s = 0.0
    for axis in range(_ndim(psi)):
        nb = np.roll(psi, -1, axis=axis)
        s += float((1.0 - np.abs(np.sum(np.conj(psi) * nb, axis=-1)) ** 2).sum())
    return s


def _site_action(psi: np.ndarray) -> np.ndarray:
    """Per-site action over all links touching each site (both directions)."""
    s = np.zeros(psi.shape[:-1])
    for axis in range(_ndim(psi)):
        for shift in (-1, 1):
            nb = np.roll(psi, shift, axis=axis)
            s += 1.0 - np.abs(np.sum(np.conj(nb) * psi, axis=-1)) ** 2
    return s


def sector_metropolis_sweep(psi: np.ndarray, beta: float,
                            rng: np.random.Generator,
                            epsilon: float = 0.5) -> np.ndarray:
    """One checkerboard Metropolis sweep of the CP field at coupling ``beta``.

    The d-dimensional generalisation of `topological_charge.cp_metropolis_sweep`
    — sites of one parity (sum of coordinates mod 2) update in parallel while
    their neighbours (the other parity) are held fixed, so detailed balance
    holds per half-sweep.
    """
    z = psi.copy()
    d = _ndim(z)
    grids = np.indices(z.shape[:-1])
    parity = grids.sum(axis=0) % 2
    for colour in (0, 1):
        mask = parity == colour
        prop = z + epsilon * (rng.standard_normal(z.shape)
                              + 1j * rng.standard_normal(z.shape))
        prop /= np.linalg.norm(prop, axis=-1, keepdims=True)
        s_old = _site_action(z)
        trial = z.copy()
        trial[mask] = prop[mask]
        s_new = _site_action(trial)
        accept = (rng.random(z.shape[:-1]) < np.exp(-beta * (s_new - s_old))) & mask
        z[accept] = prop[accept]
    return z


def sector_flow_step(psi: np.ndarray, dt: float) -> np.ndarray:
    """One projected-gradient-flow step in any dimension (cf. `cp_flow_step`).

    ``−∂S/∂ψ̄(x) = M(x)ψ(x)`` with ``M(x) = Σ_{y~x} ψ(y)ψ(y)†`` over the ``2d``
    axis neighbours; the tangent-space step keeps ``|ψ| = 1`` to ``O(dt²)`` and
    the renormalisation removes the rest.  The linearised flow is the lattice
    heat equation, stable for ``dt ≲ 1/(2d)`` (``0.125`` in 4-D).
    """
    g = np.zeros_like(psi)
    for axis in range(_ndim(psi)):
        for shift in (-1, 1):
            nb = np.roll(psi, shift, axis=axis)
            g += nb * np.sum(np.conj(nb) * psi, axis=-1, keepdims=True)
    radial = np.sum(np.conj(psi) * g, axis=-1, keepdims=True)
    z = psi + dt * (g - radial * psi)
    return z / np.linalg.norm(z, axis=-1, keepdims=True)


def sector_gradient_flow(psi: np.ndarray, *, t_max: float, dt: float = 0.05,
                         measure_every: int = 5,
                         min_frac: float = 0.25) -> tuple[np.ndarray, list[dict]]:
    """Integrate the 4-D sector flow to ``t_max``; return flowed field + trajectory.

    Records, every ``measure_every`` steps: flow time ``t``, CP action ``S``,
    composite charge ``Q``, the coherent fraction ``κ̂``, and the mean instanton
    size ``rho_mean`` from BPST peak heights of ``|q(x)|`` (the *same* 4-D size
    convention as the gauge sector — `instanton_scales.instanton_sizes_su`),
    with the lump count ``n_lumps``.
    """
    from .instanton_scales import instanton_sizes_su

    z = psi.copy()
    n_steps = max(1, int(round(t_max / dt)))
    traj: list[dict] = []

    max_rho = min(psi.shape[:-1]) / 2.0   # a lump wider than the torus isn't a size

    def record(t: float) -> None:
        f = field_strength(z)
        qd = charge_density_from_f(f)
        e = np.sum(f ** 2, axis=0) / (8.0 * np.pi ** 2)
        sizes = instanton_sizes_su(qd, min_frac=min_frac)
        sizes = sizes[sizes <= max_rho]
        traj.append({
            "t": float(t),
            "S": sector_action(z),
            "Q": float(qd.sum()),
            "kappa": coherent_fraction(e, qd),
            "rho_mean": float(sizes.mean()) if sizes.size else float("nan"),
            "n_lumps": int(sizes.size),
        })

    record(0.0)
    for step in range(1, n_steps + 1):
        z = sector_flow_step(z, dt)
        if step % measure_every == 0 or step == n_steps:
            record(step * dt)
    return z, traj


def sector_flow_diffusion_constant(size: int = 12, dim: int = 4,
                                   n_colors: int = 3, dt: float = 0.05,
                                   n_steps: int = 40) -> float:
    """Measure the d-dimensional sector flow's diffusion constant (cf. 2-D version).

    A transverse plane wave along one axis decays at ``γ = 2(1−cos k)·D`` under
    the linearised flow; fitting ``ln A(τ)`` gives ``D`` (the prediction is
    exactly 1, and `instanton_scales.smoothing_radius` uses the measured value).
    """
    k = 2.0 * np.pi / size
    z0 = np.zeros(n_colors, dtype=np.complex128)
    z0[0] = 1.0
    v = np.zeros(n_colors, dtype=np.complex128)
    v[1] = 1.0
    shape = (size,) * dim
    x_last = np.arange(size)
    wave = 0.05 * np.cos(k * x_last)                     # varies along final axis
    wave = np.broadcast_to(wave, shape)
    z = np.broadcast_to(z0, shape + (n_colors,)).copy()
    z = z + wave[..., None] * v
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)

    def amplitude(field: np.ndarray) -> float:
        comp = field @ np.conj(v)
        return float(np.abs(2.0 * np.mean(comp * wave / 0.05)))

    times, amps = [0.0], [amplitude(z)]
    for step in range(1, n_steps + 1):
        z = sector_flow_step(z, dt)
        times.append(step * dt)
        amps.append(amplitude(z))
    slope = np.polyfit(times, np.log(np.maximum(amps, 1e-300)), 1)[0]
    return float(-slope / (2.0 * (1.0 - np.cos(k))))
