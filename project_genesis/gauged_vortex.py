"""The self-consistent gauge field: the vortex becomes a gauged particle.

The phase-pinned braid (`n3_phase_pinned_braid.py`) completed the dynamical
exchange by anchoring the winding with a **background phase template**, and
named its own caveat in scope: *"a classical, local restoring force toward a
winding template (a background gauge-like coupling), **not** a dynamical gauge
field solved self-consistently."*  This module supplies the real thing — the
gauge field that a genuine charged particle carries — by coupling the vortex
field to a **U(1) gauge connection** and solving both self-consistently.  It is
the textbook **abelian Higgs / Ginzburg–Landau** model (a lattice
superconductor), whose vortex is the Abrikosov / Nielsen–Olesen flux tube:

- ``ψ`` — a complex scalar (the Higgs / order parameter) on sites;
- ``θ_μ(x)`` — U(1) **link phases**, ``U_μ(x) = e^{iθ_μ(x)}``, the lattice gauge
  connection (parallel transport ``x → x+μ``);
- the gauge-invariant **magnetic flux** through a plaquette is the wrapped
  plaquette angle ``B(x) = θ_x(x) + θ_y(x+x̂) − θ_x(x+ŷ) − θ_y(x)``.

The energy is the gauged Ginzburg–Landau functional

    E = Σ_μ |ψ(x) − U_μ(x)ψ(x+μ)|²  +  (β/2) Σ B(x)²  +  (λ/4) Σ (|ψ|²−1)²

(covariant kinetic + Maxwell magnetic + Higgs potential).  Gradient flow on it
(`relax`) descends to the self-consistent gauged vortex.

**Flux attachment (Chern–Simons proxy).**  The Aharonov–Bohm experiment read the
statistical phase off the solved Wilson loop; flux–charge binding was not
enforced dynamically.  An optional soft attachment term

    E_att = (γ/2) Σ κ(x) · (B(x) − 2π · ρ(x))²

penalises local mismatch between plaquette flux ``B`` and a charge-density
proxy ``ρ`` built from the Higgs depletion ``(1 − |ψ|²)_+``, normalised so a
winding-``q`` core carries integrated charge ``q``.  ``κ(x)`` is an optional
capacity weight (default 1): attachment costs more where capacity is available,
matching the framework's κ-weighted integration.  This is a **classical lattice
proxy** for Chern–Simons flux attachment, not a continuum CS action and not
second quantisation.

Honest scope: classical U(1) gauge field and classical attachment; no Fock
space, no ``{ψ, ψ†}``, no many-body Pauli principle.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "seed_vortices",
    "plaquette_flux",
    "local_flux",
    "covariant_laplacian",
    "energy",
    "energy_parts",
    "relax",
    "gauge_transform",
    "zero_links",
    "wilson_loop",
    "ab_phase",
    "charge_density_proxy",
    "attachment_energy",
    "displace_flux",
    "flux_core_offset",
]


def _wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles into ``(−π, π]``."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def zero_links(n: int) -> list:
    """A trivial gauge connection ``θ_μ = 0`` (``U_μ = 1``) on an ``n×n`` torus."""
    return [np.zeros((n, n)), np.zeros((n, n))]


def seed_vortices(n: int, centers, charges, core: float = 3.0) -> np.ndarray:
    """A scalar field with vortices of integer ``charges`` at ``centers``.

    ``ψ = Π_k tanh(r_k/core)·e^{i q_k arg(x − c_k)}`` (periodic distance) — the
    Higgs amplitude holed at each core, the phase winding ``q_k``.
    """
    grids = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    psi = np.ones((n, n), dtype=complex)
    for (cx, cy), q in zip(centers, charges):
        dx = ((grids[0] - cx + n / 2) % n) - n / 2
        dy = ((grids[1] - cy + n / 2) % n) - n / 2
        r = np.hypot(dx, dy)
        psi = psi * np.tanh(r / core) * np.exp(1j * q * np.arctan2(dy, dx))
    return psi


def plaquette_flux(theta) -> np.ndarray:
    """The gauge-invariant magnetic flux ``B(x)`` through each plaquette."""
    tx, ty = theta
    return _wrap(tx + np.roll(ty, -1, 0) - np.roll(tx, -1, 1) - ty)


def local_flux(theta, center, radius: float) -> float:
    """Total magnetic flux through a disk of ``radius`` about ``center``."""
    b = plaquette_flux(theta)
    n = b.shape[0]
    grids = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    dx = ((grids[0] - center[0] + n / 2) % n) - n / 2
    dy = ((grids[1] - center[1] + n / 2) % n) - n / 2
    return float(b[np.hypot(dx, dy) < radius].sum())


def charge_density_proxy(psi: np.ndarray, target_charge: float) -> np.ndarray:
    """Normalised Higgs-depletion density with integrated charge ``target_charge``.

    ``ρ₀ = max(1 − |ψ|², 0)`` peaks at vortex cores.  Rescaled so
    ``Σ ρ = target_charge`` when the total depletion is positive; otherwise a
    zero field (no charge to attach).
    """
    rho = np.maximum(1.0 - np.abs(psi) ** 2, 0.0)
    total = float(rho.sum())
    if total < 1e-12 or abs(target_charge) < 1e-12:
        return np.zeros_like(rho)
    return rho * (target_charge / total)


def attachment_energy(
    psi: np.ndarray,
    theta,
    *,
    gamma: float,
    target_charge: float,
    kappa: np.ndarray | None = None,
) -> float:
    """Soft flux-attachment energy ``(γ/2) Σ κ (B − 2π ρ)²``."""
    if gamma <= 0.0:
        return 0.0
    b = plaquette_flux(theta)
    rho = charge_density_proxy(psi, target_charge)
    mismatch = b - 2.0 * np.pi * rho
    weight = 1.0 if kappa is None else kappa
    return float(0.5 * gamma * np.sum(weight * mismatch ** 2))


def covariant_laplacian(psi: np.ndarray, theta) -> np.ndarray:
    """``(D²ψ)(x) = Σ_μ [U_μ(x)ψ(x+μ) + U_μ(x−μ)*ψ(x−μ) − 2ψ(x)]``."""
    out = np.zeros_like(psi)
    for mu in (0, 1):
        u = np.exp(1j * theta[mu])
        fwd = u * np.roll(psi, -1, mu)
        u_back = np.exp(1j * np.roll(theta[mu], 1, mu))
        bwd = np.conj(u_back) * np.roll(psi, 1, mu)
        out = out + fwd + bwd - 2.0 * psi
    return out


def energy_parts(
    psi: np.ndarray,
    theta,
    lam: float,
    beta: float,
    *,
    gamma: float = 0.0,
    target_charge: float = 0.0,
    kappa: np.ndarray | None = None,
) -> dict:
    """Energy contributions: kinetic, magnetic, potential, attachment."""
    kinetic = 0.0
    for mu in (0, 1):
        u = np.exp(1j * theta[mu])
        kinetic += float(np.sum(np.abs(psi - u * np.roll(psi, -1, mu)) ** 2))
    magnetic = float(0.5 * beta * np.sum(plaquette_flux(theta) ** 2))
    potential = float(0.25 * lam * np.sum((np.abs(psi) ** 2 - 1.0) ** 2))
    attach = attachment_energy(
        psi, theta, gamma=gamma, target_charge=target_charge, kappa=kappa
    )
    return {
        "kinetic": kinetic,
        "magnetic": magnetic,
        "potential": potential,
        "attachment": attach,
        "total": kinetic + magnetic + potential + attach,
    }


def energy(
    psi: np.ndarray,
    theta,
    lam: float,
    beta: float,
    *,
    gamma: float = 0.0,
    target_charge: float = 0.0,
    kappa: np.ndarray | None = None,
) -> float:
    """Total gauged Ginzburg–Landau energy (with optional attachment)."""
    return energy_parts(
        psi, theta, lam, beta,
        gamma=gamma, target_charge=target_charge, kappa=kappa,
    )["total"]


def _link_force(
    psi: np.ndarray,
    theta,
    beta: float,
    *,
    gamma: float = 0.0,
    target_charge: float = 0.0,
    kappa: np.ndarray | None = None,
) -> list:
    """``−∂E/∂θ_μ``: matter current + Maxwell + optional attachment."""
    b = plaquette_flux(theta)
    force = []
    for mu in (0, 1):
        u = np.exp(1j * theta[mu])
        current = 2.0 * np.imag(np.conj(psi) * u * np.roll(psi, -1, mu))
        force.append(-current)
    force[0] = force[0] - beta * (b - np.roll(b, 1, 1))
    force[1] = force[1] - beta * (-b + np.roll(b, 1, 0))

    if gamma > 0.0 and abs(target_charge) > 1e-12:
        rho = charge_density_proxy(psi, target_charge)
        weight = 1.0 if kappa is None else kappa
        dE_dB = gamma * weight * (b - 2.0 * np.pi * rho)
        force[0] = force[0] - (dE_dB - np.roll(dE_dB, 1, 1))
        force[1] = force[1] - (-dE_dB + np.roll(dE_dB, 1, 0))

    return force


def relax(
    psi: np.ndarray,
    theta,
    *,
    lam: float = 2.0,
    beta: float = 1.0,
    dt: float = 0.05,
    steps: int = 4000,
    gauge_on: bool = True,
    record_every: int = 0,
    gamma: float = 0.0,
    target_charge: float = 0.0,
    kappa: np.ndarray | None = None,
) -> dict:
    """Gradient-flow the gauged GL energy (optional flux attachment).

    With ``gamma > 0`` the soft CS-proxy term pulls plaquette flux toward
    ``2π ρ(ψ)``, locking flux to the charge-density proxy.  ``kappa`` weights
    the attachment cost spatially when supplied.
    """
    psi = np.array(psi, dtype=complex)
    theta = [np.array(theta[0], dtype=float), np.array(theta[1], dtype=float)]
    hist = []
    for i in range(steps):
        if record_every and i % record_every == 0:
            hist.append(energy(
                psi, theta, lam, beta,
                gamma=gamma, target_charge=target_charge, kappa=kappa,
            ))
        psi = psi + dt * (
            covariant_laplacian(psi, theta)
            - 0.5 * lam * (np.abs(psi) ** 2 - 1.0) * psi
        )
        if gauge_on:
            force = _link_force(
                psi, theta, beta,
                gamma=gamma, target_charge=target_charge, kappa=kappa,
            )
            theta = [theta[0] + dt * force[0], theta[1] + dt * force[1]]
    parts = energy_parts(
        psi, theta, lam, beta,
        gamma=gamma, target_charge=target_charge, kappa=kappa,
    )
    return {
        "psi": psi,
        "theta": theta,
        "energy": parts["total"],
        "parts": parts,
        "history": hist,
    }


def gauge_transform(psi: np.ndarray, theta, alpha: np.ndarray) -> tuple:
    """Apply a local gauge transformation ``α(x)``."""
    psi_g = np.exp(1j * alpha) * psi
    theta_g = [
        theta[0] + alpha - np.roll(alpha, -1, 0),
        theta[1] + alpha - np.roll(alpha, -1, 1),
    ]
    return psi_g, theta_g


def wilson_loop(theta, center, half_width: float) -> tuple:
    """The gauge holonomy of a square loop — the Aharonov–Bohm phase."""
    tx, ty = theta
    cx, cy = int(round(center[0])), int(round(center[1]))
    r = int(round(half_width))
    x0, x1, y0, y1 = cx - r, cx + r, cy - r, cy + r
    holonomy = (
        float(tx[x0:x1, y0].sum())
        + float(ty[x1, y0:y1].sum())
        - float(tx[x0:x1, y1].sum())
        - float(ty[x0, y0:y1].sum())
    )
    return holonomy, complex(np.exp(1j * holonomy))


def ab_phase(theta, center, radius: float, charge: float = 1.0) -> complex:
    """The Aharonov–Bohm phase ``e^{iQΦ}`` a charge ``Q`` accrues encircling the flux."""
    holonomy = local_flux(theta, center, radius)
    return complex(np.exp(1j * charge * holonomy))


def displace_flux(theta, shift: tuple[int, int]) -> list:
    """Rigidly translate the link-phase field by ``shift`` (periodic).

    Used to create a controlled flux–core mismatch: the Higgs stays put while
    the gauge field (and its flux) is moved.
    """
    sx, sy = shift
    return [
        np.roll(np.roll(theta[0], sx, 0), sy, 1),
        np.roll(np.roll(theta[1], sx, 0), sy, 1),
    ]


def _periodic_offset(
    a: tuple[float, float], b: tuple[float, float], n: int
) -> tuple[float, float, float]:
    """Minimum-image displacement from a → b on an n×n torus."""
    dx = ((b[0] - a[0] + n / 2) % n) - n / 2
    dy = ((b[1] - a[1] + n / 2) % n) - n / 2
    return float(dx), float(dy), float(np.hypot(dx, dy))


def flux_core_offset(
    psi: np.ndarray,
    theta,
    *,
    center: tuple[float, float] | None = None,
    flux_radius: float = 6.0,
) -> dict:
    """Separation between Higgs core and flux peak (primary: peak markers).

    **Peak mode (primary).**  Core = ``argmin |ψ|`` (deepest amplitude hole);
    flux peak = ``argmax |B|``.  Periodic minimum-image distance.  Centroids of
    broad ``(1−|ψ|²)`` and ``|B|`` clouds were too noisy (baseline offsets of
    several lattice units on well-quantised vortices); peaks track the same
    objects the displacement protocol moves.

    Also reports centroid offset (secondary), and ``core_flux`` = total flux
    through a disk of ``flux_radius`` about the Higgs peak — a lock-recovery
    observable independent of the flux-peak location.
    """
    n = psi.shape[0]
    amp = np.abs(psi)
    b = plaquette_flux(theta)
    abs_b = np.abs(b)

    # --- peaks ---
    core_ij = np.unravel_index(int(np.argmin(amp)), amp.shape)
    flux_ij = np.unravel_index(int(np.argmax(abs_b)), abs_b.shape)
    core_peak = (float(core_ij[0]), float(core_ij[1]))
    flux_peak = (float(flux_ij[0]), float(flux_ij[1]))
    dx_p, dy_p, offset_peak = _periodic_offset(core_peak, flux_peak, n)

    # --- centroids (secondary) ---
    rho = np.maximum(1.0 - amp ** 2, 0.0)
    grids = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ref = center if center is not None else core_peak

    def _centroid(weight: np.ndarray) -> tuple[float, float]:
        w = float(weight.sum())
        if w < 1e-12:
            return ref
        dx = ((grids[0] - ref[0] + n / 2) % n) - n / 2
        dy = ((grids[1] - ref[1] + n / 2) % n) - n / 2
        cx = (ref[0] + float((weight * dx).sum() / w)) % n
        cy = (ref[1] + float((weight * dy).sum() / w)) % n
        return (cx, cy)

    core_c = _centroid(rho)
    flux_c = _centroid(abs_b)
    dx_c, dy_c, offset_cent = _periodic_offset(core_c, flux_c, n)

    core_flux = local_flux(theta, core_peak, flux_radius)

    return {
        "core": core_peak,
        "flux_center": flux_peak,
        "offset": offset_peak,  # primary
        "dx": dx_p,
        "dy": dy_p,
        "offset_peak": offset_peak,
        "offset_centroid": offset_cent,
        "core_centroid": core_c,
        "flux_centroid": flux_c,
        "core_flux": float(core_flux),
        "core_flux_quanta": float(core_flux / (2.0 * np.pi)),
    }
