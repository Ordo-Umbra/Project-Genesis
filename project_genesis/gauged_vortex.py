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
(`relax`) descends to the self-consistent gauged vortex.  Three facts, each
what a real gauge field gives that the phase template did not:

1. **Flux quantization.**  A winding-``q`` vortex forces the gauge field to
   carry a *quantised* magnetic flux ``Φ = 2π·q`` through a loop about the core —
   solved by the dynamics, not imposed; ``0`` when the gauge field is frozen off.
2. **Finite energy (London screening).**  The gauge field screens the
   logarithmically-divergent energy of a *global* vortex over the London
   penetration length, so the gauged vortex is a **finite-energy soliton** whose
   energy converges as the box grows (the global vortex's does not).
3. **Gauge invariance.**  The observables (flux, energy) are invariant under a
   *local* gauge transformation ``ψ → e^{iα(x)}ψ``, ``θ_μ → θ_μ + α(x) −
   α(x+μ)`` — the signature of a genuine gauge theory, which a fixed phase
   template is not.

Honest scope: this is the **classical** gauge field — a genuine, self-consistent
U(1) gauge theory, closing the phase-template caveat and making the vortex a
finite-energy gauged particle.  It is **not** second quantization: no Fock space,
no ``{ψ, ψ†}`` anticommutator, no many-body Pauli principle.  The
Aharonov–Bohm / Chern–Simons statistical phase — the flux-attachment that turns
the gauged ½-vortex's topological exchange sign into a genuine dynamical fermion
statistic — is the standing frontier.
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
    """Total magnetic flux through a disk of ``radius`` about ``center``.

    For a winding-``q`` vortex this saturates at ``2π·q`` (one flux quantum per
    winding) once ``radius`` exceeds the London length and while it stays inside
    the region the vortex's flux occupies.
    """
    b = plaquette_flux(theta)
    n = b.shape[0]
    grids = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    dx = ((grids[0] - center[0] + n / 2) % n) - n / 2
    dy = ((grids[1] - center[1] + n / 2) % n) - n / 2
    return float(b[np.hypot(dx, dy) < radius].sum())


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


def energy_parts(psi: np.ndarray, theta, lam: float, beta: float) -> dict:
    """The three energy contributions: covariant kinetic, magnetic, potential."""
    kinetic = 0.0
    for mu in (0, 1):
        u = np.exp(1j * theta[mu])
        kinetic += float(np.sum(np.abs(psi - u * np.roll(psi, -1, mu)) ** 2))
    magnetic = float(0.5 * beta * np.sum(plaquette_flux(theta) ** 2))
    potential = float(0.25 * lam * np.sum((np.abs(psi) ** 2 - 1.0) ** 2))
    return {"kinetic": kinetic, "magnetic": magnetic, "potential": potential,
            "total": kinetic + magnetic + potential}


def energy(psi: np.ndarray, theta, lam: float, beta: float) -> float:
    """Total gauged Ginzburg–Landau energy."""
    return energy_parts(psi, theta, lam, beta)["total"]


def _link_force(psi: np.ndarray, theta, beta: float) -> list:
    """``−∂E/∂θ_μ``: the matter current plus the Maxwell ``∇×B`` term."""
    b = plaquette_flux(theta)
    force = []
    for mu in (0, 1):
        u = np.exp(1j * theta[mu])
        current = 2.0 * np.imag(np.conj(psi) * u * np.roll(psi, -1, mu))
        force.append(-current)
    # θ_x(x) enters B(x) with +1 and B(x−ŷ) with −1; θ_y(x) enters B(x) with −1
    # and B(x−x̂) with +1
    force[0] = force[0] - beta * (b - np.roll(b, 1, 1))
    force[1] = force[1] - beta * (-b + np.roll(b, 1, 0))
    return force


def relax(psi: np.ndarray, theta, *, lam: float = 2.0, beta: float = 1.0,
          dt: float = 0.05, steps: int = 4000, gauge_on: bool = True,
          record_every: int = 0) -> dict:
    """Gradient-flow the gauged Ginzburg–Landau energy to the self-consistent
    vortex.

    Descends ``∂_tψ = D²ψ − (λ/2)(|ψ|²−1)ψ`` and, with ``gauge_on``,
    ``∂_tθ_μ = −∂E/∂θ_μ`` (the gauge field solving itself to carry the flux the
    winding demands).  ``gauge_on = False`` freezes ``θ`` (the *global* vortex,
    no flux) — the control that isolates what the gauge field buys.  Returns the
    relaxed ``psi``/``theta``, the final energy parts, and (if ``record_every``)
    an energy history.
    """
    psi = np.array(psi, dtype=complex)
    theta = [np.array(theta[0], dtype=float), np.array(theta[1], dtype=float)]
    hist = []
    for i in range(steps):
        if record_every and i % record_every == 0:
            hist.append(energy(psi, theta, lam, beta))
        psi = psi + dt * (covariant_laplacian(psi, theta)
                          - 0.5 * lam * (np.abs(psi) ** 2 - 1.0) * psi)
        if gauge_on:
            force = _link_force(psi, theta, beta)
            theta = [theta[0] + dt * force[0], theta[1] + dt * force[1]]
    parts = energy_parts(psi, theta, lam, beta)
    return {"psi": psi, "theta": theta, "energy": parts["total"],
            "parts": parts, "history": hist}


def gauge_transform(psi: np.ndarray, theta, alpha: np.ndarray) -> tuple:
    """Apply a local gauge transformation ``α(x)``.

    ``ψ → e^{iα}ψ``, ``θ_μ(x) → θ_μ(x) + α(x) − α(x+μ)``.  All physical
    observables (flux, energy) are invariant under this — the test that the
    theory is a genuine gauge theory rather than a fixed phase template.
    """
    psi_g = np.exp(1j * alpha) * psi
    theta_g = [theta[0] + alpha - np.roll(alpha, -1, 0),
               theta[1] + alpha - np.roll(alpha, -1, 1)]
    return psi_g, theta_g


def wilson_loop(theta, center, half_width: float) -> tuple:
    """The gauge holonomy of a square loop — the Aharonov–Bohm phase.

    Parallel-transports a *unit* test charge around a square loop of half-width
    ``half_width`` about ``center`` by multiplying the U(1) link variables along
    the perimeter: the accumulated phase (the Wilson loop) is ``∮A·dl``, which by
    lattice Stokes equals the **enclosed magnetic flux** ``Φ``.  Returns
    ``(holonomy, loop_value)`` with ``holonomy`` the unwrapped angle (``≈ 2π·q``
    for a winding-``q`` vortex) and ``loop_value = e^{iΦ}``.  The loop is assumed
    to sit inside the lattice (it must not wrap the torus).
    """
    tx, ty = theta
    cx, cy = int(round(center[0])), int(round(center[1]))
    r = int(round(half_width))
    x0, x1, y0, y1 = cx - r, cx + r, cy - r, cy + r
    holonomy = (float(tx[x0:x1, y0].sum()) + float(ty[x1, y0:y1].sum())
                - float(tx[x0:x1, y1].sum()) - float(ty[x0, y0:y1].sum()))
    return holonomy, complex(np.exp(1j * holonomy))


def ab_phase(theta, center, radius: float, charge: float = 1.0) -> complex:
    """The Aharonov–Bohm phase ``e^{iQΦ}`` a charge ``Q`` accrues encircling the
    flux.

    ``Φ`` is the holonomy ``∮A·dl`` of the self-consistent gauge field, taken as
    the enclosed flux through a disk of ``radius`` (:func:`local_flux` — a
    circular Wilson loop, robust to the loop shape; the square
    :func:`wilson_loop` is the same holonomy by Stokes but more corner-sensitive
    at small size).  A *unit* charge around one flux quantum (``Φ = 2π``) gets
    ``e^{2πi} = 1`` (Dirac); a **half** charge gets ``e^{iπ} = −1`` — the flux
    quantum is visible to a fractional charge as a sign, the Aharonov–Bohm root
    of statistics.
    """
    holonomy = local_flux(theta, center, radius)
    return complex(np.exp(1j * charge * holonomy))
