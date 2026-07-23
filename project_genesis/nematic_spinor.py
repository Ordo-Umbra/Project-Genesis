"""Half-integer spin: a nematic director field and its ½-disclinations.

Every spin so far was **integer** — a ``U(1)`` complex order parameter (a boson).
A quark is a **half-integer spinor**: under a ``2π`` rotation it picks up ``−1``,
and only ``4π`` returns it to itself — the ``SU(2)`` double cover of ``SO(3)``.
That needs a *different* order parameter: a **nematic**, a headless director
``n̂`` with the ``Z₂`` head–tail identification ``n̂ ≡ −n̂`` (order-parameter space
``RP¹``).  Its defects are **±½ disclinations** — the director winds by only
``π`` around the core, impossible for a vector — and that ``π``-winding *is* the
double cover.

A 2-D nematic is carried faithfully by ``ψ = e^{2iθ}`` (the *doubled* angle, so
``θ`` and ``θ + π`` give the same ``ψ``): the director angle is
``θ = ½·arg(ψ)`` with ``θ ≡ θ + π``, and a ``ψ``-vortex of integer charge ``q``
is a disclination of strength ``s = q/2`` — **half-integer when ``q`` is odd**.
So the whole ``U(1)`` vortex machinery (`vortex_chiral.imprint_vortices`, the
CGL evolution) applies, but the *physical, observable* object — the director —
is double-valued in the ``ψ`` lift, and that is the spinor.

This module supplies the nematic readouts:

- `director_angle` — ``θ = ½·arg(ψ)`` (mod ``π``);
- `disclination_strength` — the half-integer winding ``s`` of the director
  around a loop (mod-``π`` differences, ``/2π``);
- `director_holonomy` — transport the *oriented* director around the defect:
  ``−1`` after one ``2π`` loop (the spinor sign flip), ``+1`` after ``4π``.

Honest scope: this is the **order-parameter (topological)** realisation of
half-integer spin — a genuine ``Z₂/RP¹`` double cover with ``π``-winding
disclinations.  It is **not** the full quantum Dirac field: no spin–statistics
(anticommutation), no Dirac equation, no dynamical fermion.  What is measured is
the half-integer winding, the ``4π`` double cover, and the disclination fusion
rules — the topological content of "spin-½".
"""

from __future__ import annotations

import numpy as np


def director_angle(psi: np.ndarray) -> np.ndarray:
    """The nematic director angle ``θ = ½·arg(ψ)`` (mod ``π``, head–tail)."""
    return 0.5 * np.angle(psi)


def plaquette_winding(psi: np.ndarray) -> np.ndarray:
    """Integer winding of ``arg ψ`` on each unit plaquette — the defect map.

    Nonzero exactly at defect cores (``±1`` for elementary defects, i.e.
    ``±½`` disclinations in the nematic reading); zero elsewhere.  Periodic;
    the entry at ``(x, y)`` is the plaquette with lower-left corner ``(x, y)``.
    """
    th = np.angle(psi)

    def wrap(d):
        return (d + np.pi) % (2.0 * np.pi) - np.pi

    d1 = wrap(np.roll(th, -1, 0) - th)
    d2 = wrap(np.roll(np.roll(th, -1, 0), -1, 1) - np.roll(th, -1, 0))
    d3 = wrap(np.roll(th, -1, 1) - np.roll(np.roll(th, -1, 0), -1, 1))
    d4 = wrap(th - np.roll(th, -1, 1))
    return np.round((d1 + d2 + d3 + d4) / (2.0 * np.pi)).astype(int)


def _loop(center, radius, npts):
    th = np.linspace(0.0, 2.0 * np.pi, npts, endpoint=False)
    return th, center[0] + radius * np.cos(th), center[1] + radius * np.sin(th)


def disclination_strength(psi: np.ndarray, center, radius: float = 10.0,
                          npts: int = 720) -> float:
    """Disclination strength ``s`` — the director winding around ``center``.

    The director ``θ = ½·arg(ψ)`` rotates by ``2π·s`` around the loop; summing the
    per-step differences *wrapped into* ``(−π/2, π/2]`` (the nematic, head–tail
    distance) and dividing by ``2π`` gives ``s``.  A ``ψ``-charge-``q`` defect
    reads ``s = q/2`` — **half-integer for odd ``q``** (the elementary
    disclination is ``±½``).
    """
    n0, n1 = psi.shape
    _, xs, ys = _loop(center, radius, npts)
    xs = np.round(xs).astype(int) % n0
    ys = np.round(ys).astype(int) % n1
    theta = 0.5 * np.angle(psi[xs, ys])
    d = np.diff(np.concatenate([theta, theta[:1]]))
    d = (d + np.pi / 2) % np.pi - np.pi / 2            # nematic (mod-π) difference
    return float(np.sum(d) / (2.0 * np.pi))


def director_holonomy(psi: np.ndarray, center, radius: float = 10.0,
                      npts: int = 720) -> tuple:
    """The double-cover holonomy of the *oriented* director around the defect.

    Lift the director to an oriented vector ``n̂ = (cos θ, sin θ)`` with
    ``θ = ½·(unwrapped arg ψ)`` and transport it **twice** around the loop.
    Returns ``(n̂·n̂ after 2π, n̂·n̂ after 4π)``: for a ``±½`` disclination the
    first is ``−1`` (the spinor sign flip) and the second ``+1`` (restored after
    ``4π``); an integer disclination gives ``+1`` already at ``2π``.
    """
    n0, n1 = psi.shape
    th = np.linspace(0.0, 4.0 * np.pi, 2 * npts, endpoint=False)   # around twice
    xs = np.round(center[0] + radius * np.cos(th % (2.0 * np.pi))).astype(int) % n0
    ys = np.round(center[1] + radius * np.sin(th % (2.0 * np.pi))).astype(int) % n1
    ang = 0.5 * np.unwrap(np.angle(psi[xs, ys]))       # oriented lift (no mod π)
    start = np.array([np.cos(ang[0]), np.sin(ang[0])])
    at_2pi = np.array([np.cos(ang[npts]), np.sin(ang[npts])])
    at_4pi = np.array([np.cos(ang[-1]), np.sin(ang[-1])])
    return float(start @ at_2pi), float(start @ at_4pi)
