"""Spin in 3-D: the vortex becomes a LINE, and its spin becomes an axial VECTOR.

In 2-D the chiral spin lived on a *point* vortex and its angular momentum was a
scalar — a sign (`vortex_chiral.py`).  A complex field ``ψ: ℝ³ → ℂ`` vanishes
generically on **codimension-2 sets — lines**, so in 3-D the defect is a vortex
*line*, and the field's angular momentum ``L = Σ (x − c) × j`` (``j = Im(ψ*∇ψ)``)
is a genuine **3-vector, aligned with the line** and quantised by the winding.
Spin has acquired a *direction*: rotate the line and its spin vector rotates
with it, pointing anywhere in space.

This module is the field-level 3-D instrument (no molecule dynamics yet):

- `vortex_line` imprints a straight line through a point along an arbitrary
  axis, winding ``q`` in the perpendicular plane;
- `phase_current_3d` / `line_angular_momentum` measure the vector ``L`` (the
  latter over a centred sphere, so a skew line reads cleanly on a cubic box);
- `line_winding` reads the enclosed integer through a principal-axis plane
  (tracker-free, as in 2-D);
- `evolve_seeded_line` co-evolves a seeded field under the κ-detuned CGL with
  **no re-imprinting** — the self-sustained test — so the winding's conservation
  and the axis's stability are the dynamics', not an imposition.

Honest scope: this ``U(1)`` field carries an **integer / axial-vector** angular
momentum — a spin-1-like (bosonic) object.  A genuine **half-integer spinor**
(the ``SU(2)`` double cover, ``4π`` periodicity — the fermion) is a *different*
field structure, and turning matter's spin into that is the open frontier toward
fermionic matter, not what a complex order parameter gives.
"""

from __future__ import annotations

import numpy as np

from .two_field import chiral_detuning, step_chiral_detuned


def ortho_frame(normal):
    """A right-handed orthonormal frame ``(n̂, e1, e2)`` with ``n̂ ∥ normal``."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ n) * n
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return n, e1, e2


def vortex_line(shape, center, normal, q, *, core: float = 3.0,
                detune: np.ndarray | None = None) -> np.ndarray:
    """A straight vortex line through ``center`` along ``normal``, winding ``q``.

    ``ψ = A(x)·exp(i q·arg⊥(x − c))`` — the phase winds ``q`` times in the plane
    perpendicular to the line; the amplitude ``A = tanh(r⊥/core)`` holes the line
    core (times the optional ``√(1−detune)`` well envelope).
    """
    shape = tuple(int(s) for s in shape)
    nhat, e1, e2 = ortho_frame(normal)
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    r = [grids[a] - center[a] for a in range(3)]
    u = r[0] * e1[0] + r[1] * e1[1] + r[2] * e1[2]
    v = r[0] * e2[0] + r[1] * e2[1] + r[2] * e2[2]
    amp = np.tanh(np.sqrt(u * u + v * v) / core)
    if detune is not None:
        amp = amp * np.sqrt(np.clip(1.0 - detune, 0.0, 1.0))
    return amp * np.exp(1j * q * np.arctan2(v, u))


def phase_current_3d(psi: np.ndarray) -> list:
    """``j = Im(ψ*∇ψ)`` — the phase current, a 3-vector field (central diff)."""
    c = np.conj(psi)
    return [np.imag(c * 0.5 * (np.roll(psi, -1, ax) - np.roll(psi, 1, ax)))
            for ax in range(3)]


def line_angular_momentum(psi: np.ndarray, center,
                          radius: float | None = None) -> np.ndarray:
    """``L = Σ (x − c) × j`` over a centred sphere — an axial 3-vector.

    The spherical mask (default: just inside the box) makes ``L`` isotropic to
    the cubic lattice, so a line at *any* orientation reads its true direction
    without corner-truncation artifacts.
    """
    jx, jy, jz = phase_current_3d(psi)
    n = psi.shape
    gx = np.arange(n[0])[:, None, None] - center[0]
    gy = np.arange(n[1])[None, :, None] - center[1]
    gz = np.arange(n[2])[None, None, :] - center[2]
    if radius is None:
        radius = min(n) / 2 - 2
    m = (gx * gx + gy * gy + gz * gz) <= radius * radius
    return np.array([float(np.sum((gy * jz - gz * jy) * m)),
                     float(np.sum((gz * jx - gx * jz) * m)),
                     float(np.sum((gx * jy - gy * jx) * m))])


def line_winding(psi: np.ndarray, center, normal_axis: int = 2,
                 radius: int = 6) -> float:
    """Enclosed integer winding on a square loop in the plane ⊥ ``normal_axis``.

    Reads the line's topological charge (the flux threading the loop), tracker-
    free.  ``normal_axis`` is the line's direction (0/1/2); the loop lives in the
    other two axes at the line's slice.
    """
    ax = [a for a in range(3) if a != normal_axis]
    cx, cy = int(round(center[ax[0]])), int(round(center[ax[1]]))
    sl = int(round(center[normal_axis]))
    n0, n1 = psi.shape[ax[0]], psi.shape[ax[1]]
    r = int(radius)
    loop = ([(cx - r, cy + t) for t in range(-r, r)]
            + [(cx + t, cy + r) for t in range(-r, r)]
            + [(cx + r, cy + t) for t in range(r, -r, -1)]
            + [(cx + t, cy - r) for t in range(r, -r, -1)])

    def val(a, b):
        idx = [0, 0, 0]
        idx[ax[0]], idx[ax[1]], idx[normal_axis] = a % n0, b % n1, sl
        return np.angle(psi[idx[0], idx[1], idx[2]])

    ph = [val(a, b) for a, b in loop]
    return float(np.sum(np.diff(np.unwrap(ph + [ph[0]]))) / (2.0 * np.pi))


def evolve_seeded_line(shape, center, normal, q, kappa, *,
                       chiral: float = 0.0, detune_gamma: float = 0.8,
                       kappa_baseline: float = 1.0, core: float = 3.0,
                       dt: float = 0.1, steps: int = 800, record_every: int = 100,
                       winding_radius: int = 6,
                       noise: float = 0.0, seed: int = 0) -> dict:
    """Seed a vortex line ONCE, then co-evolve under the κ-detuned CGL.

    No re-imprinting: the winding's conservation and the ``L``-axis's stability
    are the dynamics'.  ``normal`` must be a principal axis (0/1/2) so the
    winding loop is well defined.  Records the enclosed winding, the ``L``
    vector, its alignment to the seed axis, and ``amp_min``.
    """
    shape = tuple(int(s) for s in shape)
    normal_axis = int(np.argmax(np.abs(normal)))
    g = chiral_detuning(kappa, detune_gamma=detune_gamma,
                        kappa_baseline=kappa_baseline)
    psi = vortex_line(shape, center, normal, q, core=core, detune=g)
    if noise:
        rng = np.random.default_rng(seed)
        psi = psi + noise * (rng.standard_normal(shape)
                             + 1j * rng.standard_normal(shape))
    nhat = np.asarray(normal, float)
    nhat = nhat / np.linalg.norm(nhat)
    hist = {"time": [], "winding": [], "L": [], "align": [], "amp_min": []}
    for i in range(steps + 1):
        if i % record_every == 0:
            L = line_angular_momentum(psi, center)
            hist["time"].append(i * dt)
            hist["winding"].append(line_winding(psi, center, normal_axis,
                                                winding_radius))
            hist["L"].append(L.tolist())
            hist["align"].append(float(abs(L @ nhat) / (np.linalg.norm(L) + 1e-12)))
            hist["amp_min"].append(float(np.abs(psi).min()))
        psi = step_chiral_detuned(psi, g, chiral=chiral, dt=dt)
    hist["psi"] = psi
    return hist
