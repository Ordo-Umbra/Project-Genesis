"""Nematic defects in three dimensions: the order parameter is RP², not RP¹.

The programme's spinor and spin–statistics work is built on a complex field
``ψ`` with director ``θ = ½·arg ψ``.  A complex phase is a point on a circle, so
that construction describes a director confined to a plane — order parameter
space ``RP¹``, with

    π₁(RP¹) = **Z**

and therefore an integer-graded ladder of defects: ``±½, ±1, ±3/2, …`` are all
distinct, and ``+½`` and ``−½`` are different objects that annihilate each other.

A director free to point anywhere in three dimensions lives on the sphere with
antipodes identified — ``RP²`` — and there

    π₁(RP²) = **Z/2**

which is a different world with only *two* classes.  Consequences, all standard
liquid-crystal physics:

- a **half-integer** disclination is the non-trivial class: genuinely singular
  and topologically protected;
- an **integer** disclination is the *trivial* class: not protected, and it
  removes its own singularity by tilting the director along the line — the
  classic **escape into the third dimension**;
- ``+½`` and ``−½`` are the **same class**, so a half-integer line is its own
  antiparticle and two like-signed ones can annihilate.

None of that is available to a ``ψ``-based model, which is why this module uses
a real three-component director on a lattice rather than reusing
:mod:`project_genesis.nematic_spinor`.

**Geometry.**  A straight disclination line along ``z`` has a director field
that does not depend on ``z``, so the lattice is 2-D while the director keeps
all three components.  "Escape into the third dimension" then shows up as
``n_z`` becoming non-zero at the core — the director tilting along the line.
Setting ``confine_plane=True`` pins ``n_z = 0`` and recovers the ``RP¹`` world
the existing modules describe, so both arms run on one code path and the only
difference is whether the third component is allowed to exist.

**Dynamics.**  One-constant Lebwohl–Lasher relaxation.  The energy
``E = −Σ_⟨ij⟩ P₂(nᵢ·nⱼ)`` depends on ``nᵢ·nⱼ`` only through its square, so it is
invariant under ``n → −n`` at every site independently — headless by
construction rather than by convention.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "imprint_disclination",
    "tilted_core",
    "relax",
    "elastic_energy",
    "loop_points",
    "loop_z2_class",
    "loop_strength",
    "core_escape",
    "sample_director",
]


# --------------------------------------------------------------------- setup

def imprint_disclination(size: int, strength, centres=None, rng=None,
                         noise: float = 0.0) -> np.ndarray:
    """A director field carrying disclination line(s) of the given strength.

    ``strength`` may be a single value or one per centre.  The in-plane
    director angle is ``θ = Σ_k s_k · φ_k`` with ``φ_k`` the azimuth about
    centre ``k`` — the standard planar disclination texture, which is a valid
    starting configuration in **both** worlds since it has ``n_z = 0``.

    Whether it *stays* a valid configuration is the measurement.
    """
    strengths = np.atleast_1d(np.asarray(strength, dtype=float))
    if centres is None:
        centres = [(size / 2.0, size / 2.0)] * len(strengths)
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    if len(centres) != len(strengths):
        raise ValueError("need one centre per strength")

    y, x = np.indices((size, size)).astype(float)
    theta = np.zeros((size, size))
    for s, (cx, cy) in zip(strengths, centres):
        theta = theta + s * np.arctan2(y - cy, x - cx)

    n = np.zeros((3, size, size))
    n[0] = np.cos(theta)
    n[1] = np.sin(theta)
    if noise and rng is not None:
        n = n + noise * rng.standard_normal(n.shape)
    return _normalise(n)


def tilted_core(size: int, strength: float, core: float = 6.0) -> np.ndarray:
    """The strength-``s`` texture with its core deliberately tipped out of plane.

    ``n = (sin β·cos sφ, sin β·sin sφ, cos β)`` with ``β`` running from 0 on the
    axis to ``π/2`` by ``r ≈ core``: the director points *along* the line at the
    centre and lies in the plane far away.

    For ``s = 1`` this is the textbook **escaped radial** configuration, and it
    is non-singular.  For ``s = ½`` the same construction does *not* remove the
    singularity, because the class is non-trivial and no continuous texture can.

    Starting a relaxation here rather than from the flat texture is the point:
    the flat texture is a symmetric saddle where zero-temperature gradient
    descent has no direction to fall, so it reveals nothing.  Seeded off the
    saddle, the dynamics answers the question directly — the tilt **grows** if
    escape is downhill and **decays** if the defect is protected.
    """
    y, x = np.indices((size, size)).astype(float)
    cx = cy = size / 2.0
    r = np.hypot(x - cx, y - cy)
    phi = np.arctan2(y - cy, x - cx)
    beta = np.minimum(2.0 * np.arctan(r / float(core)), np.pi / 2.0)
    th = float(strength) * phi
    n = np.stack([np.sin(beta) * np.cos(th),
                  np.sin(beta) * np.sin(th),
                  np.cos(beta)])
    return _normalise(n)


def _normalise(n: np.ndarray) -> np.ndarray:
    return n / np.maximum(np.sqrt((n * n).sum(axis=0, keepdims=True)), 1e-12)


def _neighbours(n: np.ndarray):
    return [np.roll(n, 1, 1), np.roll(n, -1, 1), np.roll(n, 1, 2), np.roll(n, -1, 2)]


# ------------------------------------------------------------------ dynamics

def elastic_energy(n: np.ndarray) -> float:
    """Mean Lebwohl–Lasher energy ``−⟨P₂(nᵢ·nⱼ)⟩`` over lattice bonds.

    Only ``(nᵢ·nⱼ)²`` enters, so flipping any single director changes nothing —
    the headlessness is in the energy, not in a convention applied afterwards.
    """
    tot = 0.0
    for m in (np.roll(n, -1, 1), np.roll(n, -1, 2)):
        dot = (n * m).sum(axis=0)
        tot += float(np.mean(0.5 * (3.0 * dot * dot - 1.0)))
    return -tot / 2.0


def relax(n: np.ndarray, steps: int, *, dt: float = 0.1,
          confine_plane: bool = False, pin_radius: float = 0.0,
          noise: float = 0.0, rng=None) -> np.ndarray:
    """Gradient relaxation of the Lebwohl–Lasher energy on the unit sphere.

    ``confine_plane=True`` pins ``n_z = 0`` after every step, which is exactly
    the ``RP¹`` world: the director cannot leave the plane, so an integer
    disclination has nowhere to escape to.

    ``pin_radius > 0`` holds the far-field boundary fixed at its imprinted
    texture, so a defect cannot simply be swept out through the edges — the
    boundary condition is what makes the topological charge meaningful at all.
    """
    n = _normalise(np.array(n, dtype=float, copy=True))
    size = n.shape[-1]
    if confine_plane:
        n[2] = 0.0
        n = _normalise(n)

    mask = None
    if pin_radius > 0:
        y, x = np.indices((size, size)).astype(float)
        r = np.hypot(x - size / 2.0, y - size / 2.0)
        mask = r >= pin_radius
        frozen = n[:, mask].copy()

    for _ in range(int(steps)):
        # molecular field: gradient of  ½ Σ (n_i·n_j)²  w.r.t. n_i
        h = np.zeros_like(n)
        for m in _neighbours(n):
            h = h + (n * m).sum(axis=0) * m
        # project onto the tangent plane of the sphere; radial part is a no-op
        h = h - (h * n).sum(axis=0) * n
        n = n + dt * h
        if noise and rng is not None:
            n = n + noise * rng.standard_normal(n.shape)
        if confine_plane:
            n[2] = 0.0
        n = _normalise(n)
        if mask is not None:
            n[:, mask] = frozen
    return n


# -------------------------------------------------------------- measurement

def loop_points(centre, radius: float, npts: int = 512):
    """Sample points on a circle — the loop whose homotopy class we read."""
    th = np.linspace(0.0, 2.0 * np.pi, int(npts), endpoint=False)
    return (centre[0] + radius * np.cos(th), centre[1] + radius * np.sin(th))


def sample_director(n: np.ndarray, xs, ys) -> np.ndarray:
    """Nearest-site director samples, as ``(npts, 3)``."""
    size = n.shape[-1]
    xi = np.clip(np.rint(xs).astype(int), 0, size - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, size - 1)
    return n[:, yi, xi].T


def loop_z2_class(n: np.ndarray, centre, radius: float, npts: int = 512) -> int:
    """The π₁(RP²) invariant: does the director come back flipped?

    Walk the loop choosing, at each step, whichever of ``±n`` is closer to the
    previous choice — the only continuous lift available for a headless field.
    If the lift closes up on ``−n_start`` the loop is **not** contractible and
    the class is 1; if on ``+n_start`` it is 0.

    This is the whole of ``π₁(RP²) = Z/2``, and it is exactly the measurement a
    complex-phase model cannot make, because there the lift is forced to live
    on a circle.
    """
    xs, ys = loop_points(centre, radius, npts)
    d = sample_director(n, xs, ys)
    prev = d[0].copy()
    start = prev.copy()
    for k in range(1, len(d)):
        v = d[k]
        if float(v @ prev) < 0.0:
            v = -v
        prev = v
    return int(float(prev @ start) < 0.0)


def loop_strength(n: np.ndarray, centre, radius: float, npts: int = 512) -> float:
    """In-plane disclination strength ``s`` — the *integer-graded* charge.

    Accumulates the signed in-plane angle change along the lift, taking the
    head–tail ambiguity mod π.  Meaningful only while the director stays in the
    plane; once it escapes, the in-plane angle is not the whole story and this
    number stops being the invariant.  Reported alongside the Z/2 class so the
    difference between the two gradings is visible rather than assumed.
    """
    xs, ys = loop_points(centre, radius, npts)
    d = sample_director(n, xs, ys)
    ang = np.arctan2(d[:, 1], d[:, 0])
    total = 0.0
    for k in range(len(ang)):
        step = ang[(k + 1) % len(ang)] - ang[k]
        step = (step + np.pi / 2) % np.pi - np.pi / 2      # head–tail: mod π
        total += step
    return float(total / (2.0 * np.pi))


def core_escape(n: np.ndarray, centre, radius: float = 4.0) -> float:
    """Mean ``|n_z|`` inside the core — the escape order parameter.

    Zero means the director is still lying in the plane and the line is
    singular.  Approaching one means it has tilted along the line axis, which
    is precisely how an unprotected disclination removes its own singularity.
    """
    size = n.shape[-1]
    y, x = np.indices((size, size)).astype(float)
    disc = np.hypot(x - centre[0], y - centre[1]) <= float(radius)
    if not disc.any():
        return 0.0
    return float(np.abs(n[2][disc]).mean())
