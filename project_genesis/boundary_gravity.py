"""Does the boundary know the interior?  A Gauss law for κ-gravity.

The area-law result (`n3_area_law`) measured that *coherence* is
boundary-limited — integration scales with the surface while distinction scales
with the volume.  It explicitly did **not** touch gravity, and named the sharper
question it set up: **is the bulk κ field — the thing that actually gravitates —
readable from its boundary?**

That question has a precise classical form: a **Gauss law**.  For Newtonian
gravity the flux of the field through *any* closed surface equals the enclosed
mass — independent of the surface's radius and of how the mass inside is
arranged.  That is exactly "the boundary processes the interior," and it is what
makes gravity holographic in the elementary sense.

The κ field relaxes to the steady state of
``∂_t κ = D∇²κ + r(κ₀−κ) − c·load·κ``, a **screened** (Yukawa) equation with
screening length ``ξ = √(D/r)``.  On the lattice the discrete divergence theorem
makes the surface flux exact as a bulk sum,

    ∮_{∂R} ∇κ·n̂  =  Σ_{x∈R} ∇²κ(x) ,

so :func:`enclosed_flux` needs no interpolation onto a contour.

What the instruments here measure is whether ``flux / enclosed-mass`` is a
**constant** — the signature of a Gauss law — as one varies:

- the surface radius (:func:`gauss_ratio_profile`),
- the arrangement of a fixed total mass,
- and the screening length itself.

The measured answer (see `n3_boundary_gravity.py`) is that κ-gravity has **no
Gauss law while it is screened** — the boundary reading decays like
``e^{−R/ξ}`` — and that the Gauss law is **restored as ``r → 0``**.  The
screening term ``r(κ₀−κ)`` — capacity self-maintenance, the same term from which
this programme *derives dark energy* — is precisely what blinds the boundary to
the interior.
"""

from __future__ import annotations

import numpy as np

from .multiphase import periodic_laplacian

__all__ = [
    "disc_indicator",
    "enclosed_flux",
    "enclosed_mass",
    "gauss_ratio_profile",
    "flatness",
]


def disc_indicator(shape, radius: float, center=None) -> np.ndarray:
    """Indicator of a centred disc of ``radius``."""
    n0, n1 = int(shape[0]), int(shape[1])
    if center is None:
        center = (n0 / 2.0, n1 / 2.0)
    y, x = np.indices((n0, n1))
    return (np.hypot(x - center[1], y - center[0]) < radius).astype(float)


def enclosed_flux(kappa: np.ndarray, radius: float, center=None) -> float:
    """``∮_{∂R} ∇κ·n̂`` over a disc of ``radius`` — exact via the divergence
    theorem.

    Equal to ``Σ_{x∈R} ∇²κ(x)`` on the periodic lattice, so no contour
    interpolation is needed.
    """
    chi = disc_indicator(kappa.shape, radius, center)
    return float((periodic_laplacian(kappa) * chi).sum())


def enclosed_mass(load: np.ndarray, radius: float, center=None) -> float:
    """Total load enclosed by the same disc."""
    chi = disc_indicator(load.shape, radius, center)
    return float((load * chi).sum())


def gauss_ratio_profile(kappa: np.ndarray, load: np.ndarray, radii,
                        center=None) -> list:
    """``flux(R) / enclosed_mass(R)`` for each radius.

    A **Gauss law** is the statement that this ratio is *independent of R*.
    Under screening it instead decays like ``e^{−R/ξ}``.
    """
    out = []
    for r in radii:
        m = enclosed_mass(load, r, center)
        out.append(enclosed_flux(kappa, r, center) / max(m, 1e-12))
    return out


def flatness(values) -> float:
    """``max/min`` of a profile — 1.0 is a perfect Gauss law, large is none."""
    v = np.abs(np.asarray(values, dtype=float))
    return float(v.max() / max(v.min(), 1e-12))
