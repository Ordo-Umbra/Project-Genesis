"""The corpus of stable forms: topological solitons of the ψ∈ℂ³ sector field.

The URP reading of matter: a *particle* is not an arbitrary lump of field but
a **stable structure made manifest** — a localized configuration the dynamics
cannot smoothly undo.  In the CP² sector field that is exactly a **topological
soliton**: a winding of ψ carrying an integer charge ``Q ∈ π₂(CP²) = ℤ``.
Because Q cannot change continuously, such forms are protected; because the
CP action obeys a Bogomolny bound ``E ≥ (const)·|Q|``, a charge-Q form has an
*energy floor* it cannot decay below.  The admissible forms therefore make a
**discrete corpus** indexed by Q — a mass spectrum — and their energies are
their **structural (inertial) masses**.

This module builds that corpus and measures two masses of each form:

- ``structural_mass`` — the form's own CP-action energy (what it costs to
  build the structure), the inertial mass ``E``.
- ``gravitational_mass`` — the strength of the κ-well the form sources, read
  through the capacity-gravity dynamics (``capacity_gravity``), i.e. how hard
  the form *gravitates*.

Both are rooted in the same quantity — the form's **distinction density**
``d(x) = Σ_μ (1 − |ψ̄(x)·ψ(x+μ)|²)`` (the local winding/gradient content) —
which is why the framework *predicts* their equality: inertial and
gravitational mass coincide because each is the amount of stable structure
present.  That is the equivalence principle, explained rather than imposed.
"""

from __future__ import annotations

import numpy as np

from .capacity_gravity import relax_capacity
from .topological_charge import cp_action, topological_charge


def winding_form(L: int, charge: int, ring: float = 3.0,
                 sector: tuple[int, int] = (0, 1)) -> np.ndarray:
    """A charge-``Q`` CP² soliton via the rational map ``w = ((x+iy)/λ)^|Q|``.

    Embedded in ℂ³ on the ``sector`` pair of colour components.  ``λ = ring·
    |Q|`` keeps the winding ring's *width* roughly Q-independent, so forms of
    different charge are comparably resolved on the lattice (the BPS energy
    ladder ``E ∝ |Q|`` then shows cleanly).  Charge 0 returns the vacuum.
    """
    a, b = sector
    xs = np.arange(L) - L / 2.0 + 0.5
    X, Y = np.meshgrid(xs, xs)
    z = np.zeros((L, L, 3), dtype=complex)
    z[..., a] = 1.0
    if charge != 0:
        lam = ring * abs(charge)
        w = ((X + 1j * Y) / lam) ** abs(charge)
        z[..., b] = np.conj(w) if charge < 0 else w
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    return z


def trivial_bump(L: int, amplitude: float = 0.8, width: float = 5.0,
                 sector: tuple[int, int] = (0, 1)) -> np.ndarray:
    """A topologically trivial (Q = 0) localized deformation — *not* a form.

    A Gaussian bulge of ψ that carries no winding; it can relax back to the
    vacuum (no energy floor), the control against which the topological forms'
    stability is judged.
    """
    a, b = sector
    xs = np.arange(L) - L / 2.0 + 0.5
    X, Y = np.meshgrid(xs, xs)
    z = np.zeros((L, L, 3), dtype=complex)
    z[..., a] = 1.0
    z[..., b] = amplitude * np.exp(-(X ** 2 + Y ** 2) / (2.0 * width ** 2))
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    return z


def distinction_density(psi: np.ndarray) -> np.ndarray:
    """Per-site distinction ``d(x) = Σ_μ (1 − |ψ̄(x)·ψ(x+μ)|²)`` (the form's mass density).

    The local winding/gradient content of the sector field — what the κ
    capacity dynamics consumes as ``load``, and what the CP action sums to the
    structural energy.  Its total is exactly :func:`structural_mass`.
    """
    if psi.ndim != 3:
        raise ValueError("distinction_density expects a 2-D CP field (H, W, N)")
    d = np.zeros(psi.shape[:-1])
    for axis in (0, 1):
        nb = np.roll(psi, -1, axis=axis)
        d += 1.0 - np.abs(np.sum(np.conj(psi) * nb, axis=-1)) ** 2
    return d


def structural_mass(psi: np.ndarray) -> float:
    """The form's inertial mass ``E`` — its total CP-action energy."""
    return cp_action(psi)


def gravitational_mass(psi: np.ndarray, *, kappa_diffusion: float = 1.0,
                       kappa_recovery: float = 0.05,
                       kappa_consumption: float = 0.2,
                       kappa_baseline: float = 1.0, dt: float = 0.1,
                       max_iters: int = 6000, tol: float = 1e-8) -> float:
    """The form's gravitational mass — the total κ-deficit it sources.

    Relaxes the capacity field with the form's distinction density as the
    ``load`` source and returns the integrated well ``Σ_x (κ₀ − κ)`` — the
    monopole strength of the κ-well, i.e. how strongly the form gravitates.
    At weak ``kappa_consumption`` the well is linear and this is proportional
    to the structural mass; at strong consumption κ saturates (bottoms out at
    0) and the gravitational mass grows *sub-linearly* — a genuine nonlinear
    departure from the structural mass, not a redefinition of it.
    """
    load = distinction_density(psi)
    kappa = relax_capacity(load, kappa_diffusion=kappa_diffusion,
                           kappa_recovery=kappa_recovery,
                           kappa_consumption=kappa_consumption,
                           kappa_baseline=kappa_baseline, dt=dt,
                           max_iters=max_iters, tol=tol)
    return float((kappa_baseline - kappa).sum())


def form_charge(psi: np.ndarray) -> float:
    """The form's topological charge ``Q`` (integer for a genuine soliton)."""
    return topological_charge(psi)
