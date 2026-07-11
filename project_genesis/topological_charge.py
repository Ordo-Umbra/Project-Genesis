"""Geometric topological charge for the ψ∈ℂ^N (CP^(N-1)) sector field.

The normalised sector field ``ψ(x) ∈ ℂ^N`` (a unit vector, physics invariant
under an overall local phase ``ψ → e^{iθ(x)}ψ``) is a **CP^(N-1) field**.  The
2-D CP^(N-1) model is the textbook laboratory for the non-perturbative
structure the URP papers invoke for the QCD vacuum — it is asymptotically
free, confines, has a dynamical mass gap, a θ-vacuum, and genuine
**instantons** carrying an integer topological charge ``Q ∈ π₂(CP^(N-1)) =
ℤ``.  This module measures that charge, so the "instanton content" of the
sector field can be read off directly.

The estimator is the **geometric (Berg–Lüscher) construction**, which is
exactly integer on any configuration and invariant under the local phase
(the CP gauge freedom).  Each lattice square is split into two triangles;
the charge of a triangle with corners ``z₁, z₂, z₃`` is the signed area of
the geodesic triangle they span on CP^(N-1), divided by 2π:

    q(△) = (1/2π) · arg[ (z̄₁·z₂)(z̄₂·z₃)(z̄₃·z₁) ] ,   arg ∈ (−π, π]

Summing the two triangles gives the charge in a plaquette; summing all
plaquettes on the periodic lattice gives the total charge ``Q``, an integer.
The topological susceptibility ``χ_top = ⟨Q²⟩ / V`` measures the vacuum's
instanton activity.
"""

from __future__ import annotations

import numpy as np

_TWO_PI = 2.0 * np.pi


def _triangle_charge(z1: np.ndarray, z2: np.ndarray, z3: np.ndarray) -> np.ndarray:
    """Signed geodesic-triangle area / 2π for CP^(N-1) corners (vectorised).

    ``z1, z2, z3`` are ``(..., N)`` complex arrays of (not necessarily unit)
    vectors.  Returns the per-site triangle charge in ``(-½, ½]``.  Invariant
    under an independent phase on each corner (the CP gauge freedom) because
    the three overlaps' phases telescope.
    """
    o12 = np.sum(np.conj(z1) * z2, axis=-1)
    o23 = np.sum(np.conj(z2) * z3, axis=-1)
    o31 = np.sum(np.conj(z3) * z1, axis=-1)
    return np.angle(o12 * o23 * o31) / _TWO_PI


def topological_charge_density(psi: np.ndarray) -> np.ndarray:
    """Per-plaquette geometric charge of a 2-D CP^(N-1) field ``(H, W, N)``.

    Returns an ``(H, W)`` real array; the plaquette anchored at ``(y, x)`` is
    the unit square with corners ``(y,x), (y,x+1), (y+1,x+1), (y+1,x)`` on the
    periodic lattice, triangulated as ``[00,10,11] + [00,11,01]``.
    """
    if psi.ndim != 3:
        raise ValueError("topological_charge_density expects a 2-D field (H, W, N)")
    z00 = psi
    z10 = np.roll(psi, -1, axis=1)   # (y, x+1)
    z01 = np.roll(psi, -1, axis=0)   # (y+1, x)
    z11 = np.roll(np.roll(psi, -1, axis=0), -1, axis=1)  # (y+1, x+1)
    return (_triangle_charge(z00, z10, z11)
            + _triangle_charge(z00, z11, z01))


def topological_charge(psi: np.ndarray) -> float:
    """Total geometric topological charge ``Q`` (integer on a periodic lattice)."""
    return float(topological_charge_density(psi).sum())


def topological_susceptibility(charges, volume: float) -> float:
    """``χ_top = ⟨Q²⟩ / V`` from a sample of total charges over the ensemble."""
    q = np.asarray(charges, dtype=float)
    if q.size == 0:
        return 0.0
    return float(np.mean(q ** 2) / volume)


def instanton_density(psi: np.ndarray) -> float:
    """Mean absolute charge per site — the (anti-)instanton activity density."""
    return float(np.abs(topological_charge_density(psi)).mean())


def cp_action(psi: np.ndarray) -> float:
    """Gauge-invariant CP^(N-1) lattice action ``Σ_x Σ_μ (1 − |z̄·z'|²)``."""
    s = 0.0
    for axis in (0, 1):
        nb = np.roll(psi, -1, axis=axis)
        s += float((1.0 - np.abs(np.sum(np.conj(psi) * nb, axis=-1)) ** 2).sum())
    return s


def cp_action_density(psi: np.ndarray) -> np.ndarray:
    """Per-site CP action density ``e(x) = Σ_μ (1 − |z̄(x)·z(x+μ)|²)`` (forward links).

    Sums the two forward links at each site, so ``e.sum()`` equals
    :func:`cp_action`.  The energy density against which the topological charge
    density is compared in the Bogomolny coherent fraction.
    """
    e = np.zeros(psi.shape[:-1])
    for axis in (0, 1):
        nb = np.roll(psi, -1, axis=axis)
        e += 1.0 - np.abs(np.sum(np.conj(psi) * nb, axis=-1)) ** 2
    return e


def coherent_fraction(action_density: np.ndarray, charge_density: np.ndarray) -> float:
    """The Bogomolny coherent fraction ``Σ|q(x)| / Σ e(x)`` of a field.

    One dimensionless operator, applicable to any sector with a topological
    charge density ``q`` and an action density ``e`` obeying the Bogomolny bound
    ``Σe ≥ Σ|q|`` (so the fraction is in ``[0, 1]``): the share of the action
    carried by topologically coherent (self-dual) structure — the URP
    integration/exchange fraction ``κ``.  It is the same operator that
    ``gauge_topology.self_dual_fraction`` computes for the 4-D SU(N) field and
    that :func:`cp_coherent_fraction` computes for the 2-D CP sector.
    """
    e = float(np.asarray(action_density).sum())
    if e == 0.0:
        return 0.0
    q = float(np.abs(np.asarray(charge_density)).sum())
    return q / e


def cp_coherent_fraction(psi: np.ndarray) -> float:
    """The Bogomolny coherent fraction of a CP field — :func:`coherent_fraction`."""
    return coherent_fraction(cp_action_density(psi), topological_charge_density(psi))


def charge_variance_per_action(charges, actions) -> float:
    """Dimensionless topological invariant ``⟨Q²⟩ / ⟨S⟩`` (χ_top / action density).

    Both ``⟨Q²⟩/V`` (the topological susceptibility) and ``⟨S⟩/V`` (the action
    density) carry dimension ``L^{-d}``, so their ratio is dimensionless in *any*
    dimension — a scale-robust candidate for a sector-independent ``κ``.  Takes
    per-configuration total charges ``Q`` and total actions ``S`` over an
    ensemble.  (The one-κ obstruction experiment finds this also does not
    coincide between the SU(3) and CP sectors — see ``n3_kappa_obstruction``.)
    """
    q = np.asarray(charges, dtype=float)
    s = np.asarray(actions, dtype=float)
    denom = float(s.mean())
    if denom == 0.0:
        return 0.0
    return float(np.mean(q ** 2) / denom)


def cp_metropolis_sweep(psi: np.ndarray, beta: float, rng: np.random.Generator,
                        epsilon: float = 0.5) -> np.ndarray:
    """One checkerboard Metropolis sweep of the CP^(N-1) field at coupling ``beta``.

    Proposes ``z' ∝ z + ε·(complex Gaussian)`` at every site of one colour at
    once (their neighbours are the other colour, held fixed, so the parallel
    update satisfies detailed balance) and accepts by the local action change
    ``exp(−β ΔS)``.  Generates a *thermal* CP vacuum — the sector analogue of a
    heat-bath gauge ensemble — as opposed to a cooled or hot-random field.
    """
    z = psi.copy()
    L0, L1 = z.shape[:2]
    yy, xx = np.meshgrid(np.arange(L0), np.arange(L1), indexing="ij")

    def site_action(field):
        s = np.zeros(field.shape[:-1])
        for axis in (0, 1):
            for shift in (-1, 1):
                nb = np.roll(field, shift, axis=axis)
                s += 1.0 - np.abs(np.sum(np.conj(nb) * field, axis=-1)) ** 2
        return s

    for colour in (0, 1):
        mask = ((yy + xx) % 2) == colour
        prop = z + epsilon * (rng.standard_normal(z.shape)
                              + 1j * rng.standard_normal(z.shape))
        prop = prop / np.linalg.norm(prop, axis=-1, keepdims=True)
        s_old = site_action(z)
        trial = z.copy()
        trial[mask] = prop[mask]
        s_new = site_action(trial)
        accept = (rng.random((L0, L1)) < np.exp(-beta * (s_new - s_old))) & mask
        z[accept] = prop[accept]
    return z


def cool_step(psi: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """One cooling sweep: move each ``z(x)`` toward the local action-minimiser.

    Minimising the CP action at a site means maximising
    ``Σ_neighbours |z̄(x)·z_nb|²``, whose solution is the leading eigenvector
    of ``M(x) = Σ_neighbours z_nb z_nb†``.  Cooling removes short-wavelength
    (UV) lattice noise — the dislocations that inflate the raw geometric
    charge — while leaving well-separated instantons intact, the standard
    lattice route to *physical* topological content.

    ``rate`` interpolates toward that minimiser: ``rate=1`` snaps to it (a
    full sweep, the default); ``rate<1`` takes a gentle step ``z ← (1−rate)·z
    + rate·v``, so a ladder of small reflections integrates the field
    gradually rather than all at once.
    """
    H, W, N = psi.shape
    M = np.zeros((H, W, N, N), dtype=np.complex128)
    for axis in (0, 1):
        for shift in (-1, 1):
            nb = np.roll(psi, shift, axis=axis)
            M += nb[..., :, None] * np.conj(nb[..., None, :])
    _, vecs = np.linalg.eigh(M)          # ascending eigenvalues
    v = vecs[..., :, -1]                  # leading eigenvector per site
    if rate < 1.0:
        # align v's phase with z before mixing (the CP freedom), then blend
        phase = np.sum(np.conj(v) * psi, axis=-1)
        phase = np.where(np.abs(phase) > 0, phase / np.abs(phase), 1.0)
        v = v * phase[..., None]
        z = (1.0 - rate) * psi + rate * v
    else:
        z = v
    return z / np.linalg.norm(z, axis=-1, keepdims=True)


def cool(psi: np.ndarray, n_steps: int, rate: float = 1.0) -> np.ndarray:
    """Apply ``n_steps`` cooling sweeps at the given ``rate`` (fresh array)."""
    z = psi.copy()
    for _ in range(n_steps):
        z = cool_step(z, rate)
    return z
