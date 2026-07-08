"""Field-theoretic (clover) topological charge for 4-D SU(N) gauge fields.

The 2-D sector field carries CP² instantons (``topological_charge.py``); the
genuine article the URP functorial-bridge paper points at is the **4-D
SU(3) θ-vacuum**, whose instanton content is the physical integrator that
binds the degenerate winding sectors.  This module gives the gauge sector
the instrument it lacked there: the **clover** definition of the
topological charge,

    Q = (1/32π²) Σ_x ε_{μνρσ} Tr[ F_{μν}(x) F_{ρσ}(x) ] ,

with ``F_{μν}`` the clover-averaged field strength — the traceless
anti-Hermitian part of the sum of the four plaquettes ("leaves") in the
μ-ν plane that share the site ``x``.  The clover average is symmetric about
``x`` (so the charge transforms correctly and is less noisy than the
single-plaquette definition).

The field-theoretic charge is only *approximately* integer on a rough
lattice — UV dislocations shift it — so, exactly as in 2-D, the physical
topology is read after **cooling** (here, replacing each link by the SU(N)
projection of its staple sum, which descends the Wilson action while
preserving smooth instantons).  After cooling ``Q`` clusters at evenly
spaced levels: the trivial sector sits at exactly ``0``, and topological
sectors at ``Z·n`` with a multiplicative lattice renormalisation
``Z ≲ 1`` (≈ 0.85 on a coarse lattice, tending to 1 as the config is
smoothed further or the lattice is refined).  That quantised level
structure — 0 exact, higher sectors evenly spaced — is the instrument's
validation; the precise integer normalisation and the instanton/condensate
split that would test the framework's κ ≈ 0.22 need gradient flow,
scale-setting, and larger lattices (a further stage).
"""

from __future__ import annotations

import numpy as np

from .gauge import _dagger, plaquette, traceless_antihermitian

_NORM = 1.0 / (32.0 * np.pi ** 2)


def _leaf_plaquettes(links: np.ndarray, mu: int, nu: int) -> np.ndarray:
    """Clover sum ``C_{μν}(x)`` — the four plaquettes in the μ-ν plane at ``x``.

    The leaves are the plaquettes based at ``x`` occupying the four
    quadrants ``(+μ+ν), (+ν−μ), (−μ−ν), (−ν+μ)``; their sum is symmetric
    under μ↔ reflections about ``x``.
    """
    u_mu, u_nu = links[mu], links[nu]
    dag = _dagger

    def s(a, axis, step):          # s(a,axis,+1)=a(x−ê); s(a,axis,−1)=a(x+ê)
        return np.roll(a, shift=step, axis=axis)

    def s2(a, ax1, st1, ax2, st2):
        return np.roll(np.roll(a, st1, axis=ax1), st2, axis=ax2)

    # leaf 1 (+μ,+ν): U_μ(x) U_ν(x+μ) U_μ(x+ν)† U_ν(x)†
    l1 = u_mu @ s(u_nu, mu, -1) @ dag(s(u_mu, nu, -1)) @ dag(u_nu)
    # leaf 2 (+ν,−μ): U_ν(x) U_μ(x+ν−μ)† U_ν(x−μ)† U_μ(x−μ)
    l2 = u_nu @ dag(s2(u_mu, nu, -1, mu, 1)) @ dag(s(u_nu, mu, 1)) @ s(u_mu, mu, 1)
    # leaf 3 (−μ,−ν): U_μ(x−μ)† U_ν(x−μ−ν)† U_μ(x−μ−ν) U_ν(x−ν)
    l3 = (dag(s(u_mu, mu, 1)) @ dag(s2(u_nu, mu, 1, nu, 1))
          @ s2(u_mu, mu, 1, nu, 1) @ s(u_nu, nu, 1))
    # leaf 4 (−ν,+μ): U_ν(x−ν)† U_μ(x−ν) U_ν(x+μ−ν) U_μ(x)†
    l4 = dag(s(u_nu, nu, 1)) @ s(u_mu, nu, 1) @ s2(u_nu, mu, -1, nu, 1) @ dag(u_mu)
    return l1 + l2 + l3 + l4


def clover_field_strength(links: np.ndarray, mu: int, nu: int) -> np.ndarray:
    """Clover field strength ``F_{μν}(x)`` — traceless anti-Hermitian, /8."""
    c = _leaf_plaquettes(links, mu, nu)
    return traceless_antihermitian(c) / 4.0     # (C − C†)/2 · 1/4 = (C − C†)/8


def topological_charge_density(links: np.ndarray) -> np.ndarray:
    """Per-site clover topological charge density ``q(x)`` (4-D SU(N))."""
    if links.shape[0] != 4:
        raise ValueError("topological charge is defined here for 4-D links")
    f01 = clover_field_strength(links, 0, 1)
    f23 = clover_field_strength(links, 2, 3)
    f02 = clover_field_strength(links, 0, 2)
    f13 = clover_field_strength(links, 1, 3)
    f03 = clover_field_strength(links, 0, 3)
    f12 = clover_field_strength(links, 1, 2)

    def tr(a, b):
        return np.real(np.trace(a @ b, axis1=-2, axis2=-1))

    # ε_{μνρσ} Tr[F F] = 8 Re Tr[F01 F23 − F02 F13 + F03 F12]
    return _NORM * 8.0 * (tr(f01, f23) - tr(f02, f13) + tr(f03, f12))


def topological_charge(links: np.ndarray) -> float:
    """Total clover topological charge ``Q`` (≈ integer after cooling)."""
    return float(topological_charge_density(links).sum())


def topological_susceptibility(charges, volume: float) -> float:
    """``χ_top = ⟨Q²⟩ / V`` from a sample of total charges over the ensemble."""
    q = np.asarray(charges, dtype=float)
    if q.size == 0:
        return 0.0
    return float(np.mean(q ** 2) / volume)


def _project_su(m: np.ndarray) -> np.ndarray:
    """Project stacked matrices onto SU(N): unitarise (polar) then fix det=1."""
    # polar factor U = M (M†M)^(−1/2) via SVD (stable, vectorised)
    u, _, vh = np.linalg.svd(m)
    w = u @ vh
    n = w.shape[-1]
    det = np.linalg.det(w)
    phase = det ** (1.0 / n)
    return w / phase[..., None, None]


def staple_field(links: np.ndarray, mu: int) -> np.ndarray:
    """Vectorised sum of staples ``A_μ(x)`` for every site (Wilson action).

    Matches the per-site ``gauge_mc._staple_sum`` convention: the plaquettes
    touching ``U_μ(x)`` factorise as ``Re Tr[U_μ(x)·A_μ(x)]``.  ``s(a,ax,+1)``
    is the field at ``x−ê``, ``s(a,ax,−1)`` at ``x+ê`` (periodic).
    """
    ndim = links.shape[0]
    u_mu = links[mu]
    dag = _dagger

    def s(a, ax, st):
        return np.roll(a, st, axis=ax)

    def s2(a, a1, s1, a2, s2_):
        return np.roll(np.roll(a, s1, axis=a1), s2_, axis=a2)

    A = np.zeros_like(u_mu)
    for nu in range(ndim):
        if nu == mu:
            continue
        u_nu = links[nu]
        # up:   U_ν(x+μ) U_μ(x+ν)† U_ν(x)†
        A += s(u_nu, mu, -1) @ dag(s(u_mu, nu, -1)) @ dag(u_nu)
        # down: U_ν(x+μ−ν)† U_μ(x−ν)† U_ν(x−ν)
        A += dag(s2(u_nu, mu, -1, nu, 1)) @ dag(s(u_mu, nu, 1)) @ s(u_nu, nu, 1)
    return A


def cool_step(links: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """One cooling sweep: move each link toward the SU(N) action-minimiser.

    Maximising ``Re Tr[U_μ(x) A_μ(x)]`` (descending the Wilson action) at
    fixed neighbours is solved by ``U = Proj_SU(A†)``.  ``rate=1`` snaps to
    it; ``rate<1`` takes a gentle step ``U ← Proj_SU((1−rate)·U + rate·U*)``,
    so small reflections remove UV dislocations while preserving smooth
    instantons rather than shrinking them.
    """
    ndim = links.shape[0]
    out = links.copy()
    for mu in range(ndim):
        target = _project_su(_dagger(staple_field(out, mu)))
        out[mu] = target if rate >= 1.0 else _project_su(
            (1.0 - rate) * out[mu] + rate * target)
    return out


def cool(links: np.ndarray, n_steps: int, rate: float = 1.0) -> np.ndarray:
    """Apply ``n_steps`` cooling sweeps at the given ``rate`` (fresh array)."""
    z = links.copy()
    for _ in range(n_steps):
        z = cool_step(z, rate)
    return z


def mean_plaquette(links: np.ndarray) -> float:
    """Average ``(1/N) Re Tr P_{μν}`` over all plaquettes — the Wilson action density."""
    ndim = links.shape[0]
    n = links.shape[-1]
    tot, cnt = 0.0, 0
    for mu in range(ndim):
        for nu in range(mu + 1, ndim):
            p = plaquette(links, mu, nu)
            tot += float(np.real(np.trace(p, axis1=-2, axis2=-1)).mean()) / n
            cnt += 1
    return tot / cnt
