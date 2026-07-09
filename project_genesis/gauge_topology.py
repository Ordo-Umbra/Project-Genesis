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

from .gauge import (
    _dagger,
    _expm_antihermitian,
    plaquette,
    traceless_antihermitian,
)

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


# --------------------------------------------------------------------------
# Stage 2: Wilson gradient flow, scale-setting, and the self-dual fraction
# --------------------------------------------------------------------------
#
# Cooling (above) is a crude smoother — it snaps each link to a local
# action-minimiser, with no controlled step size, so the amount of smoothing
# is not a renormalisation-group-clean scale.  The **Wilson (gradient) flow**
# replaces it with a proper continuous smoothing: the links evolve by the
# gradient-descent equation of the Wilson action,
#
#     dV_t/dt = − { ∂_{x,μ} S_W(V_t) } V_t ,
#
# integrated in the flow "time" ``t`` (units of lattice-spacing²).  Lüscher
# showed the flow is a genuine RG transformation: observables at fixed
# physical ``√(8t)`` are finite and the field-theoretic topological charge
# flows to the *integer* value (the coarse-lattice ``Z ≲ 1`` of Stage 1
# becomes ``Z → 1``).  It also furnishes a scale: the dimensionless
# ``t² E(t)`` is a smooth, monotone clock, and the flow time ``t₀`` where it
# reaches the reference ``0.3`` sets the lattice spacing.


def _clover_sq_density(links: np.ndarray) -> np.ndarray:
    """Per-site ``Σ_{μ<ν} (−Re Tr[F_{μν}F_{μν}]) ≥ 0`` from the clover ``F``."""
    if links.shape[0] != 4:
        raise ValueError("clover action density is defined here for 4-D links")
    tot = None
    for mu in range(4):
        for nu in range(mu + 1, 4):
            f = clover_field_strength(links, mu, nu)
            term = -np.real(np.trace(f @ f, axis1=-2, axis2=-1))
            tot = term if tot is None else tot + term
    return tot


def flow_energy(links: np.ndarray) -> float:
    """Clover action density ``E = ⟨Σ_{μ<ν}(−Tr F_{μν}²)⟩`` (Lüscher normalisation).

    The quantity whose dimensionless combination ``t² E(t)`` defines the flow
    scale ``t₀``.
    """
    return float(_clover_sq_density(links).mean())


def action_density(links: np.ndarray) -> np.ndarray:
    """Per-site action density ``e(x) ≥ 0`` in the ``1/32π²`` topological units.

    Normalised (Bogomolny) so that ``e(x) ≥ |q(x)|`` pointwise, with equality
    iff the field is (anti-)self-dual at ``x`` — the denominator of the
    self-dual fraction.
    """
    return _NORM * 4.0 * _clover_sq_density(links)


def self_dual_fraction(links: np.ndarray) -> float:
    """Fraction of the field energy that is (anti-)self-dual: ``Σ|q| / Σe ∈ [0,1]``.

    The Bogomolny bound ``e(x) ≥ |q(x)|`` saturates on instantons, so this
    ratio is 1 for a purely self-dual field and small for structureless UV
    noise (whose local topological density averages away).  It is the
    lattice proxy for the **instanton fraction of the gluon condensate** —
    the physical content of the URP integration constant ``κ``.
    """
    e = float(action_density(links).sum())
    if e == 0.0:
        return 0.0
    q = float(np.abs(topological_charge_density(links)).sum())
    return q / e


def _flow_drift(links: np.ndarray) -> np.ndarray:
    """su(N) drift ``Z_μ(x) = −TA[U_μ(x) A_μ(x)]`` for every link (Wilson flow).

    ``A_μ`` is the staple sum, so ``U_μ A_μ`` is the sum of plaquette loops
    through the link; its traceless anti-Hermitian part is the action
    gradient.  The sign makes the flow *descend* the Wilson action.
    """
    out = np.empty_like(links)
    for mu in range(links.shape[0]):
        omega = links[mu] @ staple_field(links, mu)
        out[mu] = -traceless_antihermitian(omega)
    return out


def _flow_apply(z: np.ndarray, links: np.ndarray) -> np.ndarray:
    """Apply the group step ``V_μ ← exp(Z_μ) V_μ`` for every link."""
    out = np.empty_like(links)
    for mu in range(links.shape[0]):
        out[mu] = _expm_antihermitian(z[mu]) @ links[mu]
    return out


def flow_step(links: np.ndarray, dt: float) -> np.ndarray:
    """One Lüscher third-order Runge–Kutta gradient-flow step of size ``dt``.

    The RK3 integrator of Lüscher (2010): three drift evaluations combined so
    the step is accurate to ``O(dt³)`` while staying exactly on the group.
    """
    z0 = dt * _flow_drift(links)
    w1 = _flow_apply(0.25 * z0, links)
    z1 = dt * _flow_drift(w1)
    w2 = _flow_apply((8.0 / 9.0) * z1 - (17.0 / 36.0) * z0, w1)
    z2 = dt * _flow_drift(w2)
    return _flow_apply((3.0 / 4.0) * z2 - (8.0 / 9.0) * z1 + (17.0 / 36.0) * z0, w2)


def gradient_flow(links: np.ndarray, *, t_max: float, dt: float = 0.02,
                  measure_every: int = 5) -> tuple[np.ndarray, list[dict]]:
    """Integrate the Wilson flow to ``t_max``; return flowed links + trajectory.

    The trajectory records, every ``measure_every`` steps, the flow time and
    the observables that matter: the scale clock ``t² E(t)``, the (now nearly
    integer) topological charge ``Q(t)``, and the self-dual fraction.
    """
    z = links.copy()
    n_steps = max(1, int(round(t_max / dt)))
    traj = []

    def record(t):
        e = flow_energy(z)
        traj.append({"t": float(t), "E": e, "t2E": float(t * t * e),
                     "Q": topological_charge(z),
                     "f_sd": self_dual_fraction(z)})

    record(0.0)
    for step in range(1, n_steps + 1):
        z = flow_step(z, dt)
        if step % measure_every == 0 or step == n_steps:
            record(step * dt)
    return z, traj


def flow_scale(traj: list[dict], ref: float = 0.3) -> float:
    """Flow time ``t₀`` where ``t² E(t)`` first reaches ``ref`` (linear interp).

    Returns ``nan`` if the trajectory never reaches the reference value.
    """
    for a, b in zip(traj, traj[1:]):
        if a["t2E"] <= ref <= b["t2E"] and b["t2E"] > a["t2E"]:
            frac = (ref - a["t2E"]) / (b["t2E"] - a["t2E"])
            return float(a["t"] + frac * (b["t"] - a["t"]))
    return float("nan")


def continuum_limit(cutoff, values, weights=None) -> dict:
    """Linear extrapolation of an observable to the continuum (``cutoff → 0``).

    Lattice artifacts are ``O(a²)``, so a set of measurements at cutoff
    surrogates ``x_i`` (e.g. ``a² ∝ 1/t₀`` in lattice units) is fit to a line
    ``y = intercept + slope·x``; the intercept is the ``a → 0`` value.  With
    ``weights`` (e.g. ``1/σ²``) the fit is weighted.  Returns the intercept,
    slope, and the fitted line — the honest continuum trend, whatever it is.
    """
    x = np.asarray(cutoff, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.size < 2:
        raise ValueError("need at least two points to extrapolate")
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    W = w.sum()
    mx = (w * x).sum() / W
    my = (w * y).sum() / W
    sxx = (w * (x - mx) ** 2).sum()
    sxy = (w * (x - mx) * (y - my)).sum()
    slope = sxy / sxx
    intercept = my - slope * mx
    return {"intercept": float(intercept), "slope": float(slope)}
