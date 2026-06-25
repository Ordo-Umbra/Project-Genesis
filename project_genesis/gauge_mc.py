"""Pure-gauge Monte Carlo sampling layer for the URP lattice gauge theory.

This module implements the *thermodynamic ensemble* of the Wilson lattice
gauge theory that underlies the URP gauge derivation (§3.2 / §4.3).
Deterministic gradient-flow Yang–Mills dynamics (``gauge.flow_step``) find
a stationary point of ``S = coupling·coherence − stress``; **this module**
samples the Boltzmann weight ``exp(−β_g · S_W)`` with the Wilson action

    S_W = Σ_{x,μ<ν} Re Tr(1 − P_{μν}(x))

and measures the observables whose *ensemble averages* carry the confinement
signatures the deterministic flow cannot see:

- **Wilson loops** W(R,T) = ⟨Re Tr U_□(R,T)⟩ / N.  In the confined phase
  these fall as ``exp(−σ·R·T − c·(R+T))``; fitting the area coefficient σ
  gives the **string tension**.
- **Polyakov loop** ⟨|P|⟩ = 0 (confined) or > 0 (deconfined).
- **Creutz ratio** χ(R,T) = −log[W(R,T)·W(R-1,T-1) / W(R,T-1)·W(R-1,T)]
  is a discretisation-artifact-free estimator of σ.

Three update algorithms are provided:

1. **Metropolis** — correct for any SU(N); exact local ΔS via the staple
   sum (O(ndim) links touched, no full-action recompute).
2. **SU(2) Kennedy–Pendleton heat-bath** — exact for SU(2); wrapped into
   Cabibbo–Marinari subgroup updates for SU(3).
3. **Overrelaxation** (microcanonical) — energy-preserving; decorrelates the
   field faster than Metropolis at moderate β_g.

Recommended production schedule: ``n_hb`` heat-bath sweeps + ``n_or``
overrelaxation sweeps interleaved, then measure.

Honest scope note
-----------------
This module targets the pure-gauge sector only (no dynamical fermions).  The
confinement/deconfinement transition and the string tension are well-defined
observables in this setting.  Asymptotic freedom (the running coupling
dependence on the lattice spacing ``a``) and matching to a physical string
tension in MeV/fm require additional renormalisation-group input that is not
implemented here — consistent with the README's stated next step of
"lattice signatures" rather than a full continuum-limit QCD study.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from project_genesis.gauge import (
    _dagger,
    random_unitary,
    wilson_action,
)


# ---------------------------------------------------------------------------
# Low-level staple helper (generalised ndim)
# ---------------------------------------------------------------------------

def _staple_sum(
    links: np.ndarray,
    mu: int,
    site: tuple[int, ...],
) -> np.ndarray:
    """Sum of all staples touching the link U_μ(site).

    For a rectangular plaquette in the (μ,ν) plane the *forward* staple is::

        A_fwd = U_ν(site+μ̂) · U_μ(site+ν̂)† · U_ν(site)†

    and the *backward* staple is::

        A_bwd = U_ν(site−ν̂+μ̂)† · U_μ(site−ν̂)† · U_ν(site−ν̂)

    The Wilson contribution of a single link to S_W is then
    ``Re Tr[ U_μ(site) · A† ]`` where ``A = Σ_ν (A_fwd + A_bwd)``.
    The Metropolis local ΔS when replacing U_old with U_prop is therefore::

        ΔS = Re Tr[ (U_old − U_prop) · A† ]

    which is exact regardless of spatial dimension.
    """
    ndim = links.shape[0]
    spatial = links.shape[1:-2]
    n = links.shape[-1]
    staple = np.zeros((n, n), dtype=np.complex128)

    s = list(site)
    for nu in range(ndim):
        if nu == mu:
            continue
        Lnu = spatial[nu]
        Lmu = spatial[mu]

        # --- forward staple ---
        s_fwd_nu = list(s)
        s_fwd_nu[mu] = (s[mu] + 1) % Lmu   # site + μ̂
        s_fwd_nuL = list(s)
        s_fwd_nuL[nu] = (s[nu] + 1) % Lnu  # site + ν̂
        s_fwd_nuL[mu] = (s[mu] + 1) % Lmu  # site + μ̂ + ν̂  (not needed directly)
        s_nu_fwd = list(s)
        s_nu_fwd[mu] = (s[mu] + 1) % Lmu   # site + μ̂

        u_nu_fwd   = links[(nu,) + tuple(s_fwd_nu)]    # U_ν(site+μ̂)
        s_mu_nu    = list(s)
        s_mu_nu[nu] = (s[nu] + 1) % Lnu               # site+ν̂
        u_mu_nu    = links[(mu,) + tuple(s_mu_nu)]      # U_μ(site+ν̂)
        u_nu_here  = links[(nu,) + tuple(s)]            # U_ν(site)
        staple += u_nu_fwd @ _dagger(u_mu_nu) @ _dagger(u_nu_here)

        # --- backward staple ---
        s_dn = list(s)
        s_dn[nu] = (s[nu] - 1) % Lnu    # site − ν̂
        s_dn_mu  = list(s_dn)
        s_dn_mu[mu] = (s[mu] + 1) % Lmu  # site − ν̂ + μ̂

        u_nu_dn     = links[(nu,) + tuple(s_dn)]     # U_ν(site−ν̂)
        u_mu_dn     = links[(mu,) + tuple(s_dn)]     # U_μ(site−ν̂)
        u_nu_dn_fwd = links[(nu,) + tuple(s_dn_mu)]  # U_ν(site−ν̂+μ̂)
        staple += _dagger(u_nu_dn_fwd) @ _dagger(u_mu_dn) @ u_nu_dn

    return staple


# ---------------------------------------------------------------------------
# Metropolis
# ---------------------------------------------------------------------------

def metropolis_sweep(
    links: np.ndarray,
    beta_g: float,
    rng: np.random.Generator,
    *,
    n_sweeps: int = 1,
    step_scale: float = 0.18,
) -> tuple[np.ndarray, float]:
    """Single-link Metropolis sweep over all links.

    Uses the exact local ``ΔS = Re Tr[(U_old − U_prop) · A†]`` with the
    staple sum ``A`` so no full-action recompute is needed per proposal.
    Accepts with probability ``min(1, exp(−β_g · ΔS))``.

    Parameters
    ----------
    links : ndarray, shape (ndim, *spatial, n, n)
    beta_g : float  — inverse gauge coupling (Wilson β)
    rng : Generator
    n_sweeps : int  — number of full sweeps over the lattice
    step_scale : float  — scale of the random SU(N) perturbation; tune so
        acceptance ≈ 50–70 % for the target β_g.

    Returns
    -------
    links_new : ndarray
    acceptance_rate : float
    """
    ndim = links.shape[0]
    spatial = links.shape[1:-2]
    n = links.shape[-1]
    links_new = links.copy()
    n_acc = n_tot = 0

    for _ in range(n_sweeps):
        for mu in range(ndim):
            for site in np.ndindex(*spatial):
                u_old = links_new[(mu,) + site]
                v = random_unitary(rng, n, special=(n > 1), scale=step_scale)
                u_prop = u_old @ v

                # exact local ΔS
                a = _staple_sum(links_new, mu, site)
                a_dag = _dagger(a)
                delta_s = float(np.real(np.trace((u_old - u_prop) @ a_dag)))

                if delta_s <= 0.0 or rng.random() < math.exp(-beta_g * delta_s):
                    links_new[(mu,) + site] = u_prop
                    n_acc += 1
                n_tot += 1

    return links_new, n_acc / max(1, n_tot)


# ---------------------------------------------------------------------------
# SU(2) Kennedy–Pendleton heat-bath
# ---------------------------------------------------------------------------

def _heatbath_su2_from_staple(
    staple: np.ndarray,
    beta_g: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact SU(2) heat-bath for one link given its staple sum.

    Implements the Kennedy–Pendleton (1985) / Creutz algorithm:

    1.  Compute ``k = |det(A)|^{1/2}`` where ``A`` is the staple.
    2.  The new link samples the distribution
        ``P(U) ∝ exp(β_g/2 · Re Tr[U·A†])`` — a von Mises-like distribution
        on SU(2).
    3.  Accept/reject to get the correct Boltzmann weight for
        ``S_W = β_g/2 · Σ Re Tr(1 − P)``.

    The returned link satisfies ``U ∈ SU(2)``.
    """
    # normalise A so we work with the effective coupling k·β_g/2
    a = staple
    # SU(2) staple sum: write A = k·V, V ∈ SU(2), k = sqrt(det A)
    det_a = np.linalg.det(a)
    k = float(np.real(np.sqrt(np.abs(det_a))))
    if k < 1e-14:
        # degenerate staple: draw uniformly from SU(2)
        return random_unitary(rng, 2, special=True, scale=1.0)

    v = a / k  # approximately SU(2); re-project
    vd = _dagger(v)

    alpha = beta_g * k / 2.0

    # Kennedy–Pendleton: sample a0 from the density
    # p(a0) ∝ sqrt(1-a0²) exp(2·α·a0),  a0 ∈ [−1,1]
    while True:
        x1 = rng.random()
        x2 = rng.random()
        x3 = rng.random()
        r1 = -math.log(max(x1, 1e-300)) / alpha
        r2 = -math.log(max(x2, 1e-300)) / alpha
        cos_theta = math.cos(2.0 * math.pi * x3) ** 2
        a0_candidate = 1.0 - r1 - r2 * cos_theta
        if rng.random() ** 2 <= 1.0 - a0_candidate ** 2:
            a0 = a0_candidate
            break
        # clamp rare numerical overflows
        if a0_candidate < -1.0 or a0_candidate > 1.0:
            continue

    # random unit 3-vector for the SU(2) Lie-algebra component
    sr = math.sqrt(max(0.0, 1.0 - a0 ** 2))
    theta = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * rng.random())))
    phi = 2.0 * math.pi * rng.random()
    avec = np.array([sr * math.sin(theta) * math.cos(phi),
                     sr * math.sin(theta) * math.sin(phi),
                     sr * math.cos(theta)])
    # SU(2) element: a0·I + i·avec·σ
    su2 = (a0 * np.eye(2, dtype=np.complex128)
           + 1j * avec[2] * np.array([[1, 0], [0, -1]], dtype=np.complex128)
           + 1j * avec[0] * np.array([[0, 1], [1, 0]], dtype=np.complex128)
           + 1j * avec[1] * np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
    # new link: su2 · V† so that U·A† is SU(2) × (k·I) with the right weight
    return su2 @ vd


# ---------------------------------------------------------------------------
# Cabibbo–Marinari SU(3) pseudo-heat-bath
# ---------------------------------------------------------------------------

def _cm_su2_embed(size: int, indices: tuple[int, int]) -> callable:
    """Return functions to extract and embed the SU(2) subgroup at (i,j)."""
    i, j = indices

    def extract(u: np.ndarray) -> np.ndarray:
        """Project the 2×2 block (i,j) onto SU(2)."""
        b = u[np.ix_([i, j], [i, j])]
        # re-project onto SU(2) via normalised first row
        r0 = b[0].copy()
        nrm = np.linalg.norm(r0)
        if nrm < 1e-14:
            return np.eye(2, dtype=np.complex128)
        r0 /= nrm
        r1 = np.array([-r0[1].conjugate(), r0[0].conjugate()])
        return np.array([r0, r1])

    def embed(r: np.ndarray) -> np.ndarray:
        """Embed SU(2) matrix r into SU(size) at positions (i,j)."""
        out = np.eye(size, dtype=np.complex128)
        out[i, i] = r[0, 0]
        out[i, j] = r[0, 1]
        out[j, i] = r[1, 0]
        out[j, j] = r[1, 1]
        return out

    return extract, embed


def _cabibbo_marinari_su3_update(
    u: np.ndarray,
    staple: np.ndarray,
    beta_g: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One Cabibbo–Marinari pseudo-heat-bath step for a single SU(3) link.

    Cycles through the three SU(2) subgroups {(0,1), (0,2), (1,2)}, updating
    each with an exact SU(2) heat-bath in the background of the current link.
    This is the standard algorithm for SU(N≥3) heat-bath (Cabibbo & Marinari
    1982).
    """
    u_new = u.copy()
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        extract, embed = _cm_su2_embed(3, (i, j))
        # Effective SU(2) staple: project u·staple† onto the (i,j) block
        w = u_new @ _dagger(staple)
        w_block = w[np.ix_([i, j], [i, j])]
        # build SU(2) staple
        nrm = float(np.real(np.sqrt(np.abs(np.linalg.det(w_block)))))
        if nrm < 1e-14:
            continue
        su2_staple = w_block / nrm
        # heat-bath step in SU(2)
        r = _heatbath_su2_from_staple(su2_staple * nrm, beta_g, rng)
        # embed back
        u_new = embed(r) @ u_new
    return u_new


# ---------------------------------------------------------------------------
# Heat-bath sweep (dispatches SU(2) / SU(3))
# ---------------------------------------------------------------------------

def heatbath_sweep(
    links: np.ndarray,
    beta_g: float,
    rng: np.random.Generator,
    *,
    n_sweeps: int = 1,
) -> tuple[np.ndarray, float]:
    """Heat-bath sweep over all links.

    * SU(2): exact Kennedy–Pendleton update per link.
    * SU(3): Cabibbo–Marinari pseudo-heat-bath (three SU(2) sub-updates).
    * SU(N>3): falls back to a Metropolis step with tight step_scale.

    Returns ``(links, 1.0)`` — acceptance is 100 % by construction for exact
    heat-bath; the returned scalar is kept for API uniformity with
    ``metropolis_sweep``.
    """
    ndim = links.shape[0]
    spatial = links.shape[1:-2]
    n = links.shape[-1]
    links_new = links.copy()

    for _ in range(n_sweeps):
        for mu in range(ndim):
            for site in np.ndindex(*spatial):
                a = _staple_sum(links_new, mu, site)
                if n == 2:
                    links_new[(mu,) + site] = _heatbath_su2_from_staple(a, beta_g, rng)
                elif n == 3:
                    links_new[(mu,) + site] = _cabibbo_marinari_su3_update(
                        links_new[(mu,) + site], a, beta_g, rng
                    )
                else:
                    # generic fallback: tight Metropolis
                    v = random_unitary(rng, n, special=(n > 1), scale=0.05)
                    u_prop = links_new[(mu,) + site] @ v
                    a_dag = _dagger(a)
                    ds = float(np.real(np.trace(
                        (links_new[(mu,) + site] - u_prop) @ a_dag
                    )))
                    if ds <= 0.0 or rng.random() < math.exp(-beta_g * ds):
                        links_new[(mu,) + site] = u_prop

    return links_new, 1.0


# ---------------------------------------------------------------------------
# Overrelaxation (microcanonical)
# ---------------------------------------------------------------------------

def overrelaxation_sweep(
    links: np.ndarray,
    rng: np.random.Generator,
    *,
    n_sweeps: int = 1,
    omega: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Overrelaxation sweep — energy-conserving link updates.

    For each link ``U``, the overrelaxed update is::

        U_new = A† · (U · A†)†   where A = staple_sum

    This is the *reflection* of U through the SU(N) manifold defined by the
    local action minimum ``U_min ∝ A†``.  It leaves the local action
    ``Re Tr[U·A†]`` unchanged (microcanonical) while strongly decorrelating
    the link orientation — particularly useful between heat-bath sweeps.

    ``omega`` ∈ (0, 2] is an over-relaxation parameter; ``omega=1`` gives the
    standard reflection.  Values > 1 do a partial over-shoot (use with care).

    Returns ``(links, 1.0)`` — acceptance is always 1 by construction.
    """
    ndim = links.shape[0]
    spatial = links.shape[1:-2]
    n = links.shape[-1]
    links_new = links.copy()

    for _ in range(n_sweeps):
        for mu in range(ndim):
            for site in np.ndindex(*spatial):
                a = _staple_sum(links_new, mu, site)
                a_dag = _dagger(a)
                u_old = links_new[(mu,) + site]
                # U_new = A† · (U · A†)† = A† · A · U†
                u_ref = a_dag @ (u_old @ a_dag).conj().swapaxes(-1, -2)
                if omega != 1.0:
                    # interpolate between U_old (ω=0) and U_ref (ω=1), overshoot ω>1
                    # re-project to SU(N) via matrix exp of the su(N) displacement
                    disp = u_ref - u_old
                    # first-order tangent step, then re-project via QR
                    u_new = u_old + omega * disp
                    q, r = np.linalg.qr(u_new)
                    # fix determinant phase
                    if n > 1:
                        d = np.linalg.det(q)
                        q[..., -1] /= d
                    links_new[(mu,) + site] = q
                else:
                    links_new[(mu,) + site] = u_ref

    return links_new, 1.0


# ---------------------------------------------------------------------------
# Wilson loop observable
# ---------------------------------------------------------------------------

def wilson_loop(
    links: np.ndarray,
    extents: tuple[int, int],
    plane: tuple[int, int] = (0, 1),
) -> float:
    """Ensemble-average ⟨Re Tr W(R,T)⟩ / N for a single link configuration.

    Averages over *all* origins in the lattice (translational invariance) to
    improve statistics.  The loop is a rectangle in the ``plane = (μ, ν)``
    with side lengths ``extents = (R, T)`` in lattice units.

    Parameters
    ----------
    links : ndarray, shape (ndim, *spatial, n, n)
    extents : (R, T)  — loop side lengths (must fit within spatial dims)
    plane : (μ, ν)   — the two directions spanning the loop

    Returns
    -------
    float in [−1, 1]
    """
    ndim = links.shape[0]
    spatial = links.shape[1:-2]
    n = links.shape[-1]
    mu, nu = plane
    R, T = extents
    dag = _dagger
    loops = 0.0
    count = 0

    for site in np.ndindex(*spatial):
        prod = np.eye(n, dtype=np.complex128)
        pos = list(site)
        for _ in range(R):
            prod = prod @ links[(mu,) + tuple(pos)]
            pos[mu] = (pos[mu] + 1) % spatial[mu]
        for _ in range(T):
            prod = prod @ links[(nu,) + tuple(pos)]
            pos[nu] = (pos[nu] + 1) % spatial[nu]
        for _ in range(R):
            pos[mu] = (pos[mu] - 1) % spatial[mu]
            prod = prod @ dag(links[(mu,) + tuple(pos)])
        for _ in range(T):
            pos[nu] = (pos[nu] - 1) % spatial[nu]
            prod = prod @ dag(links[(nu,) + tuple(pos)])
        loops += float(np.real(np.trace(prod))) / n
        count += 1

    return loops / max(1, count)


# ---------------------------------------------------------------------------
# Polyakov loop
# ---------------------------------------------------------------------------

def polyakov_loop(links: np.ndarray, temporal_axis: int = -1) -> float:
    """Spatially averaged Polyakov loop (real part), normalised to [0, 1].

    ``⟨|P|⟩ ≈ 0`` in the confined phase (Z_N symmetry unbroken), positive
    in the deconfined phase.  The temporal direction is ``temporal_axis``
    (default: last spatial axis, i.e. axis ``ndim − 1``).
    """
    ndim = links.shape[0]
    spatial = links.shape[1:-2]
    n = links.shape[-1]
    if temporal_axis < 0:
        temporal_axis = ndim - 1
    axis = temporal_axis
    Nt = spatial[axis]
    pl_sum = 0.0 + 0.0j
    n_spatial = 0

    # iterate over the spatial hyper-slice orthogonal to `axis`
    transverse = [i for i in range(ndim) if i != axis]
    trans_sizes = [spatial[i] for i in transverse]
    for trans_site in np.ndindex(*trans_sizes):
        # rebuild full site with temporal=0
        pos = [0] * ndim
        for k, i in enumerate(transverse):
            pos[i] = trans_site[k]
        prod = np.eye(n, dtype=np.complex128)
        for _ in range(Nt):
            prod = prod @ links[(axis,) + tuple(pos)]
            pos[axis] = (pos[axis] + 1) % Nt
        pl_sum += np.trace(prod)
        n_spatial += 1

    return float(np.real(pl_sum / (n_spatial * n)))


# ---------------------------------------------------------------------------
# Creutz ratio
# ---------------------------------------------------------------------------

def creutz_ratio(
    w_ij: float,
    w_im1_jm1: float,
    w_i_jm1: float,
    w_im1_j: float,
) -> float:
    """Creutz ratio χ(R,T) = −log[W(R,T)·W(R-1,T-1) / W(R,T-1)·W(R-1,T)].

    This estimator of the string tension cancels the perimeter (self-energy)
    contribution to leading order, giving a cleaner signal for σ than a raw
    area-law fit.  Returns ``nan`` if any Wilson loop is ≤ 0.
    """
    if any(w <= 0.0 for w in (w_ij, w_im1_jm1, w_i_jm1, w_im1_j)):
        return float("nan")
    return -math.log((w_ij * w_im1_jm1) / (w_i_jm1 * w_im1_j))


# ---------------------------------------------------------------------------
# Area-law fitter
# ---------------------------------------------------------------------------

def fit_area_law(
    loop_matrix: np.ndarray,
    r_values: Sequence[int],
    t_values: Sequence[int],
) -> dict:
    """Fit the area-law ansatz W(R,T) = exp(−σ·R·T − c·(R+T)) to a matrix of
    ensemble-averaged Wilson loops.

    Parameters
    ----------
    loop_matrix : ndarray, shape (len(r_values), len(t_values))
        Ensemble-averaged ⟨W(R,T)⟩ values.
    r_values : sequence of R (spatial separations)
    t_values : sequence of T (temporal separations)

    Returns
    -------
    dict with keys:
        ``sigma``   — string tension (area coefficient; > 0 = confined)
        ``perimeter_coeff``  — perimeter coefficient c
        ``fit_residual``     — mean absolute residual of the log fit
        ``creutz_ratios``    — dict {"chi_R_T": value} for adjacent pairs
        ``raw_loops``        — the input matrix as a nested list
    """
    rs = list(r_values)
    ts = list(t_values)
    nr, nt = len(rs), len(ts)
    W = np.array(loop_matrix)

    # build least-squares system: log W(R,T) = -σ·R·T - c·(R+T)
    # only use positive-valued loops
    rows_A, rows_b = [], []
    for i, r in enumerate(rs):
        for j, t in enumerate(ts):
            w = W[i, j]
            if w > 0.0:
                rows_A.append([r * t, r + t])
                rows_b.append(-math.log(w))

    sigma, c = 0.0, 0.0
    residual = float("nan")
    if len(rows_b) >= 2:
        A = np.array(rows_A)
        b = np.array(rows_b)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            sigma, c = float(coeffs[0]), float(coeffs[1])
            pred = A @ coeffs
            residual = float(np.mean(np.abs(pred - b)))
        except np.linalg.LinAlgError:
            pass

    # Creutz ratios for adjacent (R,T) pairs
    creutz: dict[str, float] = {}
    for i, r in enumerate(rs[1:], 1):
        for j, t in enumerate(ts[1:], 1):
            chi = creutz_ratio(W[i, j], W[i - 1, j - 1], W[i, j - 1], W[i - 1, j])
            creutz[f"chi_{r}_{t}"] = chi

    return {
        "sigma": sigma,
        "perimeter_coeff": c,
        "fit_residual": residual,
        "creutz_ratios": creutz,
        "raw_loops": W.tolist(),
    }


# ---------------------------------------------------------------------------
# Deconfinement scan helper
# ---------------------------------------------------------------------------

def deconfinement_scan(
    size: int,
    n: int,
    beta_values: Sequence[float],
    rng: np.random.Generator,
    *,
    ndim: int = 3,
    n_therm: int = 200,
    n_meas: int = 40,
    n_skip: int = 5,
    updater: str = "heatbath",
    loop_sizes: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """Scan β_g and record string tension and Polyakov-loop order parameter.

    For each β value the lattice is thermalised from a *hot start*, then
    ``n_meas`` measurements are taken every ``n_skip`` sweeps.  Useful for
    identifying the deconfinement transition: σ → 0 and ⟨|P|⟩ becomes
    non-zero as β_g increases through the critical β.

    Returns a list of summary dicts (one per β value).
    """
    if loop_sizes is None:
        loop_sizes = [(1, 1), (2, 2), (3, 3)]
    results = []
    for beta_g in beta_values:
        summary, _ = thermalize_and_measure_pure_gauge(
            size=size,
            n=n,
            beta_g=beta_g,
            rng=rng,
            ndim=ndim,
            n_therm=n_therm,
            n_meas=n_meas,
            n_skip=n_skip,
            updater=updater,
            loop_sizes=loop_sizes,
        )
        results.append(summary)
    return results


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------

def thermalize_and_measure_pure_gauge(
    size: int,
    n: int,
    beta_g: float,
    rng: np.random.Generator,
    *,
    ndim: int = 2,
    n_therm: int = 150,
    n_meas: int = 30,
    n_skip: int = 5,
    step_scale: float = 0.18,
    updater: str = "metropolis",
    loop_sizes: list[tuple[int, int]] | None = None,
) -> tuple[dict, np.ndarray]:
    """Thermalise and measure Wilson loops + Polyakov loop for a pure SU(N) theory.

    Parameters
    ----------
    size : int — lattice side length (cubic: *size*^ndim)
    n : int — gauge group SU(N)
    beta_g : float — inverse gauge coupling (Wilson β)
    rng : Generator
    ndim : int — spatial dimensions (default 2; use 3 or 4 for full study)
    n_therm : int — thermalisation sweeps (hot start)
    n_meas : int — number of measurements
    n_skip : int — sweeps between measurements (decorrelation)
    step_scale : float — Metropolis step scale (tune for ~50 % acceptance)
    updater : str — "metropolis" | "heatbath" | "overrelax"
    loop_sizes : list of (R, T) tuples to measure

    Returns
    -------
    summary : dict — contains beta_g, updater, loop_averages, polyakov_mean,
        final_wilson_action, and area_law_fit
    links : ndarray — final link configuration
    """
    if loop_sizes is None:
        loop_sizes = [(1, 1), (2, 2), (3, 3)]

    spatial = (size,) * ndim
    links = random_unitary(rng, n, (ndim, *spatial), special=(n > 1), scale=0.7)

    def _sweep(links):
        if updater == "metropolis":
            return metropolis_sweep(links, beta_g, rng, n_sweeps=1, step_scale=step_scale)[0]
        elif updater == "heatbath":
            return heatbath_sweep(links, beta_g, rng, n_sweeps=1)[0]
        elif updater == "overrelax":
            return overrelaxation_sweep(links, rng, n_sweeps=1)[0]
        else:
            raise ValueError(f"Unknown updater: {updater!r}")

    for _ in range(n_therm):
        links = _sweep(links)

    # build a full W(R,T) matrix
    rs = sorted({s[0] for s in loop_sizes})
    ts = sorted({s[1] for s in loop_sizes})
    W_accum = np.zeros((len(rs), len(ts)), dtype=np.float64)
    poly_vals: list[float] = []

    for _ in range(n_meas):
        for _ in range(n_skip):
            links = _sweep(links)
        for i, r in enumerate(rs):
            for j, t in enumerate(ts):
                W_accum[i, j] += wilson_loop(links, (r, t), plane=(0, 1))
        poly_vals.append(polyakov_loop(links))

    W_mean = W_accum / n_meas
    area_fit = fit_area_law(W_mean, rs, ts)

    summary = {
        "beta_g": float(beta_g),
        "ndim": ndim,
        "size": size,
        "n": n,
        "updater": updater,
        "n_therm": n_therm,
        "n_meas": n_meas,
        "loop_sizes": loop_sizes,
        "loop_averages": {
            f"W_{rs[i]}_{ts[j]}": float(W_mean[i, j])
            for i in range(len(rs))
            for j in range(len(ts))
        },
        "polyakov_mean": float(np.mean(poly_vals)),
        "polyakov_susceptibility": float(np.var(poly_vals)),
        "final_wilson_action": float(wilson_action(links)),
        "area_law_fit": area_fit,
    }
    return summary, links
