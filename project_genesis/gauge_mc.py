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
import sys
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
    """Sum of all staples touching the link U_μ(site)."""
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

        s_fwd_nu = list(s)
        s_fwd_nu[mu] = (s[mu] + 1) % Lmu
        s_fwd_nuL = list(s)
        s_fwd_nuL[nu] = (s[nu] + 1) % Lnu
        s_fwd_nuL[mu] = (s[mu] + 1) % Lmu
        s_nu_fwd = list(s)
        s_nu_fwd[mu] = (s[mu] + 1) % Lmu

        u_nu_fwd   = links[(nu,) + tuple(s_fwd_nu)]
        s_mu_nu    = list(s)
        s_mu_nu[nu] = (s[nu] + 1) % Lnu
        u_mu_nu    = links[(mu,) + tuple(s_mu_nu)]
        u_nu_here  = links[(nu,) + tuple(s)]
        staple += u_nu_fwd @ _dagger(u_mu_nu) @ _dagger(u_nu_here)

        s_dn = list(s)
        s_dn[nu] = (s[nu] - 1) % Lnu
        s_dn_mu  = list(s_dn)
        s_dn_mu[mu] = (s[mu] + 1) % Lmu

        u_nu_dn     = links[(nu,) + tuple(s_dn)]
        u_mu_dn     = links[(mu,) + tuple(s_dn)]
        u_nu_dn_fwd = links[(nu,) + tuple(s_dn_mu)]
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
    """Single-link Metropolis sweep over all links."""
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
    """Exact SU(2) heat-bath for one link given its staple sum."""
    a = staple
    det_a = np.linalg.det(a)
    k = float(np.real(np.sqrt(np.abs(det_a))))
    if k < 1e-14:
        return random_unitary(rng, 2, special=True, scale=1.0)

    v = a / k
    vd = _dagger(v)
    alpha = beta_g * k / 2.0

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
        if a0_candidate < -1.0 or a0_candidate > 1.0:
            continue

    sr = math.sqrt(max(0.0, 1.0 - a0 ** 2))
    theta = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * rng.random())))
    phi = 2.0 * math.pi * rng.random()
    avec = np.array([sr * math.sin(theta) * math.cos(phi),
                     sr * math.sin(theta) * math.sin(phi),
                     sr * math.cos(theta)])
    su2 = (a0 * np.eye(2, dtype=np.complex128)
           + 1j * avec[2] * np.array([[1, 0], [0, -1]], dtype=np.complex128)
           + 1j * avec[0] * np.array([[0, 1], [1, 0]], dtype=np.complex128)
           + 1j * avec[1] * np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
    return su2 @ vd


# ---------------------------------------------------------------------------
# Cabibbo–Marinari SU(3) pseudo-heat-bath
# ---------------------------------------------------------------------------

def _cm_su2_embed(size: int, indices: tuple[int, int]) -> callable:
    """Return functions to extract and embed the SU(2) subgroup at (i,j)."""
    i, j = indices

    def extract(u: np.ndarray) -> np.ndarray:
        b = u[np.ix_([i, j], [i, j])]
        r0 = b[0].copy()
        nrm = np.linalg.norm(r0)
        if nrm < 1e-14:
            return np.eye(2, dtype=np.complex128)
        r0 /= nrm
        r1 = np.array([-r0[1].conjugate(), r0[0].conjugate()])
        return np.array([r0, r1])

    def embed(r: np.ndarray) -> np.ndarray:
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
    """One Cabibbo–Marinari pseudo-heat-bath step for a single SU(3) link."""
    u_new = u.copy()
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        extract, embed = _cm_su2_embed(3, (i, j))
        w = u_new @ _dagger(staple)
        w_block = w[np.ix_([i, j], [i, j])]
        nrm = float(np.real(np.sqrt(np.abs(np.linalg.det(w_block)))))
        if nrm < 1e-14:
            continue
        su2_staple = w_block / nrm
        r = _heatbath_su2_from_staple(su2_staple * nrm, beta_g, rng)
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
    """Heat-bath sweep over all links."""
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
    """Overrelaxation sweep — energy-conserving link updates."""
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
                u_ref = a_dag @ (u_old @ a_dag).conj().swapaxes(-1, -2)
                if omega != 1.0:
                    disp = u_ref - u_old
                    u_new = u_old + omega * disp
                    q, r = np.linalg.qr(u_new)
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
    """Ensemble-average ⟨Re Tr W(R,T)⟩ / N for a single link configuration."""
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
    """Spatially averaged Polyakov loop (real part), normalised to [0, 1]."""
    ndim = links.shape[0]
    spatial = links.shape[1:-2]
    n = links.shape[-1]
    if temporal_axis < 0:
        temporal_axis = ndim - 1
    axis = temporal_axis
    Nt = spatial[axis]
    pl_sum = 0.0 + 0.0j
    n_spatial = 0

    transverse = [i for i in range(ndim) if i != axis]
    trans_sizes = [spatial[i] for i in transverse]
    for trans_site in np.ndindex(*trans_sizes):
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
    """Creutz ratio χ(R,T) = −log[W(R,T)·W(R-1,T-1) / W(R,T-1)·W(R-1,T)]."""
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
    """Fit W(R,T) = exp(−σ·R·T − c·(R+T)) to ensemble-averaged Wilson loops."""
    rs = list(r_values)
    ts = list(t_values)
    W = np.array(loop_matrix)

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
    verbose_progress: bool = True,
) -> list[dict]:
    """Scan β_g and record string tension and Polyakov-loop order parameter.

    Parameters
    ----------
    verbose_progress : bool
        If True (default), print a one-liner per beta point to stdout so that
        CI runners and terminals show forward progress instead of appearing
        hung.  Set to False in unit tests where output is unwanted.
    """
    if loop_sizes is None:
        loop_sizes = [(1, 1), (2, 2), (3, 3)]
    beta_list = list(beta_values)
    n_beta = len(beta_list)
    results = []
    for idx, beta_g in enumerate(beta_list, 1):
        if verbose_progress:
            print(
                f"  beta {idx}/{n_beta}  β_g={beta_g:.3f}  "
                f"(therm={n_therm} meas={n_meas} skip={n_skip} updater={updater})",
                flush=True,
            )
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
        if verbose_progress:
            sigma = summary["area_law_fit"]["sigma"]
            poly  = abs(summary["polyakov_mean"])
            print(
                f"    → sigma={sigma:+.5f}  |<P>|={poly:.4f}",
                flush=True,
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
    """Thermalise and measure Wilson loops + Polyakov loop for a pure SU(N) theory."""
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
