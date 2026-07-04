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
2. **SU(2) heat-bath** — exact for SU(2), using Kennedy–Pendleton sampling
   at strong effective coupling and Creutz sampling at weak effective
   coupling so link updates never stall; wrapped into Cabibbo–Marinari
   subgroup updates for any SU(N ≥ 3).  A quenched matter field ψ may be
   coupled through ``exp(g_m·Σ Re[ψ†Uψ])`` — the matter term enters each
   link's weight exactly, as a staple addition.
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

try:  # optional JIT acceleration; the pure-Python reference path always works
    from project_genesis import gauge_mc_kernels as _nbk
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is a declared dependency
    _nbk = None
    _HAVE_NUMBA = False


def _use_numba_default(n: int, use_numba: bool | None) -> bool:
    """Resolve the ``use_numba`` tri-state: None = auto (if available)."""
    if use_numba is None:
        return _HAVE_NUMBA and n >= 2
    return bool(use_numba) and _HAVE_NUMBA


def _matter_outer(
    matter: np.ndarray,
    mu: int,
    site: tuple[int, ...],
    spatial: tuple[int, ...],
) -> np.ndarray:
    """The link's matter source ``M = ψ(x+μ̂)·ψ†(x)`` as an n×n matrix.

    ``Re[ψ†(x)·U·ψ(x+μ̂)] = Re Tr(U·M)``, so a quenched matter field enters
    the link update as an addition to the staple sum.
    """
    fwd = list(site)
    fwd[mu] = (site[mu] + 1) % spatial[mu]
    psi_here = matter[tuple(site)]
    psi_fwd = matter[tuple(fwd)]
    return np.outer(psi_fwd, np.conjugate(psi_here))


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
    matter: np.ndarray | None = None,
    matter_coupling: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Single-link Metropolis sweep over all links.

    With a quenched ``matter`` field (shape ``(*spatial, n)``), samples the
    same matter-coupled ensemble as :func:`heatbath_sweep` — kept as an
    algorithm-independent cross-check of the exact heat-bath.
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

                # S(U) = const − Re Tr(U·A) with A the staple sum, because every
                # plaquette containing U_μ(x) factorises as Tr(U_μ(x)·staple).
                a = _staple_sum(links_new, mu, site)
                delta_s = float(np.real(np.trace((u_old - u_prop) @ a)))
                delta_exponent = -beta_g * delta_s
                if matter is not None:
                    m = _matter_outer(matter, mu, site, spatial)
                    delta_exponent += matter_coupling * float(
                        np.real(np.trace((u_prop - u_old) @ m))
                    )

                if delta_exponent >= 0.0 or rng.random() < math.exp(delta_exponent):
                    links_new[(mu,) + site] = u_prop
                    n_acc += 1
                n_tot += 1

    return links_new, n_acc / max(1, n_tot)


# ---------------------------------------------------------------------------
# SU(2) Kennedy–Pendleton heat-bath
# ---------------------------------------------------------------------------

def _sample_su2_a0(c: float, rng: np.random.Generator) -> float:
    """Sample a0 ∈ [−1, 1] from P(a0) ∝ √(1 − a0²)·exp(c·a0), c ≥ 0.

    Two exact rejection samplers are combined so the expected iteration
    count is O(1) for *every* c:

    - **c > 2 — Kennedy–Pendleton.**  Draw δ = 1 − a0 from Gamma(3/2, rate c)
      via δ = −(ln x1 + cos²(2πx2)·ln x3)/c, then accept with probability
      √(1 − δ/2).  Efficient at large c (δ concentrates near 0) but the
      acceptance collapses like c^{3/2} as c → 0 — the previous
      implementation used this branch for all c, which is what made single
      link updates hang for minutes at strong coupling.
    - **c ≤ 2 — Creutz.**  Draw a0 from the truncated exponential
      ∝ exp(c·a0) on [−1, 1] by inverse CDF, then accept with probability
      √(1 − a0²).  Acceptance ≥ π·I₁(c)/(2·sinh c) ≥ 0.6 on this range.

    A finite retry cap on the KP branch falls back to the always-safe
    Creutz branch, so the sampler provably terminates.
    """
    if c > 2.0:
        for _ in range(64):
            x1 = rng.random()
            x2 = rng.random()
            x3 = rng.random()
            delta = -(math.log(max(x1, 1e-300))
                      + math.cos(2.0 * math.pi * x2) ** 2
                      * math.log(max(x3, 1e-300))) / c
            if delta <= 2.0 and rng.random() ** 2 <= 1.0 - delta / 2.0:
                return 1.0 - delta
        # Statistically unreachable for c > 2; fall through to Creutz.
    while True:
        x = rng.random()
        if c < 1e-9:
            a0 = 2.0 * x - 1.0
        else:
            # Inverse CDF of exp(c·a0) on [−1, 1], written to stay finite
            # for large c: a0 = 1 + ln(x + (1−x)e^{−2c})/c.
            a0 = 1.0 + math.log(x + (1.0 - x) * math.exp(-2.0 * c)) / c
        a0 = max(-1.0, min(1.0, a0))
        if rng.random() ** 2 <= 1.0 - a0 * a0:
            return a0


def _su2_from_a0(a0: float, rng: np.random.Generator) -> np.ndarray:
    """Build X = a0·1 + i·a⃗·σ⃗ ∈ SU(2) with a⃗ uniform on the sphere of radius √(1−a0²)."""
    sr = math.sqrt(max(0.0, 1.0 - a0 ** 2))
    cos_t = 1.0 - 2.0 * rng.random()
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t ** 2))
    phi = 2.0 * math.pi * rng.random()
    a1 = sr * sin_t * math.cos(phi)
    a2 = sr * sin_t * math.sin(phi)
    a3 = sr * cos_t
    return np.array(
        [[a0 + 1j * a3, a2 + 1j * a1],
         [-a2 + 1j * a1, a0 - 1j * a3]],
        dtype=np.complex128,
    )


def _heatbath_su2_from_staple(
    staple: np.ndarray,
    beta_g: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact SU(2) heat-bath for one link given its (effective) staple sum.

    The staple sum of SU(2) matrices is proportional to an SU(2) matrix,
    A = k·V with k = √det(A).  The Boltzmann weight for the link is
    exp(β_g·Re Tr(U·A)) = exp(2·β_g·k·a0) where X = U·V and Re Tr X = 2·a0,
    so a0 is sampled from √(1 − a0²)·exp(c·a0) with c = 2·β_g·k and the new
    link is U = X·V†.
    """
    # Project onto the quaternionic part first: for an exact SU(2) staple
    # this is a no-op, but it strips the accumulated float noise that would
    # otherwise be amplified by 1/k below and compound sweep over sweep,
    # driving the links off the group manifold.
    a = _su2_quaternion_part(staple)
    k = float(np.real(np.sqrt(np.abs(np.linalg.det(a)))))
    c = 2.0 * beta_g * k
    a0 = _sample_su2_a0(c, rng)
    x = _su2_from_a0(a0, rng)
    if k < 1e-14:
        return x  # weight is flat: any Haar sample is exact
    return x @ _dagger(a / k)


# ---------------------------------------------------------------------------
# Cabibbo–Marinari SU(3) pseudo-heat-bath
# ---------------------------------------------------------------------------

def _su2_quaternion_part(m: np.ndarray) -> np.ndarray:
    """Project a 2×2 complex matrix onto its quaternionic (∝ SU(2)) part.

    Any 2×2 complex M splits as Q + i·Q' with Q, Q' real-quaternionic
    (real combinations of {1, iσ⃗}).  For R ∈ SU(2), Re Tr(R·M) = Re Tr(R·Q),
    so only Q matters for the subgroup Boltzmann weight.  Skipping this
    projection (as the previous implementation did) both samples the wrong
    subgroup distribution and lets SU(3) links drift off the group manifold.
    """
    q0 = 0.5 * (m[0, 0] + m[1, 1].conjugate())
    q1 = 0.5 * (m[0, 1] - m[1, 0].conjugate())
    return np.array([[q0, q1], [-q1.conjugate(), q0.conjugate()]],
                    dtype=np.complex128)


def _cm_embed(size: int, indices: tuple[int, int], r: np.ndarray) -> np.ndarray:
    """Embed an SU(2) matrix r into the (i,j) subgroup of SU(size)."""
    i, j = indices
    out = np.eye(size, dtype=np.complex128)
    out[i, i] = r[0, 0]
    out[i, j] = r[0, 1]
    out[j, i] = r[1, 0]
    out[j, j] = r[1, 1]
    return out


def _su2_subgroup_pairs(n: int) -> tuple[tuple[int, int], ...]:
    """All (i, j) SU(2) subgroup index pairs of SU(n)."""
    return tuple((i, j) for i in range(n) for j in range(i + 1, n))


def _cabibbo_marinari_update(
    u: np.ndarray,
    staple: np.ndarray,
    beta_g: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One Cabibbo–Marinari pseudo-heat-bath step for a single SU(N) link.

    The link update U → R_emb·U changes the weight through
    Re Tr(R_emb·U·A) = Re Tr₂(R·(U·A)_block) + const, so each SU(2)
    subgroup rotation R is drawn by the exact SU(2) heat-bath against the
    quaternionic part of the corresponding 2×2 block of W = U·A.  Sweeping
    all N(N−1)/2 subgroups makes the update ergodic on SU(N) for any N ≥ 2.
    """
    n = u.shape[-1]
    u_new = u.copy()
    for (i, j) in _su2_subgroup_pairs(n):
        w = u_new @ staple
        w_q = _su2_quaternion_part(w[np.ix_([i, j], [i, j])])
        r = _heatbath_su2_from_staple(w_q, beta_g, rng)
        u_new = _cm_embed(n, (i, j), r) @ u_new
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
    matter: np.ndarray | None = None,
    matter_coupling: float = 1.0,
    use_numba: bool | None = None,
) -> tuple[np.ndarray, float]:
    """Heat-bath sweep over all links of the (optionally matter-coupled) ensemble.

    Samples ``exp(−β_g·S_W + g_m·Σ_{x,μ} Re[ψ†(x)·U_μ(x)·ψ(x+μ̂)])``.  A
    quenched matter field ``matter`` (shape ``(*spatial, n)``, e.g. the
    sector-membership ψ from :func:`project_genesis.gauge.sector_field_to_psi`)
    enters each link's Boltzmann weight exactly, as the staple addition
    ``(g_m/β_g)·ψ(x+μ̂)ψ†(x)`` — the quaternionic projection inside the SU(2)
    kernel extracts precisely the part of that matrix the weight depends on,
    so the update remains an *exact* heat-bath.  Requires ``β_g > 0`` when
    matter is supplied.

    SU(2) uses the direct heat-bath; every N ≥ 3 uses Cabibbo–Marinari over
    all N(N−1)/2 SU(2) subgroups.  ``use_numba=None`` (default) uses the JIT
    kernels when available; ``False`` forces the pure-Python reference path.
    Both paths implement the same update with the same random-draw order.
    """
    ndim = links.shape[0]
    spatial = links.shape[1:-2]
    n = links.shape[-1]
    if matter is not None and not beta_g > 0.0:
        raise ValueError("matter coupling requires beta_g > 0")

    if _use_numba_default(n, use_numba):
        flat, fwd, bwd, spatial = _nbk.flatten_links(links)
        if matter is None:
            psi_flat = np.zeros((1, n), dtype=np.complex128)
            cob = 0.0
        else:
            psi_flat = np.ascontiguousarray(
                matter.reshape(-1, n).astype(np.complex128)
            )
            cob = float(matter_coupling) / float(beta_g)
        _nbk.heatbath_sweep_flat(
            flat, fwd, bwd, float(beta_g), rng, n_sweeps, psi_flat, cob
        )
        return _nbk.unflatten_links(flat, spatial), 1.0

    links_new = links.copy()

    for _ in range(n_sweeps):
        for mu in range(ndim):
            for site in np.ndindex(*spatial):
                a = _staple_sum(links_new, mu, site)
                if matter is not None:
                    a = a + (matter_coupling / beta_g) * _matter_outer(
                        matter, mu, site, spatial
                    )
                if n == 2:
                    links_new[(mu,) + site] = _heatbath_su2_from_staple(a, beta_g, rng)
                else:
                    links_new[(mu,) + site] = _cabibbo_marinari_update(
                        links_new[(mu,) + site], a, beta_g, rng
                    )

    return links_new, 1.0


# ---------------------------------------------------------------------------
# Overrelaxation (microcanonical)
# ---------------------------------------------------------------------------

def _overrelax_link(u: np.ndarray, staple: np.ndarray) -> np.ndarray:
    """Microcanonical reflection of one link about its (effective) staple.

    For each SU(2) subgroup, the weight-relevant quaternionic part of the
    2×2 block of W = U·A is k·Ṽ with Ṽ ∈ SU(2); the reflection R = (Ṽ†)²
    exactly preserves Re Tr₂(R·k·Ṽ) — and hence the Wilson action (plus any
    matter term folded into ``staple``) — while moving the link as far as
    possible.  For SU(2) the single "subgroup" is the whole group, so the
    update is the exact classic reflection; for SU(N ≥ 3) it is applied per
    Cabibbo–Marinari subgroup.  The previous implementation used A†·A·U†,
    which is neither unitary nor action-preserving.
    """
    n = u.shape[-1]
    subgroups = [(0, 1)] if n == 2 else list(_su2_subgroup_pairs(n))
    u_new = u.copy()
    for (i, j) in subgroups:
        w = u_new @ staple
        w_q = _su2_quaternion_part(w[np.ix_([i, j], [i, j])])
        k = float(np.real(np.sqrt(np.abs(np.linalg.det(w_q)))))
        if k < 1e-14:
            continue
        v = w_q / k
        r = _dagger(v) @ _dagger(v)
        u_new = _cm_embed(n, (i, j), r) @ u_new if n > 2 else r @ u_new
    return u_new


def overrelaxation_sweep(
    links: np.ndarray,
    rng: np.random.Generator,
    *,
    n_sweeps: int = 1,
    omega: float = 1.0,
    matter: np.ndarray | None = None,
    matter_coupling: float = 1.0,
    beta_g: float = 1.0,
    use_numba: bool | None = None,
) -> tuple[np.ndarray, float]:
    """Overrelaxation sweep — energy-conserving link updates.

    With a quenched ``matter`` field the reflection is taken about the
    *effective* staple ``A + (g_m/β_g)·ψ(x+μ̂)ψ†(x)``, so the full exponent
    ``−β_g·S_W + g_m·Σ Re[ψ†Uψ]`` is conserved exactly (``beta_g`` is only
    used for that scaling; the microcanonical move itself has no coupling).

    ``omega`` is retained for API compatibility; the standard full
    reflection (omega = 1) is always applied, as partial reflections are
    not microcanonical after re-unitarisation.
    """
    ndim = links.shape[0]
    spatial = links.shape[1:-2]
    n = links.shape[-1]
    if matter is not None and not beta_g > 0.0:
        raise ValueError("matter coupling requires beta_g > 0")

    if _use_numba_default(n, use_numba):
        flat, fwd, bwd, spatial = _nbk.flatten_links(links)
        if matter is None:
            psi_flat = np.zeros((1, n), dtype=np.complex128)
            cob = 0.0
        else:
            psi_flat = np.ascontiguousarray(
                matter.reshape(-1, n).astype(np.complex128)
            )
            cob = float(matter_coupling) / float(beta_g)
        _nbk.overrelaxation_sweep_flat(flat, fwd, bwd, n_sweeps, psi_flat, cob)
        return _nbk.unflatten_links(flat, spatial), 1.0

    links_new = links.copy()

    for _ in range(n_sweeps):
        for mu in range(ndim):
            for site in np.ndindex(*spatial):
                a = _staple_sum(links_new, mu, site)
                if matter is not None:
                    a = a + (matter_coupling / beta_g) * _matter_outer(
                        matter, mu, site, spatial
                    )
                links_new[(mu,) + site] = _overrelax_link(
                    links_new[(mu,) + site], a
                )

    return links_new, 1.0


# ---------------------------------------------------------------------------
# Wilson loop observable
# ---------------------------------------------------------------------------

def wilson_loop(
    links: np.ndarray,
    extents: tuple[int, int],
    plane: tuple[int, int] = (0, 1),
    *,
    use_numba: bool | None = None,
) -> float:
    """Ensemble-average ⟨Re Tr W(R,T)⟩ / N for a single link configuration."""
    ndim = links.shape[0]
    spatial = links.shape[1:-2]
    n = links.shape[-1]
    mu, nu = plane
    R, T = extents

    if _HAVE_NUMBA and use_numba is not False:
        flat, fwd, bwd, _ = _nbk.flatten_links(links)
        return float(_nbk.wilson_loop_flat(flat, fwd, bwd, R, T, mu, nu))
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

def polyakov_loop(
    links: np.ndarray,
    temporal_axis: int = -1,
    *,
    use_numba: bool | None = None,
) -> float:
    """Spatially averaged Polyakov loop (real part), normalised to [0, 1]."""
    ndim = links.shape[0]
    spatial = links.shape[1:-2]
    n = links.shape[-1]
    if temporal_axis < 0:
        temporal_axis = ndim - 1
    axis = temporal_axis
    Nt = spatial[axis]

    if _HAVE_NUMBA and use_numba is not False:
        flat, fwd, bwd, _ = _nbk.flatten_links(links)
        idx = np.arange(flat.shape[1]).reshape(spatial)
        starts = np.ascontiguousarray(
            np.take(idx, 0, axis=axis).ravel().astype(np.int64)
        )
        return float(_nbk.polyakov_loop_flat(flat, fwd, axis, starts, Nt))
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
