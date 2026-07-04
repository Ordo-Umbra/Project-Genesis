"""Numba-accelerated kernels for the Monte Carlo gauge layer.

These JIT kernels implement exactly the same updates as the pure-Python
reference implementations in :mod:`project_genesis.gauge_mc` — same staple
convention, same SU(2) heat-bath sampler (Kennedy–Pendleton at strong
effective coupling, Creutz at weak), same Cabibbo–Marinari subgroup scheme,
same random-draw order — so a chain driven by these kernels samples the
identical Boltzmann ensemble ``exp(−β_g·S_W)``.

Layout
------
The lattice ``links`` array ``(ndim, *spatial, n, n)`` is flattened to
``(ndim, nsites, n, n)`` with precomputed periodic neighbour tables
``fwd[mu, s]`` / ``bwd[mu, s]`` (row-major site order, matching the
``np.ndindex`` sweep order of the reference implementation).  Small SU(2)/
SU(3) matrix products are hand-rolled so the hot loop never leaves compiled
code.

Numba's ``np.random.Generator`` support draws from the same bit stream as
NumPy, so the scalar samplers here consume randomness in the same order and
with the same values as the reference path.  (Bit-identical *chains* are
still not guaranteed, because tiny matmul rounding differences can flip a
Metropolis-style accept after many sweeps — equivalence is enforced by the
distribution-level tests in ``tests/test_gauge_mc_numba.py``.)
"""

from __future__ import annotations

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Layout helpers (pure Python)
# ---------------------------------------------------------------------------

def flatten_links(links: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Flatten ``(ndim, *spatial, n, n)`` links to ``(ndim, nsites, n, n)``.

    Returns the flat array (always a contiguous copy, so in-place kernels
    never mutate the caller's array), forward/backward neighbour tables of
    shape ``(ndim, nsites)``, and the original spatial shape.
    """
    ndim = links.shape[0]
    spatial = links.shape[1:-2]
    n = links.shape[-1]
    nsites = int(np.prod(spatial))
    flat = links.reshape(ndim, nsites, n, n).astype(np.complex128, copy=True)
    idx = np.arange(nsites).reshape(spatial)
    fwd = np.empty((ndim, nsites), dtype=np.int64)
    bwd = np.empty((ndim, nsites), dtype=np.int64)
    for mu in range(ndim):
        fwd[mu] = np.roll(idx, -1, axis=mu).ravel()
        bwd[mu] = np.roll(idx, 1, axis=mu).ravel()
    return flat, fwd, bwd, spatial


def unflatten_links(flat: np.ndarray, spatial: tuple[int, ...]) -> np.ndarray:
    """Inverse of :func:`flatten_links`."""
    ndim = flat.shape[0]
    n = flat.shape[-1]
    return flat.reshape(ndim, *spatial, n, n)


# ---------------------------------------------------------------------------
# Small-matrix helpers (complex, n = 2 or 3)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _mat_mul(a, b, out):
    """out = a @ b for small square complex matrices."""
    n = a.shape[0]
    for i in range(n):
        for j in range(n):
            acc = 0.0 + 0.0j
            for k in range(n):
                acc += a[i, k] * b[k, j]
            out[i, j] = acc


@njit(cache=True)
def _mat_mul_bdag(a, b, out):
    """out = a @ b† for small square complex matrices."""
    n = a.shape[0]
    for i in range(n):
        for j in range(n):
            acc = 0.0 + 0.0j
            for k in range(n):
                acc += a[i, k] * np.conj(b[j, k])
            out[i, j] = acc


@njit(cache=True)
def _mat_dag_mul_bdag(a, b, out):
    """out = a† @ b† for small square complex matrices."""
    n = a.shape[0]
    for i in range(n):
        for j in range(n):
            acc = 0.0 + 0.0j
            for k in range(n):
                acc += np.conj(a[k, i]) * np.conj(b[j, k])
            out[i, j] = acc


@njit(cache=True)
def _staple_sum_flat(links, fwd, bwd, mu, s, staple, t1, t2):
    """Sum of staples touching link (mu, s); same convention as the reference.

    ``staple`` is the (n, n) output buffer; ``t1``/``t2`` are scratch.
    """
    ndim = links.shape[0]
    n = links.shape[-1]
    for i in range(n):
        for j in range(n):
            staple[i, j] = 0.0 + 0.0j
    s_fwd_mu = fwd[mu, s]
    for nu in range(ndim):
        if nu == mu:
            continue
        # forward staple: U_ν(s+μ̂) · U_μ†(s+ν̂) · U_ν†(s)
        s_fwd_nu = fwd[nu, s]
        _mat_mul_bdag(links[nu, s_fwd_mu], links[mu, s_fwd_nu], t1)
        _mat_mul_bdag(t1, links[nu, s], t2)
        for i in range(n):
            for j in range(n):
                staple[i, j] += t2[i, j]
        # backward staple: U_ν†(s−ν̂+μ̂) · U_μ†(s−ν̂) · U_ν(s−ν̂)
        s_dn = bwd[nu, s]
        s_dn_mu = fwd[mu, s_dn]
        _mat_dag_mul_bdag(links[nu, s_dn_mu], links[mu, s_dn], t1)
        _mat_mul(t1, links[nu, s_dn], t2)
        for i in range(n):
            for j in range(n):
                staple[i, j] += t2[i, j]


# ---------------------------------------------------------------------------
# SU(2) sampling (same algorithm and draw order as gauge_mc._sample_su2_a0)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _sample_su2_a0_nb(c, rng):
    """Sample a0 ∈ [−1, 1] from P(a0) ∝ √(1 − a0²)·exp(c·a0)."""
    if c > 2.0:
        for _ in range(64):
            x1 = rng.random()
            x2 = rng.random()
            x3 = rng.random()
            delta = -(np.log(max(x1, 1e-300))
                      + np.cos(2.0 * np.pi * x2) ** 2
                      * np.log(max(x3, 1e-300))) / c
            if delta <= 2.0 and rng.random() ** 2 <= 1.0 - delta / 2.0:
                return 1.0 - delta
    while True:
        x = rng.random()
        if c < 1e-9:
            a0 = 2.0 * x - 1.0
        else:
            a0 = 1.0 + np.log(x + (1.0 - x) * np.exp(-2.0 * c)) / c
        a0 = max(-1.0, min(1.0, a0))
        if rng.random() ** 2 <= 1.0 - a0 * a0:
            return a0


@njit(cache=True)
def _su2_from_a0_nb(a0, rng, out):
    """Fill ``out`` with X = a0·1 + i·a⃗·σ⃗, a⃗ uniform on the √(1−a0²) sphere."""
    sr = np.sqrt(max(0.0, 1.0 - a0 ** 2))
    cos_t = 1.0 - 2.0 * rng.random()
    sin_t = np.sqrt(max(0.0, 1.0 - cos_t ** 2))
    phi = 2.0 * np.pi * rng.random()
    a1 = sr * sin_t * np.cos(phi)
    a2 = sr * sin_t * np.sin(phi)
    a3 = sr * cos_t
    out[0, 0] = a0 + 1j * a3
    out[0, 1] = a2 + 1j * a1
    out[1, 0] = -a2 + 1j * a1
    out[1, 1] = a0 - 1j * a3


@njit(cache=True)
def _quat_part(m, out):
    """Quaternionic projection of a 2×2 complex matrix; returns k = √det."""
    q0 = 0.5 * (m[0, 0] + np.conj(m[1, 1]))
    q1 = 0.5 * (m[0, 1] - np.conj(m[1, 0]))
    out[0, 0] = q0
    out[0, 1] = q1
    out[1, 0] = -np.conj(q1)
    out[1, 1] = np.conj(q0)
    return np.sqrt(abs(q0) ** 2 + abs(q1) ** 2)


@njit(cache=True)
def _heatbath_su2_nb(staple, beta_g, rng, out, aq, x):
    """Exact SU(2) heat-bath given a (2, 2) staple; fills ``out``.

    ``aq`` and ``x`` are (2, 2) scratch buffers.
    """
    k = _quat_part(staple, aq)
    c = 2.0 * beta_g * k
    a0 = _sample_su2_a0_nb(c, rng)
    _su2_from_a0_nb(a0, rng, x)
    if k < 1e-14:
        out[0, 0] = x[0, 0]
        out[0, 1] = x[0, 1]
        out[1, 0] = x[1, 0]
        out[1, 1] = x[1, 1]
        return
    for i in range(2):
        for j in range(2):
            aq[i, j] = aq[i, j] / k
    _mat_mul_bdag(x, aq, out)


@njit(cache=True)
def _su2_left_apply(r, u, i, j):
    """Left-multiply rows i, j of ``u`` by the SU(2) matrix ``r`` in place."""
    n = u.shape[0]
    for col in range(n):
        ui = u[i, col]
        uj = u[j, col]
        u[i, col] = r[0, 0] * ui + r[0, 1] * uj
        u[j, col] = r[1, 0] * ui + r[1, 1] * uj


@njit(cache=True)
def _cm_update_nb(u, staple, beta_g, rng, w, blk, aq, x, r):
    """Cabibbo–Marinari pseudo-heat-bath on a single SU(N) link, in place.

    Sweeps all N(N−1)/2 SU(2) subgroups in (i < j) lexicographic order —
    the same order as the pure-Python reference.  ``w`` is (n, n) scratch;
    ``blk``/``aq``/``x``/``r`` are (2, 2) scratch.
    """
    n = u.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            _mat_mul(u, staple, w)
            blk[0, 0] = w[i, i]
            blk[0, 1] = w[i, j]
            blk[1, 0] = w[j, i]
            blk[1, 1] = w[j, j]
            _quat_part(blk, aq)
            # aq is already quaternionic; _heatbath_su2_nb re-projects (no-op)
            _heatbath_su2_nb(aq, beta_g, rng, r, blk, x)
            _su2_left_apply(r, u, i, j)


@njit(cache=True)
def _add_matter(staple, psi, fwd, mu, s, cob):
    """Add the matter source ``cob·ψ(s+μ̂)·ψ†(s)`` to the staple, in place."""
    n = staple.shape[0]
    sf = fwd[mu, s]
    for i in range(n):
        for j in range(n):
            staple[i, j] += cob * psi[sf, i] * np.conj(psi[s, j])


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

@njit(cache=True)
def heatbath_sweep_flat(links, fwd, bwd, beta_g, rng, n_sweeps, psi, cob):
    """Heat-bath sweep(s) over flattened links, in place, for any SU(n ≥ 2).

    ``psi`` (nsites, n) is a quenched matter field added to each staple as
    ``cob·ψ(s+μ̂)ψ†(s)`` with ``cob = matter_coupling/β_g``; pass ``cob = 0``
    (with a dummy 1×n ``psi``) for the pure-gauge ensemble.
    """
    ndim = links.shape[0]
    nsites = links.shape[1]
    n = links.shape[-1]
    staple = np.empty((n, n), dtype=np.complex128)
    t1 = np.empty((n, n), dtype=np.complex128)
    t2 = np.empty((n, n), dtype=np.complex128)
    w = np.empty((n, n), dtype=np.complex128)
    blk = np.empty((2, 2), dtype=np.complex128)
    aq = np.empty((2, 2), dtype=np.complex128)
    x = np.empty((2, 2), dtype=np.complex128)
    r = np.empty((2, 2), dtype=np.complex128)
    has_matter = cob != 0.0
    for _ in range(n_sweeps):
        for mu in range(ndim):
            for s in range(nsites):
                _staple_sum_flat(links, fwd, bwd, mu, s, staple, t1, t2)
                if has_matter:
                    _add_matter(staple, psi, fwd, mu, s, cob)
                if n == 2:
                    _heatbath_su2_nb(staple, beta_g, rng, links[mu, s], aq, x)
                else:
                    _cm_update_nb(links[mu, s], staple, beta_g, rng,
                                  w, blk, aq, x, r)


@njit(cache=True)
def overrelaxation_sweep_flat(links, fwd, bwd, n_sweeps, psi, cob):
    """Microcanonical overrelaxation sweep(s) over flattened links, in place.

    Reflections are taken about the effective staple (gauge + matter), so
    the full exponent is conserved; ``cob = 0`` gives the pure-gauge case.
    """
    ndim = links.shape[0]
    nsites = links.shape[1]
    n = links.shape[-1]
    staple = np.empty((n, n), dtype=np.complex128)
    t1 = np.empty((n, n), dtype=np.complex128)
    t2 = np.empty((n, n), dtype=np.complex128)
    w = np.empty((n, n), dtype=np.complex128)
    blk = np.empty((2, 2), dtype=np.complex128)
    aq = np.empty((2, 2), dtype=np.complex128)
    r = np.empty((2, 2), dtype=np.complex128)
    has_matter = cob != 0.0
    for _ in range(n_sweeps):
        for mu in range(ndim):
            for s in range(nsites):
                _staple_sum_flat(links, fwd, bwd, mu, s, staple, t1, t2)
                if has_matter:
                    _add_matter(staple, psi, fwd, mu, s, cob)
                u = links[mu, s]
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        if n == 2 and (i, j) != (0, 1):
                            continue
                        _mat_mul(u, staple, w)
                        blk[0, 0] = w[i, i]
                        blk[0, 1] = w[i, j]
                        blk[1, 0] = w[j, i]
                        blk[1, 1] = w[j, j]
                        k = _quat_part(blk, aq)
                        if k < 1e-14:
                            continue
                        for a in range(2):
                            for b in range(2):
                                aq[a, b] = aq[a, b] / k
                        # r = (V†)² — the exact energy-preserving reflection
                        _mat_dag_mul_bdag(aq, aq, r)
                        _su2_left_apply(r, u, i, j)


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

@njit(cache=True)
def matter_metropolis_sweep_flat(
    psi, links, fwd, bwd, rng,
    temperature, matter_coupling, quartic_u,
    frac_penalty, frac_targets, fracs, step_scale,
):
    """Metropolis sweep over flattened matter sites, in place.

    Same update and random-draw order as the reference implementation in
    :mod:`project_genesis.annealed_matter` (per site: 2P standard normals
    for the proposal, then one uniform for the accept).  ``fracs`` is the
    running phase-fraction vector, updated in place on accepts.  Returns
    the acceptance fraction.
    """
    nsites = psi.shape[0]
    n = psi.shape[1]
    ndim = links.shape[0]
    inv_t = 1.0 / temperature
    p_prop = np.empty(n, dtype=np.complex128)
    delta_f = np.empty(n, dtype=np.float64)
    n_acc = 0

    for s in range(nsites):
        norm_sq = 0.0
        for a in range(n):
            zr = rng.standard_normal()
            zi = rng.standard_normal()
            p_prop[a] = psi[s, a] + step_scale * (zr + 1j * zi)
            norm_sq += abs(p_prop[a]) ** 2
        inv_norm = 1.0 / np.sqrt(norm_sq)
        for a in range(n):
            p_prop[a] *= inv_norm

        de_coh = 0.0
        for mu in range(ndim):
            sf = fwd[mu, s]
            sb = bwd[mu, s]
            for i in range(n):
                diff_i = np.conj(p_prop[i] - psi[s, i])
                acc_fwd = 0.0 + 0.0j
                for j in range(n):
                    acc_fwd += links[mu, s, i, j] * psi[sf, j]
                de_coh += (diff_i * acc_fwd).real
            for i in range(n):
                acc_bwd = 0.0 + 0.0j
                for j in range(n):
                    acc_bwd += links[mu, sb, i, j] * (p_prop[j] - psi[s, j])
                de_coh += (np.conj(psi[sb, i]) * acc_bwd).real

        de_quart = 0.0
        for a in range(n):
            de_quart += abs(p_prop[a]) ** 4 - abs(psi[s, a]) ** 4

        de_frac = 0.0
        for a in range(n):
            delta_f[a] = (abs(p_prop[a]) ** 2 - abs(psi[s, a]) ** 2) / nsites
            de_frac += (2.0 * (fracs[a] - frac_targets[a]) * delta_f[a]
                        + delta_f[a] ** 2)
        de_frac *= nsites

        gain = inv_t * (matter_coupling * de_coh
                        + quartic_u * de_quart
                        - frac_penalty * de_frac)
        if gain >= 0.0 or rng.random() < np.exp(gain):
            for a in range(n):
                psi[s, a] = p_prop[a]
                fracs[a] += delta_f[a]
            n_acc += 1

    return n_acc / nsites


@njit(cache=True)
def wilson_loop_flat(links, fwd, bwd, r_ext, t_ext, mu, nu):
    """⟨Re Tr W(R,T)⟩/n averaged over all starting sites, plane (mu, nu)."""
    nsites = links.shape[1]
    n = links.shape[-1]
    prod = np.empty((n, n), dtype=np.complex128)
    tmp = np.empty((n, n), dtype=np.complex128)
    total = 0.0
    for s in range(nsites):
        for i in range(n):
            for j in range(n):
                prod[i, j] = 1.0 + 0.0j if i == j else 0.0 + 0.0j
        pos = s
        for _ in range(r_ext):
            _mat_mul(prod, links[mu, pos], tmp)
            for i in range(n):
                for j in range(n):
                    prod[i, j] = tmp[i, j]
            pos = fwd[mu, pos]
        for _ in range(t_ext):
            _mat_mul(prod, links[nu, pos], tmp)
            for i in range(n):
                for j in range(n):
                    prod[i, j] = tmp[i, j]
            pos = fwd[nu, pos]
        for _ in range(r_ext):
            pos = bwd[mu, pos]
            _mat_mul_bdag(prod, links[mu, pos], tmp)
            for i in range(n):
                for j in range(n):
                    prod[i, j] = tmp[i, j]
        for _ in range(t_ext):
            pos = bwd[nu, pos]
            _mat_mul_bdag(prod, links[nu, pos], tmp)
            for i in range(n):
                for j in range(n):
                    prod[i, j] = tmp[i, j]
        tr = 0.0
        for i in range(n):
            tr += prod[i, i].real
        total += tr / n
    return total / nsites


@njit(cache=True)
def polyakov_loop_flat(links, fwd, axis, start_sites, nt):
    """Spatially averaged Polyakov loop (real part), normalised to [0, 1]."""
    n = links.shape[-1]
    prod = np.empty((n, n), dtype=np.complex128)
    tmp = np.empty((n, n), dtype=np.complex128)
    total = 0.0 + 0.0j
    for idx in range(start_sites.shape[0]):
        s = start_sites[idx]
        for i in range(n):
            for j in range(n):
                prod[i, j] = 1.0 + 0.0j if i == j else 0.0 + 0.0j
        pos = s
        for _ in range(nt):
            _mat_mul(prod, links[axis, pos], tmp)
            for i in range(n):
                for j in range(n):
                    prod[i, j] = tmp[i, j]
            pos = fwd[axis, pos]
        tr = 0.0 + 0.0j
        for i in range(n):
            tr += prod[i, i]
        total += tr
    return (total / (start_sites.shape[0] * n)).real
