"""Annealed sector matter — the joint (ψ, U) ensemble with back-reaction.

The quenched-matter experiments held the sector field fixed and let the
gauge links fluctuate around it.  This module closes the loop: the
sector-membership field ψ(x) ∈ ℂ^P (unit vectors) and the SU(P) links are
*both* dynamical, sampled from one joint Gibbs measure

    weight ∝ exp( −β_g·S_W
                  + (1/T)·[  g_m·Σ_{x,μ} Re(ψ†(x) U_μ(x) ψ(x+μ̂))
                           + u·Σ_x Σ_a |ψ_a(x)|⁴
                           − λ_c·N·Σ_a (f_a − f_a⁰)² ] )

- **Coherence term** (g_m): the gauge-transported integration — the same
  coupling the quenched experiments used, now felt by the matter too.
- **Corner potential** (u): Σ|ψ_a|⁴ is maximal at the P basis directions,
  the S_P-symmetric potential that makes sectors form (the thermal
  analogue of the multiphase triple-well bulk term).
- **Fraction pinning** (λ_c): f_a = ⟨|ψ_a|²⟩_x are the phase fractions;
  the soft constraint toward the targets f⁰ is the thermodynamic analogue
  of the volume-conserving dynamics — without it the system trivially
  freezes into one sector and no junction network can exist in
  equilibrium.
- **T** is the matter temperature (the melting knob); β_g stays a separate
  gauge coupling.  The links feel the matter through the effective
  coupling g_m/T, handled exactly by the matter-coupled heat-bath.

Note on gauge invariance: the corner potential and the fractions are
defined in the fixed sector basis, so local SU(P) relabelling is
*explicitly* broken — deliberately.  In the URP picture the sectors define
preferred frames and the gauge freedom is the local relabelling the
connection must repair; here the potential ties the frames down and the
sector labels/junction observables are well-defined.

Matter updates are single-site Metropolis moves: ψ′ ∝ ψ + σ·ζ with
ζ ~ complex-normal (renormalised).  Because the proposal noise is
unitarily invariant, the proposal kernel depends only on |⟨ψ, ψ′⟩| and is
exactly symmetric, so plain Metropolis acceptance is correct.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from project_genesis import gauge_mc_kernels as _nbk
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is a declared dependency
    _nbk = None
    _HAVE_NUMBA = False


def sector_amplitudes(psi: np.ndarray) -> np.ndarray:
    """|ψ_a|² as a real ``(P, *spatial)`` array (multiphase-style layout).

    This is the bridge back to the multiphase instruments: sector labels,
    junction counts, neutrality, and gradient energy are all computed on
    the amplitude field.
    """
    return np.moveaxis(np.abs(psi) ** 2, -1, 0)


def phase_fractions(psi: np.ndarray) -> np.ndarray:
    """The P phase fractions f_a = ⟨|ψ_a(x)|²⟩_x (they sum to 1)."""
    return np.abs(psi).reshape(-1, psi.shape[-1]).__pow__(2).mean(axis=0)


def thermal_junction_analysis(
    amps: np.ndarray,
    n_palette: int,
    *,
    margin: float = 0.25,
    smooth_radius: int = 1,
    hood_radius: int = 2,
) -> dict:
    """Noise-robust junction observables for a *thermal* amplitude field.

    The deterministic ``full_palette_junction_density`` reads argmax labels
    straight off the field; on a thermal field, molten cells carry random
    labels, so every neighbourhood spuriously "contains all colours" and
    the plain measure counts label noise as junctions.  This version:

    1. box-smooths each component over a ``(2r+1)^d`` neighbourhood (thermal
       single-cell noise averages out; genuine domains survive),
    2. marks a cell **ordered** only when the smoothed winner beats the
       runner-up by ``margin`` — molten cells contribute no label (they are
       disorder, not colour-neutral junctions),
    3. counts a cell as a junction when its ``(2·hood_radius+1)^d``
       neighbourhood carries ≥ 3 distinct *ordered-cell* labels.  The cell
       itself need not be ordered: genuine junction cores are exactly where
       no colour dominates, and the wider neighbourhood bridges them.

    Defaults calibrated so that a converged deterministic 3-sector network
    scores a clear nonzero density while a pure-noise field scores exactly
    zero (see ``tests/test_annealed_matter.py``).  The absolute scale
    differs from the deterministic measure — use it self-relatively (e.g.
    melting curves) or comparatively across palettes.

    Returns a dict with ``neutrality`` (full-palette junction density),
    ``junctions`` (boolean mask), ``ordered_fraction``, and ``labels``
    (argmax of the smoothed field, −1 on disordered cells).
    """
    smooth = amps.astype(np.float64).copy()
    for axis in range(1, amps.ndim):
        acc = smooth.copy()
        for shift in range(1, smooth_radius + 1):
            acc += np.roll(smooth, shift, axis=axis)
            acc += np.roll(smooth, -shift, axis=axis)
        smooth = acc / (2 * smooth_radius + 1)

    srt = np.sort(smooth, axis=0)
    gap = srt[-1] - srt[-2]
    ordered = gap > margin
    labels = np.argmax(smooth, axis=0)

    from itertools import product

    ndim = labels.ndim
    own = np.where(ordered, 1 << labels, 0)
    bits = own.copy()
    r = hood_radius
    for off in product(range(-r, r + 1), repeat=ndim):
        if not any(off):
            continue
        shifted_bits = own
        for axis, delta in enumerate(off):
            if delta:
                shifted_bits = np.roll(shifted_bits, -delta, axis=axis)
        bits = bits | shifted_bits

    def popcount(arr):
        out = np.zeros_like(arr)
        a = arr.copy()
        while np.any(a):
            out += a & 1
            a >>= 1
        return out

    distinct = popcount(bits)
    full = (1 << n_palette) - 1
    junctions = (bits == full) & (distinct >= 3)
    neutrality = float(np.mean(junctions))
    return {
        "neutrality": neutrality,
        "junctions": junctions,
        "ordered_fraction": float(ordered.mean()),
        "labels": np.where(ordered, labels, -1),
    }


def matter_energy_terms(
    psi: np.ndarray,
    links: np.ndarray,
    *,
    frac_targets: np.ndarray,
) -> dict[str, float]:
    """The three matter energy terms (unscaled by their couplings)."""
    from project_genesis.gauge import covariant_coherence

    n_sites = int(np.prod(psi.shape[:-1]))
    fracs = phase_fractions(psi)
    return {
        "coherence": covariant_coherence(psi, links),
        "quartic": float(np.sum(np.abs(psi) ** 4)),
        "fraction_penalty": float(
            n_sites * np.sum((fracs - frac_targets) ** 2)
        ),
    }


def matter_metropolis_sweep(
    psi: np.ndarray,
    links: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float,
    matter_coupling: float = 1.0,
    quartic_u: float = 1.0,
    frac_penalty: float = 0.0,
    frac_targets: np.ndarray | None = None,
    step_scale: float = 0.3,
    use_numba: bool | None = None,
) -> tuple[np.ndarray, float]:
    """One Metropolis sweep over all matter sites; returns (psi, acceptance).

    Samples ψ from the joint weight's matter factor at temperature ``T``
    with the links held at their current values.  ``use_numba=None`` uses
    the JIT kernel when available; both paths share the same random-draw
    order (2P normals for the proposal, then one uniform for the accept).
    """
    if not temperature > 0.0:
        raise ValueError("temperature must be positive")
    n = psi.shape[-1]
    spatial = psi.shape[:-1]
    ndim = len(spatial)
    n_sites = int(np.prod(spatial))
    if frac_targets is None:
        frac_targets = np.full(n, 1.0 / n)
    frac_targets = np.asarray(frac_targets, dtype=np.float64)

    if use_numba is None:
        use_numba = _HAVE_NUMBA
    if use_numba and _HAVE_NUMBA:
        flat_links, fwd, bwd, _ = _nbk.flatten_links(links)
        psi_flat = np.ascontiguousarray(
            psi.reshape(n_sites, n).astype(np.complex128)
        )
        fracs = np.ascontiguousarray(
            (np.abs(psi_flat) ** 2).mean(axis=0)
        )
        acc = _nbk.matter_metropolis_sweep_flat(
            psi_flat, flat_links, fwd, bwd, rng,
            float(temperature), float(matter_coupling), float(quartic_u),
            float(frac_penalty), frac_targets, fracs, float(step_scale),
        )
        return psi_flat.reshape(*spatial, n), float(acc)

    psi_new = psi.copy()
    fracs = (np.abs(psi_new.reshape(n_sites, n)) ** 2).mean(axis=0)
    n_acc = 0
    inv_t = 1.0 / temperature

    for site in np.ndindex(*spatial):
        p_old = psi_new[site]
        zeta = np.empty(n, dtype=np.complex128)
        for a in range(n):
            zeta[a] = rng.standard_normal() + 1j * rng.standard_normal()
        p_prop = p_old + step_scale * zeta
        p_prop = p_prop / np.linalg.norm(p_prop)

        # coherence with the 2·ndim gauge-transported neighbours
        de_coh = 0.0
        for mu in range(ndim):
            fwd_site = list(site)
            fwd_site[mu] = (site[mu] + 1) % spatial[mu]
            u_fwd = links[(mu,) + tuple(site)]
            p_fwd = psi_new[tuple(fwd_site)]
            bwd_site = list(site)
            bwd_site[mu] = (site[mu] - 1) % spatial[mu]
            u_bwd = links[(mu,) + tuple(bwd_site)]
            p_bwd = psi_new[tuple(bwd_site)]
            de_coh += float(np.real(
                np.conjugate(p_prop - p_old) @ (u_fwd @ p_fwd)
            ))
            de_coh += float(np.real(
                np.conjugate(p_bwd) @ (u_bwd @ (p_prop - p_old))
            ))

        de_quart = float(np.sum(np.abs(p_prop) ** 4 - np.abs(p_old) ** 4))

        # f_a → f_a + δ_a with δ_a = (|ψ'_a|² − |ψ_a|²)/N, so
        # Δ[N·Σ(f−f⁰)²] = N·Σ[2(f−f⁰)δ + δ²]
        delta_f = (np.abs(p_prop) ** 2 - np.abs(p_old) ** 2) / n_sites
        de_frac = float(np.sum(
            2.0 * (fracs - frac_targets) * delta_f + delta_f ** 2
        )) * n_sites

        gain = inv_t * (
            matter_coupling * de_coh
            + quartic_u * de_quart
            - frac_penalty * de_frac
        )
        if gain >= 0.0 or rng.random() < math.exp(gain):
            psi_new[site] = p_prop
            fracs = fracs + delta_f
            n_acc += 1

    return psi_new, n_acc / n_sites


def joint_sweep(
    psi: np.ndarray,
    links: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float,
    beta_g: float,
    matter_coupling: float = 1.0,
    quartic_u: float = 1.0,
    frac_penalty: float = 0.0,
    frac_targets: np.ndarray | None = None,
    step_scale: float = 0.3,
    n_or: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One compound sweep of the joint ensemble: matter, then gauge.

    The links feel the matter at the effective coupling ``g_m/T`` (the
    coherence term carries 1/T in the joint weight), through the exact
    matter-coupled heat-bath plus ``n_or`` overrelaxation sweeps.
    Returns ``(psi, links, matter_acceptance)``.
    """
    from project_genesis.gauge_mc import heatbath_sweep, overrelaxation_sweep

    psi, acc = matter_metropolis_sweep(
        psi, links, rng,
        temperature=temperature, matter_coupling=matter_coupling,
        quartic_u=quartic_u, frac_penalty=frac_penalty,
        frac_targets=frac_targets, step_scale=step_scale,
    )
    eff_coupling = matter_coupling / temperature
    links, _ = heatbath_sweep(links, beta_g, rng,
                              matter=psi, matter_coupling=eff_coupling)
    if n_or:
        links, _ = overrelaxation_sweep(links, rng, n_sweeps=n_or,
                                        matter=psi,
                                        matter_coupling=eff_coupling,
                                        beta_g=beta_g)
    return psi, links, acc
