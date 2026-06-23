'''Pure-gauge Monte Carlo sampling on the Wilson action (2-D start).

This module provides the stochastic sampling layer required for thermodynamic
confinement signatures on top of the deterministic gauge primitives.

v1 scope
--------
- Metropolis updates driven solely by the pure Wilson action (local plaquette ΔS).
- Rectangular Wilson loop measurement (2-D).
- Polyakov loop (compact temporal direction).
- Creutz ratio helper.
- High-level thermalise + measure driver for rapid experimentation.

All functions follow the documentation, naming, and numerical style of
project_genesis.gauge.
'''

from __future__ import annotations

import numpy as np

from project_genesis.gauge import (
    _dagger,
    random_unitary,
    wilson_action,
)


def _local_delta_wilson(links: np.ndarray, mu: int, site: tuple[int, int], u_prop: np.ndarray) -> float:
    """Exact ΔS contributed by the two plaquettes that contain the updated link.

    Pure Wilson action, ndim == 2. Standard local update used in lattice gauge MC.
    """
    ndim = links.shape[0]
    assert ndim == 2, "Local delta implemented only for ndim == 2"
    x, y = site
    nx, ny = links.shape[1], links.shape[2]
    nu = 1 - mu
    n = links.shape[-1]

    def re_tr_one_minus_p(p: np.ndarray) -> float:
        return float(np.real(n - np.trace(p)))

    pA_old = (
        links[mu, x, y]
        @ links[nu, (x + 1) % nx, y]
        @ _dagger(links[mu, x, (y + 1) % ny])
        @ _dagger(links[nu, x, y])
    )
    sA_old = re_tr_one_minus_p(pA_old)

    pB_old = (
        links[mu, x, (y - 1) % ny]
        @ links[nu, (x + 1) % nx, (y - 1) % ny]
        @ _dagger(links[mu, x, y])
        @ _dagger(links[nu, x, (y - 1) % ny])
    )
    sB_old = re_tr_one_minus_p(pB_old)

    s_old_local = sA_old + sB_old

    pA_new = (
        u_prop
        @ links[nu, (x + 1) % nx, y]
        @ _dagger(links[mu, x, (y + 1) % ny])
        @ _dagger(links[nu, x, y])
    )
    sA_new = re_tr_one_minus_p(pA_new)

    pB_new = (
        links[mu, x, (y - 1) % ny]
        @ links[nu, (x + 1) % nx, (y - 1) % ny]
        @ _dagger(u_prop)
        @ _dagger(links[nu, x, (y - 1) % ny])
    )
    sB_new = re_tr_one_minus_p(pB_new)

    return (sA_new + sB_new) - s_old_local


def metropolis_sweep(
    links: np.ndarray,
    beta_g: float,
    rng: np.random.Generator,
    *,
    n_sweeps: int = 1,
    step_scale: float = 0.20,
) -> tuple[np.ndarray, float]:
    """Single-link Metropolis updates on the pure Wilson action using local ΔS.

    Returns updated links and the overall acceptance rate.
    Tune step_scale to keep acceptance in ~0.30–0.50 range.
    """
    ndim = links.shape[0]
    assert ndim == 2, "v1 Metropolis implemented for ndim == 2 only"
    *spatial, n, _ = links.shape[1:]
    nx, ny = spatial

    links_new = links.copy()
    n_acc = 0
    n_tot = 0

    for _ in range(n_sweeps):
        for mu in range(ndim):
            for ix in range(nx):
                for iy in range(ny):
                    u_old = links_new[mu, ix, iy].copy()
                    v = random_unitary(rng, n, special=(n > 1), scale=step_scale)
                    u_prop = u_old @ v

                    delta_s = _local_delta_wilson(links_new, mu, (ix, iy), u_prop)

                    if delta_s <= 0.0 or rng.random() < np.exp(-beta_g * delta_s):
                        links_new[mu, ix, iy] = u_prop
                        n_acc += 1
                    n_tot += 1

    return links_new, n_acc / max(1, n_tot)


def wilson_loop_2d(links: np.ndarray, rx: int, ry: int) -> tuple[float, np.ndarray]:
    """Spatially averaged Re(Tr W(rx, ry))/N and the per-site complex loop values."""
    ndim = links.shape[0]
    assert ndim == 2
    *spatial, n, _ = links.shape[1:]
    nx, ny = spatial
    loops = np.zeros(spatial, dtype=np.complex128)
    dag = _dagger

    for ix in range(nx):
        for iy in range(ny):
            prod = np.eye(n, dtype=np.complex128)
            x, y = ix, iy
            for _ in range(rx):
                prod = prod @ links[0, x % nx, y % ny]
                x = (x + 1) % nx
            for _ in range(ry):
                prod = prod @ links[1, x % nx, y % ny]
                y = (y + 1) % ny
            for _ in range(rx):
                prod = prod @ dag(links[0, (x - 1) % nx, y % ny])
                x = (x - 1) % nx
            for _ in range(ry):
                prod = prod @ dag(links[1, x % nx, (y - 1) % ny])
                y = (y - 1) % ny
            loops[ix, iy] = np.trace(prod) / n

    return float(np.real(loops).mean()), loops


def polyakov_loop(links: np.ndarray, temporal_axis: int = 1) -> float:
    """Spatially averaged real part of the traced temporal Wilson line."""
    ndim = links.shape[0]
    assert ndim == 2
    *spatial, n, _ = links.shape[1:]
    nx, ny = spatial
    axis = temporal_axis
    pl_sum = 0.0 + 0.0j
    n_sites = 0
    for ix in range(nx):
        for iy in range(ny):
            prod = np.eye(n, dtype=np.complex128)
            pos = [ix, iy]
            for _ in range(spatial[axis]):
                prod = prod @ links[axis, pos[0] % nx, pos[1] % ny]
                pos[axis] = (pos[axis] + 1) % spatial[axis]
            pl_sum += np.trace(prod)
            n_sites += 1
    return float(np.real(pl_sum / (n_sites * n)))


def creutz_ratio(w_ij: float, w_im1_jm1: float, w_i_jm1: float, w_im1_j: float) -> float:
    """Creutz ratio χ(I,J) ≈ σ a² (string tension in lattice units)."""
    if any(w <= 0 for w in (w_ij, w_im1_jm1, w_i_jm1, w_im1_j)):
        return np.nan
    ratio = (w_ij * w_im1_jm1) / (w_i_jm1 * w_im1_j)
    return -np.log(ratio)


def thermalize_and_measure_pure_gauge(
    size: int,
    n: int,
    beta_g: float,
    rng: np.random.Generator,
    *,
    n_therm: int = 200,
    n_meas: int = 40,
    n_skip: int = 5,
    step_scale: float = 0.20,
    loop_sizes: list[tuple[int, int]] | None = None,
    temporal_axis: int = 1,
) -> tuple[dict, np.ndarray]:
    """Thermalise a pure-gauge configuration and measure Wilson + Polyakov loops.

    Returns a structured dictionary with acceptance rate, loop averages,
    sample Creutz ratios, and the final links for further analysis.
    """
    if loop_sizes is None:
        loop_sizes = [(1, 1), (2, 2), (3, 3), (4, 4)]

    spatial = (size, size)
    links = random_unitary(rng, n, (2, *spatial), special=(n > 1), scale=0.7)

    # Thermalisation
    acc_history: list[float] = []
    for sweep in range(n_therm):
        links, acc = metropolis_sweep(
            links, beta_g, rng, n_sweeps=1, step_scale=step_scale
        )
        acc_history.append(acc)
        if (sweep + 1) % max(1, n_therm // 5) == 0:
            print(f"  therm {sweep + 1:4d}/{n_therm}   ⟨acc⟩ = {np.mean(acc_history[-10:]):.3f}")

    mean_acc = float(np.mean(acc_history))

    # Production measurements
    results: dict[str, list[float]] = {f"W_{rx}_{ry}": [] for rx, ry in loop_sizes}
    polyakov_vals: list[float] = []

    for _ in range(n_meas):
        for _ in range(n_skip):
            links, _ = metropolis_sweep(
                links, beta_g, rng, n_sweeps=1, step_scale=step_scale
            )
        for rx, ry in loop_sizes:
            w_avg, _ = wilson_loop_2d(links, rx, ry)
            results[f"W_{rx}_{ry}"].append(w_avg)
        polyakov_vals.append(polyakov_loop(links, temporal_axis=temporal_axis))

    # Simple Creutz ratios
    creutz_samples = []
    if (2, 2) in loop_sizes and (1, 1) in loop_sizes:
        for i in range(min(5, len(results["W_2_2"]))):
            creutz_samples.append(creutz_ratio(
                results["W_2_2"][i],
                results["W_1_1"][i],
                results["W_1_1"][i],
                results["W_1_1"][i],
            ))

    summary = {
        "beta_g": float(beta_g),
        "size": size,
        "n": n,
        "n_therm": n_therm,
        "n_meas": n_meas,
        "mean_acceptance_therm": mean_acc,
        "loop_averages": {k: float(np.mean(v)) for k, v in results.items()},
        "polyakov_mean": float(np.mean(polyakov_vals)),
        "creutz_ratios_sample": creutz_samples,
        "final_wilson_action": float(wilson_action(links)),
    }
    return summary, links
