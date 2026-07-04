"""Unpinning the fractions: the S_P sector symmetry breaks for real.

The melting studies pinned the phase fractions — the thermodynamic
analogue of volume conservation — which forces sector *coexistence*.  In
that ensemble the global S_P permutation symmetry cannot break (all P
sectors are always present by construction), the melt is interface
dissolution, and the scaling ladder duly found a crossover.

This experiment removes the pin (``frac_penalty = 0``).  Now nothing in
the model prefers any sector: the corner potential u·Σ|ψ_a|⁴ is exactly
S_P-symmetric, and at low temperature the system must *choose* one sector
spontaneously.  The sector basis has become dynamical in the meaningful
sense — the ordered state is selected by the system, not the Hamiltonian
— and the order–disorder point can now be a genuine phase transition.
For P = 3 in 2-D the natural universality class is the 3-state Potts
model (continuous transition, γ/ν = 26/15 ≈ 1.733, ν = 5/6).

Measured on a lattice-size ladder with binned-jackknife errors:

- the Potts magnetisation m = (P·n_max − 1)/(P − 1), where n_max is the
  largest sector's site fraction (by argmax label),
- its susceptibility χ_m = N·Var(m),
- the Binder cumulant U₄ = 1 − ⟨m⁴⟩/(3⟨m²⟩²) — meaningful here precisely
  because the symmetry is spontaneous; its size-independent crossing
  locates T_c,
- the χ_max(L) scaling exponent, to compare against Potts γ/ν.

The joint gauge sector is kept (β_g, g_m as in the melting studies) so
the result is directly comparable: same model, one constraint removed.

Usage::

    python experiments/n3_potts_transition.py --output-dir artifacts/n3_potts
    python experiments/n3_potts_transition.py --quick     # smoke run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.annealed_matter import (  # noqa: E402
    joint_sweep,
    sector_amplitudes,
)
from project_genesis.gauge import random_psi, random_unitary  # noqa: E402
from project_genesis.multiphase import sector_labels  # noqa: E402

from confinement_sigma_scan import jackknife  # noqa: E402
from n3_scaling_ladder import parabolic_peak, weighted_loglog_slope  # noqa: E402

POTTS_GAMMA_OVER_NU = 26.0 / 15.0  # 2-D 3-state Potts, exact


def potts_magnetisation(amps: np.ndarray) -> float:
    """m = (P·n_max − 1)/(P − 1) from argmax sector labels ∈ [0, 1]."""
    n_palette = amps.shape[0]
    labels = sector_labels(amps)
    counts = np.bincount(labels.ravel(), minlength=n_palette)
    n_max = counts.max() / labels.size
    return float((n_palette * n_max - 1.0) / (n_palette - 1.0))


def run_point(
    *,
    n_palette: int,
    size: int,
    temperature: float,
    beta_g: float,
    matter_coupling: float,
    quartic_u: float,
    n_therm: int,
    n_meas: int,
    n_skip: int,
    bin_size: int,
    seed: int,
    step_scale: float,
    dim: int = 2,
    start: str = "cold",
) -> dict:
    """One (L, T) point of the unpinned ensemble.

    Cold (ordered) starts are the efficient default for an order–disorder
    scan: melting is downhill and fast, while ordering from a hot start
    is a coarsening process needing O(L²) sweeps.  ``start="hot"`` is
    provided for hysteresis checks (hot/cold disagreement windows are a
    first-order signature).  ``dim`` selects 2-D or 3-D.
    """
    rng = np.random.default_rng(seed)
    spatial = (size,) * dim
    if start == "cold":
        psi = random_psi(rng, n_palette, spatial) * 0.15
        psi[..., 0] += 1.0
        psi = psi / np.linalg.norm(psi, axis=-1, keepdims=True)
    elif start == "hot":
        psi = random_psi(rng, n_palette, spatial)
    else:
        raise ValueError(f"unknown start: {start!r}")
    links = random_unitary(rng, n_palette, (dim, *spatial),
                           special=True, scale=0.7)

    kw = dict(temperature=temperature, beta_g=beta_g,
              matter_coupling=matter_coupling, quartic_u=quartic_u,
              frac_penalty=0.0, step_scale=step_scale)
    acc = 0.0
    # adaptive step during thermalisation only
    step = step_scale
    block = max(10, n_therm // 12)
    done = 0
    while done < n_therm:
        acc_sum = 0.0
        for _ in range(min(block, n_therm - done)):
            kw["step_scale"] = step
            psi, links, acc = joint_sweep(psi, links, rng, **kw)
            acc_sum += acc
            done += 1
        mean_acc = acc_sum / block
        if mean_acc < 0.15:
            step = max(0.02, step * 0.7)
        elif mean_acc > 0.4:
            step = min(0.8, step * 1.3)
    kw["step_scale"] = step

    samples = np.empty((n_meas, 4))  # m, m², m⁴, order
    for m_i in range(n_meas):
        for _ in range(n_skip):
            psi, links, acc = joint_sweep(psi, links, rng, **kw)
        amps = sector_amplitudes(psi)
        m = potts_magnetisation(amps)
        samples[m_i] = (m, m * m, m ** 4,
                        float(np.max(amps, axis=0).mean()))

    def jk(cols, est):
        return jackknife(samples[:, cols], est, bin_size=bin_size)

    n_sites = size ** dim
    m_mean, m_err = jk([0], lambda v: float(v[0]))
    chi, chi_err = jk([0, 1], lambda v: float(n_sites * (v[1] - v[0] ** 2)))
    binder, binder_err = jk(
        [1, 2], lambda v: float(1.0 - v[1] / (3.0 * v[0] ** 2))
        if v[0] > 1e-12 else float("nan"))
    return {
        "size": size,
        "dim": dim,
        "start": start,
        "temperature": temperature,
        "magnetisation": m_mean,
        "magnetisation_err": m_err,
        "susceptibility": chi,
        "susceptibility_err": chi_err,
        "binder": binder,
        "binder_err": binder_err,
        "order_parameter": float(samples[:, 3].mean()),
        "acceptance": float(acc),
        "step_scale": float(step),
    }


def binder_crossing(results: dict[int, list[dict]], sizes: list[int],
                    temps: list[float]) -> float:
    """Estimate T_c from where the two largest sizes' U₄ curves cross."""
    l1, l2 = sizes[-2], sizes[-1]
    u1 = [p["binder"] for p in results[l1]]
    u2 = [p["binder"] for p in results[l2]]
    diff = np.array(u1) - np.array(u2)
    for i in range(1, len(temps)):
        if np.isfinite(diff[i - 1]) and np.isfinite(diff[i]) \
                and diff[i - 1] * diff[i] < 0:
            t0, t1 = temps[i - 1], temps[i]
            d0, d1 = diff[i - 1], diff[i]
            return float(t0 + (t1 - t0) * d0 / (d0 - d1))
    return float("nan")


def make_figure(results, sizes, temps, fit, t_c, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RAMP = ["#b8d4ea", "#7fb2d9", "#4a90c4", "#0072B2", "#004b78"]
    ORANGE, GRAY = "#E69F00", "#999999"
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    ax_m, ax_u, ax_x, ax_s = axes
    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for li, L in enumerate(sizes):
        pts = results[L]
        ts = [p["temperature"] for p in pts]
        c = RAMP[li % len(RAMP)]
        ax_m.errorbar(ts, [p["magnetisation"] for p in pts],
                      yerr=[p["magnetisation_err"] for p in pts], color=c,
                      marker="o", ms=4, lw=1.6, capsize=2, label=f"L={L}",
                      zorder=3)
        ax_u.errorbar(ts, [p["binder"] for p in pts],
                      yerr=[p["binder_err"] for p in pts], color=c,
                      marker="o", ms=4, lw=1.6, capsize=2, label=f"L={L}",
                      zorder=3)
        ax_x.errorbar(ts, [p["susceptibility"] for p in pts],
                      yerr=[p["susceptibility_err"] for p in pts], color=c,
                      marker="o", ms=4, lw=1.6, capsize=2, label=f"L={L}",
                      zorder=3)
    for ax, ylabel, title in (
        (ax_m, "Potts magnetisation m", "Spontaneous sector order"),
        (ax_u, "Binder U₄", "Binder cumulant (crossing ⇒ T_c)"),
        (ax_x, "χ_m = N·Var(m)", "Susceptibility"),
    ):
        ax.set_xlabel("matter temperature T")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    if not math.isnan(t_c):
        for ax in (ax_m, ax_u, ax_x):
            ax.axvline(t_c, color=GRAY, lw=0.9, ls=":", zorder=1)

    peaks = [fit["peaks"][str(L)] for L in sizes]
    ax_s.errorbar(sizes, [p["y_peak"] for p in peaks],
                  yerr=[p["y_peak_err"] for p in peaks], color="#0072B2",
                  marker="o", ms=6, lw=0, capsize=3, label="measured χ_max",
                  zorder=3)
    lr = np.array([min(sizes) * 0.9, max(sizes) * 1.1])
    y0 = peaks[0]["y_peak"]
    ax_s.plot(lr, y0 * (lr / sizes[0]) ** fit["slope"], color="#0072B2",
              lw=1.4,
              label=f"fit: slope {fit['slope']:.2f}±{fit['slope_err']:.2f}",
              zorder=2)
    ax_s.plot(lr, y0 * (lr / sizes[0]) ** POTTS_GAMMA_OVER_NU, color=ORANGE,
              ls="--", lw=1.2, label="3-state Potts (γ/ν ≈ 1.73)", zorder=2)
    ax_s.plot(lr, [y0, y0], color=GRAY, ls=":", lw=1.2,
              label="crossover (slope 0)", zorder=2)
    ax_s.set_xscale("log")
    ax_s.set_yscale("log")
    ax_s.set_xlabel("lattice size L")
    ax_s.set_ylabel("χ_max")
    ax_s.set_title("Peak scaling vs Potts")
    ax_s.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-phases", type=int, default=3)
    p.add_argument("--sizes", type=str, default="16,24,32,48")
    p.add_argument("--temperatures", type=str,
                   default="0.10,0.16,0.22,0.28,0.34,0.42,0.52,0.65")
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--n-therm", type=int, default=400)
    p.add_argument("--n-meas", type=int, default=250)
    p.add_argument("--n-skip", type=int, default=2)
    p.add_argument("--bin-size", type=int, default=10)
    p.add_argument("--step-scale", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_potts")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.sizes = "12,16"
        args.temperatures = "0.12,0.25,0.5"
        args.n_therm, args.n_meas = 80, 50

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    results: dict[int, list[dict]] = {}
    for L in sizes:
        pts = []
        for ti, t in enumerate(temps):
            pt = run_point(
                n_palette=args.n_phases, size=L, temperature=t,
                beta_g=args.beta_g, matter_coupling=args.matter_coupling,
                quartic_u=args.quartic_u, n_therm=args.n_therm,
                n_meas=args.n_meas, n_skip=args.n_skip,
                bin_size=args.bin_size, seed=args.seed + 10000 * L + ti,
                step_scale=args.step_scale)
            pts.append(pt)
            print(f"  L={L} T={t:5.3f}: m={pt['magnetisation']:.4f}"
                  f"±{pt['magnetisation_err']:.4f}  "
                  f"chi={pt['susceptibility']:8.3f}"
                  f"±{pt['susceptibility_err']:.3f}  "
                  f"U4={pt['binder']:.3f}±{pt['binder_err']:.3f}  "
                  f"acc={pt['acceptance']:.2f}", flush=True)
        results[L] = pts

    peaks = {}
    for L in sizes:
        pts = results[L]
        peaks[str(L)] = parabolic_peak(
            temps, [p["susceptibility"] for p in pts],
            [p["susceptibility_err"] for p in pts])
    slope, slope_err = weighted_loglog_slope(
        sizes, [peaks[str(L)]["y_peak"] for L in sizes],
        [peaks[str(L)]["y_peak_err"] for L in sizes])
    t_c = binder_crossing(results, sizes, temps)
    fit = {"peaks": peaks, "slope": slope, "slope_err": slope_err}

    lines = ["unpinned S_P transition — verdict", "=" * 42, ""]
    for L in sizes:
        pk = peaks[str(L)]
        edge = " [at grid edge]" if pk["at_edge"] else ""
        lines.append(f"L={L}: chi_max={pk['y_peak']:.3f}"
                     f"±{pk['y_peak_err']:.3f} at T={pk['x_peak']:.3f}{edge}")
    lines.append("")
    lines.append(f"Binder crossing (two largest sizes): T_c ≈ {t_c:.3f}")
    lines.append(f"chi_max(L) ~ L^b: b = {slope:.2f} ± {slope_err:.2f}")
    lines.append(f"  references: b = 0 (crossover) | "
                 f"b = {POTTS_GAMMA_OVER_NU:.3f} (2-D 3-state Potts)")
    s_zero = abs(slope) / slope_err if slope_err else float("nan")
    s_potts = (abs(slope - POTTS_GAMMA_OVER_NU) / slope_err
               if slope_err else float("nan"))
    if s_zero > 3 and s_potts < 3:
        verdict = (f"GENUINE TRANSITION: chi_max grows with volume "
                   f"({s_zero:.1f}σ from flat), consistent with the "
                   f"3-state Potts class at {s_potts:.1f}σ — unpinning the "
                   "fractions turns the crossover into a real transition")
    elif s_zero > 3:
        verdict = (f"TRANSITION-like (b {s_zero:.1f}σ from 0) but "
                   f"{s_potts:.1f}σ from the Potts exponent — scaling "
                   "window or universality question, needs a longer ladder")
    else:
        verdict = f"no volume growth detected (b = {slope:.2f}±{slope_err:.2f})"
    lines.append("verdict: " + verdict)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": vars(args),
               "results": {str(L): results[L] for L in sizes},
               "fit": fit, "t_c": t_c}
    with open(os.path.join(args.output_dir, "n3_potts.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_potts.png")
        make_figure(results, sizes, temps, fit, t_c, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
