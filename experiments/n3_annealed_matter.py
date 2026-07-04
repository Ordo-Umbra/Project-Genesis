"""Annealed matter: does the N⋆=3 junction network survive back-reaction?

The thermal-selection experiment quenched the sector field and let only
the gauge links fluctuate.  This experiment closes the loop: the sector
field ψ(x) ∈ ℂ^P and the SU(P) links co-evolve in one joint Gibbs measure
(see :mod:`project_genesis.annealed_matter`), so gauge fluctuations
back-react on the junction network and the network itself is thermal.

Protocol
--------
For each palette size P and each matter temperature T (each T started
fresh — not sequentially annealed — so every point is a well-defined
ensemble):

1. Build the converged deterministic P-sector network, embed it as ψ, pin
   the phase fractions at the network's own values, and pre-thermalise the
   links around it.
2. Run joint sweeps (matter Metropolis + exact matter-coupled heat-bath +
   overrelaxation) to equilibrium at temperature T, then measure with
   binned-jackknife errors:

   - **neutrality** — the full-palette junction density of the *annealed*
     field (the topological integration term, now thermal),
   - the sector order parameter ⟨max_a |ψ_a|²⟩ and the surviving sector
     count,
   - **retention** R = covariant coherence / naive coherence of the
     current ψ (the gauge tax, now with back-reaction),
   - ΔC from the amplitude gradient energy, and the dressed selection
     functional S = ΔC + κ·w·neutrality·R — every ingredient measured in
     the same annealed ensemble,
   - junction/wall/bulk curvature ratios (localization check, masks
     recomputed per measurement from the current labels).

Verdicts sought: the melting temperature of the junction network per P
(where neutrality falls to half its cold value), whether argmax_P S stays
at 3 below melting, and whether curvature localization appears once the
matter can fluctuate (it could not for quenched matter, where the
constraints are integrable).

Honest scope: the corner potential and pinned fractions explicitly break
local SU(P) (deliberate — sectors define preferred frames in the URP
picture); 2-D; one deterministic seed network per P; groups compared at
equal (β_g, g_m, T).

Usage::

    python experiments/n3_annealed_matter.py --output-dir artifacts/n3_annealed
    python experiments/n3_annealed_matter.py --quick     # smoke run
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
    phase_fractions,
    sector_amplitudes,
    thermal_junction_analysis,
)
from project_genesis.gauge import (  # noqa: E402
    covariant_coherence,
    curvature_density,
    naive_coherence,
    random_unitary,
)
from project_genesis.gauge_mc import heatbath_sweep  # noqa: E402
from project_genesis.multiphase import (  # noqa: E402
    _total_gradient_energy,
    interface_mask,
)

from confinement_sigma_scan import jackknife  # noqa: E402
from n3_thermal_selection import build_sector_network  # noqa: E402


def step_for_temperature(t: float) -> float:
    """Initial proposal step; refined adaptively during thermalisation."""
    return min(0.6, max(0.03, 0.4 * math.sqrt(t)))


def run_point(
    network: dict,
    *,
    temperature: float,
    beta_g: float,
    matter_coupling: float,
    quartic_u: float,
    frac_penalty: float,
    n_therm: int,
    n_meas: int,
    n_skip: int,
    bin_size: int,
    seed: int,
    beta: float,
) -> dict:
    """Anneal one (P, T) point from the deterministic network; measure."""
    psi = network["psi"].astype(np.complex128).copy()
    n = psi.shape[-1]
    spatial = psi.shape[:-1]
    ndim = len(spatial)
    n_links = ndim * int(np.prod(spatial))
    frac_targets = phase_fractions(psi)
    # fixed retention denominator: the COLD network's naive coherence, so
    # R = "fraction of the original integration retained" stays meaningful
    # even when the thermal field itself melts (its own naive coherence → 0)
    naive_cold = naive_coherence(psi) / n_links
    rng = np.random.default_rng(seed)
    step = step_for_temperature(temperature)

    # pre-thermalise links around the initial (still-cold) matter field
    links = random_unitary(rng, n, (ndim, *spatial), special=True, scale=0.7)
    for _ in range(40):
        links, _ = heatbath_sweep(
            links, beta_g, rng,
            matter=psi, matter_coupling=matter_coupling / temperature)

    def kw(step_scale):
        return dict(temperature=temperature, beta_g=beta_g,
                    matter_coupling=matter_coupling, quartic_u=quartic_u,
                    frac_penalty=frac_penalty, frac_targets=frac_targets,
                    step_scale=step_scale)

    # adaptive step tuning during thermalisation only (frozen afterwards,
    # so the measured chain uses a fixed, valid proposal)
    acc = 0.0
    block = max(10, n_therm // 12)
    done = 0
    while done < n_therm:
        acc_sum = 0.0
        for _ in range(min(block, n_therm - done)):
            psi, links, acc = joint_sweep(psi, links, rng, **kw(step))
            acc_sum += acc
            done += 1
        mean_acc = acc_sum / block
        if mean_acc < 0.15:
            step = max(0.02, step * 0.7)
        elif mean_acc > 0.4:
            step = min(0.8, step * 1.3)

    # per-meas: neutrality, order, ordered frac, coh/link,
    #           junc curv, wall curv, bulk curv, dC
    samples = np.empty((n_meas, 8))
    for m in range(n_meas):
        for _ in range(n_skip):
            psi, links, acc = joint_sweep(psi, links, rng, **kw(step))
        amps = sector_amplitudes(psi)
        tj = thermal_junction_analysis(amps, n)
        junc = tj["junctions"]
        walls = interface_mask(amps) & ~junc
        bulk = ~interface_mask(amps) & ~junc
        curv = curvature_density(links)
        samples[m] = (
            tj["neutrality"],
            float(np.max(amps, axis=0).mean()),
            tj["ordered_fraction"],
            covariant_coherence(psi, links) / n_links,
            float(curv[junc].mean()) if junc.any() else float("nan"),
            float(curv[walls].mean()) if walls.any() else float("nan"),
            float(curv[bulk].mean()) if bulk.any() else float("nan"),
            float(beta * _total_gradient_energy(amps).mean()),
        )

    def jk(cols, est):
        return jackknife(samples[:, cols], est, bin_size=bin_size)

    neut, neut_err = jk([0], lambda v: float(v[0]))
    retention, retention_err = jk(
        [3], lambda v: float(v[0] / naive_cold))
    junc_ratio, junc_ratio_err = jk(
        [4, 6], lambda v: float(v[0] / v[1]) if v[1] > 1e-12
        else float("nan"))
    wall_ratio, wall_ratio_err = jk(
        [5, 6], lambda v: float(v[0] / v[1]) if v[1] > 1e-12
        else float("nan"))

    mean_load = float(samples[:, 7].mean()) / beta
    return {
        "temperature": temperature,
        "neutrality": neut,
        "neutrality_err": neut_err,
        "order_parameter": float(samples[:, 1].mean()),
        "ordered_fraction": float(samples[:, 2].mean()),
        "retention": retention,
        "retention_err": retention_err,
        "delta_c": float(samples[:, 7].mean()),
        "kappa": 1.0 / (1.0 + mean_load),
        "junction_curvature_ratio": junc_ratio,
        "junction_curvature_ratio_err": junc_ratio_err,
        "wall_curvature_ratio": wall_ratio,
        "wall_curvature_ratio_err": wall_ratio_err,
        "fractions": [float(f) for f in phase_fractions(psi)],
        "acceptance": float(acc),
        "step_scale": float(step),
    }


def melting_temperature(points: list[dict]) -> float:
    """First T where neutrality falls below half its coldest value."""
    if not points:
        return float("nan")
    ref = points[0]["neutrality"]
    if not ref > 0:
        return float("nan")
    for pt in points:
        if pt["neutrality"] < 0.5 * ref:
            return pt["temperature"]
    return float("inf")


def make_figure(results, weights, w_ref, path):
    """Melting curves, retention, and annealed S — Okabe–Ito, CVD-safe."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {2: "#0072B2", 3: "#E69F00", 4: "#009E73", 5: "#CC79A7"}
    MARKERS = {2: "o", 3: "s", 4: "^", 5: "D"}
    phases = sorted(results)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax in (ax1, ax2, ax3):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("matter temperature T")
        ax.set_xscale("log")

    for P in phases:
        pts = results[P]
        ts = [p["temperature"] for p in pts]
        ax1.errorbar(ts, [p["neutrality"] for p in pts],
                     yerr=[p["neutrality_err"] for p in pts],
                     color=COLORS[P], marker=MARKERS[P], ms=5, lw=1.6,
                     capsize=3, label=f"P={P}", zorder=3)
        ax2.errorbar(ts, [p["retention"] for p in pts],
                     yerr=[p["retention_err"] for p in pts],
                     color=COLORS[P], marker=MARKERS[P], ms=5, lw=1.6,
                     capsize=3, label=f"P={P}", zorder=3)
        s_vals = [p["delta_c"] + p["kappa"] * w_ref * p["neutrality"]
                  * p["retention"] for p in pts]
        ax3.plot(ts, s_vals, color=COLORS[P], marker=MARKERS[P], ms=5,
                 lw=1.6, label=f"P={P}", zorder=3)

    ax1.set_ylabel("neutrality (full-palette junction density)")
    ax1.set_title("Junction-network melting")
    ax1.legend(frameon=False)
    ax2.set_ylabel("coherence retention R")
    ax2.set_title("Integration with back-reaction")
    ax2.set_ylim(bottom=0.0)
    ax2.legend(frameon=False)
    ax3.set_ylabel(f"S = ΔC + κ·w·neutrality·R   (w={w_ref:g})")
    ax3.set_title("Annealed selection functional")
    ax3.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--phases", type=str, default="2,3,4,5")
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--diffusion", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.09)
    p.add_argument("--temperatures", type=str,
                   default="0.02,0.05,0.1,0.2,0.4,0.8,1.6")
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--frac-penalty", type=float, default=20.0)
    p.add_argument("--weights", type=str, default="0.05,0.1,0.2,0.5")
    p.add_argument("--n-therm", type=int, default=300)
    p.add_argument("--n-meas", type=int, default=100)
    p.add_argument("--n-skip", type=int, default=3)
    p.add_argument("--bin-size", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_annealed")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size, args.steps = 24, 200
        args.temperatures = "0.05,0.8"
        args.n_therm, args.n_meas = 60, 30
        args.phases = "3,4"

    phases = [int(x) for x in args.phases.split(",") if x.strip()]
    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    weights = [float(w) for w in args.weights.split(",") if w.strip()]
    w_ref = weights[len(weights) // 2]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Stage 1: deterministic P-sector networks ({args.size}^2)",
          flush=True)
    networks = {}
    for P in phases:
        networks[P] = build_sector_network(
            P, size=args.size, steps=args.steps, gamma=args.gamma,
            diffusion=args.diffusion, dt=args.dt, seed=args.seed,
            beta=args.beta)
        print(f"  P={P}: neutrality(cold)={networks[P]['neutrality']:.5f}",
              flush=True)

    print(f"\nStage 2: annealed (ψ, U) ensembles "
          f"(β_g={args.beta_g}, g_m={args.matter_coupling}, "
          f"u={args.quartic_u}, λ_c={args.frac_penalty}, T ∈ {temps})",
          flush=True)
    results: dict[int, list[dict]] = {P: [] for P in phases}
    for P in phases:
        for ti, t in enumerate(temps):
            pt = run_point(
                networks[P], temperature=t, beta_g=args.beta_g,
                matter_coupling=args.matter_coupling,
                quartic_u=args.quartic_u, frac_penalty=args.frac_penalty,
                n_therm=args.n_therm, n_meas=args.n_meas,
                n_skip=args.n_skip, bin_size=args.bin_size,
                seed=args.seed + 1000 * P + ti, beta=args.beta)
            results[P].append(pt)
            print(f"  P={P} T={t:5.2f}: neut={pt['neutrality']:.5f}"
                  f"±{pt['neutrality_err']:.5f}  order={pt['order_parameter']:.3f}"
                  f"  R={pt['retention']:.3f}±{pt['retention_err']:.3f}"
                  f"  junc/bulk={pt['junction_curvature_ratio']:.3f}"
                  f"±{pt['junction_curvature_ratio_err']:.3f}"
                  f"  acc={pt['acceptance']:.2f}", flush=True)

    lines = ["annealed N*=3 — verdicts", "=" * 40, ""]
    for P in phases:
        tm = melting_temperature(results[P])
        lines.append(f"P={P}: junction-network melting T ≈ {tm:g}")
    lines.append("")
    lines.append(f"argmax_P of annealed S = ΔC + κ·w·neutrality·R (w={w_ref:g}):")
    picks = []
    for ti, t in enumerate(temps):
        s_by_p = {
            P: (results[P][ti]["delta_c"]
                + results[P][ti]["kappa"] * w_ref
                * results[P][ti]["neutrality"] * results[P][ti]["retention"])
            for P in phases
        }
        best = max(s_by_p, key=lambda pp: s_by_p[pp])
        picks.append((t, best))
    lines.append("  " + ", ".join(f"T={t:g}→P*={b}" for t, b in picks))
    n3 = sum(1 for _, b in picks if b == 3)
    lines.append(f"  selection at P=3 in {n3}/{len(picks)} temperatures")
    if 3 in phases:
        devs = [max(abs(p["junction_curvature_ratio"] - 1.0),
                    abs(p["wall_curvature_ratio"] - 1.0))
                for p in results[3]
                if not math.isnan(p["junction_curvature_ratio"])]
        if devs:
            lines.append(
                f"P=3 curvature localization: max |ratio − 1| = {max(devs):.3f}"
                " across temperatures")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {
        "params": vars(args),
        "networks": {str(P): {"neutrality_cold": networks[P]["neutrality"],
                              "delta_c_cold": networks[P]["delta_c"]}
                     for P in phases},
        "results": {str(P): results[P] for P in phases},
    }
    with open(os.path.join(args.output_dir, "n3_annealed.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_annealed.png")
        make_figure(results, weights, w_ref, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
