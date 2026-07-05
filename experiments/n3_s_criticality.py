"""What does the S-functional do at a real phase transition?

Everything in this repository exists to test the claim that recursive
systems evolve by climbing ``S = ΔC + κ·ΔI``.  The thermodynamic program
has produced something the deterministic experiments never had: a genuine
critical system — the unpinned sector model with its measured Potts-class
transition at T_c ≈ 0.1.  This experiment asks the obvious-in-retrospect
question: **measure the theory's central functional across the
transition** and see where it is maximised and whether it carries a
critical signature at all.

Per temperature (deep thermalisation — the earlier lesson about slow
melting near T_c is baked in), on the same unpinned (ψ, U) ensembles:

- **ΔC** = β·⟨Σ_a |∇η_a|²⟩ — the distinction load of the amplitude field,
- **κ**  = 1/(1 + mean load) — the diagnostic capacity proxy the
  multiphase instruments use,
- **I**  = the *standing* nonlocal coherence (``coherence_integration``:
  exponential-kernel two-point coherence, the integration half that
  survives at equilibrium),
- **S(w) = ΔC + κ·w·I** for a range of integration weights w,
- the Potts magnetisation m, to anchor T_c on the same ensembles.

The candidate verdicts are all informative: S maximised *just below* T_c
(the ordered-but-critical neighbourhood — the URP-flavoured outcome), S
monotone into the deep ordered phase (the functional rewards order per
se), or S blind to the transition entirely.

Usage::

    python experiments/n3_s_criticality.py --output-dir artifacts/n3_s_crit
    python experiments/n3_s_criticality.py --quick    # smoke run
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
from project_genesis.multiphase import (  # noqa: E402
    _total_gradient_energy,
    coherence_integration,
)

from confinement_sigma_scan import jackknife  # noqa: E402
from n3_potts_transition import potts_magnetisation  # noqa: E402


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
    beta: float,
    kernel_radius: int,
    kernel_decay: float,
) -> dict:
    """One temperature point: S-functional components with jackknife errors."""
    rng = np.random.default_rng(seed)
    spatial = (size, size)
    psi = random_psi(rng, n_palette, spatial) * 0.15
    psi[..., 0] += 1.0
    psi = psi / np.linalg.norm(psi, axis=-1, keepdims=True)
    links = random_unitary(rng, n_palette, (2, *spatial),
                           special=True, scale=0.7)

    kw = dict(temperature=temperature, beta_g=beta_g,
              matter_coupling=matter_coupling, quartic_u=quartic_u,
              frac_penalty=0.0)
    step = step_scale
    block = max(10, n_therm // 12)
    done = 0
    while done < n_therm:
        acc_sum = 0.0
        for _ in range(min(block, n_therm - done)):
            psi, links, acc = joint_sweep(psi, links, rng,
                                          step_scale=step, **kw)
            acc_sum += acc
            done += 1
        mean_acc = acc_sum / block
        if mean_acc < 0.15:
            step = max(0.02, step * 0.7)
        elif mean_acc > 0.4:
            step = min(0.8, step * 1.3)

    # per-meas: load, coherence I, magnetisation
    samples = np.empty((n_meas, 3))
    for i in range(n_meas):
        for _ in range(n_skip):
            psi, links, acc = joint_sweep(psi, links, rng,
                                          step_scale=step, **kw)
        amps = sector_amplitudes(psi)
        samples[i] = (
            float(_total_gradient_energy(amps).mean()),
            coherence_integration(amps, radius=kernel_radius,
                                  decay=kernel_decay),
            potts_magnetisation(amps),
        )

    def jk(cols, est):
        return jackknife(samples[:, cols], est, bin_size=bin_size)

    load, load_err = jk([0], lambda v: float(v[0]))
    coh, coh_err = jk([1], lambda v: float(v[0]))
    m, m_err = jk([2], lambda v: float(v[0]))
    delta_c = beta * load
    kappa = 1.0 / (1.0 + load)
    return {
        "size": size,
        "temperature": temperature,
        "mean_load": load,
        "delta_c": delta_c,
        "delta_c_err": beta * load_err,
        "kappa": kappa,
        "coherence": coh,
        "coherence_err": coh_err,
        "magnetisation": m,
        "magnetisation_err": m_err,
        "acceptance": float(acc),
    }


def s_of(pt: dict, w: float) -> float:
    return pt["delta_c"] + pt["kappa"] * w * pt["coherence"]


def s_err_of(pt: dict, w: float) -> float:
    # dominant error propagation: ΔC and coherence errors in quadrature
    return math.hypot(pt["delta_c_err"], pt["kappa"] * w * pt["coherence_err"])


def make_figure(results, sizes, weights, t_c_anchor, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#E69F00", "#009E73", "#999999"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax in (ax1, ax2, ax3):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("matter temperature T")
        ax.set_xscale("log")
        if not math.isnan(t_c_anchor):
            ax.axvline(t_c_anchor, color=GRAY, lw=0.9, ls=":", zorder=1)

    L = max(sizes)
    pts = results[L]
    ts = [p["temperature"] for p in pts]
    ax1.errorbar(ts, [p["delta_c"] for p in pts],
                 yerr=[p["delta_c_err"] for p in pts], color=BLUE,
                 marker="o", ms=4, lw=1.6, capsize=2, label="ΔC", zorder=3)
    ax1b = ax1.twinx()
    ax1b.errorbar(ts, [p["coherence"] for p in pts],
                  yerr=[p["coherence_err"] for p in pts], color=ORANGE,
                  marker="s", ms=4, lw=1.6, capsize=2,
                  label="standing coherence I", zorder=3)
    ax1b.set_ylabel("standing coherence I", color=ORANGE)
    ax1b.spines["top"].set_visible(False)
    ax1.set_ylabel("ΔC", color=BLUE)
    ax1.set_title(f"The two halves of S (L={L})")
    ax1.errorbar([], [], color=ORANGE, marker="s", label="I (right axis)")
    ax1.legend(frameon=False, fontsize=8)

    ax2.errorbar(ts, [p["magnetisation"] for p in pts],
                 yerr=[p["magnetisation_err"] for p in pts], color=GREEN,
                 marker="o", ms=4, lw=1.6, capsize=2, zorder=3)
    ax2.set_ylabel("Potts magnetisation m")
    ax2.set_title("The transition, on the same ensembles")

    ramp = ["#b8d4ea", "#7fb2d9", "#4a90c4", "#0072B2"]
    for wi, w in enumerate(weights):
        s_vals = [s_of(p, w) for p in pts]
        s_errs = [s_err_of(p, w) for p in pts]
        ax3.errorbar(ts, s_vals, yerr=s_errs, color=ramp[wi % len(ramp)],
                     marker="o", ms=4, lw=1.6, capsize=2, label=f"w={w:g}",
                     zorder=3)
        i_max = int(np.argmax(s_vals))
        ax3.plot([ts[i_max]], [s_vals[i_max]], marker="*", ms=14,
                 color=ramp[wi % len(ramp)], zorder=4)
    ax3.set_ylabel("S = ΔC + κ·w·I")
    ax3.set_title("The S-functional across the transition (★ = max)")
    ax3.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-phases", type=int, default=3)
    p.add_argument("--sizes", type=str, default="32,48")
    p.add_argument("--temperatures", type=str,
                   default="0.03,0.05,0.07,0.085,0.10,0.115,0.13,0.16,0.22,0.35,0.60")
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.09,
                   help="distinction coefficient in ΔC")
    p.add_argument("--kernel-radius", type=int, default=3)
    p.add_argument("--kernel-decay", type=float, default=1.0)
    p.add_argument("--weights", type=str, default="0.02,0.05,0.1,0.2")
    p.add_argument("--n-therm", type=int, default=1000)
    p.add_argument("--n-meas", type=int, default=250)
    p.add_argument("--n-skip", type=int, default=3)
    p.add_argument("--bin-size", type=int, default=10)
    p.add_argument("--step-scale", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--t-c", type=float, default=0.10,
                   help="T_c anchor from the Potts study (for reporting)")
    p.add_argument("--output-dir", type=str, default="artifacts/n3_s_crit")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.sizes = "16"
        args.temperatures = "0.05,0.10,0.22,0.6"
        args.n_therm, args.n_meas = 150, 60

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    weights = [float(w) for w in args.weights.split(",") if w.strip()]
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
                step_scale=args.step_scale, beta=args.beta,
                kernel_radius=args.kernel_radius,
                kernel_decay=args.kernel_decay)
            pts.append(pt)
            print(f"  L={L} T={t:5.3f}: ΔC={pt['delta_c']:.5f}  "
                  f"I={pt['coherence']:.4f}±{pt['coherence_err']:.4f}  "
                  f"κ={pt['kappa']:.4f}  m={pt['magnetisation']:.3f}  "
                  f"acc={pt['acceptance']:.2f}", flush=True)
        results[L] = pts

    lines = ["S-functional at criticality — verdict", "=" * 44, ""]
    L = max(sizes)
    pts = results[L]
    ts = [p["temperature"] for p in pts]
    for w in weights:
        s_vals = [s_of(p, w) for p in pts]
        s_errs = [s_err_of(p, w) for p in pts]
        i_max = int(np.argmax(s_vals))
        # significance of the peak against the deep-ordered and disordered ends
        z_cold = ((s_vals[i_max] - s_vals[0])
                  / math.hypot(s_errs[i_max], s_errs[0]))
        z_hot = ((s_vals[i_max] - s_vals[-1])
                 / math.hypot(s_errs[i_max], s_errs[-1]))
        rel = ("below" if ts[i_max] < args.t_c * 0.95
               else "above" if ts[i_max] > args.t_c * 1.05 else "at")
        lines.append(
            f"w={w:g}: argmax_T S = {ts[i_max]:g} ({rel} T_c={args.t_c:g}); "
            f"peak vs cold end: {z_cold:+.1f}σ, vs hot end: {z_hot:+.1f}σ")
    interior = [w for w in weights
                if 0 < int(np.argmax([s_of(p, w) for p in pts])) < len(ts) - 1]
    lines.append("")
    if interior:
        lines.append(
            f"S has an INTERIOR temperature optimum for w ∈ {interior} — "
            "the functional does not simply reward order or disorder; it "
            "selects a finite-temperature regime (see figure for where it "
            "sits relative to T_c)")
    else:
        lines.append(
            "S is monotone across the scanned range for every weight — "
            "no interior temperature optimum at these couplings")
    # does dC or I carry a visible critical signature?
    m_vals = [p["magnetisation"] for p in pts]
    i_tc = int(np.argmin([abs(t - args.t_c) for t in ts]))
    lines.append(
        f"anchors at T_c≈{args.t_c:g}: m={m_vals[i_tc]:.3f}, "
        f"ΔC={pts[i_tc]['delta_c']:.5f}, I={pts[i_tc]['coherence']:.4f}")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": vars(args),
               "results": {str(L): results[L] for L in sizes}}
    with open(os.path.join(args.output_dir, "n3_s_criticality.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_s_criticality.png")
        make_figure(results, sizes, weights, args.t_c, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
