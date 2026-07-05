"""Dynamical capacity at criticality: does a binding κ budget move the melt?

The S-at-criticality experiment used the diagnostic κ proxy.  This one
promotes κ to the **dynamical capacity field** the URP engine actually
uses — consumed by the distinction load, regenerating with slack,
diffusing between regions — and lets it gate the matter–gauge coherence
coupling *locally* in the joint (ψ, U, κ) system (an adaptive steady
state, not a fixed-Hamiltonian ensemble: URP capacity is a driven
resource, deliberately).

Two sharp questions, both descended from the toy-model observation that
κ is the main governor of which structures can emerge:

1. **Does capacity develop a trough at criticality?**  ΔC (the load κ is
   consumed by) peaks exactly at T_c — so ⟨κ⟩(T) should dip there,
   deepest at the transition, with the wall/bulk κ split showing where
   the scarcity concentrates.
2. **Does a binding budget move the transition and the S-optimum?**
   Scanning the consumption strength c: with abundant capacity the
   S-functional selects the critical neighbourhood (previous verdict);
   as κ binds, the coherence channel is taxed exactly where the load is
   highest — does T_c shift, and does argmax_T S move away from
   criticality?

S is now assembled with the *measured* capacity: S = ΔC + ⟨κ⟩·w·I.

Usage::

    python experiments/n3_kappa_criticality.py --output-dir artifacts/n3_kappa
    python experiments/n3_kappa_criticality.py --quick    # smoke run
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
    interface_mask,
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
    consumption: float,
    kappa_recovery: float,
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
    """One (T, consumption) point of the adaptive (ψ, U, κ) steady state."""
    rng = np.random.default_rng(seed)
    spatial = (size, size)
    psi = random_psi(rng, n_palette, spatial) * 0.15
    psi[..., 0] += 1.0
    psi = psi / np.linalg.norm(psi, axis=-1, keepdims=True)
    links = random_unitary(rng, n_palette, (2, *spatial),
                           special=True, scale=0.7)
    kappa = np.ones(spatial)
    kp = dict(consumption=consumption, recovery=kappa_recovery)

    kw = dict(temperature=temperature, beta_g=beta_g,
              matter_coupling=matter_coupling, quartic_u=quartic_u,
              frac_penalty=0.0, kappa_params=kp)
    step = step_scale
    block = max(10, n_therm // 12)
    done = 0
    while done < n_therm:
        acc_sum = 0.0
        for _ in range(min(block, n_therm - done)):
            psi, links, kappa, acc = joint_sweep(
                psi, links, rng, step_scale=step, kappa=kappa, **kw)
            acc_sum += acc
            done += 1
        mean_acc = acc_sum / block
        if mean_acc < 0.15:
            step = max(0.02, step * 0.7)
        elif mean_acc > 0.4:
            step = min(0.8, step * 1.3)

    # per-meas: load, coherence I, m, kappa mean, kappa wall, kappa bulk
    samples = np.empty((n_meas, 6))
    for i in range(n_meas):
        for _ in range(n_skip):
            psi, links, kappa, acc = joint_sweep(
                psi, links, rng, step_scale=step, kappa=kappa, **kw)
        amps = sector_amplitudes(psi)
        walls = interface_mask(amps)
        samples[i] = (
            float(_total_gradient_energy(amps).mean()),
            coherence_integration(amps, radius=kernel_radius,
                                  decay=kernel_decay),
            potts_magnetisation(amps),
            float(kappa.mean()),
            float(kappa[walls].mean()) if walls.any() else float("nan"),
            float(kappa[~walls].mean()) if (~walls).any() else float("nan"),
        )

    def jk(cols, est):
        return jackknife(samples[:, cols], est, bin_size=bin_size)

    load, load_err = jk([0], lambda v: float(v[0]))
    coh, coh_err = jk([1], lambda v: float(v[0]))
    m, m_err = jk([2], lambda v: float(v[0]))
    kap, kap_err = jk([3], lambda v: float(v[0]))
    return {
        "size": size,
        "temperature": temperature,
        "consumption": consumption,
        "delta_c": beta * load,
        "delta_c_err": beta * load_err,
        "coherence": coh,
        "coherence_err": coh_err,
        "magnetisation": m,
        "magnetisation_err": m_err,
        "kappa_mean": kap,
        "kappa_err": kap_err,
        "kappa_wall": float(np.nanmean(samples[:, 4])),
        "kappa_bulk": float(np.nanmean(samples[:, 5])),
        "acceptance": float(acc),
    }


def s_of(pt: dict, w: float) -> float:
    """S with the *measured* capacity: ΔC + ⟨κ⟩·w·I."""
    return pt["delta_c"] + pt["kappa_mean"] * w * pt["coherence"]


def make_figure(results, consumptions, temps, w_ref, t_c_anchor, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RAMP = ["#b8d4ea", "#7fb2d9", "#4a90c4", "#0072B2"]
    GRAY = "#999999"
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

    for ci, c in enumerate(consumptions):
        pts = results[c]
        ts = [p["temperature"] for p in pts]
        col = RAMP[ci % len(RAMP)]
        ax1.errorbar(ts, [p["kappa_mean"] for p in pts],
                     yerr=[p["kappa_err"] for p in pts], color=col,
                     marker="o", ms=4, lw=1.6, capsize=2, label=f"c={c:g}",
                     zorder=3)
        ax2.errorbar(ts, [p["magnetisation"] for p in pts],
                     yerr=[p["magnetisation_err"] for p in pts], color=col,
                     marker="o", ms=4, lw=1.6, capsize=2, label=f"c={c:g}",
                     zorder=3)
        s_vals = [s_of(p, w_ref) for p in pts]
        ax3.plot(ts, s_vals, color=col, marker="o", ms=4, lw=1.6,
                 label=f"c={c:g}", zorder=3)
        i_max = int(np.argmax(s_vals))
        ax3.plot([ts[i_max]], [s_vals[i_max]], marker="*", ms=14, color=col,
                 zorder=4)

    ax1.set_ylabel("⟨κ⟩ (dynamical capacity)")
    ax1.set_title("Capacity across the transition")
    ax1.legend(frameon=False, fontsize=8)
    ax2.set_ylabel("Potts magnetisation m")
    ax2.set_title("The transition vs consumption strength")
    ax2.legend(frameon=False, fontsize=8)
    ax3.set_ylabel(f"S = ΔC + ⟨κ⟩·w·I   (w={w_ref:g})")
    ax3.set_title("S with measured capacity (★ = max)")
    ax3.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-phases", type=int, default=3)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--temperatures", type=str,
                   default="0.03,0.05,0.07,0.085,0.10,0.115,0.13,0.16,0.22,0.35")
    p.add_argument("--consumptions", type=str, default="0.5,2.0,5.0,15.0")
    p.add_argument("--kappa-recovery", type=float, default=0.1)
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.09)
    p.add_argument("--kernel-radius", type=int, default=3)
    p.add_argument("--kernel-decay", type=float, default=1.0)
    p.add_argument("--weights", type=str, default="0.05")
    p.add_argument("--n-therm", type=int, default=800)
    p.add_argument("--n-meas", type=int, default=200)
    p.add_argument("--n-skip", type=int, default=3)
    p.add_argument("--bin-size", type=int, default=10)
    p.add_argument("--step-scale", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--t-c", type=float, default=0.10)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_kappa")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size = 16
        args.temperatures = "0.05,0.10,0.22"
        args.consumptions = "0.5,15.0"
        args.n_therm, args.n_meas = 150, 60

    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    consumptions = [float(c) for c in args.consumptions.split(",") if c.strip()]
    weights = [float(w) for w in args.weights.split(",") if w.strip()]
    w_ref = weights[0]
    os.makedirs(args.output_dir, exist_ok=True)

    results: dict[float, list[dict]] = {}
    for c in consumptions:
        pts = []
        for ti, t in enumerate(temps):
            pt = run_point(
                n_palette=args.n_phases, size=args.size, temperature=t,
                beta_g=args.beta_g, matter_coupling=args.matter_coupling,
                quartic_u=args.quartic_u, consumption=c,
                kappa_recovery=args.kappa_recovery, n_therm=args.n_therm,
                n_meas=args.n_meas, n_skip=args.n_skip,
                bin_size=args.bin_size,
                seed=args.seed + 1000 * int(c * 10) + ti,
                step_scale=args.step_scale, beta=args.beta,
                kernel_radius=args.kernel_radius,
                kernel_decay=args.kernel_decay)
            pts.append(pt)
            print(f"  c={c:5.1f} T={t:5.3f}: ⟨κ⟩={pt['kappa_mean']:.4f}"
                  f"±{pt['kappa_err']:.4f} (wall {pt['kappa_wall']:.3f} / "
                  f"bulk {pt['kappa_bulk']:.3f})  m={pt['magnetisation']:.3f}  "
                  f"ΔC={pt['delta_c']:.5f}  I={pt['coherence']:.4f}  "
                  f"acc={pt['acceptance']:.2f}", flush=True)
        results[c] = pts

    lines = ["dynamical κ at criticality — verdicts", "=" * 44, ""]
    for c in consumptions:
        pts = results[c]
        ts = [p["temperature"] for p in pts]
        kaps = [p["kappa_mean"] for p in pts]
        i_min = int(np.argmin(kaps))
        edge = " [at grid edge]" if i_min in (0, len(ts) - 1) else ""
        # transition location: steepest m drop
        ms = [p["magnetisation"] for p in pts]
        drops = [ms[i] - ms[i + 1] for i in range(len(ms) - 1)]
        i_dr = int(np.argmax(drops))
        t_half = 0.5 * (ts[i_dr] + ts[i_dr + 1])
        s_vals = [s_of(p, w_ref) for p in pts]
        i_s = int(np.argmax(s_vals))
        wall_bulk = np.nanmean([p["kappa_wall"] / p["kappa_bulk"]
                                for p in pts
                                if not math.isnan(p["kappa_wall"])])
        lines.append(
            f"c={c:g}: κ trough at T={ts[i_min]:g}{edge} "
            f"(⟨κ⟩={kaps[i_min]:.3f}); m-drop near T≈{t_half:.3f}; "
            f"argmax_T S(w={w_ref:g}) = {ts[i_s]:g}; "
            f"κ wall/bulk ≈ {wall_bulk:.3f}")
    lines.append("")
    troughs = {c: results[c][int(np.argmin([p['kappa_mean']
                for p in results[c]]))]["temperature"]
               for c in consumptions}
    near_tc = [c for c, t in troughs.items()
               if 0.5 * args.t_c <= t <= 2.0 * args.t_c]
    if near_tc:
        lines.append(
            f"κ develops its trough in the critical region for c ∈ "
            f"{sorted(near_tc)} — capacity is consumed exactly where "
            "distinction peaks (the ΔC maximum at T_c)")
    s_opts = {c: results[c][int(np.argmax([s_of(p, w_ref)
              for p in results[c]]))]["temperature"] for c in consumptions}
    lines.append("argmax_T S vs consumption: " + ", ".join(
        f"c={c:g}→T={t:g}" for c, t in sorted(s_opts.items())))
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": vars(args),
               "results": {f"{c:g}": results[c] for c in consumptions}}
    with open(os.path.join(args.output_dir, "n3_kappa_criticality.json"),
              "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_kappa_criticality.png")
        make_figure(results, consumptions, temps, w_ref, args.t_c, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
