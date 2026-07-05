"""Latent heat or not: energy histograms at the 3-D sector transition.

The 3-D first-order hunt found suggestive-but-inconclusive signatures
(persistent hysteresis, deepening Binder minima).  The decisive
small-lattice instrument is the **energy histogram**: a first-order
transition has latent heat, so at the transition the per-site matter
energy distribution is *bimodal* — two peaks (ordered/disordered branches)
whose separation Δe (the latent heat) stays finite as L grows, with a
valley that deepens.  At a continuous transition the peaks merge and
Δe → 0.

Protocol per (L, T): two long chains, one cold-started and one
hot-started (near a weakly-first-order point single chains tunnel rarely,
so the two starts sample the two branches; the *pooled* histogram then
shows both peaks — with the honest caveat that the branch weights are set
by the starts, not by the ensemble, so peak *separation* is the reliable
observable and relative peak heights are not).

Observables per (L, T):
- pooled per-site energy histogram e = −(g_m·coherence + u·quartic)/N
  (the T-conjugate part of the joint measure — the quantity whose
  discontinuity is the latent heat),
- branch separation Δe = |⟨e⟩_hot − ⟨e⟩_cold| with jackknife errors,
- a bimodality flag: whether the pooled histogram has two peaks separated
  by a valley deeper than the statistical noise.

Verdict logic: Δe finite and stable/growing with L across the transition
window ⇒ first-order behaviour; Δe shrinking toward zero with L ⇒
continuous.

Usage::

    python experiments/n3_latent_heat.py --output-dir artifacts/n3_latent
    python experiments/n3_latent_heat.py --quick    # smoke run
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
from project_genesis.gauge import (  # noqa: E402
    covariant_coherence,
    random_psi,
    random_unitary,
)

from confinement_sigma_scan import jackknife  # noqa: E402
from n3_potts_transition import potts_magnetisation  # noqa: E402


def matter_energy_per_site(psi, links, g_m, u) -> float:
    """e = −(g_m·Σ Re ψ†Uψ′ + u·Σ|ψ_a|⁴)/N — the T-conjugate energy."""
    n_sites = int(np.prod(psi.shape[:-1]))
    coh = covariant_coherence(psi, links)
    quart = float(np.sum(np.abs(psi) ** 4))
    return -(g_m * coh + u * quart) / n_sites


def run_branch(
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
    start: str,
    step_scale: float = 0.3,
) -> dict:
    """One 3-D chain from one start; returns energy samples and summaries."""
    rng = np.random.default_rng(seed)
    spatial = (size,) * 3
    if start == "cold":
        psi = random_psi(rng, n_palette, spatial) * 0.15
        psi[..., 0] += 1.0
        psi = psi / np.linalg.norm(psi, axis=-1, keepdims=True)
    else:
        psi = random_psi(rng, n_palette, spatial)
    links = random_unitary(rng, n_palette, (3, *spatial),
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
        if acc_sum / block < 0.15:
            step = max(0.02, step * 0.7)
        elif acc_sum / block > 0.4:
            step = min(0.8, step * 1.3)

    energies = np.empty(n_meas)
    mags = np.empty(n_meas)
    for i in range(n_meas):
        for _ in range(n_skip):
            psi, links, _ = joint_sweep(psi, links, rng,
                                        step_scale=step, **kw)
        energies[i] = matter_energy_per_site(psi, links,
                                             matter_coupling, quartic_u)
        mags[i] = potts_magnetisation(sector_amplitudes(psi))

    e_mean, e_err = jackknife(energies[:, None], lambda v: float(v[0]),
                              bin_size=bin_size)
    return {"start": start, "energies": energies, "e_mean": e_mean,
            "e_err": e_err, "m_mean": float(mags.mean())}


def bimodality(pooled: np.ndarray, n_bins: int = 30) -> dict:
    """Two-peak/valley structure of the pooled energy histogram."""
    hist, edges = np.histogram(pooled, bins=n_bins)
    smooth = np.convolve(hist, np.ones(3) / 3.0, mode="same")
    peaks = [i for i in range(1, n_bins - 1)
             if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]
             and smooth[i] > 0.05 * smooth.max()]
    if len(peaks) < 2:
        return {"bimodal": False, "valley_ratio": float("nan"),
                "peak_separation": 0.0}
    # take the two tallest peaks; valley = minimum between them
    peaks = sorted(sorted(peaks, key=lambda i: -smooth[i])[:2])
    i1, i2 = peaks
    valley = float(smooth[i1:i2 + 1].min())
    ratio = valley / min(smooth[i1], smooth[i2])
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"bimodal": bool(ratio < 0.7),
            "valley_ratio": ratio,
            "peak_separation": float(abs(centers[i2] - centers[i1]))}


def make_figure(scan, sizes, temps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RAMP = ["#7fb2d9", "#4a90c4", "#0072B2", "#004b78"]
    fig, axes = plt.subplots(1, len(sizes) + 1,
                             figsize=(4.2 * (len(sizes) + 1), 4.0))
    hist_axes, ax_sep = axes[:-1], axes[-1]
    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # per-size pooled histogram at the most bimodal temperature
    for ax, L in zip(hist_axes, sizes):
        best_t, best = None, None
        for t in temps:
            b = scan[(L, t)]["bimodality"]
            score = (1.0 if b["bimodal"] else 0.0, b["peak_separation"])
            if best is None or score > best:
                best, best_t = score, t
        pooled = np.concatenate([scan[(L, best_t)]["cold"]["energies"],
                                 scan[(L, best_t)]["hot"]["energies"]])
        ax.hist(pooled, bins=30, color="#4a90c4", edgecolor="white",
                linewidth=0.4, zorder=3)
        ax.set_xlabel("energy per site e")
        ax.set_title(f"L={L}, T={best_t:g} (pooled hot+cold)")
    hist_axes[0].set_ylabel("count")

    for li, L in enumerate(sizes):
        seps = [abs(scan[(L, t)]["cold"]["e_mean"]
                    - scan[(L, t)]["hot"]["e_mean"]) for t in temps]
        errs = [math.hypot(scan[(L, t)]["cold"]["e_err"],
                           scan[(L, t)]["hot"]["e_err"]) for t in temps]
        ax_sep.errorbar(temps, seps, yerr=errs, color=RAMP[li % len(RAMP)],
                        marker="o", ms=5, lw=1.6, capsize=2, label=f"L={L}",
                        zorder=3)
    ax_sep.set_xlabel("matter temperature T")
    ax_sep.set_ylabel("branch separation Δe")
    ax_sep.set_title("Latent-heat candidate Δe(L, T)")
    ax_sep.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-phases", type=int, default=3)
    p.add_argument("--sizes", type=str, default="8,10,12,14")
    p.add_argument("--temperatures", type=str, default="0.10,0.12,0.14")
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--n-therm", type=int, default=600)
    p.add_argument("--n-meas", type=int, default=500)
    p.add_argument("--n-skip", type=int, default=2)
    p.add_argument("--bin-size", type=int, default=10)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_latent")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.sizes = "6,8"
        args.temperatures = "0.12"
        args.n_therm, args.n_meas = 100, 80

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    scan: dict[tuple[int, float], dict] = {}
    for L in sizes:
        for ti, t in enumerate(temps):
            entry = {}
            for start in ("cold", "hot"):
                entry[start] = run_branch(
                    n_palette=args.n_phases, size=L, temperature=t,
                    beta_g=args.beta_g,
                    matter_coupling=args.matter_coupling,
                    quartic_u=args.quartic_u, n_therm=args.n_therm,
                    n_meas=args.n_meas, n_skip=args.n_skip,
                    bin_size=args.bin_size,
                    seed=args.seed + 10000 * L + 100 * ti, start=start)
            pooled = np.concatenate([entry["cold"]["energies"],
                                     entry["hot"]["energies"]])
            entry["bimodality"] = bimodality(pooled)
            scan[(L, t)] = entry
            b = entry["bimodality"]
            sep = abs(entry["cold"]["e_mean"] - entry["hot"]["e_mean"])
            err = math.hypot(entry["cold"]["e_err"], entry["hot"]["e_err"])
            print(f"  L={L} T={t:5.3f}: e_cold={entry['cold']['e_mean']:.4f} "
                  f"e_hot={entry['hot']['e_mean']:.4f}  Δe={sep:.4f}±{err:.4f}  "
                  f"m_cold={entry['cold']['m_mean']:.3f} "
                  f"m_hot={entry['hot']['m_mean']:.3f}  "
                  f"bimodal={b['bimodal']} (valley ratio "
                  f"{b['valley_ratio']:.2f})", flush=True)

    lines = ["3-D latent-heat hunt — verdict", "=" * 40, ""]
    # per size: max significant separation over the window
    max_sep = {}
    for L in sizes:
        best = max(
            (abs(scan[(L, t)]["cold"]["e_mean"] - scan[(L, t)]["hot"]["e_mean"]),
             math.hypot(scan[(L, t)]["cold"]["e_err"],
                        scan[(L, t)]["hot"]["e_err"]), t)
            for t in temps)
        max_sep[L] = best
        sep, err, t_at = best
        bim = any(scan[(L, t)]["bimodality"]["bimodal"] for t in temps)
        lines.append(f"L={L}: max Δe = {sep:.4f}±{err:.4f} at T={t_at:g}; "
                     f"pooled histogram bimodal: {bim}")
    lines.append("")
    seps = [max_sep[L][0] for L in sizes]
    sig = [max_sep[L][0] > 4 * max_sep[L][1] for L in sizes]
    shrinking = all(s2 <= s1 * 1.05 for s1, s2 in zip(seps, seps[1:]))
    growing_or_stable = all(s2 >= s1 * 0.7 for s1, s2 in zip(seps, seps[1:]))
    if all(sig) and growing_or_stable and not shrinking:
        verdict = ("FIRST-ORDER: a finite branch separation Δe (latent-heat "
                   "candidate) persists or grows with L, with bimodal pooled "
                   "histograms")
    elif all(sig) and growing_or_stable:
        verdict = ("first-order-consistent: Δe stays finite and significant "
                   "across the ladder (neither clearly growing nor shrinking) "
                   "— stronger than the Binder-minimum evidence, still short "
                   "of a proof; branch weights are start-selected, so Δe is "
                   "the reliable observable")
    else:
        verdict = ("Δe shrinks with L or loses significance — consistent "
                   "with a continuous transition in 3-D (which would be a "
                   "genuine universality deviation from the Potts "
                   "prediction)")
    lines.append("verdict: " + verdict)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": vars(args),
               "scan": {f"{L}_{t:g}": {
                   "cold": {k: v for k, v in scan[(L, t)]["cold"].items()
                            if k != "energies"},
                   "hot": {k: v for k, v in scan[(L, t)]["hot"].items()
                           if k != "energies"},
                   "bimodality": scan[(L, t)]["bimodality"],
               } for L in sizes for t in temps}}
    with open(os.path.join(args.output_dir, "n3_latent_heat.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_latent_heat.png")
        make_figure(scan, sizes, temps, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
