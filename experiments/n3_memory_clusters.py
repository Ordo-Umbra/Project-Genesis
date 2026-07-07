"""The spatial structure of surviving memory: does it stay connected?

The finite-size experiment (``n3_recall_finite_size.py``) measured a scalar
— at the melt (m = ½), about 71% of the soil is still fertile, on every
lattice size.  But 71% *fertile* is not the same as 71% *usable*: a
recalled seed can root anywhere fertile, yet it can only *spread* — grow a
remembered domain — across a connected patch of fertile ground.  Whether
memory is globally recoverable at criticality is therefore a **percolation**
question, not a fraction: is the surviving soil one spanning continent, or
an archipelago of disconnected islands?

The reference number is the 2-D square-lattice site-percolation threshold
``p_c ≈ 0.593``.  A *randomly* occupied mask percolates above it and
fragments below it.  The measured fertile fraction dips to ≈ 0.52–0.59
right at T_c (``n3_recall_finite_size.py``) — it *straddles* p_c — so
whether surviving memory stays connected through the transition is a
genuine knife-edge, and the thermal field's spatial structure (κ pooling
off the domain walls, not scattering at random) decides which way it falls.

At fixed high recovery (r = 0.8, the overtaking regime) this scans
temperature and measures, on the fertile mask ``κ ≥ threshold``:

1. **P∞(T)** — the largest-cluster fraction (percolation strength), with
   the Potts order m and the fertile fraction overlaid, against the p_c
   line: does the connected backbone of memory collapse where the fertile
   fraction crosses p_c, and how does that compare to where order dies?
2. **spanning probability & χ(T)** — the fraction of snapshots whose
   fertile set spans the torus, and the mean finite-cluster size χ (the
   percolation susceptibility, peaking at de-percolation).
3. **a cluster montage** — the fertile mask coloured by connected component
   at three temperatures (ordered / critical / disordered), showing the
   continent → archipelago → (recovered) story directly.

Usage::

    python experiments/n3_memory_clusters.py --output-dir artifacts/n3_clusters
    python experiments/n3_memory_clusters.py --quick    # smoke run
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

from project_genesis.annealed_matter import joint_sweep, sector_amplitudes
from project_genesis.soil_clusters import (
    SITE_PERCOLATION_PC,
    cluster_report,
    label_periodic,
)

from confinement_sigma_scan import jackknife  # noqa: E402
from n3_potts_transition import potts_magnetisation  # noqa: E402
from n3_recall_recovery import crossing_below  # noqa: E402
from n3_seed_rooting import equilibrate  # noqa: E402


def cluster_point(*, temperature: float, args, seed: int) -> dict:
    """Percolation observables of the fertile mask over measurement snapshots.

    Also returns one representative fertile-cluster label field (the last
    snapshot) for the montage.
    """
    psi, links, kappa, rng, step = equilibrate(
        size=args.size, temperature=temperature, beta_g=args.beta_g,
        matter_coupling=args.matter_coupling, quartic_u=args.quartic_u,
        consumption=args.consumption, kappa_recovery=args.recovery,
        n_therm=args.n_therm, seed=seed)
    kp = dict(consumption=args.consumption, recovery=args.recovery)
    # columns: fertile_fraction, p_inf, spans, m ; chi kept separately
    samples = np.empty((args.n_meas, 4))
    chis = np.empty(args.n_meas)
    last_labels = None
    for i in range(args.n_meas):
        for _ in range(args.n_skip):
            psi, links, kappa, _ = joint_sweep(
                psi, links, rng, temperature=temperature, beta_g=args.beta_g,
                matter_coupling=args.matter_coupling, quartic_u=args.quartic_u,
                frac_penalty=0.0, step_scale=step, kappa=kappa, kappa_params=kp)
        mask = kappa >= args.threshold
        rep = cluster_report(mask)
        samples[i] = (rep["fertile_fraction"], rep["p_inf"],
                      1.0 if rep["spans"] else 0.0,
                      potts_magnetisation(sector_amplitudes(psi)))
        chis[i] = rep["chi"]
        if i == args.n_meas - 1:
            last_labels, _ = label_periodic(mask)

    def jk(col):
        return jackknife(samples[:, [col]], lambda v: float(v[0]),
                         bin_size=args.bin_size)
    ff, ff_err = jk(0)
    pinf, pinf_err = jk(1)
    m, m_err = jk(3)
    return {"temperature": temperature,
            "fertile_fraction": ff, "fertile_fraction_err": ff_err,
            "p_inf": pinf, "p_inf_err": pinf_err,
            "spanning_prob": float(samples[:, 2].mean()),
            "chi": float(chis.mean()),
            "magnetisation": m, "magnetisation_err": m_err,
            "labels": last_labels.tolist() if last_labels is not None else None}


def make_figure(points, temps, montage_temps, order_edge, perc_edge, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, GREEN, GRAY, ORANGE, PURPLE = (
        "#0072B2", "#009E73", "#999999", "#E69F00", "#8551A6")
    fig = plt.figure(figsize=(15, 8.4))
    gs = GridSpec(2, 6, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[0, 3:6])
    for ax in (ax1, ax2):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    ts = [p["temperature"] for p in points]
    # panel 1: P∞, fertile fraction, order m vs T, with p_c reference
    ax1.errorbar(ts, [p["p_inf"] for p in points],
                 yerr=[p["p_inf_err"] for p in points], color=BLUE, marker="o",
                 ms=5, lw=1.8, capsize=2, label="P∞ (largest cluster)", zorder=4)
    ax1.errorbar(ts, [p["fertile_fraction"] for p in points],
                 yerr=[p["fertile_fraction_err"] for p in points], color=ORANGE,
                 marker="^", ms=4, lw=1.4, ls="--", capsize=2,
                 label="fertile fraction", zorder=3)
    ax1.plot(ts, [p["magnetisation"] for p in points], color=GREEN, marker="s",
             ms=4, lw=1.4, ls="--", label="Potts order m", zorder=2)
    ax1.axhline(SITE_PERCOLATION_PC, color=PURPLE, lw=1.1, ls=":",
                label=f"random p_c = {SITE_PERCOLATION_PC:g}", zorder=1)
    if math.isfinite(order_edge):
        ax1.axvline(order_edge, color=GREEN, lw=0.8, ls=":", zorder=1)
    ax1.set_xscale("log")
    ax1.set_xlabel("matter temperature T")
    ax1.set_ylabel("fraction / order")
    ax1.set_title("Percolation strength of surviving memory")
    ax1.legend(frameon=False, fontsize=8)

    # panel 2: spanning probability + percolation susceptibility χ
    ax2.plot(ts, [p["spanning_prob"] for p in points], color=BLUE, marker="o",
             ms=5, lw=1.8, label="spanning probability", zorder=4)
    ax2.axhline(0.5, color=GRAY, lw=0.8, ls=":", zorder=1)
    if math.isfinite(perc_edge):
        ax2.axvline(perc_edge, color=BLUE, lw=0.8, ls="--", zorder=1,
                    label=f"de-percolation T≈{perc_edge:.3f}")
    ax2.set_xscale("log")
    ax2.set_xlabel("matter temperature T")
    ax2.set_ylabel("spanning probability", color=BLUE)
    ax2.set_ylim(-0.03, 1.03)
    axb = ax2.twinx()
    axb.plot(ts, [p["chi"] for p in points], color=ORANGE, marker="D", ms=4,
             lw=1.4, ls="-", label="χ (mean finite cluster)", zorder=3)
    axb.set_ylabel("χ  (mean finite-cluster size)", color=ORANGE)
    for side in ("top",):
        axb.spines[side].set_visible(False)
    ax2.set_title("Where the memory backbone breaks")
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = axb.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, frameon=False, fontsize=8,
               loc="center left")

    # bottom row: cluster montage at three temperatures
    by_t = {p["temperature"]: p for p in points}
    for j, mt in enumerate(montage_temps[:3]):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        pt = by_t.get(mt)
        if pt is None or pt["labels"] is None:
            ax.axis("off")
            continue
        labels = np.array(pt["labels"])
        n = labels.max()
        # colour each cluster distinctly; background white; largest = bold blue
        rng = np.random.default_rng(0)
        palette = np.zeros((n + 1, 3))
        palette[0] = (1, 1, 1)
        if n >= 1:
            sizes = np.bincount(labels.ravel())[1:]
            biggest = int(sizes.argmax()) + 1
            for lab in range(1, n + 1):
                palette[lab] = (rng.uniform(0.55, 0.9), rng.uniform(0.55, 0.9),
                                rng.uniform(0.55, 0.9))
            palette[biggest] = (0.0, 0.45, 0.70)  # BLUE, the spanning backbone
        ax.imshow(palette[labels], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        span_txt = "spans ✓" if pt["spanning_prob"] >= 0.5 else "fragmented"
        ax.set_title(f"T={mt:g}  (m={pt['magnetisation']:.2f}, "
                     f"P∞={pt['p_inf']:.2f}, {span_txt})", fontsize=9)
    fig.suptitle("Surviving memory: continent → archipelago across the melt",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def percolation_verdict(points, order_edge, perc_edge, fertile_edge) -> list[str]:
    """Human-readable verdict lines from the measured percolation curves.

    ``points`` is the per-temperature list from :func:`cluster_point`.  The
    memory bottleneck is located by the *minimum* percolation strength P∞
    (the continuous order parameter), not by the spanning probability, which
    saturates near 1 and ties.  The refined verdict distinguishes three
    regimes: a spanning-but-stressed backbone, a genuine de-percolation, and
    an untested-edge case.
    """
    lines = ["spatial structure of surviving memory — verdict", "=" * 47, ""]
    lines.append(f"order edge (m<½):             T={order_edge:.3f}")
    lines.append(f"de-percolation edge (span<½): "
                 f"T={'∞ (never)' if math.isinf(perc_edge) else f'{perc_edge:.3f}'}")
    lines.append(f"fertile crosses random p_c:   "
                 f"T={'∞ (never)' if math.isinf(fertile_edge) else f'{fertile_edge:.3f}'}")
    lines.append("")

    # the backbone is thinnest where P∞ is smallest; χ peaks there too
    bottleneck = min(points, key=lambda p: p["p_inf"])
    chi_peak = max(points, key=lambda p: p["chi"])
    min_span = min(p["spanning_prob"] for p in points)
    lines.append(
        f"the memory backbone is driven thinnest at T={bottleneck['temperature']:g}: "
        f"P∞ collapses to {bottleneck['p_inf']:.2f} (from ≈{max(p['p_inf'] for p in points):.2f} "
        f"in the ordered phase) and the percolation susceptibility χ peaks at "
        f"{chi_peak['chi']:.0f} (T={chi_peak['temperature']:g}) — the classic "
        "near-threshold signature, in the same critical region where recall dips")

    always_spans = all(p["spanning_prob"] >= 0.5 for p in points)
    if always_spans:
        msg = (
            "surviving memory STAYS CONNECTED across the melt — but only just. "
            f"A system-spanning fertile path survives at every temperature "
            f"(spanning probability ≥ {min_span:.2f}), yet the backbone thins to "
            "a tenuous filament threaded through an archipelago of disconnected "
            "fertile islands (see the montage). ")
        if math.isfinite(fertile_edge):
            msg += (
                f"Decisively, the fertile fraction dips BELOW the random "
                f"site-percolation threshold p_c={SITE_PERCOLATION_PC:g} "
                f"(crossing it at T={fertile_edge:.3f}) while still spanning: the "
                "thermal field's spatial structure — barren soil confined to thin "
                "domain walls, fertile bulk staying contiguous — keeps memory "
                "globally connected at a density where randomly-placed soil would "
                "already fragment. Memory bends at criticality but does not break.")
        else:
            msg += (
                "The fertile fraction stays above the random threshold "
                f"p_c={SITE_PERCOLATION_PC:g} throughout, so connectivity is never "
                "seriously in doubt on this grid.")
        lines.append(msg)
    elif math.isfinite(perc_edge):
        rel = ("below" if perc_edge < order_edge - 1e-6 else
               "above" if perc_edge > order_edge + 1e-6 else "right at")
        lines.append(
            f"surviving memory DE-PERCOLATES at T={perc_edge:.3f}, {rel} the order "
            f"edge (T={order_edge:.3f}): the fertile soil fragments into "
            "disconnected islands — memory can still root locally but can no "
            "longer spread across the system. Connectivity, not fertility, is the "
            "tighter constraint on recoverable memory.")
    else:
        lines.append(
            "the fertile backbone thins but never fully de-percolates on this grid "
            "— connectivity is stressed most at criticality but a spanning path "
            "survives in the majority of snapshots throughout.")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=40)
    p.add_argument("--temperatures", type=str,
                   default="0.05,0.065,0.08,0.09,0.10,0.11,0.13,0.16,0.20")
    p.add_argument("--recovery", type=float, default=0.8)
    p.add_argument("--consumption", type=float, default=5.0)
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--montage-temps", type=str, default="0.065,0.11,0.20")
    p.add_argument("--n-therm", type=int, default=600)
    p.add_argument("--n-meas", type=int, default=80)
    p.add_argument("--n-skip", type=int, default=3)
    p.add_argument("--bin-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=47)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_clusters")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size = 20
        args.temperatures = "0.06,0.11,0.20"
        args.montage_temps = "0.06,0.11,0.20"
        args.n_therm, args.n_meas = 200, 40

    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    montage_temps = [float(t) for t in args.montage_temps.split(",") if t.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Fertile-soil percolation at r={args.recovery:g}, c={args.consumption:g}, "
          f"L={args.size} (p_c={SITE_PERCOLATION_PC:g})", flush=True)
    points = []
    for ti, t in enumerate(temps):
        pt = cluster_point(temperature=t, args=args, seed=args.seed + 100 * ti)
        points.append(pt)
        print(f"  T={t:.3f}: fertile={pt['fertile_fraction']:.3f}  "
              f"P∞={pt['p_inf']:.3f}  span={pt['spanning_prob']:.2f}  "
              f"χ={pt['chi']:.1f}  m={pt['magnetisation']:.3f}", flush=True)

    order_edge = crossing_below(temps, [p["magnetisation"] for p in points], 0.5)
    perc_edge = crossing_below(temps, [p["spanning_prob"] for p in points], 0.5)
    fertile_edge = crossing_below(
        temps, [p["fertile_fraction"] for p in points], SITE_PERCOLATION_PC)

    lines = percolation_verdict(points, order_edge, perc_edge, fertile_edge)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": vars(args),
               "edges": {"order_edge": order_edge, "perc_edge": perc_edge,
                         "fertile_pc_edge": fertile_edge},
               "points": [{k: v for k, v in p.items() if k != "labels"}
                          for p in points]}
    with open(os.path.join(args.output_dir, "n3_memory_clusters.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_memory_clusters.png")
        make_figure(points, temps, montage_temps, order_edge, perc_edge, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
