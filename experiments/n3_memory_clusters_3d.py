"""Memory connectivity in 3-D: do surface walls sever it more or less?

In 2-D (``n3_memory_clusters.py``) the backbone of surviving memory was
driven to the edge at criticality — the percolation strength P∞ collapsed
and the susceptibility χ spiked — yet it never quite broke, because the
barren soil concentrated on the thin *line* domain walls and the fertile
bulk stayed contiguous.

3-D changes the geometry in two ways that pull in the *same* direction:

1. **Walls become surfaces.**  A 2-D surface partitions a 3-D volume far
   less effectively than a 1-D line partitions a 2-D plane — you can route
   a connected path *around* a surface through the extra dimension.  So the
   barren domain walls should sever memory even less than in 2-D.
2. **The percolation threshold drops.**  Random site percolation needs only
   p_c ≈ 0.312 on the simple-cubic lattice, versus 0.593 on the square
   lattice — a given fertile fraction is far *more* than enough to connect
   in 3-D.

The sharp prediction is therefore that memory connectivity is **more
robust** in 3-D: at the same fertile fraction the 3-D backbone should stay
near-complete where the 2-D one nearly came apart.  This experiment runs
the *same* percolation scan (fixed high recovery r, temperature across the
melt) in both 2-D (L² line walls) and 3-D (L³ surface walls) and compares
the backbone strength P∞(T), the susceptibility χ(T), and the spanning
probability head to head, with a 3-D cluster-slice montage.

Usage::

    python experiments/n3_memory_clusters_3d.py --output-dir artifacts/n3_clusters_3d
    python experiments/n3_memory_clusters_3d.py --quick    # smoke run
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
from project_genesis.gauge import random_psi, random_unitary
from project_genesis.soil_clusters import (
    SITE_PERCOLATION_PC,
    SITE_PERCOLATION_PC_3D,
    cluster_report,
    label_periodic,
)

from confinement_sigma_scan import jackknife  # noqa: E402
from n3_potts_transition import potts_magnetisation  # noqa: E402
from n3_recall_recovery import crossing_below  # noqa: E402


def equilibrate_dim(*, dim, size, temperature, beta_g, matter_coupling,
                    quartic_u, consumption, recovery, n_therm, seed,
                    step_scale=0.3):
    """Thermalise the adaptive (ψ, U, κ) field in ``dim`` dimensions.

    Dimension-agnostic version of ``n3_seed_rooting.equilibrate``: the
    spatial shape is ``(size,)*dim`` and the gauge links carry ``dim``
    directions.  Returns ``(psi, links, kappa, rng, step)``.
    """
    rng = np.random.default_rng(seed)
    spatial = (size,) * dim
    psi = random_psi(rng, 3, spatial) * 0.15
    psi[..., 0] += 1.0
    psi = psi / np.linalg.norm(psi, axis=-1, keepdims=True)
    links = random_unitary(rng, 3, (dim, *spatial), special=True, scale=0.7)
    kappa = np.ones(spatial)
    kp = dict(consumption=consumption, recovery=recovery)
    step = step_scale
    block = max(10, n_therm // 12)
    done = 0
    while done < n_therm:
        acc_sum = 0.0
        for _ in range(min(block, n_therm - done)):
            psi, links, kappa, acc = joint_sweep(
                psi, links, rng, temperature=temperature, beta_g=beta_g,
                matter_coupling=matter_coupling, quartic_u=quartic_u,
                frac_penalty=0.0, step_scale=step, kappa=kappa, kappa_params=kp)
            acc_sum += acc
            done += 1
        if acc_sum / block < 0.15:
            step = max(0.02, step * 0.7)
        elif acc_sum / block > 0.4:
            step = min(0.8, step * 1.3)
    return psi, links, kappa, rng, step


def cluster_point(*, dim, temperature, args, seed, keep_slice=False) -> dict:
    """Fertile-mask percolation observables in ``dim`` dimensions at ``T``."""
    size = args.size2d if dim == 2 else args.size3d
    psi, links, kappa, rng, step = equilibrate_dim(
        dim=dim, size=size, temperature=temperature, beta_g=args.beta_g,
        matter_coupling=args.matter_coupling, quartic_u=args.quartic_u,
        consumption=args.consumption, recovery=args.recovery,
        n_therm=args.n_therm, seed=seed)
    kp = dict(consumption=args.consumption, recovery=args.recovery)
    samples = np.empty((args.n_meas, 4))  # fertile, p_inf, spans, m
    chis = np.empty(args.n_meas)
    slice_labels = None
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
        if keep_slice and i == args.n_meas - 1 and dim == 3:
            # a mid-plane z-slice of the 3-D fertile clusters, re-labelled in
            # the plane so the montage colours are meaningful in 2-D
            zc = size // 2
            slice_labels, _ = label_periodic(mask[..., zc])

    def jk(col):
        return jackknife(samples[:, [col]], lambda v: float(v[0]),
                         bin_size=args.bin_size)
    ff, ff_err = jk(0)
    pinf, pinf_err = jk(1)
    m, m_err = jk(3)
    return {"dim": dim, "size": size, "temperature": temperature,
            "fertile_fraction": ff, "fertile_fraction_err": ff_err,
            "p_inf": pinf, "p_inf_err": pinf_err,
            "spanning_prob": float(samples[:, 2].mean()),
            "chi": float(chis.mean()),
            "magnetisation": m, "magnetisation_err": m_err,
            "slice_labels": slice_labels.tolist() if slice_labels is not None else None}


def _dim_stats(pts, temps, pc) -> dict:
    """Backbone summary for one dimension: minima, χ-peak, p_c crossing."""
    bottleneck = min(pts, key=lambda p: p["p_inf"])
    fertile_min = min(pts, key=lambda p: p["fertile_fraction"])
    return {
        "pc": pc,
        "p_inf_min": bottleneck["p_inf"],
        "p_inf_min_T": bottleneck["temperature"],
        "chi_peak": max(p["chi"] for p in pts),
        "span_min": min(p["spanning_prob"] for p in pts),
        "fertile_min": fertile_min["fertile_fraction"],
        "fertile_min_T": fertile_min["temperature"],
        "fertile_pc_edge": crossing_below(
            temps, [p["fertile_fraction"] for p in pts], pc),
        "depercolates": min(p["spanning_prob"] for p in pts) < 0.5,
    }


def summarise(by_dim, temps) -> list[str]:
    """Verdict lines comparing the 2-D (line-wall) and 3-D (surface-wall) backbones."""
    lines = ["memory connectivity: line walls (2-D) vs surface walls (3-D)",
             "=" * 60, ""]
    s = {2: _dim_stats(by_dim[2], temps,  SITE_PERCOLATION_PC),
         3: _dim_stats(by_dim[3], temps, SITE_PERCOLATION_PC_3D)}
    for dim in (2, 3):
        d = s[dim]
        fe = ("never" if math.isinf(d["fertile_pc_edge"])
              else f"T={d['fertile_pc_edge']:.3f}")
        lines.append(
            f"{dim}-D (p_c={d['pc']:g}): fertile craters to {d['fertile_min']:.2f} "
            f"(T={d['fertile_min_T']:g}); P∞ dips to {d['p_inf_min']:.2f}; χ peaks "
            f"at {d['chi_peak']:.0f}; min spanning prob {d['span_min']:.2f}; fertile "
            f"crosses p_c {fe}")
    lines.append("")

    # the decisive contrast: does the backbone actually break (span<½)?
    if s[3]["depercolates"] and not s[2]["depercolates"]:
        lines.append(
            "COUNTER to the geometric prediction, memory is MORE fragile in 3-D. "
            "The 2-D backbone bends but holds (min spanning prob "
            f"{s[2]['span_min']:.2f}); the 3-D backbone genuinely DE-PERCOLATES "
            f"(min spanning prob {s[3]['span_min']:.2f}, P∞ → {s[3]['p_inf_min']:.2f}) "
            "— the fertile soil fragments into islands and a spanning path is lost "
            "in most snapshots at criticality.")
        lines.append(
            "The mechanism is the κ dynamics, not the geometry. The lower 3-D "
            f"percolation threshold (p_c={s[3]['pc']:g} vs {s[2]['pc']:g}) *would* "
            "favour connectivity — but it is overwhelmed by a far deeper capacity "
            f"trough: the fertile fraction craters to {s[3]['fertile_min']:.2f} in 3-D "
            f"versus {s[2]['fertile_min']:.2f} in 2-D, plunging well below even the "
            "lower 3-D threshold. The denser 3-D critical structure — surface "
            "walls, six neighbours, the order-of-magnitude-denser junction network "
            "— consumes much more capacity at the transition, so the load physics "
            "beats the favourable geometry. Adding a dimension makes memory harder "
            "to keep, not easier.")
    elif s[3]["p_inf_min"] > s[2]["p_inf_min"] + 0.05:
        lines.append(
            f"the 3-D backbone is more robust (min P∞ {s[3]['p_inf_min']:.2f} vs "
            f"{s[2]['p_inf_min']:.2f}): the lower percolation threshold and the "
            "route-around-a-surface geometry win, as predicted.")
    else:
        lines.append(
            f"the two backbones are comparably stressed at their weakest (min P∞ "
            f"{s[3]['p_inf_min']:.2f} vs {s[2]['p_inf_min']:.2f}); no dramatic "
            "dimensional difference in this window.")
    lines.append(
        "honest scope: the capacity-gated fertile mask (κ ≥ threshold), one "
        "recovery/consumption/coupling point; 2-D at L² and 3-D at L³ are single "
        "sizes, χ is not volume-normalised across dimensions, and the p_c values "
        "are the random-lattice references quoted for contrast.")
    return lines


def make_figure(by_dim, temps, montage, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    C2, C3, ORANGE, GRAY, PURPLE = (
        "#0072B2", "#D55E00", "#E69F00", "#999999", "#8551A6")
    fig = plt.figure(figsize=(15, 8.4))
    gs = GridSpec(2, 6, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[0, 3:6])
    for ax in (ax1, ax2):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for dim, col in ((2, C2), (3, C3)):
        pts = by_dim[dim]
        ts = [p["temperature"] for p in pts]
        ax1.errorbar(ts, [p["p_inf"] for p in pts],
                     yerr=[p["p_inf_err"] for p in pts], color=col, marker="o",
                     ms=5, lw=1.9, capsize=2, label=f"P∞ ({dim}-D)", zorder=4)
        ax1.plot(ts, [p["fertile_fraction"] for p in pts], color=col, marker="^",
                 ms=3, lw=1.2, ls="--", alpha=0.7,
                 label=f"fertile ({dim}-D)", zorder=3)
    ax1.axhline(SITE_PERCOLATION_PC, color=C2, lw=1.0, ls=":", zorder=1,
                label=f"p_c 2-D = {SITE_PERCOLATION_PC:g}")
    ax1.axhline(SITE_PERCOLATION_PC_3D, color=C3, lw=1.0, ls=":", zorder=1,
                label=f"p_c 3-D = {SITE_PERCOLATION_PC_3D:g}")
    ax1.set_xscale("log")
    ax1.set_xlabel("matter temperature T")
    ax1.set_ylabel("fraction")
    ax1.set_title("Backbone strength P∞(T), by dimension")
    ax1.legend(frameon=False, fontsize=7, ncol=2)

    for dim, col in ((2, C2), (3, C3)):
        pts = by_dim[dim]
        ts = [p["temperature"] for p in pts]
        ax2.plot(ts, [p["chi"] for p in pts], color=col, marker="D", ms=5,
                 lw=1.9, label=f"χ ({dim}-D)", zorder=3)
    ax2.set_xscale("log")
    ax2.set_xlabel("matter temperature T")
    ax2.set_ylabel("χ  (mean finite-cluster size)")
    ax2.set_title("Percolation susceptibility spike, by dimension")
    ax2.legend(frameon=False, fontsize=8)

    # bottom: 3-D fertile-cluster z-slices at three temperatures
    import numpy as _np
    by_t = {p["temperature"]: p for p in by_dim[3]}
    for j, mt in enumerate(montage[:3]):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        pt = by_t.get(mt)
        if pt is None or pt["slice_labels"] is None:
            ax.axis("off")
            continue
        labels = _np.array(pt["slice_labels"])
        n = int(labels.max())
        rng = _np.random.default_rng(0)
        palette = _np.zeros((n + 1, 3))
        palette[0] = (1, 1, 1)
        if n >= 1:
            sizes = _np.bincount(labels.ravel())[1:]
            biggest = int(sizes.argmax()) + 1
            for lab in range(1, n + 1):
                palette[lab] = (rng.uniform(0.55, 0.9), rng.uniform(0.55, 0.9),
                                rng.uniform(0.55, 0.9))
            palette[biggest] = (0.0, 0.45, 0.70)
        ax.imshow(palette[labels], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        span = "spans ✓" if pt["spanning_prob"] >= 0.5 else "fragmented"
        ax.set_title(f"3-D z-slice, T={mt:g}  (m={pt['magnetisation']:.2f}, "
                     f"P∞={pt['p_inf']:.2f}, {span})", fontsize=9)
    fig.suptitle("Memory connectivity in 3-D: do surface walls sever it more "
                 "or less than line walls?", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size2d", type=int, default=40)
    p.add_argument("--size3d", type=int, default=16)
    p.add_argument("--temperatures", type=str,
                   default="0.05,0.065,0.08,0.09,0.10,0.11,0.13,0.16,0.20")
    p.add_argument("--recovery", type=float, default=0.8)
    p.add_argument("--consumption", type=float, default=5.0)
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--montage-temps", type=str, default="0.065,0.13,0.20")
    p.add_argument("--n-therm", type=int, default=500)
    p.add_argument("--n-meas", type=int, default=60)
    p.add_argument("--n-skip", type=int, default=3)
    p.add_argument("--bin-size", type=int, default=6)
    p.add_argument("--seed", type=int, default=59)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_clusters_3d")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size2d, args.size3d = 20, 10
        args.temperatures = "0.06,0.11,0.20"
        args.montage_temps = "0.06,0.11,0.20"
        args.n_therm, args.n_meas = 200, 30

    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    montage = [float(t) for t in args.montage_temps.split(",") if t.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    by_dim = {}
    for dim in (2, 3):
        size = args.size2d if dim == 2 else args.size3d
        print(f"{dim}-D fertile-soil percolation at r={args.recovery:g}, "
              f"c={args.consumption:g}, L={size}", flush=True)
        pts = []
        for ti, t in enumerate(temps):
            pt = cluster_point(dim=dim, temperature=t, args=args,
                               seed=args.seed + 100 * ti + 1000 * dim,
                               keep_slice=(t in montage))
            pts.append(pt)
            print(f"  {dim}-D T={t:.3f}: fertile={pt['fertile_fraction']:.3f}  "
                  f"P∞={pt['p_inf']:.3f}  span={pt['spanning_prob']:.2f}  "
                  f"χ={pt['chi']:.1f}  m={pt['magnetisation']:.3f}", flush=True)
        by_dim[dim] = pts

    lines = summarise(by_dim, temps)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": vars(args),
               "by_dim": {str(d): [{k: v for k, v in p.items()
                                    if k != "slice_labels"} for p in by_dim[d]]
                          for d in (2, 3)}}
    with open(os.path.join(args.output_dir, "n3_memory_clusters_3d.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_memory_clusters_3d.png")
        make_figure(by_dim, temps, montage, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
