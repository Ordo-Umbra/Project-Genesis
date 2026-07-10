"""Cosmic structure: κ-gravity against an expanding background.

The last step toward cosmology: put the inertial κ-gravity of
`n3_orbital_gravity` in an **expanding background** — every mass given the
Hubble-law recession ``v = H·(r − r_centre)`` — and ask the question structure
formation turns on: does gravity still assemble bound structure against the
expansion, or does the expansion carry everything apart?

Two measurements, both the classic gravity-vs-expansion competition:

1. **Turnaround.**  Two receding masses: below a critical expansion rate they
   decelerate, reach a maximum separation (the **turnaround radius**), and
   **recollapse** into a bound pair; above it they escape.  The turnaround
   radius grows with ``H`` and diverges at the critical rate — the
   spherical-collapse picture, in miniature.
2. **Structure vs expansion.**  A cloud of masses in Hubble flow: at low ``H``
   gravity wins and the cloud collapses into a single bound halo; at high
   ``H`` the expansion wins and the masses disperse to singletons.  The
   largest bound fraction falls through a threshold — **expansion suppresses
   structure**, exactly as a faster-expanding universe forms less.

Honest scope: a *Newtonian, coasting-background* model — the expansion enters
as the initial peculiar-velocity field (Hubble law) and gravity is the
screened κ-force; there is no FLRW metric, no dark-energy term, and no
relativistic horizon.  It captures the essential competition — self-gravity
versus expansion, turnaround, and the suppression of structure by expansion —
not a quantitative cosmology.

Usage::

    python experiments/n3_cosmic_structure.py --output-dir artifacts/n3_cosmic
    python experiments/n3_cosmic_structure.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.capacity_dynamics import evolve_inertial, fof_groups, hubble_flow


def _fk(args):
    return dict(kappa_diffusion=args.D, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0,
                max_iters=args.max_iters)


def two_body_turnaround(args):
    """Separation history of a receding pair over a scan of expansion rate H."""
    L = args.size
    out = []
    for H in args.h_two:
        d = args.d0
        pos = [[L / 2 - d / 2, L / 2], [L / 2 + d / 2, L / 2]]
        vH = H * d / 2.0
        vel = [[-vH, 0.0], [vH, 0.0]]
        h = evolve_inertial(pos, vel, [args.mass, args.mass], shape=(L, L),
                            width=args.width, dt=args.dt, steps=args.steps,
                            escape_radius=args.escape, **_fk(args))
        sep = np.array(h["separation"])
        # escaped = the pair ran out to the escape radius (box edge); a bound
        # orbit oscillates and never reaches it, however large its turnaround
        escaped = bool(sep.max() >= args.escape - 2.0)
        R_ta = float("nan") if escaped else float(sep.max())
        out.append({"H": H, "sep": sep.tolist(), "R_turnaround": R_ta,
                    "escaped": bool(escaped)})
        print(f"  turnaround H={H:.2f}: max sep {sep.max():.1f} "
              f"{'(escape)' if escaped else f'(turnaround, recollapse to {sep[-1]:.1f})'}",
              flush=True)
    return out


def structure_vs_expansion(args):
    """Largest bound fraction of an N-body cloud vs the expansion rate H."""
    L = args.size
    out = []
    for H in args.h_cloud:
        rng = np.random.default_rng(args.seed)
        ctr = np.array([L / 2, L / 2])
        pos = [ctr + rng.uniform(-args.cloud, args.cloud, 2)
               for _ in range(args.n_bodies)]
        vel = hubble_flow(pos, H, center=ctr)
        h = evolve_inertial(pos, vel, [1.0] * args.n_bodies, shape=(L, L),
                            width=args.width, dt=args.dt, steps=args.cloud_steps,
                            **_fk(args))
        final = h["final_positions"]
        sizes = fof_groups(final, args.link)
        frac = sizes[0] / args.n_bodies
        # track cloud size (rms radius about COM) over time
        rms = [float(np.sqrt(np.mean(np.sum(
            (np.array(P) - np.array(P).mean(0)) ** 2, axis=1))))
            for P in h["positions"]]
        out.append({"H": H, "largest_fraction": float(frac),
                    "group_sizes": sizes, "rms": rms,
                    "final": [p.tolist() for p in final]})
        print(f"  cloud H={H:.2f}: largest bound group {sizes[0]}/{args.n_bodies} "
              f"({frac:.0%})  groups={sizes}", flush=True)
    return out


def summarise(turn, cloud, args) -> list[str]:
    lines = ["Cosmic structure: κ-gravity vs expansion — verdict", "=" * 52, ""]
    bound = [t for t in turn if not t["escaped"]]
    esc = [t for t in turn if t["escaped"]]
    hc = (f"between H={bound[-1]['H']:.2f} and H={esc[0]['H']:.2f}"
          if bound and esc else "outside the scanned range")
    lines.append(
        "turnaround: a receding pair decelerates under gravity, reaches a "
        "maximum separation and recollapses — for expansion below a critical "
        f"rate. The turnaround radius grows with H (" +
        ", ".join(f"{t['R_turnaround']:.1f} at H={t['H']:.2f}" for t in bound) +
        f") and gives way to escape {hc} — the spherical-collapse picture.")
    lines.append("")
    lo, hi = cloud[0], cloud[-1]
    lines.append(
        f"structure vs expansion: a cloud in Hubble flow collapses into a single "
        f"bound halo at low expansion ({lo['largest_fraction']:.0%} of the mass "
        f"in one group at H={lo['H']:.2f}) but is dispersed to fragments at high "
        f"expansion ({hi['largest_fraction']:.0%} at H={hi['H']:.2f}) — expansion "
        "suppresses structure, exactly as a faster-expanding universe forms less.")
    lines.append("")
    lines.append(
        "so κ-gravity builds structure against expansion up to a threshold and "
        "loses beyond it: the same gap→matter→gravity chain, now competing with "
        "a cosmological background — the essential ingredient of structure "
        "formation, present.")
    lines.append("")
    lines.append(
        "honest scope: a Newtonian, coasting-background model — expansion is the "
        "initial Hubble velocity field, gravity the screened κ-force; no FLRW "
        "metric, no dark-energy term, no relativistic horizon. It captures the "
        "competition (self-gravity vs expansion, turnaround, suppression of "
        "structure), not a quantitative cosmology.")
    return lines


def make_figure(turn, cloud, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY, RED, PURPLE = (
        "#0072B2", "#E69F00", "#009E73", "#999999", "#c0392b", "#8551A6")
    fig = plt.figure(figsize=(13, 9.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.26)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))
    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    palette = [BLUE, GREEN, ORANGE, RED, PURPLE, "#111111"]
    # panel 1: two-body turnaround separation(t)
    for t, col in zip(turn, palette):
        style = "--" if t["escaped"] else "-"
        ax1.plot(range(len(t["sep"])), t["sep"], color=col, lw=1.8, ls=style,
                 label=f"H={t['H']:.2f}" + (" (escape)" if t["escaped"] else ""),
                 zorder=3)
    ax1.set_xlabel("time step")
    ax1.set_ylabel("separation d(t)")
    ax1.set_title("Turnaround: rise, halt, recollapse — or escape")
    ax1.legend(frameon=False, fontsize=8)

    # panel 2: turnaround radius vs H (diverges at the critical rate)
    Hs = [t["H"] for t in turn]
    Rta = [t["R_turnaround"] for t in turn]
    bound = [(t["H"], t["R_turnaround"]) for t in turn if not t["escaped"]]
    if bound:
        ax2.plot([b[0] for b in bound], [b[1] for b in bound], color=BLUE,
                 marker="o", ms=7, lw=1.8, zorder=3)
    esc_H = [t["H"] for t in turn if t["escaped"]]
    if esc_H:
        ax2.axvspan(min(esc_H) - 0.005, max(Hs) + 0.02, color=RED, alpha=0.08,
                    zorder=1)
        ax2.text(min(esc_H), ax2.get_ylim()[1] * 0.5 if bound else 1, " escape",
                 color=RED, fontsize=9, va="center")
    ax2.set_xlabel("expansion rate H")
    ax2.set_ylabel("turnaround radius R_ta")
    ax2.set_title("Critical expansion: R_ta grows, then escape")

    # panel 3: structure metric vs H
    Hc = [c["H"] for c in cloud]
    frac = [c["largest_fraction"] for c in cloud]
    ax3.plot(Hc, frac, color=GREEN, marker="D", ms=7, lw=1.8, zorder=3)
    ax3.axhline(1.0 / args.n_bodies, color=GRAY, ls=":", lw=1.0, zorder=2)
    ax3.text(Hc[0], 1.0 / args.n_bodies + 0.02, "fully dispersed",
             color=GRAY, fontsize=8)
    ax3.set_xlabel("expansion rate H")
    ax3.set_ylabel("largest bound fraction")
    ax3.set_title("Expansion suppresses structure")
    ax3.set_ylim(0, 1.05)

    # panel 4: cloud size vs time — collapse (low H) vs dispersal (high H)
    for c, col in zip(cloud, palette):
        ax4.plot(range(len(c["rms"])), c["rms"], color=col, lw=1.8,
                 label=f"H={c['H']:.2f}", zorder=3)
    ax4.set_xlabel("time step")
    ax4.set_ylabel("cloud size (rms radius)")
    ax4.set_title("Cloud collapses (low H) or expands away (high H)")
    ax4.legend(frameon=False, fontsize=8)

    fig.suptitle("Cosmic structure: κ-gravity builds structure against "
                 "expansion — up to a critical rate", fontsize=12, y=0.98)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--mass", type=float, default=1.5)
    p.add_argument("--D", type=float, default=1.0)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--max-iters", type=int, default=4000)
    p.add_argument("--d0", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=140)
    p.add_argument("--escape", type=float, default=44.0)
    p.add_argument("--h-two", type=float, nargs="+",
                   default=[0.0, 0.10, 0.14, 0.18, 0.26])
    p.add_argument("--n-bodies", type=int, default=12)
    p.add_argument("--cloud", type=float, default=11.0)
    p.add_argument("--cloud-steps", type=int, default=110)
    p.add_argument("--link", type=float, default=6.0)
    p.add_argument("--h-cloud", type=float, nargs="+",
                   default=[0.0, 0.08, 0.16, 0.26])
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_cosmic")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size = 64
        args.steps = 70
        args.cloud_steps = 55
        args.n_bodies = 8
        args.h_two = [0.0, 0.12, 0.24]
        args.h_cloud = [0.0, 0.24]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"cosmic structure, L={args.size}^2, ξ=√(D/r)={np.sqrt(args.D/args.r):.1f}",
          flush=True)
    turn = two_body_turnaround(args)
    cloud = structure_vs_expansion(args)

    lines = summarise(turn, cloud, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args),
               "turnaround": [{"H": t["H"], "R_turnaround": t["R_turnaround"],
                               "escaped": t["escaped"]} for t in turn],
               "structure": [{"H": c["H"], "largest_fraction": c["largest_fraction"],
                              "group_sizes": c["group_sizes"]} for c in cloud]}
    with open(os.path.join(args.output_dir, "n3_cosmic_structure.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_cosmic_structure.png")
        make_figure(turn, cloud, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
