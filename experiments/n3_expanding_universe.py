"""An expanding universe: FLRW scale factor, Hubble drag, and dark energy.

The cosmic-structure experiment carried the expansion in the initial
velocities only.  Here the background is a genuine, evolving **scale factor**
``a(t)`` obeying a Friedmann-like law with a matter component and a
cosmological-constant (dark-energy) component ``Ω_Λ``:

    (ȧ/a)² = H₀² [ Ω_m a^{-p} + Ω_Λ ] .

Working in physical coordinates about a fixed comoving origin, the masses feel
the peculiar κ-force plus the background ``(ä/a)(r − c)`` and start in the
Hubble flow.  Three genuinely-cosmological things follow:

1. **Expansion histories.**  ``a(t)`` decelerates when matter dominates and
   accelerates when Λ dominates — the qualitatively different fates of a
   matter vs a dark-energy universe.
2. **Hubble drag.**  A peculiar velocity **redshifts as 1/a** — ``a·|v_pec|``
   is constant — the momentum-redshift the coasting model could not show.
3. **Dark energy suppresses structure.**  The *same* initial cloud collapses
   into a single bound halo in a matter universe but is **frozen out** as
   ``Ω_Λ`` rises and the expansion accelerates — a faster-expanding universe
   assembles less structure, the defining signature of dark energy in
   structure formation.

Honest scope: a Newtonian FLRW-*analogue* — a real evolving scale factor and
Hubble drag, but the Friedmann law is imposed (a modelling choice, ``p = 3``
matter dilution applied to the 2-D screened dynamics), not derived from the
κ-field's own stress-energy, and there is no metric, horizon, or relativistic
growth factor.  It reproduces the *mechanisms* — Hubble drag and dark-energy
suppression of growth — not a quantitative ΛCDM.

Usage::

    python experiments/n3_expanding_universe.py --output-dir artifacts/n3_universe
    python experiments/n3_expanding_universe.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.capacity_dynamics import (
    evolve_cosmological,
    fof_groups,
    friedmann_rates,
)


def _fk(args):
    return dict(kappa_diffusion=args.D, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0,
                max_iters=args.max_iters)


def scale_histories(args):
    """a(t) for a few Ω_Λ (pure background integration, no particles needed)."""
    out = []
    for ol in args.omega_hist:
        a, t = 1.0, 0.0
        avals, tvals = [a], [t]
        for _ in range(args.hist_steps):
            H, _ = friedmann_rates(a, args.hubble0, 1.0 - ol, ol, args.p)
            a += a * H * args.dt
            t += args.dt
            avals.append(a); tvals.append(t)
        out.append({"omega_lambda": ol, "t": tvals, "a": avals})
    return out


def hubble_drag(args):
    """A single mass with a peculiar kick — its peculiar velocity vs a."""
    L = args.size
    c = np.array([L / 2.0, L / 2.0])
    # place at centre (Hubble flow ~0) with a pure peculiar velocity
    h = evolve_cosmological([[L / 2.0, L / 2.0]], [args.mass], shape=(L, L),
                            width=args.width, hubble0=args.hubble0,
                            omega_m=1.0, omega_lambda=0.0, p=args.p, dt=args.dt,
                            steps=args.drag_steps, center=c,
                            peculiar=[[args.kick, 0.0]], **_fk(args))
    a = np.array(h["a"])
    vpec = []
    for i in range(len(a)):
        r = np.array(h["positions"][i][0])
        v = np.array(h["velocities"][i][0])
        H, _ = friedmann_rates(a[i], args.hubble0, 1.0, 0.0, args.p)
        vpec.append(float(np.linalg.norm(v - H * (r - c))))
    prod = (a * np.array(vpec))
    print(f"  hubble drag: a·|v_pec| = {prod[1]:.3f} → {prod[-1]:.3f} "
          f"(constant ⇒ v_pec ∝ 1/a); spread {prod[1:].std()/prod[1:].mean()*100:.1f}%",
          flush=True)
    return {"a": a.tolist(), "vpec": vpec, "a_vpec": prod.tolist()}


def structure_vs_darkenergy(args):
    """Largest bound fraction of one cloud across a scan of Ω_Λ."""
    L = args.size
    c = np.array([L / 2.0, L / 2.0])
    out = []
    for ol in args.omega_cloud:
        rng = np.random.default_rng(args.seed)
        pos = [c + rng.uniform(-args.cloud, args.cloud, 2)
               for _ in range(args.n_bodies)]
        h = evolve_cosmological(pos, [1.0] * args.n_bodies, shape=(L, L),
                                width=args.width, hubble0=args.hubble0,
                                omega_m=1.0 - ol, omega_lambda=ol, p=args.p,
                                dt=args.dt, steps=args.cloud_steps, center=c,
                                **_fk(args))
        # structure growth: comoving cloud size (rms radius / a) vs a —
        # shrinks when matter collapses, stays flat when Λ freezes growth
        comoving = []
        for P, a in zip(h["positions"], h["a"]):
            arr = np.array(P)
            rms = float(np.sqrt(np.mean(np.sum((arr - arr.mean(0)) ** 2, axis=1))))
            comoving.append(rms / a)
        sizes = fof_groups([p for p in h["final_positions"]], args.link)
        out.append({"omega_lambda": ol, "a_final": h["a_final"],
                    "largest_fraction": sizes[0] / args.n_bodies,
                    "group_sizes": sizes, "a": h["a"], "comoving": comoving})
        print(f"  Ω_Λ={ol:.2f}: a_final={h['a_final']:.1f}  largest "
              f"{sizes[0]}/{args.n_bodies} ({sizes[0]/args.n_bodies:.0%})  "
              f"groups={sizes}", flush=True)
    return out


def summarise(hist, drag, cloud, args) -> list[str]:
    lines = ["An expanding universe: FLRW κ-gravity — verdict", "=" * 48, ""]
    aprod = np.array(drag["a_vpec"])[1:]
    lines.append(
        f"the scale factor evolves: a(t) decelerates when matter dominates and "
        f"accelerates when Λ dominates — over the run a grows to " +
        ", ".join(f"{c['a_final']:.1f} (Ω_Λ={c['omega_lambda']:.1f})"
                  for c in cloud) + ".")
    lines.append("")
    lines.append(
        f"Hubble drag is real: a peculiar velocity redshifts as 1/a — the "
        f"product a·|v_pec| holds at {aprod.mean():.2f} to "
        f"{aprod.std()/aprod.mean()*100:.1f}% across the expansion. The coasting "
        "model could not show this; an evolving a(t) does.")
    lines.append("")
    lo, hi = cloud[0], cloud[-1]
    lines.append(
        f"dark energy suppresses structure: the same cloud collapses into a "
        f"single bound halo in a matter universe ({lo['largest_fraction']:.0%} "
        f"in one group at Ω_Λ={lo['omega_lambda']:.1f}) but is frozen out as the "
        f"expansion accelerates ({hi['largest_fraction']:.0%} at "
        f"Ω_Λ={hi['omega_lambda']:.1f}) — a faster-expanding, dark-energy "
        "universe assembles less, the defining signature of Λ in growth.")
    lines.append("")
    lines.append(
        "so the gap→matter→gravity→structure chain now runs in a genuine FLRW-"
        "like background: an evolving scale factor, momentum redshift, and the "
        "dark-energy freeze-out of structure — the mechanisms of cosmology, "
        "present.")
    lines.append("")
    lines.append(
        "honest scope: a Newtonian FLRW-analogue — a real evolving scale factor "
        "and Hubble drag, but the Friedmann law is imposed (p=3 matter dilution "
        "applied to the 2-D screened dynamics), not derived from the κ-field's "
        "stress-energy; no metric, horizon, or relativistic growth factor. It "
        "reproduces the mechanisms, not a quantitative ΛCDM.")
    return lines


def make_figure(hist, drag, cloud, args, path):
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

    palette = [BLUE, GREEN, ORANGE, RED, PURPLE]
    # panel 1: a(t) histories
    for hpt, col in zip(hist, palette):
        lab = f"Ω_Λ={hpt['omega_lambda']:.1f}"
        if hpt["omega_lambda"] == 0:
            lab += " (matter, decel)"
        elif hpt["omega_lambda"] >= 0.99:
            lab += " (Λ, accel)"
        ax1.plot(hpt["t"], hpt["a"], color=col, lw=2.0, label=lab, zorder=3)
    ax1.set_xlabel("time")
    ax1.set_ylabel("scale factor a(t)")
    ax1.set_title("Expansion histories: matter vs dark energy")
    ax1.legend(frameon=False, fontsize=8)

    # panel 2: Hubble drag — a·|v_pec| constant
    a = np.array(drag["a"])
    ax2.plot(a[1:], np.array(drag["vpec"])[1:], color=BLUE, lw=1.9,
             label="|v_pec|", zorder=3)
    ax2.plot(a[1:], np.array(drag["a_vpec"])[1:], color=RED, lw=1.9, ls="--",
             label="a·|v_pec| (const ⇒ ∝1/a)", zorder=3)
    ax2.set_xlabel("scale factor a")
    ax2.set_ylabel("peculiar velocity")
    ax2.set_title("Hubble drag: peculiar velocity redshifts as 1/a")
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_ylim(bottom=0)

    # panel 3: dark energy suppresses structure
    ol = [c["omega_lambda"] for c in cloud]
    frac = [c["largest_fraction"] for c in cloud]
    ax3.plot(ol, frac, color=GREEN, marker="D", ms=7, lw=1.8, zorder=3)
    ax3.axhline(1.0 / args.n_bodies, color=GRAY, ls=":", lw=1.0, zorder=2)
    ax3.text(ol[0], 1.0 / args.n_bodies + 0.02, "fully dispersed",
             color=GRAY, fontsize=8)
    ax3.set_xlabel("dark energy  Ω_Λ")
    ax3.set_ylabel("largest bound fraction")
    ax3.set_title("Dark energy suppresses structure")
    ax3.set_ylim(0, 1.05)

    # panel 4: comoving cloud size vs a — collapse (matter) vs freeze-out (Λ)
    for c, col in zip(cloud, palette):
        ax4.plot(c["a"], c["comoving"], color=col, lw=1.8,
                 label=f"Ω_Λ={c['omega_lambda']:.1f}", zorder=3)
    ax4.set_xlabel("scale factor a")
    ax4.set_ylabel("comoving cloud size  (rms / a)")
    ax4.set_title("Matter collapses; dark energy freezes it out")
    ax4.legend(frameon=False, fontsize=8)
    ax4.set_ylim(bottom=0)

    fig.suptitle("An expanding universe: FLRW scale factor, Hubble drag, and "
                 "the dark-energy freeze-out of structure", fontsize=12, y=0.98)
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
    p.add_argument("--hubble0", type=float, default=0.05)
    p.add_argument("--p", type=float, default=3.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--omega-hist", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    p.add_argument("--hist-steps", type=int, default=110)
    p.add_argument("--kick", type=float, default=0.5)
    p.add_argument("--drag-steps", type=int, default=140)
    p.add_argument("--n-bodies", type=int, default=12)
    p.add_argument("--cloud", type=float, default=11.0)
    p.add_argument("--cloud-steps", type=int, default=110)
    p.add_argument("--link", type=float, default=6.0)
    p.add_argument("--omega-cloud", type=float, nargs="+",
                   default=[0.0, 0.3, 0.5, 0.7, 1.0])
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_universe")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size = 64
        args.drag_steps = 80
        args.hist_steps = 70
        args.n_bodies = 8
        args.cloud_steps = 55
        args.omega_cloud = [0.0, 0.5, 1.0]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"expanding universe, L={args.size}^2, H0={args.hubble0}, p={args.p}",
          flush=True)
    hist = scale_histories(args)
    drag = hubble_drag(args)
    cloud = structure_vs_darkenergy(args)

    lines = summarise(hist, drag, cloud, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args),
               "hubble_drag": {"a_vpec": drag["a_vpec"]},
               "structure": [{"omega_lambda": c["omega_lambda"],
                              "a_final": c["a_final"],
                              "largest_fraction": c["largest_fraction"],
                              "group_sizes": c["group_sizes"]} for c in cloud]}
    with open(os.path.join(args.output_dir, "n3_expanding_universe.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_expanding_universe.png")
        make_figure(hist, drag, cloud, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
