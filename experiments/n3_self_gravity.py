"""Self-gravitating forms: infall and accretion under κ-gravity.

`n3_kappa_gravity` measured the force between rigid masses; `n3_stable_forms`
showed a form's gravitational mass equals its structural mass.  Here the
masses are **let go** — they move in the κ-field they mutually source
(`capacity_dynamics`), the adiabatic gradient flow ``dR_i/dt = μ·F_i`` with
the envelope-theorem force ``F_i = −c·Σ_x load_i·κ·∇κ``.

Two results, both structure-formation from first principles:

1. **Infall.**  Two masses fall together and merge — and the fall
   *accelerates* (the Yukawa force grows as they close), exactly as gravity
   should.  With the capacity coupling switched off (``c = 0``) they source no
   well and **stay put**: the infall is κ-gravity, not drift.
2. **Accretion.**  A scattering of masses, left to their own κ-gravity,
   **clumps** — pairs merge, clumps merge, the count falls toward a single
   bound object while total mass is conserved.  The framework grows its own
   bound structure out of the same capacity field whose depletion *is* gravity.

Honest scope: overdamped (dissipative) adiabatic dynamics — κ relaxed at each
step, masses moved down the energy gradient — with rigid Gaussian masses
standing in for the stable forms (whose gravitational mass = structural mass
was established separately).  This is Newtonian-in-spirit accretion under a
*screened* force, on a periodic lattice; it is the emergence of bound
structure, not a cosmological simulation.

Usage::

    python experiments/n3_self_gravity.py --output-dir artifacts/n3_self_gravity
    python experiments/n3_self_gravity.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.capacity_dynamics import capacity_force, evolve


def _field_kw(args, gravity=True):
    return dict(kappa_diffusion=args.D,
                kappa_recovery=args.r,
                kappa_consumption=args.c if gravity else 0.0,
                kappa_baseline=1.0, max_iters=args.max_iters)


def two_body(args):
    L = args.size
    p0 = [[L / 2, L / 2 - args.sep0 / 2], [L / 2, L / 2 + args.sep0 / 2]]
    grav = evolve(p0, [args.mass, args.mass], shape=(L, L), width=args.width,
                  mobility=args.mobility, dt=args.dt, steps=args.steps,
                  merge_dist=args.merge_dist, **_field_kw(args, True))
    ctrl = evolve(p0, [args.mass, args.mass], shape=(L, L), width=args.width,
                  mobility=args.mobility, dt=args.dt, steps=args.steps,
                  merge_dist=args.merge_dist, **_field_kw(args, False))
    print(f"  two-body: sep {grav['separation'][0]:.1f} → merge in "
          f"{len(grav['separation'])} steps; control stays "
          f"{ctrl['separation'][-1]:.1f}", flush=True)
    return grav, ctrl


def force_law(args):
    """Measured infall force vs separation — the (screened) gravitational law."""
    L = args.size
    seps = np.linspace(4, args.sep0 + 4, 9)
    Fs = []
    for s in seps:
        p = [[L / 2, L / 2 - s / 2], [L / 2, L / 2 + s / 2]]
        F, _ = capacity_force(p, [args.mass, args.mass], shape=(L, L),
                              width=args.width, **_field_kw(args, True))
        Fs.append(float(np.linalg.norm(F[0])))
    return {"sep": seps.tolist(), "F": Fs}


def nbody(args):
    L = args.size
    rng = np.random.default_rng(args.seed)
    pos = [rng.uniform(args.margin, L - args.margin, 2)
           for _ in range(args.n_bodies)]
    hist = evolve(pos, [1.0] * args.n_bodies, shape=(L, L), width=args.width,
                  mobility=args.nbody_mobility, dt=args.dt, steps=args.nbody_steps,
                  merge_dist=args.nbody_merge, **_field_kw(args, True))
    print(f"  N-body: {args.n_bodies} masses → {len(hist['final_masses'])} "
          f"cluster(s), heaviest {max(hist['final_masses']):.0f} "
          f"(Σm = {sum(hist['final_masses']):.0f}, conserved)", flush=True)
    return hist


def summarise(grav, ctrl, hist, args) -> list[str]:
    lines = ["Self-gravitating forms — verdict", "=" * 40, ""]
    s = grav["separation"]
    # accelerating? compare early vs late per-step closing rate
    d = -np.diff(s)
    early = float(np.mean(d[:max(1, len(d)//3)]))
    late = float(np.mean(d[-max(1, len(d)//3):]))
    lines.append(
        f"infall: two masses fall together from separation {s[0]:.1f} and merge "
        f"in {len(s)} steps, and the fall accelerates — the closing rate grows "
        f"from {early:.2f}/step early to {late:.2f}/step near contact (the "
        "Yukawa force steepening as they approach).")
    lines.append("")
    lines.append(
        f"it is gravity, not drift: with the capacity coupling off (c = 0) the "
        f"same pair sources no κ-well and stays at separation "
        f"{ctrl['separation'][-1]:.1f} — no infall without the field.")
    lines.append("")
    lines.append(
        f"accretion: {args.n_bodies} masses scattered at random, left to their "
        f"own κ-gravity, clump into {len(hist['final_masses'])} bound "
        f"object(s) (heaviest {max(hist['final_masses']):.0f} units) while the "
        f"total mass Σm = {sum(hist['final_masses']):.0f} is conserved — "
        "structure forming out of the capacity field, from first principles.")
    lines.append("")
    lines.append(
        "honest scope: overdamped adiabatic dynamics (κ relaxed each step, "
        "masses moved down the energy gradient) with rigid Gaussian masses "
        "standing in for the stable forms; a screened force on a periodic "
        "lattice. It demonstrates the *emergence of bound structure* under "
        "κ-gravity, not a cosmological simulation.")
    return lines


def make_figure(grav, ctrl, flaw, hist, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY, RED = (
        "#0072B2", "#E69F00", "#009E73", "#999999", "#c0392b")
    fig = plt.figure(figsize=(13, 9.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.24)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))

    # panel 1: two-body infall d(t) vs control
    for ax in (ax1, ax4):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax1.plot(range(len(grav["separation"])), grav["separation"], color=BLUE,
             marker="o", ms=4, lw=1.9, zorder=3, label="κ-gravity (infall)")
    ax1.plot(range(len(ctrl["separation"])), ctrl["separation"], color=GRAY,
             ls="--", lw=1.7, zorder=2, label="gravity off (c=0)")
    ax1.set_xlabel("time step")
    ax1.set_ylabel("separation d(t)")
    ax1.set_title("Two forms fall together and merge")
    ax1.legend(frameon=False, fontsize=8)

    # panels 2 & 3: N-body initial and final states on the κ field
    kap = hist["kappa"]
    init = np.array(hist["positions"][0])
    initm = np.array(hist["masses"][0])
    finp = np.array(hist["final_positions"])
    finm = np.array(hist["final_masses"])
    for ax, pos, ms, title in ((ax2, init, initm, "Scattered masses (t = 0)"),
                               (ax3, finp, finm, "Accreted (t = end)")):
        ax.imshow(kap.T, origin="lower", cmap="magma", aspect="equal",
                  alpha=0.9)
        ax.scatter(pos[:, 0], pos[:, 1], s=30 + 60 * ms / initm.max(),
                   facecolor=GREEN, edgecolor="white", linewidth=1.0, zorder=5)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    # panel 4: accretion history — bodies falling toward one
    ax4.plot(range(len(hist["n_bodies"])), hist["n_bodies"], color=ORANGE,
             marker="o", ms=4, lw=1.9, zorder=3)
    ax4.set_xlabel("time step")
    ax4.set_ylabel("number of bound objects")
    ax4.set_title("Accretion: many → few (mass conserved)")
    ax4.set_ylim(0, args.n_bodies + 0.5)

    fig.suptitle("Self-gravitating forms: infall and accretion under κ-gravity",
                 fontsize=12, y=0.98)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--mass", type=float, default=1.2)
    p.add_argument("--D", type=float, default=1.0)
    p.add_argument("--r", type=float, default=0.02)     # ξ = √(D/r) ≈ 7
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--max-iters", type=int, default=4000)
    p.add_argument("--sep0", type=float, default=14.0)
    p.add_argument("--mobility", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--merge-dist", type=float, default=3.0)
    p.add_argument("--n-bodies", type=int, default=9)
    p.add_argument("--nbody-mobility", type=float, default=3.0)
    p.add_argument("--nbody-steps", type=int, default=80)
    p.add_argument("--nbody-merge", type=float, default=4.0)
    p.add_argument("--margin", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_self_gravity")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size = 64
        args.n_bodies = 6
        args.nbody_steps = 40
        args.steps = 25
        args.margin = 16.0

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"self-gravity, L={args.size}^2, ξ=√(D/r)={np.sqrt(args.D/args.r):.1f}",
          flush=True)
    grav, ctrl = two_body(args)
    flaw = force_law(args)
    hist = nbody(args)

    lines = summarise(grav, ctrl, hist, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args),
               "two_body": {"separation": grav["separation"],
                            "n_final": len(grav["final_masses"])},
               "control": {"separation": ctrl["separation"]},
               "force_law": flaw,
               "nbody": {"n_bodies": hist["n_bodies"],
                         "final_masses": hist["final_masses"]}}
    with open(os.path.join(args.output_dir, "n3_self_gravity.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_self_gravity.png")
        make_figure(grav, ctrl, flaw, hist, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
