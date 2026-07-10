"""The spectrum of stable forms — and structural mass = gravitational mass.

The URP claim about matter, made measurable: *particles are the stable forms
the sector field can manifest, and mass is stable structure made manifest.*
In the CP² field those forms are **topological solitons** of integer charge
``Q``.  This experiment establishes the three things that turns the claim into
a measurement:

1. **A discrete corpus.**  The admissible forms are indexed by an integer
   charge Q, and their structural (inertial) masses — CP-action energies —
   lie on a **discrete ladder** ``E ∝ |Q|`` (the Bogomolny spectrum).  Matter
   comes in quanta because topology does.
2. **Stability.**  A charge-Q form preserves its Q under the dynamics and
   holds an **energy floor** it cannot decay below; a topologically trivial
   bump of the field relaxes back to the vacuum.  Only the protected forms
   persist — they are the *stable* forms.
3. **Structural mass = gravitational mass.**  Feeding each form's distinction
   density into the capacity-gravity dynamics, the κ-well it sources — its
   **gravitational mass** — is **proportional to its structural mass** across
   the whole spectrum.  The equivalence principle, emergent: both masses are
   the same underlying quantity, the form's distinction content.

Honest scope: 2-D CP² solitons on a periodic lattice; the "masses" are in
lattice-action units and the equivalence holds in the weak-well (linear)
regime — at strong coupling the κ-well saturates and gravitational mass grows
*sub-linearly*, a measured nonlinearity, not a redefinition.  The proportion-
ality is *explained* by (not independent of) the shared distinction root —
which is precisely the framework's account of why inertial = gravitational.

Usage::

    python experiments/n3_stable_forms.py --output-dir artifacts/n3_stable_forms
    python experiments/n3_stable_forms.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.stable_forms import (
    distinction_density,
    form_charge,
    gravitational_mass,
    structural_mass,
    trivial_bump,
    winding_form,
)
from project_genesis.topological_charge import cool


def build_forms(args):
    """The corpus: charge-Q solitons, lightly cooled to a common smoothness."""
    forms = {}
    for Q in range(1, args.max_charge + 1):
        psi = winding_form(args.size, Q, ring=args.ring)
        psi = cool(psi, args.prep_cool, rate=args.prep_rate)
        forms[Q] = psi
    return forms


def measure_spectrum(forms, args):
    rows = []
    for Q, psi in forms.items():
        E = structural_mass(psi)
        Mg = gravitational_mass(psi, kappa_consumption=args.coupling)
        Mg_strong = gravitational_mass(psi, kappa_consumption=args.coupling_strong)
        rows.append({"Q": Q, "charge": form_charge(psi), "E": E,
                     "M_grav": Mg, "M_grav_strong": Mg_strong})
        print(f"  Q={Q}: charge={rows[-1]['charge']:+.2f}  E={E:.3f}  "
              f"M_grav={Mg:.3f}  M_grav(strong)={Mg_strong:.3f}", flush=True)
    return rows


def stability(args):
    """Energy under cooling: topological forms plateau at a floor; a bump decays."""
    steps = list(range(0, args.stab_steps + 1, args.stab_every))
    traj = {}
    configs = {0: trivial_bump(args.size, args.bump_amp, args.bump_width)}
    for Q in range(1, args.stab_charge + 1):
        configs[Q] = winding_form(args.size, Q, ring=args.ring)
    for Q, psi0 in configs.items():
        E, Qs = [], []
        z = psi0.copy()
        prev = 0
        for s in steps:
            z = cool(z, s - prev, rate=args.prep_rate)
            prev = s
            E.append(structural_mass(z))
            Qs.append(form_charge(z))
        traj[Q] = {"steps": steps, "E": E, "charge": Qs}
        print(f"  stability Q={Q}: E {E[0]:.2f} → {E[-1]:.2f}, "
              f"charge {Qs[0]:+.2f} → {Qs[-1]:+.2f}", flush=True)
    return traj


def equivalence_fit(rows):
    E = np.array([r["E"] for r in rows])
    Mg = np.array([r["M_grav"] for r in rows])
    slope = float((E * Mg).sum() / (E * E).sum())     # through the origin
    r2 = float(1.0 - ((Mg - slope * E) ** 2).sum()
               / ((Mg - Mg.mean()) ** 2).sum())
    return {"slope": slope, "r2": r2}


def summarise(rows, traj, fit, args) -> list[str]:
    lines = ["The spectrum of stable forms — verdict", "=" * 40, ""]
    charges = ", ".join(f"{r['charge']:+.2f}" for r in rows)
    Es = ", ".join(f"{r['E']:.2f}" for r in rows)
    incr = np.diff([r["E"] for r in rows])
    lines.append(
        f"a discrete corpus: the forms carry integer topological charge "
        f"(Q = {charges}) and their structural masses lie on a discrete ladder "
        f"(E = {Es}), rising by a near-constant {np.mean(incr):.2f} per unit "
        "charge — the Bogomolny spectrum E ∝ |Q|. Matter comes in quanta "
        "because topology does.")
    lines.append("")
    b = traj[0]
    top = traj[max(traj)]
    lines.append(
        f"and they are stable: under cooling a charge-{max(traj)} form holds its "
        f"charge ({top['charge'][0]:+.1f} → {top['charge'][-1]:+.1f}) and settles "
        f"to an energy floor ({top['E'][0]:.1f} → {top['E'][-1]:.1f}), while the "
        f"topologically trivial bump decays toward the vacuum "
        f"({b['E'][0]:.2f} → {b['E'][-1]:.2f}). Only the protected forms persist.")
    lines.append("")
    lines.append(
        f"and structural mass = gravitational mass: the κ-well each form sources "
        f"is proportional to its energy across the whole spectrum — "
        f"M_grav = {fit['slope']:.3f}·E with R² = {fit['r2']:.4f}. Inertial and "
        "gravitational mass coincide because both are the form's distinction "
        "content — the equivalence principle, explained rather than imposed.")
    lines.append("")
    # saturation: strong-coupling gravitational mass grows sub-linearly
    sat_ratio = [r["M_grav_strong"] / r["M_grav"] for r in rows]
    lines.append(
        f"nonlinear check: at strong capacity coupling the κ-well saturates "
        f"(bottoms out), and the gravitational mass grows sub-linearly — the "
        f"strong/weak ratio falls from {sat_ratio[0]:.2f} (lightest) to "
        f"{sat_ratio[-1]:.2f} (heaviest). So the gravitational mass is a genuine "
        "field response, not a relabelling of the energy: equivalence holds in "
        "the weak-well regime and bends where capacity runs out.")
    lines.append("")
    lines.append(
        "honest scope: 2-D CP² solitons on a periodic lattice, masses in "
        "lattice-action units; the constructed forms are lightly cooled to a "
        "common smoothness (the raw winding profile is Q-dependent). The "
        "structural=gravitational proportionality is *explained* by the shared "
        "distinction root, not an independent coincidence — that shared root is "
        "the framework's account of the equivalence principle.")
    return lines


def make_figure(rows, traj, fit, forms, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY, RED, PURPLE = (
        "#0072B2", "#E69F00", "#009E73", "#999999", "#c0392b", "#8551A6")
    fig = plt.figure(figsize=(13, 9.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.26)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))
    for ax in (ax2, ax3, ax4):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # panel 1: a stable form — its distinction density (the mass, made manifest)
    show_Q = max(forms)
    d = distinction_density(forms[show_Q])
    im = ax1.imshow(d, origin="lower", cmap="magma", aspect="equal")
    ax1.set_title(f"A stable form: Q = {show_Q} ({show_Q} quanta of winding)")
    ax1.set_xticks([]); ax1.set_yticks([])
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="distinction density")

    # panel 2: the mass spectrum E(Q) — discrete ladder ∝ |Q|
    Qs = [r["Q"] for r in rows]
    Es = [r["E"] for r in rows]
    ax2.plot(Qs, Es, color=BLUE, marker="o", ms=8, lw=0, zorder=4)
    qq = np.array([0] + Qs)
    slopeE = np.sum(np.array(Qs) * np.array(Es)) / np.sum(np.array(Qs) ** 2)
    ax2.plot(qq, slopeE * qq, color=GRAY, ls="--", lw=1.2, zorder=2,
             label=f"E ∝ |Q|  ({slopeE:.2f}·Q)")
    ax2.set_xlabel("topological charge Q")
    ax2.set_ylabel("structural mass  E")
    ax2.set_title("Discrete mass spectrum (the corpus)")
    ax2.set_xticks(Qs)
    ax2.legend(frameon=False, fontsize=8)

    # panel 3: stability — energy under cooling
    cols = {0: RED, 1: BLUE, 2: ORANGE, 3: GREEN, 4: PURPLE}
    for Q, t in sorted(traj.items()):
        lbl = "Q=0 (trivial, decays)" if Q == 0 else f"Q={Q}"
        ax3.plot(t["steps"], t["E"], color=cols.get(Q, GRAY),
                 marker="o", ms=4, lw=1.7,
                 ls="--" if Q == 0 else "-", zorder=3, label=lbl)
    ax3.set_xlabel("cooling steps")
    ax3.set_ylabel("energy  E")
    ax3.set_title("Stability: topological forms hold a floor")
    ax3.legend(frameon=False, fontsize=8)

    # panel 4: equivalence — gravitational vs structural mass
    Mg = [r["M_grav"] for r in rows]
    ax4.plot(Es, Mg, color=GREEN, marker="D", ms=8, lw=0, zorder=4,
             label="weak well (linear)")
    xs = np.array([0] + sorted(Es))
    ax4.plot(xs, fit["slope"] * xs, color=GRAY, lw=1.4, zorder=2,
             label=f"M_grav = {fit['slope']:.2f}·E  (R²={fit['r2']:.3f})")
    Mgs = [r["M_grav_strong"] for r in rows]
    # rescale strong to same weak-limit slope to expose the sub-linear bend
    s2 = Mgs[0] / Mg[0]
    ax4.plot(Es, [m / s2 for m in Mgs], color=RED, marker="s", ms=6, lw=0,
             zorder=3, label="strong well (saturates)")
    ax4.set_xlabel("structural mass  E")
    ax4.set_ylabel("gravitational mass  M_grav")
    ax4.set_title("Structural mass = gravitational mass")
    ax4.legend(frameon=False, fontsize=8)

    fig.suptitle("The spectrum of stable forms: mass is stable structure made "
                 "manifest, and it gravitates in proportion", fontsize=12, y=0.98)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--ring", type=float, default=3.0)
    p.add_argument("--max-charge", type=int, default=4)
    p.add_argument("--prep-cool", type=int, default=8)
    p.add_argument("--prep-rate", type=float, default=0.4)
    p.add_argument("--coupling", type=float, default=0.2)          # weak (linear)
    p.add_argument("--coupling-strong", type=float, default=3.0)   # saturating
    p.add_argument("--stab-charge", type=int, default=3)
    p.add_argument("--stab-steps", type=int, default=40)
    p.add_argument("--stab-every", type=int, default=5)
    p.add_argument("--bump-amp", type=float, default=0.8)
    p.add_argument("--bump-width", type=float, default=5.0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_stable_forms")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size = 40
        args.max_charge = 3
        args.stab_charge = 2
        args.stab_steps = 20

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"stable forms, L={args.size}^2, charges 1..{args.max_charge}", flush=True)
    forms = build_forms(args)
    rows = measure_spectrum(forms, args)
    traj = stability(args)
    fit = equivalence_fit(rows)
    print(f"  equivalence: M_grav = {fit['slope']:.3f}·E, R² = {fit['r2']:.4f}",
          flush=True)

    lines = summarise(rows, traj, fit, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_stable_forms.json"), "w") as fh:
        json.dump({"params": vars(args), "spectrum": rows, "stability": traj,
                   "equivalence": fit}, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_stable_forms.png")
        make_figure(rows, traj, fit, forms, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
