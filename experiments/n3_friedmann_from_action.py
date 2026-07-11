"""The variational closure: the Friedmann relations from an action, not by hand.

The relativistic closure (`n3_stress_energy_closure.py`) built a conserved
stress-energy tensor and made the expansion its consequence — but it still *put
in by hand* the Einstein/Friedmann relations `H² = ρ`, `ä/a = −½(ρ + 3p)` that
turn `T^μ_ν` into `a(t)`.  This derives those relations from a **minisuperspace
variational principle**, so `H² = ρ` becomes a first integral of the dynamics
rather than a postulate.

For a flat FLRW background with lapse `N` and scale factor `a`
(units `8πG/3 = 1`, comoving volume dropped),

    S = ∫ dt [ −a ȧ²/N − N a³ ρ(a) ] ,

with the gravitational kinetic term `−a ȧ²` (the dynamical geometry's own
energy — the capacity field's free energy setting the global scale) and `ρ(a)`
the content of the stress-energy tensor.  Two variations give the two laws:

  • the lapse / Hamiltonian constraint `∂S/∂N = 0`  ⇒  `H² = ρ`   (Friedmann),
  • the `a` Euler–Lagrange equation (with conservation)  ⇒  `ä/a = −½(ρ + 3p)`.

Four results:

A. **One history, three derivations.**  Evolving the `a`-Euler–Lagrange equation
   (the acceleration equation) with conservation reproduces the very same `a(t)`
   as the imposed-`a^{−3}` cosmology and the stress-energy route — the action
   derivation is consistent with everything before it.
B. **Friedmann is a first integral, not an input.**  Along that acceleration
   equation the constraint `C = H² − ρ` stays at ~1e−10: `H² = ρ` is *preserved*
   by the dynamics (it is never substituted), so it is a consequence, not a
   postulate.  Analytically `Ċ = −2 H C`.
C. **…and an attractor.**  Start with a *wrong* expansion rate (constraint
   violated by ±30–60%) and `C(t) → 0`: the universe relaxes onto the Friedmann
   trajectory, because `Ċ = −2 H C` with `H > 0`.
D. **The physics is intact.**  The deceleration parameter `q = −ä a/ȧ²` from the
   variational dynamics crosses zero at the same `a_acc`: the decel→accel history
   survives the derivation.

Honest scope: a *minisuperspace* variational principle — homogeneous flat FLRW
with the single degree of freedom `a(t)`.  The Friedmann equation is genuinely
the Hamiltonian constraint and the acceleration equation the Euler–Lagrange
equation (and `H² = ρ` a preserved, attracting first integral), but the
gravitational kinetic term `−a ȧ²` is the reduced Einstein–Hilbert form *posited*
(read as the geometry's free energy), not derived from the κ-action microscopically,
and there are no inhomogeneous field equations or a solved metric.  It removes
the last hand-input at the Friedmann level; it is not a derivation of general
relativity from the κ-action.

Usage::

    python experiments/n3_friedmann_from_action.py --output-dir artifacts/n3_action
    python experiments/n3_friedmann_from_action.py --quick
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
    capacity_vacuum_density,
    integrate_friedmann_action,
    integrate_scale_factor,
    integrate_stress_energy,
)


def _q_zero_crossing(a, q):
    a = np.asarray(a); q = np.asarray(q)
    below = np.where(q < 0)[0]
    if len(below) and below[0] > 0:
        i = below[0]
        frac = q[i - 1] / (q[i - 1] - q[i])
        return float(a[i - 1] + frac * (a[i] - a[i - 1]))
    return float("nan")


def three_routes(args, rho_lambda):
    """A/D: the action-derived a(t)/q(a) vs the imposed and stress-energy routes."""
    dens, eos = [args.rho_m0, rho_lambda], [0.0, -1.0]
    act = integrate_friedmann_action(dens, eos, dt=args.dt, steps=args.steps)
    ten = integrate_stress_energy(dens, eos, dt=args.dt, steps=args.steps)
    imp = integrate_scale_factor(args.rho_m0, rho_lambda, dim=3, dt=args.dt,
                                 steps=args.steps)
    a_act = np.array(act["a"]); m = a_act <= args.a_show
    resid_ten = float(np.max(np.abs(a_act[m] - np.array(ten["a"])[m])
                             / np.maximum(np.array(ten["a"])[m], 1e-30)))
    resid_imp = float(np.max(np.abs(a_act[m] - np.array(imp["a"])[m])
                             / np.maximum(np.array(imp["a"])[m], 1e-30)))
    a_acc = _q_zero_crossing(act["a"], act["q"])
    print(f"  action a(t) vs stress-energy route: max rel |Δa|/a = {resid_ten:.1e}",
          flush=True)
    print(f"  action a(t) vs imposed-law route:   max rel |Δa|/a = {resid_imp:.1e}",
          flush=True)
    print(f"  action q(a): {act['q'][0]:+.3f} → {act['q'][-1]:+.3f}, "
          f"acceleration at a_acc={a_acc:.2f}", flush=True)
    return {"action": act, "tensor": ten, "imposed": imp,
            "resid_tensor": resid_ten, "resid_imposed": resid_imp,
            "a_acc": a_acc}


def constraint_preservation(args, rho_lambda):
    """B: C = H² − ρ preserved along the acceleration equation (first integral)."""
    h = integrate_friedmann_action([args.rho_m0, rho_lambda], [0.0, -1.0],
                                   dt=args.dt, steps=args.steps)
    a = np.array(h["a"]); C = np.array(h["constraint"]); m = a <= args.a_show
    worst = float(np.max(np.abs(C[m])))
    print(f"  Friedmann constraint |H²−ρ| along the a-EOM: max (a≤{args.a_show:g}) "
          f"= {worst:.1e} — preserved, never imposed", flush=True)
    return {"a": h["a"], "constraint": h["constraint"], "worst": worst}


def constraint_attractor(args, rho_lambda):
    """C: constraint-violating initial rates relax onto H² = ρ."""
    out = []
    rho0 = args.rho_m0 + rho_lambda
    for f in args.violate:
        adot0 = f * np.sqrt(rho0)
        h = integrate_friedmann_action([args.rho_m0, rho_lambda], [0.0, -1.0],
                                       dt=args.dt, steps=args.steps, adot0=adot0)
        a = np.array(h["a"]); C = np.array(h["constraint"])
        out.append({"factor": f, "a": h["a"], "constraint": h["constraint"],
                    "C0": float(C[0]), "C_end": float(C[-1])})
        print(f"  ȧ₀={f:.1f}·√ρ₀: C₀={C[0]:+.4f} → C_end={C[-1]:+.1e} "
              f"({'relaxes to' if abs(C[-1]) < abs(C[0]) else 'stays'} Friedmann)",
              flush=True)
    return out


def summarise(routes, cons, attr, args) -> list[str]:
    lines = ["The variational closure: the Friedmann relations from an action "
             "— verdict", "=" * 66, ""]
    lines.append(
        "the setup: for flat FLRW the minisuperspace action S = ∫dt[−a ȧ²/N − "
        "N a³ρ(a)] has the gravitational kinetic term −a ȧ² (the geometry's own "
        "free energy) and the stress-energy content ρ(a). Its lapse variation "
        "∂S/∂N=0 gives the Friedmann constraint H²=ρ; its a-Euler–Lagrange "
        "equation (with conservation) gives ä/a=−½(ρ+3p).")
    lines.append("")
    lines.append(
        f"A. one history, three derivations: evolving the a-Euler–Lagrange "
        f"equation reproduces the same a(t) as the stress-energy route "
        f"(max rel |Δa|/a={routes['resid_tensor']:.0e}) and the original "
        f"imposed-a^(−3) cosmology ({routes['resid_imposed']:.0e}) — the action "
        "derivation agrees with everything before it.")
    lines.append("")
    lines.append(
        f"B. Friedmann is a first integral, not an input: along that "
        f"acceleration equation the constraint C=H²−ρ stays below "
        f"{cons['worst']:.0e} (a≤{args.a_show:g}) — H²=ρ is preserved by the "
        "dynamics though it is never substituted. Analytically Ċ=−2HC, so C=0 is "
        "conserved: the Friedmann equation is a consequence, not a postulate.")
    lines.append("")
    relaxed = ", ".join(f"C₀={o['C0']:+.2f}→{o['C_end']:+.0e}" for o in attr)
    lines.append(
        f"C. …and an attractor: starting from a wrong expansion rate (constraint "
        f"violated) the universe relaxes onto H²=ρ — {relaxed} — because Ċ=−2HC "
        "with H>0. Even the initial condition need not satisfy Friedmann; the "
        "dynamics enforces it.")
    lines.append("")
    lines.append(
        f"D. the physics is intact: the deceleration parameter from the "
        f"variational dynamics still crosses zero (a_acc={routes['a_acc']:.2f}), "
        "so the decel→accel history survives the derivation.")
    lines.append("")
    lines.append(
        "so the last hand-input at the Friedmann level is removed: H²=ρ and "
        "ä/a=−½(ρ+3p) are the Hamiltonian constraint and Euler–Lagrange equation "
        "of one action, and H²=ρ is a preserved, attracting first integral. The "
        "geometry (the scale factor's dynamics) and its source (the field's "
        "stress-energy) now come from a single variational principle.")
    lines.append("")
    lines.append(
        "honest scope: a minisuperspace variational principle — homogeneous flat "
        "FLRW with the single degree of freedom a(t). The Friedmann equation is "
        "genuinely the Hamiltonian constraint and H²=ρ a preserved, attracting "
        "first integral, but the gravitational kinetic term −a ȧ² is the reduced "
        "Einstein–Hilbert form posited (read as the geometry's free energy), not "
        "derived from the κ-action microscopically, and there are no inhomogeneous "
        "field equations or a solved metric. It removes the last hand-input at the "
        "Friedmann level; it is not a derivation of general relativity.")
    return lines


def make_figure(routes, cons, attr, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY, RED, PURPLE = (
        "#0072B2", "#E69F00", "#009E73", "#999999", "#c0392b", "#8551A6")
    fig = plt.figure(figsize=(13, 9.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.27)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))
    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    act = routes["action"]; ten = routes["tensor"]; imp = routes["imposed"]
    ta = np.array(act["t"]); aa = np.array(act["a"]); m = aa <= args.a_show

    # panel A: one history, three derivations
    ax1.plot(np.array(imp["t"])[m], np.array(imp["a"])[m], color=GRAY, lw=5.0,
             alpha=0.45, zorder=2, label="imposed-a^{−3} cosmology")
    ax1.plot(np.array(ten["t"])[m], np.array(ten["a"])[m], color=ORANGE, lw=2.8,
             zorder=3, label="stress-energy (∇·T=0)")
    ax1.plot(ta[m], aa[m], color=BLUE, lw=1.6, zorder=4,
             label="variational (a-Euler–Lagrange)")
    ax1.set_xlabel("time")
    ax1.set_ylabel("scale factor a(t)")
    ax1.set_title(f"A. One history, three derivations "
                  f"(Δa/a ≤ {max(routes['resid_tensor'], routes['resid_imposed']):.0e})")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    # panel B: the Friedmann constraint preserved (first integral)
    ac = np.array(cons["a"]); C = np.abs(np.array(cons["constraint"]))
    mc = ac <= args.a_show
    ax2.semilogy(ac[mc], np.maximum(C[mc], 1e-16), color=BLUE, lw=2.0, zorder=3)
    ax2.set_xlabel("scale factor a")
    ax2.set_ylabel("|C| = |H² − ρ|  (constraint)")
    ax2.set_title("B. Friedmann is a first integral (never imposed)")
    ax2.text(0.97, 0.9, f"max |C| ≈ {cons['worst']:.0e}", transform=ax2.transAxes,
             ha="right", fontsize=9, color=GRAY)

    # panel C: the constraint is an attractor
    cols = [RED, ORANGE, GREEN, PURPLE]
    for o, col in zip(attr, cols):
        ac = np.array(o["a"]); C = np.abs(np.array(o["constraint"]))
        mc = ac <= args.a_show
        ax3.semilogy(ac[mc], np.maximum(C[mc], 1e-16), color=col, lw=1.9,
                     zorder=3, label=f"ȧ₀ = {o['factor']:.1f}·√ρ₀")
    ax3.set_xlabel("scale factor a")
    ax3.set_ylabel("|C| = |H² − ρ|")
    ax3.set_title("C. …and an attractor: wrong ȧ₀ relaxes onto H²=ρ")
    ax3.legend(frameon=False, fontsize=8, loc="upper right")

    # panel D: the physics is intact — q(a) from the variational dynamics
    q = np.array(act["q"])
    ax4.plot(aa[m], q[m], color=BLUE, lw=2.2, zorder=3, label="q = −ä a/ȧ²")
    ax4.axhline(0.0, color=RED, ls="--", lw=1.1, zorder=2)
    if np.isfinite(routes["a_acc"]):
        ax4.axvline(routes["a_acc"], color=GRAY, ls=":", lw=1.1, zorder=2)
        ax4.text(routes["a_acc"], q[m].max() * 0.6,
                 f" a_acc={routes['a_acc']:.2f}", color=GRAY, fontsize=8)
    ax4.text(0.97, 0.05, "accelerating ↓", transform=ax4.transAxes, ha="right",
             fontsize=8, color=GRAY)
    ax4.set_xlabel("scale factor a")
    ax4.set_ylabel("deceleration parameter q")
    ax4.set_title("D. The physics is intact: q crosses zero")
    ax4.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("The variational closure: the Friedmann equation as a first "
                 "integral of an action, not a postulate", fontsize=12, y=0.98)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rho-m0", type=float, default=0.05)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--kappa0", type=float, default=1.0)
    p.add_argument("--vac-coeff", type=float, default=0.5)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--a-show", type=float, default=6.0)
    p.add_argument("--violate", type=float, nargs="+", default=[0.6, 1.3, 1.6])
    p.add_argument("--output-dir", type=str, default="artifacts/n3_action")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.dt = 0.2
        args.steps = 350
        args.violate = [0.6, 1.4]

    os.makedirs(args.output_dir, exist_ok=True)
    rho_lambda = capacity_vacuum_density(args.r, args.kappa0, args.vac_coeff)
    print(f"Friedmann from an action: ρ_m0={args.rho_m0}, ρ_Λ={rho_lambda:.4f}, "
          f"flat FLRW minisuperspace, units 8πG/3=1", flush=True)
    print("A/D — one history from three derivations:")
    routes = three_routes(args, rho_lambda)
    print("B — the Friedmann constraint is a preserved first integral:")
    cons = constraint_preservation(args, rho_lambda)
    print("C — the constraint is an attractor:")
    attr = constraint_attractor(args, rho_lambda)

    lines = summarise(routes, cons, attr, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args), "rho_lambda": rho_lambda,
               "resid_tensor": routes["resid_tensor"],
               "resid_imposed": routes["resid_imposed"], "a_acc": routes["a_acc"],
               "constraint_worst": cons["worst"],
               "attractor": [{"factor": o["factor"], "C0": o["C0"],
                              "C_end": o["C_end"]} for o in attr]}
    with open(os.path.join(args.output_dir, "n3_friedmann_from_action.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_friedmann_from_action.png")
        make_figure(routes, cons, attr, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
