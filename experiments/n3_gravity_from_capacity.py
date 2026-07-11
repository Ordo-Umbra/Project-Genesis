"""Gravity from the capacity field: the gravitational action as capacity free energy.

The variational closure (`n3_friedmann_from_action.py`) derived the Friedmann
relations from a minisuperspace action — but it still *posited* the gravitational
kinetic term `−a ȧ²` as the reduced Einstein–Hilbert form.  This shows that term
**is** the capacity field's own kinetic free energy, closing the last posit: not
just the cosmology, but the gravitational action itself comes from the field.

Identify the scale factor with the exponential of the homogeneous capacity scalar
— the zero-mode `κ_s` whose global value sets the overall integration scale —
`a = e^{κ_s}` (so `κ_s = ln a`, `H = κ̇_s`).  The capacity scalar's kinetic free
energy on the FLRW volume measure `a³` is `a³ κ̇_s² = a ȧ²`, so

    −a ȧ²  =  −a³ κ̇_s²

— the gravitational kinetic term is *exactly* the capacity scalar's kinetic free
energy (with the wrong-sign conformal mode of gravity).  Rewriting the action in
`κ_s` then recasts the whole cosmology in the language of the field:

A. **The gravitational term is capacity free energy.**  Along the cosmic history
   the posited `−a ȧ²` and the capacity scalar's kinetic free energy `−a³ κ̇_s²`
   coincide identically — the geometry's kinetic energy *is* the field's.
B. **Friedmann is an energy balance `κ̇_s² = ρ`.**  The lapse constraint of the
   capacity action says the mean capacity's kinetic free-energy density equals
   the content density.  Expansion is the capacity field rolling; Friedmann
   balances its kinetic free energy against matter + vacuum.
C. **Expansion is the capacity scalar rolling.**  Its field equation
   `κ̈_s = −(3/2)(κ̇_s² + p)` drives the roll: `κ̈_s → 0` in the vacuum limit
   (constant roll → de Sitter) and `κ̈_s < 0` under dust.  Integrating it (the
   Friedmann relation never imposed) the balance `κ̇_s² = ρ` holds to ~1e−9 and
   is an attractor.
D. **It reproduces the whole history.**  The scale factor `a = e^{κ_s}` from the
   capacity-scalar dynamics coincides with the stress-energy and imposed-law
   cosmologies — gravity's expansion re-derived as the capacity field rolling.

Honest scope: the identification `a = e^{κ_s}` (the scale factor as the exp of the
homogeneous capacity scalar / e-folding number) is a *reading* within the URP
framework, and the kinetic free energy carries the wrong-sign conformal mode of
gravity, taken as given.  It is still minisuperspace (homogeneous, one degree of
freedom), with no inhomogeneous field equations or a solved metric.  What it
establishes: the gravitational action's kinetic term is not an independent posit
but the capacity field's kinetic free energy — the last hand-input at the
Friedmann level traced to the field.  It is not a derivation of general relativity.

Usage::

    python experiments/n3_gravity_from_capacity.py --output-dir artifacts/n3_capgrav
    python experiments/n3_gravity_from_capacity.py --quick
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
    capacity_kinetic_energy,
    capacity_vacuum_density,
    integrate_capacity_scale,
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


def capacity_history(args, rho_lambda):
    """A–D: evolve the cosmology as the capacity scalar κ_s = ln a rolling."""
    h = integrate_capacity_scale([args.rho_m0, rho_lambda], [0.0, -1.0],
                                 dt=args.dt, steps=args.steps)
    a = np.array(h["a"]); m = a <= args.a_show
    # the gravitational-term identity: −a ȧ² vs −a³ κ̇_s²
    kdot = np.array(h["kappa_dot"])
    adot = a * kdot                                    # ȧ = a κ̇_s
    grav_posited = -a * adot ** 2
    grav_capacity = -np.array([capacity_kinetic_energy(ai, adi)
                               for ai, adi in zip(a, adot)])
    identity_err = float(np.max(np.abs(grav_posited[m] - grav_capacity[m])))
    C = np.array(h["constraint"])
    balance = float(np.max(np.abs(C[m])))
    a_acc = _q_zero_crossing(a, h["q"])
    print(f"  gravitational term −a ȧ² vs capacity free energy −a³ κ̇_s²: "
          f"max|Δ|={identity_err:.1e}", flush=True)
    print(f"  Friedmann as energy balance |κ̇_s² − ρ|: max (a≤{args.a_show:g}) "
          f"= {balance:.1e}", flush=True)
    print(f"  capacity scalar rolls: κ̈_s {h['kappa_ddot'][0]:+.3f} → "
          f"{h['kappa_ddot'][-1]:+.3f} (dust → vacuum/de Sitter), "
          f"acceleration at a_acc={a_acc:.2f}", flush=True)
    return {"hist": h, "grav_posited": grav_posited.tolist(),
            "grav_capacity": grav_capacity.tolist(),
            "identity_err": identity_err, "balance": balance, "a_acc": a_acc}


def consistency(args, rho_lambda):
    """D: the capacity-scalar a(t) vs the stress-energy and imposed-law routes."""
    cap = integrate_capacity_scale([args.rho_m0, rho_lambda], [0.0, -1.0],
                                   dt=args.dt, steps=args.steps)
    ten = integrate_stress_energy([args.rho_m0, rho_lambda], [0.0, -1.0],
                                  dt=args.dt, steps=args.steps)
    imp = integrate_scale_factor(args.rho_m0, rho_lambda, dim=3, dt=args.dt,
                                 steps=args.steps)
    a = np.array(cap["a"]); m = a <= args.a_show
    r_ten = float(np.max(np.abs(a[m] - np.array(ten["a"])[m])
                         / np.maximum(np.array(ten["a"])[m], 1e-30)))
    r_imp = float(np.max(np.abs(a[m] - np.array(imp["a"])[m])
                         / np.maximum(np.array(imp["a"])[m], 1e-30)))
    print(f"  capacity a(t) vs stress-energy route: max rel |Δa|/a = {r_ten:.1e}",
          flush=True)
    print(f"  capacity a(t) vs imposed-law route:   max rel |Δa|/a = {r_imp:.1e}",
          flush=True)
    return {"tensor": ten, "imposed": imp, "resid_tensor": r_ten,
            "resid_imposed": r_imp}


def attractor(args, rho_lambda):
    """C (inset): a wrong capacity roll-rate relaxes onto the balance κ̇_s² = ρ."""
    out = []
    rho0 = args.rho_m0 + rho_lambda
    for f in args.violate:
        h = integrate_capacity_scale([args.rho_m0, rho_lambda], [0.0, -1.0],
                                     dt=args.dt, steps=args.steps,
                                     kappadot0=f * np.sqrt(rho0))
        C = np.array(h["constraint"])
        out.append({"factor": f, "a": h["a"], "constraint": h["constraint"],
                    "C0": float(C[0]), "C_end": float(C[-1])})
        print(f"  κ̇_s(0)={f:.1f}·√ρ₀: C₀={C[0]:+.4f} → C_end={C[-1]:+.1e}",
              flush=True)
    return out


def summarise(cap, cons, attr, args) -> list[str]:
    lines = ["Gravity from the capacity field: the gravitational action as "
             "capacity free energy — verdict", "=" * 72, ""]
    lines.append(
        "the identification: the scale factor is the exponential of the "
        "homogeneous capacity scalar, a = e^{κ_s} (κ_s = ln a, H = κ̇_s). The "
        "capacity scalar's kinetic free energy on the FLRW measure a³ is "
        "a³ κ̇_s² = a ȧ², so the minisuperspace gravitational term −a ȧ² is "
        "exactly −a³ κ̇_s².")
    lines.append("")
    lines.append(
        f"A. the gravitational term is capacity free energy: along the cosmic "
        f"history the posited −a ȧ² and the capacity scalar's kinetic free energy "
        f"−a³ κ̇_s² coincide to max|Δ|={cap['identity_err']:.0e} — the geometry's "
        "kinetic energy is the field's, not an independent Einstein–Hilbert posit.")
    lines.append("")
    lines.append(
        f"B. Friedmann is an energy balance κ̇_s² = ρ: the lapse constraint of the "
        f"capacity action says the mean capacity's kinetic free-energy density "
        f"equals the content; along the trajectory |κ̇_s² − ρ| stays below "
        f"{cap['balance']:.0e}. Expansion is the capacity field rolling, and "
        "Friedmann balances its kinetic free energy against matter + vacuum.")
    lines.append("")
    h = cap["hist"]
    lines.append(
        f"C. expansion is the capacity scalar rolling: its field equation "
        f"κ̈_s = −(3/2)(κ̇_s² + p) drives the roll — κ̈_s = {h['kappa_ddot'][0]:+.2f} "
        f"under dust → {h['kappa_ddot'][-1]:+.2f} in the vacuum limit (constant "
        f"roll = de Sitter). The balance κ̇_s² = ρ is never imposed yet holds, and "
        "a wrong initial roll-rate relaxes onto it (Ċ = −3 κ̇_s C).")
    lines.append("")
    lines.append(
        f"D. it reproduces the whole history: the scale factor a = e^{{κ_s}} from "
        f"the capacity-scalar dynamics coincides with the stress-energy route "
        f"(max rel |Δa|/a={cons['resid_tensor']:.0e}) and the imposed-law "
        f"cosmology ({cons['resid_imposed']:.0e}) — gravity's expansion re-derived "
        f"as the capacity field rolling, acceleration onset a_acc={cap['a_acc']:.2f}.")
    lines.append("")
    lines.append(
        "so the last posit is gone: the gravitational action's kinetic term is not "
        "an independent Einstein–Hilbert input but the capacity field's own "
        "kinetic free energy, and the cosmic expansion is that field — the mean "
        "capacity — rolling under it. Gravity, and not just its cosmology, is read "
        "off the URP field.")
    lines.append("")
    lines.append(
        "honest scope: the identification a = e^{κ_s} (scale factor as the exp of "
        "the homogeneous capacity scalar / e-folding number) is a reading within "
        "the URP framework, and the kinetic free energy carries gravity's "
        "wrong-sign conformal mode, taken as given. It is still minisuperspace "
        "(homogeneous, one degree of freedom), with no inhomogeneous field "
        "equations or solved metric. It traces the last hand-input at the "
        "Friedmann level to the field; it is not a derivation of general relativity.")
    return lines


def make_figure(cap, cons, attr, args, path):
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

    h = cap["hist"]
    a = np.array(h["a"]); m = a <= args.a_show

    # panel A: the gravitational term is capacity free energy (identity)
    gp = np.array(cap["grav_posited"]); gc = np.array(cap["grav_capacity"])
    ax1.plot(a[m], gp[m], color=GRAY, lw=5.0, alpha=0.5, zorder=2,
             label="posited  −a ȧ²  (Einstein–Hilbert)")
    ax1.plot(a[m], gc[m], color=BLUE, lw=1.8, zorder=3,
             label="capacity free energy  −a³ κ̇_s²")
    ax1.set_xlabel("scale factor a")
    ax1.set_ylabel("gravitational kinetic term")
    ax1.set_title(f"A. The gravitational term is capacity free energy "
                  f"(Δ ≤ {cap['identity_err']:.0e})")
    ax1.legend(frameon=False, fontsize=8, loc="lower left")

    # panel B: Friedmann as energy balance κ̇_s² = ρ
    kdot = np.array(h["kappa_dot"]); rho = np.array(h["rho_tot"])
    ax2.plot(a[m], (kdot ** 2)[m], color=BLUE, lw=2.4, zorder=3,
             label="κ̇_s²  (capacity kinetic free-energy density)")
    ax2.plot(a[m], rho[m], color=ORANGE, lw=1.6, ls="--", zorder=4,
             label="ρ  (matter + vacuum)")
    ax2.set_xlabel("scale factor a")
    ax2.set_ylabel("density")
    ax2.set_yscale("log")
    ax2.set_title("B. Friedmann as an energy balance: κ̇_s² = ρ")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")

    # panel C: the capacity scalar rolling — κ_s(t) and κ̈_s(a)
    t = np.array(h["t"]); ks = np.array(h["kappa_s"]); kddot = np.array(h["kappa_ddot"])
    ax3.plot(t[m], ks[m], color=BLUE, lw=2.2, zorder=3, label="κ_s = ln a")
    ax3.set_xlabel("time")
    ax3.set_ylabel("capacity scalar  κ_s", color=BLUE)
    ax3.tick_params(axis="y", labelcolor=BLUE)
    axc = ax3.twinx()
    axc.plot(t[m], kddot[m], color=GREEN, lw=1.8, zorder=3, label="κ̈_s (roll accel.)")
    axc.axhline(0.0, color=GRAY, ls=":", lw=1.0)
    axc.set_ylabel("κ̈_s = −(3/2)(κ̇_s²+p)", color=GREEN)
    axc.tick_params(axis="y", labelcolor=GREEN)
    for s in ("top",):
        axc.spines[s].set_visible(False)
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = axc.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
    ax3.set_title("C. Expansion is the capacity scalar rolling")

    # panel D: reproduces the whole history + attractor inset
    ten = cons["tensor"]; imp = cons["imposed"]
    ax4.plot(np.array(imp["t"])[m], np.array(imp["a"])[m], color=GRAY, lw=5.0,
             alpha=0.45, zorder=2, label="imposed-a^{−3}")
    ax4.plot(np.array(ten["t"])[m], np.array(ten["a"])[m], color=ORANGE, lw=2.8,
             zorder=3, label="stress-energy (∇·T=0)")
    ax4.plot(t[m], a[m], color=BLUE, lw=1.6, zorder=4, label="capacity scalar rolling")
    ax4.set_xlabel("time")
    ax4.set_ylabel("scale factor a = e^{κ_s}")
    ax4.set_title(f"D. Same history (Δa/a ≤ "
                  f"{max(cons['resid_tensor'], cons['resid_imposed']):.0e})")
    ax4.legend(frameon=False, fontsize=8, loc="upper left")
    axin = ax4.inset_axes([0.58, 0.10, 0.38, 0.42])
    cols = [RED, GREEN, PURPLE]
    for o, col in zip(attr, cols):
        aa = np.array(o["a"]); C = np.abs(np.array(o["constraint"]))
        mm = aa <= args.a_show
        axin.semilogy(aa[mm], np.maximum(C[mm], 1e-16), color=col, lw=1.4,
                      label=f"{o['factor']:.1f}·√ρ₀")
    axin.set_title("attractor: |κ̇_s²−ρ|", fontsize=7)
    axin.tick_params(labelsize=6)
    axin.legend(frameon=False, fontsize=6)

    fig.suptitle("Gravity from the capacity field: the gravitational action is "
                 "the capacity scalar's kinetic free energy", fontsize=12, y=0.98)
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
    p.add_argument("--violate", type=float, nargs="+", default=[0.6, 1.4])
    p.add_argument("--output-dir", type=str, default="artifacts/n3_capgrav")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.dt = 0.2
        args.steps = 350

    os.makedirs(args.output_dir, exist_ok=True)
    rho_lambda = capacity_vacuum_density(args.r, args.kappa0, args.vac_coeff)
    print(f"gravity from capacity: ρ_m0={args.rho_m0}, ρ_Λ={rho_lambda:.4f}, "
          f"a = e^(κ_s), flat FLRW minisuperspace, units 8πG/3=1", flush=True)
    print("A–C — the capacity scalar rolling (gravitational term = its free energy):")
    cap = capacity_history(args, rho_lambda)
    print("D — consistency with the stress-energy and imposed-law routes:")
    cons = consistency(args, rho_lambda)
    print("C — the balance κ̇_s² = ρ is an attractor:")
    attr = attractor(args, rho_lambda)

    lines = summarise(cap, cons, attr, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args), "rho_lambda": rho_lambda,
               "identity_err": cap["identity_err"], "balance": cap["balance"],
               "a_acc": cap["a_acc"], "resid_tensor": cons["resid_tensor"],
               "resid_imposed": cons["resid_imposed"],
               "attractor": [{"factor": o["factor"], "C0": o["C0"],
                              "C_end": o["C_end"]} for o in attr]}
    with open(os.path.join(args.output_dir, "n3_gravity_from_capacity.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_gravity_from_capacity.png")
        make_figure(cap, cons, attr, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
