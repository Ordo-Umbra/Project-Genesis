"""The relativistic closure: expansion as a consequence of a stress-energy tensor.

Everything the cosmology needs is now measured from the field: the matter
density and its dilution (Links 7–8) and the equation of state of each component
(Link 9 — forms are dust `w = 0`, the capacity vacuum is `w = −1`).  This
assembles them into a **perfect-fluid stress-energy tensor** in a 3+1 FLRW
background and lets the expansion *follow* from it, rather than imposing the
Friedmann matter law by hand:

    T^μ_ν = diag(−ρ, p, p, p),   p_i = w_i·ρ_i,   ρ = Σρ_i,  p = Σp_i .

Its covariant conservation `∇_μ T^{μν} = 0` has, in FLRW, one non-trivial
component — the continuity equation `ρ̇_i + 3H(ρ_i + p_i) = 0` — and closing with
the Einstein/Friedmann relations `H² = ρ`, `ä/a = −½(ρ + 3p)` makes `a(t)` a
consequence of the field's own stress-energy.  Four results:

A. **The tensor and its evolving equation of state.**  Built from the dust of
   forms (`w = 0`) and the capacity vacuum (`w = −1`), the *effective* equation
   of state `w_eff = p/ρ` runs from ~0 while matter dominates to `−1` once the
   vacuum does, crossing the acceleration threshold `w_eff = −1/3`.
B. **Conservation *derives* the dilution laws.**  Integrating
   `ρ̇ = −3H(ρ + p)` for each component reproduces `ρ ∝ a^{−3(1+w)}` to ~1e−5 —
   dust `a^{−3}`, the vacuum constant, radiation `a^{−4}`.  The `a^{−dim}` matter
   law we had *imposed* is now an output of covariant conservation.
C. **Expansion as an output.**  The deceleration parameter `q = ½(1 + 3w_eff)`
   crosses zero when `w_eff = −1/3`, so `a(t)` decelerates then accelerates —
   with nothing about the dilution assumed, only the measured equations of state
   and conservation.
D. **Consistency with the imposed-law cosmology.**  The `a(t)` from the coupled
   tensor coincides with the earlier imposed-`a^{−3}` cosmology
   (`integrate_scale_factor`) to the integration precision — the new derivation
   *explains* the old input.

Honest scope: a perfect-fluid `T^μ_ν = diag(−ρ, p, p, p)` in a homogeneous FLRW
background with the standard Einstein/Friedmann relations (units `8πG/3 = 1`).
This makes the *expansion law* a consequence of the field's stress-energy and its
measured equations of state — it is **not** a derivation of the Einstein field
equations from the κ-action, and it stays homogeneous (no perturbations, metric
solved, or covariant field theory of κ itself).  It closes the Friedmann level:
`T_{μν}` from the field, expansion as its consequence.

Usage::

    python experiments/n3_stress_energy_closure.py --output-dir artifacts/n3_tmunu
    python experiments/n3_stress_energy_closure.py --quick
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
    integrate_scale_factor,
    integrate_stress_energy,
    stress_energy_tensor,
)


def _q_zero_crossing(a, q):
    """Interpolated scale factor where q crosses zero (acceleration onset)."""
    q = np.asarray(q); a = np.asarray(a)
    below = np.where(q < 0)[0]
    if len(below) and below[0] > 0:
        i = below[0]
        frac = q[i - 1] / (q[i - 1] - q[i])
        return float(a[i - 1] + frac * (a[i] - a[i - 1]))
    return float("nan")


def main_history(args, rho_lambda):
    """A/C: the coupled dust + vacuum tensor evolved — nothing imposed."""
    h = integrate_stress_energy([args.rho_m0, rho_lambda], [0.0, -1.0],
                                dt=args.dt, steps=args.steps)
    a = np.array(h["a"]); q = np.array(h["q"]); w = np.array(h["w_eff"])
    a_acc = _q_zero_crossing(a, q)
    print(f"  tensor history: w_eff {w[0]:+.3f} → {w[-1]:+.3f}, "
          f"q {q[0]:+.3f} → {q[-1]:+.3f}, acceleration at a_acc={a_acc:.2f} "
          f"(q=0 ⇔ w_eff=−1/3)", flush=True)
    return {"a": h["a"], "t": h["t"], "q": h["q"], "w_eff": h["w_eff"],
            "rho_tot": h["rho_tot"], "a_acc": a_acc}


def conservation_check(args, rho_lambda):
    """B: covariant conservation reproduces ρ ∝ a^{−3(1+w)} for each component."""
    dens0 = [args.rho_m0, rho_lambda, args.rho_r0]
    eos = [0.0, -1.0, 1.0 / 3.0]
    names = ["form dust (w=0)", "capacity vacuum (w=−1)", "radiation (w=1/3)"]
    h = integrate_stress_energy(dens0, eos, dt=args.dt, steps=args.steps)
    a = np.array(h["a"]); comp = np.array(h["rho_components"])
    out = []
    for i, (name, w) in enumerate(zip(names, eos)):
        rho = comp[:, i]
        analytic = rho[0] * a ** (-3.0 * (1.0 + w))
        window = a <= args.a_show
        err = float(np.max(np.abs(rho[window] - analytic[window])
                           / np.maximum(analytic[window], 1e-30)))
        out.append({"name": name, "w": w, "a": a.tolist(),
                    "rho": rho.tolist(), "analytic": analytic.tolist(),
                    "max_rel_err": err})
        print(f"  {name:26s}: ρ(a) vs a^(−{3*(1+w):.0f}) max rel err "
              f"(a≤{args.a_show:g}) = {err:.1e}", flush=True)
    return out


def consistency_check(args, rho_lambda):
    """D: coupled-tensor a(t) vs the earlier imposed-a^{−3} cosmology."""
    hb = integrate_stress_energy([args.rho_m0, rho_lambda], [0.0, -1.0],
                                 dt=args.dt, steps=args.steps)
    hi = integrate_scale_factor(args.rho_m0, rho_lambda, dim=3, dt=args.dt,
                                steps=args.steps)
    a_t = np.array(hb["a"]); a_i = np.array(hi["a"])
    window = a_t <= args.a_show
    resid = float(np.max(np.abs(a_t[window] - a_i[window])
                         / np.maximum(a_i[window], 1e-30)))
    print(f"  coupled-tensor vs imposed-law: max rel |Δa|/a (a≤{args.a_show:g}) "
          f"= {resid:.1e}", flush=True)
    return {"t": hb["t"], "a_tensor": hb["a"], "a_imposed": hi["a"],
            "max_rel_resid": resid}


def summarise(hist, cons, consist, args) -> list[str]:
    lines = ["The relativistic closure: expansion from a stress-energy tensor "
             "— verdict", "=" * 66, ""]
    T = stress_energy_tensor([args.rho_m0, capacity_vacuum_density(
        args.r, args.kappa0, args.vac_coeff)], [0.0, -1.0])
    lines.append(
        f"A. the tensor: assembling T^μ_ν = diag(−ρ, p, p, p) from the dust of "
        f"forms (w=0) and the capacity vacuum (w=−1) gives, at a=1, "
        f"ρ={T['rho']:.3f}, p={T['pressure']:+.3f}, w_eff={T['w_eff']:+.3f}; as "
        f"the universe expands w_eff runs {hist['w_eff'][0]:+.2f} → "
        f"{hist['w_eff'][-1]:+.2f} (matter → vacuum), crossing the acceleration "
        "threshold w_eff=−1/3.")
    lines.append("")
    worst = max(c["max_rel_err"] for c in cons)
    lines.append(
        "B. conservation derives the dilution laws: integrating the continuity "
        "equation ρ̇=−3H(ρ+p) that is the (0-)component of ∇_μT^{μν}=0 reproduces "
        "ρ ∝ a^(−3(1+w)) for every component — " +
        ", ".join(f"{c['name'].split(' (')[0]} → a^(−{3*(1+c['w']):.0f})"
                  for c in cons) +
        f" — to better than {worst:.0e}. The a^(−dim) matter law we had imposed "
        "is now an output of covariant conservation.")
    lines.append("")
    lines.append(
        f"C. expansion as an output: with H²=ρ and ä/a=−½(ρ+3p), the deceleration "
        f"parameter q=½(1+3w_eff) crosses zero at a_acc={hist['a_acc']:.2f}, so "
        f"a(t) decelerates then accelerates — nothing about the dilution assumed, "
        "only the measured equations of state and conservation.")
    lines.append("")
    lines.append(
        f"D. consistency: the a(t) from the coupled tensor coincides with the "
        f"earlier imposed-a^(−3) cosmology to max rel |Δa|/a={consist['max_rel_resid']:.0e} "
        f"(a≤{args.a_show:g}) — the new derivation explains the old input rather "
        "than replacing it.")
    lines.append("")
    lines.append(
        "so the Friedmann level is closed: a stress-energy tensor built from the "
        "field's own content (dust of forms + capacity vacuum) with measured "
        "equations of state, made to conserve covariantly, produces the expansion "
        "history — matter dilution, the decel→accel turnover, and the de Sitter "
        "limit — as a consequence. The Friedmann equation is now an output.")
    lines.append("")
    lines.append(
        "honest scope: a perfect-fluid T^μ_ν = diag(−ρ,p,p,p) in a homogeneous "
        "FLRW background with the standard Einstein/Friedmann relations "
        "(8πG/3=1). It makes the expansion law a consequence of the field's "
        "stress-energy and its measured equations of state — it is not a "
        "derivation of the Einstein field equations from the κ-action, and it "
        "stays homogeneous (no perturbations or covariant field theory of κ "
        "itself).")
    return lines


def make_figure(hist, cons, consist, args, path):
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

    # panel A: the evolving effective equation of state w_eff(a)
    a = np.array(hist["a"]); w = np.array(hist["w_eff"])
    m = a <= args.a_show
    ax1.plot(a[m], w[m], color=BLUE, lw=2.2, zorder=3, label="w_eff = p/ρ")
    ax1.axhline(0.0, color=GRAY, ls=":", lw=1.0, zorder=2)
    ax1.axhline(-1.0, color=GRAY, ls=":", lw=1.0, zorder=2)
    ax1.axhline(-1.0 / 3.0, color=RED, ls="--", lw=1.2, zorder=2,
                label="accel threshold  w_eff = −1/3")
    ax1.text(a[m][0], 0.0, " dust (w=0)", color=GRAY, fontsize=8, va="bottom")
    ax1.text(a[m][-1], -1.0, "Λ (w=−1) ", color=GRAY, fontsize=8, va="top",
             ha="right")
    ax1.set_xlabel("scale factor a")
    ax1.set_ylabel("effective equation of state  w_eff")
    ax1.set_title("A. The tensor's equation of state: matter → Λ")
    ax1.legend(frameon=False, fontsize=8, loc="center right")

    # panel B: conservation reproduces the dilution laws
    cols = [BLUE, ORANGE, RED]
    for c, col in zip(cons, cols):
        ac = np.array(c["a"]); rho = np.array(c["rho"]); an = np.array(c["analytic"])
        mm = ac <= args.a_show
        ax2.loglog(ac[mm], an[mm], color=col, lw=1.4, ls="--", zorder=2)
        ax2.loglog(ac[mm][::max(1, mm.sum() // 12)],
                   rho[mm][::max(1, mm.sum() // 12)], color=col, marker="o",
                   lw=0, ms=5, zorder=4,
                   label=f"{c['name']}  (a^−{3*(1+c['w']):.0f})")
    ax2.set_xlabel("scale factor a")
    ax2.set_ylabel("component density ρ_i")
    ax2.set_title("B. Conservation derives the dilution laws (∇·T=0)")
    ax2.legend(frameon=False, fontsize=8, loc="lower left")

    # panel C: expansion as an output — a(t) and q crossing zero
    t = np.array(hist["t"]); q = np.array(hist["q"])
    ax3.plot(t[m], a[m], color=BLUE, lw=2.2, zorder=3, label="a(t)")
    ax3.set_xlabel("time")
    ax3.set_ylabel("scale factor a(t)", color=BLUE)
    ax3.tick_params(axis="y", labelcolor=BLUE)
    axc = ax3.twinx()
    axc.plot(t[m], q[m], color=GREEN, lw=1.8, zorder=3, label="q(a)")
    axc.axhline(0.0, color=RED, ls="--", lw=1.1, zorder=2)
    axc.set_ylabel("deceleration parameter q", color=GREEN)
    axc.tick_params(axis="y", labelcolor=GREEN)
    for s in ("top",):
        axc.spines[s].set_visible(False)
    if np.isfinite(hist["a_acc"]):
        ia = int(np.argmin(np.abs(a - hist["a_acc"])))
        ax3.plot([t[ia]], [a[ia]], marker="o", color=RED, ms=8, zorder=5)
        ax3.text(t[ia], a[ia], "  accel begins (q=0)", color=RED, fontsize=8,
                 va="center")
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = axc.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
    ax3.set_title("C. Expansion as an output: q = ½(1+3w_eff)")

    # panel D: consistency with the imposed-law cosmology
    td = np.array(consist["t"]); at = np.array(consist["a_tensor"])
    ai = np.array(consist["a_imposed"])
    md = at <= args.a_show
    ax4.plot(td[md], ai[md], color=GRAY, lw=4.0, alpha=0.5, zorder=2,
             label="imposed-a^{−3} cosmology")
    ax4.plot(td[md], at[md], color=BLUE, lw=1.8, zorder=4,
             label="coupled T^μ_ν (conservation)")
    ax4.set_xlabel("time")
    ax4.set_ylabel("scale factor a(t)")
    ax4.set_title(f"D. Same history: max rel |Δa|/a = "
                  f"{consist['max_rel_resid']:.0e}")
    ax4.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("The relativistic closure: a stress-energy tensor from the "
                 "field, and expansion as its consequence", fontsize=12, y=0.98)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rho-m0", type=float, default=0.05)     # dust of forms
    p.add_argument("--rho-r0", type=float, default=0.02)     # radiation (panel B)
    p.add_argument("--r", type=float, default=0.02)          # capacity recovery
    p.add_argument("--kappa0", type=float, default=1.0)
    p.add_argument("--vac-coeff", type=float, default=0.5)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--a-show", type=float, default=6.0)      # plotting window
    p.add_argument("--output-dir", type=str, default="artifacts/n3_tmunu")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.dt = 0.2
        args.steps = 350

    os.makedirs(args.output_dir, exist_ok=True)
    rho_lambda = capacity_vacuum_density(args.r, args.kappa0, args.vac_coeff)
    print(f"stress-energy closure: ρ_m0={args.rho_m0}, ρ_Λ={rho_lambda:.4f} "
          f"(=½·r·κ₀²), 3+1 FLRW, units 8πG/3=1", flush=True)
    print("A/C — the coupled tensor history:")
    hist = main_history(args, rho_lambda)
    print("B — covariant conservation reproduces the dilution laws:")
    cons = conservation_check(args, rho_lambda)
    print("D — consistency with the imposed-law cosmology:")
    consist = consistency_check(args, rho_lambda)

    lines = summarise(hist, cons, consist, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args), "rho_lambda": rho_lambda,
               "a_acc": hist["a_acc"], "w_eff_range": [hist["w_eff"][0],
                                                       hist["w_eff"][-1]],
               "conservation_max_rel_err": [c["max_rel_err"] for c in cons],
               "consistency_max_rel_resid": consist["max_rel_resid"]}
    with open(os.path.join(args.output_dir, "n3_stress_energy_closure.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_stress_energy_closure.png")
        make_figure(hist, cons, consist, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
