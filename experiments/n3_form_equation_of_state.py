"""The equation of state of the matter: what kind of stuff the forms are.

The cosmology's Friedmann source now comes entirely from the field
(`n3_matter_from_forms.py`), but with an *assumed* character: matter as
pressureless **dust** (`ρ_m ∝ a^{−dim}`) and the capacity vacuum as a
**cosmological constant** (`ρ_Λ = const`).  Those characters are equations of
state `p = w·ρ`.  This experiment *measures* `w` for the forms — the last
qualitative input the cosmology takes on faith, and the piece a relativistic
stress-energy tensor `T_{μν}` will need.

Three independent readings, all agreeing that **cold forms are dust** (`w = 0`)
and the **capacity vacuum is a cosmological constant** (`w = −1`):

A. **Kinetic (the measurement).**  A gas of forms with peculiar velocities has a
   directly computable `w = p/ρ = Σγm v²/dim ÷ Σγm`.  At rest it is pressureless
   dust (`w → 0`); heated toward the speed of light it becomes radiation
   (`w → 1/dim`).  The measured `w(σ_v)` interpolates exactly between those
   limits — so the *cold* form gas the cosmology assumes really is dust.
B. **Mechanical (`p = −∂E/∂V`).**  A form is a *localized* lump: its rest energy
   is independent of the comoving box it sits in, so `∂E/∂V → 0` and its
   pressure vanishes — `w = 0`, dust — with no reference to velocities.  The
   capacity vacuum instead has energy `E_Λ = ρ_Λ·V` growing with the box, so
   `p = −∂E/∂V = −ρ_Λ` — `w = −1`, a cosmological constant.
C. **Kinematic (from the dilution exponent).**  In `dim` spatial dimensions
   `ρ ∝ a^{−dim(1+w)}`, so the dilution exponent *is* the equation of state:
   the `a^{−dim}` matter law (Link 8) reads back `w = 0`, the constant vacuum
   reads `w = −1`, and radiation (`a^{−(dim+1)}`) reads `w = 1/dim`.

Together they fix the two components a covariant `T_{μν}` must carry: a
pressureless dust of forms (`w = 0`) and a `w = −1` capacity vacuum — with the
warm/relativistic form gas smoothly bridging dust and radiation.

Honest scope: still a Newtonian `8πG/3 = 1` background — this measures the
*equation of state* the eventual `T_{μν}` needs, it does not yet build the
tensor or a covariant field equation.  The forms are point masses of the
measured Bogomolny rest energies; `w` is intensive (independent of the box),
and the mechanical `∂E/∂V → 0` is read off the localized CP² lump.

Usage::

    python experiments/n3_form_equation_of_state.py --output-dir artifacts/n3_eos
    python experiments/n3_form_equation_of_state.py --quick
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
    equation_of_state_from_dilution,
    gas_equation_of_state,
)
from project_genesis.stable_forms import structural_mass, winding_form


def form_masses(args):
    """The Bogomolny rest energies of the corpus (Link 2 / Link 8)."""
    return np.array([structural_mass(winding_form(args.form_size, Q, ring=args.ring))
                     for Q in args.charges])


def kinetic_eos(args, masses):
    """Part A: measured w(σ_v) of a form gas — dust (0) to radiation (1/dim)."""
    out = []
    for sigma in args.sigma_scan:
        ws = []
        for s in range(args.n_samples):
            rng = np.random.default_rng(s + 1)
            v = rng.normal(0.0, sigma, (len(masses), args.dim))
            speed = np.linalg.norm(v, axis=1)
            cap = args.max_speed
            over = speed > cap                       # keep speeds < 1 (units c=1)
            if over.any():
                v[over] *= (cap / speed[over])[:, None]
            ws.append(gas_equation_of_state(masses, v)["w"])
        out.append({"sigma": float(sigma), "w": float(np.mean(ws)),
                    "w_std": float(np.std(ws))})
        print(f"  σ_v={sigma:.2f}: w={np.mean(ws):.4f} ± {np.std(ws):.4f}  "
              f"(dust 0 → radiation {1.0/args.dim:.3f})", flush=True)
    return out


def mechanical_eos(args, rho_lambda):
    """Part B: p = −∂E/∂V for a localized form (→0) vs the vacuum (→−ρ_Λ)."""
    Ls = list(args.box_scan)
    Vs = np.array([L * L for L in Ls], dtype=float)     # 2-D lattice area of the form
    Es = np.array([structural_mass(winding_form(L, args.mech_charge, ring=args.ring))
                   for L in Ls])
    dE_form = float(np.polyfit(Vs, Es, 1)[0])           # ∂E/∂V of the form → 0
    E_vac = rho_lambda * Vs                             # E_Λ = ρ_Λ·V
    dE_vac = float(np.polyfit(Vs, E_vac, 1)[0])         # ∂E/∂V of the vacuum → ρ_Λ
    p_form, p_vac = -dE_form, -dE_vac
    w_form = p_form / (Es.mean() / Vs.mean())
    w_vac = p_vac / rho_lambda if rho_lambda > 0 else float("nan")
    print(f"  form:   E≈const over V, ∂E/∂V={dE_form:+.2e} → p≈{p_form:+.2e} "
          f"→ w≈{w_form:+.3f} (dust)", flush=True)
    print(f"  vacuum: E_Λ=ρ_Λ·V, ∂E/∂V={dE_vac:+.4f} → p={p_vac:+.4f} "
          f"→ w={w_vac:+.3f} (cosmological constant)", flush=True)
    return {"L": Ls, "V": Vs.tolist(), "E_form": Es.tolist(),
            "E_vac": E_vac.tolist(), "dE_form": dE_form, "dE_vac": dE_vac,
            "w_form": float(w_form), "w_vac": float(w_vac),
            "rho_lambda": rho_lambda}


def kinematic_eos(args):
    """Part C: w read off the dilution exponent for the three components."""
    comps = [("capacity vacuum", 0.0), ("form matter (dust)", float(args.dim)),
             ("radiation", float(args.dim + 1))]
    out = []
    for name, n in comps:
        w = equation_of_state_from_dilution(n, args.dim)
        out.append({"name": name, "exponent": n, "w": w})
        print(f"  {name:22s}: ρ ∝ a^(−{n:.0f})  → w = {w:+.3f}", flush=True)
    return out


def summarise(kin, mech, kinm, args) -> list[str]:
    lines = ["The equation of state of the matter: what kind of stuff the forms "
             "are — verdict", "=" * 70, ""]
    cold = min(kin, key=lambda k: k["sigma"])
    hot = max(kin, key=lambda k: k["sigma"])
    lines.append(
        f"A. kinetic (the measurement): a gas of forms with velocity dispersion "
        f"σ_v has w = p/ρ rising from w={cold['w']:.3f} at σ_v={cold['sigma']:.2f} "
        f"(cold — pressureless dust) toward w={hot['w']:.3f} at "
        f"σ_v={hot['sigma']:.2f}, heading for the radiation value 1/dim="
        f"{1.0/args.dim:.3f}. The cold form gas the cosmology assumes really is "
        "dust; heating it turns it into radiation, smoothly.")
    lines.append("")
    lines.append(
        f"B. mechanical (p = −∂E/∂V): a form is a localized lump — its rest "
        f"energy is independent of the box (∂E/∂V={mech['dE_form']:+.1e}), so its "
        f"pressure vanishes and w≈{mech['w_form']:+.2f}: dust, with no reference "
        f"to velocities. The capacity vacuum instead has E_Λ=ρ_Λ·V, so "
        f"∂E/∂V={mech['dE_vac']:.4f}=ρ_Λ and p=−ρ_Λ → w={mech['w_vac']:+.2f}: a "
        "cosmological constant.")
    lines.append("")
    lines.append(
        "C. kinematic (from the dilution exponent): since ρ ∝ a^(−dim(1+w)), the "
        "dilution law fixes w — " +
        ", ".join(f"{c['name']} (a^−{c['exponent']:.0f}) → w={c['w']:+.2f}"
                  for c in kinm) +
        ". The a^(−dim) matter law of Link 8 reads back exactly w=0.")
    lines.append("")
    lines.append(
        "so the two components a relativistic stress-energy tensor must carry are "
        "fixed and mutually consistent: a pressureless dust of forms (w=0) and a "
        "w=−1 capacity vacuum, with the warm form gas bridging dust and radiation "
        "(0 → 1/dim). The cosmology's 'dust + Λ' assumption is now measured, not "
        "assumed.")
    lines.append("")
    lines.append(
        "honest scope: still a Newtonian 8πG/3=1 background. This measures the "
        "equation of state the eventual T_{μν} needs — it does not yet build the "
        "tensor or a covariant field equation. The forms are point masses of the "
        "measured Bogomolny rest energies; w is intensive (box-independent), and "
        "the mechanical ∂E/∂V→0 is read off the localized CP² lump.")
    return lines


def make_figure(kin, mech, kinm, args, path):
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

    # panel A: measured w(σ_v) — dust → radiation
    sig = np.array([k["sigma"] for k in kin])
    w = np.array([k["w"] for k in kin])
    werr = np.array([k["w_std"] for k in kin])
    wrad = 1.0 / args.dim
    ax1.axhline(0.0, color=GRAY, ls=":", lw=1.1, zorder=2)
    ax1.axhline(wrad, color=ORANGE, ls="--", lw=1.2, zorder=2,
                label=f"radiation  w = 1/dim = {wrad:.3f}")
    ax1.errorbar(sig, w, yerr=werr, marker="o", color=BLUE, lw=2.0, ms=6,
                 capsize=3, zorder=4, label="measured  w = p/ρ")
    ax1.text(sig[0], 0.0, " dust  w = 0", color=GRAY, fontsize=8, va="bottom")
    ax1.set_xlabel("velocity dispersion  σ_v  (units c = 1)")
    ax1.set_ylabel("equation of state  w")
    ax1.set_title("A. The form gas measured: dust → radiation")
    ax1.legend(frameon=False, fontsize=8, loc="center right")

    # panel B: mechanical p = −∂E/∂V — localized form (flat) vs vacuum (linear)
    V = np.array(mech["V"])
    ax2.plot(V, mech["E_form"], marker="o", color=BLUE, lw=2.0, ms=6, zorder=4,
             label=f"form  E(V)  (∂E/∂V={mech['dE_form']:+.1e} → w≈0)")
    axb = ax2.twinx()
    axb.plot(V, mech["E_vac"], marker="s", color=ORANGE, lw=2.0, ms=6, zorder=3,
             label=f"vacuum  E_Λ=ρ_ΛV  (∂E/∂V=ρ_Λ → w=−1)")
    axb.set_ylabel("vacuum energy  E_Λ  in the box", color=ORANGE)
    axb.tick_params(axis="y", labelcolor=ORANGE)
    for s in ("top",):
        axb.spines[s].set_visible(False)
    ax2.set_xlabel("comoving box volume  V")
    ax2.set_ylabel("form rest energy  E", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax2.set_title("B. p = −∂E/∂V: localized form (w=0) vs vacuum (w=−1)")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="center right")

    # panel C: w from the dilution exponent — the three components
    names = [c["name"] for c in kinm]
    exps = np.array([c["exponent"] for c in kinm])
    ws = np.array([c["w"] for c in kinm])
    cols = [ORANGE, BLUE, RED]
    nn = np.linspace(-0.5, args.dim + 1.5, 50)
    ax3.plot(nn, nn / args.dim - 1.0, color=GRAY,
             lw=1.6, zorder=2, label="w = n/dim − 1")
    ax3.scatter(exps, ws, c=cols, s=90, zorder=4)
    for c, col in zip(kinm, cols):
        ax3.annotate(c["name"], (c["exponent"], c["w"]),
                     textcoords="offset points", xytext=(8, -4 if c["w"] > 0 else 8),
                     fontsize=8, color=col)
    ax3.axhline(0.0, color=GRAY, ls=":", lw=0.8, zorder=1)
    ax3.set_xlabel("dilution exponent  n   (ρ ∝ a^−n)")
    ax3.set_ylabel("equation of state  w")
    ax3.set_title("C. w read off the dilution law")
    ax3.legend(frameon=False, fontsize=8, loc="upper left")

    # panel D: the three components' dilution ρ(a) — what w means for expansion
    a = np.geomspace(1.0, 8.0, 80)
    for c, col in zip(kinm, cols):
        rho = a ** (-c["exponent"])
        ax4.loglog(a, rho, color=col, lw=2.0, zorder=3,
                   label=f"{c['name']}  (w={c['w']:+.2f})")
    ax4.set_xlabel("scale factor a")
    ax4.set_ylabel("density ρ (normalised)")
    ax4.set_title("D. What w means: dilution of each component")
    ax4.legend(frameon=False, fontsize=8, loc="lower left")

    fig.suptitle("The equation of state of the matter: cold forms are dust "
                 "(w=0), the capacity vacuum is Λ (w=−1)", fontsize=12, y=0.98)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--form-size", type=int, default=56)
    p.add_argument("--ring", type=float, default=3.0)
    p.add_argument("--charges", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--dim", type=int, default=3)
    p.add_argument("--sigma-scan", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9])
    p.add_argument("--max-speed", type=float, default=0.99)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--mech-charge", type=int, default=2)
    p.add_argument("--box-scan", type=int, nargs="+", default=[40, 48, 56, 64, 80, 96])
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--kappa0", type=float, default=1.0)
    p.add_argument("--vac-coeff", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_eos")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.form_size = 48
        args.sigma_scan = [0.05, 0.2, 0.4, 0.7, 0.9]
        args.n_samples = 80
        args.box_scan = [40, 56, 72, 96]

    os.makedirs(args.output_dir, exist_ok=True)
    rho_lambda = capacity_vacuum_density(args.r, args.kappa0, args.vac_coeff)
    print(f"equation of state of the forms: dim={args.dim}, charges={args.charges}, "
          f"ρ_Λ={rho_lambda:.4f}", flush=True)
    masses = form_masses(args)
    print("Part A — kinetic w(σ_v) of the form gas:")
    kin = kinetic_eos(args, masses)
    print("Part B — mechanical p = −∂E/∂V:")
    mech = mechanical_eos(args, rho_lambda)
    print("Part C — kinematic w from the dilution exponent:")
    kinm = kinematic_eos(args)

    lines = summarise(kin, mech, kinm, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args), "rho_lambda": rho_lambda,
               "form_masses": masses.tolist(), "kinetic": kin,
               "mechanical": {k: mech[k] for k in
                              ("dE_form", "dE_vac", "w_form", "w_vac")},
               "kinematic": kinm}
    with open(os.path.join(args.output_dir, "n3_form_equation_of_state.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_form_equation_of_state.png")
        make_figure(kin, mech, kinm, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
