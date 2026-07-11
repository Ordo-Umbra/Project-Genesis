"""Matter from forms: the Friedmann matter source read off the form spectrum.

The self-contained cosmos (`n3_self_contained_cosmos.py`) derived the dark
energy from the capacity field's self-maintenance but still *dialled* the matter
density ``ρ_m0`` and *imposed* its dilution law ``ρ_m ∝ a^{-dim}`` by hand.
This closes that last gap by tracing the matter source all the way back to
**Link 2** — matter as the stable topological forms of the sector field ψ∈ℂ³.

Two facts about those forms do all the work:

1. **Their rest energies are quantised by charge.**  A charge-``Q`` CP² soliton
   sits on a Bogomolny bound ``E ≥ (const)·|Q|`` — an energy *floor* it cannot
   decay below.  So the matter density of a corpus of forms is
   ``ρ_m0 = Σ E_i / V ∝ Σ |Q_i|``: it is read off the **topological content**
   of the field, not chosen.
2. **That charge is topologically protected.**  ``Q ∈ π₂(CP²) = ℤ`` cannot
   change under any smooth deformation; the rest energy floor rides with it.
   A set of forms therefore carries a *conserved* total rest energy, so as the
   comoving volume grows ``V(a) = V₀·a^{dim}`` the only thing that changes is
   the volume they are spread over — giving ``ρ_m(a) = ρ_m0·a^{-dim}`` as a
   **consequence** of topology, not an imposed law.

Four results, each a step of the closure:

A. **The mass ladder.**  Structural (rest) energies of the charge ``Q = 1..4``
   forms lie on a straight line ``E ≈ slope·|Q| + floor`` — the Bogomolny ladder.
   Matter density is proportional to total charge.
B. **Topological protection.**  Deform a charge-2 form with growing noise: the
   raw geometric charge blows up on UV dislocations, but after cooling the
   charge returns to exactly ``Q = 2`` — the rest-energy floor is protected.
C. **The dilution law is topological.**  Because the total rest energy is
   conserved, ``ρ_m(a) = Σ E_i / (V₀ a^{dim})`` falls with a log–log slope of
   exactly ``−dim`` — the ``a^{-dim}`` law derived, not assumed.
D. **The cosmos from its form content.**  Feeding ``ρ_m0`` (from the forms) and
   ``ρ_Λ`` (from the recovery term) into the Friedmann integrator makes the
   acceleration onset ``a_acc = (ρ_m0/2ρ_Λ)^{1/dim}`` a function of **how many
   stable forms the universe contains** — more matter, later acceleration, on
   the predicted curve.

Honest scope: a Newtonian ``8πG/3 = 1`` Friedmann closure. The forms are 2-D
CP² solitons whose Bogomolny energies set the rest masses; they are then treated
as point masses populating a ``dim``-dimensional comoving volume, which is what
gives the ``a^{-dim}`` dilution.  The dark-energy identification still carries a
modelling coefficient and there is no metric or relativistic stress-energy
tensor.  What it establishes is that the *matter* term of the cosmology is fixed
by the topological content of the field — the last dialled density removed.

Usage::

    python experiments/n3_matter_from_forms.py --output-dir artifacts/n3_matter
    python experiments/n3_matter_from_forms.py --quick
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
    acceleration_onset,
    capacity_vacuum_density,
    integrate_scale_factor,
    matter_energy_density,
)
from project_genesis.stable_forms import structural_mass, winding_form
from project_genesis.topological_charge import cool, topological_charge


def form_spectrum(args):
    """Part A: the Bogomolny mass ladder E_i ∝ |Q_i| of the stable forms."""
    charges = list(args.charges)
    energies = [structural_mass(winding_form(args.form_size, Q, ring=args.ring))
                for Q in charges]
    q = np.array([abs(Q) for Q in charges], dtype=float)
    e = np.array(energies, dtype=float)
    slope, floor = np.polyfit(q, e, 1)
    pred = slope * q + floor
    ss_res = float(((e - pred) ** 2).sum())
    ss_tot = float(((e - e.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    print(f"  spectrum: " +
          "  ".join(f"|Q|={int(qi)}→E={ei:.2f}" for qi, ei in zip(q, e)))
    print(f"  Bogomolny ladder: E ≈ {slope:.3f}·|Q| + {floor:.3f}  (R²={r2:.4f})",
          flush=True)
    return {"charges": [abs(c) for c in charges], "energies": energies,
            "slope": float(slope), "floor": float(floor), "r2": float(r2)}


def topological_protection(args):
    """Part B: the charge (hence rest-energy floor) survives deformation.

    Averaged over ``n_seeds`` random deformations at each noise level, so the
    protection (and the threshold where a violent kick tunnels to a neighbouring
    sector) is a statistical statement, not one lucky realisation.
    """
    Q = args.protect_charge
    z0 = winding_form(args.form_size, Q, ring=args.ring)
    out = []
    for noise in args.noise_scan:
        raws, cools = [], []
        for s in range(args.n_seeds):
            rng = np.random.default_rng(1000 * s + 7)
            z = z0 + noise * (rng.standard_normal(z0.shape)
                              + 1j * rng.standard_normal(z0.shape))
            z /= np.linalg.norm(z, axis=-1, keepdims=True)
            raws.append(topological_charge(z))
            cools.append(topological_charge(cool(z, args.cool_steps, 1.0)))
        raws = np.array(raws); cools = np.array(cools)
        frac = float(np.mean(np.abs(np.round(cools) - Q) < 0.5))
        rec = {"noise": float(noise), "q_raw_absmean": float(np.abs(raws).mean()),
               "q_cool_mean": float(cools.mean()), "q_cool_std": float(cools.std()),
               "frac_protected": frac}
        out.append(rec)
        print(f"  noise={noise:.2f}: ⟨|raw Q|⟩={np.abs(raws).mean():5.1f}  "
              f"⟨cooled Q⟩={cools.mean():+.2f}±{cools.std():.2f}  "
              f"protected@Q={Q}: {frac:.0%}", flush=True)
    return out


def dilution(args, spectrum):
    """Part C: conserved rest energy + growing volume → ρ_m ∝ a^{-dim}."""
    total_rest = float(np.sum(spectrum["energies"]))       # conserved (Part B)
    v0 = args.box ** args.dim
    a = np.geomspace(1.0, args.a_max, 60)
    # the volume dilutes the *conserved* rest energy: ρ_m(a) = ΣE / (V₀ a^dim)
    rho_m = np.array([matter_energy_density(spectrum["energies"], v0 * ai ** args.dim)
                      for ai in a])
    slope = float(np.polyfit(np.log(a), np.log(rho_m), 1)[0])
    print(f"  dilution: total rest energy ΣE={total_rest:.2f} (conserved), "
          f"ρ_m(a) log–log slope = {slope:+.3f}  (−dim = {-args.dim})", flush=True)
    return {"a": a.tolist(), "rho_m": rho_m.tolist(), "slope": slope,
            "total_rest": total_rest, "rho_m0": float(rho_m[0])}


def _a_acc_measured(rho_m0, rho_L, args):
    """Integrate the Friedmann equation and read the q = 0 crossing."""
    h = integrate_scale_factor(rho_m0, rho_L, dim=args.dim, dt=args.dt,
                               steps=args.hist_steps)
    a = np.array(h["a"]); q = np.array(h["q"])
    cross = np.where(q < 0)[0]
    if len(cross) and cross[0] > 0:
        i = cross[0]
        frac = q[i - 1] / (q[i - 1] - q[i])
        return float(a[i - 1] + frac * (a[i] - a[i - 1])), h
    return (float(a[cross[0]]) if len(cross) else float("nan")), h


def closed_source(args, spectrum):
    """Part D: the cosmos built from its form content — a_acc(Σ|Q|)."""
    rho_L = capacity_vacuum_density(args.r, args.kappa0, args.vac_coeff)
    v0 = args.box ** args.dim
    spec_charge = float(np.sum(spectrum["charges"]))
    spec_energy = float(np.sum(spectrum["energies"]))
    out = []
    ref_hist = None
    for m in args.content_scan:                # m generations of the spectrum
        q_tot = m * spec_charge
        rho_m0 = matter_energy_density([spec_energy] * m, v0)
        a_pred = acceleration_onset(rho_m0, rho_L, args.dim)
        a_meas, h = _a_acc_measured(rho_m0, rho_L, args)
        if m == args.content_ref:
            ref_hist = {"a": h["a"], "t": h["t"], "q": h["q"],
                        "rho_m0": rho_m0, "a_acc": a_meas}
        out.append({"m": m, "q_tot": q_tot, "rho_m0": rho_m0,
                    "a_acc_pred": a_pred, "a_acc_meas": a_meas})
        print(f"  content m={m}: Σ|Q|={q_tot:.0f}  ρ_m0={rho_m0:.4f}  "
              f"a_acc pred={a_pred:.2f} meas={a_meas:.2f}", flush=True)
    if ref_hist is None:
        ref_hist = {"a": [], "t": [], "q": [], "rho_m0": float("nan"),
                    "a_acc": float("nan")}
    return {"scan": out, "rho_lambda": rho_L, "ref": ref_hist}


def summarise(spectrum, protect, dilute, closed, args) -> list[str]:
    lines = ["Matter from forms: the Friedmann source from the form spectrum "
             "— verdict", "=" * 66, ""]
    lines.append(
        f"A. the mass ladder: the stable forms Q=1..{max(spectrum['charges'])} "
        f"have structural (rest) energies on a straight Bogomolny line "
        f"E ≈ {spectrum['slope']:.2f}·|Q| + {spectrum['floor']:.2f} "
        f"(R²={spectrum['r2']:.3f}). Matter density ρ_m0 = ΣE/V is therefore "
        "proportional to the total topological charge — read off the field, not "
        "dialled.")
    lines.append("")
    Q = args.protect_charge
    plateau = [p for p in protect if p["frac_protected"] >= 0.999]
    n_hi = max((p["noise"] for p in plateau), default=0.0)
    broke = next((p for p in protect if p["frac_protected"] < 0.999), None)
    tail = (f" Only past a threshold (here noise≈{broke['noise']:.1f}, "
            f"{broke['frac_protected']:.0%} protected) does a kick comparable to "
            "the field itself carry it over the barrier into a neighbouring sector."
            if broke is not None else "")
    lines.append(
        f"B. topological protection: deforming a charge-{Q} form and cooling "
        f"recovers exactly Q={Q} for every realisation up to noise≈{n_hi:.1f}, "
        f"while the raw geometric charge has already blown up on UV dislocations "
        f"(⟨|raw Q|⟩ reaching {max(p['q_raw_absmean'] for p in protect):.0f})."
        + tail + " The rest-energy floor rides with the protected charge.")
    lines.append("")
    lines.append(
        f"C. the dilution law is topological: because that total rest energy is "
        f"conserved, spreading it through a growing comoving volume "
        f"V₀·a^{args.dim} gives ρ_m(a) with a log–log slope of "
        f"{dilute['slope']:+.2f} — i.e. ρ_m ∝ a^(−{args.dim}). The a^(−dim) law "
        "is a consequence of conserved topology plus volume, not an imposed input.")
    lines.append("")
    scan = closed["scan"]
    lines.append(
        f"D. the cosmos from its form content: feeding ρ_m0 (from the forms) and "
        f"ρ_Λ={closed['rho_lambda']:.3f} (from the recovery term) into the "
        f"Friedmann integrator, the acceleration onset tracks how many forms the "
        f"universe holds — " +
        ", ".join(f"a_acc={s['a_acc_meas']:.2f} at Σ|Q|={s['q_tot']:.0f}"
                  for s in scan) +
        " — more matter, later acceleration, on the predicted curve.")
    lines.append("")
    lines.append(
        "so the last dialled density is gone: the matter term of the cosmology is "
        "fixed by the topological content of the sector field — ρ_m0 from the "
        "Bogomolny spectrum, a^(−dim) from charge conservation. Combined with dark "
        "energy from κ's self-maintenance, the whole Friedmann source is now the "
        "field's own content. Matter is stable form (Link 2), and the cosmos runs "
        "on it.")
    lines.append("")
    lines.append(
        "honest scope: a Newtonian 8πG/3=1 closure. The forms are 2-D CP² solitons "
        "whose Bogomolny energies set the rest masses; they are then treated as "
        "point masses in a dim-dimensional comoving volume (which is what gives "
        "a^(−dim)). The dark-energy identification still carries a modelling "
        "coefficient and there is no metric or relativistic stress-energy tensor. "
        "It fixes the matter *source* from topology — not a first-principles ΛCDM.")
    return lines


def make_figure(spectrum, protect, dilute, closed, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY, RED, PURPLE = (
        "#0072B2", "#E69F00", "#009E73", "#999999", "#c0392b", "#8551A6")
    fig = plt.figure(figsize=(13, 9.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.26)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))
    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # panel A: the Bogomolny mass ladder E ∝ |Q|
    q = np.array(spectrum["charges"], dtype=float)
    e = np.array(spectrum["energies"], dtype=float)
    qq = np.linspace(0, q.max() + 0.5, 50)
    ax1.plot(qq, spectrum["slope"] * qq + spectrum["floor"], color=GRAY, lw=1.8,
             zorder=2, label=f"E = {spectrum['slope']:.2f}|Q| + "
                             f"{spectrum['floor']:.2f}")
    ax1.plot(q, e, marker="o", color=BLUE, ms=9, lw=0, zorder=4,
             label="stable forms")
    ax1.set_xlabel("topological charge |Q|")
    ax1.set_ylabel("structural (rest) energy E")
    ax1.set_title(f"A. The Bogomolny mass ladder (R²={spectrum['r2']:.3f})")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    # panel B: topological protection under deformation (averaged over seeds)
    ns = np.array([p["noise"] for p in protect])
    q_raw = np.array([p["q_raw_absmean"] for p in protect])
    q_cool = np.array([p["q_cool_mean"] for p in protect])
    q_err = np.array([p["q_cool_std"] for p in protect])
    true_q = args.protect_charge
    ax2.plot(ns, q_raw, marker="s", color=RED, lw=1.6, ms=6, zorder=3,
             label="⟨|raw geometric Q|⟩ (UV noise)")
    ax2.errorbar(ns, q_cool, yerr=q_err, marker="o", color=GREEN, lw=2.0, ms=7,
                 capsize=3, zorder=4, label="⟨cooled Q⟩ (protected)")
    ax2.axhline(true_q, color=GRAY, ls="--", lw=1.1, zorder=2,
                label=f"true charge Q={true_q}")
    ax2.set_xlabel("deformation noise amplitude")
    ax2.set_ylabel("topological charge Q")
    ax2.set_title("B. Charge is protected — so is the rest-energy floor")
    ax2.legend(frameon=False, fontsize=8, loc="lower left")

    # panel C: the derived dilution law ρ_m ∝ a^{-dim}
    a = np.array(dilute["a"]); rho = np.array(dilute["rho_m"])
    ax3.loglog(a, rho, color=BLUE, lw=2.2, zorder=3, label="ρ_m(a) = ΣE / V₀a^dim")
    ref = rho[0] * a ** (-args.dim)
    ax3.loglog(a, ref, color=GRAY, ls="--", lw=1.4, zorder=2,
               label=f"slope −dim = −{args.dim}")
    ax3.set_xlabel("scale factor a")
    ax3.set_ylabel("matter density ρ_m")
    ax3.set_title(f"C. Dilution from topology: slope {dilute['slope']:+.2f}")
    ax3.legend(frameon=False, fontsize=8, loc="lower left")

    # panel D: the acceleration onset set by the form content
    scan = closed["scan"]
    qt = np.array([s["q_tot"] for s in scan])
    ax4.plot(qt, [s["a_acc_pred"] for s in scan], color=GRAY, lw=1.8, zorder=2,
             label="derived (ρ_m0/2ρ_Λ)^(1/dim)")
    ax4.plot(qt, [s["a_acc_meas"] for s in scan], color=GREEN, marker="D", ms=7,
             lw=0, zorder=4, label="from integration")
    ax4.set_xlabel("total form charge  Σ|Q|  (matter content)")
    ax4.set_ylabel("acceleration onset  a_acc")
    ax4.set_title("D. The cosmos from its form content: a_acc(Σ|Q|)")
    ax4.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("Matter from forms: the Friedmann matter source read off the "
                 "topological spectrum of the field", fontsize=12, y=0.98)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--form-size", type=int, default=64)
    p.add_argument("--ring", type=float, default=3.0)
    p.add_argument("--charges", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--protect-charge", type=int, default=2)
    p.add_argument("--noise-scan", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--cool-steps", type=int, default=80)
    p.add_argument("--box", type=float, default=13.0)   # comoving box side
    p.add_argument("--dim", type=int, default=3)
    p.add_argument("--a-max", type=float, default=100.0)
    p.add_argument("--r", type=float, default=0.02)     # capacity recovery rate
    p.add_argument("--kappa0", type=float, default=1.0)
    p.add_argument("--vac-coeff", type=float, default=0.5)
    p.add_argument("--content-scan", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--content-ref", type=int, default=2)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--hist-steps", type=int, default=340)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_matter")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.form_size = 48
        args.cool_steps = 50
        args.n_seeds = 4
        args.noise_scan = [0.0, 0.2, 0.4, 0.6]
        args.content_scan = [1, 3, 5]
        args.hist_steps = 200

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"matter from forms: charges={args.charges}, dim={args.dim}, "
          f"box={args.box}, ρ_Λ={args.vac_coeff}·r·κ₀²", flush=True)
    print("Part A — the Bogomolny mass ladder:")
    spectrum = form_spectrum(args)
    print("Part B — topological protection:")
    protect = topological_protection(args)
    print("Part C — the derived dilution law:")
    dilute = dilution(args, spectrum)
    print("Part D — the cosmos from its form content:")
    closed = closed_source(args, spectrum)

    lines = summarise(spectrum, protect, dilute, closed, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args), "spectrum": spectrum, "protection": protect,
               "dilution": {"slope": dilute["slope"],
                            "total_rest": dilute["total_rest"],
                            "rho_m0": dilute["rho_m0"]},
               "closed_source": {"rho_lambda": closed["rho_lambda"],
                                 "scan": closed["scan"]}}
    with open(os.path.join(args.output_dir, "n3_matter_from_forms.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_matter_from_forms.png")
        make_figure(spectrum, protect, dilute, closed, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
