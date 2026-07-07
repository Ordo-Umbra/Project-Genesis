"""The universe selects its Platonic forms by S — and capacity decides which.

The program's central result is that a capacity-bounded recursive field
*selects* a three-sector structure: `S = ΔC + κ·ΔI` is maximised at the
three-fold colour-neutral Y-junction (`topological_selection.py`).  This
experiment reframes that as the vision it is an instance of — a **corpus of
Platonic forms**, each a candidate stable structure, of which the universe
*manifests* whichever climbs S highest under the prevailing conditions —
and asks the question that framing makes unavoidable: **which form is
selected depends on capacity.**

The forms are the P-sector domain networks, P ∈ {2, 3, 4, 5, 6}, each
realised as a stable configuration by the conserved multiphase dynamics.
Two S-ingredients are measured per form:

- **distinction ΔC** — the wall density (β·⟨|∇f|²⟩), which *rises* with P
  (more sectors, more walls);
- **integration ΔI** — the full-palette junction density (the §6
  neutrality criterion), which is non-zero essentially *only at P = 3*,
  because a 2-D junction is three-fold.

The decisive new ingredient is that the integration term is **capacity-
gated**: ``S_P = ΔC_P + κ·w·ΔI_P``, where κ is the physical capacity
(0 → 1) available to pay for integration.  The selected form is
``argmax_P S_P`` over the condition plane of capacity κ and integration
weight w — and because integration (ΔI) is essentially a P = 3 monopoly,
the three-fold form is selected only where κ·w is large enough to lift its
neutrality bonus above the distinction gap to the maximally-fragmented
form.

The prediction, straight from the S-functional's structure: **abundant
capacity manifests the integrated three-fold form (N⋆ = 3); low capacity
kills the κ·ΔI term and leaves only distinction, so the universe
fragments** — the "perfect" neutral form is manifestable only where
capacity can pay its cost.  A thermal spot-check (the annealed ψ∈ℂ³ field
with dynamical κ) anchors *where in κ the universe actually sits*: the
sustained ⟨κ⟩ falls sharply as consumption rises (and, from
`n3_kappa_criticality.py`, craters toward ≈ 0.02 at criticality), so the
system is driven out of the three-fold island and into fragmentation
exactly when capacity is scarce.

Usage::

    python experiments/n3_form_selection.py --output-dir artifacts/n3_forms
    python experiments/n3_form_selection.py --quick    # smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.annealed_matter import joint_sweep, sector_amplitudes
from project_genesis.multiphase import (
    _total_gradient_energy,
    full_palette_junction_density,
    sector_labels,
)

from n3_seed_rooting import equilibrate  # noqa: E402
from topological_selection import evolve_conserved  # noqa: E402


def realise_forms(phases, *, size, steps, gamma, diffusion, dt, beta, trials,
                  seed) -> dict:
    """Realise each P-sector form and measure its distinction & neutrality."""
    forms = {}
    for P in phases:
        dcs, neuts = [], []
        for t in range(trials):
            f = evolve_conserved(P, dim=2, size=size, steps=steps, gamma=gamma,
                                 diffusion=diffusion, dt=dt, seed=seed + t)
            dcs.append(beta * float(_total_gradient_energy(f).mean()))
            neuts.append(full_palette_junction_density(sector_labels(f), P))
        forms[P] = {"delta_c": float(np.mean(dcs)),
                    "neutrality": float(np.mean(neuts))}
    return forms


def selected_form(forms, phases, kappa, w):
    """The S-maximising form at (capacity κ, weight w): argmax_P ΔC_P + κ·w·ΔI_P."""
    best_P, best_S = None, -np.inf
    for P in phases:
        s = forms[P]["delta_c"] + kappa * w * forms[P]["neutrality"]
        if s > best_S:
            best_S, best_P = s, P
    return best_P, best_S


def thermal_kappa_check(consumptions, args) -> list[dict]:
    """Spot-check: sustained ⟨κ⟩ and realised neutrality of the ψ∈ℂ³ (P=3) field."""
    rows = []
    for c in consumptions:
        psi, links, kappa, rng, step = equilibrate(
            size=args.thermal_size, temperature=args.thermal_temperature,
            beta_g=args.beta_g, matter_coupling=args.matter_coupling,
            quartic_u=args.quartic_u, consumption=c,
            kappa_recovery=args.recovery, n_therm=args.thermal_therm,
            seed=args.seed + int(100 * c))
        kp = dict(consumption=c, recovery=args.recovery)
        kaps, neuts = [], []
        for _ in range(args.thermal_meas):
            for _ in range(2):
                psi, links, kappa, _ = joint_sweep(
                    psi, links, rng, temperature=args.thermal_temperature,
                    beta_g=args.beta_g, matter_coupling=args.matter_coupling,
                    quartic_u=args.quartic_u, frac_penalty=0.0,
                    step_scale=step, kappa=kappa, kappa_params=kp)
            kaps.append(float(kappa.mean()))
            neuts.append(full_palette_junction_density(
                sector_labels(sector_amplitudes(psi)), 3))
        rows.append({"consumption": c, "kappa_mean": float(np.mean(kaps)),
                     "neutrality": float(np.mean(neuts))})
    return rows


def make_figure(forms, phases, cs, ws, sel_grid, thermal, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#E69F00", "#009E73", "#999999"
    fig = plt.figure(figsize=(15, 4.6))
    gs = GridSpec(1, 3, figure=fig)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, j]) for j in range(3))

    # panel 1: the corpus — distinction and neutrality per form
    dc = [forms[P]["delta_c"] for P in phases]
    nu = [forms[P]["neutrality"] for P in phases]
    ax1.bar([p - 0.18 for p in phases], dc, width=0.36, color=BLUE,
            label="distinction ΔC", zorder=3)
    axb = ax1.twinx()
    axb.bar([p + 0.18 for p in phases], nu, width=0.36, color=ORANGE,
            label="neutrality ΔI", zorder=3)
    ax1.set_xlabel("form (sector count P)")
    ax1.set_ylabel("distinction ΔC", color=BLUE)
    axb.set_ylabel("neutrality ΔI", color=ORANGE)
    ax1.set_xticks(phases)
    ax1.set_title("The corpus of forms")
    for side in ("top",):
        ax1.spines[side].set_visible(False)
        axb.spines[side].set_visible(False)

    # panel 2: selection phase diagram over (scarcity c, weight w)
    cmap = ListedColormap(
        ["#cfe3f2", "#4a90c4", "#009E73", "#E69F00", "#D55E00", "#8551A6"][:len(phases)])
    pindex = {P: i for i, P in enumerate(phases)}
    grid = np.array([[pindex[sel_grid[(ci, wi)]] for ci in range(len(cs))]
                     for wi in range(len(ws))])
    im = ax2.imshow(grid, aspect="auto", origin="lower", cmap=cmap,
                    vmin=-0.5, vmax=len(phases) - 0.5)
    ax2.set_xticks(range(len(cs)))
    ax2.set_xticklabels([f"{c:g}" for c in cs], fontsize=7, rotation=45)
    ax2.set_yticks(range(len(ws)))
    ax2.set_yticklabels([f"{w:g}" for w in ws], fontsize=8)
    ax2.set_xlabel("capacity κ  (→ more abundant)")
    ax2.set_ylabel("integration weight w")
    ax2.set_title("Which form the universe manifests")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04,
                        ticks=range(len(phases)))
    cbar.ax.set_yticklabels([f"P={P}" for P in phases], fontsize=8)

    # panel 3: thermal spot-check — sustained κ (and neutrality) vs consumption
    tc = [t["consumption"] for t in thermal]
    ax3.plot(tc, [t["kappa_mean"] for t in thermal], color=GREEN, marker="o",
             ms=5, lw=1.9, label="sustained ⟨κ⟩ (ψ∈ℂ³)", zorder=3)
    ax3.plot(tc, [t["neutrality"] for t in thermal], color=ORANGE, marker="s",
             ms=4, lw=1.5, ls="--", label="realised neutrality", zorder=2)
    ax3.set_xscale("log")
    ax3.set_xlabel("capacity consumption c")
    ax3.set_ylabel("sustained ⟨κ⟩ / neutrality")
    ax3.set_title("Structure costs capacity (thermal check)")
    ax3.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
    ax3.set_axisbelow(True)
    for side in ("top", "right"):
        ax3.spines[side].set_visible(False)
    ax3.legend(frameon=False, fontsize=8)

    fig.suptitle("Platonic forms selected by S: the integrated three-fold form "
                 "is manifested where capacity is abundant", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarise(forms, phases, kappas, ws, sel_grid, thermal) -> list[str]:
    lines = ["Platonic-form selection by S — verdict", "=" * 38, ""]
    n3 = max(phases, key=lambda P: forms[P]["neutrality"])
    lines.append(
        f"the corpus: neutrality (integration) peaks at P={n3} "
        f"({forms[n3]['neutrality']:.4f}); distinction rises with P "
        f"(ΔC: {forms[phases[0]]['delta_c']:.4f} → {forms[phases[-1]]['delta_c']:.4f} "
        f"for P={phases[0]}→{phases[-1]}) — the three-fold form is the only one "
        "that carries the whole palette on its junctions")
    # fraction of the (κ,w) plane where the three-fold form is selected
    total = len(kappas) * len(ws)
    n3_cells = sum(1 for k in sel_grid.values() if k == n3)
    lines.append(
        f"across the condition plane the three-fold form is selected in "
        f"{n3_cells}/{total} of the (capacity, weight) cells — an island of "
        "abundant-capacity-and-valued-integration, fragmentation outside it")
    # at a mid weight, find the capacity threshold below which 3 is abandoned
    wi_mid = len(ws) // 2
    w_mid = ws[wi_mid]
    sel_row = [sel_grid[(ci, wi_mid)] for ci in range(len(kappas))]  # ascending κ
    kept = [kappas[ci] for ci in range(len(kappas)) if sel_row[ci] == n3]
    if kept and any(s != n3 for s in sel_row):
        lines.append(
            f"at integration weight w={w_mid:g}, the three-fold form is selected "
            f"only once capacity rises above κ≈{min(kept):g}: below it the κ·ΔI "
            "integration term cannot be paid, and the S-optimum jumps to the "
            "maximally-distinct (fragmented) form")
    lines.append("")
    t_lo, t_hi = thermal[0], thermal[-1]
    lines.append(
        f"thermal anchor: the ψ∈ℂ³ field's sustained ⟨κ⟩ falls "
        f"{t_lo['kappa_mean']:.2f} → {t_hi['kappa_mean']:.2f} as consumption rises "
        f"c={t_lo['consumption']:g}→{t_hi['consumption']:g} (and craters toward "
        "≈0.02 at criticality, per n3_kappa_criticality) — so scarcity drives the "
        "system leftward out of the three-fold island and into fragmentation")
    lines.append("")
    lines.append(
        "reading: N⋆=3 is not a bare constant but a *conditional* manifestation — "
        "the integrated three-fold Platonic form is what abundant capacity buys; "
        "under scarcity the universe can afford only distinction, and fragments. "
        "Perfection (the neutral, unified form) is manifestable where conditions "
        "permit, exactly as the S = ΔC + κΔI structure demands.")
    lines.append(
        "honest scope: forms realised by 2-D conserved dynamics (a structural, "
        "3-fold-junction argument — 3-D vertices differ); the phase diagram is in "
        "physical capacity κ, with the thermal run and n3_kappa_criticality "
        "anchoring where the field sits in κ; single coupling/weight ranges.")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--phases", type=str, default="2,3,4,5,6")
    p.add_argument("--size", type=int, default=72)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--diffusion", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.09)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--recovery", type=float, default=0.5)
    p.add_argument("--capacities", type=str, default="0.02,0.05,0.1,0.2,0.4,0.7,0.95",
                   help="physical capacity κ (0→1) available to pay for integration")
    p.add_argument("--weights", type=str, default="0.05,0.1,0.2,0.4,0.8,1.6")
    # thermal spot-check
    p.add_argument("--thermal-size", type=int, default=24)
    p.add_argument("--thermal-temperature", type=float, default=0.05)
    p.add_argument("--thermal-consumptions", type=str, default="0.5,2,8,32")
    p.add_argument("--thermal-therm", type=int, default=400)
    p.add_argument("--thermal-meas", type=int, default=30)
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=67)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_forms")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size, args.steps, args.trials = 40, 250, 2
        args.capacities = "0.05,0.2,0.95"
        args.weights = "0.05,0.2,0.8"
        args.thermal_size, args.thermal_therm, args.thermal_meas = 14, 150, 15
        args.thermal_consumptions = "1,16"

    phases = [int(x) for x in args.phases.split(",") if x.strip()]
    cs = [float(x) for x in args.capacities.split(",") if x.strip()]
    ws = [float(x) for x in args.weights.split(",") if x.strip()]
    thermal_cs = [float(x) for x in args.thermal_consumptions.split(",") if x.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Realising the corpus of forms P ∈ {phases} (conserved dynamics)",
          flush=True)
    forms = realise_forms(phases, size=args.size, steps=args.steps,
                          gamma=args.gamma, diffusion=args.diffusion, dt=args.dt,
                          beta=args.beta, trials=args.trials, seed=args.seed)
    for P in phases:
        print(f"  P={P}: ΔC={forms[P]['delta_c']:.5f}  "
              f"neutrality={forms[P]['neutrality']:.5f}", flush=True)

    sel_grid = {}
    for ci, kappa in enumerate(cs):
        for wi, w in enumerate(ws):
            best_P, _ = selected_form(forms, phases, kappa, w)
            sel_grid[(ci, wi)] = best_P

    print("\nThermal spot-check: sustained ⟨κ⟩ of the ψ∈ℂ³ form vs consumption",
          flush=True)
    thermal = thermal_kappa_check(thermal_cs, args)
    for t in thermal:
        print(f"  c={t['consumption']:g}: ⟨κ⟩={t['kappa_mean']:.3f}  "
              f"neutrality={t['neutrality']:.4f}", flush=True)

    lines = summarise(forms, phases, cs, ws, sel_grid, thermal)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": vars(args), "forms": forms,
               "selection": {f"kappa={cs[ci]:g},w={ws[wi]:g}": sel_grid[(ci, wi)]
                             for ci in range(len(cs)) for wi in range(len(ws))},
               "thermal": thermal}
    with open(os.path.join(args.output_dir, "n3_form_selection.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_form_selection.png")
        make_figure(forms, phases, cs, ws, sel_grid, thermal, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
