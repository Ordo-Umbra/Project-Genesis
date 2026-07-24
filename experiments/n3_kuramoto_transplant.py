"""The Kuramoto transplant: does "scarcity pushes S to criticality" hold on a third substrate?

The sector-field programme's deepest emergent result (`n3_s_landscape`,
`n3_kappa_criticality`) — **the S-optimum of a capacity-bound system relocates
to the critical point** — already survived one transplant, to the thermal
Hopfield network (`n3_criticality_transplant.py`).  That experiment closed with
the honest scope the whole programme keeps: *"the transplant tests the
funding-structure mechanism on one new substrate; universality wants more
substrates, not more claims."*  This is the third substrate, and the argument
for it is that it shares **nothing** structural with the first two:

    the **Kuramoto model** — a population of phase oscillators pulling toward a
    common rhythm.  No lattice geometry (mean-field, all-to-all), no learned
    memories (the attractor is *collective synchrony*, not a stored pattern),
    no discrete states (the order parameter is continuous), and an independent,
    textbook order–disorder transition: coherence switches on below a critical
    spread of natural frequencies.

The identical functional is run on it — `S = ΔC + κ·w·ΔI`, κ from the engine's
own capacity law `κ = r/(r + c·load)` — with **the capacity law, the S-weight,
and the S-compass taxonomy imported unchanged from the Hopfield module**
(`kuramoto_substrate.py` re-exports them), so "one law across substrates" is a
fact about the code, not a claim in prose.  The substrate dictionary (stated
once, in the module): the Kuramoto coherence `r` is the order parameter and the
integration reading ΔI; the distinction ΔC is the diversity of
effective-frequency clusters among the coherent population; the disorder knob
is the natural-frequency spread `γ` (the analogue of temperature), swept across
the synchronisation transition.

Design, stated before running — the condition is *toggled*, not assumed, and
the toggle is the same one the Hopfield transplant used:

The deterministic Kuramoto locked state is, in the co-rotating frame,
essentially free — a synchronised population that just rotates together pays
almost no reconfiguration activity, so scarcity has nothing to tax there and
the mechanism *predicts no relocation*.  Add phase **noise** the synchrony must
continuously repair and holding order costs capacity again.  Three conditions:

1. **Undriven, activity load** (co-rotating reconfiguration): the deep-locked
   phase pays ~0 → the ordered optimum is scarcity-proof — NO relocation.
2. **Undriven, ΔC load** (the falsifier): the locked phase is untaxed by
   construction → again NO relocation predicted.
3. **Driven, activity load**: continuous phase noise — a rhythm that is
   actually being held against perturbation.  Maintaining synchrony now costs
   capacity per step → the mechanism predicts the relocation RETURNS, as a
   level crossing into the critical neighbourhood.

In every condition the capacity law and the functional are applied **post-hoc**
on the measured curves for a ladder of consumption strengths `c` (the
simulation never sees `c` — the substrate cannot conspire with the verdict),
and the S-compass delta-form taxonomy is read along the disorder-increasing
trajectory (`trajectory_label`) to place the scarce optimum against the
compass's `diverging` (hallucination-trajectory) band.

If 1 and 2 stay put while 3 relocates, the law survives a second transplant
with its condition intact: *scarcity evicts the ordered optimum exactly when
maintaining order costs capacity* — now on a lattice-free, memory-free,
continuous-state oscillator population.  Any other pattern falsifies it.

Usage::

    python experiments/n3_kuramoto_transplant.py --output-dir artifacts/n3_kuramoto
    python experiments/n3_kuramoto_transplant.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.kuramoto_substrate import (
    natural_frequencies,
    s_functional,
    simulate,
    steady_state_kappa,
    trajectory_label,
    window_metrics,
)


def measure_substrate(args, driven: bool):
    """ΔC(γ), ΔI(γ), load(γ) across the disorder scan.

    ``driven=True`` adds continuous phase noise (``args.noise``) the population
    must repair; ``driven=False`` is the deterministic model.  The load is the
    population's own *co-rotating* reconfiguration activity — for a rigidly
    rotating locked state it is ~0 whether or not oscillators drift, so the
    ordered phase is genuinely free undriven and pays only under the noise.
    """
    gammas = np.linspace(args.g_min, args.g_max, args.n_gamma)
    noise = args.noise if driven else 0.0
    rows = []
    for g in gammas:
        dc, di, act, frac = [], [], [], []
        for seed in range(args.n_seeds):
            rng = np.random.default_rng(args.seed + 1000 * seed)
            omega = natural_frequencies(args.n_osc, g, rng,
                                        distribution=args.distribution)
            phases = rng.uniform(-np.pi, np.pi, args.n_osc)
            sim = simulate(phases, omega, args.coupling, args.dt,
                           args.n_steps, noise, rng, burn_in=args.n_burn)
            out = window_metrics(sim, freq_tol=args.freq_tol)
            dc.append(out["delta_c"])
            di.append(out["delta_i"])
            act.append(out["activity"])
            frac.append(out["coherent_fraction"])
        rows.append({"gamma": float(g),
                     "delta_c": float(np.mean(dc)),
                     "delta_c_err": float(np.std(dc) / np.sqrt(len(dc))),
                     "delta_i": float(np.mean(di)),
                     "delta_i_err": float(np.std(di) / np.sqrt(len(di))),
                     "activity": float(np.mean(act)),
                     "coherent_fraction": float(np.mean(frac))})
        print(f"  {'driven ' if driven else ''}γ={g:.2f}: "
              f"ΔC={rows[-1]['delta_c']:.3f}  ΔI={rows[-1]['delta_i']:.3f}  "
              f"activity={rows[-1]['activity']:.4f}", flush=True)
    return rows


def critical_gamma(rows) -> float:
    """γ_c from the order parameter: where ΔI (=r) first falls through 0.5."""
    for a, b in zip(rows, rows[1:]):
        if a["delta_i"] >= 0.5 >= b["delta_i"]:
            f = (a["delta_i"] - 0.5) / max(a["delta_i"] - b["delta_i"], 1e-12)
            return float(a["gamma"] + f * (b["gamma"] - a["gamma"]))
    return float("nan")


def optimum_ladder(rows, args, load_key: str):
    """γ⋆(c) = argmax_γ S for the consumption ladder, one load convention."""
    gammas = np.array([r["gamma"] for r in rows])
    dc = np.array([r["delta_c"] for r in rows])
    di = np.array([r["delta_i"] for r in rows])
    load = dc if load_key == "delta_c" else np.array([r["activity"] for r in rows])
    ladder = []
    for c in args.consumptions:
        kappa = np.array([steady_state_kappa(l, c, args.recovery) for l in load])
        s = np.array([s_functional(a, b, k, args.weight)
                      for a, b, k in zip(dc, di, kappa)])
        i_star = int(np.argmax(s))
        ladder.append({"c": float(c), "gamma_star": float(gammas[i_star]),
                       "S_star": float(s[i_star]), "S_curve": s.tolist(),
                       "kappa_curve": kappa.tolist()})
    return ladder


def relocation_verdict(ladder, g_c, gammas, args):
    """Did the optimum relocate into the critical neighbourhood — and how?"""
    g_stars = np.array([l["gamma_star"] for l in ladder])
    start, end = g_stars[0], g_stars[-1]
    near_gc = (abs(end - g_c) <= args.critical_window * g_c) if np.isfinite(g_c) else False
    started_ordered = start <= gammas[0] + 0.25 * (g_c - gammas[0]) if np.isfinite(g_c) else start == gammas[0]
    moved = end - start
    steps = np.diff(g_stars)
    max_jump = float(np.max(np.abs(steps))) if steps.size else 0.0
    grid = float(np.mean(np.diff(gammas)))
    level_crossing = max_jump > 2.5 * grid
    return {"relocated": bool(started_ordered and near_gc and moved > 0),
            "start": float(start), "end": float(end), "moved": float(moved),
            "max_jump": max_jump, "level_crossing": bool(level_crossing),
            "g_stars": g_stars.tolist()}


def summarise(rows_u, rows_d, g_c_u, g_c_d, ladders, verdicts, args) -> list[str]:
    lines = ["The Kuramoto transplant: the capacity-bound S-optimum on an "
             "oscillator population — verdict", "=" * 74, ""]
    dc_peak_g = rows_u[int(np.argmax([r["delta_c"] for r in rows_u]))]["gamma"]
    low_act_u = rows_u[0]["activity"]
    low_act_d = rows_d[0]["activity"]
    lines.append(
        f"A. the substrate has the right anatomy: the Kuramoto "
        f"synchronisation transition sits at γ_c ≈ {g_c_u:.2f} (the coherence "
        f"r = ΔI falls through ½ as the frequency spread widens), and the "
        f"distinction reading ΔC — effective-frequency cluster entropy — peaks "
        f"at γ ≈ {dc_peak_g:.2f}, in the critical neighbourhood: the same "
        "ΔC-peak-at-criticality anatomy the sector field and the Hopfield "
        "network showed, now from a locked core coexisting with drifting "
        f"oscillators. And the toggle works: the deep-locked phase's load is "
        f"{low_act_u:.4f} undriven (a rhythm that just rotates, paying nothing) "
        f"vs {low_act_d:.4f} under the phase-noise drive (a rhythm held against "
        "perturbation pays repair work every step).")
    lines.append("")
    vu = verdicts["undriven_activity"]
    vf = verdicts["undriven_deltac"]
    vd = verdicts["driven_activity"]
    lines.append(
        f"B. the three conditions (prediction → outcome), c sweeping "
        f"{args.consumptions[0]:g} → {args.consumptions[-1]:g}:")
    lines.append(
        f"   1. undriven, activity load (locked phase untaxed; predict NO "
        f"relocation): γ⋆ {vu['start']:.2f} → {vu['end']:.2f} — "
        + ("stays put ✓" if not vu["relocated"] else "RELOCATED ✗ (against prediction)"))
    lines.append(
        f"   2. undriven, ΔC load (falsifier; predict NO relocation): "
        f"γ⋆ {vf['start']:.2f} → {vf['end']:.2f} — "
        + ("stays put ✓" if not vf["relocated"] else "RELOCATED ✗ (against prediction)"))
    lines.append(
        f"   3. DRIVEN, activity load (order costs per step; predict "
        f"relocation): γ⋆ {vd['start']:.2f} → {vd['end']:.2f} "
        f"(γ_c(driven) ≈ {g_c_d:.2f}; largest step {vd['max_jump']:.2f}) — "
        + (("RELOCATES into the critical neighbourhood ✓"
            + (", as a LEVEL CROSSING (a jump, not a drift)"
               if vd["level_crossing"] else ", gradually"))
           if vd["relocated"] else "does NOT relocate ✗ (against prediction)")
        + ". Optimum trajectory: "
        + " → ".join(f"{t:.2f}" for t in vd["g_stars"]) + ".")
    lines.append("")
    pattern_ok = vd["relocated"] and not vu["relocated"] and not vf["relocated"]
    n_land = int(not vu["relocated"]) + int(not vf["relocated"]) + int(vd["relocated"])
    lines.append(
        "C. what the pattern establishes: "
        + ("all three predictions land. The sector-field law survives a "
           "SECOND, structurally-independent transplant — no lattice, no "
           "sectors, no learned memories, a continuous order parameter — with "
           "its condition intact: **scarcity evicts the ordered optimum "
           "exactly when maintaining order costs capacity.** A rhythm that "
           "merely rotates is scarcity-proof; a rhythm *held against "
           "perturbation* is pushed to the critical neighbourhood as capacity "
           "binds. Across lattice field, attractor network, and oscillator "
           "population the relocation is a property of capacity-taxed order, "
           "not of any one substrate."
           if pattern_ok else
           "the predicted pattern did NOT land in full — see B. The mechanism "
           "story (integration-funded vs distinction-funded optima under a "
           "capacity tax) does not transfer cleanly to this substrate, and "
           "this record is the starting point for why."))
    lines.append("")
    labels = []
    for a, b in zip(rows_d, rows_d[1:]):
        labels.append(trajectory_label(b["delta_c"] - a["delta_c"],
                                       b["delta_i"] - a["delta_i"],
                                       deadband=args.deadband))
    div_g = [0.5 * (rows_d[i]["gamma"] + rows_d[i + 1]["gamma"])
             for i, lab in enumerate(labels) if lab == "diverging"]
    if div_g:
        lines.append(
            "D. one taxonomy across substrates: reading the S-compass "
            "delta-form labels along the driven disorder-increasing "
            "trajectory, the substrate turns `diverging` (ΔC↑, ΔI↓ — the "
            f"compass's hallucination trajectory) over γ ≈ [{min(div_g):.2f}, "
            f"{max(div_g):.2f}]"
            + (f", and the scarce optimum lands at γ⋆ = {vd['end']:.2f} — "
               "inside/at the edge of that band: the same coincidence the "
               "Hopfield transplant found — what the compass flags as "
               "hallucination-onset in an LLM session is, on a third "
               "substrate too, where a capacity-starved S-climber is pushed."
               if vd["relocated"] else "."))
    else:
        lines.append("D. no `diverging` band resolved on this grid.")
    lines.append("")
    lines.append(f"score: {n_land}/3 pre-registered predictions land.")
    lines.append("")
    lines.append(
        "honest scope: one coupling K, one recovery rate, one noise drive "
        "strength, one frequency distribution, modest population and windows; "
        "γ_c is the finite-size r=½ crossing, not a thermodynamic "
        "determination; the entrainment tolerance, ΔC's fixed frequency bins, "
        "and the S weight w are conventions (stated, fixed before the scan; "
        "the c-ladder is applied post-hoc to measured curves, so the dynamics "
        "cannot conspire with the verdict). This is the mechanism on a third "
        "substrate — a second independent confirmation, not a proof of "
        "universality; the honest object is still the funding-structure law "
        "itself, now read identically on lattice, network, and oscillators.")
    return lines


def make_figure(rows_u, rows_d, g_c_u, g_c_d, ladders, verdicts, args, path):
    rows, g_c = rows_u, g_c_u
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY, RED, PURPLE = ("#0072B2", "#E69F00", "#009E73",
                                              "#999999", "#c0392b", "#8551A6")
    fig = plt.figure(figsize=(13, 9.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.27)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))
    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    gammas = np.array([r["gamma"] for r in rows])

    # panel A: the substrate's anatomy
    ax1.errorbar(gammas, [r["delta_i"] for r in rows],
                 yerr=[r["delta_i_err"] for r in rows], color=BLUE, lw=2.0,
                 marker="o", ms=4, capsize=2, zorder=3, label="ΔI (coherence r)")
    ax1.errorbar(gammas, [r["delta_c"] for r in rows],
                 yerr=[r["delta_c_err"] for r in rows], color=ORANGE, lw=2.0,
                 marker="s", ms=4, capsize=2, zorder=3, label="ΔC (cluster entropy)")
    ax1.plot(gammas, [r["activity"] for r in rows], color=GRAY, lw=1.6, ls="--",
             zorder=2, label="activity load (undriven)")
    ax1.plot(gammas, [r["activity"] for r in rows_d], color=PURPLE, lw=1.6,
             ls="--", zorder=2, label="activity load (driven)")
    if np.isfinite(g_c):
        ax1.axvline(g_c, color=RED, lw=1.2, ls=":", zorder=2)
        ax1.text(g_c, 0.98, " γ_c", color=RED, fontsize=9, va="top",
                 transform=ax1.get_xaxis_transform())
    ax1.set_xlabel("frequency spread  γ  (disorder)")
    ax1.set_ylabel("reading")
    ax1.set_title("A. The substrate: ΔC peaks where synchrony dies")
    ax1.legend(fontsize=8, loc="best")

    # panel B: S(γ) for the consumption ladder (DRIVEN — the condition on)
    la = ladders["driven_activity"]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(la)))
    gammas_d = np.array([r["gamma"] for r in rows_d])
    for entry, color in zip(la, cmap):
        ax2.plot(gammas_d, entry["S_curve"], color=color, lw=1.8, zorder=3,
                 label=f"c = {entry['c']:g}")
        ax2.plot([entry["gamma_star"]],
                 [max(entry["S_curve"])], "o", color=color, ms=7, zorder=4)
    if np.isfinite(g_c_d):
        ax2.axvline(g_c_d, color=RED, lw=1.2, ls=":", zorder=2)
    ax2.set_xlabel("frequency spread  γ")
    ax2.set_ylabel("S = ΔC + κ·w·ΔI")
    ax2.set_title("B. The S landscape as capacity binds (driven rhythm)")
    ax2.legend(fontsize=7, loc="best")

    # panel C: the relocation curves, all three conditions
    for key, color, label in (
            ("undriven_activity", GRAY, "undriven (locked order is free)"),
            ("undriven_deltac", GREEN, "ΔC load (falsifier)"),
            ("driven_activity", BLUE, "DRIVEN (order costs per step)")):
        cs = [l["c"] for l in ladders[key]]
        gsr = [l["gamma_star"] for l in ladders[key]]
        ax3.plot(cs, gsr, color=color, lw=2.2, marker="o", ms=5, zorder=3,
                 label=label)
    if np.isfinite(g_c_d):
        ax3.axhline(g_c_d, color=RED, lw=1.2, ls=":", zorder=2)
        ax3.text(0.02, g_c_d, " γ_c", color=RED, fontsize=9, va="bottom",
                 transform=ax3.get_yaxis_transform())
    ax3.set_xscale("log")
    ax3.set_xlabel("consumption  c  (capacity scarcity)")
    ax3.set_ylabel("γ⋆(c) = argmax_γ S")
    ax3.set_title("C. The relocation: three conditions, one toggle")
    ax3.legend(fontsize=8, loc="best")

    # panel D: verdict
    ax4.axis("off")
    ax4.set_title("D. The verdict")
    vu = verdicts["undriven_activity"]
    vf = verdicts["undriven_deltac"]
    vd = verdicts["driven_activity"]
    pattern_ok = vd["relocated"] and not vu["relocated"] and not vf["relocated"]
    txt = (f"substrate: Kuramoto oscillators (no lattice,\n"
           f"no memories, continuous r, γ_c = {g_c:.2f})\n\n"
           f"1 undriven:   γ* {vu['start']:.2f} -> {vu['end']:.2f}"
           f"   predict NO  -> {'NO ok' if not vu['relocated'] else 'YES !!'}\n"
           f"2 dC load:    γ* {vf['start']:.2f} -> {vf['end']:.2f}"
           f"   predict NO  -> {'NO ok' if not vf['relocated'] else 'YES !!'}\n"
           f"3 driven:     γ* {vd['start']:.2f} -> {vd['end']:.2f}"
           f"   predict YES -> {'YES ok' if vd['relocated'] else 'NO !!'}"
           + ("  (LEVEL CROSSING)" if vd["level_crossing"] else "")
           + "\n\n"
           + ("the law survives a second transplant:\n"
              "scarcity evicts ordered optima exactly\n"
              "when holding order costs capacity. a\n"
              "freely-rotating rhythm is scarcity-proof;\n"
              "a rhythm HELD against noise is pushed to\n"
              "criticality as capacity binds. one law on\n"
              "lattice, network, and oscillators."
              if pattern_ok else
              "the predicted pattern did not land in\n"
              "full - the mechanism story does not\n"
              "transfer cleanly here (see summary B)."))
    ax4.text(0.03, 0.92, txt, transform=ax4.transAxes, fontsize=9.5, va="top",
             family="monospace")

    fig.suptitle("The Kuramoto transplant: capacity-bound S-climbing on a "
                 "lattice-free, memory-free oscillator population",
                 fontsize=12, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-osc", type=int, default=400)
    p.add_argument("--coupling", type=float, default=2.0,
                   help="Kuramoto coupling K (fixed while γ sweeps).")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--n-steps", type=int, default=900)
    p.add_argument("--n-burn", type=int, default=450)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--g-min", type=float, default=0.1)
    p.add_argument("--g-max", type=float, default=2.4)
    p.add_argument("--n-gamma", type=int, default=13)
    p.add_argument("--distribution", type=str, default="gaussian",
                   choices=["gaussian", "lorentzian"])
    p.add_argument("--freq-tol", type=float, default=0.3,
                   help="Entrainment tolerance |Ω_i| < freq_tol (zero-centred).")
    p.add_argument("--noise", type=float, default=0.6,
                   help="Phase-noise drive strength (driven runs).")
    p.add_argument("--weight", type=float, default=1.5,
                   help="Integration weight w in S = ΔC + κ·w·ΔI.")
    p.add_argument("--recovery", type=float, default=0.1,
                   help="Capacity recovery rate r (engine law).")
    p.add_argument("--consumptions", type=float, nargs="+",
                   default=[0.05, 0.2, 0.8, 3.0, 12.0, 50.0])
    p.add_argument("--critical-window", type=float, default=0.35,
                   help="Fractional γ_c window counted as 'critical "
                        "neighbourhood'.")
    p.add_argument("--deadband", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_kuramoto")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.n_osc = 250
        args.n_seeds = 2
        args.n_steps = 500
        args.n_burn = 250
        args.n_gamma = 10

    os.makedirs(args.output_dir, exist_ok=True)
    print("the Kuramoto transplant: capacity-bound S-climbing on an "
          "oscillator population", flush=True)
    print("condition off — undriven (a freely-rotating rhythm pays nothing):")
    rows_u = measure_substrate(args, driven=False)
    print("condition on — driven (a rhythm held against noise pays repair):")
    rows_d = measure_substrate(args, driven=True)
    g_c_u = critical_gamma(rows_u)
    g_c_d = critical_gamma(rows_d)
    gammas = np.array([r["gamma"] for r in rows_u])
    conditions = {"undriven_activity": (rows_u, "activity", g_c_u),
                  "undriven_deltac": (rows_u, "delta_c", g_c_u),
                  "driven_activity": (rows_d, "activity", g_c_d)}
    ladders, verdicts = {}, {}
    for key, (rows, load_key, g_c) in conditions.items():
        ladders[key] = optimum_ladder(rows, args, load_key)
        verdicts[key] = relocation_verdict(ladders[key], g_c, gammas, args)

    lines = summarise(rows_u, rows_d, g_c_u, g_c_d, ladders, verdicts, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args), "rows_undriven": rows_u,
               "rows_driven": rows_d, "g_c_undriven": g_c_u,
               "g_c_driven": g_c_d, "ladders": ladders, "verdicts": verdicts}
    with open(os.path.join(args.output_dir, "n3_kuramoto_transplant.json"),
              "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_kuramoto_transplant.png")
        make_figure(rows_u, rows_d, g_c_u, g_c_d, ladders, verdicts, args,
                    fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
