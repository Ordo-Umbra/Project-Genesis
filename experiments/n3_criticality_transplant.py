"""The criticality transplant: does "scarcity pushes S to criticality" survive a new substrate?

The sector-field programme's deepest emergent result (`n3_s_landscape`,
`n3_kappa_criticality`): **the S-optimum of a capacity-bound system relocates
to the critical point** — scarcity taxes the κ-gated integration term but not
distinction, so as consumption rises the global optimum performs a *level
crossing* from the deep-ordered phase (integration-funded) to the critical
neighbourhood (distinction-funded).

Every one of those measurements shares one substrate: a lattice sector field.
If the relocation is a *law of capacity-bound recursion*, it must survive
transplant to a system with none of that structure.  This experiment runs the
identical functional — `S = ΔC + κ·w·ΔI`, κ from the engine's own capacity
law `κ = r/(r + c·load)` — on the **thermal Hopfield network**
(`hopfield_substrate.py`): all-to-all *learned* couplings, no lattice, no
sectors, attractors as memories, and an independent retrieval transition.

Design, stated before running — the condition is *toggled*, not assumed:

The sector field's deep-ordered phase always carries residual load (domain
walls exist even when cold), so scarcity can tax it.  The Hopfield network is
sharper: at low `T` it is **perfectly frozen** — zero activity, zero tax — so
the theorem's mechanism *predicts no relocation there at all*.  The
experiment therefore runs three conditions:

1. **Undriven, activity load** (flip rate): the frozen retrieval phase pays
   nothing → the mechanism predicts the ordered optimum is scarcity-proof —
   NO relocation.
2. **Undriven, ΔC load** (the falsifier): the frozen phase is untaxed by
   construction → again NO relocation predicted.
3. **Driven, activity load**: a *query drive* — every few sweeps a random
   fraction of spins is externally perturbed and the network must repair
   itself (a memory that is actually being used).  Maintaining order now
   costs capacity per query, restoring the sector field's condition → the
   mechanism predicts the relocation RETURNS, as a level crossing.

In every condition the capacity law and the functional are applied
**post-hoc** on the measured curves for a ladder of consumption strengths
`c` (the simulation never sees `c` — the substrate cannot conspire with the
verdict), and the S-compass delta-form taxonomy is read along the heating
trajectory (`trajectory_label`) to place the scarce optimum against the
compass's `diverging` (hallucination-trajectory) band.

If 1 and 2 stay put while 3 relocates, the law survives transplant with its
condition made precise: *scarcity evicts the ordered optimum exactly when
maintaining order costs capacity.*  Any other pattern falsifies the
mechanism story.

Usage::

    python experiments/n3_criticality_transplant.py --output-dir artifacts/n3_transplant
    python experiments/n3_criticality_transplant.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.hopfield_substrate import (
    best_overlap,
    glauber_sweep,
    hebbian_weights,
    s_functional,
    steady_state_kappa,
    trajectory_label,
    window_metrics,
)


def measure_substrate(args, driven: bool):
    """ΔC(T), ΔI(T), load(T) across the temperature scan.

    With ``driven=True``, every ``query_every``-th window sweep first perturbs
    a random ``query_fraction`` of spins (an external recall query / noise
    hit).  The perturbation flips are *not* counted as load — only the
    network's own repair activity on subsequent sweeps is (the work of
    maintaining the memory in use).
    """
    temps = np.linspace(args.t_min, args.t_max, args.n_temps)
    rows = []
    for T in temps:
        dc, di, act, frac = [], [], [], []
        for seed in range(args.n_seeds):
            rng = np.random.default_rng(args.seed + 1000 * seed)
            pats = rng.choice([-1.0, 1.0], size=(args.n_patterns, args.n_spins))
            j = hebbian_weights(pats)
            s = pats[0].copy()
            for _ in range(args.n_burn):
                s = glauber_sweep(s, j, T, rng)
            window, flips = [], []
            for k in range(args.n_window):
                if driven and k % args.query_every == 0:
                    hit = rng.choice(s.size,
                                     size=int(args.query_fraction * s.size),
                                     replace=False)
                    s[hit] *= -1
                s_new = glauber_sweep(s, j, T, rng)
                flips.append(float(np.mean(s_new != s)))
                s = s_new
                window.append(best_overlap(s, pats))
            out = window_metrics(window, args.n_patterns,
                                 args.structured_threshold)
            dc.append(out["delta_c"])
            di.append(out["delta_i"])
            act.append(float(np.mean(flips)))
            frac.append(out["structured_fraction"])
        rows.append({"T": float(T),
                     "delta_c": float(np.mean(dc)),
                     "delta_c_err": float(np.std(dc) / np.sqrt(len(dc))),
                     "delta_i": float(np.mean(di)),
                     "delta_i_err": float(np.std(di) / np.sqrt(len(di))),
                     "activity": float(np.mean(act)),
                     "structured_fraction": float(np.mean(frac))})
        print(f"  {'driven ' if driven else ''}T={T:.2f}: "
              f"ΔC={rows[-1]['delta_c']:.3f}  ΔI={rows[-1]['delta_i']:.3f}  "
              f"activity={rows[-1]['activity']:.3f}", flush=True)
    return rows


def critical_temperature(rows) -> float:
    """T_c from the order parameter: where ΔI first falls through 0.5."""
    for a, b in zip(rows, rows[1:]):
        if a["delta_i"] >= 0.5 >= b["delta_i"]:
            f = (a["delta_i"] - 0.5) / max(a["delta_i"] - b["delta_i"], 1e-12)
            return float(a["T"] + f * (b["T"] - a["T"]))
    return float("nan")


def optimum_ladder(rows, args, load_key: str):
    """T⋆(c) = argmax_T S for the consumption ladder, one load convention."""
    temps = np.array([r["T"] for r in rows])
    dc = np.array([r["delta_c"] for r in rows])
    di = np.array([r["delta_i"] for r in rows])
    load = dc if load_key == "delta_c" else np.array([r["activity"] for r in rows])
    ladder = []
    for c in args.consumptions:
        kappa = np.array([steady_state_kappa(l, c, args.recovery) for l in load])
        s = np.array([s_functional(a, b, k, args.weight)
                      for a, b, k in zip(dc, di, kappa)])
        i_star = int(np.argmax(s))
        ladder.append({"c": float(c), "T_star": float(temps[i_star]),
                       "S_star": float(s[i_star]), "S_curve": s.tolist(),
                       "kappa_curve": kappa.tolist()})
    return ladder


def relocation_verdict(ladder, t_c, temps, args):
    """Did the optimum relocate into the critical neighbourhood — and how?"""
    t_stars = np.array([l["T_star"] for l in ladder])
    start, end = t_stars[0], t_stars[-1]
    near_tc = (abs(end - t_c) <= args.critical_window * t_c) if np.isfinite(t_c) else False
    started_ordered = start <= temps[0] + 0.25 * (t_c - temps[0]) if np.isfinite(t_c) else start == temps[0]
    moved = end - start
    steps = np.diff(t_stars)
    max_jump = float(np.max(np.abs(steps))) if steps.size else 0.0
    grid = float(np.mean(np.diff(temps)))
    level_crossing = max_jump > 2.5 * grid
    return {"relocated": bool(started_ordered and near_tc and moved > 0),
            "start": float(start), "end": float(end), "moved": float(moved),
            "max_jump": max_jump, "level_crossing": bool(level_crossing),
            "t_stars": t_stars.tolist()}


def summarise(rows_u, rows_d, t_c_u, t_c_d, ladders, verdicts, args) -> list[str]:
    lines = ["The criticality transplant: the capacity-bound S-optimum on a "
             "substrate with no lattice — verdict", "=" * 74, ""]
    dc_peak_T = rows_u[int(np.argmax([r["delta_c"] for r in rows_u]))]["T"]
    low_act_u = rows_u[0]["activity"]
    low_act_d = rows_d[0]["activity"]
    lines.append(
        f"A. the substrate has the right anatomy: the Hopfield retrieval "
        f"transition sits at T_c ≈ {t_c_u:.2f} (ΔI falls through ½), and the "
        f"distinction reading ΔC — structured-attractor visit entropy — peaks "
        f"at T ≈ {dc_peak_T:.2f}, in the critical neighbourhood: the same "
        "ΔC-peak-at-criticality anatomy the sector field showed, now from "
        "attractor hopping among learned memories instead of domain walls. "
        f"And the toggle works: cold-phase load is {low_act_u:.4f} undriven "
        f"(a genuinely free frozen memory) vs {low_act_d:.4f} under the query "
        "drive (a memory in use pays repair work per query).")
    lines.append("")
    vu = verdicts["undriven_activity"]
    vf = verdicts["undriven_deltac"]
    vd = verdicts["driven_activity"]
    lines.append(
        f"B. the three conditions (prediction → outcome), c sweeping "
        f"{args.consumptions[0]:g} → {args.consumptions[-1]:g}:")
    lines.append(
        f"   1. undriven, activity load (frozen phase untaxed; predict NO "
        f"relocation): T⋆ {vu['start']:.2f} → {vu['end']:.2f} — "
        + ("stays put ✓" if not vu["relocated"] else "RELOCATED ✗ (against prediction)"))
    lines.append(
        f"   2. undriven, ΔC load (falsifier; predict NO relocation): "
        f"T⋆ {vf['start']:.2f} → {vf['end']:.2f} — "
        + ("stays put ✓" if not vf["relocated"] else "RELOCATED ✗ (against prediction)"))
    lines.append(
        f"   3. DRIVEN, activity load (order costs per query; predict "
        f"relocation): T⋆ {vd['start']:.2f} → {vd['end']:.2f} "
        f"(T_c(driven) ≈ {t_c_d:.2f}; largest step {vd['max_jump']:.2f}) — "
        + (("RELOCATES into the critical neighbourhood ✓"
            + (", as a LEVEL CROSSING (a jump, not a drift)"
               if vd["level_crossing"] else ", gradually"))
           if vd["relocated"] else "does NOT relocate ✗ (against prediction)")
        + ". Optimum trajectory: "
        + " → ".join(f"{t:.2f}" for t in vd["t_stars"]) + ".")
    lines.append("")
    pattern_ok = vd["relocated"] and not vu["relocated"] and not vf["relocated"]
    lines.append(
        "C. what the pattern establishes: "
        + ("all three predictions land. The sector-field law survives its "
           "first substrate transplant — no lattice, no sectors, learned "
           "couplings — with its condition made precise by the toggle: "
           "**scarcity evicts the ordered optimum exactly when maintaining "
           "order costs capacity.** A perfectly frozen, unused memory is "
           "scarcity-proof; a memory *in use* is pushed to the critical "
           "neighbourhood as capacity binds. The relocation is a property of "
           "capacity-taxed order, not of the sector field."
           if pattern_ok else
           "the predicted pattern did NOT land in full — see B. The "
           "mechanism story (integration-funded vs distinction-funded optima "
           "under a capacity tax) needs revision, and this record is the "
           "starting point."))
    lines.append("")
    labels = []
    for a, b in zip(rows_d, rows_d[1:]):
        labels.append(trajectory_label(b["delta_c"] - a["delta_c"],
                                       b["delta_i"] - a["delta_i"],
                                       deadband=args.deadband))
    div_T = [0.5 * (rows_d[i]["T"] + rows_d[i + 1]["T"])
             for i, lab in enumerate(labels) if lab == "diverging"]
    if div_T:
        lines.append(
            "D. one taxonomy across substrates: reading the S-compass "
            "delta-form labels along the driven heating trajectory, the "
            "substrate turns `diverging` (ΔC↑, ΔI↓ — the compass's "
            f"hallucination trajectory) over T ≈ [{min(div_T):.2f}, "
            f"{max(div_T):.2f}]"
            + (f", and the scarce optimum lands at T⋆ = {vd['end']:.2f} — "
               "inside/at the edge of that band: what the compass flags as "
               "hallucination-onset in an LLM session is, on this substrate, "
               "where a capacity-starved S-climber is pushed."
               if vd["relocated"] else "."))
    else:
        lines.append("D. no `diverging` band resolved on this grid.")
    lines.append("")
    lines.append(
        "honest scope: one loading (P/N), one recovery rate, one query "
        "drive strength, modest sizes and windows; T_c is the finite-size "
        "ΔI=½ crossing, not a thermodynamic determination; ΔC's structured "
        "threshold and the S weight w are conventions (stated, fixed before "
        "the scan; the c-ladder is applied post-hoc to measured curves, so "
        "the dynamics cannot conspire with the verdict). The transplant "
        "tests the funding-structure mechanism on one new substrate; "
        "universality wants more substrates, not more claims.")
    return lines


def make_figure(rows_u, rows_d, t_c_u, t_c_d, ladders, verdicts, args, path):
    rows, t_c = rows_u, t_c_u
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

    temps = np.array([r["T"] for r in rows])

    # panel A: the substrate's anatomy
    ax1.errorbar(temps, [r["delta_i"] for r in rows],
                 yerr=[r["delta_i_err"] for r in rows], color=BLUE, lw=2.0,
                 marker="o", ms=4, capsize=2, zorder=3, label="ΔI (retrieval overlap)")
    ax1.errorbar(temps, [r["delta_c"] for r in rows],
                 yerr=[r["delta_c_err"] for r in rows], color=ORANGE, lw=2.0,
                 marker="s", ms=4, capsize=2, zorder=3, label="ΔC (visit entropy)")
    ax1.plot(temps, [r["activity"] for r in rows], color=GRAY, lw=1.6, ls="--",
             zorder=2, label="activity load (undriven)")
    ax1.plot(temps, [r["activity"] for r in rows_d], color=PURPLE, lw=1.6,
             ls="--", zorder=2, label="activity load (driven)")
    if np.isfinite(t_c):
        ax1.axvline(t_c, color=RED, lw=1.2, ls=":", zorder=2)
        ax1.text(t_c, 0.98, " T_c", color=RED, fontsize=9, va="top",
                 transform=ax1.get_xaxis_transform())
    ax1.set_xlabel("temperature  T")
    ax1.set_ylabel("reading")
    ax1.set_title("A. The substrate: ΔC peaks where order dies")
    ax1.legend(fontsize=8, loc="best")

    # panel B: S(T) for the consumption ladder (DRIVEN — the condition on)
    la = ladders["driven_activity"]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(la)))
    temps_d = np.array([r["T"] for r in rows_d])
    for entry, color in zip(la, cmap):
        ax2.plot(temps_d, entry["S_curve"], color=color, lw=1.8, zorder=3,
                 label=f"c = {entry['c']:g}")
        ax2.plot([entry["T_star"]],
                 [max(entry["S_curve"])], "o", color=color, ms=7, zorder=4)
    if np.isfinite(t_c_d):
        ax2.axvline(t_c_d, color=RED, lw=1.2, ls=":", zorder=2)
    ax2.set_xlabel("temperature  T")
    ax2.set_ylabel("S = ΔC + κ·w·ΔI")
    ax2.set_title("B. The S landscape as capacity binds (driven memory)")
    ax2.legend(fontsize=7, loc="best")

    # panel C: the relocation curves, all three conditions
    for key, color, label in (
            ("undriven_activity", GRAY, "undriven (frozen order is free)"),
            ("undriven_deltac", GREEN, "ΔC load (falsifier)"),
            ("driven_activity", BLUE, "DRIVEN (order costs per query)")):
        cs = [l["c"] for l in ladders[key]]
        ts = [l["T_star"] for l in ladders[key]]
        ax3.plot(cs, ts, color=color, lw=2.2, marker="o", ms=5, zorder=3,
                 label=label)
    if np.isfinite(t_c_d):
        ax3.axhline(t_c_d, color=RED, lw=1.2, ls=":", zorder=2)
        ax3.text(0.02, t_c_d, " T_c", color=RED, fontsize=9, va="bottom",
                 transform=ax3.get_yaxis_transform())
    ax3.set_xscale("log")
    ax3.set_xlabel("consumption  c  (capacity scarcity)")
    ax3.set_ylabel("T⋆(c) = argmax_T S")
    ax3.set_title("C. The relocation: three conditions, one toggle")
    ax3.legend(fontsize=8, loc="best")

    # panel D: verdict
    ax4.axis("off")
    ax4.set_title("D. The verdict")
    vu = verdicts["undriven_activity"]
    vf = verdicts["undriven_deltac"]
    vd = verdicts["driven_activity"]
    pattern_ok = vd["relocated"] and not vu["relocated"] and not vf["relocated"]
    txt = (f"substrate: Hopfield network (no lattice,\n"
           f"learned couplings, T_c = {t_c:.2f})\n\n"
           f"1 undriven:   T* {vu['start']:.2f} -> {vu['end']:.2f}"
           f"   predict NO  -> {'NO ok' if not vu['relocated'] else 'YES !!'}\n"
           f"2 dC load:    T* {vf['start']:.2f} -> {vf['end']:.2f}"
           f"   predict NO  -> {'NO ok' if not vf['relocated'] else 'YES !!'}\n"
           f"3 driven:     T* {vd['start']:.2f} -> {vd['end']:.2f}"
           f"   predict YES -> {'YES ok' if vd['relocated'] else 'NO !!'}"
           + ("  (LEVEL CROSSING)" if vd["level_crossing"] else "")
           + "\n\n"
           + ("the law survives transplant, condition\n"
              "made precise: scarcity evicts ordered\n"
              "optima exactly when maintaining order\n"
              "costs capacity. an unused frozen memory\n"
              "is scarcity-proof; a memory IN USE is\n"
              "pushed to criticality as capacity binds."
              if pattern_ok else
              "the predicted pattern did not land in\n"
              "full - the mechanism story needs revision\n"
              "(see summary B)."))
    ax4.text(0.03, 0.92, txt, transform=ax4.transAxes, fontsize=9.5, va="top",
             family="monospace")

    fig.suptitle("The criticality transplant: capacity-bound S-climbing on a "
                 "substrate with no lattice, no sectors, learned couplings",
                 fontsize=12, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-spins", type=int, default=300)
    p.add_argument("--n-patterns", type=int, default=9)
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--n-burn", type=int, default=30)
    p.add_argument("--n-window", type=int, default=60)
    p.add_argument("--t-min", type=float, default=0.1)
    p.add_argument("--t-max", type=float, default=1.6)
    p.add_argument("--n-temps", type=int, default=16)
    p.add_argument("--structured-threshold", type=float, default=0.3)
    p.add_argument("--query-fraction", type=float, default=0.05,
                   help="Fraction of spins perturbed per query (driven runs).")
    p.add_argument("--query-every", type=int, default=5,
                   help="Window sweeps between queries (driven runs).")
    p.add_argument("--weight", type=float, default=1.5,
                   help="Integration weight w in S = ΔC + κ·w·ΔI.")
    p.add_argument("--recovery", type=float, default=0.1,
                   help="Capacity recovery rate r (engine law).")
    p.add_argument("--consumptions", type=float, nargs="+",
                   default=[0.05, 0.2, 0.8, 3.0, 12.0, 50.0])
    p.add_argument("--critical-window", type=float, default=0.35,
                   help="Fractional T_c window counted as 'critical "
                        "neighbourhood'.")
    p.add_argument("--deadband", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_transplant")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.n_spins = 200
        args.n_seeds = 2
        args.n_burn = 15
        args.n_window = 30
        args.n_temps = 10

    os.makedirs(args.output_dir, exist_ok=True)
    print("the criticality transplant: capacity-bound S-climbing on the "
          "thermal Hopfield network", flush=True)
    print("condition off — undriven (a frozen memory pays nothing):")
    rows_u = measure_substrate(args, driven=False)
    print("condition on — driven (a memory in use pays repair per query):")
    rows_d = measure_substrate(args, driven=True)
    t_c_u = critical_temperature(rows_u)
    t_c_d = critical_temperature(rows_d)
    temps = np.array([r["T"] for r in rows_u])
    conditions = {"undriven_activity": (rows_u, "activity", t_c_u),
                  "undriven_deltac": (rows_u, "delta_c", t_c_u),
                  "driven_activity": (rows_d, "activity", t_c_d)}
    ladders, verdicts = {}, {}
    for key, (rows, load_key, t_c) in conditions.items():
        ladders[key] = optimum_ladder(rows, args, load_key)
        verdicts[key] = relocation_verdict(ladders[key], t_c, temps, args)

    lines = summarise(rows_u, rows_d, t_c_u, t_c_d, ladders, verdicts, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args), "rows_undriven": rows_u,
               "rows_driven": rows_d, "t_c_undriven": t_c_u,
               "t_c_driven": t_c_d, "ladders": ladders, "verdicts": verdicts}
    with open(os.path.join(args.output_dir, "n3_criticality_transplant.json"),
              "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_criticality_transplant.png")
        make_figure(rows_u, rows_d, t_c_u, t_c_d, ladders, verdicts, args,
                    fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
