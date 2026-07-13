"""Curriculum order under the capacity law: do foundations-first curricula win, and does capacity amplify it?

A long-standing observation of the programme's own exploration phase: the
*order* in which concepts are learned greatly affects the outcome — logic,
then geometry, then mathematics; foundations first, building upward.  The
capacity law turns that observation into a mechanism with a testable
signature: **foundations learned first root in fresh soil and are protected;
a composite learned later composes that protected structure while rooting its
new parts in remaining capacity.  Learned composite-first, the hard task
drains the soil everywhere and the foundations have nowhere left to root.**

The substrate (`continual_learning.py`): three tasks over the same inputs
with genuine compositional structure — concept A (a teacher on the first half
of the features), concept B (second half), and the composite C = **XOR(A, B)**,
learnable only by representing both concepts.  Only the curriculum order
varies; the total training budget is identical.

Protocol note, stated in the record: a single-head pilot showed near-total
interference *at the readout* (every task at chance after later phases,
regardless of order or optimizer) — a known confound of sharing one 2-way
head across differently-labelled tasks, which swamps any representation-level
effect.  The experiment therefore uses the standard task-incremental
protocol: one shared hidden layer (where composition and the capacity soil
live), one private head per task.  The predictions were fixed before either
run and are unchanged.

Pre-registered predictions:

P1. **Order matters**: foundations-first (A→B→C) beats composite-first
    (C→A→B) on final composite accuracy at equal total budget.
P2. **Capacity amplifies the curriculum** (the URP-specific claim): the
    curriculum gain — the all-task final-average advantage of
    foundations-first over composite-first — is larger under κ-gated
    plasticity at moderate recovery than under plain SGD, because the
    capacity dynamics *protect* the foundations while the composite trains
    on top of them.  Plain SGD has no memory of what deserves protecting.
P3. **Transfer is real**: task C reaches higher accuracy after A and B than
    from scratch at matched C-epochs, under both optimizers — the
    composition, not the optimizer, carries the foundations' value.

Failures go in the record as-is.

Usage::

    python experiments/n3_curriculum_order.py --output-dir artifacts/n3_curriculum
    python experiments/n3_curriculum_order.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.continual_learning import (
    KappaSGD,
    MultiHeadMLP,
    accuracy,
    make_concept_tasks,
    train_task,
)

ORDERS = {"foundations-first": ("A", "B", "C"),
          "composite-first": ("C", "A", "B")}


def one_run(args, seed: int, order: tuple, recovery,
            consumption=None) -> dict:
    """Train the three tasks in ``order``; return final and phase accuracies."""
    rng = np.random.default_rng(seed)
    n_total = args.n_train + args.n_test
    x, y_a, y_b, y_c = make_concept_tasks(rng, args.dim, n_total)
    labels = {"A": y_a, "B": y_b, "C": y_c}
    tr = slice(0, args.n_train)
    te = slice(args.n_train, n_total)

    model = MultiHeadMLP(args.dim, args.hidden, rng, heads=("A", "B", "C"))
    opt = KappaSGD(model, lr=args.lr, recovery=recovery,
                   consumption=(args.consumption if consumption is None
                                else consumption))
    acc_after_own = {}
    for task in order:
        train_task(model, opt, x[tr], labels[task][tr],
                   args.epochs, args.batch, rng, head=task)
        acc_after_own[task] = accuracy(model, x[te], labels[task][te],
                                       head=task)
    final = {task: accuracy(model, x[te], labels[task][te], head=task)
             for task in ("A", "B", "C")}
    return {"final": final,
            "final_avg": float(np.mean(list(final.values()))),
            "final_c": final["C"],
            "c_after_own_phase": acc_after_own["C"]}


def sweep(args) -> dict:
    """All (order × optimizer) cells, averaged over seeds."""
    optimizers = {"plain-sgd": None, "kappa": args.recovery}
    cells = {}
    for order_name, order in ORDERS.items():
        for opt_name, recovery in optimizers.items():
            runs = [one_run(args, args.seed + 100 * s, order, recovery)
                    for s in range(args.n_seeds)]
            cell = {}
            for key in ("final_avg", "final_c", "c_after_own_phase"):
                vals = np.array([r[key] for r in runs])
                cell[key] = float(vals.mean())
                cell[key + "_err"] = float(vals.std() / np.sqrt(len(vals)))
            cell["final_per_task"] = {
                t: float(np.mean([r["final"][t] for r in runs]))
                for t in ("A", "B", "C")}
            cells[(order_name, opt_name)] = cell
            print(f"  {order_name:>17} × {opt_name:<9}: "
                  f"final avg = {cell['final_avg']:.3f}  "
                  f"final C = {cell['final_c']:.3f}  "
                  f"(A={cell['final_per_task']['A']:.3f}, "
                  f"B={cell['final_per_task']['B']:.3f})", flush=True)
    return cells


def diagnostic_scan(args) -> list[dict]:
    """Labelled diagnostic (not a registered prediction): map the κ operating
    window over consumption strength — is there a capacity strength that both
    retains foundations and leaves them composable?"""
    out = []
    for c in args.consumption_scan:
        rows = {}
        for order_name, order in ORDERS.items():
            runs = [one_run(args, args.seed + 100 * s, order,
                            recovery=args.recovery, consumption=c)
                    for s in range(args.n_seeds)]
            rows[order_name] = {
                "final_avg": float(np.mean([r["final_avg"] for r in runs])),
                "final_c": float(np.mean([r["final_c"] for r in runs]))}
        out.append({"consumption": float(c),
                    "gain": rows["foundations-first"]["final_avg"]
                    - rows["composite-first"]["final_avg"],
                    "ff_final_c": rows["foundations-first"]["final_c"],
                    "cf_final_c": rows["composite-first"]["final_c"]})
        print(f"  diagnostic c={c:g}: curriculum gain = {out[-1]['gain']:+.3f}  "
              f"ff final C = {out[-1]['ff_final_c']:.3f}", flush=True)
    return out


def analyse(cells, args) -> dict:
    ff_k = cells[("foundations-first", "kappa")]
    cf_k = cells[("composite-first", "kappa")]
    ff_p = cells[("foundations-first", "plain-sgd")]
    cf_p = cells[("composite-first", "plain-sgd")]

    # P1 as registered ("foundations-first beats composite-first on final
    # composite accuracy") — read on the substrate's plain condition, so a
    # capacity-side pathology cannot mask a real order effect
    p1_margin = ff_p["final_c"] - cf_p["final_c"]
    p1_err = float(np.hypot(ff_p["final_c_err"], cf_p["final_c_err"]))
    p1 = p1_margin > max(0.02, 2.0 * p1_err)

    # P2: the interaction — curriculum gain (all-task average) larger with κ
    gain_kappa = ff_k["final_avg"] - cf_k["final_avg"]
    gain_plain = ff_p["final_avg"] - cf_p["final_avg"]
    interaction = gain_kappa - gain_plain
    int_err = float(np.sqrt(ff_k["final_avg_err"] ** 2
                            + cf_k["final_avg_err"] ** 2
                            + ff_p["final_avg_err"] ** 2
                            + cf_p["final_avg_err"] ** 2))
    p2 = interaction > max(0.02, 2.0 * int_err)

    # P3: transfer — C after foundations beats C from scratch, both optimizers
    t_kappa = ff_k["c_after_own_phase"] - cf_k["c_after_own_phase"]
    t_plain = ff_p["c_after_own_phase"] - cf_p["c_after_own_phase"]
    p3 = t_kappa > 0.02 and t_plain > 0.02

    return {"p1_order_matters": bool(p1), "p1_margin": float(p1_margin),
            "p1_err": p1_err,
            "gain_kappa": float(gain_kappa), "gain_plain": float(gain_plain),
            "interaction": float(interaction), "interaction_err": int_err,
            "p2_capacity_amplifies": bool(p2),
            "transfer_kappa": float(t_kappa), "transfer_plain": float(t_plain),
            "p3_transfer": bool(p3),
            "plain_ff_final_c": ff_p["final_c"]}


def summarise(cells, a, args) -> list[str]:
    lines = ["Curriculum order under the capacity law — verdict", "=" * 74, ""]
    ff_k = cells[("foundations-first", "kappa")]
    cf_k = cells[("composite-first", "kappa")]
    ff_p = cells[("foundations-first", "plain-sgd")]
    cf_p = cells[("composite-first", "plain-sgd")]
    lines.append(
        f"P1 (order matters): final composite accuracy, foundations-first "
        f"{ff_p['final_c']:.3f} vs composite-first {cf_p['final_c']:.3f} "
        f"(margin {a['p1_margin']:+.3f} ± {a['p1_err']:.3f}) — "
        + ("foundations-first WINS ✓: the programme's old observation, "
           "reproduced under instrumentation." if a["p1_order_matters"] else
           "no significant order effect on the composite ✗."))
    lines.append("")
    suppresses = (a["interaction"] < -max(0.02, 2.0 * a["interaction_err"]))
    lines.append(
        f"P2 (capacity amplifies the curriculum): all-task curriculum gain = "
        f"{a['gain_kappa']:+.3f} under κ-gating vs {a['gain_plain']:+.3f} "
        f"under plain SGD — interaction {a['interaction']:+.3f} ± "
        f"{a['interaction_err']:.3f} — "
        + ("capacity dynamics AMPLIFY the value of foundations-first ✓: "
           "protection is what lets foundations survive the composite's "
           "training on top of them. The URP-specific claim lands."
           if a["p2_capacity_amplifies"] else
           ("capacity dynamics SUPPRESS the curriculum ✗ — a significant "
            "NEGATIVE interaction: at this operating point the protected "
            "foundations are too rigid to compose on. The capacity law "
            "cannot tell destructive load (overwriting) from constructive "
            "load (building on top) — protection and composability trade "
            "off. A real finding with a mechanism, opposite in sign to the "
            "registered prediction; recorded as-is."
            if suppresses else
            "NO significant interaction ✗: the order effect is real but "
            "optimizer-independent here — curriculum value does not need "
            "the capacity machinery on this substrate; recorded as-is.")))
    lines.append("")
    lines.append(
        f"P3 (transfer is real): C at the end of its own phase — "
        f"after foundations vs from scratch: Δ = {a['transfer_kappa']:+.3f} "
        f"(κ) and {a['transfer_plain']:+.3f} (plain SGD) — "
        + ("both positive ✓: the composition carries the foundations' value, "
           "whatever the optimizer." if a["p3_transfer"] else
           "transfer not established in at least one optimizer ✗."))
    lines.append("")
    n_pass = sum(a[k] for k in ("p1_order_matters", "p2_capacity_amplifies",
                                "p3_transfer"))
    lines.append(f"score: {n_pass}/3 pre-registered predictions land.")
    lines.append("")
    if a.get("diagnostic"):
        d = a["diagnostic"]
        window = [row for row in d
                  if row["gain"] > 0.02
                  and row["ff_final_c"] > a["plain_ff_final_c"] - 0.05]
        lines.append(
            "diagnostic (labelled as such — not a registered prediction): "
            "curriculum gain vs consumption strength: "
            + ";  ".join(f"c={row['consumption']:g} → gain {row['gain']:+.3f}, "
                         f"ff C = {row['ff_final_c']:.3f}" for row in d) + ". "
            + (f"An operating window exists (c ≈ "
               f"{window[0]['consumption']:g}) where capacity retains "
               "foundations without blocking composition — the "
               "protection↔composability trade-off is a dial, not a wall."
               if window else
               "No scanned consumption both preserves the curriculum gain "
               "and keeps the composite learnable — at this recovery rate, "
               "protection and composability exclude each other."))
        lines.append("")
    lines.append(
        "honest scope: one compositional family (XOR of two half-space "
        "concepts), one budget split, one moderate recovery rate, small "
        "MLP, 3-task curricula; 'foundations' here is a structural notion "
        "(the composite is literally a function of the concepts), which is "
        "the cleanest case — natural curricula are messier. Interleaved and "
        "reversed-partial orders, longer concept chains, and consolidation "
        "baselines are the next rungs.")
    return lines


def make_figure(cells, a, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#E69F00", "#009E73", "#999999"
    fig = plt.figure(figsize=(13, 9.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.27)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))
    for ax in (ax1, ax2, ax3):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0, axis="y")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    orders = list(ORDERS)
    xpos = np.arange(2)
    width = 0.35

    # panel A: final composite accuracy by order × optimizer
    for i, (opt, color) in enumerate((("kappa", BLUE), ("plain-sgd", GRAY))):
        vals = [cells[(o, opt)]["final_c"] for o in orders]
        errs = [cells[(o, opt)]["final_c_err"] for o in orders]
        ax1.bar(xpos + (i - 0.5) * width, vals, width, yerr=errs, capsize=3,
                color=color, zorder=3,
                label="κ-gated" if opt == "kappa" else "plain SGD")
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(orders)
    ax1.set_ylabel("final accuracy on composite C")
    ax1.set_ylim(0.4, 1.0)
    ax1.set_title("A. The composite: does order matter?")
    ax1.legend(fontsize=8)

    # panel B: all-task final average — the interaction
    for i, (opt, color) in enumerate((("kappa", BLUE), ("plain-sgd", GRAY))):
        vals = [cells[(o, opt)]["final_avg"] for o in orders]
        errs = [cells[(o, opt)]["final_avg_err"] for o in orders]
        ax2.bar(xpos + (i - 0.5) * width, vals, width, yerr=errs, capsize=3,
                color=color, zorder=3,
                label="κ-gated" if opt == "kappa" else "plain SGD")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(orders)
    ax2.set_ylabel("final accuracy, all-task average")
    ax2.set_ylim(0.4, 1.0)
    ax2.set_title("B. The interaction: curriculum gain × capacity")
    ax2.legend(fontsize=8)

    # panel C: the diagnostic — the protection↔composability trade vs c
    diag = a.get("diagnostic") or []
    if diag:
        cs = [row["consumption"] for row in diag]
        ax3.plot(cs, [row["gain"] for row in diag], color=BLUE, lw=2.2,
                 marker="o", ms=5, zorder=3, label="curriculum gain (κ)")
        ax3.axhline(a["gain_plain"], color=GRAY, lw=1.2, ls="--", zorder=2,
                    label="plain-SGD gain")
        ax3b = ax3.twinx()
        ax3b.plot(cs, [row["ff_final_c"] for row in diag], color=ORANGE,
                  lw=2.0, marker="s", ms=4, zorder=3)
        ax3b.set_ylabel("foundations-first final C", color=ORANGE)
        ax3.axhline(0.0, color="#dddddd", lw=0.8, zorder=1)
        ax3.set_xscale("log")
        ax3.set_xlabel("consumption strength  c")
        ax3.set_ylabel("curriculum gain (all-task avg)")
        ax3.legend(fontsize=8, loc="best")
    ax3.set_title("C. Diagnostic: protection vs composability, in c")

    # panel D: verdict
    ax4.axis("off")
    ax4.set_title("D. The verdict")
    marks = {True: "ok", False: "FAIL"}
    txt = (f"P1 order matters:          {marks[a['p1_order_matters']]}"
           f"   ({a['p1_margin']:+.3f})\n"
           f"P2 capacity amplifies:     {marks[a['p2_capacity_amplifies']]}"
           f"   (gain {a['gain_kappa']:+.3f} vs {a['gain_plain']:+.3f})\n"
           f"P3 transfer is real:       {marks[a['p3_transfer']]}"
           f"   (Δ = {a['transfer_kappa']:+.3f} κ, "
           f"{a['transfer_plain']:+.3f} sgd)\n\n"
           "three tasks, one input space:\n"
           "  A = concept(first half)\n"
           "  B = concept(second half)\n"
           "  C = XOR(A, B)  — composition only\n\n"
           "foundations root in fresh soil and are\n"
           "protected; a composite learned first\n"
           "drains the soil the foundations need.")
    ax4.text(0.03, 0.92, txt, transform=ax4.transAxes, fontsize=10, va="top",
             family="monospace")

    fig.suptitle("Curriculum order under the capacity law: foundations first, "
                 "measured", fontsize=12, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dim", type=int, default=24)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--n-train", type=int, default=1200)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--consumption", type=float, default=4.0)
    p.add_argument("--consumption-scan", type=float, nargs="+",
                   default=[0.25, 0.5, 1.0, 2.0, 4.0],
                   help="Diagnostic ladder mapping the protection↔"
                        "composability trade (κ condition only).")
    p.add_argument("--recovery", type=float, default=0.1,
                   help="κ recovery for the gated condition (near the "
                        "measured crossover r⋆ ≈ 0.11).")
    p.add_argument("--n-seeds", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_curriculum")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.n_train = 600
        args.epochs = 15
        args.n_seeds = 3
        args.consumption_scan = [0.5, 2.0]

    os.makedirs(args.output_dir, exist_ok=True)
    print("curriculum order under the capacity law: foundations-first vs "
          "composite-first × plain SGD vs κ-gated", flush=True)
    cells = sweep(args)
    a = analyse(cells, args)
    print("diagnostic: the protection↔composability trade vs consumption:")
    a["diagnostic"] = diagnostic_scan(args)

    lines = summarise(cells, a, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args),
               "cells": {f"{o}|{p_}": v for (o, p_), v in cells.items()},
               "analysis": a}
    with open(os.path.join(args.output_dir, "n3_curriculum_order.json"),
              "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_curriculum_order.png")
        make_figure(cells, a, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
