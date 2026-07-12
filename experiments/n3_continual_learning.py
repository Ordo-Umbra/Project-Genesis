"""Continual learning under the capacity law: the persistence↔plasticity dial, on real learning.

The sector field measured a sharp persistence↔plasticity crossover in the
capacity recovery rate (`n3_memory_competition`); the Hopfield transplant
showed the law leaves the lattice.  This experiment takes the third step —
the first substrate with **external ground truth**: continual learning, where
catastrophic forgetting vs intransigence *is* the persistence↔plasticity
dilemma, and where the mechanism must earn its keep against a baseline.

Setup (`continual_learning.py`): a small numpy MLP learns task A, then task B
(permuted features, same teacher — the classic interference structure).
Every parameter carries its own capacity κ_j obeying the engine's law,
consumed by that parameter's own update activity and regenerating at rate
``r``; gradients are gated by κ ("rooting consumes soil", for weights).

Two interference conditions — the mechanism's own condition, toggled (the
design pattern the criticality transplant validated).  Per-parameter capacity
can only outdo a global learning rate when the *load is heterogeneous* across
parameters; when every parameter is implicated in both tasks, capacity drains
uniformly and gating degenerates to global LR decay:

- **dense** (permuted features): every parameter carries both tasks — the
  *negative control*.
- **split** (disjoint informative halves): task loads are heterogeneous —
  selective protection has something to select.

Four predictions, registered before the run (P1/P2/P4 read on the split
condition, the mechanism's home regime):

P1. **The dial is monotone**: retention of task A falls with ``r``;
    acquisition of task B rises with ``r``.
P2. **The crossover is sharp**: the overwrite fraction (how much of A's
    learned signal task B erased) crosses ½ at some ``r⋆`` within a narrow
    band of the ladder — a dial, not a smooth trade (the field's r⋆ ≈ 0.36
    result, transplanted).
P3. **The honest value-add test, both ways**: under *dense* interference the
    κ-dial must show NO significant advantage over a tuned constant learning
    rate (the control — capacity gating is predicted to be LR tuning in
    disguise there); under *split* interference it must BEAT the tuned
    constant LR (selectivity pays).  Either half failing goes in the record.
P4. **Criticality echo**: the best *average* learner sits near the
    crossover — the persistence↔plasticity edge, where the transplant says
    capacity-bound S-climbers live.

Usage::

    python experiments/n3_continual_learning.py --output-dir artifacts/n3_continual
    python experiments/n3_continual_learning.py --quick
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
    MLP,
    accuracy,
    make_split_pair,
    make_task,
    make_teacher,
    permute_task,
    train_task,
)


def one_run(args, seed: int, recovery, lr: float, condition: str) -> dict:
    """Train A then B; return retention/acquisition/forgetting metrics."""
    rng = np.random.default_rng(seed)
    n_total = args.n_train + args.n_test
    if condition == "dense":
        teacher = make_teacher(rng, args.dim)
        xa, ya = make_task(rng, teacher, n_total, args.dim)
        xb, yb = permute_task(xa, ya, rng)
    else:
        (xa, ya), (xb, yb) = make_split_pair(rng, args.dim, n_total)
    xa_tr, ya_tr = xa[:args.n_train], ya[:args.n_train]
    xa_te, ya_te = xa[args.n_train:], ya[args.n_train:]
    xb_tr, yb_tr = xb[:args.n_train], yb[:args.n_train]
    xb_te, yb_te = xb[args.n_train:], yb[args.n_train:]

    model = MLP(args.dim, args.hidden, rng)
    opt = KappaSGD(model, lr=lr, recovery=recovery,
                   consumption=args.consumption)
    train_task(model, opt, xa_tr, ya_tr, args.epochs, args.batch, rng)
    acc_a_after_a = accuracy(model, xa_te, ya_te)
    kappa_after_a = opt.mean_kappa()
    train_task(model, opt, xb_tr, yb_tr, args.epochs, args.batch, rng)
    retention = accuracy(model, xa_te, ya_te)
    acquisition = accuracy(model, xb_te, yb_te)
    learned = max(acc_a_after_a - 0.5, 1e-9)
    overwrite = float(np.clip((acc_a_after_a - retention) / learned, 0.0, 1.0))
    return {"acc_a_after_a": acc_a_after_a, "retention": retention,
            "acquisition": acquisition, "overwrite": overwrite,
            "average": 0.5 * (retention + acquisition),
            "kappa_after_a": kappa_after_a,
            "kappa_final": opt.mean_kappa()}


def sweep(args, condition: str, recoveries=None, lrs=None) -> list[dict]:
    """Average `one_run` over seeds along a ladder (κ dial or LR baseline)."""
    out = []
    ladder = recoveries if recoveries is not None else lrs
    for value in ladder:
        runs = [one_run(args, args.seed + 100 * s,
                        recovery=value if recoveries is not None else None,
                        lr=args.lr if recoveries is not None else value,
                        condition=condition)
                for s in range(args.n_seeds)]
        entry = {"value": float(value)}
        for key in ("acc_a_after_a", "retention", "acquisition", "overwrite",
                    "average", "kappa_after_a", "kappa_final"):
            vals = np.array([r[key] for r in runs])
            entry[key] = float(vals.mean())
            entry[key + "_err"] = float(vals.std() / np.sqrt(len(vals)))
        out.append(entry)
        tag = "r" if recoveries is not None else "lr"
        print(f"  {tag}={value:g}: A-after-A={entry['acc_a_after_a']:.3f}  "
              f"retention={entry['retention']:.3f}  "
              f"acquisition={entry['acquisition']:.3f}  "
              f"overwrite={entry['overwrite']:.2f}", flush=True)
    return out


def _crossing(xs, ys, level):
    """First x where ys crosses ``level`` (linear in log-x for r ladders)."""
    for (x1, y1), (x2, y2) in zip(zip(xs, ys), list(zip(xs, ys))[1:]):
        if (y1 - level) * (y2 - level) <= 0 and y1 != y2:
            lx1 = np.log(max(x1, 1e-4))
            lx2 = np.log(max(x2, 1e-4))
            f = (level - y1) / (y2 - y1)
            return float(np.exp(lx1 + f * (lx2 - lx1)))
    return float("nan")


def _frontier(kappa_scan, baseline_scan) -> dict:
    """Best-average comparison of the κ dial against the tuned constant LR."""
    best_kappa = max(kappa_scan, key=lambda e: e["average"])
    best_base = max(baseline_scan, key=lambda e: e["average"])
    margin = best_kappa["average"] - best_base["average"]
    err = float(np.hypot(best_kappa["average_err"], best_base["average_err"]))
    return {"best_kappa_avg": best_kappa["average"],
            "best_kappa_r": best_kappa["value"],
            "best_base_avg": best_base["average"],
            "best_base_lr": best_base["value"],
            "margin": float(margin), "err": err,
            "significant_advantage": bool(margin > max(0.02, 2.0 * err))}


def analyse(scans, args) -> dict:
    kappa_scan = scans["split"]["kappa"]        # the mechanism's home regime
    rs = [e["value"] for e in kappa_scan]
    ret = [e["retention"] for e in kappa_scan]
    acq = [e["acquisition"] for e in kappa_scan]
    over = [e["overwrite"] for e in kappa_scan]
    avg = [e["average"] for e in kappa_scan]

    # P1: monotone dial (endpoints + majority of steps)
    ret_steps = np.diff(ret)
    acq_steps = np.diff(acq)
    p1 = (ret[-1] < ret[0] and acq[-1] > acq[0]
          and (ret_steps <= 0.02).mean() >= 0.7
          and (acq_steps >= -0.02).mean() >= 0.7)

    # P2: sharp crossover in the overwrite fraction
    r_star = _crossing(rs, over, 0.5)
    r_lo = _crossing(rs, over, 0.25)
    r_hi = _crossing(rs, over, 0.75)
    width_decades = (np.log10(r_hi / r_lo)
                     if np.isfinite(r_lo) and np.isfinite(r_hi) and r_lo > 0
                     else float("nan"))
    p2 = np.isfinite(r_star) and np.isfinite(width_decades) and width_decades < 1.0

    # P3, both ways: control (dense — expect no advantage) and value (split)
    dense = _frontier(scans["dense"]["kappa"], scans["dense"]["baseline"])
    split = _frontier(scans["split"]["kappa"], scans["split"]["baseline"])
    p3_control = not dense["significant_advantage"]
    p3_value = split["significant_advantage"]

    # P4: the best average learner sits near the crossover
    r_opt = rs[int(np.argmax(avg))]
    p4 = (np.isfinite(r_star) and r_star > 0
          and abs(np.log10(max(r_opt, 1e-4) / r_star)) <= 0.7)

    return {"p1_monotone": bool(p1), "r_star": r_star,
            "crossover_width_decades": float(width_decades),
            "p2_sharp": bool(p2),
            "dense": dense, "split": split,
            "p3_control_holds": bool(p3_control),
            "p3_value_add": bool(p3_value),
            "r_opt": float(r_opt), "p4_edge_optimum": bool(p4)}


def summarise(scans, a, args) -> list[str]:
    kappa_scan = scans["split"]["kappa"]
    lines = ["Continual learning under the capacity law: the "
             "persistence↔plasticity dial on real learning — verdict",
             "=" * 74, ""]
    lines.append(
        f"P1 (the dial is monotone; split condition): retention falls "
        f"{kappa_scan[0]['retention']:.3f} → {kappa_scan[-1]['retention']:.3f} "
        f"and acquisition rises {kappa_scan[0]['acquisition']:.3f} → "
        f"{kappa_scan[-1]['acquisition']:.3f} across r ∈ "
        f"[{kappa_scan[0]['value']:g}, {kappa_scan[-1]['value']:g}] — "
        + ("MONOTONE ✓: the recovery rate is the stability–plasticity dial, "
           "as the field measured." if a["p1_monotone"] else
           "NOT cleanly monotone ✗ — the dial claim fails on this substrate."))
    lines.append("")
    lines.append(
        f"P2 (the crossover is sharp): the overwrite fraction crosses ½ at "
        f"r⋆ ≈ {a['r_star']:.3f}, with the 25%→75% band spanning "
        f"{a['crossover_width_decades']:.2f} decades — "
        + ("SHARP ✓: a genuine dial with a threshold, the "
           "n3_memory_competition crossover transplanted to weights."
           if a["p2_sharp"] else
           "not sharp ✗ (or no crossing): the transition is a smooth trade "
           "here, unlike the field's."))
    lines.append("")
    d, s = a["dense"], a["split"]
    lines.append(
        f"P3 (the honest value-add test, both ways):")
    lines.append(
        f"   control, dense interference (predict NO advantage): κ-dial "
        f"{d['best_kappa_avg']:.3f} vs tuned LR {d['best_base_avg']:.3f}, "
        f"margin {d['margin']:+.3f} ± {d['err']:.3f} — "
        + ("no advantage, as predicted ✓ (uniform load degenerates capacity "
           "gating to LR decay)." if a["p3_control_holds"] else
           "an advantage appeared where none was predicted ✗ — the "
           "heterogeneity story needs revision."))
    lines.append(
        f"   value, split interference (predict advantage): κ-dial "
        f"{s['best_kappa_avg']:.3f} (r = {s['best_kappa_r']:g}) vs tuned LR "
        f"{s['best_base_avg']:.3f} (η = {s['best_base_lr']:g}), margin "
        f"{s['margin']:+.3f} ± {s['err']:.3f} — "
        + ("selective regenerating protection BEATS anything a global "
           "constant learning rate reaches ✓: where load is heterogeneous, "
           "the mechanism is not LR tuning in disguise."
           if a["p3_value_add"] else
           "NO significant advantage ✗ — even in its home regime the "
           "capacity machinery does not pay for itself here; recorded as-is."))
    lines.append("")
    lines.append(
        f"P4 (criticality echo): the best average learner sits at "
        f"r_opt = {a['r_opt']:g} against the crossover r⋆ ≈ {a['r_star']:.3f} — "
        + ("the optimum lives at the persistence↔plasticity edge ✓, where "
           "the transplant says capacity-bound S-climbers are pushed."
           if a["p4_edge_optimum"] else
           "the optimum does not track the crossover ✗."))
    lines.append("")
    n_pass = sum(a[k] for k in ("p1_monotone", "p2_sharp", "p3_control_holds",
                                "p3_value_add", "p4_edge_optimum"))
    lines.append(
        f"score: {n_pass}/5 pre-registered predictions land. "
        + ("The capacity law's learning phenomenology transfers to a system "
           "with external ground truth, with its condition (heterogeneous "
           "load) demonstrated by the control — the first result in the "
           "programme that a continual-learning benchmark could now falsify "
           "at scale." if n_pass == 5 else
           "The failures above are the programme's next questions — kept "
           "bright, not buried."))
    lines.append("")
    lines.append(
        "honest scope: one task pair per condition, one consumption strength, "
        "a small numpy MLP, 2-task sequences only; the baseline is a tuned "
        "constant LR — synaptic-consolidation baselines (EWC, SI) are the "
        "necessary next comparison, and standard benchmarks (split/permuted "
        "MNIST, longer task sequences) the next substrate. The overwrite "
        "normalisation and ladders are stated conventions.")
    return lines


def make_figure(scans, a, args, path):
    kappa_scan = scans["split"]["kappa"]
    baseline_scan = scans["split"]["baseline"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY, RED = ("#0072B2", "#E69F00", "#009E73",
                                      "#999999", "#c0392b")
    fig = plt.figure(figsize=(13, 9.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.27)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))
    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    rs = np.array([max(e["value"], 1e-3) for e in kappa_scan])

    # panel A: the dial
    for key, color, label in (("retention", BLUE, "retention of A"),
                              ("acquisition", GREEN, "acquisition of B"),
                              ("average", GRAY, "average")):
        ax1.errorbar(rs, [e[key] for e in kappa_scan],
                     yerr=[e[key + "_err"] for e in kappa_scan],
                     color=color, lw=2.0, marker="o", ms=4, capsize=2,
                     zorder=3, label=label)
    ax1.set_xscale("log")
    ax1.set_xlabel("capacity recovery rate  r")
    ax1.set_ylabel("test accuracy")
    ax1.set_title("A. The dial: persistence ↔ plasticity in r")
    ax1.legend(fontsize=8, loc="best")

    # panel B: the crossover
    ax2.errorbar(rs, [e["overwrite"] for e in kappa_scan],
                 yerr=[e["overwrite_err"] for e in kappa_scan],
                 color=ORANGE, lw=2.2, marker="s", ms=5, capsize=2, zorder=3)
    ax2.axhline(0.5, color=GRAY, lw=1.0, ls="--", zorder=2)
    if np.isfinite(a["r_star"]):
        ax2.axvline(a["r_star"], color=RED, lw=1.2, ls=":", zorder=2)
        ax2.text(a["r_star"], 0.03, f" r⋆ ≈ {a['r_star']:.2f}", color=RED,
                 fontsize=9, transform=ax2.get_xaxis_transform())
    ax2.set_xscale("log")
    ax2.set_xlabel("capacity recovery rate  r")
    ax2.set_ylabel("overwrite fraction of task A")
    ax2.set_title("B. The crossover: write-once → plastic")

    # panel C: the frontier test, both conditions
    ax3.plot([e["retention"] for e in kappa_scan],
             [e["acquisition"] for e in kappa_scan],
             color=BLUE, lw=2.2, marker="o", ms=5, zorder=4,
             label="split: κ-dial (by r)")
    ax3.plot([e["retention"] for e in baseline_scan],
             [e["acquisition"] for e in baseline_scan],
             color=GRAY, lw=2.0, marker="^", ms=5, ls="--", zorder=3,
             label="split: constant LR (by η)")
    ax3.plot([e["retention"] for e in scans["dense"]["kappa"]],
             [e["acquisition"] for e in scans["dense"]["kappa"]],
             color=BLUE, lw=1.2, marker="o", ms=3, alpha=0.35, zorder=2,
             label="dense: κ-dial (control)")
    ax3.plot([e["retention"] for e in scans["dense"]["baseline"]],
             [e["acquisition"] for e in scans["dense"]["baseline"]],
             color=GRAY, lw=1.2, marker="^", ms=3, ls="--", alpha=0.35,
             zorder=2, label="dense: constant LR")
    ax3.set_xlabel("retention of task A")
    ax3.set_ylabel("acquisition of task B")
    ax3.set_title("C. The honest test: κ frontier vs LR frontier")
    ax3.legend(fontsize=7.5, loc="best")

    # panel D: verdict
    ax4.axis("off")
    ax4.set_title("D. The verdict")
    marks = {True: "ok", False: "FAIL"}
    d, s = a["dense"], a["split"]
    txt = (f"P1 dial monotone:         {marks[a['p1_monotone']]}\n"
           f"P2 crossover sharp:       {marks[a['p2_sharp']]}"
           f"   (r* = {a['r_star']:.2f}, "
           f"{a['crossover_width_decades']:.2f} dec)\n"
           f"P3 control (dense) null:  {marks[a['p3_control_holds']]}"
           f"   ({d['margin']:+.3f})\n"
           f"P3 value (split) beats:   {marks[a['p3_value_add']]}"
           f"   ({s['best_kappa_avg']:.3f} vs {s['best_base_avg']:.3f})\n"
           f"P4 optimum at the edge:   {marks[a['p4_edge_optimum']]}"
           f"   (r_opt = {a['r_opt']:g})\n\n"
           "capacity-gated plasticity:\n"
           "  dk/dt = r(1-k) - c|g|k,  dw = -lr k g\n\n"
           "the field's soil law on real learning:\n"
           "selective regenerating protection pays\n"
           "exactly where load is heterogeneous —\n"
           "the condition toggled, the baseline\n"
           "given a fair shot.")
    ax4.text(0.03, 0.92, txt, transform=ax4.transAxes, fontsize=9.5, va="top",
             family="monospace")

    fig.suptitle("Continual learning under the capacity law: the "
                 "persistence↔plasticity dial meets external ground truth",
                 fontsize=12, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dim", type=int, default=24)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--consumption", type=float, default=4.0)
    p.add_argument("--recoveries", type=float, nargs="+",
                   default=[0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
    p.add_argument("--baseline-lrs", type=float, nargs="+",
                   default=[0.02, 0.05, 0.1, 0.2, 0.35, 0.5])
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_continual")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.n_train = 400
        args.epochs = 15
        args.n_seeds = 3
        args.recoveries = [0.0, 0.05, 0.2, 0.8, 3.2]
        args.baseline_lrs = [0.05, 0.2, 0.5]

    os.makedirs(args.output_dir, exist_ok=True)
    print("continual learning under the capacity law: the "
          "persistence↔plasticity dial, with the condition toggled and a "
          "fair baseline", flush=True)
    scans = {}
    for condition in ("dense", "split"):
        label = ("dense interference (negative control)"
                 if condition == "dense" else
                 "split interference (the mechanism's home regime)")
        print(f"{label} — κ-dial ladder:")
        kappa_scan = sweep(args, condition, recoveries=args.recoveries)
        print(f"{label} — constant-LR baseline ladder:")
        baseline_scan = sweep(args, condition, lrs=args.baseline_lrs)
        scans[condition] = {"kappa": kappa_scan, "baseline": baseline_scan}
    a = analyse(scans, args)

    lines = summarise(scans, a, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args), "scans": scans, "analysis": a}
    with open(os.path.join(args.output_dir, "n3_continual_learning.json"),
              "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_continual_learning.png")
        make_figure(scans, a, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
