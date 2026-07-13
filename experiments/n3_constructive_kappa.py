"""Constructive-load κ: teaching the capacity law the difference between building and breaking.

The curriculum experiment (`n3_curriculum_order.py`) measured the base law's
blind spot: κ taxes *building-upon* exactly like *overwriting*, so at strong
consumption the protected foundations are too rigid to compose on — a
significant negative capacity×curriculum interaction.  The theory question it
posed: can a capacity that recognises **constructive load** keep the base
law's protection *and* the curriculum's composability?

The extension (`continual_learning.ConstructiveKappaSGD`): per parameter, the
learned displacement ``d = w − anchor`` (anchor = weights at the last task
boundary) records commitment; updates *against* the displacement
(``g·d > 0`` — undoing) are destructive, everything else constructive.
**Only destructive updates are gated by κ, and only destructive load consumes
it.**  Protection against undoing; free passage for building on.

Three optimizers meet on two suites, at the HARSH operating point
(``c = 4`` — exactly where the base law failed):

- **plain SGD** — no protection,
- **standard κ** — the base law (protection that also blocks building),
- **constructive κ** — the extension.

Pre-registered predictions:

Q1. **Composability restored**: on the curriculum suite (A → B → XOR(A,B),
    foundations-first vs composite-first), constructive κ's curriculum gain
    is at least plain SGD's — the suppression the base law showed at c = 4
    is gone.
Q2. **Protection retained**: on the interference suite (split-task pair,
    single shared head — the protocol of `n3_continual_learning`),
    constructive κ's retention of task A matches standard κ's (within noise)
    and beats plain SGD's — distinguishing construction must not reopen the
    door to destruction.
Q3. **The payoff**: on the curriculum suite, constructive κ's final all-task
    average beats BOTH plain SGD and standard κ — the engineering advantage
    the base law could not demonstrate.

Failures go in the record as-is.

Usage::

    python experiments/n3_constructive_kappa.py --output-dir artifacts/n3_constructive
    python experiments/n3_constructive_kappa.py --quick
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
    ConstructiveKappaSGD,
    KappaSGD,
    MLP,
    MultiHeadMLP,
    accuracy,
    make_concept_tasks,
    make_split_pair,
    train_task,
)

OPTS = ("plain-sgd", "standard-kappa", "constructive-kappa")
ORDERS = {"foundations-first": ("A", "B", "C"),
          "composite-first": ("C", "A", "B")}


def _make_opt(name: str, model, args):
    if name == "plain-sgd":
        return KappaSGD(model, lr=args.lr)
    if name == "standard-kappa":
        return KappaSGD(model, lr=args.lr, recovery=args.recovery,
                        consumption=args.consumption)
    return ConstructiveKappaSGD(model, lr=args.lr, recovery=args.recovery,
                                consumption=args.consumption)


def curriculum_run(args, seed: int, order: tuple, opt_name: str) -> dict:
    rng = np.random.default_rng(seed)
    n_total = args.n_train + args.n_test
    x, y_a, y_b, y_c = make_concept_tasks(rng, args.dim, n_total)
    labels = {"A": y_a, "B": y_b, "C": y_c}
    tr, te = slice(0, args.n_train), slice(args.n_train, n_total)
    model = MultiHeadMLP(args.dim, args.hidden, rng, heads=("A", "B", "C"))
    opt = _make_opt(opt_name, model, args)
    for task in order:
        opt.new_task()
        train_task(model, opt, x[tr], labels[task][tr],
                   args.epochs, args.batch, rng, head=task)
    final = {t: accuracy(model, x[te], labels[t][te], head=t)
             for t in ("A", "B", "C")}
    return {"final_avg": float(np.mean(list(final.values()))),
            "final_c": final["C"]}


def interference_run(args, seed: int, opt_name: str) -> dict:
    rng = np.random.default_rng(seed)
    n_total = args.n_train + args.n_test
    (xa, ya), (xb, yb) = make_split_pair(rng, args.dim, n_total)
    tr, te = slice(0, args.n_train), slice(args.n_train, n_total)
    model = MLP(args.dim, args.hidden, rng)
    opt = _make_opt(opt_name, model, args)
    opt.new_task()
    train_task(model, opt, xa[tr], ya[tr], args.epochs, args.batch, rng)
    acc_a = accuracy(model, xa[te], ya[te])
    opt.new_task()
    train_task(model, opt, xb[tr], yb[tr], args.epochs, args.batch, rng)
    return {"retention": accuracy(model, xa[te], ya[te]),
            "acquisition": accuracy(model, xb[te], yb[te]),
            "acc_a_after_a": acc_a}


def sweep(args) -> dict:
    out = {"curriculum": {}, "interference": {}}
    for opt_name in OPTS:
        rows = {}
        for order_name, order in ORDERS.items():
            runs = [curriculum_run(args, args.seed + 100 * s, order, opt_name)
                    for s in range(args.n_seeds)]
            rows[order_name] = {
                k: float(np.mean([r[k] for r in runs]))
                for k in ("final_avg", "final_c")}
            rows[order_name].update({
                k + "_err": float(np.std([r[k] for r in runs])
                                  / np.sqrt(args.n_seeds))
                for k in ("final_avg", "final_c")})
        out["curriculum"][opt_name] = rows
        runs = [interference_run(args, args.seed + 100 * s, opt_name)
                for s in range(args.n_seeds)]
        out["interference"][opt_name] = {
            k: float(np.mean([r[k] for r in runs]))
            for k in ("retention", "acquisition", "acc_a_after_a")}
        out["interference"][opt_name].update({
            k + "_err": float(np.std([r[k] for r in runs])
                              / np.sqrt(args.n_seeds))
            for k in ("retention", "acquisition")})
        cur = out["curriculum"][opt_name]
        intf = out["interference"][opt_name]
        print(f"  {opt_name:>18}: curriculum ff avg = "
              f"{cur['foundations-first']['final_avg']:.3f} "
              f"(C = {cur['foundations-first']['final_c']:.3f}), "
              f"gain = {cur['foundations-first']['final_avg'] - cur['composite-first']['final_avg']:+.3f};  "
              f"interference retention = {intf['retention']:.3f}, "
              f"acquisition = {intf['acquisition']:.3f}", flush=True)
    return out


def analyse(res, args) -> dict:
    def gain(opt):
        c = res["curriculum"][opt]
        return c["foundations-first"]["final_avg"] - c["composite-first"]["final_avg"]

    def gain_err(opt):
        c = res["curriculum"][opt]
        return float(np.hypot(c["foundations-first"]["final_avg_err"],
                              c["composite-first"]["final_avg_err"]))

    g_plain, g_std, g_con = (gain(o) for o in OPTS)
    q1_margin = g_con - g_plain
    q1_err = float(np.hypot(gain_err("constructive-kappa"),
                            gain_err("plain-sgd")))
    q1 = q1_margin > -max(0.02, 2.0 * q1_err)     # at least plain's gain

    r_plain = res["interference"]["plain-sgd"]["retention"]
    r_std = res["interference"]["standard-kappa"]["retention"]
    r_con = res["interference"]["constructive-kappa"]["retention"]
    r_con_err = res["interference"]["constructive-kappa"]["retention_err"]
    r_std_err = res["interference"]["standard-kappa"]["retention_err"]
    q2 = (r_con > r_plain + 0.02
          and r_con > r_std - max(0.05, 2.0 * np.hypot(r_con_err, r_std_err)))

    ff = {o: res["curriculum"][o]["foundations-first"]["final_avg"]
          for o in OPTS}
    ff_err = {o: res["curriculum"][o]["foundations-first"]["final_avg_err"]
              for o in OPTS}
    m_plain = ff["constructive-kappa"] - ff["plain-sgd"]
    m_std = ff["constructive-kappa"] - ff["standard-kappa"]
    e_plain = float(np.hypot(ff_err["constructive-kappa"], ff_err["plain-sgd"]))
    e_std = float(np.hypot(ff_err["constructive-kappa"], ff_err["standard-kappa"]))
    q3 = (m_plain > max(0.02, 2.0 * e_plain)
          and m_std > max(0.02, 2.0 * e_std))

    return {"gain_plain": float(g_plain), "gain_standard": float(g_std),
            "gain_constructive": float(g_con),
            "q1_margin": float(q1_margin), "q1_err": q1_err,
            "q1_composability": bool(q1),
            "retention_plain": r_plain, "retention_standard": r_std,
            "retention_constructive": r_con, "q2_protection": bool(q2),
            "ff_avg": ff, "q3_margin_plain": float(m_plain),
            "q3_margin_std": float(m_std), "q3_err_plain": e_plain,
            "q3_err_std": e_std, "q3_payoff": bool(q3)}


def summarise(res, a, args) -> list[str]:
    lines = ["Constructive-load κ: building vs breaking, taught to the "
             "capacity law — verdict", "=" * 74, ""]
    lines.append(
        f"Q1 (composability restored, c = {args.consumption:g}): curriculum "
        f"gain — plain {a['gain_plain']:+.3f}, standard κ "
        f"{a['gain_standard']:+.3f}, constructive κ "
        f"{a['gain_constructive']:+.3f} (margin over plain "
        f"{a['q1_margin']:+.3f} ± {a['q1_err']:.3f}) — "
        + ("the suppression is GONE ✓: distinguishing constructive load "
           "restores the curriculum at the operating point where the base "
           "law killed it." if a["q1_composability"] else
           "still suppressed ✗ — the anchor-displacement definition of "
           "'constructive' does not capture composition here."))
    lines.append("")
    lines.append(
        f"Q2 (protection retained): interference retention — plain "
        f"{a['retention_plain']:.3f}, standard κ {a['retention_standard']:.3f}, "
        f"constructive κ {a['retention_constructive']:.3f} — "
        + ("protection intact ✓: free passage for building does not reopen "
           "the door to overwriting." if a["q2_protection"] else
           "protection lost ✗ — the constructive channel leaks destruction."))
    lines.append("")
    lines.append(
        f"Q3 (the payoff): curriculum all-task average — plain "
        f"{a['ff_avg']['plain-sgd']:.3f}, standard κ "
        f"{a['ff_avg']['standard-kappa']:.3f}, constructive κ "
        f"{a['ff_avg']['constructive-kappa']:.3f} "
        f"(margins {a['q3_margin_plain']:+.3f} ± {a['q3_err_plain']:.3f} over "
        f"plain, {a['q3_margin_std']:+.3f} ± {a['q3_err_std']:.3f} over "
        f"standard) — "
        + ("constructive κ BEATS BOTH ✓ — the capacity machinery's first "
           "demonstrated engineering advantage, earned by the distinction "
           "the theory was missing." if a["q3_payoff"] else
           "no clean double advantage ✗; recorded as-is."))
    lines.append("")
    n_pass = sum(a[k] for k in ("q1_composability", "q2_protection",
                                "q3_payoff"))
    lines.append(f"score: {n_pass}/3 pre-registered predictions land.")
    lines.append("")
    lines.append(
        "honest scope, and what failure means here: 'constructive' is "
        "operationalised as displacement-sign alignment against "
        "initialization anchors. A v1 with task-boundary anchors was "
        "catastrophic (each task's drift away from old solutions read as "
        "constructive; retention fell BELOW plain SGD) and is kept in the "
        "record. If this init-anchored v2 also fails, the mechanism is "
        "already visible: a prior task's solution is an OPTIMUM, so "
        "movement in EITHER direction destroys it — extending displacement "
        "(overshooting) is as destructive as undoing it, and no "
        "per-parameter sign test can see that. The distinction the theory "
        "needs would then be FUNCTIONAL, not parametric: destructive load "
        "= what degrades prior function (pointing to function-space gating "
        "— κ taxing disagreement with prior-task behaviour, the "
        "rehearsal/projection family). One harsh operating point, "
        "miniature substrates, boundaries given.")
    return lines


def make_figure(res, a, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GRAY, GREEN = "#0072B2", "#E69F00", "#999999", "#009E73"
    COLORS = {"plain-sgd": GRAY, "standard-kappa": ORANGE,
              "constructive-kappa": BLUE}
    LABELS = {"plain-sgd": "plain SGD", "standard-kappa": "standard κ",
              "constructive-kappa": "constructive κ"}
    fig = plt.figure(figsize=(13, 9.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.27)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))
    for ax in (ax1, ax2, ax3):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0, axis="y")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    x3 = np.arange(3)

    # panel A: curriculum gain by optimizer (Q1)
    gains = [a["gain_plain"], a["gain_standard"], a["gain_constructive"]]
    ax1.bar(x3, gains, 0.6, color=[COLORS[o] for o in OPTS], zorder=3)
    ax1.axhline(0.0, color="#bbbbbb", lw=0.8)
    ax1.set_xticks(x3)
    ax1.set_xticklabels([LABELS[o] for o in OPTS], fontsize=9)
    ax1.set_ylabel("curriculum gain (ff − cf, all-task avg)")
    ax1.set_title(f"A. Q1 — composability at c = {args.consumption:g}")

    # panel B: interference retention (Q2)
    rets = [res["interference"][o]["retention"] for o in OPTS]
    errs = [res["interference"][o]["retention_err"] for o in OPTS]
    ax2.bar(x3, rets, 0.6, yerr=errs, capsize=3,
            color=[COLORS[o] for o in OPTS], zorder=3)
    ax2.set_xticks(x3)
    ax2.set_xticklabels([LABELS[o] for o in OPTS], fontsize=9)
    ax2.set_ylabel("retention of task A after task B")
    ax2.set_ylim(0.4, 1.0)
    ax2.set_title("B. Q2 — protection against genuine overwriting")

    # panel C: the payoff (Q3)
    ffs = [a["ff_avg"][o] for o in OPTS]
    ff_errs = [res["curriculum"][o]["foundations-first"]["final_avg_err"]
               for o in OPTS]
    ax3.bar(x3, ffs, 0.6, yerr=ff_errs, capsize=3,
            color=[COLORS[o] for o in OPTS], zorder=3)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([LABELS[o] for o in OPTS], fontsize=9)
    ax3.set_ylabel("curriculum all-task average (ff)")
    ax3.set_ylim(0.4, 1.0)
    ax3.set_title("C. Q3 — the payoff")

    # panel D: verdict
    ax4.axis("off")
    ax4.set_title("D. The verdict")
    marks = {True: "ok", False: "FAIL"}
    txt = (f"Q1 composability restored:  {marks[a['q1_composability']]}"
           f"   (gain {a['gain_constructive']:+.3f} vs "
           f"{a['gain_standard']:+.3f} std)\n"
           f"Q2 protection retained:     {marks[a['q2_protection']]}"
           f"   ({a['retention_constructive']:.3f} vs "
           f"{a['retention_plain']:.3f} plain)\n"
           f"Q3 beats both:              {marks[a['q3_payoff']]}"
           f"   ({a['ff_avg']['constructive-kappa']:.3f} vs "
           f"{a['ff_avg']['plain-sgd']:.3f} / "
           f"{a['ff_avg']['standard-kappa']:.3f})\n\n"
           "the extension:\n"
           "  d = w - anchor   (commitment)\n"
           "  destructive iff g*d > 0   (undoing)\n"
           "  kappa gates ONLY destruction;\n"
           "  only destruction consumes kappa\n\n"
           "capacity taxes the undoing of\n"
           "integration - not integration itself.")
    ax4.text(0.03, 0.92, txt, transform=ax4.transAxes, fontsize=10, va="top",
             family="monospace")

    fig.suptitle("Constructive-load κ: protection against breaking, free "
                 "passage for building", fontsize=12, y=0.99)
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
    p.add_argument("--consumption", type=float, default=4.0,
                   help="The harsh operating point where the base law "
                        "suppressed the curriculum.")
    p.add_argument("--recovery", type=float, default=0.1)
    p.add_argument("--n-seeds", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str,
                   default="artifacts/n3_constructive")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.n_train = 600
        args.epochs = 15
        args.n_seeds = 3

    os.makedirs(args.output_dir, exist_ok=True)
    print("constructive-load kappa: three optimizers, two suites, the harsh "
          "operating point", flush=True)
    res = sweep(args)
    a = analyse(res, args)

    lines = summarise(res, a, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args), "results": res, "analysis": a}
    with open(os.path.join(args.output_dir, "n3_constructive_kappa.json"),
              "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_constructive_kappa.png")
        make_figure(res, a, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
