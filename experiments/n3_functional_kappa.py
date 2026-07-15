"""Function-space κ: the constructive/destructive distinction, done at the level it lives.

The parametric constructive-κ experiment (`n3_constructive_kappa.py`) failed
and named the reason: destruction is *functional* — a prior task's solution
is an optimum, invisible to per-parameter geometry.  This experiment tests
the formulation that failure demanded (`FunctionalKappaSGD`): a small memory
buffer per completed task; per step, the gradient's **damaging component**
(its projection onto the prior-task gradient, when they conflict) is gated
by κ and consumes capacity, while the orthogonal remainder — construction —
passes free.  Consolidation is part of the law: `remember()` sets the
destructive budget to the steady state under full conflicting load
(`κ ← r/(r+c)`, no new parameter), so protection is present from the first
conflicting step; recovery `r` then returns plasticity over time.

Development-record notes carried into this registration: two intermediate
formulations failed and are documented in the class (elementwise sign =
noise-blind; global scalar gate = too weak), and a probe measured that on
*totally* conflicting tasks (permuted features, shared head) even pure A-GEM
barely beats plain SGD — single-buffer projection needs heterogeneous
conflict to have an escape direction.  The suites below are the same as the
parent experiment's.

Pre-registered predictions (four optimizers: plain SGD, standard κ,
parametric constructive κ, functional κ; harsh point c = 4):

F1. **Composability restored**: functional κ's curriculum gain ≥ plain
    SGD's (the suppression standard κ showed at c = 4 is gone).
F2. **Protection retained**: on the split-interference suite, functional κ's
    retention beats plain SGD's and is within noise of standard κ's.
F3. **The payoff**: functional κ's curriculum all-task average beats BOTH
    plain SGD and standard κ — the advantage neither parametric variant
    could show.

Honest information note: functional κ uses stored exemplars the others do
not; rehearsal/A-GEM are the fair external family, named for the next round.

Usage::

    python experiments/n3_functional_kappa.py --output-dir artifacts/n3_functional
    python experiments/n3_functional_kappa.py --quick
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
    FunctionalKappaSGD,
    KappaSGD,
    MLP,
    MultiHeadMLP,
    accuracy,
    make_concept_tasks,
    make_split_pair,
    train_task,
)

OPTS = ("plain-sgd", "standard-kappa", "constructive-kappa", "functional-kappa")
ORDERS = {"foundations-first": ("A", "B", "C"),
          "composite-first": ("C", "A", "B")}


def _make_opt(name, model, args):
    if name == "plain-sgd":
        return KappaSGD(model, lr=args.lr)
    if name == "standard-kappa":
        return KappaSGD(model, lr=args.lr, recovery=args.recovery,
                        consumption=args.consumption)
    if name == "constructive-kappa":
        return ConstructiveKappaSGD(model, lr=args.lr, recovery=args.recovery,
                                    consumption=args.consumption)
    return FunctionalKappaSGD(model, lr=args.lr, recovery=args.recovery,
                              consumption=args.consumption,
                              buffer_size=args.buffer_size)


def curriculum_run(args, seed, order, opt_name):
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
        opt.remember(x[tr], labels[task][tr], head=task)
    final = {t: accuracy(model, x[te], labels[t][te], head=t)
             for t in ("A", "B", "C")}
    return {"final_avg": float(np.mean(list(final.values())))}


def interference_run(args, seed, opt_name):
    rng = np.random.default_rng(seed)
    n_total = args.n_train + args.n_test
    (xa, ya), (xb, yb) = make_split_pair(rng, args.dim, n_total)
    tr, te = slice(0, args.n_train), slice(args.n_train, n_total)
    model = MLP(args.dim, args.hidden, rng)
    opt = _make_opt(opt_name, model, args)
    train_task(model, opt, xa[tr], ya[tr], args.epochs, args.batch, rng)
    opt.remember(xa[tr], ya[tr])
    opt.new_task()
    train_task(model, opt, xb[tr], yb[tr], args.epochs, args.batch, rng)
    return {"retention": accuracy(model, xa[te], ya[te]),
            "acquisition": accuracy(model, xb[te], yb[te])}


def sweep(args):
    res = {"gain": {}, "ff_avg": {}, "ff_avg_err": {}, "gain_err": {},
           "retention": {}, "retention_err": {}, "acquisition": {}}
    for opt_name in OPTS:
        cur = {}
        for order_name, order in ORDERS.items():
            vals = np.array([curriculum_run(args, args.seed + 100 * s, order,
                                            opt_name)["final_avg"]
                             for s in range(args.n_seeds)])
            cur[order_name] = (float(vals.mean()),
                               float(vals.std() / np.sqrt(args.n_seeds)))
        res["gain"][opt_name] = cur["foundations-first"][0] - cur["composite-first"][0]
        res["gain_err"][opt_name] = float(np.hypot(cur["foundations-first"][1],
                                                   cur["composite-first"][1]))
        res["ff_avg"][opt_name] = cur["foundations-first"][0]
        res["ff_avg_err"][opt_name] = cur["foundations-first"][1]
        runs = [interference_run(args, args.seed + 100 * s, opt_name)
                for s in range(args.n_seeds)]
        rets = np.array([r["retention"] for r in runs])
        res["retention"][opt_name] = float(rets.mean())
        res["retention_err"][opt_name] = float(rets.std() / np.sqrt(args.n_seeds))
        res["acquisition"][opt_name] = float(
            np.mean([r["acquisition"] for r in runs]))
        print(f"  {opt_name:>19}: gain = {res['gain'][opt_name]:+.3f}, "
              f"ff avg = {res['ff_avg'][opt_name]:.3f}, "
              f"retention = {res['retention'][opt_name]:.3f}, "
              f"acquisition = {res['acquisition'][opt_name]:.3f}", flush=True)
    return res


def analyse(res, args):
    fn, pl, st = "functional-kappa", "plain-sgd", "standard-kappa"
    f1_margin = res["gain"][fn] - res["gain"][pl]
    f1_err = float(np.hypot(res["gain_err"][fn], res["gain_err"][pl]))
    f1 = f1_margin > -max(0.02, 2.0 * f1_err)
    ret_err = float(np.hypot(res["retention_err"][fn], res["retention_err"][st]))
    f2 = (res["retention"][fn] > res["retention"][pl] + 0.02
          and res["retention"][fn] > res["retention"][st]
          - max(0.05, 2.0 * ret_err))
    m_pl = res["ff_avg"][fn] - res["ff_avg"][pl]
    m_st = res["ff_avg"][fn] - res["ff_avg"][st]
    e_pl = float(np.hypot(res["ff_avg_err"][fn], res["ff_avg_err"][pl]))
    e_st = float(np.hypot(res["ff_avg_err"][fn], res["ff_avg_err"][st]))
    f3 = m_pl > max(0.02, 2.0 * e_pl) and m_st > max(0.02, 2.0 * e_st)
    return {"f1_margin": float(f1_margin), "f1_err": f1_err,
            "f1_composability": bool(f1), "f2_protection": bool(f2),
            "f3_margin_plain": float(m_pl), "f3_margin_std": float(m_st),
            "f3_err_plain": e_pl, "f3_err_std": e_st, "f3_payoff": bool(f3)}


def summarise(res, a, args):
    lines = ["Function-space κ — verdict", "=" * 74, ""]
    g = res["gain"]
    lines.append(
        f"F1 (composability, c = {args.consumption:g}): curriculum gain — "
        f"plain {g['plain-sgd']:+.3f}, standard κ {g['standard-kappa']:+.3f}, "
        f"parametric {g['constructive-kappa']:+.3f}, FUNCTIONAL "
        f"{g['functional-kappa']:+.3f} (margin over plain "
        f"{a['f1_margin']:+.3f} ± {a['f1_err']:.3f}) — "
        + ("restored ✓." if a["f1_composability"] else "still suppressed ✗."))
    r = res["retention"]
    lines.append(
        f"F2 (protection): retention — plain {r['plain-sgd']:.3f}, standard κ "
        f"{r['standard-kappa']:.3f}, parametric {r['constructive-kappa']:.3f}, "
        f"FUNCTIONAL {r['functional-kappa']:.3f} (acquisition "
        f"{res['acquisition']['functional-kappa']:.3f}) — "
        + ("intact ✓." if a["f2_protection"] else "lost ✗."))
    f = res["ff_avg"]
    lines.append(
        f"F3 (payoff): curriculum all-task average — plain "
        f"{f['plain-sgd']:.3f}, standard κ {f['standard-kappa']:.3f}, "
        f"parametric {f['constructive-kappa']:.3f}, FUNCTIONAL "
        f"{f['functional-kappa']:.3f} (margins {a['f3_margin_plain']:+.3f} ± "
        f"{a['f3_err_plain']:.3f} / {a['f3_margin_std']:+.3f} ± "
        f"{a['f3_err_std']:.3f}) — "
        + ("beats both ✓ — the capacity machinery's first double advantage."
           if a["f3_payoff"] else "no double advantage ✗; recorded as-is."))
    n = sum(a[k] for k in ("f1_composability", "f2_protection", "f3_payoff"))
    lines.append("")
    lines.append(f"score: {n}/3 pre-registered predictions land.")
    lines.append("")
    lines.append(
        "honest scope: functional κ uses stored exemplars the other "
        "optimizers do not (64/task) — rehearsal and A-GEM are the fair "
        "external family, and the measured single-buffer ceiling on totally "
        "conflicting tasks applies here too. Consolidation-to-steady-state "
        "is the law's own number but a design choice; one harsh operating "
        "point; miniature substrates; boundaries given.")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dim", type=int, default=24)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--n-train", type=int, default=1200)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--consumption", type=float, default=4.0)
    p.add_argument("--recovery", type=float, default=0.1)
    p.add_argument("--buffer-size", type=int, default=64)
    p.add_argument("--n-seeds", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_functional")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_train, args.epochs, args.n_seeds = 600, 15, 3

    os.makedirs(args.output_dir, exist_ok=True)
    print("function-space kappa: four optimizers, two suites, the harsh point",
          flush=True)
    res = sweep(args)
    a = analyse(res, args)
    text = "\n".join(summarise(res, a, args))
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_functional_kappa.json"), "w") as fh:
        json.dump({"params": vars(args), "results": res, "analysis": a},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
