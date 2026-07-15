"""The combined benchmark: composability and protection demanded in one sequence.

Function-space κ (`n3_functional_kappa.py`) matched plain SGD where only
composability matters and standard κ where only protection matters — the open
question its F3 left: is there a demonstrable advantage where a single
sequence demands BOTH?  This benchmark is that sequence:

    A → B → C = XOR(A, B) → D (permuted-features conflict)

on one shared trunk (per-task heads): the first three phases reward building,
the last densely attacks everything built.  Final score: the all-task average
over {A, B, C, D}.  A fair-information baseline is included: **naive
rehearsal** (plain SGD + replay of the same 64-exemplar buffers functional κ
uses).

Pre-registered predictions:

G1. functional κ > plain SGD on the combined average (D wipes an
    unprotected trunk).
G2. functional κ > standard κ (rigidity costs the composite).
G3. functional κ ≥ naive rehearsal within noise — the capacity response must
    justify itself against simple replay of the same information.

Usage::

    python experiments/n3_combined_benchmark.py --output-dir artifacts/n3_combined
    python experiments/n3_combined_benchmark.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.continual_learning import (
    FunctionalKappaSGD,
    KappaSGD,
    MultiHeadMLP,
    accuracy,
    make_concept_tasks,
    train_task,
)

OPTS = ("plain-sgd", "standard-kappa", "functional-kappa", "rehearsal")


def one_run(args, seed: int, opt_name: str) -> dict:
    rng = np.random.default_rng(seed)
    n_total = args.n_train + args.n_test
    x, y_a, y_b, y_c = make_concept_tasks(rng, args.dim, n_total)
    # task D: dense conflict — permuted features, fresh labels via A's rule
    perm = rng.permutation(args.dim)
    xd = x[:, perm]
    labels = {"A": (x, y_a), "B": (x, y_b), "C": (x, y_c), "D": (xd, y_a)}
    tr, te = slice(0, args.n_train), slice(args.n_train, n_total)
    model = MultiHeadMLP(args.dim, args.hidden, rng, heads=tuple(labels))
    if opt_name == "plain-sgd" or opt_name == "rehearsal":
        opt = KappaSGD(model, lr=args.lr)
    elif opt_name == "standard-kappa":
        opt = KappaSGD(model, lr=args.lr, recovery=args.recovery,
                       consumption=args.consumption)
    else:
        opt = FunctionalKappaSGD(model, lr=args.lr, recovery=args.recovery,
                                 consumption=args.consumption,
                                 buffer_size=args.buffer_size)
    buffers: list[tuple[np.ndarray, np.ndarray, str]] = []
    for task, (xt, yt) in labels.items():
        opt.new_task()
        if opt_name == "rehearsal" and buffers:
            # interleave: one replay pass per epoch across stored tasks
            n = args.n_train
            for _ in range(args.epochs):
                order = rng.permutation(n)
                for start in range(0, n, args.batch):
                    idx = order[start:start + args.batch]
                    _, grads = model.loss_and_grads(xt[tr][idx], yt[tr][idx],
                                                    head=task)
                    opt.step(grads)
                for bx, by, bh in buffers:
                    _, grads = model.loss_and_grads(bx, by, head=bh)
                    opt.step(grads)
        else:
            train_task(model, opt, xt[tr], yt[tr], args.epochs, args.batch,
                       rng, head=task)
        opt.remember(xt[tr][:args.buffer_size], yt[tr][:args.buffer_size],
                     head=task)
        buffers.append((xt[tr][:args.buffer_size].copy(),
                        yt[tr][:args.buffer_size].copy(), task))
    final = {t: accuracy(model, labels[t][0][te], labels[t][1][te], head=t)
             for t in labels}
    return {"avg": float(np.mean(list(final.values()))), "final": final}


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
    p.add_argument("--recovery", type=float, default=0.1)
    p.add_argument("--buffer-size", type=int, default=64)
    p.add_argument("--n-seeds", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_combined")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_train, args.epochs, args.n_seeds = 600, 15, 3

    os.makedirs(args.output_dir, exist_ok=True)
    print("the combined benchmark: A → B → XOR(A,B) → dense conflict, "
          "four optimizers", flush=True)
    res = {}
    for opt_name in OPTS:
        runs = [one_run(args, args.seed + 100 * s, opt_name)
                for s in range(args.n_seeds)]
        avgs = np.array([r["avg"] for r in runs])
        res[opt_name] = {
            "avg": float(avgs.mean()),
            "err": float(avgs.std() / np.sqrt(args.n_seeds)),
            "final": {t: float(np.mean([r["final"][t] for r in runs]))
                      for t in ("A", "B", "C", "D")}}
        f = res[opt_name]["final"]
        print(f"  {opt_name:>16}: combined avg = {res[opt_name]['avg']:.3f} "
              f"± {res[opt_name]['err']:.3f}  "
              f"(A={f['A']:.2f} B={f['B']:.2f} C={f['C']:.2f} D={f['D']:.2f})",
              flush=True)

    fn = res["functional-kappa"]

    def beat(other, slack=0.0):
        m = fn["avg"] - res[other]["avg"]
        e = float(np.hypot(fn["err"], res[other]["err"]))
        return m, e, m > max(0.02, 2.0 * e) - slack

    g1m, g1e, g1 = beat("plain-sgd")
    g2m, g2e, g2 = beat("standard-kappa")
    g3m = fn["avg"] - res["rehearsal"]["avg"]
    g3e = float(np.hypot(fn["err"], res["rehearsal"]["err"]))
    g3 = g3m > -max(0.02, 2.0 * g3e)

    lines = ["The combined benchmark — verdict", "=" * 74, ""]
    lines.append(f"G1 (beats plain SGD): {g1m:+.3f} ± {g1e:.3f} — "
                 + ("✓ the unprotected trunk pays for D." if g1 else "✗."))
    lines.append(f"G2 (beats standard κ): {g2m:+.3f} ± {g2e:.3f} — "
                 + ("✓ rigidity costs the composite." if g2 else "✗."))
    lines.append(f"G3 (competitive with rehearsal): {g3m:+.3f} ± {g3e:.3f} — "
                 + ("✓ the capacity response holds its own against replay "
                    "of the same information." if g3 else
                    "✗ simple replay wins — recorded as-is."))
    n = int(g1) + int(g2) + int(g3)
    lines.append("")
    lines.append(f"score: {n}/3 pre-registered predictions land.")
    lines.append("")
    lines.append(
        "honest scope: one sequence design, one harsh operating point, "
        "miniature substrates, boundaries given; rehearsal here is one "
        "naive schedule (one replay pass per epoch) — tuned replay and "
        "A-GEM proper are stronger fair-information baselines for the "
        "next round.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_combined_benchmark.json"),
              "w") as fh:
        json.dump({"params": vars(args), "results": res,
                   "analysis": {"g1": g1, "g2": g2, "g3": g3,
                                "g1_margin": g1m, "g2_margin": g2m,
                                "g3_margin": g3m}}, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
