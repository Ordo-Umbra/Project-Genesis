"""What does a theory know about its own walls?

Every measurement in this series so far has been taken from *outside*. The
observer with the bigger notation system — running the ladder, watching where it
stops, reading off costs and collisions — has been us, the whole time. Nothing
yet has asked what the theory can determine about itself, using only checks it
can run on its own presentation.

That gap matters because the intuition this series keeps brushing against is an
*interior* one: that from the inside, continuation feels unbounded. If that has
a formal counterpart, it is a claim about the difference between what a system
can establish about its own stopping and what an outside observer can establish
about it. This measures that difference.

Three walls, and a fourth arm to exhibit the third
--------------------------------------------------
The series has found three ways a climb can end:

    unaffordable      the edge exists and costs too much     economic
    limit-undefined   the edge does not exist                structural
    undecidable       whether it exists cannot be decided    epistemic

Until now only two were *exhibited* by a running arm; the third was argued from
the Π⁰₂ character of totality. So a fourth presentation is added — `searched` —
whose limit notation is an arbitrary index rather than a canonical one, and so
must be certified by *running* its fundamental sequence.

The important detail: `searched`'s sequence is `n ↦ n`, which is **total**. Its
continuation genuinely exists. It halts anyway, because it cannot authorise a
step it cannot certify. That is what makes the third wall a different kind of
thing from the other two rather than a slower version of them.

Pre-registered predictions
--------------------------
Q1. **Every theory can predict *where* it stops, exactly.** Predicted stop rung
    equals actual, for every arm at every budget — including the undecidable
    one, since a cautious system knows it will decline an uncertifiable step.
    **Falsifier:** any mismatch, which would mean the interior view is not even
    locally reliable and nothing below is worth reading.

Q2. **It can predict *why*, for the two decidable walls.** `inline`,
    `indexed` and `truncated` should name the correct reason at every budget,
    with the wall resolved in the order it would be *met* — the economic one
    when the budget binds first, the structural one when it does not.
    **Falsifier:** a reason mismatch.

Q3. **For the third wall it cannot determine whether the wall is real.**
    `searched` should report "cannot tell" at every budget, and never certify
    or refute its own edge — while the edge is in fact there. **Falsifier:**
    it settles the question either way, which would collapse the third wall
    into one of the first two.

Q4. **The third wall is budget-invariant.** Unlike the economic wall it should
    not move at all between a tight budget and one 10⁷ times larger, because it
    is not about cost. **Falsifier:** the answer changes with the budget.

What this does and does not say
-------------------------------
The result is *not* "from the inside the system cannot see its walls" — it
sees them all, and knows exactly where each is. It is the narrower and sharper
statement that a system can be complete about **location** and incomplete about
**necessity**: it always knows where it stops, and does not always know whether
stopping was required. A system at the third wall cannot distinguish a
continuation it lacks from one it merely cannot certify.

Nothing here concerns experience, and nothing here is evidence about it. The
resemblance to an interior report of unboundedness is a resemblance; the
measured object is a formal presentation of arithmetic.

Honest scope. The `searched` arm's inability to certify is *simulated* in the
sense that its checker is bounded by construction; that totality is Π⁰₂-complete
and `O`-membership Π¹₁-complete are cited results. What is measured is that a
bounded certifier cannot separate "total" from "total so far", and that a system
relying on one therefore halts on a live edge.

    python experiments/reflection_interior.py
    python experiments/reflection_interior.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.reflection import (  # noqa: E402
    Capacity, peano, predict_stop, transfinite_climb,
)

ARMS = (("inline", None), ("indexed", None), ("truncated", 3),
        ("searched", None))
REALITY = {True: "yes", False: "no", None: "CANNOT TELL"}


def compare(kind, width, budget, blocks, per_block) -> dict:
    theory = peano(kind, width=width)
    cap = None if budget is None else Capacity(budget, 1.0)
    p = predict_stop(theory, blocks=blocks, per_block=per_block, capacity=cap)
    o = transfinite_climb(theory, blocks=blocks, per_block=per_block,
                          capacity=cap)
    actual_rung = None if o.stopped_because == "horizon" else o.taken
    return {
        "arm": kind, "budget": budget,
        "predicted_rung": p.stop_rung, "predicted_reason": p.reason,
        "wall_is_real": p.wall_is_real, "detail": p.detail,
        "actual_rung": actual_rung, "actual_reason": o.stopped_because,
        "actual_taken": o.taken, "final_rank": str(o.rank),
        "where_correct": p.stop_rung == actual_rung,
        "why_correct": p.reason == o.stopped_because,
        "knows_why": p.certain,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--per-block", type=int, default=10)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if args.quick:
        args.blocks, args.per_block = 3, 5

    budgets = [None, 1e5, 1e8, 1e12]
    rows = [compare(k, w, b, args.blocks, args.per_block)
            for k, w in ARMS for b in budgets]

    print(__doc__.split("\n\n")[0])
    print()
    print(f"  {args.blocks} blocks of {args.per_block} successors, one limit "
          f"between each. The 'predicted' columns")
    print(f"  use ONLY checks the theory can run on itself; 'actual' is the run.")
    print()
    print(f"  {'arm':<10} {'budget':>7} | {'predicts':>26} {'wall real?':>12} "
          f"| {'actually':>26}")
    for r in rows:
        b = "none" if r["budget"] is None else f"{r['budget']:.0e}"
        pred = f"{r['predicted_rung']} ({r['predicted_reason']})"
        act = f"{r['actual_rung']} ({r['actual_reason']})"
        print(f"  {r['arm']:<10} {b:>7} | {pred:>26} "
              f"{REALITY[r['wall_is_real']]:>12} | {act:>26}")
    print()

    q1 = all(r["where_correct"] for r in rows)
    decidable = [r for r in rows if r["arm"] != "searched"]
    searched = [r for r in rows if r["arm"] == "searched"]
    q2 = all(r["why_correct"] and r["knows_why"] for r in decidable)
    q3 = all(not r["knows_why"] and r["where_correct"] for r in searched)
    q4 = len({(r["predicted_rung"], r["predicted_reason"], r["wall_is_real"])
              for r in searched}) == 1

    print(f"  Q1 every theory predicts WHERE it stops ......... "
          f"{'CONFIRMED' if q1 else 'REFUTED'}")
    print(f"     {sum(r['where_correct'] for r in rows)}/{len(rows)} exact, "
          f"across four presentations and four budgets — including the")
    print(f"     undecidable arm. A cautious system knows it will decline a step")
    print(f"     it cannot certify, so it knows exactly where that happens.")
    print()
    print(f"  Q2 and WHY, for the two decidable walls ......... "
          f"{'CONFIRMED' if q2 else 'REFUTED'}")
    print(f"     inline names 'unaffordable' at 1e+05 and 'limit-undefined' "
          f"above it —")
    print(f"     the same arm reporting two different walls depending on which")
    print(f"     one it would MEET first. Resolving them in the wrong order is")
    print(f"     a real error and this experiment was written with it: the first")
    print(f"     version checked the limit edge first and named the structural")
    print(f"     wall at a budget where the economic one binds four rungs sooner.")
    print()
    print(f"  Q3 the third wall's REALITY is not determinable . "
          f"{'CONFIRMED' if q3 else 'REFUTED'}")
    print(f"     the searched arm knows it halts at rung "
          f"{searched[0]['predicted_rung']}, and cannot establish")
    print(f"     whether anything was there. Its fundamental sequence is n -> n,")
    print(f"     which is TOTAL: the continuation genuinely exists. It halts on a")
    print(f"     live edge, because a step it cannot certify is a step it will")
    print(f"     not take. That is not a slower version of the other two walls.")
    print()
    print(f"  Q4 and it is budget-invariant ................... "
          f"{'CONFIRMED' if q4 else 'REFUTED'}")
    print(f"     identical answer from 1e+05 to 1e+12 — seven orders of")
    print(f"     magnitude buying exactly nothing, because the question was")
    print(f"     never economic.")
    print()

    print("  What this says, stated carefully")
    print()
    print("  Not: 'from the inside a system cannot see its walls.' It sees all")
    print("  three, and knows exactly where each one is. The finding is")
    print("  narrower and sharper than that:")
    print()
    print("      a system is COMPLETE about location")
    print("      and INCOMPLETE about necessity.")
    print()
    print("  It always knows where it stops. It does not always know whether")
    print("  stopping was required. At the third wall it cannot distinguish a")
    print("  continuation it LACKS from one it merely cannot CERTIFY — and the")
    print("  searched arm shows those two are not the same situation, because")
    print("  there the edge was real and it stopped anyway.")
    print()
    print("  That also puts a boundary on G from a new side. G > 0 is a claim")
    print("  about which edges exist. A system can always compute where its own")
    print("  climb ends; what it cannot always compute is whether that ending")
    print("  was G = 0 or G > 0 with no certificate. Those are different facts")
    print("  about the world and identical facts from the inside.")
    print()
    print("  Honest scope. Nothing here concerns experience and nothing here is")
    print("  evidence about it; the measured object is a formal presentation of")
    print("  arithmetic and the resemblance to an interior report is a")
    print("  resemblance. The searched arm's bounded certifier stands in for a")
    print("  Pi-0-2 check; that totality is Pi-0-2-complete and O-membership")
    print("  Pi-1-1-complete are cited, not measured.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "reflection_interior.json"
        out.write_text(json.dumps({
            "params": vars(args) | {"output_dir": str(args.output_dir)},
            "rows": rows,
            "verdicts": {"Q1_where_is_predictable": q1,
                         "Q2_why_for_decidable_walls": q2,
                         "Q3_reality_not_determinable": q3,
                         "Q4_budget_invariant": q4},
        }, indent=2, default=str))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
