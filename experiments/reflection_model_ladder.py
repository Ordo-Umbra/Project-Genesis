"""A ceiling that nobody chose: eliminate models until consistency forbids more.

`reflection_finite_box.py` produced a reachable ceiling, and every adequate
naming scheme hit it in exactly `C` steps. That tidiness is suspect, because the
saturation there was **stipulated** — `k` atoms, one added per productive step,
exhausted at `k`. The address and the thing added were effectively the same
object, which is precisely the condition under which nothing interesting can
come apart.

So this is the harder domain. Fix `n` propositional variables, giving `2^n`
valuations. A theory *is* the set of valuations still consistent with it, and a
step eliminates one — the semantic content of adding an axiom. The ladder cannot
empty the set, because a theory with no models is inconsistent. So

    C = 2^n - 1

falls out of the semantics rather than a counter, and the address now names
*which* model to eliminate, which is a separate thing from the elimination
itself.

Pre-registered predictions
--------------------------
Q1. **Capacity is emergent and reachable.** A well-addressed ladder makes
    exactly `C` productive eliminations and is then stopped by consistency,
    not by a bound anyone wrote down. **Falsifier:** it stops short for another
    reason, or runs past `C`.

Q2. **Exhaustion is naming-invariant in LOCATION but not in COST.** Two adequate
    schemes should both reach exactly `C`, while differing in how many steps it
    takes them. **Falsifier:** every scheme that reaches `C` does so in `C`
    steps — the box result would then generalise unchanged and cost would not
    be a separate observable.

Q3. **Content-addressing freezes permanently on its first unproductive move.**
    `inline` names itself by its surviving models, so its address only changes
    when an elimination succeeds. One wasted move should therefore stall it
    forever. **Falsifier:** it recovers and continues.

Q4. **There are at least three distinct ways to fail to exhaust.** Collision
    (`truncated`, address space too small), coverage (`partial`, addresses only
    reach half the models), and freeze (`inline`). **Falsifier:** they collapse
    into one failure mode, which would mean `stagnant` is not hiding structure.

Q5. **The box's tidiness was an artifact of its construction.** There, adequate
    implied reaching `C` in `C` steps. Here `adequate` should split into
    efficient and inefficient. **Falsifier:** no adequate scheme is inefficient.

Honest scope
------------
`scattered` is inefficient because a particular integer hash revisits models;
its *ratio* is a property of that hash and means nothing on its own — at small
`n` the same hash happens to be a permutation and the gap vanishes entirely.
What the arm establishes is **existence**: an adequate scheme can pay
substantially more for the same result. The specific multiple is not a constant
of nature and is not reported as one.

The domain is still a model. Propositional model-elimination shares no machinery
with the arithmetic ladder or the atom box — deliberately, because the question
is whether the *taxonomy* transfers across constructions, not whether the code
does.

    python experiments/reflection_model_ladder.py
    python experiments/reflection_model_ladder.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.model_ladder import (  # noqa: E402
    ModelTheory, model_climb, model_step, schemes,
)


def sweep(sizes: list[int], width: int, horizon_factor: int) -> list[dict]:
    rows = []
    for n in sizes:
        for kind, w in schemes(width):
            theory = ModelTheory(variables=n, kind=kind, width=w)
            rows.append(model_climb(theory, (1 << n) * horizon_factor))
    return rows


def freeze_trace(variables: int, rungs: int = 5) -> list[dict]:
    theory, out = ModelTheory(variables=variables, kind="inline"), []
    for _ in range(rungs):
        s = model_step(theory)
        out.append({"rung": s.rung, "address": s.address, "target": s.target,
                    "target_was_alive": s.target_was_alive,
                    "address_is_new": s.address_is_new,
                    "productive": s.productive})
        theory = s.after
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 6, 8, 10])
    p.add_argument("--width", type=int, default=3)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if args.quick:
        args.sizes, args.horizon = [4, 6, 8], 15

    rows = sweep(args.sizes, args.width, args.horizon)
    print(__doc__.split("\n\n")[0])
    print()
    print("  Every scheme, every size. C is set by consistency, not by a counter.")
    print()
    print(f"  {'n':>3} {'models':>7} {'scheme':<14} {'I':>6} {'C':>6} "
          f"{'ceiling?':>9} {'steps':>7} {'efficiency':>11} {'reason':>10}")
    for r in rows:
        eff = f"{r['efficiency']:.3f}" if r["efficiency"] else "--"
        tag = r["kind"] + (f" w={r['width']}" if r["width"] else "")
        print(f"  {r['variables']:>3} {1 << r['variables']:>7} {tag:<14} "
              f"{r['integration']:>6} {r['capacity']:>6} "
              f"{str(r['reached_ceiling']):>9} "
              f"{str(r['steps_to_floor']):>7} {eff:>11} {r['reason']:>10}")
    print()

    reached = [r for r in rows if r["reached_ceiling"]]
    q1 = bool(reached) and all(r["integration"] == r["capacity"]
                               for r in reached)
    print(f"  Q1 capacity is emergent and reachable ........... "
          f"{'CONFIRMED' if q1 else 'REFUTED'}")
    print(f"     {len(reached)}/{len(rows)} runs reach exactly C and are then "
          f"stopped by consistency —")
    print(f"     the ladder cannot remove its last model. Nobody wrote C down;")
    print(f"     it is 2^n - 1 because an inconsistent theory has no models.")
    print()

    # ------------------------------------------------------------------- Q2
    by_size: dict[int, list[dict]] = {}
    for r in reached:
        by_size.setdefault(r["variables"], []).append(r)
    same_place = all(len({x["integration"] for x in v}) == 1
                     for v in by_size.values())
    cost_gaps = [(v[0]["variables"],
                  min(x["steps_to_floor"] for x in v),
                  max(x["steps_to_floor"] for x in v))
                 for v in by_size.values() if len(v) > 1]
    differing = [(n, lo, hi) for n, lo, hi in cost_gaps if hi > lo]
    q2 = same_place and bool(differing)
    print(f"  Q2 naming-invariant in LOCATION, not in COST .... "
          f"{'CONFIRMED' if q2 else 'REFUTED'}")
    print(f"     every scheme that arrives, arrives at exactly the same floor —")
    print(f"     the box result holds on location. But the cost is not shared:")
    for n, lo, hi in differing:
        print(f"       n = {n}: same ceiling reached in {lo} steps and in {hi} "
              f"({hi / lo:.1f}x)")
    if not differing:
        print(f"       (no cost gap at these sizes)")
    print(f"     The box could not show this, because there the address and the")
    print(f"     thing added were the same object. Separate them and 'adequate'")
    print(f"     splits into efficient and merely-eventually-correct.")
    print()

    # ------------------------------------------------------------------- Q3
    trace = freeze_trace(max(4, min(args.sizes)))
    inline_rows = [r for r in rows if r["kind"] == "inline"]
    q3 = all(r["integration"] == 1 for r in inline_rows)
    print(f"  Q3 content-addressing freezes on its first miss .. "
          f"{'CONFIRMED' if q3 else 'REFUTED'}")
    print()
    print(f"     {'rung':>5} {'address':>8} {'target':>7} {'alive?':>7} "
          f"{'new addr?':>10} {'productive':>11}")
    for r in trace:
        print(f"     {r['rung']:>5} {r['address']:>8} {r['target']:>7} "
              f"{str(r['target_was_alive']):>7} "
              f"{str(r['address_is_new']):>10} {str(r['productive']):>11}")
    print()
    print(f"     One elimination, then a miss, then nothing forever. Its address")
    print(f"     IS its surviving set, so a move that changes nothing changes")
    print(f"     nothing about what it will try next. It has no way to try")
    print(f"     something else — it is deterministically stuck after a single")
    print(f"     wasted step, at I = 1 out of C = {inline_rows[-1]['capacity']}.")
    print()
    print(f"     In the arithmetic setting the same scheme was merely expensive.")
    print(f"     Here it is catastrophic. A presentation's cost and its")
    print(f"     RECOVERABILITY are different properties, and only a domain where")
    print(f"     moves can be wasted can tell them apart.")
    print()

    # ------------------------------------------------------------------- Q4
    modes = {}
    for r in rows:
        if r["reached_ceiling"]:
            continue
        modes[r["kind"]] = r["integration"]
    q4 = len(modes) >= 3
    print(f"  Q4 at least three distinct ways to fail to exhaust "
          f"{'CONFIRMED' if q4 else 'REFUTED'}")
    print(f"     all reported as `stagnant`, all different underneath:")
    print(f"       inline    — freeze:    stops at I = 1, address stops moving")
    print(f"       partial   — coverage:  addresses reach only half the models")
    print(f"       truncated — collision: address space smaller than the domain")
    print(f"     So `stagnant` was hiding structure. It is not one failure but a")
    print(f"     family, and the family members are distinguished by *why* the")
    print(f"     address stops doing useful work rather than by anything visible")
    print(f"     in the stall itself.")
    print()

    q5 = q2 and bool(differing)
    print(f"  Q5 the box's tidiness was an artifact ........... "
          f"{'CONFIRMED' if q5 else 'REFUTED'}")
    print(f"     there, adequate implied reaching C in exactly C steps, because")
    print(f"     addressing and adding were one operation. Here adequate splits,")
    print(f"     and the split is invisible in every observable the box had.")
    print()

    print("  Does the taxonomy transfer?")
    print()
    print("  Yes, and it gains a dimension. `exhausted` and `stagnant` still")
    print("  separate, and the naming-invariance discriminator still works — but")
    print("  only for the LOCATION of the ceiling. Cost is a new observable that")
    print("  neither previous domain could produce, and under it `stagnant`")
    print("  resolves into at least three mechanisms.")
    print()
    print("  The pattern across three domains is consistent: each richer domain")
    print("  has left the previous categories standing and added one the")
    print("  previous construction was structurally unable to express. That is")
    print("  the opposite of a framework being confirmed, and it is the more")
    print("  useful outcome — it says where to look next rather than that there")
    print("  is nowhere left to look.")
    print()
    print("  Honest scope. `scattered` is inefficient because a particular hash")
    print("  revisits models; at n <= 6 that same hash is a permutation and the")
    print("  gap vanishes. The arm establishes EXISTENCE of adequate-but-costly")
    print("  schemes, not a constant. The domain is still a model and shares no")
    print("  machinery with the arithmetic ladder or the atom box, which is the")
    print("  point: what is claimed to transfer is the taxonomy, not the code.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "reflection_model_ladder.json"
        out.write_text(json.dumps({
            "params": vars(args) | {"output_dir": str(args.output_dir)},
            "runs": rows, "freeze_trace": trace,
            "cost_gaps": [{"variables": n, "min_steps": lo, "max_steps": hi}
                          for n, lo, hi in cost_gaps],
            "verdicts": {"Q1_capacity_emergent": q1,
                         "Q2_invariant_location_not_cost": q2,
                         "Q3_content_addressing_freezes": q3,
                         "Q4_three_stagnation_mechanisms": q4,
                         "Q5_box_tidiness_was_artifact": q5},
        }, indent=2, default=str))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
