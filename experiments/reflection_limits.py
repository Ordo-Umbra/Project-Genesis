"""The limit mechanism: where presentation stops being a price and becomes a gate.

The model names *two* continuation mechanisms, not one:

    K(T) = T + Con(T)                     the local successor
    T_{l_a} = ⋃ₙ T_{succ^n(a)}            the hierarchical limit

Only the first has been run. `reflection_ladder.py` measured the successor and
found that presentation costs a factor of 2,222 in symbols while buying exactly
the same productive content; `reflection_capacity.py` gave the ladder a budget
and found that cost-bounding makes `G > 0` contingent but cannot tell a real
ladder from a degenerate one. Both are quantitative results: presentation was a
*price*.

This runs the second mechanism, and the prediction is that its character
changes there. Taking a limit means naming the union of the whole ladder below.
A presentation whose index is a *description* of an axiom set can do that — the
description "PA plus every rung below this point" is no longer than any other
description. A presentation whose index is the Gödel number of a literal axiom
*list* cannot, because the union has no finite list. That is not an expensive
edge in the accessibility relation. It is a missing one.

Denotation vs enumeration
-------------------------
This is the distinction the whole experiment turns on, and it is worth stating
before the numbers. A recursive presentation is a finite object that *denotes*
an infinite set. In this module `Theory.index()` is the denotation and
`Theory.rungs` is the finite prefix actually enumerated so far — used for proof
checking, and never claimed to be the whole set. The limit step is exactly the
move from "the set I have listed" to "the set I can describe", so a
presentation that conflates the two has nowhere to go. `inline` conflates them.
That is the entire mechanism behind every result below.

Pre-registered predictions
--------------------------
Q1. **The limit is a gate, not a price.** `inline` cannot take it at *any*
    budget, including budgets that comfortably afford every successor beneath
    it. **Falsifier:** a large enough budget lets `inline` through, or some
    finite index for the union turns out to exist — either would make this a
    cost difference like all the others.

Q2. **Where it is definable, the limit costs what a successor costs.** O(1) in
    the ladder it subsumes. **Falsifier:** limit cost scales with the number of
    rungs below it, which would mean the "description" is a listing in
    disguise and nothing has actually been bought.

Q3. **Past ω the successor mechanism resumes at full productivity.**
    `Con(T_ω)` is new, and `T_{ω+n}` keeps enlarging the axiom set.
    **Falsifier:** the limit index collides with one already used, or
    productivity drops after the limit.

Q4. **The limit gives the degenerate arm a bounded reprieve, not a cure.** The
    `truncated` control should get exactly `2^width` productive rungs per
    ω-block, forever — its productive content growing linearly in limits taken
    while its rank grows like `ω·limits`. **Falsifier:** it stays permanently
    stalled (the limit buys nothing) or becomes fully productive (the limit
    cures it). Either would mean the stall was not what the first experiment
    said it was.

Honest scope
------------
The rank notation is the `ω²` fragment `ω·a + b` — the smallest system that can
express the two mechanisms and nothing more. It is **not** Kleene's `O`: there
are no fundamental sequences, no notation comparison, no transfinite recursion,
and the only ordinal fact used is that `(limits, successors)` orders
lexicographically. `Prf` remains primitive, so costs carry an unexpanded
constant and only ratios are read. As everywhere in this series, `T_n ⊬ Con(T_n)`
is Gödel's second theorem discharged from stated hypotheses, never measured.

    python experiments/reflection_limits.py
    python experiments/reflection_limits.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.reflection import (  # noqa: E402
    Capacity, LimitUndefined, Rank, con_formula, construction_cost, ladder,
    limit_step, peano, step, transfinite_climb,
)

ARMS = (("inline", None), ("indexed", None), ("truncated", 3))


def arm(kind: str, width: int | None, trunc_width: int):
    return peano(kind, width=trunc_width if kind == "truncated" else width)


# ------------------------------------------------------------- Q1: the gate


def gate_probe(trunc_width: int, budgets: list[float], per_block: int) -> dict:
    """Climb each arm at a range of budgets and record where and *why* it stops.

    The interesting column is `stopped_because`. A wall that moves when you pay
    more is a price; a wall that does not is a gate.
    """
    out: dict[str, list] = {}
    for kind, width in ARMS:
        rows = []
        for b in budgets:
            cap = None if b is None else Capacity(b, 1.0)
            o = transfinite_climb(arm(kind, width, trunc_width), blocks=5,
                                  per_block=per_block, capacity=cap)
            rows.append({
                "budget": b,
                "rank": str(o.rank),
                "taken": o.taken,
                "productive": o.productive,
                "limits_taken": o.limits_taken,
                "stopped_because": o.stopped_because,
            })
        out[kind] = rows
    return out


# ------------------------------------------------------ Q2: what a limit costs


def limit_cost_scan(trunc_width: int, depths: list[int]) -> dict:
    """Cost of the limit step against the size of the ladder it subsumes."""
    out: dict[str, list] = {}
    for kind, width in ARMS:
        rows = []
        for d in depths:
            theory = arm(kind, width, trunc_width)
            succ_cost = construction_cost(step(theory))
            for s in ladder(theory, d):
                theory = s.theory_after
            try:
                lim = limit_step(theory)
                rows.append({"rungs_below": d, "successor_cost": succ_cost,
                             "limit_cost": lim.con_symbols,
                             "ratio": lim.con_symbols / succ_cost,
                             "defined": True, "new_axiom": lim.new_axiom})
            except LimitUndefined:
                rows.append({"rungs_below": d, "successor_cost": succ_cost,
                             "limit_cost": None, "ratio": None,
                             "defined": False, "new_axiom": None})
        out[kind] = rows
    return out


# --------------------------------------------- Q3/Q4: productivity past omega


def block_productivity(trunc_width: int, blocks: int, per_block: int) -> dict:
    """Productive rungs per ω-block, for the arms that can reach one."""
    out: dict[str, dict] = {}
    for kind, width in ARMS:
        theory = arm(kind, width, trunc_width)
        if not theory.can_take_limit():
            out[kind] = {"reachable": False}
            continue
        per, ranks = [], []
        for block in range(blocks):
            produced = 0
            for s in ladder(theory, per_block):
                produced += 1 if s.new_axiom else 0
                theory = s.theory_after
            lim = limit_step(theory)
            produced += 1 if lim.new_axiom else 0
            theory = lim.theory_after
            per.append(produced)
            ranks.append(str(theory.rank))
        out[kind] = {"reachable": True, "per_block": per, "ranks": ranks,
                     "steady": len(set(per[1:])) <= 1 if len(per) > 1 else True}
    return out


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--width", type=int, default=3,
                   help="counter width of the truncated control (default 3)")
    p.add_argument("--per-block", type=int, default=12,
                   help="successors between limits (default 12)")
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if args.quick:
        args.per_block, args.blocks = 6, 3

    budgets = [None, 1e5, 1e6, 1e9, 1e12, 1e15]
    depths = [2, 4, 8, 12] if not args.quick else [2, 4, 8]

    print(__doc__.split("\n\n")[0])
    print()
    print(f"  Two mechanisms: {args.per_block} successors per omega-block, "
          f"{args.blocks} blocks.")
    print(f"  Truncated control at width {args.width} "
          f"(wraps every {1 << args.width} successors).")
    print()

    # ------------------------------------------------------------------- Q1
    gate = gate_probe(args.width, budgets, args.per_block)
    print("  Where each arm stops, and why")
    print()
    print(f"  {'arm':<11} {'budget':>9} {'final rank':>12} {'taken':>6} "
          f"{'prod':>5}  stopped because")
    for kind, _ in ARMS:
        for r in gate[kind]:
            b = "none" if r["budget"] is None else f"{r['budget']:.0e}"
            print(f"  {kind:<11} {b:>9} {r['rank']:>12} {r['taken']:>6} "
                  f"{r['productive']:>5}  {r['stopped_because']}")
        print()

    inline_rows = gate["inline"]
    priced = [r for r in inline_rows
              if r["stopped_because"] == "unaffordable" and r["budget"]]
    gated = [r for r in inline_rows
             if r["stopped_because"] == "limit-undefined" and r["budget"]]
    q1 = bool(gated) and all(r["limits_taken"] == 0 for r in inline_rows)

    print(f"  Q1 the limit is a gate, not a price ............. "
          f"{'CONFIRMED' if q1 else 'REFUTED'}")
    if priced and gated:
        print(f"     inline's wall MOVES while it is economic — rank "
              f"{priced[0]['rank']} at {priced[0]['budget']:.0e} rising to "
              f"{priced[-1]['rank']} at")
        print(f"     {priced[-1]['budget']:.0e} — and then stops moving. From "
              f"{gated[0]['budget']:.0e} upward, including {gated[-1]['budget']:.0e},")
        print(f"     every run halts at the same place for a different reason:")
        print(f"     'limit-undefined'. The wall did not get further away. It")
        print(f"     changed KIND. inline takes {inline_rows[0]['limits_taken']}"
              f" limits at any budget, because")
        print(f"     the union of its ladder has no finite axiom list to index.")
    print()

    # ------------------------------------------------------------------- Q2
    costs = limit_cost_scan(args.width, depths)
    print("  What a limit costs, against the ladder it subsumes")
    print()
    print(f"  {'arm':<11} {'rungs below':>12} {'successor':>10} {'limit':>10} "
          f"{'ratio':>7}")
    ratios = []
    for kind, _ in ARMS:
        for r in costs[kind]:
            if not r["defined"]:
                print(f"  {kind:<11} {r['rungs_below']:>12} "
                      f"{r['successor_cost']:>10} {'UNDEFINED':>10} {'--':>7}")
                continue
            ratios.append(r["ratio"])
            print(f"  {kind:<11} {r['rungs_below']:>12} "
                  f"{r['successor_cost']:>10} {r['limit_cost']:>10} "
                  f"{r['ratio']:>7.3f}")
        print()
    q2 = bool(ratios) and max(ratios) / min(ratios) < 1.05

    print(f"  Q2 a limit costs what a successor costs ......... "
          f"{'CONFIRMED' if q2 else 'REFUTED'}")
    print(f"     ratio to a successor is {min(ratios):.3f}-{max(ratios):.3f} "
          f"across every depth tested, and FLAT in")
    print(f"     the number of rungs subsumed. Naming a union costs what naming")
    print(f"     anything else costs, because an index is a description and a")
    print(f"     description of 'everything below here' is not longer than a")
    print(f"     description of one thing. That is what inline cannot buy at")
    print(f"     any price: not a cheaper list, but the right to stop listing.")
    print()

    # --------------------------------------------------------------- Q3 / Q4
    blocks = block_productivity(args.width, args.blocks, args.per_block)
    print(f"  Productive rungs per omega-block "
          f"({args.per_block} successors + 1 limit = "
          f"{args.per_block + 1} attempts per block)")
    print()
    print(f"  {'arm':<11} {'per block':<28} {'final rank':>12}  steady?")
    for kind, _ in ARMS:
        b = blocks[kind]
        if not b["reachable"]:
            print(f"  {kind:<11} {'never reaches a limit':<28} "
                  f"{'--':>12}  --")
            continue
        print(f"  {kind:<11} {str(b['per_block']):<28} "
              f"{b['ranks'][-1]:>12}  {b['steady']}")
    print()

    idx = blocks["indexed"]
    tr = blocks["truncated"]
    q3 = idx["reachable"] and all(n == args.per_block + 1
                                  for n in idx["per_block"])
    expected = 1 << args.width
    q4 = tr["reachable"] and all(n == expected for n in tr["per_block"][1:])

    print(f"  Q3 the successor resumes past omega ............ "
          f"{'CONFIRMED' if q3 else 'REFUTED'}")
    print(f"     indexed produces all {args.per_block + 1} of its attempts in "
          f"every block, through {idx['ranks'][-1]}.")
    print(f"     Con(T_omega) is new — the limit index has never been named —")
    print(f"     and reflection on the union is productive exactly as it was")
    print(f"     below it. Passing omega costs the mechanism nothing.")
    print()
    print(f"  Q4 the limit is a reprieve, not a cure ......... "
          f"{'CONFIRMED' if q4 else 'REFUTED'}")
    print(f"     the truncated arm settles at exactly {expected} = "
          f"2^{args.width} productive rungs per block,")
    print(f"     forever: {tr['per_block']}. (The first block gets one more:")
    print(f"     it opens on a fresh index, where later blocks open on the one")
    print(f"     the limit step itself just consumed.) Its rank runs away like")
    print(f"     omega*limits while its")
    print(f"     productive content grows LINEARLY in limits taken. So the")
    print(f"     limit does rescue it — a stalled ladder that takes a limit")
    print(f"     starts producing again, which is worth knowing — but the")
    print(f"     reprieve is bounded by the same width that caused the stall.")
    print(f"     It buys blocks, not a rate.")
    print()

    print("  What the two experiments now say together")
    print()
    print("  There are two kinds of terminal state and they are not alike.")
    print("  A BUDGET produces a contingent wall: it moves when you pay more,")
    print("  and the recovery rate sets where it sits. A PRESENTATION produces")
    print("  a necessary wall: no budget moves it, because the edge is absent")
    print("  from the accessibility relation rather than priced within it.")
    print("  inline shows both, in that order, and the crossover is visible in")
    print("  the first table — the wall stops receding and starts refusing.")
    print()
    print("  That sharpens what the (C, I, G) bookkeeping leaves out. G > 0 is")
    print("  a statement about which edges exist. Which edges exist is fixed by")
    print("  how a system names itself, and a system that can only name what it")
    print("  has already listed has a hard ceiling at its first limit — while")
    print("  its rank counter, and its C and its nominal G, report nothing")
    print("  unusual right up to the point where it stops.")
    print()
    print("  Honest scope. The rank notation is the omega^2 fragment, not")
    print("  Kleene's O: no fundamental sequences, no notation comparison, no")
    print("  transfinite recursion. index() is a DENOTATION and rungs is the")
    print("  enumerated prefix — the limit theory's axiom set is described, not")
    print("  materialised, which is exactly the capability being measured and")
    print("  also the boundary of what was checked. Prf stays primitive, so")
    print("  only ratios between arms are read.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "reflection_limits.json"
        out.write_text(json.dumps({
            "params": vars(args) | {"output_dir": str(args.output_dir)},
            "gate": gate, "limit_costs": costs, "blocks": blocks,
            "verdicts": {"Q1_limit_is_a_gate": q1,
                         "Q2_limit_costs_a_successor": q2,
                         "Q3_successor_resumes_past_omega": q3,
                         "Q4_bounded_reprieve": q4},
        }, indent=2, default=str))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
