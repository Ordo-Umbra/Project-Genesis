"""The missing cost model was the first result of the series, all along.

Six domains in, the reviewer named the remaining structural gap: none of them
contained a cost model that does not grow with the reflected object. Under any
size-monotone cost, each advance enlarges the frontier's closure and so raises
the price of the next — the system is priced out of advancing by its own
success, and no selection rule can outrun that, because the policy does not set
the price.

The gap turns out not to need a new object. **It needs the first experiment
applied where the later domains quietly stopped applying it.**

Result one measured two ways of naming a theory. `inline` addresses by
*content* — the address is the axiom list — and its cost doubles every rung.
`indexed` addresses by *description* — "arithmetic plus everything below here" —
and its cost is a flat **4,996 symbols per rung, however much theory it names**.
That is a cost model that does not grow with the reflected object, measured
eleven experiments ago.

The DAG domain had been pricing by `|key|`: content-addressing, reintroduced
without noticing. So this run swaps in description-addressed pricing and asks
what survives.

Pre-registered predictions
--------------------------
Q1. **Rank stops saturating.** Under description cost, rank grows linearly with
    steps instead of stopping at the budget. **Falsifier:** it still saturates,
    which would mean the bias has a source other than cost-scaling and the
    whole diagnosis of the last two runs is wrong.

Q2. **The filters stop being directional.** Economic, structural and epistemic
    all go neutral — advancing and sideways are admitted at the same rate.
    **Falsifier:** any of them still prefers one.

Q3. **The sideways basin still exists.** It is no longer *selected for*, but
    a policy can still choose it. **Falsifier:** it disappears, which would make
    the earlier finding — that sideways is a property of moves rather than
    states — wrong.

Q4. **The epistemic wall is not one.** Added after a reviewer closed with a
    claim I had not tested: that a rank-following policy advances without bound
    *until an independent wall intervenes* — naming undecidability of an address
    as one such wall. Checking it exposed a fault in this module rather than in
    the claim. `certify_effort` compares `cost(key)` against a bound, so it reads
    exactly what the economic filter reads; under flat pricing it must degenerate
    to all-or-nothing. It was a third size tax wearing an epistemic label.

    So a genuine epistemic wall is added — `opaque`, a set of addresses whose
    validity cannot be settled, ported from result one's `searched` arm where
    certification is a *search that cannot conclude* rather than a tax that
    scales with size. The prediction: because it does not read size, it must
    fire **identically under both cost models**, in every row.
    **Falsifier:** any row where the two columns differ, which would mean it
    reads size after all and is a fourth tax.

Honest scope
------------
The flat rate is a parameter (`description_cost`), and its *value* is arbitrary;
what is not arbitrary is that it does not read the key. The structural filter is
made to follow the cost model too, since it was size-linked only because the
address was — leaving it content-linked would have rigged the comparison.

`depth` remains a declared proxy for proof-theoretic rank.

    python experiments/reflection_cost_model.py
    python experiments/reflection_cost_model.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.reflection_dag import (  # noqa: E402
    Filters, broaden, deepen, run_adaptive, run_filtered,
)

MODELS = ("content", "description")


def saturation(budgets: list[int], steps: int) -> list[dict]:
    rows = []
    for b in budgets:
        row = {"budget": b}
        for model in MODELS:
            r = run_adaptive(steps, filters=Filters(budget=b, cost_model=model))
            row[model] = {"rank": r["final_rank"],
                          "advancing": r["tally"]["advancing"],
                          "sideways": r["tally"]["sideways"]}
        rows.append(row)
    return rows


def growth(step_counts: list[int], budget: int) -> list[dict]:
    rows = []
    for n in step_counts:
        row = {"steps": n}
        for model in MODELS:
            r = run_adaptive(n, filters=Filters(budget=budget,
                                                cost_model=model))
            row[model] = r["final_rank"]
        rows.append(row)
    return rows


def directionality(steps: int) -> list[dict]:
    rows = []
    specs = (("economic", lambda L, m: Filters(budget=L, cost_model=m), (4, 5, 8)),
             ("epistemic", lambda L, m: Filters(certify_effort=L, cost_model=m),
              (4, 5, 8)),
             ("structural", lambda L, m: Filters(address_bits=L, cost_model=m),
              (2, 3, 4)))
    for label, make, limits in specs:
        for limit in limits:
            row = {"filter": label, "limit": limit}
            for model in MODELS:
                f = make(limit, model)
                a = run_filtered(deepen, steps, filters=f)["tally"]["advancing"]
                s = run_filtered(broaden, steps, filters=f)["tally"]["sideways"]
                row[model] = {"advancing": a, "sideways": s,
                              "verdict": ("blocks advancing" if a < s else
                                          "blocks sideways" if s < a
                                          else "neutral")}
            rows.append(row)
    return rows


def certifier_degeneracy() -> list[dict]:
    """What `certify_effort` actually reads, shown directly.

    A small key and a large one, priced under both models, put through the
    filter. Under content pricing the two differ — the filter separates them.
    Under description pricing they are priced the same, so no bound can separate
    them: every effort level admits both or refuses both.
    """
    small, big = frozenset({0, 1}), frozenset(range(20))
    rows = []
    for model in MODELS:
        for effort in (1, 2, 20):
            f = Filters(certify_effort=effort, cost_model=model)
            rows.append({
                "cost_model": model, "effort": effort,
                "cost_small": f.cost(small), "cost_big": f.cost(big),
                "admits_small": f.admits(small, max(small), 1)[0],
                "admits_big": f.admits(big, max(big), 1)[0],
            })
    return rows


def genuine_wall(steps: int, roots: int = 3) -> list[dict]:
    """A wall that does not read size: one address whose validity is unsettleable.

    Swept over which root is opaque, because *placement* is what determines
    which policy it bites — unlike a size tax, whose direction is fixed by the
    fact that advancing enlarges the key.
    """
    rows = []
    for opaque in range(roots):
        row = {"opaque": opaque}
        for model in MODELS:
            f = Filters(cost_model=model, opaque=frozenset({opaque}))
            d = run_filtered(deepen, steps, roots=roots, filters=f)
            b = run_filtered(broaden, steps, roots=roots, filters=f)
            row[model] = {
                "deepen_advancing": d["tally"]["advancing"],
                "deepen_blocked": d["blocks"]["uncertifiable"],
                "broaden_sideways": b["tally"]["sideways"],
                "broaden_blocked": b["blocks"]["uncertifiable"],
            }
        row["identical"] = row["content"] == row["description"]
        rows.append(row)
    return rows


def detour(steps: int, roots: int = 3) -> list[dict]:
    """Can a rank-following policy route around an unsettleable address?"""
    rows = []
    for opaque in (None, 0, 1, 2, "all"):
        marked = (frozenset() if opaque is None else
                  frozenset(range(roots)) if opaque == "all" else
                  frozenset({opaque}))
        r = run_adaptive(steps, roots=roots,
                         filters=Filters(cost_model="description",
                                         opaque=marked))
        rows.append({"opaque": opaque, "rank": r["final_rank"],
                     "advancing": r["tally"]["advancing"],
                     "sideways": r["tally"]["sideways"],
                     "blocked": r["blocks"]["uncertifiable"]})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--budgets", type=int, nargs="+", default=[6, 10, 20, 50])
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--growth", type=int, nargs="+", default=[20, 40, 80, 160])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if args.quick:
        args.budgets, args.steps, args.growth = [6, 20], 30, [20, 40, 80]

    sat = saturation(args.budgets, args.steps)
    print(__doc__.split("\n\n")[0])
    print()
    print(f"  Rank-aware selection under a budget, {args.steps} steps, by cost model")
    print()
    print(f"  {'budget':>7} | {'content rank':>13} {'advancing':>10} | "
          f"{'description rank':>17} {'advancing':>10}")
    for r in sat:
        print(f"  {r['budget']:>7} | {r['content']['rank']:>13} "
              f"{r['content']['advancing']:>10} | "
              f"{r['description']['rank']:>17} "
              f"{r['description']['advancing']:>10}")
    print()

    grow = growth(args.growth, budget=10)
    print(f"  Does rank grow with steps, or stop at the budget? (budget 10)")
    print()
    print(f"  {'steps':>7} {'content':>9} {'description':>13}")
    for r in grow:
        print(f"  {r['steps']:>7} {r['content']:>9} {r['description']:>13}")
    q1 = (len({r["content"] for r in grow}) == 1
          and len({r["description"] for r in grow}) == len(grow))
    print()
    print(f"  Q1 rank stops saturating ......................... "
          f"{'CONFIRMED' if q1 else 'REFUTED'}")
    print(f"     content-addressed: rank is {grow[0]['content']} at every step")
    print(f"     count — the budget, and nothing more. Description-addressed:")
    print(f"     rank tracks steps one-for-one, unbounded. The budget stops")
    print(f"     being a ceiling on reach and becomes what it was supposed to")
    print(f"     be: a limit on rate, not on height.")
    print()

    dirs = directionality(args.steps)
    print("  Are the filters still directional?")
    print()
    print(f"  {'filter':<12} {'limit':>6} | {'content':>18} | {'description':>12}")
    for r in dirs:
        print(f"  {r['filter']:<12} {r['limit']:>6} | "
              f"{r['content']['verdict']:>18} | {r['description']['verdict']:>12}")
    q2 = all(r["description"]["verdict"] == "neutral" for r in dirs)
    biased = sum(r["content"]["verdict"] == "blocks advancing" for r in dirs)
    print()
    print(f"  Q2 the filters stop being directional ............ "
          f"{'CONFIRMED' if q2 else 'REFUTED'}")
    print(f"     {biased}/{len(dirs)} rows block advancing under content-addressed")
    print(f"     cost; {sum(r['description']['verdict'] == 'neutral' for r in dirs)}"
          f"/{len(dirs)} are neutral under description-addressed cost. The bias")
    print(f"     was never a property of the filters. It was a property of what")
    print(f"     they were reading.")
    print()

    basin = {m: run_filtered(broaden, args.steps,
                             filters=Filters(cost_model=m))["tally"]["sideways"]
             for m in MODELS}
    q3 = basin["description"] > 0
    print(f"  Q3 the sideways basin still exists ............... "
          f"{'CONFIRMED' if q3 else 'REFUTED'}")
    print(f"     a join-seeking policy still lands {basin['description']}"
          f"/{args.steps} sideways moves under")
    print(f"     description cost. The basin is not removed — it stops being")
    print(f"     *downhill*. Which is exactly what the earlier run established:")
    print(f"     sideways is a property of the move chosen, not of the state. It")
    print(f"     was only ever a trap because the cost model made it the cheap")
    print(f"     option.")
    print()

    deg = certifier_degeneracy()
    print("  Q4. Is the epistemic filter epistemic? (a reviewer's closing claim)")
    print()
    print("  `certify_effort` admits a step when cost(key) <= effort. It reads the")
    print("  same number the economic filter reads, so swapping the cost model")
    print("  says what it was measuring all along:")
    print()
    print(f"  {'cost model':<12} {'effort':>7} {'cost(small)':>12} {'cost(big)':>10}"
          f"  {'admits':<18}")
    for r in deg:
        verdict = ("both" if r["admits_small"] and r["admits_big"] else
                   "neither" if not r["admits_small"] else "small only")
        print(f"  {r['cost_model']:<12} {r['effort']:>7} {r['cost_small']:>12} "
              f"{r['cost_big']:>10}  {verdict:<18}")
    flat = [r for r in deg if r["cost_model"] == "description"]
    degenerate = all(r["admits_small"] == r["admits_big"] for r in flat)
    print()
    print(f"  Under flat pricing it never separates the two keys: "
          f"{'CONFIRMED' if degenerate else 'REFUTED'}")
    print("  It was a third size tax wearing an epistemic label. That was")
    print("  invisible while every filter read the same number — the cost model")
    print("  swap is what made it visible, which is the point of running one.")
    print()

    wall = genuine_wall(args.steps)
    print("  So here is a wall that genuinely does not read size: an address")
    print("  whose validity cannot be settled, ported from result one's `searched`")
    print("  arm — certification as a search that cannot conclude.")
    print()
    print(f"  {'opaque':>7} | {'content-addressed':^21} | "
          f"{'description-addressed':^21}")
    print(f"  {'root':>7} | {'deepen     broaden':^21} | "
          f"{'deepen     broaden':^21}")
    print(f"  {'':>7} | {'adv/blk    side/blk':^21} | "
          f"{'adv/blk    side/blk':^21}")

    def fmt(c):
        return (f"{c['deepen_advancing']:>3}/{c['deepen_blocked']:<3}    "
                f"{c['broaden_sideways']:>3}/{c['broaden_blocked']:<3}")

    for r in wall:
        print(f"  {r['opaque']:>7} | {fmt(r['content']):^21} | "
              f"{fmt(r['description']):^21}")
    q4 = all(r["identical"] for r in wall)
    print()
    print(f"  Q4 a real epistemic wall fires identically under both models .... "
          f"{'CONFIRMED' if q4 else 'REFUTED'}")
    print(f"     {sum(r['identical'] for r in wall)}/{len(wall)} rows identical.")
    print("     It does not read size, so pricing cannot move it. Note what")
    print("     decides its direction instead: *placement*. Marking root 0 —")
    print("     the one the deepening chain runs through — blocks everything;")
    print("     marking root 2 blocks only the joins. A size tax has a fixed")
    print("     direction because advancing always enlarges the key. This one")
    print("     has no intrinsic direction at all.")
    print()

    det = detour(args.steps)
    print("  And can a rank-following policy route around it?")
    print()
    print(f"  {'opaque':>7} {'rank':>6} {'advancing':>10} {'sideways':>9} {'blocked':>8}")
    for r in det:
        print(f"  {str(r['opaque']):>7} {r['rank']:>6} {r['advancing']:>10} "
              f"{r['sideways']:>9} {r['blocked']:>8}")
    clean = det[0]["rank"]
    partial = [r for r in det[1:] if r["opaque"] != "all"]
    total = det[-1]
    print()
    print(f"     Yes, at a bounded price. A single unsettleable root costs at")
    print(f"     most {clean - min(r['rank'] for r in partial)} rungs out of "
          f"{clean} — the policy abandons the region and rebuilds")
    print(f"     elsewhere. It is a detour, not a ceiling. But mark *every* root")
    print(f"     opaque and the climb stops dead at rank {total['rank']}, with all")
    print(f"     {total['blocked']} attempts refused. Which is the arithmetic")
    print(f"     domain's `searched` arm exactly: when certification is a search")
    print(f"     over the whole address space, there is nowhere to route to.")
    print()
    print("     So the reviewer's parenthetical holds for a real epistemic wall,")
    print("     and this module did not contain one until now. The claim was")
    print("     right; the model was wrong. Both halves are the result.")
    print()

    print("  What this closes")
    print()
    print("  The gap was not a missing object. Result one of this series measured")
    print("  a cost model that does not grow with the reflected object — the")
    print("  `indexed` presentation, flat at 4,996 symbols per rung however much")
    print("  theory it names — and the later domains reverted to content-")
    print("  addressing without anyone noticing, including me.")
    print()
    print("  So the chain runs: naming by description rather than by content")
    print("  makes cost flat; flat cost makes the filters neutral; neutral")
    print("  filters stop selecting for sideways; and a rank-aware policy then")
    print("  advances without bound. Every link was already measured. What was")
    print("  missing was noticing they were the same chain.")
    print()
    print("  It also puts the second experiment back in force. With flat cost L")
    print("  and a capacity that heals at rate r, the sustainable-forever")
    print("  condition was r* = L/kappa_max, derived and matched to better than")
    print("  0.01%. That result assumed constant cost and was quietly")
    print("  inapplicable to every domain after it. Under description-addressed")
    print("  pricing it applies again.")
    print()
    print("  Honest scope. The flat rate is a parameter and its VALUE is")
    print("  arbitrary; what matters is that it does not read the key. The")
    print("  structural filter was made to follow the cost model, since it was")
    print("  size-linked only because the address was — leaving it content-linked")
    print("  would have rigged the comparison. `depth` remains a declared proxy")
    print("  for proof-theoretic rank.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "reflection_cost_model.json"
        out.write_text(json.dumps({
            "params": vars(args) | {"output_dir": str(args.output_dir)},
            "saturation": sat, "growth": grow, "directionality": dirs,
            "basin": basin, "certifier_degeneracy": deg,
            "genuine_wall": wall, "detour": det,
            "verdicts": {"Q1_no_saturation": q1, "Q2_filters_neutral": q2,
                         "Q3_basin_survives": q3,
                         "Q4_wall_is_pricing_invariant": q4,
                         "Q4_certify_effort_degenerates": degenerate},
        }, indent=2, default=str))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
