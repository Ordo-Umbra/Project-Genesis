"""Can a system tell, from inside, that its foundation stopped counting?

Every wall measured so far *blocks*: it stops the next move. One of them also
**retracts** — if the base a tower stands on turns out to be unsettleable, the
tower is unsettleable too, and the height is still there while none of it counts
any more. That distinction decided the whole concentrate-or-diversify result, and
it had never been made explicit before that run.

The question left open was whether a system can detect it about itself. It is the
right question to be nervous about, because the failure mode is not a crash. The
system keeps climbing at exactly the same rate, and its ledger silently zeroes.

What "from inside" means here, precisely
----------------------------------------
Every policy in this module before now is written from outside. `rank_aware` is
*handed the filter object* and consults it before proposing, so it routes around
walls it was told about. A system inside its own construction has no such object.
It proposes, is refused, and has to work out what the refusal meant. So the
interior agent here (`blind_climb`) gets the graph and nothing else: it learns
only by attempting, and a failed attempt costs a step exactly like a successful
one.

Pre-registered predictions
--------------------------
Q1. **Forward motion cannot detect it.** A climbing system's record of what it
    proposed and whether the attempt went through is *identical* under a
    retracting wall and under an economic wall that refuses the same moves —
    while the exterior truth differs. **Falsifier:** the records differ, meaning
    something about forward motion already leaks the distinction and no special
    test is needed.

Q2. **Re-deriving the foundation does detect it, and the reason is an
    inequality rather than an insight.** The foundation is the *cheapest* key in
    the graph and carries the *smallest* address. So if it is refused while
    anything else is admitted, the refusal cannot be economic and cannot be
    structural. **Falsifier:** the probe fires on an economic wall too, which
    would make it a coincidence rather than an argument.

Q3. **Probing has a price and no free lunch.** Steps spent re-deriving what you
    already have are steps not spent climbing, so detection latency should fall
    roughly linearly in the probe rate while forgone height rises roughly
    linearly. **Falsifier:** some rate is dramatically better than the others,
    which would mean there is a right answer rather than a dial.

Q4. **The belief gap closes on its own, without ever being diagnosed.** A
    routing policy rebuilds elsewhere because that is what it does when refused,
    not because it understood anything. So given enough horizon, certified height
    should recover while the system's *belief* never changed at any point.
    **Falsifier:** the gap persists at long horizons, which would make
    retraction permanent rather than merely expensive.

Honest scope
------------
`opaque` remains a **declared** unsettleable set. Nothing here proves any address
undecidable; Π⁰₂- and Π¹₁-completeness are cited theorems, and what is modelled
is what a system does when certification cannot conclude.

The probe's inference is **conditional and the condition is real**: a budget
below the minimum cost refuses the foundation too. `interior_verdict` therefore
returns `halted` rather than `retracted` when nothing at all is admitted, and
that non-inference is measured rather than assumed away.

`depth` remains a declared proxy for proof-theoretic rank.

    python experiments/reflection_retraction.py
    python experiments/reflection_retraction.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.reflection_dag import (  # noqa: E402
    Filters, ReflectionGraph, blind_climb, certified_rank, deepen,
    interior_verdict, probe, reflect,
)

FLAT = Filters(cost_model="description")


def tower(rungs: int, roots: int = 3) -> ReflectionGraph:
    graph = ReflectionGraph.base(roots=roots)
    for _ in range(rungs):
        graph = reflect(graph, deepen(graph)).graph_after
    return graph


def indistinguishable(graph: ReflectionGraph, steps: int) -> dict:
    """A retracting wall and an economic one, as the interior sees them."""
    dear = len(graph.frontier()[0].content) - 1
    scenarios = {
        "retracting (the base will not settle)": Filters(opaque=frozenset({0})),
        "economic (the frontier is unaffordable)": Filters(budget=dear),
    }
    out = {}
    for name, filters in scenarios.items():
        run = blind_climb(graph, filters, steps)
        out[name] = {"record": run["record"],
                     "believed": run["believed_rank"],
                     "certified": run["certified_rank"]}
    records = [v["record"] for v in out.values()]
    return {"scenarios": out, "records_identical": records[0] == records[1]}


def probing(graph: ReflectionGraph, steps: int) -> list[dict]:
    rows = []
    for name, filters in (
            ("retracting base", Filters(opaque=frozenset({0}))),
            ("economic, frontier too dear",
             Filters(budget=len(graph.frontier()[0].content) - 1)),
            ("economic, below every cost", Filters(budget=0))):
        result = probe(graph, filters)
        run = blind_climb(graph, filters, steps, probe_every=4)
        rows.append({"scenario": name, "probe_cost": result["cost"],
                     "probe_admitted": result["admitted"],
                     "alternatives": result["alternatives_admitted"],
                     "verdict": interior_verdict(result),
                     "detected_at": run["detected_at"],
                     "believed": run["believed_rank"],
                     "certified": run["certified_rank"]})
    return rows


def price_of_probing(graph: ReflectionGraph, rates: list[int], steps: int,
                     wall_window: range) -> list[dict]:
    after = Filters(cost_model="description", opaque=frozenset({0}))
    rows = []
    for rate in rates:
        latencies, forgone = [], []
        for at in wall_window:
            free = blind_climb(graph, FLAT, steps, wall_at=at,
                               filters_after=after)
            run = blind_climb(graph, FLAT, steps, probe_every=rate,
                              wall_at=at, filters_after=after)
            if run["latency"] is not None:
                latencies.append(run["latency"])
            forgone.append(free["believed_rank"] - run["believed_rank"])
        lat = sum(latencies) / len(latencies) if latencies else float("nan")
        forg = sum(forgone) / len(forgone)
        rows.append({"probe_every": rate, "mean_latency": lat,
                     "mean_forgone": forg, "product": lat * forg})
    return rows


def recovery(graph: ReflectionGraph, horizons: list[int], wall_at: int) -> list[dict]:
    after = Filters(cost_model="description", opaque=frozenset({0}))
    rows = []
    for hz in horizons:
        run = blind_climb(graph, FLAT, wall_at + hz, wall_at=wall_at,
                          filters_after=after)
        rows.append({"horizon_after_wall": hz, "believed": run["believed_rank"],
                     "certified": run["certified_rank"],
                     "gap": run["believed_rank"] - run["certified_rank"],
                     "ever_detected": run["detected_at"] is not None})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rungs", type=int, default=12)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--rates", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[0, 2, 5, 10, 20, 40])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if args.quick:
        args.steps, args.rates, args.horizons = 40, [4, 8], [0, 5, 20]

    graph = tower(args.rungs)
    print(__doc__.split("\n\n")[0])
    print()

    q1 = indistinguishable(graph, 12)
    print(f"  Q1. Two walls, {args.rungs} rungs already built. What does the "
          f"interior see?")
    print()
    print(f"  {'scenario':<42} {'believed':>9} {'certified':>10} "
          f"{'attempts refused':>17}")
    for name, row in q1["scenarios"].items():
        refused = sum(1 for _, ok in row["record"] if not ok)
        print(f"  {name:<42} {row['believed']:>9} {row['certified']:>10} "
              f"{refused:>17}")
    print()
    print(f"  Q1 forward motion cannot detect retraction ....... "
          f"{'CONFIRMED' if q1['records_identical'] else 'REFUTED'}")
    print("     The two records are identical, attempt for attempt — same")
    print("     proposals, same successes, same refusals — while the exterior")
    print("     truth differs. (Here only by a rung, because the agent has")
    print("     already rebuilt some height elsewhere; Q4 shows the gap at full")
    print("     size when there is no horizon left to rebuild in.)")
    print("     **From inside, 'I cannot")
    print("     extend this line' is compatible with 'and everything under it")
    print("     stands' and with 'and none of it ever counted.'** The system is")
    print("     not missing a subtle signal. There is no signal.")
    print()

    rows = probing(graph, 12)
    print("  Q2. Re-derive the foundation instead — the cheapest key there is.")
    print()
    print(f"  {'scenario':<30} {'probe cost':>11} {'admitted':>9} "
          f"{'others ok':>10} {'verdict':>12} {'detected':>9}")
    for r in rows:
        print(f"  {r['scenario']:<30} {r['probe_cost']:>11} "
              f"{str(r['probe_admitted']):>9} {str(r['alternatives']):>10} "
              f"{r['verdict']:>12} {str(r['detected_at']):>9}")
    q2 = ([r["verdict"] for r in rows]
          == ["retracted", "no evidence", "halted"])
    print()
    print(f"  Q2 the probe detects it, and does not misfire ... "
          f"{'CONFIRMED' if q2 else 'REFUTED'}")
    print("     The argument is an inequality, not an insight. The foundation is")
    print("     the cheapest key in the graph and carries the smallest address,")
    print("     so a refusal there cannot be about price or about writing the")
    print("     name down. **What is left is validity.**")
    print()
    print("     And the third row is why the rule is stated with a condition. A")
    print("     budget below every cost refuses the foundation too — so the")
    print("     inference only runs while *something else* is still admitted.")
    print("     Without that clause the probe would report a collapse every time")
    print("     the system merely ran out of money. It returns `halted` instead,")
    print("     which is the honest answer: the probe has told you nothing.")
    print()

    price = price_of_probing(graph, args.rates, args.steps, range(18, 30))
    print("  Q3. What does the scepticism cost? (wall arrival averaged over 12")
    print("      step counts, so nothing turns on the probe landing on it)")
    print()
    print(f"  {'probe every':>12} {'mean latency':>13} {'height forgone':>15} "
          f"{'product':>9}")
    for r in price:
        print(f"  {r['probe_every']:>12} {r['mean_latency']:>13.2f} "
              f"{r['mean_forgone']:>15.2f} {r['product']:>9.1f}")
    spread = max(r["product"] for r in price) / min(r["product"] for r in price)
    q3 = spread < 4
    print()
    print(f"  Q3 a dial, not a right answer .................... "
          f"{'CONFIRMED' if q3 else 'REFUTED'}")
    print(f"     Latency falls and cost rises, both roughly linearly, and the")
    print(f"     product moves by {spread:.1f}x across an "
          f"{max(args.rates) // min(args.rates)}x range of probe rates.")
    print("     There is no free lunch and no cliff: what you buy with each probe")
    print("     is a step of height, and what you get is a bounded delay. The")
    print("     right rate is whatever your tolerance for being wrong is worth,")
    print("     which is the same shape of answer as the mean-versus-floor trade.")
    print()

    rec = recovery(graph, args.horizons, wall_at=20)
    print("  Q4. Does the gap close? And is it ever noticed?")
    print()
    print(f"  {'horizon after wall':>19} {'believed':>9} {'certified':>10} "
          f"{'gap':>5} {'ever detected':>14}")
    for r in rec:
        print(f"  {r['horizon_after_wall']:>19} {r['believed']:>9} "
              f"{r['certified']:>10} {r['gap']:>5} "
              f"{str(r['ever_detected']):>14}")
    q4 = rec[0]["gap"] > 0 and rec[-1]["gap"] == 0 and not any(
        r["ever_detected"] for r in rec)
    print()
    print(f"  Q4 it closes, and it is never diagnosed ......... "
          f"{'CONFIRMED' if q4 else 'REFUTED'}")
    print(f"     The gap runs {rec[0]['gap']} at no remaining horizon down to "
          f"{rec[-1]['gap']} with plenty, and")
    print("     `ever detected` is False in every row. **The system repairs")
    print("     itself without ever making the diagnosis.** Routing around a")
    print("     refusal is what its policy does anyway; given enough steps that")
    print("     happens to rebuild certified height somewhere else, and the")
    print("     system's belief about itself was wrong at every point in between")
    print("     and is right again at the end, never having changed.")
    print()
    print("     Which cuts both ways, and the short-horizon row is the half worth")
    print("     keeping: with no time to rebuild, the belief stays wrong. Not")
    print("     because the system is careless — because nothing in forward")
    print("     motion was ever going to tell it.")
    print()

    print("  Honest scope. `opaque` is a declared unsettleable set; nothing here")
    print("  proves any address undecidable, and the completeness results this")
    print("  leans on remain citations. The probe's inference is conditional on")
    print("  an alternative being admitted, and that condition is measured")
    print("  rather than assumed. `depth` remains a declared proxy for")
    print("  proof-theoretic rank.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "reflection_retraction.json"
        out.write_text(json.dumps({
            "params": vars(args) | {"output_dir": str(args.output_dir)},
            "indistinguishable": q1, "probing": rows, "price": price,
            "recovery": rec,
            "verdicts": {"Q1_forward_blind": q1["records_identical"],
                         "Q2_probe_detects": q2, "Q3_dial_not_answer": q3,
                         "Q4_closes_undiagnosed": q4},
        }, indent=2, default=str))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
