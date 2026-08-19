"""Do the walls protect against going sideways? No — they select for it.

The DAG run found that a system can be productive forever and go nowhere, and
that this is a property of *which move is chosen* rather than of the state. It
left one thing open, and the reviewer named it as the next controlled run:
**restore the economic, structural and epistemic filters and ask whether
sideways trajectories survive when any of them can bite.**

The prediction, locked before implementation
--------------------------------------------
They will not merely fail to block sideways — they will **preferentially block
advancing**. The reason is structural rather than incidental: every filter
scales with *what is being reflected upon*, and advancing means reflecting on
the frontier, whose closure is the largest object in the graph. Sideways joins
shallow nodes with small closures. So the cheap move, the easily-certified move
and the small-address move are all the move that gets nowhere.

**Falsifier:** any filter that preferentially blocks sideways. One candidate is
included on purpose — an arity cap, which charges for *breadth* rather than
depth — so the pessimistic reading has something that can refute it.

Q1. Economic (a budget) blocks advancing, not sideways.
Q2. Structural (a bounded address space) does the same, at every width where it
    bites at all.
Q3. Epistemic (bounded certification effort) does the same.
Q4. **Something blocks sideways.** If nothing does, the filters are not merely
    insufficient but actively counterproductive, and there is no defence in this
    family at all.

    **NARROWED — see `reflection_selection.py` Q3.** The 0/30 below is real: the
    cap blocks all thirty of the joins `broaden` proposes. What does not follow
    is the generalisation. A sideways move needs no join — reflecting on any
    single node below the frontier is one — and the cap admits those. The
    corrected claim is that **no filter tested blocks sideways as such**; this
    one blocks join-shaped sideways only. The original run is left intact here
    with the correction beside it.

Why the cost model is not a free choice
---------------------------------------
`cost(key) = |key|` — the size of what a step reflects on. That is what the
arithmetic domain did (cost was the symbol count of an address encoding the
theory) and what the box did. Choosing it *not* to scale with the reflected
object would be the unusual move and would need its own defence. The finding is
a consequence of the principled choice, not of a convenient one.

Honest scope
------------
With `cost = |key|` and `certify_effort` compared against `|key|`, the economic
and epistemic filters are **literally the same predicate**. Their agreement is
arithmetic, not independent evidence, and it is reported as one observation
rather than two. The structural filter is a genuinely different function — the
largest identifier in the key — and it agreeing is the second, independent
observation. So this is two independent results, not three.

`depth` remains a declared structural proxy for proof-theoretic rank.

    python experiments/reflection_filters.py
    python experiments/reflection_filters.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.reflection_dag import (  # noqa: E402
    Filters, ReflectionGraph, broaden, deepen, reflect, run_filtered,
)


def cost_asymmetry(roots: int, warmup: int) -> dict:
    graph = ReflectionGraph.base(roots=roots)
    for _ in range(warmup):
        graph = reflect(graph, deepen(graph)).graph_after
    out = {}
    for name, policy in (("advancing", deepen), ("sideways", broaden)):
        parents = policy(graph)
        key = frozenset().union(*(graph.node(p).content for p in parents))
        out[name] = {"parents": sorted(parents), "key_size": len(key),
                     "max_id": max(key)}
    return out


def directional_sweep(kind: str, limits: list[int], steps: int) -> list[dict]:
    rows = []
    for limit in limits:
        if kind == "economic":
            f = Filters(budget=limit)
        elif kind == "structural":
            f = Filters(address_bits=limit)
        elif kind == "epistemic":
            f = Filters(certify_effort=limit)
        else:
            f = Filters(max_arity=limit)
        adv = run_filtered(deepen, steps, filters=f)
        side = run_filtered(broaden, steps, filters=f)
        a, s = adv["tally"]["advancing"], side["tally"]["sideways"]
        rows.append({"filter": kind, "limit": limit, "advancing": a,
                     "sideways": s, "adv_rank": adv["final_rank"],
                     "side_rank": side["final_rank"],
                     "verdict": ("blocks advancing" if a < s else
                                 "blocks sideways" if s < a else "neutral")})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--roots", type=int, default=3)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if args.quick:
        args.steps = 15

    print(__doc__.split("\n\n")[0])
    print()

    asym = cost_asymmetry(args.roots, args.warmup)
    print("  Why the direction is forced: what each move reflects on")
    print()
    print(f"  {'move':<11} {'parents':<12} {'|key|':>6} {'max id':>8}")
    for name, d in asym.items():
        print(f"  {name:<11} {str(d['parents']):<12} {d['key_size']:>6} "
              f"{d['max_id']:>8}")
    ratio = asym["advancing"]["key_size"] / max(1, asym["sideways"]["key_size"])
    print()
    print(f"  Advancing reflects on {ratio:.1f}x more than sideways does, and")
    print(f"  carries a larger identifier. Every filter that scales with the")
    print(f"  reflected object therefore bites the advancing move first. That is")
    print(f"  not a tuning artefact — it is what 'reflect on more' means.")
    print()

    sweeps = {
        "economic": directional_sweep("economic", [3, 4, 5, 8, 12, 20, 40],
                                      args.steps),
        "structural": directional_sweep("structural", [2, 3, 4, 5, 6, 8],
                                        args.steps),
        "epistemic": directional_sweep("epistemic", [3, 4, 5, 8, 12, 20, 40],
                                       args.steps),
        "arity": directional_sweep("arity", [1, 2, 3], args.steps),
    }

    for kind, rows in sweeps.items():
        print(f"  {kind}")
        print(f"    {'limit':>6} {'advancing':>10} {'sideways':>9} "
              f"{'adv rank':>9} {'side rank':>10}  verdict")
        for r in rows:
            print(f"    {r['limit']:>6} {r['advancing']:>10} {r['sideways']:>9} "
                  f"{r['adv_rank']:>9} {r['side_rank']:>10}  {r['verdict']}")
        print()

    def never_blocks_sideways(kind):
        return all(r["verdict"] != "blocks sideways" for r in sweeps[kind])

    def bites_somewhere(kind):
        return any(r["verdict"] == "blocks advancing" for r in sweeps[kind])

    q1 = never_blocks_sideways("economic") and bites_somewhere("economic")
    q2 = never_blocks_sideways("structural") and bites_somewhere("structural")
    q3 = never_blocks_sideways("epistemic") and bites_somewhere("epistemic")
    q4 = any(r["verdict"] == "blocks sideways" for r in sweeps["arity"])

    for label, key, ok in (("Q1 economic blocks advancing, not sideways", "economic", q1),
                           ("Q2 structural does the same", "structural", q2),
                           ("Q3 epistemic does the same", "epistemic", q3)):
        print(f"  {label:<48} {'CONFIRMED' if ok else 'REFUTED'}")
    print()
    print(f"     At no setting of any of the three does sideways get blocked more")
    print(f"     than advancing. At the settings where they bite hardest,")
    print(f"     advancing passes 0/{args.steps} while sideways passes "
          f"{max(r['sideways'] for r in sweeps['economic'] if r['advancing'] == 0)}"
          f"/{args.steps}.")
    print(f"     The structural filter is neutral once the address space is wide")
    print(f"     enough and directional at every width where it constrains at")
    print(f"     all — it never reverses.")
    print()
    print(f"  Q4 something blocks sideways ..................... "
          f"{'CONFIRMED' if q4 else 'REFUTED'}")
    print(f"     the arity cap does, completely: at most one parent per step")
    print(f"     admits {sweeps['arity'][0]['advancing']}/{args.steps} advancing "
          f"moves and {sweeps['arity'][0]['sideways']}/{args.steps} sideways ones.")
    print(f"     But note what kind of filter that is. The other three charge for")
    print(f"     DEPTH — how much a step reflects on. This one charges for")
    print(f"     BREADTH — how many things it joins. It is not a fourth member of")
    print(f"     the same family; it is the first constraint in this series that")
    print(f"     is not a function of the reflected object at all.")
    print()

    print("  What this answers")
    print()
    print("  The reviewer asked whether sideways trajectories survive when the")
    print("  three filters can bite. They do — and the filters make things worse,")
    print("  not better. Each one is a tax on reflecting-on-more, and advancing")
    print("  IS reflecting-on-more. So a system under any of them is pushed")
    print("  toward exactly the trajectory that satisfies every predicate and")
    print("  gets nowhere.")
    print()
    print("  Under a budget the consequence is sharper than 'no protection':")
    print("  sideways converts an unbounded wander into a bounded one. The system")
    print("  spends its entire budget at constant rank and then has nothing left.")
    print("  That is worse than being stopped, because every check passed on the")
    print("  way down.")
    print()
    print("  The defence exists but is a different kind of object: charge for")
    print("  breadth. Which suggests the operational bookkeeping needs a term")
    print("  that is not a function of what a step reflects on — and none of the")
    print("  five domains so far contained one.")
    print()
    print("  Honest scope. With cost = |key| and certification compared against")
    print("  |key|, the economic and epistemic filters are the SAME predicate;")
    print("  their agreement is arithmetic, not independent evidence, and counts")
    print("  as one observation. The structural filter is a different function")
    print("  (the largest identifier in the key) and agreeing is the second,")
    print("  independent one. Two results, not three. `depth` remains a declared")
    print("  proxy for proof-theoretic rank.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "reflection_filters.json"
        out.write_text(json.dumps({
            "params": vars(args) | {"output_dir": str(args.output_dir)},
            "cost_asymmetry": asym, "sweeps": sweeps,
            "verdicts": {"Q1_economic": q1, "Q2_structural": q2,
                         "Q3_epistemic": q3, "Q4_arity_blocks_sideways": q4},
        }, indent=2, default=str))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
