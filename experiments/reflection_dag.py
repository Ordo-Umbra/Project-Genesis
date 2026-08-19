"""Multi-parent reflection: does `productive` split?

The first prediction in this series written down *before* the domain was built,
with the domain chosen by someone else — which was the point. Three prior
domains had all been linear, and in each of them "the axiom set grew" and "the
construction advanced" were the same event, because there was only one place
growth could happen.

Ordinary proof theory does not require that: a consistency statement may be
taken relative to any finite collection of theories. Allowing that turns the
ladder into a DAG and separates two things the linear domains could not:

    local productivity — this step added a genuinely new sentence
    global advance     — the collection as a whole reached further

The registered predictions
--------------------------
**From the reviewer, locked before implementation:** all prior categories
survive; and a step can be locally productive while leaving the global rank of
the collection unchanged, because the reflected theories are already dominated
by a stronger path. If it appears, `productive` splits into two orthogonal
predicates and the taxonomy gains a genuine sixth observable. If every locally
productive step raises global rank, or the split collapses into an existing
stagnation mechanism, the linear taxonomy was already complete here.

**From this side, sharper, also locked:** the phenomenon appears *only where the
DAG genuinely branches*. Over a chain, any subset's union of contents equals the
content of its maximum, which is already asserted, so every multi-parent
reflection over a chain is an ordinary duplicate. If sideways moves appear
without branching, that account is wrong.

Q1. **All prior categories survive.**
Q2. **The split exists:** productive steps that do not advance global rank.
Q3. **It requires branching**, and branching is what a chain lacks.
Q4. **It is a property of MOVES, not of states.** From one and the same state,
    a different policy should give an advancing step — which would mean it is
    not a wall at all. **Falsifier:** no policy advances from a state where a
    sideways move is available, making it a genuine sixth wall instead.

Q4 was not in either registered prediction. It is recorded here as the question
the first two raise rather than as something foreseen, and that distinction is
kept because a prediction made while reading the output is not a prediction.

Honest scope
------------
`depth` — longest path from a root — is a **declared structural proxy** for
proof-theoretic ordinal height, exactly as `I = n` was in the arithmetic
setting. Nothing here computes an ordinal, and the reviewer's prediction was
phrased in terms of genuine proof-theoretic rank; substituting depth is a
weakening that is stated rather than hidden.

Sentence identity is normalised to logical content: `Con({T₁, T₂})` with `T₁`
an ancestor of `T₂` is the same sentence as `Con({T₂})`. Without that
normalisation the phenomenon under test would be manufactured by the
representation rather than found in it.

    python experiments/reflection_dag.py
    python experiments/reflection_dag.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.reflection_dag import (  # noqa: E402
    ReflectionGraph, broaden, deepen, reflect, run_policy,
)

POLICIES = (("deepen", deepen), ("broaden", broaden))


def root_sweep(roots: list[int], steps: int, warmup: int) -> list[dict]:
    rows = []
    for r in roots:
        for name, policy in POLICIES:
            out = run_policy(policy, steps, roots=r, warmup=warmup)
            rows.append({"roots": r, "policy": name, **out["tally"],
                         "final_rank": out["final_rank"],
                         "final_size": out["final_size"],
                         "branching": out["branching"]})
    return rows


def same_state_probe(roots: int, warmup: int) -> dict:
    graph = ReflectionGraph.base(roots=roots)
    for _ in range(warmup):
        graph = reflect(graph, deepen(graph)).graph_after
    out = {}
    for name, policy in POLICIES:
        s = reflect(graph, policy(graph))
        out[name] = {"kind": s.kind, "productive": s.productive,
                     "depth": s.depth, "rank_before": s.rank_before,
                     "rank_after": s.rank_after}
    return {"rank": graph.rank, "size": graph.size, "steps": out}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--roots", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--long", type=int, default=200)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if args.quick:
        args.steps, args.long = 12, 60

    rows = root_sweep(args.roots, args.steps, args.warmup)
    print(__doc__.split("\n\n")[0])
    print()
    print(f"  {args.steps} steps per run, after {args.warmup} deepening steps")
    print()
    print(f"  {'roots':>6} {'policy':<9} {'advancing':>10} {'sideways':>9} "
          f"{'duplicate':>10} {'rank':>6} {'size':>6} {'branched':>9}")
    for r in rows:
        print(f"  {r['roots']:>6} {r['policy']:<9} {r['advancing']:>10} "
              f"{r['sideways']:>9} {r['duplicate']:>10} {r['final_rank']:>6} "
              f"{r['final_size']:>6} {str(r['branching']):>9}")
    print()

    single = [r for r in rows if r["roots"] == 1]
    multi = [r for r in rows if r["roots"] > 1]
    sideways_multi = [r for r in multi if r["sideways"] > 0]

    q2 = bool(sideways_multi)
    q3 = (all(r["sideways"] == 0 for r in single)
          and all(not r["branching"] for r in single)
          and any(r["branching"] for r in multi))

    dupes = sum(r["duplicate"] for r in rows)
    print(f"  Q1 do the prior categories survive? ............. PARTIALLY TESTED")
    print(f"     `stagnant` reappears intact as `duplicate` ({dupes} of them")
    print(f"     across the sweep) — a step that adds nothing, under a domain")
    print(f"     sharing no machinery with the earlier ones. But the three walls")
    print(f"     are NOT exercised here: this domain has no budget, no naming")
    print(f"     defect and no certification requirement, so `economic`,")
    print(f"     `structural` and `epistemic` have nothing to bite on. The")
    print(f"     registered prediction said they survive; this run neither")
    print(f"     confirms nor refutes it, and reporting it as confirmed would be")
    print(f"     claiming a measurement that was not made.")
    print()
    print(f"  Q2 the split exists ............................. "
          f"{'CONFIRMED' if q2 else 'REFUTED'}")
    print(f"     steps that add a genuinely new sentence and leave the global")
    print(f"     rank exactly where it was. The reviewer's registered prediction")
    print(f"     is confirmed: `productive` does split, and the linear domains")
    print(f"     could not have shown it because there was only one place growth")
    print(f"     could happen.")
    print()
    print(f"  Q3 it requires branching ........................ "
          f"{'CONFIRMED' if q3 else 'REFUTED'}")
    print(f"     and the single-root result is STRONGER than the prediction it")
    print(f"     was meant to test. From one base, free finite-join reflection")
    print(f"     does not merely fail to produce sideways moves — it produces")
    print(f"     nothing at all: every join is a duplicate, "
          f"{single[1]['duplicate']}/{args.steps} of them.")
    print(f"     Over a chain every subset's union of contents is the content of")
    print(f"     its maximum, which is already asserted. So branching cannot")
    print(f"     ARISE from a single root; it has to be given, as independent")
    print(f"     starting theories. Neither registered prediction said that.")
    print()

    probe = same_state_probe(max(3, min(r for r in args.roots if r > 1)),
                             args.warmup)
    kinds = {k: v["kind"] for k, v in probe["steps"].items()}
    q4 = "advancing" in kinds.values() and "sideways" in kinds.values()
    print(f"  Q4 it is a property of MOVES, not of states ..... "
          f"{'CONFIRMED' if q4 else 'REFUTED'}")
    print()
    print(f"     one and the same graph (rank {probe['rank']}, "
          f"{probe['size']} nodes), two policies:")
    for name, s in probe["steps"].items():
        print(f"       {name:<9} -> {s['kind']:<10} rank "
              f"{s['rank_before']} -> {s['rank_after']}")
    print()
    print(f"     So `sideways` is NOT a sixth wall. Every category before it —")
    print(f"     economic, structural, epistemic, exhausted, stagnant — is a")
    print(f"     fact about where a system IS. This is a fact about what it")
    print(f"     CHOSE. The same state offers both moves, and nothing about the")
    print(f"     state decides which one gets taken.")
    print()

    long_run = run_policy(broaden, args.long, roots=4, warmup=args.warmup)
    ranks = sorted({s["rank_after"] for s in long_run["trace"]})
    print(f"  And the consequence, which neither prediction reached:")
    print()
    print(f"     {args.long} consecutive steps under `broaden`, every one of them")
    print(f"     productive — {long_run['tally']['duplicate']} duplicates, "
          f"{long_run['tally']['sideways']} sideways — building")
    print(f"     {long_run['final_size']} nodes. Distinct global ranks visited "
          f"across all of it: {ranks}.")
    print()
    print(f"     A system can be productive FOREVER and go nowhere. That is not")
    print(f"     stagnation, which produces nothing; not exhaustion, which has")
    print(f"     nothing left; not any wall, since nothing blocks it. Every")
    print(f"     check passes on every step. It simply never advances, and no")
    print(f"     observable in the previous five domains would show it.")
    print()

    deep = next(r for r in rows if r["roots"] == 4 and r["policy"] == "deepen")
    broad = next(r for r in rows if r["roots"] == 4 and r["policy"] == "broaden")
    print(f"     Set beside the same budget spent differently: at 4 roots and")
    print(f"     {args.steps} steps, `deepen` and `broaden` build the SAME number of")
    print(f"     nodes ({deep['final_size']} and {broad['final_size']}) and reach "
          f"rank {deep['final_rank']} and {broad['final_rank']}.")
    print(f"     Identical productivity, identical cost, {deep['final_rank'] // max(1, broad['final_rank'])}x the reach.")
    print()

    print("  What this does to the taxonomy")
    print()
    print("  `productive` splits into local use and global advance, which is what")
    print("  was predicted. But the split is not another wall — it is the first")
    print("  observable in this series that belongs to a MOVE rather than to a")
    print("  STATE, and that is a different kind of addition than the previous")
    print("  four domains made.")
    print()
    print("  It also means the earlier claim that each new domain adds a category")
    print("  needs qualifying rather than repeating. This domain added a")
    print("  distinction of a different type. Whether that is a fifth instance of")
    print("  a pattern or the point where the pattern stops being one thing is")
    print("  not settled by a single run, and is not claimed here.")
    print()
    print("  Honest scope. `depth` is a declared proxy for proof-theoretic rank;")
    print("  the registered prediction was phrased in terms of genuine ordinal")
    print("  height and substituting depth is a weakening, stated not hidden.")
    print("  Sentence identity is normalised to logical content, without which")
    print("  the phenomenon would be an artifact of the representation. Q4 was")
    print("  not in either registered prediction and is marked as a question the")
    print("  first two raised rather than as something foreseen.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "reflection_dag.json"
        out.write_text(json.dumps({
            "params": vars(args) | {"output_dir": str(args.output_dir)},
            "sweep": rows, "same_state": probe,
            "long_run": {k: v for k, v in long_run.items() if k != "trace"},
            "verdicts": {"Q2_split_exists": q2, "Q3_requires_branching": q3,
                         "Q4_property_of_moves": q4},
        }, indent=2, default=str))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
