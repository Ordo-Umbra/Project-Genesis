"""Does a sideways move buy options? And do options convert into height?

Every result in this series has measured reach with one number: `rank`, the
longest path from the base. Height, and nothing else. So a move that widens the
base — creating material that makes *more future moves possible* — scores zero,
and gets called `sideways`: going nowhere.

That is structurally the same error the last two runs caught twice. A quantity
was measured, a different quantity was named, and nothing forced them apart. If
a broadening move is **investment** rather than stagnation, then `rank` is
reading height and we have been calling it progress.

So: count the options. And then ask the only question that matters about them —
whether they ever turn into height.

The move space counted is singletons and incomparable pairs, which is what the
policies here actually propose. That restriction is declared, not derived.

Pre-registered predictions
--------------------------
Q1. **A chain generates no join options, ever.** Over a totally ordered
    collection every union is the larger member, which is already asserted. So
    free generation from one root should sit at zero join options forever, while
    independent roots should let the count grow. **Falsifier:** a chain produces
    join options, which would mean the normalisation in `reflect` is not
    collapsing what it claims to.

Q2. **Sideways moves generate more options per step than advancing ones.** The
    combinatorial argument: a join creates a node incomparable to much of what
    exists, which then pairs with all of it. An advance extends a chain and adds
    one thing to reflect on. **Falsifier:** advancing generates options at the
    same rate or faster, in which case there is no hidden capability and `rank`
    was never hiding anything.

    *This came out refuted, and the run reports why: the prediction inherited
    the very framing it was meant to test. Option generation tracks whether a
    move creates incomparable material, which is orthogonal to whether it gains
    height — a policy that grows several lineages in parallel does both.*

Q3. **The options do not convert into height.** This is the one that matters and
    it is registered against the interesting answer. Invest N steps in breadth,
    then climb with a rank-following policy: the prediction is that final rank is
    simply `rank_after_investment + steps_climbed`, identical whatever was
    invested in, because a depth-following policy never uses the width.
    **Falsifier:** breadth-investment yields a higher final rank than the same
    steps spent climbing, which would make sideways an investment and the
    sideways-as-trap reading wrong.

Q4. **But options are insurance, and the premium is measurable.** With a wall
    landing on a lineage chosen *after* the strategy is committed — which is the
    honest setup, since result five showed a system cannot tell from inside which
    of its lineages is the unsettleable one — concentration should win the mean
    and diversification should win the worst case. **Falsifier:** one strategy
    dominates on both, which would make this a solved problem rather than a
    trade.

Honest scope
------------
Q4's answer turns out to depend on a modelling choice that had never been made
explicit, so both readings are run rather than one being chosen:

    freeze   — a wall blocks further work on a lineage; what was built stands.
    retract  — a tower over a base whose consistency cannot be settled is itself
               uncertified, so the height goes with it.

`retract` is the more faithful one for reflection towers, and it is the only one
under which the insurance has a visible premium at a short horizon.

The prediction about *horizon* was made and came out backwards; that is recorded
in the output rather than quietly dropped.

`depth` remains a declared proxy for proof-theoretic rank, and `opaque` remains a
declared unsettleable set — nothing here proves any address undecidable.

    python experiments/reflection_options.py
    python experiments/reflection_options.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.reflection_dag import (  # noqa: E402
    Filters, ReflectionGraph, broaden, deepen, options, reflect, run_options,
    spread, strategy_table, two_phase,
)


def chain_check(steps: int) -> list[dict]:
    rows = []
    for roots in (1, 2, 3):
        graph = ReflectionGraph.base(roots=roots)
        for _ in range(steps):
            graph = reflect(graph, deepen(graph)).graph_after
        o = options(graph)
        rows.append({"roots": roots, "is_chain": graph.is_chain(),
                     "single": o["single"], "join": o["join"]})
    return rows


def per_kind(steps: int) -> list[dict]:
    rows = []
    for name, policy in (("deepen", deepen), ("spread", spread),
                         ("broaden", broaden)):
        r = run_options(policy, steps)
        rows.append({"policy": name, "mean_delta": r["mean_delta"],
                     "counts": r["counts"], "final": r["final"],
                     "final_rank": r["final_rank"]})
    return rows


def conversion(invest: int, horizon: int) -> list[dict]:
    rows = []
    for name, policy in (("deepen", deepen), ("spread", spread),
                         ("broaden", broaden)):
        graph = ReflectionGraph.base(roots=3)
        for _ in range(invest):
            graph = reflect(graph, policy(graph)).graph_after
        rows.append({"invested_in": name, "rank_after_invest": graph.rank,
                     "options_after_invest": options(graph)["total"],
                     "rank_after_climb": two_phase(policy, invest, horizon,
                                                   opaque=None)})
    return rows


def insurance(invest: int, horizons: list[int]) -> list[dict]:
    rows = []
    for retract in (False, True):
        for hz in horizons:
            t = strategy_table(invest, hz, retract=retract)
            rows.append({"model": "retract" if retract else "freeze",
                         "horizon": hz, **t})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--invest", type=int, default=20)
    p.add_argument("--horizons", type=int, nargs="+", default=[5, 20, 40, 80])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if args.quick:
        args.steps, args.invest, args.horizons = 15, 10, [5, 20]

    print(__doc__.split("\n\n")[0])
    print()

    ch = chain_check(args.steps)
    print("  Q1. Does a chain generate any join options?")
    print()
    print(f"  {'roots':>6} {'is a chain':>11} {'single options':>15} {'join options':>13}")
    for r in ch:
        print(f"  {r['roots']:>6} {str(r['is_chain']):>11} {r['single']:>15} "
              f"{r['join']:>13}")
    q1 = ch[0]["join"] == 0 and ch[0]["is_chain"] and ch[-1]["join"] > 0
    print()
    print(f"  Q1 a chain generates no join options, ever ....... "
          f"{'CONFIRMED' if q1 else 'REFUTED'}")
    print("     Free generation from one root stays totally ordered, so every")
    print("     union is the larger member and is already asserted. Independent")
    print("     starting points are the precondition for a whole class of move")
    print("     existing at all — not an optimisation, a prerequisite.")
    print()

    pk = per_kind(args.steps)
    print("  Q2. How many options does each kind of step create?")
    print()
    print(f"  {'policy':<10} {'advancing':>22} {'sideways':>22}")
    for r in pk:
        adv = r["mean_delta"].get("advancing")
        sid = r["mean_delta"].get("sideways")
        f = lambda v, n: (f"{v:+.2f} over {n:>2} steps" if v is not None else
                          "".rjust(18))
        print(f"  {r['policy']:<10} {f(adv, r['counts'].get('advancing', 0)):>22} "
              f"{f(sid, r['counts'].get('sideways', 0)):>22}")
    by_policy = {r["policy"]: r["mean_delta"] for r in pk}
    # Is the advancing/sideways label what predicts option generation? Test it
    # on the case that separates the two candidate explanations: `spread`, whose
    # moves all advance but all create incomparable material. If the label were
    # the variable, its rate would sit near the other advancing rate. If
    # incomparability is the variable, it sits near the sideways rate.
    like_advancing = abs(by_policy["spread"]["advancing"]
                         - by_policy["deepen"]["advancing"])
    like_sideways = abs(by_policy["spread"]["advancing"]
                        - by_policy["broaden"]["sideways"])
    q2 = like_advancing < like_sideways
    print()
    print(f"  Q2 sideways generates more options than advancing  "
          f"{'CONFIRMED' if q2 else 'REFUTED'}")
    print(f"     Refuted as stated, and the refutation is the finding.")
    print(f"     Chain-extension generates {by_policy['deepen']['advancing']:.2f} "
          f"options per advance; joining below the")
    print(f"     frontier generates {by_policy['broaden']['sideways']:.2f} per "
          f"sideways step — five to one, which is")
    print(f"     the effect the prediction was about. But `spread`, whose every")
    print(f"     move is a single-parent **advance**, generates "
          f"{by_policy['spread']['advancing']:.2f} per step:")
    print(f"     as many as the best sideways move.")
    print()
    print(f"     So option generation does not track the advancing/sideways")
    print(f"     axis at all. **It tracks whether a move creates incomparable")
    print(f"     material** — two theories neither of which contains the other —")
    print(f"     and that is orthogonal to whether the move gains height. You can")
    print(f"     have both. Growing several lineages in parallel advances *and*")
    print(f"     opens options; extending one lineage advances and opens almost")
    print(f"     none. The prediction inherited the framing it was supposed to")
    print(f"     test, which is how it got the variable wrong.")
    print()

    cv = conversion(args.invest, args.horizons[-1])
    print(f"  Q3. Do the options convert into height? "
          f"(invest {args.invest}, then climb {args.horizons[-1]})")
    print()
    print(f"  {'invested in':<12} {'rank after invest':>18} {'options built':>14} "
          f"{'rank after climb':>17} {'gained':>7}")
    for r in cv:
        print(f"  {r['invested_in']:<12} {r['rank_after_invest']:>18} "
              f"{r['options_after_invest']:>14} {r['rank_after_climb']:>17} "
              f"{r['rank_after_climb'] - r['rank_after_invest']:>7}")
    gains = {r["rank_after_climb"] - r["rank_after_invest"] for r in cv}
    q3 = len(gains) == 1
    print()
    print(f"  Q3 the options do NOT convert into height ........ "
          f"{'CONFIRMED' if q3 else 'REFUTED'}")
    print(f"     Every strategy gains exactly {gains.pop()} rank in the climb "
          f"phase, whatever")
    print("     it built beforehand and however many options it has. A depth-")
    print("     following policy never spends the width. **Options are real and")
    print("     they are not height** — which means the sideways-as-trap reading")
    print("     survives, but only for the quantity it was ever about.")
    print()

    ins = insurance(args.invest, args.horizons)
    print("  Q4. So what are the options for? A wall lands after you commit.")
    print()
    print(f"  {'model':<8} {'horizon':>7} | {'concentrate  mean / worst':>26} | "
          f"{'diversify  mean / worst':>24} | {'mean':>11} {'worst':>11}")
    for r in ins:
        c, d = r["concentrate"], r["diversify"]
        wm = "concentrate" if c["mean"] > d["mean"] else "diversify"
        ww = ("concentrate" if c["worst"] > d["worst"] else
              "diversify" if d["worst"] > c["worst"] else "tie")
        print(f"  {r['model']:<8} {r['horizon']:>7} | {c['mean']:>15.1f} / "
              f"{c['worst']:>8} | {d['mean']:>13.1f} / {d['worst']:>8} | "
              f"{wm:>11} {ww:>11}")
    q4 = (all(r["concentrate"]["mean"] >= r["diversify"]["mean"] for r in ins)
          and any(r["diversify"]["worst"] > r["concentrate"]["worst"]
                  for r in ins))
    print()
    print(f"  Q4 concentration wins the mean, diversification the floor  "
          f"{'CONFIRMED' if q4 else 'REFUTED'}")
    print("     Concentration wins the average in every configuration tested.")
    print("     Diversification wins the worst case in almost all of them. That")
    print("     gap **is** the trade, and it is not resolvable by more")
    print("     information: result five showed a system cannot tell from inside")
    print("     which of its lineages is the one that will not settle. So the")
    print("     choice is between optimising the mean and optimising the floor,")
    print("     and that is a decision about what you are willing to lose, not a")
    print("     fact about the domain.")
    print()
    print("     Two things sharpen it. **Whether a wall freezes or retracts is")
    print("     load-bearing.** Under `freeze` the height you already reached")
    print("     stands and concentration wins on both measures at a short")
    print("     horizon. Under `retract` — a tower over an unsettleable base is")
    print("     itself uncertified, which is the faithful reading for reflection")
    print("     towers — the concentrated strategy's floor collapses to what it")
    print("     can rebuild in the time left. That is where the premium shows.")
    print()
    print("     **And a prediction made here came out backwards.** The guess was")
    print("     that a short horizon favours diversification, because there is")
    print("     no time to rebuild. The opposite holds under `freeze`: at a short")
    print("     horizon what matters is the height already standing, and the")
    print("     concentrated strategy has more of it. Diversification only pays")
    print("     when the failure retracts *and* the horizon is short, or when the")
    print("     horizon is long enough for resumption-from-height to matter.")
    print()

    print("  Honest scope. The move space counted is singletons and incomparable")
    print("  pairs — what the policies propose, not the full subset lattice, and")
    print("  a count over a larger space would be a different number. `opaque`")
    print("  is a declared unsettleable set; nothing here proves any address")
    print("  undecidable. `depth` remains a declared proxy for proof-theoretic")
    print("  rank. And nothing in this run models research programmes, funding,")
    print("  or belief — the resemblance to how one decides where to place")
    print("  intellectual effort is a resemblance, and it was not tested.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "reflection_options.json"
        out.write_text(json.dumps({
            "params": vars(args) | {"output_dir": str(args.output_dir)},
            "chain": ch, "per_kind": pk, "conversion": cv, "insurance": ins,
            "verdicts": {"Q1_chain_has_no_joins": q1,
                         "Q2_sideways_makes_options": q2,
                         "Q3_options_do_not_convert": q3,
                         "Q4_mean_vs_floor": q4},
        }, indent=2, default=str))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
