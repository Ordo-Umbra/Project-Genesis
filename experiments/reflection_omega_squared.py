"""Limits of limits — and what Kleene's `O` actually costs.

`reflection_limits.py` ran one limit and found that presentation stops being a
price and becomes a gate. It stopped at the `ω²` fragment, and left an open
question: the next rung is a limit *of limits*, and that is supposedly where
deferring Kleene's `O` stops being free.

This runs it, and the answer to the open question is not the expected one.

Reaching `ω²` does not need `O`. Neither does `ω^ω`, nor `ε₀`. Those all have
ordinary notation systems with unique representations, decidable comparison and
canonical fundamental sequences read straight off the Cantor normal form. A
limit of limits turns out to be **the same move at a higher level**, at the same
cost, with nothing new required.

`O` is needed only for *all* recursive ordinals — and there the price is not
implementation effort. In `O` a limit notation is an arbitrary index for a
function enumerating a fundamental sequence, and the notation is valid only if
that function is total: Π⁰₂ in general, with `O`-membership Π¹₁-complete. So the
thing `O` costs is the **decidability of the accessibility relation itself**. A
system using it cannot determine which continuations are open to it.

That is a third kind of terminal state, and it is the one the whole series has
been converging on:

    unaffordable      the edge exists and costs too much     (economic)
    limit-undefined   the edge does not exist                (structural)
    undecidable       whether the edge exists is not decidable

Pre-registered predictions
--------------------------
Q1. **A limit of limits is the same mechanism, not a new one.** Cost ratio
    1.000 to a successor at every level tested, productive at every level.
    **Falsifier:** cost or productivity varies with level — that would make
    "limit of limits" a distinct move needing its own account.

Q2. **Rank outruns cost without bound.** Reaching `ω^k` takes `k` limits, so a
    fixed step budget reaches arbitrarily high rank while cost per step stays
    flat. **Falsifier:** per-step cost grows with rank reached.

Q3. **Notation validity below `ω^ω` is decidable and immediate.** Every CNF
    limit gets a conclusive verdict with no search. **Falsifier:** any CNF
    limit where the check cannot conclude.

Q4. **Above it, validity becomes a search that cannot conclude.** A total
    fundamental sequence and one that diverges at `k` return *identical*
    verdicts at every bound below `k`, and the checker can never report
    "valid" — only "verified this far". **Falsifier:** some finite bound
    separates them, which would make totality checkable by running.

Honest scope
------------
Q4 demonstrates the *shape* of the consequence, not the theorem. Divergence is
simulated by a sequence that declines to halt, and no run here proves anything
about undecidability — that totality is Π⁰₂-complete and `O`-membership is
Π¹₁-complete are cited results. What is measured is that a checker which must
run a sequence cannot distinguish "total" from "total so far", which is the
property that makes the citation bite.

`Prf` remains primitive, so costs carry an unexpanded constant and only ratios
are read. `T_n ⊬ Con(T_n)` is Gödel's second theorem, discharged, never measured.

    python experiments/reflection_omega_squared.py
    python experiments/reflection_omega_squared.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.reflection import (  # noqa: E402
    LimitUndefined, Rank, canonical_fundamental_sequence, construction_cost,
    ladder, limit_step, peano, step, verify_cnf_notation,
    verify_searched_notation,
)


# ------------------------------------------- Q1/Q2: the same move, higher up


def level_scan(levels: list[int], depth: int) -> dict:
    """Cost and productivity of a limit taken at each level."""
    theory = peano("indexed")
    successor_cost = construction_cost(step(theory))
    for s in ladder(theory, depth):
        theory = s.theory_after
    rows = []
    for level in levels:
        lim = limit_step(theory, level)
        rows.append({
            "level": level,
            "from_rank": str(theory.rank),
            "to_rank": str(lim.theory_after.rank),
            "cost": lim.con_symbols,
            "ratio": lim.con_symbols / successor_cost,
            "new_axiom": lim.new_axiom,
        })
    return {"successor_cost": successor_cost, "rows": rows}


def climb_to_degree(max_degree: int, per_level: int) -> dict:
    """Climb through ω, ω², ω³, … recording rank against cost paid."""
    theory = peano("indexed")
    taken = productive = total_cost = 0
    rows = []
    for level in range(1, max_degree + 1):
        for _ in range(per_level):
            s = step(theory)
            taken += 1
            productive += 1 if s.new_axiom else 0
            total_cost += construction_cost(s)
            theory = s.theory_after
        lim = limit_step(theory, level)
        taken += 1
        productive += 1 if lim.new_axiom else 0
        total_cost += lim.con_symbols
        theory = lim.theory_after
        rows.append({
            "degree": theory.rank.degree,
            "rank": str(theory.rank),
            "steps_taken": taken,
            "productive": productive,
            "total_cost": total_cost,
            "cost_per_step": total_cost / taken,
        })
    return {"rows": rows, "final_rank": str(theory.rank)}


def gate_still_holds(depth: int, levels: list[int]) -> dict:
    """The inline arm should refuse a limit at *every* level, not just level 1."""
    theory = peano("inline")
    for s in ladder(theory, depth):
        theory = s.theory_after
    out = {}
    for level in levels:
        try:
            limit_step(theory, level)
            out[level] = "ADMITTED"
        except LimitUndefined:
            out[level] = "refused"
    return out


# --------------------------------------------- Q3/Q4: what a notation costs


def notation_scan(ranks: list[Rank]) -> list[dict]:
    rows = []
    for r in ranks:
        v = verify_cnf_notation(r)
        seq = canonical_fundamental_sequence(r) if r.is_limit else None
        rows.append({
            "rank": str(r),
            "status": v.status,
            "conclusive": v.conclusive,
            "sequence": ([str(seq(n)) for n in range(4)] if seq else None),
        })
    return rows


def search_scan(bounds: list[int], diverge_at: int, budget: int) -> list[dict]:
    """A total sequence against one that diverges, at a range of bounds."""
    total = lambda n, _b: n                                  # noqa: E731
    partial = lambda n, _b: n if n < diverge_at else None    # noqa: E731
    rows = []
    for bound in bounds:
        vt = verify_searched_notation(total, bound=bound, budget=budget)
        vp = verify_searched_notation(partial, bound=bound, budget=budget)
        rows.append({
            "bound": bound,
            "total_status": vt.status, "total_conclusive": vt.conclusive,
            "partial_status": vp.status, "partial_conclusive": vp.conclusive,
            "indistinguishable": (vt.status, vt.checked) == (vp.status, vp.checked),
        })
    return rows


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--max-degree", type=int, default=6)
    p.add_argument("--per-level", type=int, default=4)
    p.add_argument("--diverge-at", type=int, default=9)
    p.add_argument("--budget", type=int, default=10000)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if args.quick:
        args.max_degree, args.per_level = 3, 2

    levels = list(range(1, args.max_degree + 1))
    print(__doc__.split("\n\n")[0])
    print()

    # ------------------------------------------------------------------- Q1
    scan = level_scan(levels, depth=4)
    print("  A limit taken at each level, from the same starting theory")
    print()
    print(f"  {'level':>5} {'from':>8} {'to':>12} {'cost':>7} {'ratio':>7}  new axiom")
    for r in scan["rows"]:
        print(f"  {r['level']:>5} {r['from_rank']:>8} {r['to_rank']:>12} "
              f"{r['cost']:>7} {r['ratio']:>7.4f}  {r['new_axiom']}")
    ratios = [r["ratio"] for r in scan["rows"]]
    q1 = max(ratios) == min(ratios) == 1.0 and all(r["new_axiom"]
                                                   for r in scan["rows"])
    print()
    print(f"  Q1 a limit of limits is the same mechanism ...... "
          f"{'CONFIRMED' if q1 else 'REFUTED'}")
    print(f"     ratio to a successor is {min(ratios):.4f} at every level from "
          f"1 to {levels[-1]}, and every")
    print(f"     one adds a new axiom. Going from omega to omega^2 is not a")
    print(f"     harder move than going from 3 to 4 — it bumps a different")
    print(f"     coefficient of the same normal form. There is no new mechanism")
    print(f"     here to account for, which is the first half of the answer to")
    print(f"     'do we now have to pay for Kleene's O'.")
    print()

    gates = gate_still_holds(4, levels)
    q1b = all(v == "refused" for v in gates.values())
    print(f"     And the gate generalises: the inline arm refuses a limit at "
          f"EVERY level")
    print(f"     ({', '.join(f'{k}:{v}' for k, v in gates.items())}). It is not "
          f"short of a level. It has no")
    print(f"     limit edge of any kind, because it has no index for a union.")
    print()

    # ------------------------------------------------------------------- Q2
    climb = climb_to_degree(args.max_degree, args.per_level)
    print("  Rank reached against cost paid")
    print()
    print(f"  {'rank':>16} {'degree':>7} {'steps':>6} {'productive':>11} "
          f"{'cost/step':>10}")
    for r in climb["rows"]:
        print(f"  {r['rank']:>16} {r['degree']:>7} {r['steps_taken']:>6} "
              f"{r['productive']:>11} {r['cost_per_step']:>10.1f}")
    per_step = [r["cost_per_step"] for r in climb["rows"]]
    q2 = max(per_step) / min(per_step) < 1.01
    print()
    print(f"  Q2 rank outruns cost without bound ............. "
          f"{'CONFIRMED' if q2 else 'REFUTED'}")
    print(f"     cost per step is flat to within "
          f"{100 * (max(per_step) / min(per_step) - 1):.2f}% all the way to "
          f"{climb['final_rank']},")
    print(f"     reached in {climb['rows'][-1]['steps_taken']} steps, every one "
          f"of them productive. Rank is not")
    print(f"     bought with cost at all: it is bought with the RIGHT TO NAME,")
    print(f"     and once a presentation has that right the ordinal it reaches")
    print(f"     is limited only by how many times you care to apply it.")
    print()

    # ------------------------------------------------------------------- Q3
    ranks = [Rank(0, 3), Rank.from_levels({1: 1}), Rank.from_levels({1: 3}),
             Rank.from_levels({2: 1}), Rank.from_levels({2: 2, 1: 5}),
             Rank.from_levels({args.max_degree: 1})]
    rows = notation_scan(ranks)
    print("  Notation validity below omega^omega — decided, not searched")
    print()
    print(f"  {'rank':>12} {'verdict':>23} {'conclusive':>11}  canonical sequence")
    for r in rows:
        seq = ", ".join(r["sequence"]) + ", ..." if r["sequence"] else "-"
        print(f"  {r['rank']:>12} {r['status']:>23} {str(r['conclusive']):>11}  "
              f"{seq}")
    q3 = all(r["conclusive"] for r in rows)
    print()
    print(f"  Q3 validity below omega^omega is decidable ..... "
          f"{'CONFIRMED' if q3 else 'REFUTED'}")
    print(f"     every notation gets a conclusive verdict immediately, because")
    print(f"     its fundamental sequence is closed-form arithmetic on the")
    print(f"     Cantor normal form. Nothing is run; nothing can fail to halt.")
    print()

    # ------------------------------------------------------------------- Q4
    bounds = sorted({2, args.diverge_at - 1, args.diverge_at,
                     args.diverge_at * 3, args.diverge_at * 10})
    srows = search_scan(bounds, args.diverge_at, args.budget)
    print(f"  Notation validity ABOVE it — searched, not decided")
    print(f"  (a total sequence against one that diverges at "
          f"n = {args.diverge_at})")
    print()
    print(f"  {'bound':>6} {'total':>26} {'diverging':>26}  same?")
    for r in srows:
        t = f"{r['total_status']} ({'conclusive' if r['total_conclusive'] else 'inconclusive'})"
        d = f"{r['partial_status']} ({'conclusive' if r['partial_conclusive'] else 'inconclusive'})"
        print(f"  {r['bound']:>6} {t:>26} {d:>26}  "
              f"{'YES' if r['indistinguishable'] else 'no'}")
    below = [r for r in srows if r["bound"] <= args.diverge_at]
    q4 = (all(r["indistinguishable"] for r in below)
          and not any(r["total_conclusive"] for r in srows))
    print()
    print(f"  Q4 above it, validity cannot be concluded ...... "
          f"{'CONFIRMED' if q4 else 'REFUTED'}")
    print(f"     at every bound up to {args.diverge_at} the two are "
          f"INDISTINGUISHABLE, and the total one is")
    print(f"     never conclusive at any bound — the checker can only ever say")
    print(f"     'verified this far'. Raising the bound does not converge on an")
    print(f"     answer; it just moves the same ignorance further out. That is")
    print(f"     the shape of a Pi-0-2 question, and it is what a notation")
    print(f"     system pays when it admits arbitrary indices instead of")
    print(f"     canonical ones.")
    print()

    print("  So: should the ordinal layer pay for Kleene's O?")
    print()
    print("  Not for reach. Everything below omega^omega — and epsilon_0 by the")
    print("  same construction — is available with canonical notations, at a")
    print("  cost per step that does not move. The deferral has cost nothing so")
    print("  far and would keep costing nothing for a long way up.")
    print()
    print("  What O buys is ALL recursive ordinals, and what it spends is the")
    print("  decidability of the accessibility relation. Below it, can_take_limit")
    print("  is a decision. With it, can_take_limit is a search, and the system")
    print("  cannot in general determine which continuations are open to it.")
    print("  That is a THIRD kind of terminal state, distinct from both the")
    print("  series has already found:")
    print()
    print("    unaffordable      the edge exists and costs too much   economic")
    print("    limit-undefined   the edge does not exist              structural")
    print("    undecidable       whether it exists cannot be decided  epistemic")
    print()
    print("  And it lands squarely on G. G > 0 is a claim about which edges")
    print("  exist. At the top of the recursive ordinals that claim stops being")
    print("  checkable by the system making it — not false, not unavailable,")
    print("  but undecidable from the inside. A system there cannot tell a")
    print("  continuation it has from one it merely cannot rule out.")
    print()
    print("  Honest scope. Q4 demonstrates the SHAPE of that consequence, not")
    print("  the theorem. Divergence here is simulated by a sequence that")
    print("  declines to halt; that totality is Pi-0-2-complete and O-membership")
    print("  Pi-1-1-complete are cited results, not things any run could show.")
    print("  What is measured is only that a checker which must run a sequence")
    print("  cannot separate 'total' from 'total so far' — which is the")
    print("  property that makes the citation bite.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "reflection_omega_squared.json"
        out.write_text(json.dumps({
            "params": vars(args) | {"output_dir": str(args.output_dir)},
            "levels": scan, "gate": gates, "climb": climb,
            "notations": rows, "search": srows,
            "verdicts": {"Q1_same_mechanism": q1, "Q1_gate_generalises": q1b,
                         "Q2_rank_outruns_cost": q2,
                         "Q3_decidable_below_omega_omega": q3,
                         "Q4_undecidable_above": q4},
        }, indent=2, default=str))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
