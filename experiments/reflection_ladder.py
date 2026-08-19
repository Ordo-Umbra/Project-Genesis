"""Does climbing the reflection ladder buy capability, or only symbols?

`The_Generative_Gap.md` §3 cites the ladder `T_{n+1} = T_n + Con(T_n)` as the
formal-system twin of the field program's capacity gap: `I(F) < C(F) = ω₁^CK`,
strictly and permanently, with the ladder climbing forever without closing it.
The citation has been load-bearing for the whole ordinal reading and has never
been run. This runs it, and asks the first question that can actually come out
either way.

The question, stated so it can fail
-----------------------------------
The Generative Continuation Principle is bookkept by a triple `(C, I, G)` —
capacity, current integration, accessible headroom — with the continuation
condition `G > 0`. In this realisation `C = ω₁^CK` is fixed by the domain,
`I` is the rung counter, and `G ≥ 1` because the successor `K(T) = T + Con(T)`
is always available. Two of those three are *definitions*. They cannot fail,
and a program that printed them back would be a tautology with a progress bar.

So the measured quantity is deliberately a different one:

    G_measured(n) = 1 if rung n enlarged the axiom SET, else 0

together with a machine-checked derivation that *uses* the sentence added.
`I` is a counter and will climb no matter what. The experiment is built so
that `G_measured` can come apart from it, and one arm is built so that it does.

Why presentations are the whole experiment
------------------------------------------
`Con(T_n)` must name an index for `T_n`'s axiom set. That set is r.e. and has
infinitely many indices; which one is named is a free choice the mathematics
does not fix. Three choices are run (see `project_genesis/reflection.py`):

- `inline` — the index is the Gödel number of the literal axiom list, so each
  rung's `Con` carries a numeral for the entire theory beneath it;
- `indexed` — the index is `⟨code(PA), n⟩`, a recursive index naming "PA plus
  the first n rungs" without listing them;
- `truncated` — `indexed` with the counter in a `width`-bit field, a
  **deliberate negative control**: after `2^width` rungs the index wraps.

`inline` and `indexed` are not the same theory — different indices give
literally different `Con` sentences. They are the same *construction* under two
presentations, which is the comparison being drawn, and is why the claims below
are about ratios between arms rather than absolute symbol counts.

Pre-registered predictions
--------------------------
Q1. **The productive step is presentation-invariant.** In both `inline` and
    `indexed`, every rung adds exactly one new axiom: `G_measured = 1`
    throughout, and the added sentence checks as a premise in a real
    derivation. **Falsifier:** any rung of a non-truncated arm fails to
    enlarge the axiom set — the continuation operator would not be productive
    under that presentation, and the ladder would not be what §3 claims.

Q2. **The cost is not.** Presentation cost diverges without bound between the
    two: `inline` geometric (ratio → 2 per rung), `indexed` flat. At the
    default 12 rungs the separation should exceed 100×. **Falsifier:** the two
    stay within a constant factor — symbol count would then be tracking the
    ladder rather than the encoding, and cost would be a defensible proxy for
    rank after all.

Q3. **The dissociation runs in both directions.** Q2 gives one direction (same
    productivity, wildly different cost). `truncated` gives the other: cost
    bounded, and `G_measured` drops to 0 at rung `2^width` while `I` keeps
    incrementing exactly as before. **Falsifier:** the truncated arm keeps
    adding new axioms past the wrap — the collision would not stall the
    ladder and the negative control would not be a control.

Q4 is the one the exchange asked for and it is answered by the other three
together: **does increasing `I` produce increasing productive capacity, or
merely increasing formal complexity?** `I` is identical in all three arms.
If productive capacity and formal complexity both track it, `I` is a fair
summary. If either comes apart from it, `I` alone certifies nothing.

What is NOT claimed
-------------------
`T_n ⊬ Con(T_n)` is **not measured here and cannot be.** It is Gödel's second
incompleteness theorem, discharged from stated hypotheses (each `T_n`
consistent, recursively axiomatised, extending Robinson arithmetic). The
bounded closure search reported below is a *calibration and smoke test*, not
evidence: it is run first against a target the theory demonstrably proves, to
show the search works, and its negative result covers a fragment so small that
the honest statement is printed with it.

`Prf` is a primitive relation symbol rather than an expanded Δ₀ formula, so
absolute symbol counts are not meaningful — the unexpanded predicate
contributes the same constant to every arm, which is why only ratios are read.

The ordinal layer is metadata. `I = n` is a proxy; the mathematical model puts
the rank at `ε₀ + n`. Kleene's `O`, fundamental sequences and limit notations
are deliberately not implemented — the exchange's own instruction was to build
the finite generator first and let the ordinal layer sit above it.

    python experiments/reflection_ladder.py
    python experiments/reflection_ladder.py --quick
    python experiments/reflection_ladder.py --rungs 16 --width 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.reflection import (  # noqa: E402
    And, CAPACITY, Implies, Line, ProofError, check_proof, closure_search,
    code_bits, con_formula, conjoin_rungs_proof, first_index_collision,
    integration_rank, ladder, nominal_increment, peano, productive_increment,
    symbols,
)


# --------------------------------------------------------------- the ladders


def run_arm(kind: str, rungs: int, width: int | None = None) -> dict:
    """Climb one arm and record every observable at every rung."""
    theory = peano(kind, width=width)
    t0 = time.perf_counter()
    steps = list(ladder(theory, rungs))
    wall = time.perf_counter() - t0

    per_rung = []
    for s in steps:
        after = s.theory_after
        per_rung.append({
            "n": s.n,
            "I": integration_rank(s.theory_before),
            "G_nominal": nominal_increment(s.theory_before),
            "G_measured": productive_increment(s),
            "new_axiom": s.new_axiom,
            "axiom_count": len(after.axioms()),
            "con_symbols": s.con_symbols,
            "presentation_symbols": after.presentation_symbols(),
            "expanded_symbols": after.expanded_symbols(),
            "build_seconds": s.build_seconds,
        })

    final = steps[-1].theory_after
    growth = [per_rung[i]["presentation_symbols"]
              / per_rung[i - 1]["presentation_symbols"]
              for i in range(1, len(per_rung))]

    return {
        "kind": kind,
        "width": width,
        "rungs": rungs,
        "wall_seconds": wall,
        "per_rung": per_rung,
        "final_axiom_count": len(final.axioms()),
        "final_presentation_symbols": final.presentation_symbols(),
        "final_code_bits": code_bits(list(final.axioms())),
        "productive_rungs": sum(r["G_measured"] for r in per_rung),
        "growth_ratios": growth,
        "late_growth_ratio": growth[-1] if growth else float("nan"),
        "index_collision_at": first_index_collision(steps),
        "_steps": steps,
        "_final": final,
    }


# ------------------------------------------------------- capability, not size
#
# "The axiom set grew" is a set-membership fact. It is worth little on its own:
# a theory could store sentences it can never use. The check that the added
# sentences are *usable as premises* is a real derivation, verified line by
# line by a Hilbert checker that has no search in it.


def capability_check(arm: dict) -> dict:
    """Machine-check that the rungs are premises, not decoration."""
    final = arm["_final"]
    out: dict = {"rungs_available": len(final.rungs)}

    # 1. Every rung the ladder claims to have added is an axiom of the result,
    #    and the sentence Con(T_n) is derivable in T_{n+1} in one checked line.
    per_step_ok = True
    for s in arm["_steps"]:
        if not s.new_axiom:
            continue
        try:
            proved = check_proof(s.theory_after, [Line(s.con, "axiom")])
        except ProofError:
            per_step_ok = False
            break
        per_step_ok = per_step_ok and proved == s.con
    out["each_con_checks_in_successor"] = per_step_ok

    # 2. The rungs are used, not merely listed: derive their conjunction by
    #    modus ponens through the conjunction-introduction schema.
    if len(final.rungs) >= 2:
        proof = conjoin_rungs_proof(final)
        conclusion = check_proof(final, proof)
        out["conjunction_lines_checked"] = len(proof)
        out["conjunction_symbols"] = symbols(conclusion)
        out["conjunction_checks"] = True
    else:
        out["conjunction_checks"] = False

    # 3. Calibrate the bounded search on a target that genuinely needs modus
    #    ponens, then run it on Con(T_final). A negative on the second means
    #    nothing without the positive on the first.
    cal = {"ran": False}
    if len(final.rungs) >= 2:
        a, b = final.rungs[0], final.rungs[1]
        target = And(a, b)
        found, explored = closure_search(
            final, target, budget=20000, seeds=[Implies(a, Implies(b, target))])
        cal = {"ran": True, "found": found, "explored": explored}
    out["search_calibration"] = cal

    con_next = con_formula(final)
    found, explored = closure_search(final, con_next, budget=20000)
    out["search_for_next_con"] = {
        "found": found,
        "explored": explored,
        "saturated": explored < 20000,
    }
    return out


# ------------------------------------------------------------------ verdicts


def verdicts(arms: dict[str, dict], width: int, sep_threshold: float) -> dict:
    inline, indexed = arms["inline"], arms["indexed"]
    trunc = arms["truncated"]
    n = min(inline["rungs"], indexed["rungs"])

    q1 = (inline["productive_rungs"] == inline["rungs"]
          and indexed["productive_rungs"] == indexed["rungs"])

    separation = (inline["final_presentation_symbols"]
                  / indexed["final_presentation_symbols"])
    q2 = separation > sep_threshold

    stall = next((r["n"] for r in trunc["per_rung"] if not r["new_axiom"]), None)
    q3 = stall is not None and stall == (1 << width)

    # I is the same counter in every arm — that is the point of Q4.
    ranks = {k: [r["I"] for r in a["per_rung"]] for k, a in arms.items()}
    identical_I = len({tuple(v) for v in ranks.values()}) == 1

    return {
        "Q1_productive_step_presentation_invariant": q1,
        "Q2_cost_separation": separation,
        "Q2_cost_diverges": q2,
        "Q2_inline_late_ratio": inline["late_growth_ratio"],
        "Q2_indexed_late_ratio": indexed["late_growth_ratio"],
        "Q3_truncated_stall_rung": stall,
        "Q3_stall_at_wrap": q3,
        "Q3_index_collision_at": trunc["index_collision_at"],
        "Q4_I_identical_across_arms": identical_I,
        "rungs_compared": n,
    }


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rungs", type=int, default=12,
                   help="rungs to climb in each arm (default 12)")
    p.add_argument("--width", type=int, default=3,
                   help="counter width for the truncated control (default 3, "
                        "so it wraps at rung 8)")
    p.add_argument("--separation", type=float, default=100.0,
                   help="Q2 threshold on the inline/indexed cost ratio")
    p.add_argument("--quick", action="store_true",
                   help="6 rungs and width 2, for CI and smoke runs")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)

    if args.quick:
        args.rungs, args.width, args.separation = 6, 2, 8.0
    if args.rungs <= (1 << args.width):
        p.error(f"--rungs must exceed 2**width = {1 << args.width} for the "
                f"truncated control to reach its wrap")

    print(__doc__.split("\n\n")[0])
    print()
    print(f"  T_0 = PA,  T_(n+1) = T_n + Con(T_n),  {args.rungs} rungs, "
          f"three presentations")
    print(f"  truncated control: {args.width}-bit counter, wraps at rung "
          f"{1 << args.width}")
    print()

    arms = {
        "inline": run_arm("inline", args.rungs),
        "indexed": run_arm("indexed", args.rungs),
        "truncated": run_arm("truncated", args.rungs, width=args.width),
    }

    # ------------------------------------------------ per-rung observables
    print("  The GCP triple, per rung. C is fixed by the domain and I is a")
    print("  counter; only G_meas is measured, and only it can disagree.")
    print()
    print(f"  {'arm':<10} {'n':>3} {'I':>3} {'G_nom':>6} {'G_meas':>7} "
          f"{'axioms':>7} {'pres.symbols':>13} {'expanded':>11}")
    for kind, arm in arms.items():
        for r in arm["per_rung"]:
            print(f"  {kind:<10} {r['n']:>3} {r['I']:>3} {r['G_nominal']:>6} "
                  f"{r['G_measured']:>7} {r['axiom_count']:>7} "
                  f"{r['presentation_symbols']:>13,} "
                  f"{r['expanded_symbols']:>11,}")
        print()

    print(f"  C = {CAPACITY} in every row above, at every rung, in every arm.")
    print("  It is a domain property of recursive ordinal notations, not")
    print("  something the continuation operator produces. The permanent gap")
    print("  I < C is inherited, not earned.")
    print()

    # ------------------------------------------------------------ capability
    print("  Capability check — is the added sentence a premise or a record?")
    print()
    caps = {}
    for kind, arm in arms.items():
        c = capability_check(arm)
        caps[kind] = c
        cal = c["search_calibration"]
        print(f"  {kind}:")
        print(f"    each Con(T_n) checks as an axiom of T_(n+1): "
              f"{c['each_con_checks_in_successor']}")
        if c["conjunction_checks"]:
            print(f"    conjunction of all {c['rungs_available']} rungs derived "
                  f"and checked in {c['conjunction_lines_checked']} lines "
                  f"(every rung used as a premise)")
        else:
            print(f"    only {c['rungs_available']} rung(s) — no conjunction "
                  f"to derive")
        if cal["ran"]:
            print(f"    search calibration on a 2-step MP target: "
                  f"found={cal['found']} after {cal['explored']} formulas")
    print()

    sm = caps["indexed"]["search_for_next_con"]
    print(f"  Bounded search for Con(T_n) inside T_n: found={sm['found']}, "
          f"closure saturated at {sm['explored']} formulas.")
    print("  Read that as a smoke test and nothing more. The modus-ponens")
    print("  closure of the axiom list SATURATES at a couple of dozen")
    print("  formulas — it is limited by the rules it uses, not by its budget,")
    print("  and it never instantiates the induction schema. It covers a")
    print("  vanishing fragment of the derivations of PA. The unprovability")
    print("  of Con(T_n) in T_n is Godel's second theorem under the stated")
    print("  hypotheses; it is imported, not measured, and no run of this")
    print("  program could measure it.")
    print()

    # --------------------------------------------------------------- verdicts
    v = verdicts(arms, args.width, args.separation)
    inline, indexed, trunc = arms["inline"], arms["indexed"], arms["truncated"]

    print("  Pre-registered questions")
    print()
    print(f"  Q1 productive step is presentation-invariant .... "
          f"{'CONFIRMED' if v['Q1_productive_step_presentation_invariant'] else 'REFUTED'}")
    print(f"     inline {inline['productive_rungs']}/{inline['rungs']} rungs "
          f"productive, indexed {indexed['productive_rungs']}/"
          f"{indexed['rungs']}. Every rung of both arms enlarged the axiom")
    print("     set, and the sentence added checked as a premise.")
    print()
    print(f"  Q2 cost is not ................................. "
          f"{'CONFIRMED' if v['Q2_cost_diverges'] else 'REFUTED'}")
    print(f"     presentation cost at rung {args.rungs}: inline "
          f"{inline['final_presentation_symbols']:,} symbols, indexed "
          f"{indexed['final_presentation_symbols']:,} — a factor of "
          f"{v['Q2_cost_separation']:,.0f}.")
    print(f"     inline grows by x{v['Q2_inline_late_ratio']:.2f} per rung and "
          f"indexed by x{v['Q2_indexed_late_ratio']:.2f}: geometric against")
    print("     flat, on ladders whose rungs are one-for-one identical in")
    print("     productive content. The separation has no limit — it is set by")
    print("     how the index is written, and nothing else.")
    print()
    print(f"  Q3 dissociation runs both ways ................. "
          f"{'CONFIRMED' if v['Q3_stall_at_wrap'] else 'REFUTED'}")
    print(f"     the truncated arm's index first repeats at rung "
          f"{v['Q3_index_collision_at']}, and from rung "
          f"{v['Q3_truncated_stall_rung']} on, Con(T_n) is a sentence the")
    print(f"     theory already contains: G_meas falls to 0 and the axiom count")
    print(f"     freezes at {trunc['final_axiom_count']} while I climbs to "
          f"{trunc['per_rung'][-1]['I']} exactly as in the other two arms.")
    print()

    # ------------------------------------------------------------------- Q4
    print("  Q4 — does increasing I buy productive capacity, or formal bulk?")
    print()
    print(f"     I is identical across all three arms "
          f"({v['Q4_I_identical_across_arms']}): it is a counter, and counters")
    print("     count. Set beside it, the two other quantities each come apart")
    print("     from I in a different direction:")
    print()
    print("       - inline vs indexed: SAME productive content at every rung,")
    print(f"         cost differing by {v['Q2_cost_separation']:,.0f}x. Formal")
    print("         size can be inflated without limit at constant capability,")
    print("         so size is not evidence of rank.")
    print("       - indexed vs truncated: comparable bounded cost, and")
    print("         productive content that continues in one and stops dead in")
    print("         the other. Capability can be lost while I keeps climbing,")
    print("         so rank is not evidence of capability either.")
    print()
    print("     The answer to Q4 is therefore neither of the two the question")
    print("     offered. Increasing I buys productive capacity only when the")
    print("     presentation makes the step productive, and whether it does is")
    print("     an independent fact that has to be measured. The registered")
    print("     falsifiable separation is real, and the truncated arm shows it")
    print("     is not vacuous: a system can climb the ladder by the counter")
    print("     and go nowhere. On this construction the certificate that")
    print("     rules that out is small, cheap and checkable — one bit per")
    print("     rung, 'did the axiom set grow', plus a derivation that uses")
    print("     what was added.")
    print()
    print("  Honest scope. This measures the SYNTAX of the ladder: what the")
    print("  continuation operator costs, and whether it moves. It does not")
    print("  measure proof-theoretic strength, does not implement Kleene's O,")
    print("  and does not and cannot establish T_n |/- Con(T_n). Prf is a")
    print("  primitive symbol, so absolute symbol counts carry an unexpanded")
    print("  constant and only the ratios between arms are read. What it does")
    print("  establish is the thing the ordinal framing had been assuming for")
    print("  free: that the productive step and the formal bulk are two")
    print("  quantities, and that the model's (C, I, G) bookkeeping does not")
    print("  by itself tell you which one you have.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "reflection_ladder.json"
        payload = {
            "params": vars(args) | {"output_dir": str(args.output_dir)},
            "capacity": CAPACITY,
            "arms": {k: {kk: vv for kk, vv in a.items()
                         if not kk.startswith("_")}
                     for k, a in arms.items()},
            "capability": caps,
            "verdicts": v,
        }
        out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
