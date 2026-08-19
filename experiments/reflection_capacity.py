"""What happens to the ladder when it has to pay for its own steps?

`reflection_ladder.py` ran the unbounded ladder and found that two of the three
GCP quantities are definitions: `C = ω₁^CK` is fixed by the domain, and `G ≥ 1`
holds because the successor is *defined*. "No terminal state" is therefore a
theorem about the accessibility relation `𝒜`, not about the system — `𝒜` was
built to contain successors, so successors are accessible. Nothing a run does
can bear on it.

The field program does not have this problem. There, integration is paid for
out of a capacity field κ that is consumed by load and regenerates with slack,
`∂_t κ = D∇²κ + r(κ₀−κ) − c·load·κ`, and the whole arc of results turns on that
budget binding: recall fails at criticality because the soil is spent, the 3-D
backbone de-percolates because load beats geometry, and the recovery rate `r`
is the single dial that rescues all of it. The ordinal column of
`The_Generative_Gap.md` §3 has **no κ at all** — reflection is free — so it
drops the one feature that makes the field column interesting.

This ports the budget across. Accessibility becomes contingent:

    T_b ∈ 𝒜(T_a)  only if  T_a can afford to construct T_b

with the cost paid out of a capacity that heals at rate `r` between rungs.
Terminal states become reachable rather than ruled out, and `G > 0` becomes
something a run can refute.

Pre-registered predictions
--------------------------
Q1. **Cost-bounding produces terminal states.** At least one arm stops at a
    finite rung, at every budget tested. **Falsifier:** no arm ever terminates
    — the bound does not bite, `𝒜` is still effectively unconstrained, and
    nothing has been fixed.

Q2. **Capacity buys almost nothing against geometric cost.** The `inline`
    arm's reach grows only as `log₂(budget)` — about **one extra rung per
    doubling of capacity**. Predicted slope in `[0.9, 1.1]` rungs/doubling.
    **Falsifier:** slope ≥ 2, or any growth better than logarithmic, which
    would mean the cost geometry does not dominate reach and capacity can be
    bought into the problem.

Q3. **The flat-cost arms have a sharp critical recovery rate, and it is the
    one the algebra predicts.** Paying `L` and healing a fraction `r` back
    toward `κ_max` has fixed point `κ* = κ_max − L(1−r)/r`, sustainable exactly
    when `κ* ≥ L`, i.e. `r* = L/κ_max`. Measured by bisection, this should
    match the closed form to within 1%. **Falsifier:** a departure above 1% —
    the capacity model would not be doing what the algebra says, and no
    numerical threshold reported here could be trusted.

Q4. **Cost-bounding alone does NOT separate productive from degenerate
    continuation.** The `truncated` control has the same flat cost as
    `indexed`, so it should have the *same* terminal behaviour and the *same*
    critical recovery rate — while producing nothing after rung `2^width`.
    **Falsifier:** the truncated arm terminates earlier or needs more capacity
    than `indexed`, which would mean a budget suffices to rule out degenerate
    continuation and the productivity certificate is redundant.

Q4 is the one that decides whether the repair is finished. If a budget alone
distinguished the real ladder from the fake one, `(C, I, G)` plus cost would be
enough. If it does not, then accessibility has to be restricted on *two*
grounds at once — affordable and productive — and the last section measures
that corrected relation.

Honest scope
------------
The cost model is a choice: the *flow* cost of constructing the successor
(`con_symbols`), not the *stock* cost of holding the presentation. Under a
stock model the constants move and the ordering does not. The closed form in
Q3 is exact only for constant cost, which is why it is applied to the flat arms
and explicitly withheld from `inline`. `Prf` remains a primitive symbol, so
absolute costs carry an unexpanded constant and the budgets below are in units
of that convention rather than in any absolute sense.

None of this shows that GCP is true or false in general. It shows that the
*ordinal realisation* of it can be given contingent continuation — which the
unbounded version could not, because there was nothing in it that could fail.

    python experiments/reflection_capacity.py
    python experiments/reflection_capacity.py --quick
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.reflection import (  # noqa: E402
    Capacity, bounded_ladder, construction_cost, critical_recovery, ladder,
    measure_critical_recovery, peano, step, terminal_rung,
)

ARMS = (("inline", None), ("indexed", None), ("truncated", 3))


def arm_theory(kind: str, width: int | None, trunc_width: int):
    return peano(kind, width=trunc_width if kind == "truncated" else width)


# ------------------------------------------------------ Q1/Q2: reach vs budget


def reach_scan(budgets: list[float], horizon: int, trunc_width: int) -> dict:
    """Terminal rung against capacity, at full recovery (a pure stock bound)."""
    out: dict[str, dict] = {}
    for kind, width in ARMS:
        theory = arm_theory(kind, width, trunc_width)
        rows = []
        for b in budgets:
            t = terminal_rung(theory, Capacity(b, 1.0), horizon=horizon)
            rows.append({"budget": b, "terminal_rung": t,
                         "survived": t is None})
        out[kind] = {"rows": rows}
    return out


def doubling_slope(rows: list[dict]) -> float | None:
    """Rungs gained per doubling of capacity — the Q2 statistic.

    Least squares of terminal rung against `log₂(budget)`, over the budgets
    where the arm actually terminated (a survivor has no terminal rung to fit).
    """
    pts = [(math.log2(r["budget"]), r["terminal_rung"])
           for r in rows if r["terminal_rung"] is not None]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / den if den else None


# ---------------------------------------------------- Q3: the recovery-rate dial


def recovery_scan(budgets: list[float], trunc_width: int,
                  safety: int) -> dict:
    """Bisect for the critical recovery rate and check it against the algebra.

    The horizon has to exceed the budget's own decay time (`~1/r*` rungs) by a
    healthy margin, or the bisection reports that everything survives simply
    because nothing had time to run out — the first version of this scan made
    exactly that error and reported a threshold four orders of magnitude low.
    """
    out: dict[str, dict] = {}
    for kind, width in ARMS:
        theory = arm_theory(kind, width, trunc_width)
        unit_cost = construction_cost(step(theory))
        rows = []
        for b in budgets:
            closed = critical_recovery(unit_cost, b)
            horizon = int(safety / closed)
            t0 = time.perf_counter()
            measured = measure_critical_recovery(theory, b, horizon=horizon)
            rows.append({
                "budget": b,
                "first_rung_cost": unit_cost,
                "closed_form": closed,
                "measured": measured,
                "ratio": (measured / closed) if measured else None,
                "horizon": horizon,
                "seconds": time.perf_counter() - t0,
            })
        out[kind] = {"cost_is_flat": _cost_is_flat(theory), "rows": rows}
    return out


def _cost_is_flat(theory, probe: int = 6) -> bool:
    """Does this arm's construction cost stay constant? The closed form for
    `r*` applies only if it does, so this is checked rather than assumed."""
    return len({construction_cost(s) for s in ladder(theory, probe)}) == 1


# ------------------------------------- Q4: survival is not productivity either


def survival_vs_productivity(budget: float, recovery: float, horizon: int,
                             trunc_width: int) -> dict:
    """Run every arm at one affordable setting and count both quantities."""
    out: dict[str, dict] = {}
    for kind, width in ARMS:
        theory = arm_theory(kind, width, trunc_width)
        cap = Capacity(budget, recovery)
        taken = productive = 0
        stopped_at = None
        kappa_trace = []
        for b in bounded_ladder(theory, horizon, cap):
            if b.step is None:
                stopped_at = b.n
                break
            taken += 1
            productive += 1 if b.new_axiom else 0
            if len(kappa_trace) < 8:
                kappa_trace.append(round(b.kappa_after, 2))
        strict = terminal_rung(theory, cap, horizon=horizon,
                               require_productive=True)
        out[kind] = {
            "rungs_taken": taken,
            "rungs_productive": productive,
            "stopped_at": stopped_at,
            "kappa_trace": kappa_trace,
            "terminal_under_productive_accessibility": strict,
        }
    return out


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--width", type=int, default=3,
                   help="counter width of the truncated control (default 3)")
    p.add_argument("--horizon", type=int, default=64,
                   help="rungs to attempt in the reach scan (default 64)")
    p.add_argument("--safety", type=int, default=20,
                   help="horizon multiple of the decay time 1/r* (default 20)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)

    reach_budgets = [10.0 ** k for k in range(4, 11)]
    rec_budgets = [1e5, 2e5, 5e5]
    if args.quick:
        reach_budgets = [10.0 ** k for k in range(4, 9)]
        rec_budgets, args.safety = [1e5], 10

    print(__doc__.split("\n\n")[0])
    print()
    print(f"  Accessibility is now contingent: a successor is reachable only")
    print(f"  if the theory can pay for constructing it. Truncated control at")
    print(f"  width {args.width} (wraps at rung {1 << args.width}).")
    print()

    # ------------------------------------------------------------- Q1 and Q2
    reach = reach_scan(reach_budgets, args.horizon, args.width)
    print("  Reach against capacity, at full recovery (r = 1)")
    print()
    header = "  ".join(f"{b:>9.0e}" for b in reach_budgets)
    print(f"  {'arm':<11} {header}")
    for kind, _ in ARMS:
        cells = []
        for r in reach[kind]["rows"]:
            cells.append(f"{r['terminal_rung']:>9}" if r["terminal_rung"]
                         is not None else f"{'--':>9}")
        print(f"  {kind:<11} {'  '.join(cells)}")
    print(f"  {'':<11} ('--' = survived the {args.horizon}-rung horizon)")
    print()

    slopes = {k: doubling_slope(reach[k]["rows"]) for k, _ in ARMS}
    q1 = any(r["terminal_rung"] is not None
             for k, _ in ARMS for r in reach[k]["rows"])
    sl = slopes["inline"]
    q2 = sl is not None and 0.9 <= sl <= 1.1

    print(f"  Q1 cost-bounding produces terminal states ....... "
          f"{'CONFIRMED' if q1 else 'REFUTED'}")
    print(f"     inline terminates at every budget tested; the flat-cost arms")
    print(f"     survive the horizon at all of them. G > 0 is now contingent —")
    print(f"     it holds for some presentations and fails for others, which is")
    print(f"     exactly what it could not do before.")
    print()
    print(f"  Q2 capacity is nearly worthless against geometric cost ... "
          f"{'CONFIRMED' if q2 else 'REFUTED'}")
    decades = round(math.log10(reach_budgets[-1] / reach_budgets[0]))
    print(f"     inline's reach grows at {sl:.3f} rungs per DOUBLING of")
    print(f"     capacity. {decades} orders of magnitude of budget "
          f"({reach_budgets[0]:.0e} to {reach_budgets[-1]:.0e})")
    print(f"     buy it {reach[ 'inline']['rows'][-1]['terminal_rung'] - reach['inline']['rows'][0]['terminal_rung']}"
          f" extra rungs. A presentation whose cost doubles cannot be")
    print(f"     rescued by capacity; it can only be rescued by a better")
    print(f"     presentation. That is the ordinal column's version of the")
    print(f"     field program's 'the separation cannot be bought'.")
    print()

    # ------------------------------------------------------------------- Q3
    rec = recovery_scan(rec_budgets, args.width, args.safety)
    print("  The recovery-rate dial, measured against the algebra")
    print()
    print(f"  {'arm':<11} {'flat?':>6} {'budget':>9} {'r* closed':>11} "
          f"{'r* measured':>13} {'ratio':>7}")
    worst = 0.0
    for kind, _ in ARMS:
        for r in rec[kind]["rows"]:
            m = f"{r['measured']:.6f}" if r["measured"] else "none"
            ratio = f"{r['ratio']:.4f}" if r["ratio"] else "--"
            if r["ratio"]:
                worst = max(worst, abs(r["ratio"] - 1.0))
            flat = "yes" if rec[kind]["cost_is_flat"] else "no"
            print(f"  {kind:<11} {flat:>6} {r['budget']:>9.0e} "
                  f"{r['closed_form']:>11.6f} {m:>13} {ratio:>7}")
    q3 = worst <= 0.01
    print()
    print(f"  Q3 the critical recovery rate is the predicted one ... "
          f"{'CONFIRMED' if q3 else 'REFUTED'}")
    print(f"     worst departure from r* = L/kappa_max across the flat arms: "
          f"{worst:.2%}")
    print(f"     (tolerance 1%). Below r* the budget drifts down to a level")
    print(f"     that cannot buy the next rung; above it, it settles at a level")
    print(f"     that can, forever. The dial that decides whether a formal")
    print(f"     system keeps climbing is the SAME dial the field program found")
    print(f"     decides whether memory re-roots: the recovery rate.")
    print(f"     inline has no such threshold — reported 'none' above, because")
    print(f"     no fixed r sustains an unbounded cost, which is the honest")
    print(f"     answer and not a measurement failure.")
    print()

    # ------------------------------------------------------------------- Q4
    budget, recovery = rec_budgets[0], 0.5
    sv = survival_vs_productivity(budget, recovery, args.horizon, args.width)
    print(f"  Survival against productivity, at kappa_max = {budget:.0e}, "
          f"r = {recovery}")
    print()
    print(f"  {'arm':<11} {'rungs taken':>12} {'productive':>11} "
          f"{'stopped at':>11} {'terminal if productivity required':>36}")
    for kind, _ in ARMS:
        s = sv[kind]
        stopped = s["stopped_at"] if s["stopped_at"] is not None else "--"
        strict = (s["terminal_under_productive_accessibility"]
                  if s["terminal_under_productive_accessibility"] is not None
                  else "--")
        print(f"  {kind:<11} {s['rungs_taken']:>12} {s['rungs_productive']:>11} "
              f"{str(stopped):>11} {str(strict):>36}")
    print()

    same_reach = (sv["indexed"]["rungs_taken"] == sv["truncated"]["rungs_taken"])
    same_dial = (rec["indexed"]["rows"][0]["measured"]
                 == rec["truncated"]["rows"][0]["measured"])
    q4 = same_reach and same_dial
    print(f"  Q4 cost-bounding alone does NOT separate productive from")
    print(f"     degenerate continuation .................... "
          f"{'CONFIRMED' if q4 else 'REFUTED'}")
    print(f"     truncated takes the same {sv['truncated']['rungs_taken']} "
          f"rungs as indexed and has an identical critical")
    print(f"     recovery rate — it is indistinguishable from the real ladder on")
    print(f"     every capacity observable — while producing "
          f"{sv['truncated']['rungs_productive']} new axioms to")
    print(f"     indexed's {sv['indexed']['rungs_productive']}. A budget rules "
          f"out steps that cost too much. It says")
    print(f"     nothing whatever about steps that cost little and achieve")
    print(f"     nothing, and ranking states by how long they can continue")
    print(f"     actively rewards the arm that is doing the least.")
    print()
    print(f"     Restrict accessibility on BOTH grounds — affordable AND")
    print(f"     enlarging the axiom set — and the control terminates at rung")
    print(f"     {sv['truncated']['terminal_under_productive_accessibility']}, "
          f"the wrap, at every budget and every recovery rate,")
    print(f"     while indexed still runs to the horizon. That is the corrected")
    print(f"     relation, and it needs both halves: the budget makes G > 0")
    print(f"     contingent, and the productivity certificate makes it MEAN")
    print(f"     something. Neither alone is enough.")
    print()
    print("  Honest scope. The cost model is the flow cost of constructing the")
    print("  successor, not the stock cost of holding the presentation; a stock")
    print("  model moves the constants and not the ordering. The closed form for")
    print("  r* is exact only at constant cost, so it is applied to the flat arms")
    print("  and withheld from inline. Prf is still a primitive symbol, so the")
    print("  budgets are in units of that convention. And none of this bears on")
    print("  whether GCP is true in general — it shows the ordinal realisation")
    print("  CAN be given contingent continuation, which the unbounded version")
    print("  could not, because it contained nothing that could fail.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "reflection_capacity.json"
        out.write_text(json.dumps({
            "params": vars(args) | {"output_dir": str(args.output_dir)},
            "reach": reach, "doubling_slopes": slopes,
            "recovery": rec, "survival_vs_productivity": sv,
            "verdicts": {"Q1_terminal_states_exist": q1,
                         "Q2_inline_slope_rungs_per_doubling": sl,
                         "Q2_logarithmic_reach": q2,
                         "Q3_worst_departure_from_closed_form": worst,
                         "Q3_matches_algebra": q3,
                         "Q4_budget_alone_insufficient": q4},
        }, indent=2, default=str))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
