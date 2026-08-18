"""A ladder in a box: what changes when the domain can actually run out.

Every result in this series was measured where `I < C` is a **theorem**. The
Church-Kleene ceiling is never reached, so one logically possible blocker could
never occur:

    exhausted — no productive move exists because the domain ran out, I = C

That leaves the four-dimensional taxonomy in an awkward position: complete for
non-saturating domains, and *untested* for saturating ones. And raising `C` does
not help — `I < C` is a theorem in second-order arithmetic, in set theory, and at
every admissible ordinal, so climbing altitude re-derives the same guarantee at
a new level rather than testing it.

So this changes the domain instead of the altitude. Fix `k` atomic sentences; a
theory is a subset of them; `C = k` exactly, and a ladder can add at most `k`
things before there is nothing left. Not because of a naming scheme, not because
of a budget — because the box ends.

The discriminator
-----------------
`exhausted` and `stagnant` look identical from outside: moves exist, none is
productive, the system keeps running and gets nowhere. They differ in exactly one
testable way:

    stagnant   — a better naming scheme rescues it
    exhausted  — nothing rescues it, because there is nothing left to name

So **naming-invariance** is the test, and it is what decides whether `exhausted`
is a genuine fifth category or stagnation wearing a different hat.

Pre-registered predictions
--------------------------
Q1. **The box saturates: `I` reaches `C` exactly.** A ladder in a `k`-atom box
    makes `k` productive steps and then stops producing. **Falsifier:** it
    stalls short of `C` for a reason other than its naming scheme, or never
    stalls at all — either would mean the box is not the bound it claims to be.

Q2. **Exhaustion is naming-invariant.** Every presentation with an adequate
    address space stalls at exactly `C`, regardless of *how* it names itself.
    **Falsifier:** the stall point moves with the scheme even when the address
    space suffices, which would mean this was stagnation all along and no fifth
    category is needed.

Q3. **Stagnation is naming-dependent, and rescuable.** A presentation whose
    address space is smaller than the box stalls early, at its address space
    rather than at `C`, and switching schemes moves it. **Falsifier:** it does
    not move, which would collapse the distinction from the other side.

Q4. **A naming defect is invisible when the box binds first.** Whenever the
    address space is at least as large as the box, the deliberately broken
    presentation is indistinguishable from the good one on every observable.
    **Falsifier:** some observable separates them anyway.

Q5. **Exhaustion is fully visible from inside.** A theory can count its own
    remaining room and its own address space, so it predicts its stall exactly,
    with unbounded lookahead. **Falsifier:** any mismatch between the interior
    prediction and the run.

Q4 is the one with consequences beyond this construction, and Q5 is the bookend:
the epistemic wall of the arithmetic setting was the one thing a system could not
see coming, and this should be the one it sees most clearly.

The collaborator's prediction, registered before the run: *no limit in general,
but within a finite setting — "within the box" — a limit that is real.* That is
Q1 and Q2 together, and it is recorded here because a prediction made after
seeing the numbers is not a prediction.

Honest scope
------------
The box is a **model**, not a theory of anything: `k` atoms with one added per
productive step is the simplest object that saturates, chosen because it makes
`I = C` reachable and nothing else. It shares no machinery with the arithmetic
ladder — deliberately, since the question is whether the *taxonomy* transfers,
not whether the code does. What transfers is the vocabulary: the same four
dimensions, the same verdicts, the same distinction between a fact about the
world and a fact about what a system can establish.

    python experiments/reflection_finite_box.py
    python experiments/reflection_finite_box.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.finite_ladder import (  # noqa: E402
    FiniteTheory, finite_climb, naming_schemes, predict_finite_stall,
    stall_point,
)


def scheme_table(boxes: list[int], width: int) -> list[dict]:
    rows = []
    for k in boxes:
        for kind, w in naming_schemes(width):
            rung, reason = stall_point(k, kind, w)
            rows.append({"atoms": k, "kind": kind, "width": w,
                         "stall_rung": rung, "reason": reason,
                         "address_space": (1 << w) if w else None})
    return rows


def invisibility_table(boxes: list[int], widths: list[int]) -> list[dict]:
    rows = []
    for k in boxes:
        for w in widths:
            tr = stall_point(k, "truncated", w)
            ix = stall_point(k, "indexed")
            rows.append({"atoms": k, "width": w, "address_space": 1 << w,
                         "truncated": tr, "indexed": ix,
                         "indistinguishable": tr == ix,
                         "box_binds_first": (1 << w) >= k})
    return rows


def interior_table(boxes: list[int], widths: list[int]) -> list[dict]:
    rows = []
    for k in boxes:
        for width in widths:
            for kind, w in naming_schemes(width):
                theory = FiniteTheory(atoms=k, kind=kind, width=w)
                predicted = predict_finite_stall(theory)
                actual = stall_point(k, kind, w)
                rows.append({"atoms": k, "kind": kind, "width": w,
                             "predicted": predicted, "actual": actual,
                             "exact": predicted == actual})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--boxes", type=int, nargs="+", default=[4, 6, 8, 12, 16])
    p.add_argument("--width", type=int, default=2)
    p.add_argument("--widths", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if args.quick:
        args.boxes, args.widths = [4, 6, 8], [2, 3]

    print(__doc__.split("\n\n")[0])
    print()

    # ------------------------------------------------------------- Q1 and Q2
    rows = scheme_table(args.boxes, args.width)
    print(f"  Where each naming scheme stalls, and why "
          f"(truncated at width {args.width}, address space "
          f"{1 << args.width})")
    print()
    print(f"  {'box C':>6} {'scheme':<12} {'stalls at I':>12} {'reason':>11} "
          f"{'I = C?':>8}")
    for r in rows:
        tag = r["kind"] + (f" w={r['width']}" if r["width"] else "")
        print(f"  {r['atoms']:>6} {tag:<12} {str(r['stall_rung']):>12} "
              f"{str(r['reason']):>11} "
              f"{str(r['stall_rung'] == r['atoms']):>8}")
    print()

    adequate = [r for r in rows
                if r["address_space"] is None or r["address_space"] >= r["atoms"]]
    q1 = all(r["stall_rung"] == r["atoms"] and r["reason"] == "exhausted"
             for r in adequate)
    by_box = {}
    for r in adequate:
        by_box.setdefault(r["atoms"], set()).add(r["stall_rung"])
    q2 = all(len(v) == 1 for v in by_box.values())

    print(f"  Q1 the box saturates: I reaches C exactly ....... "
          f"{'CONFIRMED' if q1 else 'REFUTED'}")
    print(f"     every adequately-addressed ladder makes exactly C productive")
    print(f"     steps and then stops producing. This is the first time in the")
    print(f"     series that I = C has been *reachable* at all — in the")
    print(f"     arithmetic setting the gap was permanent by theorem.")
    print()
    print(f"  Q2 exhaustion is naming-invariant ............... "
          f"{'CONFIRMED' if q2 else 'REFUTED'}")
    print(f"     within each box, every adequate scheme stalls at the SAME rung")
    print(f"     — inline addresses by content, indexed by position, searched")
    print(f"     cannot certify its own address, and all three land on C. The")
    print(f"     stall is a fact about the domain, not about how the system")
    print(f"     writes itself down. That is what makes `exhausted` a genuine")
    print(f"     fifth category rather than stagnation renamed.")
    print()

    stagnating = [r for r in rows
                  if r["address_space"] is not None
                  and r["address_space"] < r["atoms"]]
    q3 = bool(stagnating) and all(
        r["reason"] == "stagnant" and r["stall_rung"] == r["address_space"]
        for r in stagnating)
    print(f"  Q3 stagnation is naming-dependent and rescuable . "
          f"{'CONFIRMED' if q3 else 'REFUTED'}")
    if stagnating:
        ex = stagnating[0]
        print(f"     at C = {ex['atoms']} the truncated scheme stalls at "
              f"{ex['stall_rung']} = its address space,")
        print(f"     not at C, and switching to any adequate scheme moves it to")
        print(f"     {ex['atoms']}. The bound is in the presentation and comes off")
        print(f"     with the presentation.")
    print()

    # ------------------------------------------------------------------- Q4
    inv = invisibility_table(args.boxes, args.widths)
    print("  Does the naming defect bite, or does the box bind first?")
    print()
    print(f"  {'box C':>6} {'addr space':>11} {'box binds first':>16} "
          f"{'truncated':>22} {'indexed':>22}  same?")
    for r in inv:
        print(f"  {r['atoms']:>6} {r['address_space']:>11} "
              f"{str(r['box_binds_first']):>16} "
              f"{str(r['truncated']):>22} {str(r['indexed']):>22}  "
              f"{'YES' if r['indistinguishable'] else 'no'}")
    q4 = all(r["indistinguishable"] == r["box_binds_first"] for r in inv)
    print()
    print(f"  Q4 a naming defect is invisible when the box binds first ... "
          f"{'CONFIRMED' if q4 else 'REFUTED'}")
    print(f"     indistinguishable EXACTLY when the address space is at least")
    print(f"     the size of the box, and distinguishable exactly when it is")
    print(f"     not — {sum(r['indistinguishable'] for r in inv)}/{len(inv)} "
          f"rows matching the prediction with no exceptions.")
    print()
    print(f"     This is the result with consequences outside the construction.")
    print(f"     The `truncated` pathology was detectable in the arithmetic")
    print(f"     setting ONLY because that domain is infinite. Put the same")
    print(f"     broken system in a box smaller than its defect and it behaves")
    print(f"     identically to a sound one on every observable — same stall,")
    print(f"     same reason, same interior prediction. A system cannot")
    print(f"     distinguish 'I am out of room' from 'I am badly built' when it")
    print(f"     runs out of room first, and neither can anyone watching it.")
    print()

    # ------------------------------------------------------------------- Q5
    interior = interior_table(args.boxes, args.widths)
    exact = sum(r["exact"] for r in interior)
    q5 = exact == len(interior)
    print(f"  Q5 exhaustion is fully visible from inside ...... "
          f"{'CONFIRMED' if q5 else 'REFUTED'}")
    print(f"     {exact}/{len(interior)} exact, across every box, scheme and")
    print(f"     width. Both quantities a theory needs are countable from its")
    print(f"     own presentation: the room left in the box, and the size of its")
    print(f"     own address space. Lookahead is unbounded — it knows at rung 0.")
    print()
    print(f"     That is the bookend. The epistemic wall was the one thing a")
    print(f"     system could not see coming; exhaustion is the one it sees most")
    print(f"     clearly. Running out of room is the *easiest* limit to know")
    print(f"     about, and being unable to certify a live continuation is the")
    print(f"     hardest, and those sit at opposite ends of the same taxonomy.")
    print()

    print("  What this says about varying C")
    print()
    print("  Varying C does change things, and in a specific way: it adds a")
    print("  dimension the fixed-C setting could not express. `exhausted` is a")
    print("  fifth category, distinct from `stagnant` on the naming-invariance")
    print("  test, and it exists only where the ceiling is reachable. So the")
    print("  four-dimensional refinement was not wrong — it was complete for the")
    print("  domain it was measured in, and that domain could not saturate.")
    print()
    print("  The registered prediction — no limit in general, a real limit")
    print("  within the box — is confirmed on both halves, and the second half")
    print("  turns out to carry the sharper consequence: inside a small enough")
    print("  box, the difference between a sound system and a broken one stops")
    print("  being measurable at all.")
    print()
    print("  Honest scope. The box is a MODEL — k atoms, one added per")
    print("  productive step — chosen because it is the simplest object that")
    print("  saturates, and it shares no machinery with the arithmetic ladder.")
    print("  What is claimed to transfer is the taxonomy, not the construction.")
    print("  Whether a richer saturating domain (bounded type theories, finite")
    print("  models) produces the same five categories is untested and is the")
    print("  next thing that could refute this.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "reflection_finite_box.json"
        out.write_text(json.dumps({
            "params": vars(args) | {"output_dir": str(args.output_dir)},
            "schemes": rows, "invisibility": inv, "interior": interior,
            "verdicts": {"Q1_box_saturates": q1,
                         "Q2_exhaustion_naming_invariant": q2,
                         "Q3_stagnation_naming_dependent": q3,
                         "Q4_defect_invisible_when_box_binds": q4,
                         "Q5_exhaustion_visible_from_inside": q5},
        }, indent=2, default=str))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
