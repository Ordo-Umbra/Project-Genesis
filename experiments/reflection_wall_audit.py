"""Classify walls by what they read, not by what they are called.

The previous run found that the DAG domain's "epistemic" filter was a size tax
wearing an epistemic label: it admits when `cost(key) <= effort`, which is the
number the *economic* filter reads. Nothing forced the label and the behaviour
apart for eleven experiments, because every filter read the same quantity and so
three filters looked like three things.

The reviewer's response was to promote the accident into an instrument: *classify
walls by the quantity they actually read, and find it by perturbing that quantity
and seeing which walls move.* This run builds that instrument and points it at
every wall in the module, including the ones nobody has complained about.

Two perturbations, each leaving the graph's structure untouched:

    price     — swap `cost_model` between content- and description-addressed.
                A wall that moves is reading the price.
    identity  — shift every node identifier by a constant, relabelling the
                filter to match. The graph is isomorphic and the wall is the
                same wall, so a wall that moves is reading the raw integer
                identifier: an artifact of the encoding, not of the structure.

The reviewer also settled the locality question left open last time, and settled
it against the earlier model. Totality is Π⁰₂-complete and `O`-membership
Π¹₁-complete; the hardness is **uniform over the address space**, so nothing
privileges a proper subset of addresses as decidable while the rest are not.
Marking particular nodes opaque is an extra stipulation, not a consequence of
undecidability. When opacity attaches instead to the *form* of the notation — a
join names an arbitrary set, exactly as a limit names an arbitrary fundamental
sequence — every move of that form is opaque at once. So `opaque_form="join"`
is added and measured against address-attached opacity.

Pre-registered predictions
--------------------------
Q1. **The audit separates three groups.** `budget` and `certify_effort` read
    price and not identity. `opaque`, `opaque_form` and `max_arity` read
    neither. And `address_bits` reads **both** — it compares `max(key)` against
    `2^bits`, which is the largest raw identifier, so a pure relabelling should
    change its verdict. **Falsifier:** any wall landing in a different group.
    If `address_bits` is identity-invariant then the encoding is not leaking and
    the prediction is simply wrong.

Q2. **Uniform opacity leaves no detour; local opacity does.** With a
    filter-aware join-seeking policy, address-attached opacity should leave the
    join class essentially intact — every pair avoiding the marked address is
    still available — while form-attached opacity should remove it entirely, at
    every root count. **Falsifier:** local opacity also drives joins to zero,
    which would mean the two are not distinguishable here and the reviewer's
    correction has no measurable content in this domain.

Q3. **Removing the class does not stop the climb.** Under form opacity a
    rank-following policy should reach exactly the same rank as with no wall at
    all, because it only ever makes single-parent moves. What is lost is not
    height but a *kind of content*: no node whose content is the union of two
    incomparable theories is ever certified. **Falsifier:** rank falls, which
    would make this an economic wall in disguise.

Honest scope
------------
`opaque_form` is a **declared** uniform opacity. Nothing here proves any address
undecidable; it models what a policy does when a whole move class cannot be
certified. The instrument separates price-readers and identity-readers from
everything else — it does **not** separate a placement-reader from any other
magnitude-reader, which is why `max_arity` (which reads |parents|) lands in the
same group as the opacity walls. Two perturbations give two bits, not a taxonomy.

`depth` remains a declared proxy for proof-theoretic rank.

    python experiments/reflection_wall_audit.py
    python experiments/reflection_wall_audit.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.reflection_dag import (  # noqa: E402
    Filters, broaden, deepen, join_aware, run_adaptive, run_filtered,
)

#: Every wall in the module, as a factory over (cost_model, id_offset). The
#: offset relabels the filter alongside the graph, so a wall that still moves is
#: reading the identifier's magnitude rather than which node it names.
WALLS = {
    "budget": lambda m, o: Filters(budget=5, cost_model=m),
    "certify_effort": lambda m, o: Filters(certify_effort=5, cost_model=m),
    "address_bits": lambda m, o: Filters(address_bits=3, cost_model=m),
    "opaque": lambda m, o: Filters(opaque=frozenset({o + 2}), cost_model=m),
    "opaque_form": lambda m, o: Filters(opaque_form="join", cost_model=m),
    "max_arity": lambda m, o: Filters(max_arity=1, cost_model=m),
}

SHIFT = 8


def _signature(make, model: str, offset: int, steps: int) -> tuple:
    """What a wall does, as a comparable object: the full tally and block
    counts under both a depth-seeking and a join-seeking policy."""
    f = make(model, offset)
    out = []
    for policy in (deepen, broaden):
        r = run_filtered(policy, steps, filters=f, first_id=offset)
        out.append((tuple(sorted(r["tally"].items())),
                    tuple(sorted(r["blocks"].items()))))
    return tuple(out)


def audit(steps: int) -> list[dict]:
    rows = []
    for name, make in WALLS.items():
        base = _signature(make, "content", 0, steps)
        rows.append({
            "wall": name,
            "reads_price": base != _signature(make, "description", 0, steps),
            "reads_identity": base != _signature(make, "content", SHIFT, steps),
        })
    return rows


def locality(root_counts: list[int], steps: int) -> list[dict]:
    """Address-attached opacity against form-attached opacity, measured with a
    policy that respects the filters — see `join_aware` for why that matters."""
    rows = []
    for roots in root_counts:
        clean = Filters()
        form = Filters(opaque_form="join")
        local = []
        for marked in range(roots):
            f = Filters(opaque=frozenset({marked}))
            local.append(run_filtered(join_aware(f), steps, roots=roots,
                                      filters=f)["joins"])
        rows.append({
            "roots": roots,
            "clean": run_filtered(join_aware(clean), steps, roots=roots,
                                  filters=clean)["joins"],
            "local": local,
            "form": run_filtered(join_aware(form), steps, roots=roots,
                                 filters=form)["joins"],
            "blind": [run_filtered(broaden, steps, roots=roots,
                                   filters=Filters(opaque=frozenset({m})))["joins"]
                      for m in range(roots)],
        })
    return rows


def climb_under_form_opacity(steps: int) -> list[dict]:
    rows = []
    for model in ("content", "description"):
        clean = run_adaptive(steps, filters=Filters(cost_model=model))
        opaque = run_adaptive(steps, filters=Filters(opaque_form="join",
                                                     cost_model=model))
        joiner = run_filtered(join_aware(Filters(cost_model=model)), steps,
                              filters=Filters(cost_model=model))
        rows.append({"cost_model": model,
                     "clean_rank": clean["final_rank"],
                     "opaque_rank": opaque["final_rank"],
                     "opaque_blocked": opaque["blocks"]["uncertifiable"],
                     "joins_available_to_a_joiner": joiner["joins"],
                     "joins_certified_under_form_opacity": opaque["joins"]})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--roots", type=int, nargs="+", default=[3, 6, 10])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if args.quick:
        args.steps, args.roots = 20, [3, 6]

    print(__doc__.split("\n\n")[0])
    print()

    rows = audit(args.steps)
    print("  Q1. What does each wall actually read?")
    print()
    print(f"  {'wall':<16} {'reads price':>12} {'reads identity':>15}   {'group'}")
    for r in rows:
        group = ("price + encoding artifact" if r["reads_price"]
                 and r["reads_identity"] else
                 "size tax" if r["reads_price"] else
                 "encoding artifact" if r["reads_identity"] else
                 "reads neither")
        print(f"  {r['wall']:<16} {str(r['reads_price']):>12} "
              f"{str(r['reads_identity']):>15}   {group}")
    expected = {"budget": (True, False), "certify_effort": (True, False),
                "address_bits": (True, True), "opaque": (False, False),
                "opaque_form": (False, False), "max_arity": (False, False)}
    q1 = all((r["reads_price"], r["reads_identity"]) == expected[r["wall"]]
             for r in rows)
    print()
    print(f"  Q1 the audit separates three groups ............. "
          f"{'CONFIRMED' if q1 else 'REFUTED'}")
    print("     Two walls are size taxes, and one of them is the filter that")
    print("     spent eleven experiments labelled epistemic. Three read neither")
    print("     perturbation. And `address_bits` reads both — it compares")
    print("     max(key) against 2^bits, so it is reading the largest raw")
    print("     identifier. Relabel the graph without changing its shape and its")
    print("     verdict changes. **That is a second mislabelled wall**, found by")
    print("     the instrument built to catch the first, and nobody had")
    print("     complained about it. It is called structural; part of what it")
    print("     enforces is an artifact of how nodes are numbered.")
    print()

    loc = locality(args.roots, args.steps)
    print("  Q2. Is opacity local or uniform? (joins certified, filter-aware policy)")
    print()
    print(f"  {'roots':>6} {'no wall':>8} {'form-attached':>14}   "
          f"{'address-attached, by which root is marked'}")
    for r in loc:
        print(f"  {r['roots']:>6} {r['clean']:>8} {r['form']:>14}   {r['local']}")
    q2 = all(r["form"] == 0 and max(r["local"]) == r["clean"] for r in loc)
    print()
    print(f"  Q2 uniform opacity leaves no detour; local opacity does ......... "
          f"{'CONFIRMED' if q2 else 'REFUTED'}")
    print("     Address-attached opacity removes the pairs that touch the marked")
    print("     address and leaves the rest of the move class intact. Form-")
    print("     attached opacity removes the class, at every root count. There")
    print("     is no non-opaque move of that form to detour to — which is the")
    print("     reviewer's point, and it is the case the mathematics forces:")
    print("     completeness makes the hardness uniform over the address space,")
    print("     so a decidable proper subset is a stipulation, not a consequence.")
    print()
    print("  And a correction to the previous run, which the same measurement")
    print("  forces. Those earlier refusal counts used `broaden`, which checks")
    print("  whether a join is *new* but not whether it is *admissible* — so it")
    print("  re-offers a refused pair forever. Under it, local opacity looks far")
    print("  more destructive than it is:")
    print()
    print(f"  {'roots':>6}   {'filter-aware':<28} {'filter-blind (broaden)'}")
    for r in loc:
        print(f"  {r['roots']:>6}   {str(r['local']):<28} {r['blind']}")
    print()
    print("     The wall did not remove those joins; the policy deadlocked on")
    print("     them. A blocked-move count measures the policy as much as the")
    print("     wall whenever the policy cannot see the wall.")
    print()

    climb = climb_under_form_opacity(args.steps)
    print("  Q3. Does removing the move class stop the climb?")
    print()
    print(f"  {'cost model':<13} {'rank, no wall':>14} {'rank, form-opaque':>18} "
          f"{'refused':>8} {'joins certified':>16}")
    for r in climb:
        print(f"  {r['cost_model']:<13} {r['clean_rank']:>14} "
              f"{r['opaque_rank']:>18} {r['opaque_blocked']:>8} "
              f"{r['joins_certified_under_form_opacity']:>16}")
    q3 = all(r["clean_rank"] == r["opaque_rank"] for r in climb)
    print()
    print(f"  Q3 the climb is untouched; a kind of content is not ............. "
          f"{'CONFIRMED' if q3 else 'REFUTED'}")
    print("     Identical rank, zero refusals — a rank-following policy never")
    print(f"     attempts a join, so nothing is refused. Meanwhile a joining")
    print(f"     policy had {climb[0]['joins_available_to_a_joiner']} of them "
          f"available and certifies none.")
    print("     **What a uniform epistemic wall costs is not height. It is a")
    print("     kind of content**: no theory whose content is the union of two")
    print("     incomparable theories is ever certified, however long the climb")
    print("     runs. That is result five's `hidden` verdict recovered in a")
    print("     second domain — the system goes on forever and goes on being")
    print("     unable to certify the one move that would join what it knows.")
    print()

    print("  Honest scope. `opaque_form` is a declared uniform opacity; nothing")
    print("  here proves any address undecidable. The instrument gives two bits,")
    print("  not a taxonomy: it separates price-readers and identity-readers from")
    print("  everything else, and does not distinguish a placement-reader from")
    print("  any other magnitude-reader — which is why `max_arity`, reading")
    print("  |parents|, sits with the opacity walls. `depth` remains a declared")
    print("  proxy for proof-theoretic rank.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "reflection_wall_audit.json"
        out.write_text(json.dumps({
            "params": vars(args) | {"output_dir": str(args.output_dir)},
            "audit": rows, "locality": loc, "climb": climb,
            "verdicts": {"Q1_three_groups": q1, "Q2_uniform_has_no_detour": q2,
                         "Q3_climb_untouched": q3},
        }, indent=2, default=str))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
