"""Has the split this programme keeps quoting ever actually been demonstrated?

Six audits have reported the same result: sweep a convention, and *locations*
move while *comparisons* hold. It has been quoted in §2, §3 and §5, and it is
the reason several claims in `The_Principle.md` are stated as comparisons in the
first place. It has never been checked.

`n3_gap_conventions` is why it needs checking. Its differential sweep moves
`n_I` and cannot move `n_C` at all, so the gap's spread came out *equal* to
`n_I`'s to fourteen decimals. Scored against the usual bar — "the comparison
stayed inside the tolerance" — that reads as a sixth confirmation. It is a
subtraction. The bar cannot tell the two apart, and the bar is what every one of
the six was scored against.

So this re-scores the earlier sweeps with instruments that can tell them apart.
There is only one statistic — *how close did the claim come to breaking, in
units of how much the sweep actually moved things* — in the three forms the
claims take (`project_genesis.robustness`):

======================  ==========================  ====================
claim                   shape                       diagnostic
======================  ==========================  ====================
§2 one-dimension gap    difference of two readings  ``cancellation``
§2/§3 ``P = 3`` cliff   a ranking                   ``ordinal_headroom``
§5 eviction condition   a boolean with a margin     ``threshold_headroom``
======================  ==========================  ====================

This reads the JSON the published experiments already write, rather than
recomputing anything, so it is scoring the same numbers the document quotes and
cannot drift from them.

Pre-registered predictions:

R1. **At least one of the three was never exercised.** The bar's blind spot is
    not specific to the gap; sweeps get designed to move the thing under study,
    and the comparison's stability is then read off for free. **Falsifier:** all
    three come back exercised, meaning the six confirmations were real
    measurements and the gap was an isolated accident.

R2. **The `P = 3` cliff is structural, not robust.** Its rivals are reported at
    *identically zero* at the published radius. If that holds across the sweep
    the ranking cannot flip, and "the ranking survives the convention" is a
    categorical fact about the geometry rather than evidence about the
    convention — a **stronger** claim, wrongly filed. **Falsifier:** a live
    rival with real headroom, which would make it an ordinary robustness result.

R3. **The eviction condition's `16/16` is a count, not a margin.** Its headroom
    exceeds the tolerance, so no arm came near failing. **Falsifier:** some arm
    approached the threshold, which would make `16/16` the strong evidence it
    has been quoted as.

What this cannot do. It re-scores sweeps that were already run; it does not
widen them. "Not exercised" here means *this sweep could not have refuted the
claim*, never *the claim is false* — and for R2 the honest reading would be that
the claim is stronger than advertised, not weaker. Distinguishing those is the
whole point, and conflating them would be a worse error than the one being
corrected.

Two conventions of its own, both disclosed with every number: ``headroom_tol``
(how near a claim must come to breaking before its survival counts as evidence)
and ``cancel_floor`` (how much a part must move to count as responsive). Neither
is derived. An audit of underived constants that hid its own would be worth
nothing.

*Scale, found the hard way.* ``ordinal_headroom`` measures a lead additively by
default, and on the junction densities that is wrong: at the tightest probe the
rivals are exactly zero and the winner is merely small, so the difference is
smallest there and reads as a near miss when the winner is in fact infinitely
ahead. The first run of this experiment reported the wrong rival for that
reason. Densities are scored on a log scale, matching the ratio the published
analysis quotes.

    python experiments/n3_robustness_retrospective.py --from RESULTS_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.robustness import (  # noqa: E402
    cancellation, ordinal_headroom, threshold_headroom,
)

BANNER = "=" * 78

# `crossing.measured_crossing`'s search ladder top. The `evicted` flag is
# `isfinite(c_evict)`, so this — not any declared budget — is what the published
# boolean actually tests against.
LADDER_CEILING = 1e5


def load(directory: Path, name: str):
    p = directory / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def gap_claim(data, floor):
    """§2 — the one-dimension gap, under each of its conventions."""
    out = {}
    for label, key in (("fitting window (common-mode)", "window"),
                       ("noise-floor band", "floor"),
                       ("noise-floor magnitude", "floor_magnitude_posthoc")):
        rows = data.get(key)
        if not rows:
            continue
        vals = list(rows.values())
        out[label] = cancellation([r["n_C"] for r in vals],
                                  [r["n_I_raw"] for r in vals], floor=floor)
    return out


def cliff_claim(data, tol):
    """§2/§3 — does P=3 keep winning as the probe widens?"""
    params = data.get("params", {})
    palettes = params.get("palettes", [2, 3, 4, 5, 6])
    radii = params.get("radii", [1, 2, 3])
    results = data.get("results", {})
    out = {}
    for dim in ("2-D", "3-D"):
        dens = results.get(f"{dim}_density")
        if not dens:
            continue
        sweep = []
        for r in radii:
            row = {P: dens.get(f"P{P}_r{r}") for P in palettes}
            if any(v is None for v in row.values()):
                sweep = []
                break
            sweep.append(row)
        if sweep:
            # log scale: these are densities spanning orders of magnitude, and
            # the published analysis quotes a *margin* (a ratio) for that
            # reason. On a linear scale the smallest lead lands at the tightest
            # probe — where the rivals are exactly zero and the winner is merely
            # small — and reads as a near miss when the winner is in fact
            # infinitely ahead.
            out[f"{dim} — argmax across radii {radii}"] = ordinal_headroom(
                sweep, headroom_tol=tol, scale="log")
    return out


def eviction_claim(data, tol):
    """§5 — the eviction condition, and the consumption underneath it.

    The condition is ``evicted``, and as coded that is ``isfinite(c_evict)`` —
    true whenever the argmax leaves the ordered point anywhere on a search
    ladder running to ``c = 1e5``. The *declared* budget is much smaller
    (``c_max``, 50 by default), so the meaningful margin is how far below the
    affordable ceiling the eviction actually lands:

        margin = log10(c_max / c_evict)

    in decades, because ``c`` is searched geometrically. An arm that never
    evicts gives ``-inf`` and fails, as it should.

    Not ``excess_frac``: that is the chord excess, which is negative in several
    arms *while they evict*, so scoring it against zero reports a failure that
    did not happen. The two are different claims and only one of them is what
    §5 quotes.
    """
    arms = data.get("arms")
    if not arms:
        return {}
    c_max = float(data.get("params", {}).get("c_max", 50.0))
    out = {}
    # Two ceilings, because the experiment uses two and says so nowhere. The
    # `evicted` flag is `isfinite(c_evict)`, which passes for any crossing found
    # on `measured_crossing`'s search ladder — a computational bound running to
    # c = 1e5. The experiment separately declares c_max as the scarcity it
    # considers, and uses it for the capacity floor. Scored against the ladder
    # everything passes trivially; scored against the declared budget the answer
    # is different. Both are shown rather than one being chosen here.
    for ceiling, label in ((c_max, f"declared budget c_max={c_max:g}"),
                           (LADDER_CEILING, f"search ladder c={LADDER_CEILING:g}")):
        for family in ("hopfield", "kuramoto"):
            rows = [v for k, v in arms.items() if k.startswith(family)]
            ce = [r.get("c_evict") for r in rows if r.get("c_evict") is not None]
            if len(ce) < 2:
                continue
            with np.errstate(divide="ignore"):
                margins = [float(np.log10(ceiling / c)) if c > 0
                           else float("inf") for c in ce]
            r = threshold_headroom(margins, threshold=0.0, headroom_tol=tol)
            # The ladder rows re-score the same claim against a different
            # ceiling. They are a sensitivity check, not extra claims, and
            # counting them in the tally would inflate it with presentation.
            r["sensitivity"] = ceiling != c_max
            out[f"{family} vs {label} — decades, {len(margins)} arms"] = r
    return out


def show_cancellation(name, c):
    print(f"  {name}")
    print(f"      part spreads {c['spread_a']:.4f} / {c['spread_b']:.4f}"
          f"   difference {c['spread_difference']:.4f}"
          f"   ratio {c['ratio_vs_smaller']:.2f}")
    print(f"      -> {c['verdict']}")


def show_headroom(name, h):
    if "winner" in h:
        extra = f"winner {h['winner']}, closest rival {h['closest_rival']}"
        detail = (f"min lead {h['min_lead']:.5f}, "
                  f"lead moved {h['lead_spread']:.5f}")
    else:
        extra = f"holds: {'yes' if h['holds'] else 'NO'}"
        detail = (f"min margin {h['min_margin']:.4f}, "
                  f"motion {h['motion']:.4f}")
    hd = h["headroom"]
    hdtxt = "inf" if np.isinf(hd) else f"{hd:.2f}"
    print(f"  {name}")
    print(f"      {extra}")
    print(f"      {detail}   headroom {hdtxt}x")
    print(f"      -> {h['verdict']}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--from", dest="src", type=Path, required=True,
                   help="directory holding the published experiments' JSON")
    p.add_argument("--headroom-tol", type=float, default=3.0,
                   help="a claim counts as tested if it came within this many "
                        "of its own motion of breaking")
    p.add_argument("--cancel-floor", type=float, default=0.02,
                   help="how much a part must move to count as responsive")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    print(BANNER)
    print("HAS THE ORDINAL SPLIT EVER ACTUALLY BEEN DEMONSTRATED?")
    print(BANNER)
    print(f"  re-scoring published sweeps from {args.src}")
    print(f"  a claim counts as TESTED if it came within {args.headroom_tol:g}x")
    print("  its own motion of breaking; both tolerances are conventions and")
    print("  are printed with every number below")
    print()
    print("  Pre-registered:")
    print("    R1  at least one of the three claims was never exercised")
    print("    R2  the P=3 cliff is structural, not robust")
    print("    R3  the eviction condition's 16/16 is a count, not a margin")
    print()

    gap = load(args.src, "gap_conventions.json")
    cliff = load(args.src, "junction_scale.json")
    evict = load(args.src, "anatomy.json")

    missing = [n for n, d in (("gap_conventions", gap), ("junction_scale", cliff),
                              ("anatomy", evict)) if d is None]
    if missing:
        print(f"  NOTE: no JSON for {', '.join(missing)} — those claims are")
        print("  reported as UNAVAILABLE below, not as passing.")
        print()

    results = {}

    print(BANNER)
    print("§2 — THE ONE-DIMENSION GAP  (a difference of two readings)")
    print(BANNER)
    g = gap_claim(gap, args.cancel_floor) if gap else {}
    if not g:
        print("  UNAVAILABLE")
    for name, c in g.items():
        show_cancellation(name, c)
    results["gap"] = g

    print()
    print(BANNER)
    print("§2/§3 — THE P=3 CLIFF  (a ranking)")
    print(BANNER)
    c = cliff_claim(cliff, args.headroom_tol) if cliff else {}
    if not c:
        print("  UNAVAILABLE")
    for name, h in c.items():
        show_headroom(name, h)
    results["cliff"] = c

    print()
    print(BANNER)
    print("§5 — THE EVICTION CONDITION  (a boolean with a margin)")
    print(BANNER)
    e = eviction_claim(evict, args.headroom_tol) if evict else {}
    if not e:
        print("  UNAVAILABLE")
    for name, h in e.items():
        show_headroom(name, h)
    results["eviction"] = e

    # ------------------------------------------------------------- verdicts
    every = [x for x in list(g.values()) + list(c.values()) + list(e.values())
             if not x.get("sensitivity")]
    tested = [x for x in every if x.get("exercised")]
    structural = [x for x in c.values() if x.get("structural")]

    print()
    print(BANNER)
    print("R1  WAS ANY OF THIS EVER A MEASUREMENT?")
    print(BANNER)
    print(f"  {len(tested)} of {len(every)} swept claims came close enough to "
          f"breaking to count as tested.")
    print()
    r1 = bool(every) and len(tested) < len(every)
    if r1:
        print("  PREDICTION HELD. The bar the six audits were scored against —")
        print("  'the comparison stayed inside the tolerance' — passes on")
        print("  sweeps that could not have moved the comparison at all. Those")
        print("  claims are not refuted; they are unevidenced by the sweeps")
        print("  cited for them, which is a different and more fixable problem.")
    elif not every:
        print("  NO DATA — nothing was re-scored, so nothing is concluded.")
    else:
        print("  PREDICTION FAILED — every claim was genuinely exercised, and")
        print("  the gap's inheritance was an isolated accident rather than a")
        print("  blind spot in the bar. The six confirmations stand as")
        print("  measurements.")

    print()
    print(BANNER)
    print("R2  IS THE P=3 CLIFF STRUCTURAL RATHER THAN ROBUST?")
    print(BANNER)
    r2 = bool(structural)
    if not c:
        print("  UNAVAILABLE — no junction sweep to score.")
        r2 = False
    elif r2:
        print("  PREDICTION HELD — and this one is an upgrade, not a demotion.")
        print("  The rivals are not trailing, they are absent: identically zero")
        print("  at every setting. The ranking cannot flip, so the sweep never")
        print("  tested it — but the reason it cannot flip is that the claim is")
        print("  CATEGORICAL. §2 should say the rivals are zero, which is a")
        print("  sharper statement than 'the ranking survives the convention'.")
    else:
        print("  PREDICTION FAILED — a live rival carries real headroom, so")
        print("  this is an ordinary robustness result after all, and the")
        print("  categorical reading would have been too strong.")

    print()
    print(BANNER)
    print("R3  IS THE EVICTION CONDITION'S 16/16 A MARGIN OR A COUNT?")
    print(BANNER)
    primary = {k: v for k, v in e.items() if not v.get("sensitivity")}
    crossed = {k: v for k, v in primary.items() if not v.get("holds")}
    r3 = bool(primary) and not any(x.get("exercised") for x in primary.values())
    if not e:
        print("  UNAVAILABLE — no anatomy sweep to score.")
    elif crossed:
        print("  PREDICTION FAILED, and in the direction that matters. Against")
        print("  the search ladder the condition holds everywhere, as published.")
        print("  Against the budget the experiment itself declares, it does not:")
        for k in crossed:
            print(f"    {k}")
        print("  The `evicted` flag is `isfinite(c_evict)`, which asks only")
        print("  whether a crossing exists somewhere on a ladder running to")
        print(f"  c = {LADDER_CEILING:g} — {LADDER_CEILING / 50:.0f}x past the "
              f"scarcity the run declares.")
        print("  So the two ceilings disagree, and nothing in the experiment")
        print("  says which one the claim is about. That is not a refutation of")
        print("  the eviction result; it is an undeclared constant deciding a")
        print("  published boolean, which is the same finding this arc keeps")
        print("  making, now on §5's most-quoted number.")
    elif r3:
        print("  PREDICTION HELD — no arm came near the threshold, so '16/16'")
        print("  counts settings rather than measuring closeness. The condition")
        print("  may well be robust; this sweep did not establish it, and the")
        print("  honest form of the claim quotes the margin, not the tally.")
    else:
        print("  PREDICTION FAILED — an arm did approach the threshold and")
        print("  held. Then 16/16 is the strong evidence it has been quoted as.")

    print()
    print(BANNER)
    n = int(r1) + int(r2) + int(r3)
    print(f"VERDICT: {n}/3 pre-registered predictions held.")
    print(BANNER)
    print("  The transferable point is about the bar, not any single claim.")
    print("  'It did not move' is not a result until you show the sweep could")
    print("  have moved it. Every robustness number in this repo should be")
    print("  quoted with its headroom, and the ones that cannot be are")
    print("  descriptions of the setup rather than findings.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        o = args.output_dir / "robustness_retrospective.json"
        o.write_text(json.dumps(
            {"params": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
             "unavailable": missing, "results": results,
             "score": n}, indent=2, default=float))
        print(f"\n  wrote {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
