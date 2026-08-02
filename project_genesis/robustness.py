"""Was a robustness claim ever actually tested, or could it not have failed?

Six audits in this programme have reported the same split: sweep a convention,
and *locations* move while *comparisons* hold. Each time the comparison was
scored the same way — it stayed inside a tolerance, or the winner stayed the
winner — and each time that was read as evidence the comparison is the robust
kind of quantity.

`n3_gap_conventions` broke that reading. Sweeping the noise floor moves ``n_I``
and cannot move ``n_C`` at all, so the gap's spread came out equal to ``n_I``'s
to fourteen decimal places. The gap "survived" nothing; it inherited. A bar that
scores that as a pass is reporting a subtraction as a result.

The failure generalises past differences, which is why this is a module rather
than a helper in one experiment. Every robustness claim has the same shape —
*this quantity did not change under that convention* — and every one of them
admits the same two readings:

- the convention had real grip and the quantity resisted it (**a measurement**);
- the convention could not have changed the quantity given how little anything
  moved, or given that the quantity is a function of things it never touched
  (**an identity, wearing a measurement's clothes**).

Distinguishing them requires asking how close the claim came to breaking, in
units of how much the sweep actually moved things. That is the one statistic
here, in the three forms the programme's claims actually take:

:func:`cancellation`
    for a **difference of two co-measured quantities** — did both parts move, so
    that the difference had something to cancel? (§2's `n_C - n_I` gap.)

:func:`ordinal_headroom`
    for a **ranking** — did any rival come near enough to overtaking the winner
    that the convention could plausibly have flipped it? (§2/§3's `P = 3`.)

:func:`threshold_headroom`
    for a **boolean condition** with a continuous margin underneath it — did the
    margin ever approach the threshold? (§5's eviction condition, reported as
    holding in `16/16` arms without anyone asking by how much.)

Each reports a raw number first and a verdict second, because the verdict needs a
threshold and the threshold is exactly the sort of undeclared constant this
programme keeps finding underneath confident readings. The thresholds are
arguments with disclosed defaults, and no result should be quoted without the
number beside it.

A deliberate asymmetry in the verdicts: **not exercised is not the same as
false.** A claim that could not have failed here may still be true, and one of
the outcomes below — ``structural`` — is stronger than robustness rather than
weaker. What none of them is, is *evidence gathered by this sweep*.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "spread",
    "cancellation",
    "ordinal_headroom",
    "threshold_headroom",
]


def spread(values) -> float:
    """Full max-minus-min range of the finite entries; ``nan`` if there are none.

    Deliberately not a standard deviation. These sweeps have a handful of arms
    chosen for being defensible rather than for being sampled from anything, so
    a dispersion statistic would imply a population that does not exist. The
    honest summary of "the convention was moved over this set" is the range it
    produced.

    ``nan`` rather than ``0.0`` for an empty result, because zero reads as
    "perfectly stable" — the opposite of "nothing was measured".
    """
    v = [float(x) for x in values if np.isfinite(x)]
    return float(max(v) - min(v)) if v else float("nan")


def cancellation(part_a, part_b, floor: float = 0.02) -> dict:
    """Did a difference of two quantities have anything to cancel?

    ``part_a`` and ``part_b`` are the two co-measured quantities, one entry per
    convention setting, aligned. The difference ``a - b`` is the claim under
    test.

    The reasoning. If a convention enters both quantities the same way, it
    cancels out of their difference — that is the mechanism the programme has
    been leaning on. For the difference's stability to be *evidence* of that,
    both parts must actually respond to the convention. If only one moves,
    ``spread(a - b)`` equals that one's spread by arithmetic and the difference
    is not protected at all: it inherits the fragile part exactly.

    So the meaningful comparison is the difference's spread against the
    **smaller** of the two part-spreads, and it means nothing unless that
    smaller spread clears ``floor``. ``exercised`` records that precondition.

    ``floor`` is a judgement call about what counts as a part having moved at
    all, and it is reported back in the result so it travels with the number.
    """
    a = np.asarray(list(part_a), dtype=float)
    b = np.asarray(list(part_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError("the two parts must have one entry per setting")
    s_a, s_b, s_d = spread(a), spread(b), spread(a - b)
    lo, hi = min(s_a, s_b), max(s_a, s_b)
    exercised = bool(np.isfinite(lo) and lo > floor)
    return {
        "spread_a": s_a,
        "spread_b": s_b,
        "spread_difference": s_d,
        "smaller_part": lo,
        "larger_part": hi,
        "ratio_vs_smaller": float(s_d / lo) if lo > 0 else float("inf"),
        "ratio_vs_larger": float(s_d / hi) if hi > 0 else float("inf"),
        "exercised": exercised,
        "cancels": bool(exercised and s_d < lo),
        "floor": float(floor),
        "verdict": ("CANCELS" if exercised and s_d < lo else
                    "no cancellation" if exercised else
                    "NOT EXERCISED — a part barely moved"),
    }


def _lead(w: float, v: float, scale: str) -> float:
    """One setting's lead of the winner ``w`` over a rival ``v``."""
    if scale == "linear":
        return w - v
    if scale != "log":
        raise ValueError("scale must be 'linear' or 'log'")
    if v > 0 and w > 0:
        return float(np.log(w) - np.log(v))
    if v <= 0 and w > 0:
        return float("inf")     # infinitely ahead in ratio: the rival is absent
    if v > 0 and w <= 0:
        return float("-inf")
    return float("nan")         # both absent: no comparison exists


def ordinal_headroom(sweep, headroom_tol: float = 3.0,
                     scale: str = "linear") -> dict:
    """Could the convention plausibly have flipped this ranking?

    ``sweep`` is one mapping per convention setting, each from option label to
    its measured value — e.g. ``[{2: 0.0, 3: 0.41, 4: 0.02}, ...]``, one dict
    per radius. The claim under test is *the same option wins throughout*.

    For the winner to lose, some rival's **lead** ``winner - rival`` must reach
    zero. Each rival therefore has its own answer to "how close did this come":
    the smallest lead it ever left, measured in units of how much that lead
    moved across the sweep. The rival minimising that ratio is the one that came
    nearest, and its ``headroom`` is the statistic. Roughly, a headroom of ``k``
    says the sweep would have to move things about ``k`` times harder than it
    did to overturn the result.

    Four outcomes, and only one of them is a robustness measurement:

    ``FLIPS``
        the winner changes. The claim is refuted, which *is* information.
    ``EXERCISED``
        a rival came within ``headroom_tol`` leads of overtaking and did not.
        This is the only case where "the ranking is robust" was earned here.
    ``STRUCTURAL``
        some rival is identically zero at every setting. Then the ranking is
        categorical rather than robust — a different and generally *stronger*
        claim, but not one this sweep tested, and it should be stated as what
        it is. (§2's ``P = 3`` cliff is this case.)
    ``NOT EXERCISED``
        the lead dwarfs anything the convention did. The winner could not have
        lost, so its winning carries no information about the convention.

    ``scale`` decides what "close" means, and it is not a detail. On a linear
    scale the lead is ``winner - rival``; on a log scale it is
    ``log(winner/rival)``. **For a quantity spanning orders of magnitude — a
    density, a rate, anything non-negative — linear is the wrong choice and
    gives a badly misleading answer.** §2's junction densities are the worked
    example: at the tightest probe every rival is exactly zero while the winner
    is merely small, so the *difference* is at its smallest there and a linear
    reading calls that a near miss. In ratio terms the winner leads by infinity.
    The published analysis quotes a margin (a ratio) for exactly this reason, so
    a diagnostic scoring it must use the same geometry. Linear remains the
    default because it is the only safe choice for a quantity that can be
    negative.

    ``headroom_tol`` is the other free constant and it is a convention itself;
    both are returned in the result so they cannot be quoted away from the
    number.
    """
    rows = [{k: float(v) for k, v in dict(s).items()} for s in sweep]
    if not rows:
        return {"winner": None, "flips": False, "headroom": float("nan"),
                "exercised": False, "structural": False,
                "verdict": "empty sweep", "headroom_tol": float(headroom_tol),
                "scale": scale}

    keys = list(rows[0])
    if any(set(r) != set(keys) for r in rows):
        raise ValueError("every setting must score the same options")

    winners = [max(r, key=lambda k: r[k]) for r in rows]
    winner = winners[0]
    flips = len(set(map(str, winners))) > 1

    rivals = [k for k in keys if k != winner]
    if not rivals:
        return {"winner": winner, "flips": False, "headroom": float("inf"),
                "exercised": False, "structural": False,
                "verdict": "NOT EXERCISED — only one option",
                "headroom_tol": float(headroom_tol), "scale": scale}

    zero_rivals = [k for k in rivals if all(r[k] == 0.0 for r in rows)]

    best = None
    for k in rivals:
        leads = [_lead(r[winner], r[k], scale) for r in rows]
        usable = [x for x in leads if not np.isnan(x)]
        if not usable:
            continue
        s = spread(usable)
        lead = min(usable)
        if k in zero_rivals:
            # A rival pinned at zero contributes no closing force, and the
            # lead's apparent motion is entirely the winner's — the same
            # arithmetic that made the gap "survive" the noise floor. For a
            # non-negative quantity the lead cannot reach zero while the rival
            # sits at the bottom of the range, so the flip is unreachable and
            # the headroom is infinite regardless of how much the winner moves.
            head = float("inf")
        else:
            head = float(lead / s) if s > 0 else float("inf")
        if best is None or head < best["headroom"]:
            best = {"rival": k, "min_lead": float(lead), "lead_spread": s,
                    "headroom": head}

    if best is None:
        return {"winner": winner, "flips": flips, "structural": False,
                "zero_rivals": zero_rivals, "closest_rival": None,
                "min_lead": float("nan"), "lead_spread": float("nan"),
                "headroom": float("nan"), "exercised": False,
                "verdict": "no comparable rival at any setting",
                "headroom_tol": float(headroom_tol), "scale": scale}

    structural = best["rival"] in zero_rivals
    exercised = bool(np.isfinite(best["headroom"])
                     and best["headroom"] < headroom_tol)
    if flips:
        verdict = "FLIPS — the ranking is not stable"
    elif structural:
        verdict = "STRUCTURAL — every rival is identically zero; categorical, not robust"
    elif exercised:
        verdict = "EXERCISED — a rival came close and did not overtake"
    else:
        verdict = "NOT EXERCISED — the lead dwarfs the sweep's motion"

    return {"winner": winner, "flips": flips, "structural": structural,
            "zero_rivals": zero_rivals,
            "closest_rival": best["rival"], "min_lead": best["min_lead"],
            "lead_spread": best["lead_spread"], "headroom": best["headroom"],
            "exercised": exercised, "verdict": verdict,
            "headroom_tol": float(headroom_tol), "scale": scale}


def threshold_headroom(margins, threshold: float = 0.0,
                       headroom_tol: float = 3.0) -> dict:
    """Could the convention have flipped a **boolean** condition?

    ``margins`` is the continuous quantity behind a yes/no claim, one entry per
    convention setting; the claim holds where ``margin > threshold``. §5's
    eviction condition is the case this was written for: reported as holding in
    ``16/16`` arms, with a signed chord excess underneath it that nobody looked
    at.

    "Sixteen out of sixteen" is a count of settings, not a measure of how nearly
    any of them failed. If the margin sat ten times its own motion above the
    threshold in every arm, the condition could not have flipped and the count
    is a restatement of the setup rather than a result. The statistic is the
    same as :func:`ordinal_headroom`'s — closest approach in units of observed
    motion — because it is the same question.

    Unlike a rival pinned at zero, ``threshold`` here is a boundary the quantity
    is free to cross in either direction; the caller declares it, so no inference
    about reachability is made on the caller's behalf.
    """
    m = np.asarray(list(margins), dtype=float)
    # Drop only *missing* readings. An infinite margin is not missing data: it
    # is a setting where the condition failed as badly as it can (or held by an
    # unbounded amount), and filtering it out would delete the very arm that
    # decides the claim. That deletion is silent and looks exactly like a pass,
    # which is the failure mode this whole module exists to prevent.
    m = m[~np.isnan(m)]
    if m.size == 0:
        return {"holds": False, "min_margin": float("nan"),
                "motion": float("nan"), "headroom": float("nan"),
                "exercised": False, "verdict": "no usable margins",
                "threshold": float(threshold),
                "headroom_tol": float(headroom_tol)}
    lead = float(m.min() - threshold)
    motion = spread(m)
    holds = bool(lead > 0.0)
    head = float(lead / motion) if motion > 0 else float("inf")
    exercised = bool(np.isfinite(head) and abs(head) < headroom_tol)
    if not holds:
        verdict = "CROSSES — the condition fails somewhere in the sweep"
    elif exercised:
        verdict = "EXERCISED — came close to the threshold and held"
    else:
        verdict = "NOT EXERCISED — the margin dwarfs the sweep's motion"
    return {"holds": holds, "min_margin": lead, "motion": motion,
            "headroom": head, "exercised": exercised, "verdict": verdict,
            "threshold": float(threshold), "headroom_tol": float(headroom_tol)}
