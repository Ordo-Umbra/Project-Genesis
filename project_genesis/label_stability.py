"""Does the S-compass taxonomy survive its own deadband?

`trajectory_label` classifies a step in ``(ΔC, ΔI)`` as ``expanding``,
``consolidating``, ``diverging``, ``contracting`` or ``steady``. It is the one
piece of this programme that already points at agents: the same taxonomy is
read on LLM session traces in the sibling repo, and both criticality
transplants report that their substrate "turns ``diverging`` — the compass's
hallucination trajectory" over a band of the control parameter, then place the
capacity-starved optimum inside it.

That reading rests on a number nobody derived. ``trajectory_label`` takes a
``deadband``: a step smaller than it in *either* coordinate is treated as no
movement, so the label becomes ``steady``. Both transplants pass `0.01`. Change
it and the label sequence changes — the only question is how.

`n3_anatomy` and `n3_convention_manifold` established the shape of that
question for the distinction reading, and the answer there was worth having: the
freedom was real but it was *one knob*, and on one substrate the consequence did
not move at all. This module asks the same three-way question of the compass,
because a taxonomy used to flag hallucination onset in a running agent should
not be one whose flags depend on an undocumented constant.

The three outcomes, and what each would mean for using the compass:

- **invariant** — the band is the same across every defensible deadband, and the
  reading is safe to act on;
- **monotone** — the band shrinks or grows predictably, so the deadband is an
  explicit sensitivity dial and its value has to be reported alongside any
  flag;
- **scattered or knife-edge** — the band appears and disappears, or `0.01`
  happens to sit on a boundary, in which case the published `diverging` reading
  is an artefact and should not be used to flag anything.

:func:`plateau` is the one that matters for the third case: it measures how wide
a range of deadbands leaves the band unchanged *around a given value*, which is
the difference between a robust operating point and a lucky one.
"""

from __future__ import annotations

import numpy as np

from .hopfield_substrate import trajectory_label

__all__ = [
    "label_sequence",
    "band_extent",
    "invariant_fraction",
    "plateau",
]


def label_sequence(rows, deadband: float) -> list:
    """Label every consecutive step of a measured scan at one deadband."""
    out = []
    for a, b in zip(rows, rows[1:]):
        out.append(trajectory_label(b["delta_c"] - a["delta_c"],
                                    b["delta_i"] - a["delta_i"],
                                    deadband=float(deadband)))
    return out


def band_extent(labels, x, name: str = "diverging") -> dict:
    """Where a given label occurs, as an interval in the control parameter.

    Steps sit *between* scan points, so each is placed at the midpoint of the
    pair that produced it. Returns the interval spanned, how many steps carry
    the label, and whether it appears at all — a band that vanishes is the
    outcome that matters most, and reporting it as an empty interval rather than
    as ``nan`` keeps that visible.
    """
    x = np.asarray(x, dtype=float)
    mids = 0.5 * (x[:-1] + x[1:])
    hit = [m for m, lab in zip(mids, labels) if lab == name]
    if not hit:
        return {"present": False, "lo": float("nan"), "hi": float("nan"),
                "width": 0.0, "count": 0}
    return {"present": True, "lo": float(min(hit)), "hi": float(max(hit)),
            "width": float(max(hit) - min(hit)), "count": len(hit)}


def invariant_fraction(sequences) -> float:
    """Fraction of step positions whose label is the same at every deadband."""
    seqs = [list(s) for s in sequences]
    if not seqs:
        return float("nan")
    n = min(len(s) for s in seqs)
    same = sum(1 for i in range(n) if len({s[i] for s in seqs}) == 1)
    return same / n if n else float("nan")


def plateau(deadbands, bands, index: int) -> dict:
    """How far the deadband can move from ``deadbands[index]`` before the band does.

    Walks outward from the given setting while the band's interval is unchanged,
    and reports the surviving range as a multiplicative factor. A published
    value sitting on a boundary — factor 1.0 in one direction — means the
    reading is knife-edge and a reader who chose a slightly different constant
    would have seen something else.
    """
    d = np.asarray(deadbands, dtype=float)
    ref = bands[index]

    def same(b):
        if b["present"] != ref["present"]:
            return False
        if not b["present"]:
            return True
        return (abs(b["lo"] - ref["lo"]) < 1e-12
                and abs(b["hi"] - ref["hi"]) < 1e-12)

    lo = index
    while lo - 1 >= 0 and same(bands[lo - 1]):
        lo -= 1
    hi = index
    while hi + 1 < len(bands) and same(bands[hi + 1]):
        hi += 1
    return {"lo": float(d[lo]), "hi": float(d[hi]),
            "factor_down": float(d[index] / d[lo]) if d[lo] > 0 else float("inf"),
            "factor_up": float(d[hi] / d[index]) if d[index] > 0 else float("inf"),
            "n_settings": int(hi - lo + 1)}
