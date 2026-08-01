"""What shapes the ``(ΔI, ΔC)`` cloud — and therefore where the optimum lands.

`n3_crossing_prediction` and `n3_kuramoto_repair` between them settled the load
side of the criticality transplants: the eviction *condition* is
substrate-independent once the oscillator load stops reading back its own
drive.  Both also found the same thing about the other half, and neither could
do anything about it.  The optimum's **route** is the upper convex hull of the
scan's points in the ``(ΔI, ΔC)`` plane, and the load only sets the timing of
each hop along it.  The Hopfield network's hull has two vertices — a single
jump onto the ΔC peak.  The driven oscillators' has three, so the optimum stops
first at a partially-synchronised state that is neither ordered nor critical.

That difference is not a convention and no choice of load metric reaches it.
This module is the instrument for asking what it *is*.

The whole question reduces to one scalar.  A third hull vertex exists exactly
when some point on the scan lies **above the chord** joining the ordered point
to the ΔC peak:

    E = max_x [ ΔC_x − chord(ΔI_x) ]      over points between the two ends

``E > 0`` means the substrate has a *viable intermediate state* — a
configuration holding substantial integration **and** near-peak distinction at
once, good enough to be the S-optimum for some range of capacity.  ``E ≤ 0``
means it does not, and the optimum crosses once.  :func:`chord_excess` computes
it, and it is a property of the measured curves alone: no ladder, no capacity,
no load convention.

The obvious candidate for what sets ``E`` is how **broad** the order–disorder
transition is.  If integration collapses exactly where distinction peaks there
is nothing in between; if integration decays gradually while distinction is
already rising, the states in between are precisely the ones that sit above the
chord.  :func:`transition_width` measures that directly, and the experiment
(`n3_anatomy.py`) is what tests whether the two are actually related — this
module supplies the readings and asserts nothing.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "chord_excess",
    "transition_width",
    "hull_size",
    "anatomy_reading",
]


def chord_excess(rows, *, ordered_index: int = 0) -> dict:
    """How far the best intermediate state rises above the ordered→peak chord.

    Positive means a viable intermediate exists and the optimum will walk
    through it; zero or negative means the relocation is a single crossing.
    The sign is exactly the condition for a third convex-hull vertex, so this
    is the hull question asked as a number that can be swept.

    ``excess`` is reported in raw ΔC units and ``excess_frac`` relative to the
    distinction gap the critical point offers, which is the scale that matters
    — an excess of 0.01 means something different against a gap of 0.03 than
    against 0.5.
    """
    di = np.array([r["delta_i"] for r in rows], dtype=float)
    dc = np.array([r["delta_c"] for r in rows], dtype=float)
    o = int(ordered_index)
    s = int(np.argmax(dc))
    gap = float(dc[s] - dc[o])

    span = di[o] - di[s]
    if span <= 0:
        return {"excess": float("nan"), "excess_frac": float("nan"),
                "index": -1, "gap": gap}

    # the chord from the ordered point to the DC peak, read at each point's DI
    chord = dc[o] + (dc[s] - dc[o]) * (di[o] - di) / span
    inside = (di < di[o]) & (di > di[s])
    if not inside.any():
        return {"excess": float("-inf"), "excess_frac": float("-inf"),
                "index": -1, "gap": gap}

    resid = np.where(inside, dc - chord, -np.inf)
    j = int(np.argmax(resid))
    e = float(resid[j])
    return {"excess": e,
            "excess_frac": float(e / gap) if gap > 0 else float("nan"),
            "index": j, "gap": gap}


def transition_width(rows, x_key: str, *, hi: float = 0.8,
                     lo: float = 0.2) -> dict:
    """How broad the order–disorder transition is, in the control parameter.

    The range of ``x`` over which the order parameter ΔI falls from ``hi`` to
    ``lo``, divided by the midpoint of that range.  Dividing makes it
    dimensionless and comparable across substrates whose control parameters are
    a temperature and a frequency spread — the raw widths are in different
    units and cannot be set beside each other.

    Both crossings are found by linear interpolation on the measured curve, so
    the reading is limited by the scan's resolution; the experiment reports the
    grid spacing alongside it.
    """
    x = np.array([r[x_key] for r in rows], dtype=float)
    di = np.array([r["delta_i"] for r in rows], dtype=float)

    def crossing(level: float) -> float:
        for a, b in zip(range(len(di) - 1), range(1, len(di))):
            if di[a] >= level >= di[b]:
                d = di[a] - di[b]
                f = (di[a] - level) / d if d > 0 else 0.0
                return float(x[a] + f * (x[b] - x[a]))
        return float("nan")

    x_hi, x_lo = crossing(hi), crossing(lo)
    mid = 0.5 * (x_hi + x_lo)
    width = x_lo - x_hi
    return {"x_hi": x_hi, "x_lo": x_lo, "width": float(width),
            "midpoint": float(mid),
            "relative_width": float(width / mid) if mid > 0 else float("nan")}


def hull_size(rows) -> int:
    """Number of vertices on the upper hull — the number of stops on the route."""
    from .crossing import optimum_walk
    return len(optimum_walk(rows))


def anatomy_reading(rows, x_key: str, *, ordered_index: int = 0) -> dict:
    """Every anatomy quantity for one measured scan, in one call."""
    e = chord_excess(rows, ordered_index=ordered_index)
    w = transition_width(rows, x_key)
    x = [r[x_key] for r in rows]
    return {"excess": e["excess"], "excess_frac": e["excess_frac"],
            "gap": e["gap"],
            "x_intermediate": float(x[e["index"]]) if e["index"] >= 0
            else float("nan"),
            "relative_width": w["relative_width"], "width": w["width"],
            "midpoint": w["midpoint"],
            "hull": hull_size(rows),
            "grid": float(np.mean(np.diff(np.asarray(x, dtype=float))))}
