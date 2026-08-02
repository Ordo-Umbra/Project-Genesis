"""At what scale does "three is special" hold — and is that scale forced?

§2 and §3 of `The_Principle.md` rest on one measurement:
:func:`~project_genesis.multiphase.full_palette_junction_density` is non-zero
essentially only at ``P = 3``, and stays that way across a 10x sweep in
capacity. Everything downstream — matter as a tessellation, generations as
``d+1``, the whole selection argument — is a consequence. It is also the least
audited claim in the document, and the four audits that preceded this module
all found the same thing: a confident reading resting on an underived constant.

So the first question is what the constant *is* here, and the answer is more
interesting than it was for ΔC. The measure has **no numeric threshold**. It
asks two exact questions of each cell — does the neighbourhood carry every
colour (``bits == full``), and is it a junction rather than a wall
(``distinct >= 3``) — and the ``3`` is the definition of a vertex, not a tuned
cut. The only free choice is the **neighbourhood**, hardcoded as the immediate
``3^d`` ring.

That choice has something ΔC's threshold never had: an argument. On a lattice
the ``3^d`` ring is the *smallest* window in which a junction can appear at all,
so it is the natural probe for vertex-scale structure — a wider window measures
a region instead. This module parameterises it so the argument can be tested
rather than asserted.

The prediction that follows mechanically: a radius-2 window is 5x5, which can
hold four or more colours **even where every vertex is strictly three-fold**.
If the selection is genuinely vertex geometry — Plateau's law doing the work,
as §3 now says — it should degrade as the radius grows, and the honest reading
of the result becomes *"P = 3 is selected relative to a vertex-scale probe"*
rather than the unqualified *"three is special"*.

What matters most is which part of the claim survives. Today's other audits
found a clean split: **locations** move under a convention change and
**ordinal comparisons** do not, because the compared quantities share the
contaminated term and shift together. :func:`selection_profile` reports both —
the margin (a magnitude, expected to be fragile) and the ranking (a comparison,
expected to hold) — so the split can be checked here rather than assumed.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from .multiphase import _popcount

__all__ = [
    "neighbourhood_offsets",
    "neighbourhood_label_bits",
    "full_palette_density",
    "valence_profile",
    "selection_profile",
]


def neighbourhood_offsets(ndim: int, radius: int = 1) -> list:
    """All non-zero offsets in the ``(2r+1)^ndim`` box neighbourhood.

    ``radius = 1`` reproduces `multiphase._neighbour_offsets` exactly — the
    published probe. Larger radii are strictly larger boxes, so the bit-set they
    collect is monotone in the radius: whatever a small window sees, a bigger
    one sees too.
    """
    r = int(radius)
    if r < 0:
        raise ValueError("radius must be non-negative")
    return [o for o in product(range(-r, r + 1), repeat=int(ndim)) if any(o)]


def neighbourhood_label_bits(labels: np.ndarray, radius: int = 1) -> np.ndarray:
    """Per-cell bit-set of the sector labels present within ``radius``.

    Generalises `multiphase._neighbourhood_label_bits`, which is this at
    ``radius = 1``; the equivalence is pinned in the tests rather than assumed,
    because the whole audit is worthless if the published case is not
    reproduced exactly.
    """
    bits = 1 << labels
    for off in neighbourhood_offsets(labels.ndim, radius):
        shifted = labels
        for axis, delta in enumerate(off):
            if delta:
                shifted = np.roll(shifted, -delta, axis=axis)
        bits = bits | (1 << shifted)
    return bits


def full_palette_density(labels: np.ndarray, n_palette: int,
                         radius: int = 1) -> float:
    """Density of cells whose ``radius`` neighbourhood carries the whole palette.

    Identical to `multiphase.full_palette_junction_density` at ``radius = 1``.
    The two conditions are unchanged and both remain exact: the neighbourhood
    must contain every colour, and it must be a junction (three or more distinct
    sectors) rather than a wall.
    """
    bits = neighbourhood_label_bits(labels, radius)
    full = (1 << int(n_palette)) - 1
    return float(np.mean((bits == full) & (_popcount(bits) >= 3)))


def valence_profile(labels: np.ndarray, radius: int = 1) -> dict:
    """How many distinct sectors the typical neighbourhood sees.

    The mechanism check behind the prediction: if a wider window degrades the
    selection because it simply *fits more colours*, that has to show up here as
    a rising mean valence. If the mean valence barely moves and the selection
    still degrades, something else is going on and the vertex-scale story is
    wrong.
    """
    d = _popcount(neighbourhood_label_bits(labels, radius))
    counts = np.bincount(d.ravel())
    return {"mean": float(d.mean()),
            "max": int(d.max()),
            "junction_fraction": float(np.mean(d >= 3)),
            "histogram": counts.tolist()}


def selection_profile(densities: dict) -> dict:
    """Split a palette sweep into the magnitude and the ordinal claim.

    ``densities`` maps palette size to the measured density. Returns:

    - ``argmax`` — which palette wins, the **ordinal** claim;
    - ``margin`` — winner divided by the best rival, the **magnitude** claim;
    - ``selects`` — whether anything is selected at all (a winner with a
      non-zero reading).

    Reported separately because the preceding audits found these behave
    differently under a convention change: the ranking tends to survive where
    the margin does not, since the compared quantities share whatever the
    convention distorts.
    """
    items = {int(k): float(v) for k, v in densities.items()}
    if not items:
        return {"argmax": None, "margin": float("nan"), "selects": False}
    best = max(items, key=lambda k: items[k])
    rivals = [v for k, v in items.items() if k != best]
    top = items[best]
    runner = max(rivals) if rivals else 0.0
    if top <= 0.0:
        margin = 0.0
    elif runner <= 0.0:
        margin = float("inf")
    else:
        margin = top / runner
    return {"argmax": best, "best_density": top, "runner_up": runner,
            "margin": float(margin), "selects": bool(top > 0.0)}
