"""Where does the capacity-bound optimum cross over? — the scale-free form.

Three experiments (`n3_kappa_criticality`, `n3_criticality_transplant`,
`n3_kuramoto_transplant`) report the same qualitative result on three
substrates: sweep a disorder knob, apply ``S = ΔC + κ·w·ΔI`` with the engine's
capacity law ``κ = r·κ₀/(r + c·load)`` post-hoc, and as the consumption ``c``
rises the S-optimum **relocates** from the deep-ordered phase to the critical
neighbourhood.  All three report the relocation as a fact about ``c``: it
happens somewhere on a ladder ``c ∈ {0.05 … 50}``.

That reporting is not defensible, and this module is what is left after taking
the objection seriously.

**The objection.** ``load`` is defined differently on every substrate — domain
wall density on the lattice, spin flip rate on the Hopfield network,
co-rotating phase-increment spread on the oscillators — and at least one of
those definitions has **no intrinsic scale**.  The Kuramoto activity is a
*per-step* quantity, so it carries a factor of ``√dt`` from the integrator:
halving the time step halves the measured load and doubles the ``c`` at which
anything happens, with no change whatsoever to the physics.  A crossing
"at ``c ≈ 3``" is therefore a statement about the integrator, not the system.

**What survives.**  Two reductions rescue a real quantity:

1. ``r`` and ``c`` enter the capacity law *only through their ratio*.  Writing
   ``u = c/r`` gives ``κ = 1/(1 + u·load)``, so the recovery rate is not a
   second free parameter — it is the unit of the ``c`` axis.  (Both transplants
   fix ``r = 0.1``; that choice is invisible to every dimensionless statement
   below.)

2. The crossing condition, written in terms of the **capacity at the ordered
   point** rather than of ``c``, loses the load scale entirely.  Comparing the
   deep-ordered point ``o`` against the critical point ``⋆``:

       ΔC_o + w·κ_o·ΔI_o  =  ΔC_⋆ + w·κ_⋆·ΔI_⋆

   and because ``u·L_o = 1/κ_o − 1``, the critical point's capacity is fixed by
   ``κ_o`` and the **load ratio** ``ρ = L_⋆/L_o`` alone:

       κ_⋆ = 1 / (1 + ρ·(1/κ_o − 1))

   So the crossing sits at a ``κ_o⋆`` determined by five measured dimensionless
   numbers — ``ΔC_o, ΔC_⋆, ΔI_o, ΔI_⋆, ρ`` — plus the convention ``w``.  The
   absolute load scale has cancelled: it survives only in the conversion back
   to ``c⋆ = r·(1/κ_o⋆ − 1)/L_o``, which is exactly the step that reintroduces
   the arbitrary units.

**Three things this makes checkable that the ladder does not.**

- :func:`crossing_exists` — a **precondition** on the measured curves, computed
  before any consumption ladder is applied: a relocation is possible at all
  only if the ordered point starts ahead, ``w(ΔI_o − ΔI_⋆) > ΔC_⋆ − ΔC_o``.
- :func:`capacity_floor` — how far ``κ_o`` can actually be driven down over the
  ladder's range.  This is the mechanism's own statement of its condition: when
  the ordered phase is genuinely free (``L_o → 0``) the floor is 1, ``κ_o``
  never falls, and **no amount of scarcity can evict the ordered optimum**.
  "Scarcity evicts order exactly when maintaining order costs capacity" is that
  sentence, and here it is a number rather than a story.
- :func:`kappa_at_crossing` — the location, in the one variable that does not
  depend on how the substrate happens to measure its own load.

**What ``n3_crossing_prediction`` found using this, in short.**  The
precondition and the floor reproduce the published four-condition toggle from
the measured curves alone.  On the Hopfield network the relocation *is* the
two-point crossing, to 0.1%.  On the driven Kuramoto population it is not, and
the reason is :func:`optimum_walk`: with a flat load the optimum walks the
upper convex hull of the ``(ΔI, ΔC)`` cloud instead of crossing once, and that
hull has three vertices there against two on the Hopfield network.  Whether the
crossing sits at a substrate-independent ``κ`` is **not settled** — the
Kuramoto substrate's load turns out to be dominated by the injected drive
rather than by the system's response, so it cannot answer the question.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "capacity",
    "two_point_reduction",
    "crossing_exists",
    "capacity_floor",
    "kappa_partner",
    "kappa_at_crossing",
    "consumption_at_crossing",
    "measured_crossing",
    "optimum_walk",
]


def capacity(load, u: float) -> np.ndarray:
    """``κ = 1/(1 + u·load)`` with ``u = c/r``.

    Identical to ``steady_state_kappa(load, c, r)`` with ``κ₀ = 1``, written in
    the one combination that actually appears.  Keeping ``c`` and ``r`` separate
    invites reading them as two independent knobs; they are one.
    """
    return 1.0 / (1.0 + np.asarray(u, dtype=float)
                  * np.asarray(load, dtype=float))


def two_point_reduction(rows, *, x_key: str, ordered_index: int = 0) -> dict:
    """Reduce a measured disorder scan to the numbers the crossing depends on.

    ``rows`` is the per-disorder-value list either transplant experiment
    produces, each entry carrying ``delta_c``, ``delta_i``, ``activity`` and the
    control parameter under ``x_key`` (``"T"`` or ``"gamma"``).

    The two points are picked **from the curves alone**, with no reference to
    any consumption value: the *ordered* point is the low-disorder end of the
    scan, and the *critical* point is where ΔC peaks — which is the substrate's
    own definition of the critical neighbourhood in both transplants, and the
    thing the relocation is supposed to relocate to.

    Returns the five dimensionless quantities plus the two load readings (the
    loads are kept only because converting back to ``c`` needs ``L_o``; nothing
    dimensionless uses them except through their ratio ``rho``).
    """
    dc = np.array([r["delta_c"] for r in rows], dtype=float)
    di = np.array([r["delta_i"] for r in rows], dtype=float)
    load = np.array([r["activity"] for r in rows], dtype=float)
    x = np.array([r[x_key] for r in rows], dtype=float)

    o = int(ordered_index)
    s = int(np.argmax(dc))
    l_o, l_s = float(load[o]), float(load[s])
    return {"x_ordered": float(x[o]), "x_critical": float(x[s]),
            "i_ordered": o, "i_critical": s,
            "dc_o": float(dc[o]), "dc_s": float(dc[s]),
            "di_o": float(di[o]), "di_s": float(di[s]),
            "load_o": l_o, "load_s": l_s,
            "rho": float(l_s / l_o) if l_o > 0 else float("inf"),
            "gap": float(dc[s] - dc[o])}


def crossing_exists(red: dict, weight: float) -> bool:
    """Can the optimum start at the ordered point at all?

    At ``c = 0`` capacity is unity everywhere and ``S = ΔC + w·ΔI``.  The
    ordered point leads iff ``w(ΔI_o − ΔI_⋆) > ΔC_⋆ − ΔC_o``: its integration
    advantage must outweigh the critical point's distinction advantage.  If it
    does not, the optimum is *already* critical with capacity abundant, there is
    nothing to evict, and a "relocation" reported on such a scan is an artefact
    of the grid rather than the mechanism.

    Parameter-free apart from the shared convention ``w``, and computable before
    any consumption ladder exists.
    """
    return bool(weight * (red["di_o"] - red["di_s"]) > red["gap"])


def capacity_floor(red: dict, u_max: float) -> float:
    """Lowest ``κ_o`` the ladder can reach: ``1/(1 + u_max·L_o)``.

    The mechanism's condition, as a number.  A perfectly free ordered phase has
    ``L_o = 0``, a floor of exactly 1, and is **scarcity-proof at every ``c``**
    — not approximately, identically.  The transplants establish that by
    toggling a drive on and off; this is the same fact read directly off the
    measured load, without needing the ladder to be run.
    """
    return float(1.0 / (1.0 + float(u_max) * red["load_o"]))


def kappa_partner(kappa_o, rho: float) -> np.ndarray:
    """``κ_⋆`` given ``κ_o`` and the load ratio ``ρ = L_⋆/L_o``.

    Both points sit on the same capacity law at the same ``u``, so their
    capacities are not independent: ``u·L_o = 1/κ_o − 1`` fixes
    ``u·L_⋆ = ρ(1/κ_o − 1)``.  This is where the absolute load scale cancels.
    """
    k = np.asarray(kappa_o, dtype=float)
    if not np.isfinite(rho):
        return np.zeros_like(k)
    return 1.0 / (1.0 + float(rho) * (1.0 / np.maximum(k, 1e-15) - 1.0))


def kappa_at_crossing(red: dict, weight: float, *, grid: int = 20001) -> float:
    """``κ_o⋆``: the ordered point's capacity when the critical point overtakes it.

    Solves ``w(κ_o·ΔI_o − κ_⋆·ΔI_⋆) = ΔC_⋆ − ΔC_o`` for ``κ_o``, scanning down
    from ``κ_o = 1`` because that is the direction consumption drives it.

    The root is **unique** whenever the precondition holds, so the scan
    direction is a convention rather than a choice.  Writing
    ``F(κ) = w(κ·ΔI_o − κ_⋆(κ)·ΔI_⋆) − gap`` and
    ``κ_⋆(κ) = κ/(κ + ρ(1−κ))``, the derivative is
    ``F'(κ) = w(ΔI_o − ΔI_⋆·ρ/(κ + ρ(1−κ))²)``, whose bracket is monotone in
    ``κ`` — so ``F`` is concave for ``ρ > 1`` and convex for ``ρ < 1``.  Either
    way a single sign change is forced by ``F(0) = −gap < 0`` and
    ``F(1) = w(ΔI_o − ΔI_⋆) − gap > 0``, the latter being exactly
    :func:`crossing_exists`.

    Returns ``nan`` when no crossing exists (:func:`crossing_exists` false, or
    no sign change on the grid).

    A finite return value says *where* the crossing is, not that it is
    **reachable**: a free ordered phase (``L_o = 0``, ``ρ = ∞``) still yields a
    perfectly good ``κ_o⋆``, and it is simply never attained because ``κ_o``
    stays pinned at 1.  Always read this against :func:`capacity_floor`; the
    crossing happens iff ``floor < κ_o⋆``.
    """
    if not crossing_exists(red, weight):
        return float("nan")
    k = np.linspace(1.0, 1e-6, int(grid))
    f = weight * (k * red["di_o"] - kappa_partner(k, red["rho"]) * red["di_s"]) - red["gap"]
    sign_change = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) <= 0)[0]
    if sign_change.size == 0:
        return float("nan")
    i = int(sign_change[0])                       # first from κ_o = 1 downward
    a, b = k[i], k[i + 1]
    fa, fb = f[i], f[i + 1]
    if fa == fb:
        return float(a)
    return float(a + (b - a) * fa / (fa - fb))    # linear interpolation


def consumption_at_crossing(kappa_o: float, load_o: float, recovery: float) -> float:
    """``c⋆ = r·(1/κ_o⋆ − 1)/L_o`` — the step back into arbitrary units.

    Kept explicit rather than folded in, because this is exactly where the
    substrate's own load convention re-enters and the number stops being
    comparable across substrates.
    """
    if not np.isfinite(kappa_o) or load_o <= 0:
        return float("nan")
    return float(recovery * (1.0 / kappa_o - 1.0) / load_o)


def measured_crossing(rows, *, x_key: str, weight: float, recovery: float,
                      i_ordered: int = 0, c_lo: float = 1e-4,
                      c_hi: float = 1e5, n_c: int = 4001) -> dict:
    """Where the *full* argmax actually leaves the ordered point, on a dense ladder.

    The prediction above is a two-point comparison; this is the thing it has to
    match, and they are not the same computation — the true optimum ranges over
    every point on the scan and may leave the ordered point for somewhere that
    is neither endpoint, or move in several hops.  Both are recorded:

    - ``c_evict`` — the smallest ``c`` at which ``argmax_x S`` is no longer the
      ordered point;
    - ``c_arrive`` — the smallest ``c`` at which it is the ΔC-peak point;
    - ``kappa_evict``/``kappa_arrive`` — the same two moments in the scale-free
      variable;
    - ``hops`` — how many times the argmax index changes across the ladder (1 is
      a clean level crossing; more means the optimum walked).
    """
    dc = np.array([r["delta_c"] for r in rows], dtype=float)
    di = np.array([r["delta_i"] for r in rows], dtype=float)
    load = np.array([r["activity"] for r in rows], dtype=float)
    x = np.array([r[x_key] for r in rows], dtype=float)
    i_crit = int(np.argmax(dc))

    cs = np.geomspace(float(c_lo), float(c_hi), int(n_c))
    stars = []
    for c in cs:
        s = dc + capacity(load, c / recovery) * weight * di
        stars.append(int(np.argmax(s)))
    stars = np.asarray(stars)

    def _first(mask):
        idx = np.nonzero(mask)[0]
        return int(idx[0]) if idx.size else -1

    walk = [int(stars[0])] + [int(b) for a, b in zip(stars, stars[1:]) if a != b]
    j_ev = _first(stars != i_ordered)
    j_ar = _first(stars == i_crit) if i_crit != i_ordered else j_ev
    k_o = capacity(load[i_ordered], cs / recovery)
    hops = int(np.count_nonzero(np.diff(stars) != 0))
    return {"c_evict": float(cs[j_ev]) if j_ev >= 0 else float("nan"),
            "c_arrive": float(cs[j_ar]) if j_ar >= 0 else float("nan"),
            "kappa_evict": float(k_o[j_ev]) if j_ev >= 0 else float("nan"),
            "kappa_arrive": float(k_o[j_ar]) if j_ar >= 0 else float("nan"),
            "x_start": float(x[stars[0]]), "x_end": float(x[stars[-1]]),
            "x_critical": float(x[i_crit]), "hops": hops,
            "walk": [float(x[i]) for i in walk]}


def optimum_walk(rows) -> list:
    """Every point the optimum can ever occupy, for *any* uniform capacity.

    When the load is flat across the scan — every point paying the same —
    ``κ`` is a single number multiplying the whole ΔI curve, and
    ``argmax(ΔC + κ·w·ΔI)`` as ``κ`` falls from 1 to 0 traces the **upper
    convex hull** of the points in the ``(ΔI, ΔC)`` plane.  The optimum then
    does not cross once; it *walks*, stopping at each hull vertex in turn.

    That distinction is the difference between the two transplants.  With a
    strongly rising load (the Hopfield network, ``ρ ≈ 34``) the ordered point is
    taxed far harder than the critical one and the relocation is a single jump,
    which :func:`kappa_at_crossing` predicts exactly.  With a flat load (the
    driven Kuramoto population, ``ρ → 1``) it is a hull walk through
    intermediate points, and a two-point comparison necessarily misses — not
    because the algebra is wrong but because the relocation is a different
    event.

    Returns the hull vertices as ``(index, ΔI, ΔC)``, ordered from the
    high-integration end (the ``κ = 1`` optimum) to the high-distinction end
    (the ``κ = 0`` optimum).
    """
    di = np.array([r["delta_i"] for r in rows], dtype=float)
    dc = np.array([r["delta_c"] for r in rows], dtype=float)
    # most integrated first; ties broken by distinction, so the survivor of a
    # tie is the one that is not dominated
    order = sorted(range(len(di)), key=lambda i: (-di[i], -dc[i]))

    def lam(a: int, b: int) -> float:
        """Weight at which b overtakes a: ΔC_b + λΔI_b = ΔC_a + λΔI_a."""
        d = di[a] - di[b]
        return (dc[b] - dc[a]) / d if d > 0 else float("inf")

    hull: list[int] = []
    for i in order:
        if hull and dc[i] <= dc[hull[-1]]:
            continue                       # dominated: less of both, never optimal
        while len(hull) >= 2 and lam(hull[-1], i) >= lam(hull[-2], hull[-1]):
            hull.pop()                     # i overtakes hull[-1] before it ever leads
        hull.append(int(i))
    return [(int(i), float(di[i]), float(dc[i])) for i in hull]
