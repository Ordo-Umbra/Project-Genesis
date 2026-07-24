"""Spin–statistics: the exchange sign of nematic ½-disclinations.

The spinor work (`nematic_spinor.py`, `n3_spinor.py`) measured the **spin** side
of a topological fermion — a ``±½`` disclination whose oriented director picks up
``−1`` under a ``2π`` self-rotation (``+1`` only after ``4π``), the ``SU(2)``
double cover.  The hadron work (`n3_hadron_spin.py`) measured the **fusion** side
— ``n`` half-integer constituents adding to spin ``s = n/2`` — and named, in its
own honest scope, the piece it did **not** reach: *"the statistics here are the
composite's far-field double-cover class, not quantum spin–statistics — no
anticommutation, no Pauli principle between identical composites; that frontier
stands."*

This module measures the missing piece — the **statistics** side, the exchange
sign — and ties it to the spin.  The spin–statistics *connection* is the
statement that **exchanging two identical spin-``s`` objects multiplies the
configuration by ``(−1)^{2s}``**, the *same* sign the object's own ``2π``
rotation produces: ``−1`` for a half-integer spinor (fermions, antisymmetric),
``+1`` for an integer one (bosons).  For classical order-parameter defects this
is the **Finkelstein–Rubinstein** construction: exchange is homotopic to a
``2π`` rotation of the pair's frame, so the two carry the same holonomy.

The measurement is a **braid**.  Two ``ψ``-vortices of charge ``q`` (each a
disclination of strength ``s = q/2``) sit symmetrically about a midpoint; the
pair is rotated rigidly about that midpoint by ``turns`` full turns
(``turns = ½`` swaps the two — one exchange; ``turns = 1`` is a full braid, two
exchanges).  Because a ``½``-turn advances each defect's relative angle by
``π``, the phase ``arg ψ`` at the braid **centre** winds by exactly ``q·π`` per
exchange, so the oriented director ``θ = ½ arg ψ`` there winds by ``q·π/2`` and
the exchange sign is

    (−1)^k ,   k = winding of arg ψ at the centre / 2π  =  q  (per exchange),

i.e. ``(−1)^q = (−1)^{2s}``.  Far from the pair the winding is ``0`` (the far
field sees only the total charge — that is *fusion*, ``½+½ = 1``); the exchange
sign lives in the pair's *internal* configuration, read at the centre.

- `braid_positions` — the moving ``(A, B)`` centres for a braid of ``turns``.
- `exchange_holonomy` — the centre-winding ``k`` and the sign ``(−1)^k`` over a
  braid (``turns = ½`` for a single exchange).
- `self_rotation_sign` — the single-defect ``2π`` self-rotation director
  holonomy (`nematic_spinor.director_holonomy`), the **spin** side the
  connection equates the exchange to.

Honest scope: this is the **topological / geometric** exchange sign of classical
disclinations (Finkelstein–Rubinstein) — the same level at which the programme
realises the spin (an order-parameter double cover), and the exact content of
the spin–statistics *connection* for these objects.  It is **not** quantum-field
anticommutation: no Fock space, no operator ``{ψ, ψ†}``, no many-body Pauli
principle between identical excitations.  The braid here is **kinematic** (the
imprint centres are moved through configuration space); a fully dynamical braid
(κ-pinned cores co-evolved under the CGL) is the natural next step and is
expected to agree, the winding being a topological invariant of the path.
"""

from __future__ import annotations

import numpy as np

from .vortex_chiral import imprint_vortices, winding_number
from .nematic_spinor import director_holonomy, plaquette_winding
from .two_field import chiral_detuning, step_chiral_detuned


def braid_positions(mid, half_sep: float, turns: float, arc_sign: float,
                    n_steps: int):
    """The moving ``(A, B)`` centre pairs for a braid of ``turns`` full turns.

    The pair sits at ``mid ± half_sep`` and is rotated rigidly about ``mid`` by
    ``arc_sign · 2π · turns``.  ``turns = 0.5`` swaps the two centres (one
    exchange); ``turns = 1`` returns them to their start having braided once
    (two exchanges).  Yields ``n_steps`` pairs.
    """
    for t in np.linspace(0.0, 1.0, n_steps):
        ang = arc_sign * 2.0 * np.pi * turns * t
        a = (mid[0] + half_sep * np.cos(np.pi + ang),
             mid[1] + half_sep * np.sin(np.pi + ang))
        b = (mid[0] + half_sep * np.cos(ang),
             mid[1] + half_sep * np.sin(ang))
        yield a, b


def exchange_holonomy(shape, mid, half_sep: float, charges, *,
                      turns: float = 0.5, arc_sign: float = 1.0,
                      core: float = 3.0, n_steps: int = 361,
                      probe=None) -> dict:
    """Centre-winding ``k`` and oriented-director exchange sign ``(−1)^k``.

    Rigidly braids the two ``charges`` about ``mid`` by ``turns`` full turns
    (``0.5`` = one exchange), accumulating the winding of ``arg ψ`` at ``probe``
    (the braid centre ``mid`` by default).  Returns ``{"k", "sign"}`` with
    ``sign = cos(π k)`` — the sign the oriented director ``θ = ½ arg ψ`` carries.
    """
    if probe is None:
        probe = mid
    n0, n1 = int(shape[0]), int(shape[1])
    pi, pj = int(round(probe[0])) % n0, int(round(probe[1])) % n1
    prev = None
    acc = 0.0
    for a, b in braid_positions(mid, half_sep, turns, arc_sign, n_steps):
        psi = imprint_vortices((n0, n1), [a, b], list(charges), core=core)
        v = float(np.angle(psi[pi, pj]))
        if prev is not None:
            acc += (v - prev + np.pi) % (2.0 * np.pi) - np.pi
        prev = v
    k = acc / (2.0 * np.pi)
    return {"k": float(k), "sign": float(np.cos(np.pi * k))}


def self_rotation_sign(shape, center, charge: int, *, core: float = 3.0,
                       radius: float = 12.0) -> float:
    """The single-defect ``2π`` self-rotation director holonomy — the spin side.

    ``−1`` for odd ``q`` (a half-integer disclination, ``s = q/2``), ``+1`` for
    even ``q`` (integer).  The value the spin–statistics connection equates the
    exchange sign to.
    """
    psi = imprint_vortices((int(shape[0]), int(shape[1])), [tuple(center)],
                           [int(charge)], core=core)
    h2, _ = director_holonomy(psi, tuple(center), radius=radius)
    return float(h2)


# ---------------------------------------------------------------------------
# The dynamical braid: co-evolved (not imprinted) defects on moving κ wells
# ---------------------------------------------------------------------------
#
# The exchange sign above is *kinematic*: the field is re-imprinted at every
# braid step, so the winding is imposed, not carried.  The dynamical question is
# whether a defect that co-evolves under the κ-detuned CGL — pinned to a κ well
# but never re-imprinted — keeps its winding as the well is moved, so that the
# braid, and its −1, are realised by the dynamics.  The honest finding (see
# `n3_dynamical_braid.py`): the κ well pins the defect's *amplitude*, not its
# phase winding, so transport is **adiabatic with a speed limit** — below a
# critical well speed the winding follows, above it the vacated core heals
# (unwinds) before the winding can migrate.  Single-defect transport is clean;
# a full two-body braid is at the edge of amplitude-only pinning.


def gaussian_wells(shape, centers, width: float, depth: float,
                   baseline: float = 1.0) -> np.ndarray:
    """A capacity field ``κ`` with Gaussian dips (holes) at ``centers``.

    ``κ = baseline·(1 − depth·Σ_k exp(−r_k²/2w²))`` (periodic distance).  A deep,
    moderately tight well pins a defect's core (an amplitude hole) with a steep
    enough gradient to drag it; too wide/shallow and the pin is too weak to
    transport the winding.
    """
    shape = tuple(int(s) for s in shape)
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    kappa = np.full(shape, float(baseline))
    for cx, cy in centers:
        dx = ((grids[0] - cx + shape[0] / 2) % shape[0]) - shape[0] / 2
        dy = ((grids[1] - cy + shape[1] / 2) % shape[1]) - shape[1] / 2
        kappa -= baseline * depth * np.exp(-(dx * dx + dy * dy)
                                           / (2.0 * width * width))
    return kappa


def transport_defect(shape, start, end, charge: int, *, width: float = 5.0,
                     depth: float = 0.9, detune_gamma: float = 0.85,
                     core: float = 3.0, dt: float = 0.08, speed: float = 0.05,
                     relax: int = 600, radius: int = 6) -> dict:
    """Drag one defect along a straight line by a moving κ well, co-evolving.

    Seeds a charge-``charge`` vortex pinned to a well at ``start``, relaxes, then
    steps the well to ``end`` at ``speed`` lattice-units per unit time while the
    field evolves only under `step_chiral_detuned` (no re-imprint).  Returns the
    enclosed winding at the start and end and whether it was conserved — the
    adiabatic-transport test.
    """
    shape = tuple(int(s) for s in shape)
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    dist = float(np.hypot(*(end - start)))
    unit = (end - start) / max(dist, 1e-9)
    # steps of the well between field updates, chosen so the well advances
    # `speed·(sub·dt)` per update; use a fixed small displacement per update
    delta = 0.25
    updates = max(1, int(dist / delta))
    sub = max(1, int(round(delta / (speed * dt))))

    kappa = gaussian_wells(shape, [tuple(start)], width, depth)
    g = chiral_detuning(kappa, detune_gamma=detune_gamma)
    psi = imprint_vortices(shape, [tuple(start)], [int(charge)], core=core,
                           detune=g)
    for _ in range(relax):
        psi = step_chiral_detuned(psi, g, dt=dt)
    w_start = winding_number(psi, tuple(start), radius)
    for i in range(1, updates + 1):
        pos = start + unit * (dist * i / updates)
        g = chiral_detuning(gaussian_wells(shape, [tuple(pos)], width, depth),
                            detune_gamma=detune_gamma)
        for _ in range(sub):
            psi = step_chiral_detuned(psi, g, dt=dt)
    w_end = winding_number(psi, tuple(end), radius)
    return {"w_start": float(w_start), "w_end": float(w_end),
            "conserved": bool(abs(abs(w_end) - abs(w_start)) < 0.5),
            "speed": float(speed), "sub": sub, "updates": updates}


def dynamical_braid(shape, mid, half_sep: float, charges, *, turns: float = 0.5,
                    width: float = 5.0, depth: float = 0.9,
                    detune_gamma: float = 0.85, core: float = 3.0,
                    dt: float = 0.08, delta: float = 0.3, sub: int = 90,
                    relax: int = 600, arc_sign: float = 1.0,
                    region: int = 40) -> dict:
    """Braid two co-evolved defects on moving κ wells (no re-imprint).

    Seeds the pair pinned to κ wells at ``mid ± half_sep``, relaxes, then rotates
    the two wells about ``mid`` by ``turns`` full turns while the field evolves
    under the κ-detuned CGL, accumulating the winding of ``arg ψ`` at the braid
    centre (the exchange-sign observable) and tracking core survival via the
    plaquette-winding defect count in the central ``region``.  Returns the centre
    winding ``k`` and sign, the survival trajectory, and the fraction of the
    braid over which both cores kept their winding.
    """
    shape = tuple(int(s) for s in shape)
    mid = (float(mid[0]), float(mid[1]))
    n_pos = max(1, int(turns * 2.0 * np.pi * half_sep / delta))
    n0, n1 = shape
    pc = (int(round(mid[0])) % n0, int(round(mid[1])) % n1)
    n_seed = abs(int(charges[0]))

    def wells(frac):
        ang = arc_sign * 2.0 * np.pi * turns * frac
        return [(mid[0] + half_sep * np.cos(np.pi + ang),
                 mid[1] + half_sep * np.sin(np.pi + ang)),
                (mid[0] + half_sep * np.cos(ang),
                 mid[1] + half_sep * np.sin(ang))]

    cs = wells(0.0)
    g = chiral_detuning(gaussian_wells(shape, cs, width, depth),
                        detune_gamma=detune_gamma)
    psi = imprint_vortices(shape, cs, list(charges), core=core, detune=g)
    for _ in range(relax):
        psi = step_chiral_detuned(psi, g, dt=dt)

    prev = float(np.angle(psi[pc]))
    acc = 0.0
    survival = []
    survived = 0
    for i in range(n_pos + 1):
        frac = i / n_pos
        g = chiral_detuning(gaussian_wells(shape, wells(frac), width, depth),
                            detune_gamma=detune_gamma)
        for _ in range(sub):
            psi = step_chiral_detuned(psi, g, dt=dt)
        v = float(np.angle(psi[pc]))
        acc += (v - prev + np.pi) % (2.0 * np.pi) - np.pi
        prev = v
        w = plaquette_winding(psi)
        cx, cy = int(mid[0]), int(mid[1])
        pos_cores = int((w[cx - region:cx + region,
                           cy - region:cy + region] > 0).sum())
        alive = pos_cores >= 2 * n_seed
        survived += int(alive)
        if i % max(1, n_pos // 10) == 0:
            survival.append(pos_cores)
    k = acc / (2.0 * np.pi)
    return {"k": float(k), "sign": float(np.cos(np.pi * k)),
            "survival": survival, "survived_fraction": survived / (n_pos + 1),
            "n_pos": n_pos, "steps": n_pos * sub}
