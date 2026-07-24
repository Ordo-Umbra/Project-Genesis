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

from .vortex_chiral import imprint_vortices
from .nematic_spinor import director_holonomy


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
