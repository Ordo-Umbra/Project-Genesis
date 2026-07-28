"""Expansion: does a growing box change what survives?

The standing objection to heat death as an *ending* is that it is derived for a
sealed box of fixed volume, and the universe is not that.  In this repo the
objection has teeth, because the box is not a metaphor — it is what kills runs.
Measured in 2-D with the corrected kernel: a 18² lattice monopolises by step
300, 32² and larger do not monopolise at all within 4000 steps.  Same physics,
same bath; the only thing that changed was how much room there was.

So make the room grow and see what happens.

**How expansion enters.**  For a relaxational (first-order) order parameter,
comoving coordinates give it for free.  With physical position ``r = a(t)·x``,

    ∇²_phys = a⁻² ∇²_com

so the Allen-Cahn equation on a fixed comoving lattice is

    ∂_t η_a = (D/a(t)²) ∇²_com η_a − ∂f/∂η_a.

Expansion is a **diluting Laplacian** and nothing else.  There is no Hubble
friction term: that belongs to second-order wave equations, and this dynamics
is first order in time.  Nothing else in the model has to change, which is what
makes the experiment clean — one scalar schedule, same kernel.

**Why survival alone is the wrong score.**  Push expansion hard and ``D/a²``
goes to zero, at which point neighbouring sites stop talking to each other.
The field does not freeze so much as **decouple**: every site relaxes into its
own well independently, sectors scatter at lattice scale, and no single sector
can take the lattice.  By a survival test that reads as a triumph.  It is not.
It is causal disconnection — the de Sitter horizon, arriving as loss of contact
rather than as a wall — and it destroys structure just as thoroughly as
monopoly does, in the opposite direction.

So the score here is the framework's own *Good seed* criterion, unchanged from
:mod:`project_genesis.selection`: **integration × churn**, what binds times
what is still moving.  Monopoly kills the first factor.  Decoupling kills it
too, by scattering the palette so finely that nothing binds.  If there is a
window where a growing box does better than a fixed one, that criterion is
what will find it.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "SCHEDULES",
    "scale_factor",
    "diffusion_schedule",
]


SCHEDULES = ("flat", "radiation", "matter", "desitter")


def scale_factor(kind: str, t_sim: float, rate: float) -> float:
    """``a(t)`` for a simulation time ``t_sim`` (steps × dt).

    ``rate`` is the inverse of the expansion timescale, so ``rate = 0`` is a
    static box for every schedule and larger means faster.

    - ``flat``       — ``a = 1``; the sealed box, and the control.
    - ``radiation``  — ``a ∝ t^{1/2}`` at late time.
    - ``matter``     — ``a ∝ t^{2/3}`` at late time.
    - ``desitter``   — ``a = e^{rate·t}``; constant Λ, accelerating.
    """
    if kind not in SCHEDULES:
        raise ValueError(f"unknown schedule {kind!r}; expected one of {SCHEDULES}")
    if rate <= 0.0 or kind == "flat":
        return 1.0
    x = float(rate) * float(t_sim)
    if kind == "radiation":
        return (1.0 + x) ** 0.5
    if kind == "matter":
        return (1.0 + x) ** (2.0 / 3.0)
    return math.exp(x)                       # desitter


def diffusion_schedule(kind: str, rate: float, dt: float, *, floor: float = 0.0):
    """A callable ``step -> diffusion`` implementing ``D(t) = 1/a(t)²``.

    ``floor`` clamps the effective diffusion from below.  It defaults to 0 —
    genuine decoupling is a real outcome and should be allowed to happen rather
    than being quietly prevented by a fudge.
    """
    def diffusion_of_step(step: int) -> float:
        a = scale_factor(kind, float(step) * float(dt), rate)
        return max(1.0 / (a * a), float(floor))

    return diffusion_of_step
