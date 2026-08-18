"""A reflection ladder in a box: what happens when the domain can actually run out.

Every result in `reflection.py` was measured where `I < C` is a *theorem* — the
Church-Kleene ceiling is never reached, so one logically possible blocker could
never occur:

    exhausted — no productive move exists because the domain has run out, I = C

That gap matters. The four-dimensional taxonomy (structural, affordable,
productive, certifiable) is complete for *non-saturating* domains and untested
for saturating ones, and climbing altitude does not help: `I < C` is a theorem in
second-order arithmetic, in set theory, and at every admissible ordinal too, so
raising `C` just re-derives the same guarantee at a new level.

This builds a domain that genuinely saturates.

The box
-------
Fix `k` atomic sentences. A theory is a subset of them, so there are at most `k`
things any ladder can ever add and `C = k` exactly. The ladder adds one atom per
productive step, and when all `k` are asserted there is nothing left — not
because of the naming scheme, not because of a budget, but because the domain
ends.

Two independent ways to stall
-----------------------------
The construction keeps them separable on purpose, because telling them apart is
the whole question:

- `exhausted` — `len(axioms) = atoms`. A fact about the domain.
- `stagnant` — the address repeats, so the step re-derives something already
  held. A fact about the naming scheme.

The discriminator is **naming-invariance**: change the presentation and a
stagnant ladder moves its stall point, while an exhausted one does not. That is
what makes `exhausted` a candidate fifth category rather than stagnation wearing
a different hat.

And note what follows when the box is small enough: a naming defect is only
*observable* if it binds before the domain does. That is measured here rather
than argued, and it is the reason the arithmetic setting could see the
`truncated` pathology at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class FiniteTheory:
    """A theory inside a finite box of `atoms` atomic sentences."""

    atoms: int
    axioms: frozenset[int] = frozenset()
    kind: str = "indexed"
    width: int | None = None
    rung: int = 0
    seen: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if self.atoms <= 0:
            raise ValueError("a box needs at least one atom")
        if self.kind not in ("inline", "indexed", "truncated", "searched"):
            raise ValueError(f"unknown presentation kind {self.kind!r}")
        if self.kind == "truncated" and not self.width:
            raise ValueError("truncated presentations need a positive width")

    # ------------------------------------------------------------ quantities

    @property
    def capacity(self) -> int:
        """`C` — and here it is a finite number the ladder can actually reach."""
        return self.atoms

    @property
    def integration(self) -> int:
        """`I` — how much of the box is used."""
        return len(self.axioms)

    @property
    def room_left(self) -> int:
        """`C - I`. In the arithmetic setting this was always infinite."""
        return self.atoms - len(self.axioms)

    # -------------------------------------------------------------- naming

    def index(self) -> int:
        """The address this presentation offers for itself.

        `inline` addresses by content, so its address stops changing exactly
        when its content does — which means it cannot distinguish "I ran out of
        room" from "my address repeated". The indexed schemes address by
        position and keep moving regardless.
        """
        match self.kind:
            case "inline":
                return sum(1 << a for a in self.axioms)
            case "indexed" | "searched":
                return (1 << self.atoms) + self.rung
            case "truncated":
                return (1 << self.atoms) + (self.rung % (1 << self.width))
        raise ValueError(f"unknown presentation kind {self.kind!r}")

    def limit_status(self) -> str:
        """`available` | `unknown`. The finite box has no structural wall — an
        address always exists — but a `searched` presentation still cannot
        certify its own, which keeps the epistemic case in play."""
        return "unknown" if self.kind == "searched" else "available"


@dataclass(frozen=True)
class FiniteStep:
    """One attempted step, with both stall causes recorded separately."""

    rung: int
    address: int
    address_is_new: bool
    domain_has_room: bool
    productive: bool
    before: FiniteTheory
    after: FiniteTheory

    @property
    def blocked_by(self) -> str | None:
        """Which condition failed. Domain first: if the box is full, that is
        the binding fact whatever the address did."""
        if not self.domain_has_room:
            return "exhausted"
        if not self.address_is_new:
            return "stagnant"
        return None


def finite_step(theory: FiniteTheory) -> FiniteStep:
    """Apply the continuation operator once inside the box."""
    address = theory.index()
    address_is_new = address not in theory.seen
    domain_has_room = theory.integration < theory.atoms
    productive = address_is_new and domain_has_room
    axioms = (theory.axioms | {theory.integration}) if productive else theory.axioms
    after = replace(theory, axioms=axioms, rung=theory.rung + 1,
                    seen=theory.seen | {address})
    return FiniteStep(rung=theory.rung, address=address,
                      address_is_new=address_is_new,
                      domain_has_room=domain_has_room, productive=productive,
                      before=theory, after=after)


def finite_climb(theory: FiniteTheory, rungs: int):
    """Run the ladder and report where it stalls and why.

    Returns `(steps, stall_rung, stall_reason)`. `stall_rung` is the rung of the
    first unproductive step — the ladder keeps *running* past it, exactly as the
    arithmetic stalled arm did, because neither exhaustion nor stagnation halts
    anything. They just stop it getting anywhere.
    """
    steps, current = [], theory
    stall_rung, stall_reason = None, None
    for _ in range(rungs):
        s = finite_step(current)
        steps.append(s)
        if not s.productive and stall_rung is None:
            stall_rung, stall_reason = s.rung, s.blocked_by
        current = s.after
    return steps, stall_rung, stall_reason


def stall_point(atoms: int, kind: str, width: int | None = None,
                rungs: int | None = None) -> tuple[int | None, str | None]:
    """Where a given box and naming scheme first stops producing."""
    horizon = rungs if rungs is not None else atoms * 4 + 8
    _, rung, reason = finite_climb(
        FiniteTheory(atoms=atoms, kind=kind, width=width), horizon)
    return rung, reason


def naming_schemes(width: int) -> tuple[tuple[str, int | None], ...]:
    """The presentations to compare. Naming-invariance of a stall point is the
    test that separates exhaustion from stagnation, so all four are run against
    every box."""
    return (("inline", None), ("indexed", None), ("searched", None),
            ("truncated", width))


def predict_finite_stall(theory: FiniteTheory) -> tuple[int | None, str | None]:
    """What the theory can work out about its own stall, from inside.

    Both quantities are countable from the presentation: the room left in the
    box, and the size of its own address space. So unlike the epistemic wall of
    the arithmetic setting, this one is fully visible in advance — which is the
    bookend worth checking rather than assuming.

    Uses no run. Compare against `stall_point` to get the lookahead.
    """
    domain_limit = theory.atoms
    if theory.kind == "truncated":
        naming_limit = 1 << theory.width
        if naming_limit < domain_limit:
            return naming_limit, "stagnant"
    if theory.kind == "inline":
        # Addresses by content, so its address space is exactly its content
        # space: it can never stall for naming reasons before the box is full.
        pass
    return domain_limit, "exhausted"
