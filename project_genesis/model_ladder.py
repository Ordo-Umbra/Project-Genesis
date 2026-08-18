"""A ladder whose ceiling is semantic: eliminate models until one remains.

`finite_ladder.py` produced a reachable ceiling, but it *stipulated* one — `k`
atoms, one added per productive step, exhausted at `k`. That is the simplest
object that saturates, and its tidiness is suspect precisely because the
saturation was hand-set rather than emergent.

This is the harder version. Fix `n` propositional variables, giving `2^n`
valuations. A theory is the set of valuations still consistent with it, and a
step *eliminates* one — the semantic content of adding an axiom. The ladder
cannot empty the set, because a theory with no models is inconsistent, so the
floor is `1` remaining model and the capacity is

    C = 2^n - 1 productive steps

which nobody chose. It falls out of the semantics of "an axiom rules something
out" plus "the theory must stay consistent".

What this can complicate
------------------------
In the box, every adequate scheme reached `C` in exactly `C` steps. Here that
need not hold: a scheme addresses *which* model to eliminate, and a scheme
whose addresses revisit models already gone will take more steps to get to the
same floor. So exhaustion can be naming-invariant in its **location** while
being naming-dependent in its **cost** — a distinction the box could not show,
because there the address and the thing added were the same object.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelTheory:
    """A propositional theory, held semantically as its surviving models."""

    variables: int
    alive: frozenset[int] = None            # type: ignore[assignment]
    kind: str = "indexed"
    width: int | None = None
    rung: int = 0
    seen: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if self.variables <= 0:
            raise ValueError("need at least one variable")
        if self.kind not in ("inline", "indexed", "truncated", "scattered",
                             "partial"):
            raise ValueError(f"unknown presentation kind {self.kind!r}")
        if self.kind == "truncated" and not self.width:
            raise ValueError("truncated presentations need a positive width")
        if self.alive is None:
            object.__setattr__(self, "alive",
                               frozenset(range(1 << self.variables)))

    @property
    def models(self) -> int:
        return 1 << self.variables

    @property
    def capacity(self) -> int:
        """`C` — eliminations available before consistency forbids more."""
        return self.models - 1

    @property
    def integration(self) -> int:
        """`I` — eliminations actually made."""
        return self.models - len(self.alive)

    @property
    def room_left(self) -> int:
        return len(self.alive) - 1

    def address(self) -> int:
        """The presentation's name for its own current state."""
        match self.kind:
            case "inline":
                # Addresses by content: the surviving set itself.
                return sum(1 << m for m in self.alive)
            case "indexed":
                return self.rung
            case "truncated":
                return self.rung % (1 << self.width)
            case "scattered" | "partial":
                # Both address by position and never repeat an address. They
                # differ only in where those addresses *point*, which is the
                # separation this pair exists to make.
                return self.rung
        raise ValueError(f"unknown presentation kind {self.kind!r}")

    def target(self) -> int:
        """Which model this step tries to eliminate.

        Three distinct ways to be wrong, kept apart:

        - `scattered` hashes its address into the model space. It reaches every
          model eventually, but revisits, so it pays coupon-collector cost:
          adequate naming that is nonetheless inefficient.
        - `partial` steps by two, so it can only ever address half the models.
          Its addresses never repeat and it still cannot exhaust — a *coverage*
          failure rather than a collision one.
        - the rest map their address straight through.
        """
        if self.kind == "scattered":
            h = (self.rung * 2654435761) ^ (self.rung >> 3) ^ 0x9E3779B9
            return (h * 2246822519 % (1 << 61)) % self.models
        if self.kind == "partial":
            return (self.rung * 2) % self.models
        return self.address() % self.models


@dataclass(frozen=True)
class ModelStep:
    rung: int
    address: int
    target: int
    address_is_new: bool
    target_was_alive: bool
    consistency_allows: bool
    productive: bool
    before: ModelTheory
    after: ModelTheory

    @property
    def blocked_by(self) -> str | None:
        if not self.consistency_allows:
            return "exhausted"
        if not self.target_was_alive:
            return "stagnant"
        if not self.address_is_new:
            return "stagnant"
        return None


def model_step(theory: ModelTheory) -> ModelStep:
    """Try to eliminate one model. Consistency is the binding semantic law."""
    address = theory.address()
    target = theory.target()
    address_is_new = address not in theory.seen
    target_was_alive = target in theory.alive
    consistency_allows = len(theory.alive) > 1
    productive = address_is_new and target_was_alive and consistency_allows
    alive = (theory.alive - {target}) if productive else theory.alive
    after = replace(theory, alive=alive, rung=theory.rung + 1,
                    seen=theory.seen | {address})
    return ModelStep(rung=theory.rung, address=address, target=target,
                     address_is_new=address_is_new,
                     target_was_alive=target_was_alive,
                     consistency_allows=consistency_allows,
                     productive=productive, before=theory, after=after)


def model_climb(theory: ModelTheory, rungs: int) -> dict:
    """Run the ladder; report where it lands and what it cost to get there."""
    current, productive, steps_to_floor = theory, 0, None
    reason = None
    for i in range(rungs):
        s = model_step(current)
        if s.productive:
            productive += 1
        elif reason is None and s.blocked_by == "exhausted":
            reason, steps_to_floor = "exhausted", i
        current = s.after
        if current.room_left == 0 and steps_to_floor is None:
            steps_to_floor = i + 1
    if reason is None:
        reason = "exhausted" if current.room_left == 0 else "stagnant"
    return {
        "kind": theory.kind, "width": theory.width,
        "variables": theory.variables,
        "capacity": theory.capacity,
        "integration": current.integration,
        "reached_ceiling": current.integration == theory.capacity,
        "productive_steps": productive,
        "steps_taken": rungs,
        "steps_to_floor": steps_to_floor,
        "efficiency": (theory.capacity / steps_to_floor
                       if steps_to_floor else None),
        "reason": reason,
    }


def schemes(width: int) -> tuple[tuple[str, int | None], ...]:
    return (("inline", None), ("indexed", None), ("scattered", None),
            ("partial", None), ("truncated", width))
