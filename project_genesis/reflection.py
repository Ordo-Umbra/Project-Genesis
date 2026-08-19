"""The reflection ladder as an executable object: T_0 = PA, T_{n+1} = T_n + Con(T_n).

`The_Generative_Gap.md` §3 borrows a theorem from proof theory to name the gap
the field program keeps measuring: `I(F) < C(F) = ω₁^CK`, and the observation
that closing it by adding `Con(F)` only moves `I` to a new recursive ordinal,
so the ladder `F_{n+1} = F_n + Con(F_n)` climbs forever without saturating.
Until now that ladder was cited, not run. This module runs it.

What is genuinely computable here, and what is not
--------------------------------------------------
Constructing `Con(T_n)` is computable: it is a syntactic operation on a finite
presentation. Deciding whether `Con(T_n)` is *true*, or whether `T_n` proves
it, is not. This module does the first and never pretends to do the second.
`T_n ⊬ Con(T_n)` is imported from Gödel's second incompleteness theorem under
stated hypotheses (each `T_n` consistent, recursively axiomatised, extending
Robinson arithmetic); it is a discharged premise, not a measurement.

The declared abstraction boundary
---------------------------------
In a full arithmetisation, `Prf(e, p, c)` — "p codes a proof of the sentence
coded by c from the axiom set with index e" — is a Δ₀ formula of the language
of arithmetic, several thousand symbols of primitive-recursive bookkeeping.
Here it is a primitive ternary relation symbol whose arguments are honest
terms. Everything downstream of that symbol — the shape of `Con`, the growth
of the presentations, the ladder's step structure — is exact; the expansion of
`Prf` itself is not carried out. That boundary is drawn on purpose and it is
where any claim about absolute symbol counts stops being meaningful. What
survives it are *ratios between presentations*, which is what this module
measures, because the unexpanded `Prf` contributes the same constant to every
arm.

Presentations are the point
---------------------------
The axiom set of `T_n` is r.e., and an r.e. set has many indices. `Con(T_n)`
has to name one of them, and *which* one it names is a free choice that the
mathematics does not fix. Three choices are implemented:

- `inline` — the index is the Gödel number of the literal list of axioms.
  Honest, extensional, and self-inflating: each rung's `Con` must carry a
  numeral for the whole theory below it, so the presentation roughly doubles
  in symbols per rung.
- `indexed` — the index is `⟨code(base), n⟩`, a recursive index naming "PA
  plus the first n rungs of this ladder" without listing them. Flat: the
  pairing is dominated by the fixed base code, so the rung counter costs
  nothing measurable until `n` exceeds that code, and `O(log n)` after.
- `truncated` — `indexed` with the counter stored in a `width`-bit field. It
  is a *deliberate negative control*: after `2^width` rungs the index wraps,
  `Con(T_n)` is a sentence the theory already contains, and the ladder stops
  moving while the rung counter keeps climbing.

`inline` and `indexed` are not the same theory — they name different indices,
so their `Con` sentences are literally different sentences. They are the same
*construction* under two presentations, which is the comparison being drawn.

Numerals are counted in the binary (efficient) convention: a numeral for `v`
costs `v.bit_length()` symbols, the standard cost of building `v` from `0`,
`S` and doubling. Under unary numerals the `inline` arm's cost is not merely
geometric but iterated-exponential, which changes the size of the effect and
none of its direction.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, replace
from typing import Iterable, Iterator, Sequence

# --------------------------------------------------------------------- syntax
#
# Terms and formulas of the language of arithmetic {0, S, +, ×, =, <}, extended
# with the primitive proof predicate Prf described in the module docstring.
# Frozen dataclasses so that syntactic equality is structural equality and
# formulas can live in sets — the ladder's "is this axiom new?" test is exactly
# a set membership, so this has to be exact.


class Term:
    """Base class for terms. Subclasses are frozen and compare structurally."""


@dataclass(frozen=True)
class Var(Term):
    name: str


@dataclass(frozen=True)
class Zero(Term):
    pass


@dataclass(frozen=True)
class Num(Term):
    """A numeral for a (possibly astronomical) natural number.

    Stored as a Python int rather than as `S(S(...S(0)))` so that a numeral for
    a Gödel number is representable at all. Its *symbol cost* is its bit
    length, the binary-numeral convention declared in the module docstring.
    """

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("numerals denote naturals")


@dataclass(frozen=True)
class Succ(Term):
    arg: Term


@dataclass(frozen=True)
class Plus(Term):
    left: Term
    right: Term


@dataclass(frozen=True)
class Times(Term):
    left: Term
    right: Term


class Formula:
    """Base class for formulas. Frozen subclasses, structural equality."""


@dataclass(frozen=True)
class Eq(Formula):
    left: Term
    right: Term


@dataclass(frozen=True)
class Lt(Formula):
    left: Term
    right: Term


@dataclass(frozen=True)
class Prf(Formula):
    """`proof` codes a derivation of the sentence coded by `sentence` in the
    theory whose axiom set has index `theory`. Primitive here; see the module
    docstring for why, and for what that does and does not license."""

    theory: Term
    proof: Term
    sentence: Term


@dataclass(frozen=True)
class Not(Formula):
    arg: Formula


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Implies(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Forall(Formula):
    var: str
    body: Formula


@dataclass(frozen=True)
class Exists(Formula):
    var: str
    body: Formula


# ------------------------------------------------------------- symbol counting


def symbols(node: Term | Formula) -> int:
    """Number of syntactic symbols, numerals costed in binary (see docstring)."""
    match node:
        case Var() | Zero():
            return 1
        case Num(value=v):
            return max(1, v.bit_length())
        case Succ(arg=a) | Not(arg=a):
            return 1 + symbols(a)
        case Plus(left=x, right=y) | Times(left=x, right=y):
            return 1 + symbols(x) + symbols(y)
        case Eq(left=x, right=y) | Lt(left=x, right=y):
            return 1 + symbols(x) + symbols(y)
        case Prf(theory=t, proof=p, sentence=c):
            return 1 + symbols(t) + symbols(p) + symbols(c)
        case And(left=x, right=y) | Or(left=x, right=y) | Implies(left=x, right=y):
            return 1 + symbols(x) + symbols(y)
        case Forall(body=b) | Exists(body=b):
            return 2 + symbols(b)
    raise TypeError(f"unknown node {node!r}")


def total_symbols(nodes: Iterable[Term | Formula]) -> int:
    return sum(symbols(n) for n in nodes)


# ------------------------------------------------------------ Gödel numbering
#
# A canonical prefix-free serialisation to bytes, read back as an integer. This
# is injective and computable, which is all a Gödel numbering has to be; it is
# not the prime-power scheme of the textbooks, because that one is not
# runnable. Injectivity is what the ladder depends on (distinct theories must
# get distinct indices) and it is asserted in the tests, not assumed.

_TAGS: dict[type, int] = {
    Var: 1, Zero: 2, Num: 3, Succ: 4, Plus: 5, Times: 6,
    Eq: 10, Lt: 11, Prf: 12, Not: 13, And: 14, Or: 15, Implies: 16,
    Forall: 17, Exists: 18,
}


def _varint(n: int) -> bytes:
    """Length-prefixed big-endian encoding; self-delimiting, so concatenation
    of encoded children stays uniquely decodable."""
    body = n.to_bytes((n.bit_length() + 7) // 8 or 1, "big")
    return len(body).to_bytes(4, "big") + body


def serialize(node: Term | Formula) -> bytes:
    """Canonical prefix-free byte encoding of a term or formula."""
    tag = _TAGS[type(node)].to_bytes(1, "big")
    match node:
        case Var(name=nm):
            raw = nm.encode()
            return tag + _varint(len(raw)) + raw
        case Zero():
            return tag
        case Num(value=v):
            return tag + _varint(v)
        case Succ(arg=a) | Not(arg=a):
            return tag + serialize(a)
        case Plus(left=x, right=y) | Times(left=x, right=y):
            return tag + serialize(x) + serialize(y)
        case Eq(left=x, right=y) | Lt(left=x, right=y):
            return tag + serialize(x) + serialize(y)
        case Prf(theory=t, proof=p, sentence=c):
            return tag + serialize(t) + serialize(p) + serialize(c)
        case And(left=x, right=y) | Or(left=x, right=y) | Implies(left=x, right=y):
            return tag + serialize(x) + serialize(y)
        case Forall(var=v, body=b) | Exists(var=v, body=b):
            raw = v.encode()
            return tag + _varint(len(raw)) + raw + serialize(b)
    raise TypeError(f"unknown node {node!r}")


def serialize_all(nodes: Sequence[Term | Formula]) -> bytes:
    """Encoding of a finite sequence: a count, then the members."""
    return _varint(len(nodes)) + b"".join(serialize(n) for n in nodes)


def godel_number(node: Term | Formula | Sequence[Term | Formula]) -> int:
    """Injective computable code. A leading sentinel byte keeps the map
    injective across encodings whose first byte would otherwise be dropped by
    integer conversion."""
    raw = serialize_all(node) if isinstance(node, (list, tuple)) else serialize(node)
    return int.from_bytes(b"\x01" + raw, "big")


def code_bits(node: Term | Formula | Sequence[Term | Formula]) -> int:
    return godel_number(node).bit_length()


def pair(a: int, b: int) -> int:
    """Cantor pairing — injective ℕ² → ℕ, used to build recursive indices."""
    s = a + b
    return s * (s + 1) // 2 + b


def _code_tuple(values: Sequence[int]) -> int:
    """Injective code for a finite sequence — length first, then fold."""
    code = len(values)
    for v in values:
        code = pair(code, v)
    return code


# ------------------------------------------------------------------ variables


def free_vars(node: Term | Formula) -> frozenset[str]:
    match node:
        case Var(name=nm):
            return frozenset({nm})
        case Zero() | Num():
            return frozenset()
        case Succ(arg=a) | Not(arg=a):
            return free_vars(a)
        case Plus(left=x, right=y) | Times(left=x, right=y):
            return free_vars(x) | free_vars(y)
        case Eq(left=x, right=y) | Lt(left=x, right=y):
            return free_vars(x) | free_vars(y)
        case Prf(theory=t, proof=p, sentence=c):
            return free_vars(t) | free_vars(p) | free_vars(c)
        case And(left=x, right=y) | Or(left=x, right=y) | Implies(left=x, right=y):
            return free_vars(x) | free_vars(y)
        case Forall(var=v, body=b) | Exists(var=v, body=b):
            return free_vars(b) - {v}
    raise TypeError(f"unknown node {node!r}")


def _fresh(taken: frozenset[str], hint: str) -> str:
    i, name = 0, hint
    while name in taken:
        i += 1
        name = f"{hint}{i}"
    return name


def substitute(node, var: str, term: Term):
    """Capture-avoiding substitution of `term` for free occurrences of `var`."""
    match node:
        case Var(name=nm):
            return term if nm == var else node
        case Zero() | Num():
            return node
        case Succ(arg=a):
            return Succ(substitute(a, var, term))
        case Plus(left=x, right=y):
            return Plus(substitute(x, var, term), substitute(y, var, term))
        case Times(left=x, right=y):
            return Times(substitute(x, var, term), substitute(y, var, term))
        case Eq(left=x, right=y):
            return Eq(substitute(x, var, term), substitute(y, var, term))
        case Lt(left=x, right=y):
            return Lt(substitute(x, var, term), substitute(y, var, term))
        case Prf(theory=t, proof=p, sentence=c):
            return Prf(substitute(t, var, term), substitute(p, var, term),
                       substitute(c, var, term))
        case Not(arg=a):
            return Not(substitute(a, var, term))
        case And(left=x, right=y):
            return And(substitute(x, var, term), substitute(y, var, term))
        case Or(left=x, right=y):
            return Or(substitute(x, var, term), substitute(y, var, term))
        case Implies(left=x, right=y):
            return Implies(substitute(x, var, term), substitute(y, var, term))
        case Forall(var=v, body=b) | Exists(var=v, body=b):
            ctor = Forall if isinstance(node, Forall) else Exists
            if v == var:
                return node
            if v in free_vars(term):
                fresh = _fresh(free_vars(term) | free_vars(b) | {var}, v)
                b = substitute(b, v, Var(fresh))
                v = fresh
            return ctor(v, substitute(b, var, term))
    raise TypeError(f"unknown node {node!r}")


# ------------------------------------------------------------------ base: PA

_X, _Y, _P = Var("x"), Var("y"), Var("p")


def pa_base_axioms() -> tuple[Formula, ...]:
    """The finitely many non-schematic axioms of first-order Peano arithmetic."""
    return (
        Forall("x", Not(Eq(Succ(_X), Zero()))),
        Forall("x", Forall("y", Implies(Eq(Succ(_X), Succ(_Y)), Eq(_X, _Y)))),
        Forall("x", Eq(Plus(_X, Zero()), _X)),
        Forall("x", Forall("y", Eq(Plus(_X, Succ(_Y)), Succ(Plus(_X, _Y))))),
        Forall("x", Eq(Times(_X, Zero()), Zero())),
        Forall("x", Forall("y",
                           Eq(Times(_X, Succ(_Y)), Plus(Times(_X, _Y), _X)))),
        Forall("x", Not(Lt(_X, Zero()))),
        Forall("x", Forall("y",
                           Implies(Lt(_X, Succ(_Y)), Or(Lt(_X, _Y), Eq(_X, _Y))))),
    )


def induction_instance(phi: Formula, var: str) -> Formula:
    """`(φ(0) ∧ ∀x(φ(x) → φ(Sx))) → ∀x φ(x)`."""
    return Implies(
        And(substitute(phi, var, Zero()),
            Forall(var, Implies(phi, substitute(phi, var, Succ(Var(var)))))),
        Forall(var, phi),
    )


def is_induction_instance(f: Formula) -> bool:
    """Structural check: destructure, then reconstruct and compare."""
    match f:
        case Implies(left=And(left=base, right=Forall(var=v, body=Implies(
                left=hyp, right=step))), right=Forall(var=w, body=phi)):
            if v != w or hyp != phi:
                return False
            return (base == substitute(phi, v, Zero())
                    and step == substitute(phi, v, Succ(Var(v))))
    return False


#: The canonical falsity whose unprovability `Con` asserts: `0 = S0`.
FALSITY: Formula = Eq(Zero(), Succ(Zero()))
FALSITY_CODE: int = godel_number(FALSITY)


# ------------------------------------------------------------------- theories


def _normalise(levels: dict[int, int]) -> tuple[int, ...]:
    """Coefficients in Cantor normal form, highest exponent first, no leading
    zeros. `{1: 2, 0: 3}` (= ω·2+3) becomes `(2, 3)`."""
    if any(c < 0 for c in levels.values()):
        raise ValueError("ordinal coefficients are natural numbers")
    top = max((e for e, c in levels.items() if c), default=-1)
    return tuple(levels.get(e, 0) for e in range(top, -1, -1))


class Rank:
    """An ordinal below `ω^ω`, in Cantor normal form.

    `ω^k·a_k + … + ω·a_1 + a_0`, stored as the coefficient tuple
    `(a_k, …, a_1, a_0)` with the highest exponent first. This is a genuine
    ordinal notation system — unique representations, decidable comparison,
    computable canonical fundamental sequences — and it is *not* Kleene's `O`.

    The distinction matters and is the subject of `reflection_omega_squared.py`.
    Reaching `ω²`, or `ω^ω`, or even `ε₀`, does not require `O`: those all have
    decidable notation systems. `O` is needed only for *all* recursive ordinals,
    and there notation validity is Π¹₁-complete — so the price of `O` is not
    implementation effort, it is the decidability of the accessibility relation
    itself. A system using it cannot in general determine which continuations
    are open to it.
    """

    __slots__ = ("coeffs",)

    def __init__(self, limits: int = 0, successors: int = 0) -> None:
        object.__setattr__(self, "coeffs", _normalise({1: limits,
                                                       0: successors}))

    @classmethod
    def from_levels(cls, levels: dict[int, int]) -> "Rank":
        r = cls.__new__(cls)
        object.__setattr__(r, "coeffs", _normalise(levels))
        return r

    @classmethod
    def from_coeffs(cls, coeffs: Sequence[int]) -> "Rank":
        top = len(coeffs) - 1
        return cls.from_levels({top - i: c for i, c in enumerate(coeffs)})

    def coefficient(self, exponent: int) -> int:
        i = len(self.coeffs) - 1 - exponent
        return self.coeffs[i] if 0 <= i < len(self.coeffs) else 0

    @property
    def limits(self) -> int:
        """Coefficient of `ω`, kept so the ω² fragment reads as it did."""
        return self.coefficient(1)

    @property
    def successors(self) -> int:
        return self.coefficient(0)

    @property
    def degree(self) -> int:
        """Highest exponent carrying a nonzero coefficient; -1 for zero."""
        return len(self.coeffs) - 1 if self.coeffs else -1

    @property
    def is_limit(self) -> bool:
        return bool(self.coeffs) and self.successors == 0

    def successor(self) -> "Rank":
        return Rank.from_levels(self._levels() | {0: self.successors + 1})

    def limit(self, level: int = 1) -> "Rank":
        """The next multiple of `ω^level`: bump that coefficient, zero below.

        `level=1` is the ordinary limit of a ladder of successors; `level=2` is
        the limit of a ladder of *those* — `ω, ω·2, ω·3, … → ω²`. The mechanism
        is the same shape at every level, which is the point.
        """
        if level < 1:
            raise ValueError("a limit is taken at level >= 1")
        levels = {e: c for e, c in self._levels().items() if e > level}
        levels[level] = self.coefficient(level) + 1
        return Rank.from_levels(levels)

    def _levels(self) -> dict[int, int]:
        return {self.degree - i: c for i, c in enumerate(self.coeffs)}

    def _key(self) -> tuple[int, tuple[int, ...]]:
        return (len(self.coeffs), self.coeffs)

    def __eq__(self, other) -> bool:
        return isinstance(other, Rank) and self.coeffs == other.coeffs

    def __lt__(self, other: "Rank") -> bool:
        return self._key() < other._key()

    def __le__(self, other: "Rank") -> bool:
        return self._key() <= other._key()

    def __gt__(self, other: "Rank") -> bool:
        return self._key() > other._key()

    def __ge__(self, other: "Rank") -> bool:
        return self._key() >= other._key()

    def __hash__(self) -> int:
        return hash(self.coeffs)

    def __repr__(self) -> str:
        return f"Rank.from_coeffs({self.coeffs!r})"

    def __str__(self) -> str:
        if not self.coeffs:
            return "0"
        parts = []
        for exponent, c in sorted(self._levels().items(), reverse=True):
            if not c:
                continue
            if exponent == 0:
                parts.append(str(c))
                continue
            base = "ω" if exponent == 1 else f"ω^{exponent}"
            parts.append(base if c == 1 else f"{base}·{c}")
        return "+".join(parts)


class LimitUndefined(Exception):
    """Raised when a presentation cannot name the union it is asked to take.

    This is not a budget failure. A presentation whose index is the Gödel
    number of a literal axiom list has no index to offer for an infinite union,
    at any price — so the limit edge is absent from the accessibility relation
    rather than merely expensive.
    """


@functools.lru_cache(maxsize=None)
def _base_code(base: tuple[Formula, ...], schemas: tuple[str, ...]) -> int:
    """Gödel number of the fixed part of a presentation.

    Memoised because it is constant along a ladder and re-serialising PA at
    every rung dominates the runtime of the capacity scans, which need
    thousands of rungs to resolve a threshold.
    """
    return godel_number(list(base) + [Var(s) for s in schemas])


@dataclass(frozen=True)
class Theory:
    """A finite presentation of a recursively axiomatised theory.

    `rungs` holds the consistency sentences added so far, in order. It is the
    *axiom set* that matters for the ladder, so `rungs` is kept duplicate-free:
    a step that would re-add a sentence already present leaves it unchanged,
    which is exactly the stall the `truncated` arm is built to exhibit.
    """

    kind: str
    rung: int
    base: tuple[Formula, ...]
    schemas: tuple[str, ...]
    rungs: tuple[Formula, ...]
    width: int | None = None
    #: Limits taken so far. With `rung` counting successors since the last
    #: limit, `(limits, rung)` is the rank `ω·limits + rung`.
    limits: int = 0
    #: Coefficients above `ω`, highest exponent first — the part of the rank
    #: that the ω² fragment could not express. `(1,)` is `ω²`, `(2, 3)` is
    #: `ω³·2 + ω²·3`. Empty for any rank below `ω²`, so every result recorded
    #: before this field existed is unchanged.
    higher: tuple[int, ...] = ()
    #: Indices already reflected on. `Con(T)` is a pure function of `T`'s
    #: index, so "have we named this index before" decides whether a rung can
    #: add anything — and decides it in O(1), where scanning the rung formulas
    #: is O(n) in comparisons over astronomical numerals. The two agree on any
    #: ladder, which `test_new_axiom_agrees_with_formula_membership` asserts
    #: directly rather than leaving to the reader.
    seen: frozenset[int] = frozenset()

    @property
    def rank(self) -> Rank:
        levels = {0: self.rung, 1: self.limits}
        top = len(self.higher) + 1
        for i, c in enumerate(self.higher):
            levels[top - i] = c
        return Rank.from_levels(levels)

    @property
    def name(self) -> str:
        return "PA" if not (self.rung or self.limits) else f"T_{self.rank}"

    def axioms(self) -> tuple[Formula, ...]:
        """The explicit axioms — base plus rungs. The induction schema is
        carried in `schemas` and instantiated on demand, which is what keeps
        the presentation finite."""
        return self.base + self.rungs

    def index(self) -> int:
        """The natural number this presentation offers as its own index — the
        thing `Con` has to name. This is where the three arms diverge."""
        base_code = _base_code(self.base, self.schemas)
        match self.kind:
            case "inline":
                return godel_number(list(self.axioms())
                                    + [Var(s) for s in self.schemas])
            case "indexed" | "searched":
                return pair(base_code, _code_tuple(self.rank.coeffs))
            case "truncated":
                if self.width is None:
                    raise ValueError("truncated presentations need a width")
                # Only the successor counter is truncated; the limit levels are
                # named in full. That is what lets a limit escape the stall.
                truncated = replace(self, rung=self.rung % (1 << self.width))
                return pair(base_code, _code_tuple(truncated.rank.coeffs))
        raise ValueError(f"unknown presentation kind {self.kind!r}")

    def limit_status(self, *, bound: int = 64,
                     budget: int = 10000) -> "LimitStatus":
        """What can this theory determine, *about itself*, about its own limit?

        Every measurement in this series so far has been taken from outside —
        by an observer with a bigger notation system, which is to say by us.
        This is the first thing a theory can run on itself, and the three
        answers are not symmetric:

        - `available` — the indexed presentations. An index is a *description*,
          "PA plus every rung below this point" is a description like any other,
          and the canonical fundamental sequence certifies it with no search.
        - `absent` — `inline`. Its index is the Gödel number of a literal list,
          and the union has no finite list. A syntactic check, decided at once.
        - `unknown` — `searched`. Its limit notation is an arbitrary index, so
          certifying it means checking a fundamental sequence for totality by
          *running* it. The check can refute but never confirm.

        The asymmetry is the point. Two of these walls a system can see coming
        from any distance. The third it cannot see at all — and note that the
        `searched` arm's sequence is in fact total, so its continuation really
        does exist. It still cannot proceed on it, because it cannot authorise
        a step it cannot certify.
        """
        match self.kind:
            case "inline":
                return LimitStatus(
                    "absent", SequenceVerdict("diverges-at", 0, "no finite list"),
                    "the index is a literal axiom list; a union has none")
            case "indexed" | "truncated":
                return LimitStatus(
                    "available", verify_cnf_notation(self.rank.limit(1)),
                    "canonical fundamental sequence, certified without search")
            case "searched":
                verdict = verify_searched_notation(_opaque_sequence,
                                                   bound=bound, budget=budget)
                status = "absent" if verdict.status == "diverges-at" else "unknown"
                return LimitStatus(status, verdict,
                                   "an arbitrary index must be certified by "
                                   "running it, and running it cannot confirm")
        raise ValueError(f"unknown presentation kind {self.kind!r}")

    def can_take_limit(self) -> bool:
        """Whether the limit edge is *certified* available.

        Note what this returns for `searched`: `False`, because an uncertified
        edge is not one the theory may take — not because the edge is absent.
        `limit_status` is the honest three-valued version and should be
        preferred wherever the distinction matters.
        """
        return self.limit_status().status == "available"

    def presentation_symbols(self) -> int:
        """Cost of the presentation the generator actually maintains.

        For `inline` that is the literal axiom list, so this equals
        `expanded_symbols` by construction. For the indexed arms it is the base
        plus the numeral for the index — the recursive presentation, which
        never lists the rungs it denotes, and whose cost is therefore bounded
        by the base code rather than by the height reached."""
        if self.kind == "inline":
            return total_symbols(self.axioms()) + len(self.schemas)
        return (total_symbols(self.base) + len(self.schemas)
                + symbols(Num(self.index())))

    def expanded_symbols(self) -> int:
        """Cost of writing every axiom out, for every arm alike."""
        return total_symbols(self.axioms()) + len(self.schemas)


def peano(kind: str = "indexed", width: int | None = None) -> Theory:
    """`T_0 = PA` under the named presentation."""
    if kind not in ("inline", "indexed", "truncated", "searched"):
        raise ValueError(f"unknown presentation kind {kind!r}")
    if kind == "truncated" and not width:
        raise ValueError("truncated presentations need a positive width")
    return Theory(kind=kind, rung=0, base=pa_base_axioms(),
                  schemas=("induction",), rungs=(), width=width)


def con_formula(theory: Theory) -> Formula:
    """`Con(T) := ¬∃p Prf(⌜T⌝, p, ⌜0 = S0⌝)` — no proof of falsity from T.

    Constructing this is computable and cheap. Deciding its truth is neither,
    and nothing in this module tries to.
    """
    return Not(Exists("p", Prf(Num(theory.index()), _P, Num(FALSITY_CODE))))


@dataclass(frozen=True)
class Step:
    """One rung of the ladder, with everything measured about taking it."""

    n: int
    con: Formula
    theory_before: Theory
    theory_after: Theory
    #: Did the axiom *set* actually grow? The one measured productivity bit.
    new_axiom: bool
    #: The index `Con` named at this rung — the number identifying the theory
    #: being reflected on. Repeats across a ladder are what precede a stall,
    #: so this is recorded per rung and audited over the whole run.
    index: int
    con_symbols: int
    build_seconds: float


def step(theory: Theory) -> Step:
    """Apply the continuation operator `K(T) = T + Con(T)` once."""
    t0 = time.perf_counter()
    index = theory.index()
    con = con_formula(theory)
    new_axiom = index not in theory.seen
    rungs = theory.rungs + (con,) if new_axiom else theory.rungs
    after = replace(theory, rung=theory.rung + 1, rungs=rungs,
                    seen=theory.seen | {index})
    elapsed = time.perf_counter() - t0
    return Step(n=theory.rung, con=con, theory_before=theory, theory_after=after,
                new_axiom=new_axiom, index=index, con_symbols=symbols(con),
                build_seconds=elapsed)


def ladder(theory: Theory, rungs: int) -> Iterator[Step]:
    """Generate `rungs` successive applications of K, starting from `theory`."""
    current = theory
    for _ in range(rungs):
        s = step(current)
        yield s
        current = s.theory_after


def _at_rank(theory: Theory, rank: Rank) -> Theory:
    """The same theory re-labelled at `rank`, splitting the CNF coefficients
    across the fields `Theory` stores them in."""
    top = rank.degree
    higher = tuple(rank.coefficient(e) for e in range(top, 1, -1))
    return replace(theory, rung=rank.successors, limits=rank.limits,
                   higher=higher)


def limit_step(theory: Theory, level: int = 1) -> Step:
    """Apply the hierarchical mechanism at `level`: pass to the union.

    At `level=1` this is `⋃ₙ T_{succ^n(a)}` — the limit of a ladder of
    successors, landing on the next multiple of `ω`. At `level=2` it is the
    limit of a ladder of *those* (`ω, ω·2, ω·3, … → ω²`), and so on up. The
    mechanism has the same shape at every level, which is the finding: a limit
    of limits is not a new kind of move, and it does not cost more.

    The union's axiom set is what the ladder has already accumulated — the
    rungs are unchanged. What is new is the *index*: the limit theory names
    that union as a single r.e. set. Reflecting on it then produces a
    consistency sentence about the whole ladder below, which no rung beneath
    it asserts.

    Raises `LimitUndefined` for a presentation with no index to offer. The
    successor mechanism is always available; the limit mechanism is not, and
    that asymmetry is the thing this function exists to expose.
    """
    target = theory.rank.limit(level)
    status = theory.limit_status()
    if status.status == "absent":
        raise LimitUndefined(
            f"the {theory.kind!r} presentation indexes a literal axiom list, "
            f"and the union at rank {target} has no finite list to index — "
            f"no budget makes this edge exist")
    if not status.decided:
        raise LimitUndefined(
            f"the {theory.kind!r} presentation cannot certify its own edge at "
            f"rank {target}: {status.reason}. The edge may well be there; this "
            f"theory cannot establish that, and declines the step it cannot "
            f"authorise")
    t0 = time.perf_counter()
    index = theory.index()
    at_limit = _at_rank(theory, target)
    con = con_formula(at_limit)
    new_axiom = at_limit.index() not in theory.seen
    rungs = theory.rungs + (con,) if new_axiom else theory.rungs
    after = replace(_at_rank(theory, target.successor()), rungs=rungs,
                    seen=theory.seen | {at_limit.index()})
    return Step(n=theory.rung, con=con, theory_before=theory,
                theory_after=after, new_axiom=new_axiom, index=index,
                con_symbols=symbols(con),
                build_seconds=time.perf_counter() - t0)


def first_index_collision(steps: Sequence[Step]) -> int | None:
    """The first rung whose index repeats one already used by an earlier rung.

    An r.e. axiom set has many indices, and nothing in the mathematics forbids
    a presentation from reusing one. When it does, `Con` names a theory the
    ladder has already reflected on and the step adds a sentence already
    present. This is the audit that separates a ladder that is climbing from
    one that is only counting.
    """
    seen: dict[int, int] = {}
    for s in steps:
        if s.index in seen:
            return s.n
        seen[s.index] = s.n
    return None


# ------------------------------------------------------------- proof checking
#
# A Hilbert system: propositionally complete, with ∀-elimination and
# generalisation. Enough to *check* that the sentences the ladder adds are
# usable as premises rather than merely listed — which is the difference
# between a theory gaining an axiom and a theory gaining a capability.

_LOGICAL_SCHEMAS = ("K", "S", "contraposition", "and-left", "and-right",
                    "and-intro", "or-left", "or-right")


def _matches_logical_schema(f: Formula) -> bool:
    match f:
        # A → (B → A)
        case Implies(left=a, right=Implies(left=_, right=a2)) if a == a2:
            return True
    match f:
        # (A → (B → C)) → ((A → B) → (A → C))
        case Implies(left=Implies(left=a, right=Implies(left=b, right=c)),
                     right=Implies(left=Implies(left=a2, right=b2),
                                   right=Implies(left=a3, right=c2))):
            if a == a2 == a3 and b == b2 and c == c2:
                return True
    match f:
        # (¬A → ¬B) → (B → A)
        case Implies(left=Implies(left=Not(arg=a), right=Not(arg=b)),
                     right=Implies(left=b2, right=a2)):
            if a == a2 and b == b2:
                return True
    match f:
        # (A ∧ B) → A   and   (A ∧ B) → B
        case Implies(left=And(left=a, right=b), right=c) if c in (a, b):
            return True
    match f:
        # A → (B → (A ∧ B))
        case Implies(left=a, right=Implies(left=b, right=And(left=a2, right=b2))):
            if a == a2 and b == b2:
                return True
    match f:
        # A → (A ∨ B)   and   B → (A ∨ B)
        case Implies(left=c, right=Or(left=a, right=b)) if c in (a, b):
            return True
    return False


def is_axiom(theory: Theory, f: Formula) -> bool:
    """Membership in the axiom set, schema instances included."""
    if f in theory.axioms():
        return True
    return "induction" in theory.schemas and is_induction_instance(f)


@dataclass(frozen=True)
class Line:
    """One line of a derivation: a formula and its justification.

    `rule` is one of `axiom`, `logical`, `mp`, `gen`, `ui`. `refs` names earlier
    lines by index; `var`/`term` carry the parameters of `gen` and `ui`.
    """

    formula: Formula
    rule: str
    refs: tuple[int, ...] = ()
    var: str | None = None
    term: Term | None = None


class ProofError(Exception):
    """A derivation failed to check, with the offending line named."""


def check_proof(theory: Theory, lines: Sequence[Line]) -> Formula:
    """Verify a Hilbert derivation line by line; return the formula proved.

    Raises `ProofError` on the first line that does not follow. There is no
    partial credit and no search — this checks, it does not prove.
    """
    if not lines:
        raise ProofError("empty derivation proves nothing")
    for i, line in enumerate(lines):
        match line.rule:
            case "axiom":
                ok = is_axiom(theory, line.formula)
            case "logical":
                ok = _matches_logical_schema(line.formula)
            case "mp":
                if len(line.refs) != 2 or any(r >= i for r in line.refs):
                    raise ProofError(f"line {i}: mp needs two earlier lines")
                major, minor = (lines[r].formula for r in line.refs)
                ok = (isinstance(major, Implies) and major.left == minor
                      and major.right == line.formula)
            case "gen":
                if len(line.refs) != 1 or line.refs[0] >= i or line.var is None:
                    raise ProofError(f"line {i}: gen needs one earlier line "
                                     f"and a variable")
                ok = line.formula == Forall(line.var, lines[line.refs[0]].formula)
            case "ui":
                if len(line.refs) != 1 or line.refs[0] >= i or line.term is None:
                    raise ProofError(f"line {i}: ui needs one earlier line "
                                     f"and a term")
                src = lines[line.refs[0]].formula
                ok = (isinstance(src, Forall)
                      and line.formula == substitute(src.body, src.var, line.term))
            case _:
                raise ProofError(f"line {i}: unknown rule {line.rule!r}")
        if not ok:
            raise ProofError(f"line {i}: {line.rule} does not justify this line")
    return lines[-1].formula


def conjoin_rungs_proof(theory: Theory) -> list[Line]:
    """A derivation, in `theory`, of the conjunction of all its rungs.

    This is the module's answer to "is the added sentence a capability or just
    an entry in a list?". Each rung enters as an axiom and is then *used*: the
    proof drives the conjunction-introduction schema through modus ponens,
    `3` lines per rung, and every line is machine-checked by `check_proof`.
    A theory that merely stored the sentences could not run this.
    """
    if not theory.rungs:
        raise ValueError("no rungs to conjoin")
    lines: list[Line] = [Line(theory.rungs[0], "axiom")]
    acc = theory.rungs[0]
    for rung in theory.rungs[1:]:
        acc_at = len(lines) - 1
        lines.append(Line(rung, "axiom"))
        rung_at = len(lines) - 1
        target = And(acc, rung)
        lines.append(Line(Implies(acc, Implies(rung, target)), "logical"))
        lines.append(Line(Implies(rung, target), "mp", (len(lines) - 1, acc_at)))
        lines.append(Line(target, "mp", (len(lines) - 1, rung_at)))
        acc = target
    return lines


def closure_search(theory: Theory, target: Formula, *, budget: int,
                   seeds: Sequence[Formula] = ()) -> tuple[bool, int]:
    """Forward-chain modus ponens from the axioms and report whether `target`
    appears within `budget` derived formulas.

    Read this for what it is. A negative result is *not* evidence that the
    theory cannot prove the target: the search covers a vanishing fragment of
    an infinite derivation space and does not instantiate the induction schema
    at all. Its real job is calibration — it is run against targets the theory
    demonstrably does prove, so that a negative result at least means the
    search works and the horizon is the only thing limiting it.

    Returns `(found, explored)`.
    """
    known: list[Formula] = list(theory.axioms()) + list(seeds)
    seen = set(known)
    if target in seen:
        return True, len(seen)
    frontier = list(known)
    while frontier and len(seen) < budget:
        current = frontier.pop(0)
        for other in list(known):
            for major, minor in ((current, other), (other, current)):
                if not isinstance(major, Implies) or major.left != minor:
                    continue
                derived = major.right
                if derived in seen:
                    continue
                seen.add(derived)
                known.append(derived)
                frontier.append(derived)
                if derived == target:
                    return True, len(seen)
                if len(seen) >= budget:
                    return False, len(seen)
    return False, len(seen)


# ------------------------------------------------------------ GCP observables
#
# The triple (C, I, G). Two of these three are model bookkeeping and one is
# measured, and the whole point of the experiment is that they can be made to
# disagree — so they are kept in separate functions with separate docstrings
# rather than blended into one "score".

#: The representational ceiling: the Church-Kleene ordinal. Fixed, symbolic,
#: and never computed with — it is the domain property `|a| < ω₁^CK` that makes
#: the gap permanent, not anything the generator does.
CAPACITY = "ω₁^CK"


def integration_rank(theory: Theory) -> int:
    """`I(T_n)` — the current reflective rank, as the ladder's own counter.

    Declared metadata, not a measurement. The mathematical model puts the rank
    at a recursive ordinal (`ε₀ + n` for this ladder over PA); the finite
    generator carries `n` as its proxy and does not implement Kleene's `O`.
    Reporting it as if it were measured is precisely the error the experiment
    is built to catch, which is why it is never compared against cost without
    the measured productivity bit alongside it.
    """
    return theory.rung


def nominal_increment(_theory: Theory) -> int:
    """`G` as the model asserts it: the successor is always available, so the
    accessible increment in `S` is at least 1 and the GCP condition `G > 0`
    holds by construction."""
    return 1


@dataclass(frozen=True)
class Continuation:
    """`G`, derived from four measured dimensions rather than posited.

    The original model carried `G` as a primitive and asserted `G > 0`. Five
    experiments then found four separate things that can independently stop a
    system, none of which `G` could express. This computes `G` *from* them.

    The four dimensions, each measured somewhere in the series:

    - `structural` — does the move exist at all under this naming scheme?
    - `affordable` — can the current budget pay for it?
    - `productive` — would it actually enlarge the axiom set?
    - `certifiable` — can the *system itself* establish `structural`?
      `None` means it cannot determine that either way.

    The first three are facts about the world; the fourth is a fact about what
    the system can know, which is why `G` splits in two. The gap between them
    is the finding of experiment five, and it has a name in each direction.
    """

    structural: bool
    affordable: bool
    productive: bool
    certifiable: bool | None
    #: Has the *domain* run out — `I = C`? Always `False` in the arithmetic
    #: setting, where `I < C` is a theorem and the ceiling is unreachable.
    #: It took a saturating domain (`finite_ladder.py`) to produce a state
    #: where this is `True`, which is exactly why it was missing from the
    #: first four dimensions rather than overlooked in them.
    domain_exhausted: bool = False

    @property
    def g_actual(self) -> int:
        """`G` as an outside observer with full information computes it."""
        return int(self.structural and self.affordable and self.productive)

    @property
    def g_certified(self) -> int:
        """`G` as the system itself can establish it. Never exceeds `g_actual`
        — a system cannot certify a move that is not there, which is asserted
        in the tests rather than assumed here."""
        return int(bool(self.certifiable) and self.affordable and self.productive)

    @property
    def moves_exist(self) -> bool:
        """Is there *any* move — productive or not — the system can make?

        The distinction between having no move and having only useless ones is
        the difference between halting and running forever without getting
        anywhere, and those are not the same failure.
        """
        return self.structural and self.affordable

    @property
    def verdict(self) -> str:
        """`terminal` | `stagnant` | `hidden` | `recognised`.

        Four, not three. An earlier version returned `terminal` whenever
        `g_actual` was 0, which silently merged two different situations: a
        system with no move at all, and a system with moves that achieve
        nothing. The second one does not halt — it runs to the horizon — so
        calling it terminal contradicts the measurement that found it.

        - `exhausted` — the domain itself ran out, `I = C`. Like `stagnant`
          it does not halt, but unlike `stagnant` no naming scheme rescues it:
          there is nothing left to name. Measured in `reflection_finite_box.py`;
          unreachable in the arithmetic setting by theorem.
        - `terminal` — no move exists at all. The system stops.
        - `stagnant` — moves exist, none is productive. The system *continues*
          and gets nowhere, with every wall-detector reading normal.
        - `hidden` — a productive move is there and cannot be certified.
        - `recognised` — a productive move is there and is certified.

        Note the ordering: `terminal` is checked first, so the degenerate case
        `g_actual = g_certified = 0` cannot also read as `recognised`. That
        ambiguity is real in the informal table version of this rule and is why
        the rule lives here rather than in prose.
        """
        if self.domain_exhausted:
            return "exhausted"
        if not self.moves_exist:
            return "terminal"
        if not self.productive:
            return "stagnant"
        return "recognised" if self.g_certified else "hidden"

    @property
    def halts(self) -> bool:
        """Does this verdict stop the system? `stagnant` notably does not."""
        return self.verdict in ("terminal", "hidden")

    @property
    def blocked_by(self) -> str | None:
        """Which dimension failed, named in the order a climb would meet it."""
        if self.domain_exhausted:
            return "exhausted"
        if not self.affordable:
            return "economic"
        if not self.structural:
            return "structural"
        if not self.productive:
            return "unproductive"
        if not self.certifiable:
            return "epistemic"
        return None


def derive_continuation(theory: Theory, *, move: str = "limit",
                        capacity: Capacity | None = None,
                        kappa: float | None = None,
                        cost_of=None) -> Continuation:
    """Measure the four dimensions of one candidate move and derive `G`.

    `move` is `"successor"` or `"limit"`. A successor is always structurally
    available and always certifiable — that asymmetry with the limit is the
    whole reason the series needed two mechanisms.
    """
    cost_of = cost_of or construction_cost
    if move == "successor":
        s = step(theory)
        structural, certifiable = True, True
        productive, cost = s.new_axiom, cost_of(s)
    elif move == "limit":
        status = theory.limit_status()
        structural = status.status != "absent"
        certifiable = (True if status.status == "available"
                       else (False if status.status == "absent" else None))
        if structural:
            probe = limit_step(replace(theory, kind="indexed"))
            productive, cost = probe.new_axiom, cost_of(probe)
        else:
            productive, cost = False, 0
    else:
        raise ValueError(f"unknown move {move!r}")

    budget = kappa if kappa is not None else (
        capacity.kappa_max if capacity else None)
    affordable = budget is None or budget >= cost
    return Continuation(structural=structural, affordable=affordable,
                        productive=productive, certifiable=certifiable)


def productive_increment(s: Step) -> int:
    """`G` as measured: 1 if the step actually enlarged the axiom set, else 0.

    This is the only quantity here that can come out other than the model says,
    and the `truncated` arm makes it do so.
    """
    return 1 if s.new_axiom else 0


# ------------------------------------------------- cost-bounded accessibility
#
# The unbounded ladder above cannot terminate: the accessibility relation was
# defined to contain successors, so successors are accessible and `G > 0` is
# true by inspection of the definition. That is the tautology.
#
# The field program does not have this problem, because there integration is
# paid for out of a capacity field κ that is consumed by load and regenerates
# with slack. The ordinal column has no such budget — reflection is free — and
# so it drops the one feature that makes the field column interesting.
#
# This section ports the budget across. Accessibility becomes contingent: a
# successor is reachable only if the theory can afford to construct it, out of
# a capacity that the construction consumes and that heals at a rate `r`. Then
# terminal states are reachable rather than ruled out, and `G > 0` becomes
# something a run can refute.


@dataclass(frozen=True)
class Capacity:
    """A capacity budget with the field program's own dynamics.

    Discretised onto the ladder, `∂_t κ = r(κ₀ − κ) − load` becomes: pay the
    construction cost out of κ, then heal a fraction `r` of the way back to
    `kappa_max`. `recovery = 1` is full healing between rungs (a pure stock
    constraint); `recovery → 0` is a budget that never comes back.
    """

    kappa_max: float
    recovery: float = 1.0

    def __post_init__(self) -> None:
        if self.kappa_max <= 0:
            raise ValueError("capacity must be positive")
        if not 0 < self.recovery <= 1:
            raise ValueError("recovery rate must lie in (0, 1]")

    def spend(self, kappa: float, cost: float) -> float:
        """Pay `cost`, then regenerate toward the ceiling."""
        left = kappa - cost
        return left + self.recovery * (self.kappa_max - left)


def construction_cost(s: Step) -> int:
    """Default cost model: the symbols of the sentence being constructed.

    This is a *flow* cost — what it takes to build the successor — as against
    the *stock* cost of holding the presentation, which `presentation_symbols`
    reports. The flow is the one the budget meters, because it is the act of
    continuation that has to be afforded.
    """
    return s.con_symbols


@dataclass(frozen=True)
class BoundedStep:
    """One rung attempted under a budget."""

    n: int
    cost: int
    kappa_before: float
    kappa_after: float
    #: Could the theory pay for this successor at all?
    affordable: bool
    #: Did it enlarge the axiom set? Meaningless unless `affordable`.
    new_axiom: bool
    step: Step | None


def bounded_ladder(theory: Theory, rungs: int, capacity: Capacity, *,
                   cost_of=construction_cost,
                   require_productive: bool = False) -> Iterator[BoundedStep]:
    """Climb under a capacity budget, stopping at the first rung it cannot buy.

    With `require_productive`, accessibility is restricted further: a step
    counts as taken only if it is *both* affordable and enlarges the axiom set.
    That is the corrected relation — the plain budget rules out steps that cost
    too much, but says nothing at all about steps that cost little and achieve
    nothing.
    """
    current, kappa = theory, capacity.kappa_max
    for _ in range(rungs):
        s = step(current)
        cost = cost_of(s)
        affordable = kappa >= cost
        productive_enough = s.new_axiom or not require_productive
        if not affordable or not productive_enough:
            yield BoundedStep(n=s.n, cost=cost, kappa_before=kappa,
                              kappa_after=kappa, affordable=affordable,
                              new_axiom=s.new_axiom, step=None)
            return
        after = capacity.spend(kappa, cost)
        yield BoundedStep(n=s.n, cost=cost, kappa_before=kappa,
                          kappa_after=after, affordable=True,
                          new_axiom=s.new_axiom, step=s)
        current, kappa = s.theory_after, after


def terminal_rung(theory: Theory, capacity: Capacity, *, horizon: int,
                  cost_of=construction_cost,
                  require_productive: bool = False) -> int | None:
    """The rung at which the ladder can no longer continue, or `None` if it
    survives the whole horizon."""
    for b in bounded_ladder(theory, horizon, capacity, cost_of=cost_of,
                            require_productive=require_productive):
        if b.step is None:
            return b.n
    return None


# ------------------------------------------ notations, and the price of Kleene
#
# A limit notation is only meaningful if it comes with a *fundamental sequence*:
# a computable increasing sequence converging to it, so the union it names is
# actually enumerable. Below ω^ω that sequence is canonical — read straight off
# the Cantor normal form, closed form, total by construction, no search. That is
# why every result above is cheap and decidable.
#
# Kleene's O drops that guarantee. There a limit notation is an arbitrary index
# `e` for a function enumerating the sequence, and for the notation to be valid
# that function must be *total* — a Π⁰₂ question in general, and O-membership as
# a whole is Π¹₁-complete. Those are cited theorems, not measurements; nothing
# here proves them and nothing here could.
#
# What the code below *does* show is the shape of the consequence: a checker
# that must search cannot return "valid", only "verified this far", and no
# finite amount of verification distinguishes a total sequence from one that
# diverges later. So the price of O is not implementation effort. It is that
# `can_take_limit` stops being a decision and becomes a search — the system can
# no longer determine which continuations are open to it.


@dataclass(frozen=True)
class SequenceVerdict:
    """What a bounded check of a fundamental sequence was able to conclude."""

    status: str          # total-by-construction | verified-to | diverges-at
    checked: int
    detail: str

    @property
    def conclusive(self) -> bool:
        """Did the check settle validity, rather than merely fail to refute it?"""
        return self.status in ("total-by-construction", "diverges-at")


def canonical_fundamental_sequence(rank: Rank):
    """The standard fundamental sequence for a CNF limit, in closed form.

    For `ω^(k+1)` it is `n ↦ ω^k·n`; for `ω^k·(m+1)` it is
    `n ↦ ω^k·m + ω^(k-1)·n`, and at `k = 1` simply `n ↦ ω·m + n`. Total by
    construction: it is arithmetic on the coefficients, with no search in it.
    """
    if not rank.is_limit or not rank.coeffs:
        raise ValueError(f"{rank} is not a limit and has no fundamental sequence")
    k = rank.degree
    m = rank.coefficient(k) - 1
    below = {e: rank.coefficient(e) for e in range(k + 1, rank.degree + 1)}

    def seq(n: int) -> Rank:
        levels = dict(below)
        levels[k] = m
        if k >= 1:
            levels[k - 1] = levels.get(k - 1, 0) + n
        return Rank.from_levels(levels)

    return seq


def verify_cnf_notation(rank: Rank) -> SequenceVerdict:
    """Decide whether a CNF limit notation is valid. Total, and immediate."""
    if not rank.is_limit:
        return SequenceVerdict("total-by-construction", 0,
                               f"{rank} is a successor or zero — no sequence needed")
    seq = canonical_fundamental_sequence(rank)
    for n in range(3):
        if not seq(n) < seq(n + 1) or not seq(n) < rank:
            return SequenceVerdict("diverges-at", n,
                                   f"canonical sequence misbehaved at {n}")
    return SequenceVerdict(
        "total-by-construction", 0,
        f"the fundamental sequence for {rank} is closed-form arithmetic on the "
        f"Cantor normal form; totality needs no search")


def verify_searched_notation(seq, *, bound: int, budget: int) -> SequenceVerdict:
    """Check an *opaque* fundamental sequence by running it.

    `seq(n, budget)` returns the n-th element, or `None` if it did not halt
    within `budget` steps — standing in for the divergence a genuine index may
    exhibit. The verdict is deliberately unable to say "valid": a run that
    halted for every `n < bound` has refuted nothing about `n ≥ bound`, which
    is exactly the gap that makes totality undecidable.
    """
    last = None
    for n in range(bound):
        value = seq(n, budget)
        if value is None:
            return SequenceVerdict("diverges-at", n,
                                   f"element {n} did not halt within {budget} steps")
        if last is not None and value <= last:
            return SequenceVerdict("diverges-at", n,
                                   f"sequence not increasing at {n}")
        last = value
    return SequenceVerdict(
        "verified-to", bound,
        f"halted and increased for every n < {bound}; nothing is known about "
        f"n >= {bound}, and no finite bound changes that")


#: The opaque fundamental sequence the `searched` presentation must certify.
#: It is *total* — `n ↦ n` — so the continuation it names genuinely exists.
#: The point is that no bounded check can establish that, which is why the arm
#: cannot proceed on an edge that is really there.
def _opaque_sequence(n: int, _budget: int) -> int:
    return n


@dataclass(frozen=True)
class LimitStatus:
    """A theory's own three-valued verdict on its own limit edge."""

    status: str          # available | absent | unknown
    verdict: SequenceVerdict
    reason: str

    @property
    def decided(self) -> bool:
        """Could the theory settle the question at all? `unknown` is the whole
        content of the third wall: not that the answer is no, but that the
        system cannot reach an answer."""
        return self.status in ("available", "absent")


@dataclass(frozen=True)
class Prediction:
    """What a theory can work out about its own stopping point, from inside.

    Two questions, and they come apart — which is the whole finding:

    - `stop_rung`: *where* will I stop? `None` means "I reach the horizon".
    - `wall_is_real`: is the thing that stops me an actual absence of
      continuation, or merely one I cannot certify? `None` means the theory
      cannot determine this.

    A cautious system — one that declines steps it cannot certify — always
    knows *where* it will halt. What it may not know is whether halting was
    necessary. That is a sharper statement than "it cannot see the wall", and
    it is the one the measurement supports.
    """

    stop_rung: int | None
    reason: str
    wall_is_real: bool | None
    detail: str

    @property
    def certain(self) -> bool:
        """Did the theory settle *why* it stops, not merely where?"""
        return self.wall_is_real is not None


def predict_stop(theory: Theory, *, blocks: int, per_block: int,
                 capacity: "Capacity | None" = None,
                 cost_of=None) -> Prediction:
    """The interior view: what the theory determines about its own walls.

    Uses only checks the theory can run on itself — its own presentation, its
    own cost function, its own budget, its own `limit_status`. It never
    consults the outcome of a run, which is what makes comparing it against an
    actual climb meaningful rather than circular.

    Walls are resolved *in the order they would be met*, not in order of
    interest. Checking the limit edge first is wrong whenever the budget binds
    sooner, and gets the reason wrong even when it happens to get the rung
    right.
    """
    cost_of = cost_of or construction_cost
    kappa = capacity.kappa_max if capacity else None
    current, taken = theory, 0

    def afford(s: Step) -> bool:
        nonlocal kappa
        if capacity is None:
            return True
        if kappa < cost_of(s):
            return False
        kappa = capacity.spend(kappa, cost_of(s))
        return True

    for block in range(blocks):
        for _ in range(per_block):
            s = step(current)
            if not afford(s):
                return Prediction(taken, "unaffordable", True,
                                  f"budget {kappa:.0f} < cost {cost_of(s)}")
            current, taken = s.theory_after, taken + 1
        if block == blocks - 1:
            break
        status = current.limit_status()
        if status.status == "absent":
            return Prediction(taken, "limit-undefined", True, status.reason)
        if not status.decided:
            # It knows it will halt here — a step it cannot certify is a step
            # it will not take. What it cannot determine is whether the
            # continuation was there all along.
            return Prediction(taken, "undecidable", None, status.reason)
        lim = limit_step(current)
        if not afford(lim):
            return Prediction(taken, "unaffordable", True, "limit unaffordable")
        current, taken = lim.theory_after, taken + 1
    return Prediction(None, "horizon", True, "reaches the horizon")


@dataclass(frozen=True)
class ClimbOutcome:
    """Where a mixed successor/limit climb stopped, and why.

    `stopped_because` is the whole point: a climb can end for two structurally
    different reasons. `unaffordable` is *economic* — the budget could not buy
    the next successor, and a larger budget moves it. `limit-undefined` is
    *structural* — the presentation has no index for the union, and no budget
    moves it at all. The capacity experiment found the first kind of terminal
    state; this finds the second, and they are not the same thing.
    """

    rank: Rank
    productive: int
    taken: int
    stopped_because: str
    limits_taken: int


def _uncertified_limit_step(theory: Theory, level: int = 1) -> Step:
    """Take a limit edge the theory could not certify.

    Reached only from a climb running `require_certified=False`, modelling a
    system that proceeds on an edge it cannot authorise. Kept as a separate
    function so the ordinary path cannot arrive here by accident: proceeding
    uncertified is a deliberate posture, not a fallback. Whether it is *right*
    is exactly what the system cannot determine — in this construction it
    happens to be, because the sequence is total, and that is luck rather than
    knowledge.
    """
    if theory.limit_status().status == "absent":
        raise LimitUndefined(f"{theory.kind!r} has no limit edge to take")
    return limit_step(replace(theory, kind="indexed"), level)


def transfinite_climb(theory: Theory, *, blocks: int, per_block: int,
                      capacity: Capacity | None = None,
                      cost_of=None,
                      require_certified: bool = True) -> ClimbOutcome:
    """Climb `blocks` many ω-blocks of `per_block` successors each.

    Alternates the two mechanisms the model names: `per_block` applications of
    `K(T) = T + Con(T)`, then one limit, repeated. With a `capacity`, each
    successor must be affordable; the limit is charged the same way.
    """
    cost_of = cost_of or construction_cost
    current = theory
    kappa = capacity.kappa_max if capacity else None
    productive = taken = 0

    def charge(s: Step) -> bool:
        nonlocal kappa
        if capacity is None:
            return True
        if kappa < cost_of(s):
            return False
        kappa = capacity.spend(kappa, cost_of(s))
        return True

    for block in range(blocks):
        for _ in range(per_block):
            s = step(current)
            if not charge(s):
                return ClimbOutcome(current.rank, productive, taken,
                                    "unaffordable", current.limits)
            taken += 1
            productive += 1 if s.new_axiom else 0
            current = s.theory_after
        if block == blocks - 1:
            break
        status = current.limit_status()
        if status.status == "absent":
            return ClimbOutcome(current.rank, productive, taken,
                                "limit-undefined", current.limits)
        if not status.decided and require_certified:
            # The halt here is a POLICY, not a fact about the world. With
            # `require_certified=False` the same theory takes the same step and
            # continues — correctly, as it happens, since its sequence is
            # total. The third wall's position is set by how much certainty the
            # system demands, which is what makes it unlike the other two.
            return ClimbOutcome(current.rank, productive, taken,
                                "undecidable", current.limits)
        s = (limit_step(current) if status.decided
             else _uncertified_limit_step(current))
        if not charge(s):
            return ClimbOutcome(current.rank, productive, taken,
                                "unaffordable", current.limits)
        taken += 1
        productive += 1 if s.new_axiom else 0
        current = s.theory_after
    return ClimbOutcome(current.rank, productive, taken, "horizon",
                        current.limits)


def critical_recovery(cost: float, kappa_max: float) -> float:
    """Closed form for the sustainable recovery rate at *constant* cost `L`.

    Paying `L` and healing a fraction `r` back toward `κ_max` has the fixed
    point `κ* = κ_max − L(1−r)/r`. The ladder survives indefinitely exactly
    when `κ* ≥ L`, i.e. when `κ_max ≥ L/r`, so the threshold is

        r* = L / κ_max.

    Below it the budget drifts down to a level that cannot buy the next rung;
    above it, it settles at a level that can, forever. Arms whose cost grows
    have no such threshold — no fixed `r` sustains an unbounded cost — and this
    formula does not apply to them. It exists to be checked against a measured
    bisection, which is the only reason to trust the numerical answer.
    """
    return cost / kappa_max


def measure_critical_recovery(theory: Theory, capacity_max: float, *,
                              horizon: int, cost_of=construction_cost,
                              tol: float = 1e-6) -> float | None:
    """Bisect for the smallest recovery rate that survives `horizon` rungs.

    Returns `None` if even full recovery (`r = 1`) fails — which is the honest
    answer for an arm whose cost grows without bound.
    """
    def survives(r: float) -> bool:
        cap = Capacity(kappa_max=capacity_max, recovery=r)
        return terminal_rung(theory, cap, horizon=horizon,
                             cost_of=cost_of) is None

    if not survives(1.0):
        return None
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if mid <= 0 or not survives(mid):
            lo = mid
        else:
            hi = mid
    return hi
