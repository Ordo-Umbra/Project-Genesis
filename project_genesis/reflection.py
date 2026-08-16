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

    @property
    def name(self) -> str:
        return f"T_{self.rung}" if self.rung else "PA"

    def axioms(self) -> tuple[Formula, ...]:
        """The explicit axioms — base plus rungs. The induction schema is
        carried in `schemas` and instantiated on demand, which is what keeps
        the presentation finite."""
        return self.base + self.rungs

    def index(self) -> int:
        """The natural number this presentation offers as its own index — the
        thing `Con` has to name. This is where the three arms diverge."""
        base_code = godel_number(list(self.base) + [Var(s) for s in self.schemas])
        match self.kind:
            case "inline":
                return godel_number(list(self.axioms())
                                    + [Var(s) for s in self.schemas])
            case "indexed":
                return pair(base_code, self.rung)
            case "truncated":
                if self.width is None:
                    raise ValueError("truncated presentations need a width")
                return pair(base_code, self.rung % (1 << self.width))
        raise ValueError(f"unknown presentation kind {self.kind!r}")

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
    if kind not in ("inline", "indexed", "truncated"):
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
    new_axiom = con not in theory.axioms()
    rungs = theory.rungs + (con,) if new_axiom else theory.rungs
    after = replace(theory, rung=theory.rung + 1, rungs=rungs)
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


def productive_increment(s: Step) -> int:
    """`G` as measured: 1 if the step actually enlarged the axiom set, else 0.

    This is the only quantity here that can come out other than the model says,
    and the `truncated` arm makes it do so.
    """
    return 1 if s.new_axiom else 0
