"""Multi-parent reflection: consistency taken over any finite set of theories.

Three domains so far have all been *linear*: a chain of successors, a box filled
one atom at a time, a model set shrunk one valuation at a time. In each, "the
axiom set grew" and "the construction advanced" were the same event, because
there was only one place growth could happen.

Ordinary proof theory does not require that. A consistency statement can be
taken relative to any finite collection of theories, not merely the previous one
in a chain. Allowing that turns the ladder into a **directed acyclic graph**, and
separates two things the linear domains could not:

    local productivity — this step added a sentence that is genuinely new
    global advance     — the collection as a whole reached further

Content, and why it must be normalised
--------------------------------------
A node asserts the consistency of everything in the transitive closure of its
parents. So `Con({T₁, T₂})` where `T₁` is an ancestor of `T₂` asserts exactly
what `Con({T₂})` asserts — the same closure, the same sentence. Collapsing those
is not an optimisation; without it, re-reflecting on a chain would look
productive forever and the phenomenon under test would be manufactured by the
representation rather than found in it.

That normalisation has a consequence worth stating before any run: **over a pure
chain, every multi-parent reflection is a duplicate**, since any subset's union
closure equals the closure of its maximum, which was already asserted. So the
separation this domain exists to test should require genuine branching. That is
a prediction, and it is falsifiable here.

Rank is a declared proxy
------------------------
`depth` — longest path from the base — stands in for proof-theoretic ordinal
height, exactly as `I = n` did in the arithmetic setting. It is a structural
measure of how far the reflection has been iterated, not a computed ordinal, and
nothing here computes one.
"""

from __future__ import annotations

from itertools import combinations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Node:
    """One theory in the reflection graph."""

    ident: int
    parents: frozenset[int]
    #: Everything this node is *about*: the union of its parents' contents plus
    #: itself. The sentence a node adds is keyed by the union of its parents'
    #: contents, so two reflections over sets with the same union assert the
    #: same thing and only one of them is productive.
    content: frozenset[int]
    depth: int


@dataclass(frozen=True)
class ReflectionGraph:
    """A collection of theories closed under finite-join reflection."""

    nodes: tuple[Node, ...] = ()
    #: Sentence-keys already asserted, so a repeat is detectable in O(1).
    asserted: frozenset[frozenset[int]] = frozenset()
    #: Identifier of the first root. Shifting it relabels every node without
    #: touching the graph's shape, which is what makes an encoding artifact
    #: visible: any wall whose counts move under a pure relabelling is reading
    #: the raw identifier rather than anything structural.
    first_id: int = 0

    @classmethod
    def base(cls, roots: int = 1, first_id: int = 0) -> "ReflectionGraph":
        """Start from `roots` mutually independent theories.

        `roots = 1` is the free generation from a single base, and it is worth
        running: it collapses to a chain, because every join over a chain has
        the union of a nested family as its key, which is the largest member's
        content and is therefore already asserted. Branching — and so everything
        this domain exists to test — requires more than one starting point.
        """
        nodes = tuple(Node(ident=i, parents=frozenset(),
                           content=frozenset({i}), depth=0)
                      for i in range(first_id, first_id + roots))
        return cls(nodes=nodes, asserted=frozenset(), first_id=first_id)

    def node(self, ident: int) -> Node:
        return self.nodes[ident - self.first_id]

    @property
    def rank(self) -> int:
        """Global rank: the deepest the collection has reached. Declared proxy."""
        return max(n.depth for n in self.nodes)

    @property
    def size(self) -> int:
        return len(self.nodes)

    def frontier(self) -> tuple[Node, ...]:
        top = self.rank
        return tuple(n for n in self.nodes if n.depth == top)

    def shallow(self, at_most: int) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes if n.depth <= at_most)

    def branches(self) -> bool:
        """Are there two nodes whose contents are **incomparable**?

        Not "does some node have two parents" — that is the wrong test and it
        cost two failing assertions to learn. Joining the frontier with anything
        beneath it produces a node with two parents whose key is *identical* to
        reflecting the frontier alone: nominal branching, no new content. What
        matters is whether the collection holds two theories neither of which
        subsumes the other, because that is the only situation in which a join
        says something no single reflection does.
        """
        for i, a in enumerate(self.nodes):
            for b in self.nodes[i + 1:]:
                if not (a.content <= b.content or b.content <= a.content):
                    return True
        return False

    def is_chain(self) -> bool:
        """Totally ordered by content — the condition under which every join is
        redundant with reflecting its maximum."""
        return not self.branches()


@dataclass(frozen=True)
class DagStep:
    """One attempted reflection over a chosen set of parents."""

    parents: frozenset[int]
    key: frozenset[int]
    depth: int
    productive: bool
    rank_before: int
    rank_after: int
    graph_after: ReflectionGraph

    @property
    def rank_advanced(self) -> bool:
        return self.rank_after > self.rank_before

    @property
    def kind(self) -> str:
        """The distinction this domain exists to expose.

        `advancing`  — added something new AND the collection reached further
        `sideways`   — added something new and the collection did NOT advance
        `duplicate`  — added nothing; ordinary stagnation
        """
        if not self.productive:
            return "duplicate"
        return "advancing" if self.rank_advanced else "sideways"


def reflect(graph: ReflectionGraph, parents: frozenset[int]) -> DagStep:
    """Add `Con(parents)` — the consistency of everything they transitively
    assert — as a new node, if that sentence is not already present."""
    if not parents:
        raise ValueError("reflection needs at least one parent")
    key = frozenset().union(*(graph.node(p).content for p in parents))
    depth = 1 + max(graph.node(p).depth for p in parents)
    productive = key not in graph.asserted
    rank_before = graph.rank
    if productive:
        ident = graph.first_id + graph.size
        node = Node(ident=ident, parents=parents, content=key | {ident},
                    depth=depth)
        after = replace(graph, nodes=graph.nodes + (node,),
                        asserted=graph.asserted | {key})
    else:
        after = graph
    return DagStep(parents=parents, key=key, depth=depth,
                   productive=productive, rank_before=rank_before,
                   rank_after=after.rank, graph_after=after)


def options(graph: ReflectionGraph, filters: "Filters | None" = None) -> dict:
    """How many distinct productive moves are available right now.

    The move space is singletons and incomparable pairs — what the policies
    here actually propose. That restriction is declared rather than derived: the
    full space is every subset, which is exponential and which no policy
    searches. A count over a larger space would be a different number.

    This exists because `rank` measures **height only**. A move that widens the
    base, and so makes more future moves possible, scores zero on it. Whether
    that is a real capability the rank measure cannot see — or whether option
    growth simply never converts into height — is the question, and it cannot
    be asked without counting the options.
    """
    f = filters or Filters()
    single = sum(1 for n in graph.nodes
                 if n.content not in graph.asserted
                 and f.admits(n.content, max(n.content), 1)[0])
    join = 0
    for a, b in combinations(graph.nodes, 2):
        if a.content <= b.content or b.content <= a.content:
            continue
        key = a.content | b.content
        if key not in graph.asserted and f.admits(key, max(key), 2)[0]:
            join += 1
    return {"single": single, "join": join, "total": single + join}


def certified_rank(graph: ReflectionGraph, filters: "Filters") -> int:
    """Height the system can still stand behind.

    `rank` counts every node ever built. But a reflection tower over a base
    whose consistency cannot be settled is itself unsettleable — the height is
    there, and none of it is certified. Which of the two measures is right is
    not a detail: it decides whether a wall *freezes* your work or *retracts*
    it, and that turns out to be what the whole concentrate-or-diversify
    question hinges on.
    """
    live = [n.depth for n in graph.nodes
            if filters.admits(n.content, max(n.content), 1)[0]]
    return max(live) if live else 0


# --------------------------------------------------------------- strategies
#
# How a system chooses what to reflect on. Not a naming scheme — a content set
# names itself here — but a *policy*, and it is the policy that decides whether
# a productive step also advances the collection.


def deepen(graph: ReflectionGraph) -> frozenset[int]:
    """Always reflect on the deepest node: the linear ladder, recovered."""
    return frozenset({graph.frontier()[0].ident})


def spread(graph: ReflectionGraph) -> frozenset[int]:
    """Grow every lineage evenly: reflect on the shallowest unasserted node.

    Deliberate diversification — many shallow towers rather than one tall one.
    It is not a join-seeking policy; every move it makes is a single-parent
    reflection, so what it buys is not new *kinds* of content but a spread of
    live frontiers.
    """
    for n in sorted(graph.nodes, key=lambda n: (n.depth, n.ident)):
        if n.content not in graph.asserted:
            return frozenset({n.ident})
    return frozenset({graph.frontier()[0].ident})


def broaden(graph: ReflectionGraph) -> frozenset[int]:
    """Join shallow, incomparable nodes.

    Once the graph has independent roots, a join of two incomparable shallow
    nodes has a key nobody has asserted — so the step is productive — while its
    depth stays far below the frontier, so the collection does not advance.
    """
    ceiling = max(1, graph.rank // 2)
    pool = graph.shallow(ceiling) or graph.nodes[:2]
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            key = pool[i].content | pool[j].content
            if key not in graph.asserted:
                return frozenset({pool[i].ident, pool[j].ident})
    return frozenset({graph.nodes[0].ident})


def run_policy(policy, steps: int, *, roots: int = 1,
               warmup: int = 0) -> dict:
    """Run a policy and count how each step landed.

    `warmup` deepens first, so that a later `broaden` step is demonstrably
    *below* an established frontier rather than at it.
    """
    graph = ReflectionGraph.base(roots=roots)
    for _ in range(warmup):
        graph = reflect(graph, deepen(graph)).graph_after

    tally = {"advancing": 0, "sideways": 0, "duplicate": 0}
    trace = []
    for _ in range(steps):
        s = reflect(graph, policy(graph))
        tally[s.kind] += 1
        trace.append({"parents": sorted(s.parents), "depth": s.depth,
                      "productive": s.productive,
                      "rank_before": s.rank_before,
                      "rank_after": s.rank_after, "kind": s.kind})
        graph = s.graph_after
    return {"tally": tally, "trace": trace, "final_rank": graph.rank,
            "final_size": graph.size, "roots": roots,
            "branching": graph.branches(),
            "nominal_multiparent": any(len(n.parents) > 1
                                       for n in graph.nodes)}


def joins(graph: ReflectionGraph) -> int:
    """How many nodes were built by joining two or more theories.

    This is the capability a chain does not have: a node whose content is the
    union of two incomparable theories asserts something no single reflection
    asserts. Counting it separates "the climb continued" from "the climb kept
    everything it could reach", which rank alone cannot see.
    """
    return sum(1 for n in graph.nodes if len(n.parents) > 1)


# ------------------------------------------------------ the three filters
#
# Restoring the walls the DAG domain lacked, so that sideways trajectories can
# be asked whether they survive when anything can bite.
#
# Every filter here scales with *what is being reflected upon*, which is not a
# free choice — it is what the arithmetic domain did (cost was the symbol count
# of an address that encoded the theory) and what the box did. Reflecting on
# more costs more, takes longer to certify, and needs a bigger name.


@dataclass(frozen=True)
class Filters:
    """Economic, structural and epistemic constraints on a reflection step."""

    #: Symbols available per step. `None` disables the economic filter.
    budget: float | None = None
    #: Bits of address space. `None` disables the structural filter.
    address_bits: int | None = None
    #: Certification effort available per step. `None` disables the epistemic
    #: filter. A step is certifiable when `effort >= |key|`.
    certify_effort: int | None = None
    #: Maximum parents a step may join. `None` disables it. Unlike the three
    #: above this charges for **breadth** rather than depth, and it is the one
    #: candidate defence against sideways trajectories — included precisely so
    #: the pessimistic reading has something that could refute it.
    max_arity: int | None = None
    #: Addresses whose validity cannot be settled. A step whose key touches
    #: one is uncertifiable — the genuine epistemic wall, ported from
    #: `reflection.py`'s `searched` arm, where certification is a *search* that
    #: cannot conclude rather than a tax that scales with size.
    #:
    #: This exists because `certify_effort` turned out **not** to be an
    #: epistemic wall at all. It compares `cost(key)` against a bound, so under
    #: description pricing — where cost is constant — it degenerates to
    #: all-or-nothing, firing for every step or none regardless of the graph.
    #: It was a third size tax wearing the label. The distinction only became
    #: visible once the cost model was swapped, which is itself the point.
    opaque: frozenset[int] = frozenset()
    #: Opacity attached to the **form of the move** rather than to particular
    #: addresses. `"join"` marks every step joining two or more parents as
    #: uncertifiable, whatever it joins and wherever it sits.
    #:
    #: This is the reviewer's correction to `opaque`, and it is the case the
    #: mathematics actually forces. Totality is Π⁰₂-complete and `O`-membership
    #: Π¹₁-complete: the hardness is *uniform over the address space*, so nothing
    #: privileges a proper subset of addresses as decidable while the rest are
    #: not. Marking particular nodes opaque is an extra stipulation. When the
    #: opacity instead attaches to the form of the notation — a join names an
    #: arbitrary set, exactly as a limit names an arbitrary fundamental sequence
    #: — every move of that form is opaque at once, and there is no
    #: non-opaque move of that form left to detour to.
    opaque_form: str | None = None
    #: How a step is priced. `"content"` charges for the size of what is
    #: reflected on; `"description"` charges a flat rate.
    cost_model: str = "content"
    #: The flat rate, when pricing by description.
    description_cost: int = 2

    def cost(self, key: frozenset[int]) -> int:
        """What this step costs.

        `content` — the size of what it reflects on. This is what the
        arithmetic domain's `inline` presentation did (the address *was* the
        axiom list, so cost grew with the theory) and it is why advancing
        becomes the expensive move: each advance enlarges the frontier's
        closure, raising the price of the next.

        `description` — a flat rate, independent of what is reflected on. This
        is not a hypothetical: it is exactly what the `indexed` presentation
        measured in the very first experiment of this series, a constant 4,996
        symbols per rung however much theory it names. Naming by description
        rather than by content is what buys it.

        So the "cost model that does not grow with the reflected object" is not
        a new object to invent — it is the first result of the series, applied
        where the later domains had quietly reverted to content-addressing.
        """
        if self.cost_model == "description":
            return self.description_cost
        if self.cost_model != "content":
            raise ValueError(f"unknown cost model {self.cost_model!r}")
        return len(key)

    def admits(self, key: frozenset[int], address: int,
               arity: int = 1) -> tuple[bool, str | None]:
        """Does every filter let this step through, and if not, which bit?"""
        if self.budget is not None and self.cost(key) > self.budget:
            return False, "economic"
        if self.address_bits is not None:
            named = (self.description_cost if self.cost_model == "description"
                     else address)
            if named >= (1 << self.address_bits):
                return False, "structural"
        if self.certify_effort is not None and self.cost(key) > self.certify_effort:
            return False, "epistemic"
        if self.opaque and (key & self.opaque):
            return False, "uncertifiable"
        if self.opaque_form == "join" and arity > 1:
            return False, "uncertifiable"
        if self.opaque_form not in (None, "join"):
            raise ValueError(f"unknown opaque form {self.opaque_form!r}")
        if self.max_arity is not None and arity > self.max_arity:
            return False, "arity"
        return True, None


def filtered_step(graph: ReflectionGraph, parents: frozenset[int],
                  filters: Filters):
    """Attempt a reflection subject to the three filters.

    Returns `(step_or_None, blocked_by)`. A blocked step changes nothing — the
    filters refuse it before any content is added.
    """
    key = frozenset().union(*(graph.node(p).content for p in parents))
    address = max(key) if key else 0
    ok, blocked = filters.admits(key, address, arity=len(parents))
    if not ok:
        return None, blocked
    return reflect(graph, parents), None


def run_filtered(policy, steps: int, *, roots: int = 3, warmup: int = 5,
                 filters: Filters | None = None, first_id: int = 0) -> dict:
    """Run a policy under filters and count what got through and what advanced."""
    filters = filters or Filters()
    graph = ReflectionGraph.base(roots=roots, first_id=first_id)
    for _ in range(warmup):
        graph = reflect(graph, deepen(graph)).graph_after

    tally = {"advancing": 0, "sideways": 0, "duplicate": 0}
    blocks = {"economic": 0, "structural": 0, "epistemic": 0, "arity": 0,
              "uncertifiable": 0}
    for _ in range(steps):
        parents = policy(graph)
        s, blocked = filtered_step(graph, parents, filters)
        if s is None:
            blocks[blocked] += 1
            continue
        tally[s.kind] += 1
        graph = s.graph_after
    return {"tally": tally, "blocks": blocks, "final_rank": graph.rank,
            "final_size": graph.size, "joins": joins(graph),
            "passed": sum(tally.values()), "refused": sum(blocks.values())}


def run_options(policy, steps: int, *, roots: int = 3,
                filters: "Filters | None" = None) -> dict:
    """Trace how the option count moves, split by what kind of step moved it."""
    f = filters or Filters()
    graph = ReflectionGraph.base(roots=roots)
    deltas: dict[str, list[int]] = {}
    trace = []
    for _ in range(steps):
        before = options(graph, f)["total"]
        step = reflect(graph, policy(graph))
        graph = step.graph_after
        after = options(graph, f)["total"]
        deltas.setdefault(step.kind, []).append(after - before)
        trace.append({"kind": step.kind, "before": before, "after": after})
    return {"final": options(graph, f), "final_rank": graph.rank,
            "final_size": graph.size, "joins": joins(graph), "trace": trace,
            "mean_delta": {k: sum(v) / len(v) for k, v in deltas.items()},
            "counts": {k: len(v) for k, v in deltas.items()}}


def two_phase(invest_policy, invest_steps: int, horizon: int, *,
              roots: int = 3, opaque: int | None = None,
              retract: bool = False,
              cost_model: str = "description") -> int:
    """Invest, then have a wall land somewhere, then climb with what is left.

    The point of the two phases is that the wall arrives **after** the strategy
    is committed. Nothing in the interior view tells a system which of its
    lineages will turn out to be the unsettleable one — that was result five —
    so the honest comparison is over every placement, not a chosen one.
    """
    filters = Filters(cost_model=cost_model,
                      opaque=frozenset() if opaque is None else frozenset({opaque}))
    graph = ReflectionGraph.base(roots=roots)
    for _ in range(invest_steps):
        graph = reflect(graph, invest_policy(graph)).graph_after
    for _ in range(horizon):
        step, _ = filtered_step(graph, rank_aware(graph, filters), filters)
        if step is not None:
            graph = step.graph_after
    return certified_rank(graph, filters) if retract else graph.rank


def strategy_table(invest_steps: int, horizon: int, *, roots: int = 3,
                   retract: bool = False) -> dict:
    """Concentrate against diversify, over every place the wall could land."""
    out = {}
    for name, policy in (("concentrate", deepen), ("diversify", spread)):
        ranks = [two_phase(policy, invest_steps, horizon, roots=roots,
                           opaque=o, retract=retract) for o in range(roots)]
        out[name] = {"by_placement": ranks, "mean": sum(ranks) / len(ranks),
                     "worst": min(ranks), "best": max(ranks),
                     "no_wall": two_phase(policy, invest_steps, horizon,
                                          roots=roots, opaque=None,
                                          retract=retract)}
    return out


def rank_aware(graph: ReflectionGraph, filters: "Filters"):
    """Prefer the deepest **admissible** node; fall back to a join if refused.

    The reviewer's proposal: a selection term that does not read object size but
    reads the *rank order* instead. It asks for a maximal element whenever one
    is admissible, and only then settles for something else.

    Note what it is up against. The frontier is the most expensive move
    precisely *because* it is the frontier, and every advance grows the
    frontier's closure by one — so each success raises the price of the next.
    Whether that self-defeats is the measurement.
    """
    ordered = sorted(graph.nodes, key=lambda n: n.depth, reverse=True)
    for node in ordered:
        parents = frozenset({node.ident})
        key = node.content
        ok, _ = filters.admits(key, max(key), arity=1)
        if ok and key not in graph.asserted:
            return parents
    return broaden(graph)


def join_aware(filters: "Filters"):
    """A join-seeking policy that respects the filters, as `rank_aware` does for
    depth. Returns a `policy(graph)` closure.

    This exists because plain `broaden` re-offers a refused pair forever: it
    checks whether a join is *new*, not whether it is *admissible*. So a zero
    join count under `broaden` can mean the policy deadlocked rather than the
    wall removed the move class — and telling those apart is the whole point of
    asking whether opacity is local or uniform. When no admissible join is left
    it falls back to deepening, so the system does something rather than stall.
    """

    def policy(graph: ReflectionGraph) -> frozenset[int]:
        pool = graph.nodes
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                a, b = pool[i], pool[j]
                if a.content <= b.content or b.content <= a.content:
                    continue
                key = a.content | b.content
                if key in graph.asserted:
                    continue
                if filters.admits(key, max(key), arity=2)[0]:
                    return frozenset({a.ident, b.ident})
        return frozenset({graph.frontier()[0].ident})

    return policy


def run_adaptive(steps: int, *, roots: int = 3, warmup: int = 5,
                 filters: "Filters | None" = None, first_id: int = 0) -> dict:
    """Run the rank-aware policy, recording when it stops being able to advance."""
    filters = filters or Filters()
    graph = ReflectionGraph.base(roots=roots, first_id=first_id)
    for _ in range(warmup):
        graph = reflect(graph, deepen(graph)).graph_after

    tally = {"advancing": 0, "sideways": 0, "duplicate": 0}
    blocks = {"economic": 0, "structural": 0, "epistemic": 0, "arity": 0,
              "uncertifiable": 0}
    last_advance, rank_trace = None, []
    for i in range(steps):
        parents = rank_aware(graph, filters)
        s, blocked = filtered_step(graph, parents, filters)
        if s is None:
            blocks[blocked] += 1
            rank_trace.append(graph.rank)
            continue
        tally[s.kind] += 1
        if s.kind == "advancing":
            last_advance = i
        graph = s.graph_after
        rank_trace.append(graph.rank)
    return {"tally": tally, "blocks": blocks, "final_rank": graph.rank,
            "final_size": graph.size, "last_advance_at": last_advance,
            "rank_trace": rank_trace, "joins": joins(graph),
            "passed": sum(tally.values()), "refused": sum(blocks.values())}
