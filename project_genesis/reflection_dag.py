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

    @classmethod
    def base(cls, roots: int = 1) -> "ReflectionGraph":
        """Start from `roots` mutually independent theories.

        `roots = 1` is the free generation from a single base, and it is worth
        running: it collapses to a chain, because every join over a chain has
        the union of a nested family as its key, which is the largest member's
        content and is therefore already asserted. Branching — and so everything
        this domain exists to test — requires more than one starting point.
        """
        nodes = tuple(Node(ident=i, parents=frozenset(),
                           content=frozenset({i}), depth=0)
                      for i in range(roots))
        return cls(nodes=nodes, asserted=frozenset())

    def node(self, ident: int) -> Node:
        return self.nodes[ident]

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
        ident = graph.size
        node = Node(ident=ident, parents=parents, content=key | {ident},
                    depth=depth)
        after = replace(graph, nodes=graph.nodes + (node,),
                        asserted=graph.asserted | {key})
    else:
        after = graph
    return DagStep(parents=parents, key=key, depth=depth,
                   productive=productive, rank_before=rank_before,
                   rank_after=after.rank, graph_after=after)


# --------------------------------------------------------------- strategies
#
# How a system chooses what to reflect on. Not a naming scheme — a content set
# names itself here — but a *policy*, and it is the policy that decides whether
# a productive step also advances the collection.


def deepen(graph: ReflectionGraph) -> frozenset[int]:
    """Always reflect on the deepest node: the linear ladder, recovered."""
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
