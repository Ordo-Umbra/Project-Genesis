"""Tests for multi-parent reflection and the local/global split.

Two things must hold or the finding is an artifact:

1. **Sentence identity must be normalised to logical content.** `Con({T₁, T₂})`
   with `T₁` an ancestor of `T₂` has to be the *same sentence* as `Con({T₂})`.
   Without that, re-reflecting over a chain would look productive forever and
   the phenomenon under test would be manufactured by the representation.

2. **`sideways` must be a property of moves, not states.** If a state offering a
   sideways move offered no advancing one, this would be a sixth wall rather
   than a sixth observable, and the whole reading changes.

The single-root collapse is pinned too, because it came out stronger than the
prediction it was written to test and is the easiest thing to break by a
careless change to the content model.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.reflection_dag import (
    ReflectionGraph, broaden, deepen, reflect, run_policy,
)


def deepened(roots: int, steps: int) -> ReflectionGraph:
    g = ReflectionGraph.base(roots=roots)
    for _ in range(steps):
        g = reflect(g, deepen(g)).graph_after
    return g


# ------------------------------------------------- 1. content is normalised


class TestContentNormalisation(unittest.TestCase):

    def test_a_join_over_a_chain_is_the_same_sentence_as_its_maximum(self):
        g = deepened(1, 4)
        chain = [n.ident for n in g.nodes]
        for a in chain:
            for b in chain:
                if a == b:
                    continue
                joint = reflect(g, frozenset({a, b}))
                lone = reflect(g, frozenset({max(a, b)}))
                self.assertEqual(joint.key, lone.key, f"{a},{b}")

    def test_a_join_over_a_chain_adds_nothing_a_single_reflection_would_not(self):
        """The corrected claim. An earlier version asserted that every join over
        a chain is a *duplicate*, and that is false: joining the frontier with
        anything beneath it is productive. What is true — and is the property
        the finding rests on — is that such a join is **redundant**, carrying
        exactly the key of its maximum. It never says anything new."""
        g = deepened(1, 5)
        productive_joins = 0
        for a in range(g.size):
            for b in range(a + 1, g.size):
                joint = reflect(g, frozenset({a, b}))
                lone = reflect(g, frozenset({max(a, b)}))
                self.assertEqual(joint.key, lone.key)
                self.assertEqual(joint.productive, lone.productive)
                productive_joins += joint.productive
        self.assertGreater(productive_joins, 0,
                           "the point is redundancy, not sterility")

    def test_joins_strictly_below_the_frontier_are_duplicates(self):
        g = deepened(1, 5)
        top = max(n.ident for n in g.nodes)
        for a in range(top):
            for b in range(a + 1, top):
                self.assertFalse(reflect(g, frozenset({a, b})).productive)

    def test_a_node_is_about_itself_and_its_parents(self):
        g = ReflectionGraph.base(roots=1)
        s = reflect(g, frozenset({0}))
        self.assertTrue(s.productive)
        new = s.graph_after.nodes[-1]
        self.assertEqual(new.content, frozenset({0, new.ident}))
        self.assertEqual(s.key, frozenset({0}))

    def test_reflection_needs_a_parent(self):
        with self.assertRaises(ValueError):
            reflect(ReflectionGraph.base(), frozenset())


# ------------------------------------------ 2. the single-root collapse


class TestSingleRootCollapse(unittest.TestCase):

    def test_one_root_produces_nothing_under_joins(self):
        r = run_policy(broaden, 25, roots=1, warmup=5)
        self.assertEqual(r["tally"]["sideways"], 0)
        self.assertEqual(r["tally"]["advancing"], 0)
        self.assertEqual(r["tally"]["duplicate"], 25)

    def test_one_root_never_branches(self):
        """Branching means two nodes with *incomparable* contents. A chain has
        none however many parents its nodes nominally have."""
        for policy in (deepen, broaden):
            r = run_policy(policy, 20, roots=1)
            self.assertFalse(r["branching"], policy.__name__)

    def test_nominal_multiparent_is_not_branching(self):
        """The distinction that cost two failing assertions: a node can have two
        parents and contribute no incomparability at all."""
        r = run_policy(broaden, 20, roots=1)
        self.assertTrue(r["nominal_multiparent"])
        self.assertFalse(r["branching"])

    def test_branching_needs_independent_roots(self):
        self.assertFalse(run_policy(broaden, 20, roots=1, warmup=4)["branching"])
        self.assertTrue(run_policy(broaden, 20, roots=3, warmup=4)["branching"])


# ------------------------------------------------------ 3. the split is real


class TestLocalVersusGlobal(unittest.TestCase):

    def test_productive_steps_that_do_not_advance_exist(self):
        r = run_policy(broaden, 30, roots=3, warmup=5)
        self.assertGreater(r["tally"]["sideways"], 0)
        self.assertEqual(r["tally"]["advancing"], 0)
        for s in r["trace"]:
            if s["kind"] == "sideways":
                self.assertTrue(s["productive"])
                self.assertEqual(s["rank_before"], s["rank_after"])

    def test_a_long_run_stays_productive_and_never_advances(self):
        """The headline: productive forever, going nowhere."""
        r = run_policy(broaden, 150, roots=4, warmup=5)
        self.assertEqual(r["tally"]["duplicate"], 0)
        self.assertEqual(r["tally"]["sideways"], 150)
        self.assertEqual({s["rank_after"] for s in r["trace"]},
                         {r["final_rank"]})

    def test_the_same_budget_buys_very_different_reach(self):
        deep = run_policy(deepen, 30, roots=4, warmup=5)
        broad = run_policy(broaden, 30, roots=4, warmup=5)
        self.assertEqual(deep["final_size"], broad["final_size"])
        self.assertGreater(deep["final_rank"], broad["final_rank"] * 4)


# ------------------------------------- 4. it is a move property, not a state


class TestMoveNotState(unittest.TestCase):

    def test_one_state_offers_both_kinds_of_move(self):
        """The discriminator. Every earlier category is a fact about where a
        system is; this is a fact about what it chose."""
        for roots in (2, 3, 4):
            g = deepened(roots, 5)
            a = reflect(g, broaden(g))
            b = reflect(g, deepen(g))
            self.assertEqual(a.kind, "sideways", roots)
            self.assertEqual(b.kind, "advancing", roots)
            self.assertTrue(a.productive and b.productive)

    def test_sideways_is_not_a_wall(self):
        """Nothing blocks it — every check passes on a sideways step."""
        g = deepened(3, 5)
        s = reflect(g, broaden(g))
        self.assertTrue(s.productive)
        self.assertGreater(s.graph_after.size, g.size)
        self.assertEqual(s.rank_after, s.rank_before)

    def test_switching_policy_mid_run_recovers_advance(self):
        g = deepened(3, 4)
        for _ in range(10):
            g = reflect(g, broaden(g)).graph_after
        flat = g.rank
        g = reflect(g, deepen(g)).graph_after
        self.assertEqual(g.rank, flat + 1)


if __name__ == "__main__":
    unittest.main()
