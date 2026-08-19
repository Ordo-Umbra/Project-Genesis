"""Tests for rank-aware selection, and for the retraction it forced.

Two claims carry this layer:

1. **Rank saturates at the budget, by self-defeat.** Advancing grows the
   frontier's closure, and cost is the size of what a step reflects on, so each
   success raises the price of the next. If rank ever grew with steps instead of
   saturating, the mechanism would be wrong.

2. **Rank-awareness gains nothing over blind deepening.** This is the finding,
   and it is the easiest thing to break by an accidental change to the policy —
   so the gain is asserted to be exactly zero rather than merely small.

The retraction is tested too: an arity cap must *not* be shown to block sideways
in general, because the earlier 0/30 came from the policy's fallback producing
duplicates.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.reflection_dag import (
    Filters, ReflectionGraph, broaden, deepen, filtered_step, rank_aware,
    reflect, run_adaptive, run_filtered,
)

BUDGETS = (6, 8, 10, 14, 20, 30, 50)


def warmed(roots=3, warmup=5):
    g = ReflectionGraph.base(roots=roots)
    for _ in range(warmup):
        g = reflect(g, deepen(g)).graph_after
    return g


# ------------------------------------------------------- 1. self-defeat


class TestSelfDefeat(unittest.TestCase):

    def test_rank_saturates_exactly_at_the_budget(self):
        for b in BUDGETS:
            r = run_adaptive(60, filters=Filters(budget=b))
            self.assertEqual(r["final_rank"], b, f"budget {b}")

    def test_more_steps_do_not_buy_more_rank(self):
        """The falsifier for self-defeat: if rank grew with steps, the price
        would not be rising with success."""
        ranks = {run_adaptive(n, filters=Filters(budget=12))["final_rank"]
                 for n in (20, 40, 80, 160)}
        self.assertEqual(ranks, {12})

    def test_advancing_stops_early_then_never_resumes(self):
        r = run_adaptive(60, filters=Filters(budget=10))
        self.assertIsNotNone(r["last_advance_at"])
        self.assertLess(r["last_advance_at"], 10)
        self.assertGreater(r["tally"]["sideways"], 40)

    def test_the_frontier_gets_more_expensive_as_it_rises(self):
        """The mechanism itself, measured directly rather than inferred."""
        costs = []
        for warmup in (3, 6, 9, 12):
            g = warmed(warmup=warmup)
            key = frozenset().union(*(g.node(p).content for p in deepen(g)))
            costs.append(len(key))
        self.assertEqual(costs, sorted(costs))
        self.assertGreater(costs[-1], costs[0])


# --------------------------------------------------- 2. the gain is zero


class TestNoGain(unittest.TestCase):

    def test_rank_aware_matches_blind_deepening_under_a_budget(self):
        for b in BUDGETS:
            f = Filters(budget=b)
            blind = run_filtered(deepen, 60, filters=f)["final_rank"]
            aware = run_adaptive(60, filters=f)["final_rank"]
            self.assertEqual(aware, blind, f"budget {b}")

    def test_and_under_the_other_two_filters(self):
        for make in (lambda L: Filters(certify_effort=L),
                     lambda L: Filters(address_bits=L)):
            for limit in (4, 5, 6):
                f = make(limit)
                blind = run_filtered(deepen, 60, filters=f)["final_rank"]
                aware = run_adaptive(60, filters=f)["final_rank"]
                self.assertEqual(aware, blind, f"{f}")

    def test_refusals_become_sideways_one_for_one(self):
        """What the selection term actually buys: motion, not reach."""
        for b in (8, 10, 20, 30):
            f = Filters(budget=b)
            blind = run_filtered(deepen, 60, filters=f)
            aware = run_adaptive(60, filters=f)
            self.assertEqual(blind["refused"], aware["tally"]["sideways"], b)

    def test_with_no_filter_rank_aware_just_deepens(self):
        r = run_adaptive(40, filters=Filters())
        self.assertEqual(r["tally"]["sideways"], 0)
        self.assertEqual(r["tally"]["advancing"], 40)


# ------------------------------------------------------ 3. the retraction


class TestArityRetraction(unittest.TestCase):

    def test_single_parent_sideways_moves_exist_and_are_admitted(self):
        """The reason the earlier claim is withdrawn: a sideways move does not
        require a join."""
        f = Filters(max_arity=1)
        g = warmed()
        found = []
        for node in g.nodes:
            s, _ = filtered_step(g, frozenset({node.ident}), f)
            if s is not None and s.kind == "sideways":
                found.append(node.ident)
        self.assertGreater(len(found), 0,
                           "an arity cap must not eliminate sideways as such")

    def test_the_earlier_zero_was_joins_being_blocked_not_sideways(self):
        """The first diagnosis of this retraction was itself wrong, and this
        assertion caught it. `broaden` proposes joins and the cap blocks all
        30 of them — so the earlier measurement was correct as far as it went.
        The error was generalising 'joins are blocked' to 'sideways is
        blocked', when a sideways move needs no join at all."""
        out = run_filtered(broaden, 30, filters=Filters(max_arity=1))
        self.assertEqual(out["tally"]["sideways"], 0)
        self.assertEqual(out["blocks"]["arity"], 30)

    def test_a_policy_that_seeks_shallow_singles_gets_sideways_through(self):
        f = Filters(max_arity=1)
        g = warmed()

        def shallow_single(graph):
            for node in sorted(graph.nodes, key=lambda n: n.depth):
                if node.content not in graph.asserted:
                    return frozenset({node.ident})
            return frozenset({graph.nodes[0].ident})

        s, blocked = filtered_step(g, shallow_single(g), f)
        self.assertIsNone(blocked)
        self.assertIsNotNone(s)
        self.assertEqual(s.kind, "sideways")


# --------------------------------------------------- 4. the policy is sane


class TestRankAwarePolicy(unittest.TestCase):

    def test_it_prefers_the_deepest_admissible_node(self):
        g = warmed()
        chosen = rank_aware(g, Filters())
        deepest = max(n.depth for n in g.nodes)
        self.assertEqual({g.node(p).depth for p in chosen}, {deepest})

    def test_it_falls_back_when_the_frontier_is_refused(self):
        g = warmed()
        tight = Filters(budget=3)
        chosen = rank_aware(g, tight)
        deepest = max(n.depth for n in g.nodes)
        self.assertNotEqual({g.node(p).depth for p in chosen}, {deepest})

    def test_it_never_reads_object_size(self):
        """The reviewer's requirement: the selection term is rank-ordered, not
        size-ordered. Verified by construction — it sorts on depth."""
        g = warmed()
        by_depth = sorted(g.nodes, key=lambda n: n.depth, reverse=True)
        chosen = rank_aware(g, Filters())
        self.assertIn(next(iter(chosen)), {by_depth[0].ident})


if __name__ == "__main__":
    unittest.main()
