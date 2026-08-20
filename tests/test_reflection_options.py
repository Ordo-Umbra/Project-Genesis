"""Tests for option counting, and for what options are and are not good for.

`rank` measures height only, so a move that widens the base scores zero on it.
This module asks whether that hides a real capability. Four things have to hold:

1. **The option count must be a count of real moves.** If it counted moves that
   are not available — already asserted, or refused by the filters — the whole
   measure would be inflated and every downstream comparison meaningless.

2. **A chain must generate no join options.** That is the normalisation in
   `reflect` doing its job: over a totally ordered collection every union is the
   larger member. If a chain produced join options, the domain's central claim
   about redundancy would be false.

3. **The registered prediction was wrong, and the correction is pinned.** Option
   generation does not track the advancing/sideways label — it tracks whether a
   move creates *incomparable material*. `spread` is the case that separates
   them: every move it makes advances, and it generates options at the sideways
   rate. That is asserted here so the corrected claim cannot silently rot back.

4. **Options must not convert into height, and the insurance trade must be a
   trade.** If breadth-investment bought rank, `sideways` would have been an
   investment all along. If one strategy dominated on both mean and worst case,
   there would be no trade to report.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.reflection_dag import (
    Filters, ReflectionGraph, broaden, certified_rank, deepen, options, reflect,
    run_options, spread, strategy_table, two_phase,
)


def grown(policy, steps, roots=3):
    g = ReflectionGraph.base(roots=roots)
    for _ in range(steps):
        g = reflect(g, policy(g)).graph_after
    return g


# ------------------------------------------------- 1. the count is of real moves


class TestOptionCount(unittest.TestCase):

    def test_a_fresh_graph_offers_each_root_and_each_pair(self):
        g = ReflectionGraph.base(roots=3)
        o = options(g)
        self.assertEqual(o["single"], 3)
        self.assertEqual(o["join"], 3)          # 3 choose 2, all incomparable
        self.assertEqual(o["total"], 6)

    def test_a_taken_move_is_not_offered_again(self):
        g = ReflectionGraph.base(roots=3)
        self.assertNotIn(frozenset({0, 1}), g.asserted)
        g = reflect(g, frozenset({0, 1})).graph_after
        self.assertIn(frozenset({0, 1}), g.asserted)
        for a, b in ((0, 1), (1, 0)):
            key = g.node(a).content | g.node(b).content
            self.assertIn(key, g.asserted)

    def test_reflecting_on_a_node_consumes_its_singleton_option(self):
        g = ReflectionGraph.base(roots=3)
        self.assertEqual(options(g)["single"], 3)
        g = reflect(g, frozenset({0})).graph_after
        singles = [n for n in g.nodes if n.content not in g.asserted]
        self.assertNotIn(0, [n.ident for n in singles])

    def test_it_never_counts_a_move_the_filters_refuse(self):
        g = grown(spread, 12)
        wide = options(g, Filters())["total"]
        shut = options(g, Filters(budget=0))["total"]
        self.assertGreater(wide, 0)
        self.assertEqual(shut, 0)

    def test_form_opacity_removes_exactly_the_join_options(self):
        g = grown(spread, 12)
        base = options(g, Filters())
        opaque = options(g, Filters(opaque_form="join"))
        self.assertEqual(opaque["join"], 0)
        self.assertEqual(opaque["single"], base["single"])

    def test_the_default_filter_is_permissive(self):
        g = grown(spread, 10)
        self.assertEqual(options(g), options(g, Filters()))


# ------------------------------------------------------ 2. a chain has no joins


class TestChainsGenerateNothing(unittest.TestCase):

    def test_free_generation_from_one_root_stays_a_chain(self):
        g = grown(deepen, 25, roots=1)
        self.assertTrue(g.is_chain())
        self.assertEqual(options(g)["join"], 0)

    def test_and_its_option_count_never_grows(self):
        counts = [options(grown(deepen, n, roots=1))["total"] for n in (5, 15, 25)]
        self.assertEqual(set(counts), {1})

    def test_independent_roots_are_the_precondition(self):
        self.assertGreater(options(grown(deepen, 25, roots=3))["join"], 0)


# ------------------------------- 3. the registered prediction, and its correction


class TestWhatActuallyGeneratesOptions(unittest.TestCase):

    def setUp(self):
        self.rates = {name: run_options(p, 25)["mean_delta"]
                      for name, p in (("deepen", deepen), ("spread", spread),
                                      ("broaden", broaden))}

    def test_chain_extension_generates_almost_nothing(self):
        self.assertLess(self.rates["deepen"]["advancing"], 3)

    def test_joining_below_the_frontier_generates_many(self):
        self.assertGreater(self.rates["broaden"]["sideways"],
                           4 * self.rates["deepen"]["advancing"])

    def test_the_label_is_not_the_variable(self):
        """The correction. `spread` only ever advances, and generates options at
        the sideways rate — so the advancing/sideways axis does not explain the
        count. Incomparability does."""
        spread_adv = self.rates["spread"]["advancing"]
        self.assertLess(abs(spread_adv - self.rates["broaden"]["sideways"]),
                        abs(spread_adv - self.rates["deepen"]["advancing"]))

    def test_a_policy_can_advance_and_open_options_at_once(self):
        r = run_options(spread, 25)
        self.assertGreater(r["counts"]["advancing"], 0)
        self.assertGreater(r["final"]["join"], options(grown(deepen, 25))["join"])


# --------------------------------- 4. options are not height; the trade is real


class TestOptionsDoNotConvert(unittest.TestCase):

    def test_every_strategy_gains_the_same_rank_in_the_climb(self):
        gains = set()
        for policy in (deepen, spread, broaden):
            before = grown(policy, 20).rank
            gains.add(two_phase(policy, 20, 40, opaque=None) - before)
        self.assertEqual(len(gains), 1, "width should buy no height at all")

    def test_breadth_leaves_more_options_and_less_height(self):
        wide, tall = grown(spread, 20), grown(deepen, 20)
        self.assertGreater(options(wide)["total"], options(tall)["total"])
        self.assertLess(wide.rank, tall.rank)


class TestTheInsuranceTrade(unittest.TestCase):

    def test_concentration_wins_the_mean_everywhere_tested(self):
        for retract in (False, True):
            for hz in (5, 20, 40):
                t = strategy_table(20, hz, retract=retract)
                self.assertGreaterEqual(t["concentrate"]["mean"],
                                        t["diversify"]["mean"], (retract, hz))

    def test_diversification_wins_the_floor_at_a_long_horizon(self):
        t = strategy_table(20, 40)
        self.assertGreater(t["diversify"]["worst"], t["concentrate"]["worst"])

    def test_the_premium_only_shows_at_a_short_horizon_when_walls_retract(self):
        """The load-bearing modelling choice, stated as an assertion: under
        `freeze` the concentrated strategy keeps the height it already reached,
        so it wins the floor too. Under `retract` that height is uncertified and
        the floor collapses to what it can rebuild in the time left."""
        freeze = strategy_table(20, 5, retract=False)
        retract = strategy_table(20, 5, retract=True)
        self.assertGreater(freeze["concentrate"]["worst"],
                           freeze["diversify"]["worst"])
        self.assertLess(retract["concentrate"]["worst"],
                        retract["diversify"]["worst"])

    def test_diversification_has_no_spread_across_placements(self):
        """Which is what makes it insurance: the outcome does not depend on
        which lineage turns out to be the unsettleable one."""
        t = strategy_table(20, 20, retract=True)
        self.assertEqual(len(set(t["diversify"]["by_placement"])), 1)
        self.assertGreater(len(set(t["concentrate"]["by_placement"])), 1)

    def test_certified_rank_discounts_a_tower_over_an_opaque_base(self):
        g = grown(deepen, 10, roots=3)
        clean, opaque = Filters(), Filters(opaque=frozenset({0}))
        self.assertEqual(certified_rank(g, clean), g.rank)
        self.assertLess(certified_rank(g, opaque), g.rank)


if __name__ == "__main__":
    unittest.main()
