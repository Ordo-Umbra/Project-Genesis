"""Tests for description-addressed pricing.

The claim is that the "missing cost model" was already measured in result one —
the `indexed` presentation's flat per-rung cost — and that the later domains had
reverted to content-addressing. Three things must hold:

1. **The cost model must actually not read the key.** If `description` pricing
   ever varied with what a step reflects on, the whole result would be a
   relabelling.

2. **Saturation must disappear, not merely recede.** Rank has to track steps
   without bound, at every budget. A larger ceiling would not be the finding.

3. **The basin must survive.** Sideways moves have to remain *available* — the
   claim is that they stop being downhill, not that they stop existing. If they
   vanished, the earlier result (sideways is a property of the move, not the
   state) would be contradicted.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.reflection_dag import (
    Filters, ReflectionGraph, broaden, deepen, reflect, run_adaptive,
    run_filtered,
)


def warmed(roots=3, warmup=5):
    g = ReflectionGraph.base(roots=roots)
    for _ in range(warmup):
        g = reflect(g, deepen(g)).graph_after
    return g


# ------------------------------------------------- 1. the price ignores the key


class TestCostModel(unittest.TestCase):

    def test_description_cost_does_not_read_the_key(self):
        f = Filters(cost_model="description", description_cost=7)
        for key in (frozenset({0}), frozenset(range(50)), frozenset(range(500))):
            self.assertEqual(f.cost(key), 7)

    def test_content_cost_does_read_the_key(self):
        f = Filters(cost_model="content")
        self.assertEqual(f.cost(frozenset(range(9))), 9)
        self.assertEqual(f.cost(frozenset(range(2))), 2)

    def test_an_unknown_cost_model_is_refused(self):
        with self.assertRaises(ValueError):
            Filters(cost_model="vibes").cost(frozenset({1}))

    def test_content_is_the_default(self):
        """So that every earlier result keeps the pricing it was measured under."""
        self.assertEqual(Filters().cost_model, "content")
        self.assertEqual(Filters().cost(frozenset(range(4))), 4)


# ------------------------------------------------------- 2. saturation is gone


class TestNoSaturation(unittest.TestCase):

    def test_rank_tracks_steps_without_bound(self):
        ranks = [run_adaptive(n, filters=Filters(budget=10,
                                                 cost_model="description"))
                 ["final_rank"] for n in (20, 40, 80, 160)]
        self.assertEqual(len(set(ranks)), 4)
        self.assertEqual(ranks, sorted(ranks))
        for a, b in zip(ranks, ranks[1:]):
            self.assertGreater(b, a)

    def test_content_pricing_still_saturates_at_the_budget(self):
        """The control: the earlier result must not have quietly changed."""
        for b in (6, 10, 20, 50):
            r = run_adaptive(60, filters=Filters(budget=b))
            self.assertEqual(r["final_rank"], b)

    def test_the_budget_stops_capping_reach(self):
        ranks = {run_adaptive(60, filters=Filters(budget=b,
                                                  cost_model="description"))
                 ["final_rank"] for b in (6, 10, 20, 50)}
        self.assertEqual(len(ranks), 1,
                         "reach should no longer depend on the budget")

    def test_every_step_advances_under_description_pricing(self):
        r = run_adaptive(40, filters=Filters(budget=10,
                                             cost_model="description"))
        self.assertEqual(r["tally"]["advancing"], 40)
        self.assertEqual(r["tally"]["sideways"], 0)

    def test_a_budget_below_the_flat_rate_still_blocks_everything(self):
        """Flat is not free — the filter must still be able to bite."""
        r = run_adaptive(20, filters=Filters(budget=1, cost_model="description",
                                             description_cost=5))
        self.assertEqual(r["tally"]["advancing"], 0)
        self.assertEqual(r["refused"], 20)


# ------------------------------------------------- 3. the filters go neutral


class TestNeutrality(unittest.TestCase):

    def _verdict(self, f):
        a = run_filtered(deepen, 30, filters=f)["tally"]["advancing"]
        s = run_filtered(broaden, 30, filters=f)["tally"]["sideways"]
        return a, s

    def test_economic_is_neutral_under_description_pricing(self):
        for limit in (4, 5, 8):
            a, s = self._verdict(Filters(budget=limit,
                                         cost_model="description"))
            self.assertEqual(a, s, f"limit {limit}")

    def test_epistemic_is_neutral_under_description_pricing(self):
        for limit in (4, 5, 8):
            a, s = self._verdict(Filters(certify_effort=limit,
                                         cost_model="description"))
            self.assertEqual(a, s, f"limit {limit}")

    def test_structural_is_neutral_under_description_pricing(self):
        for limit in (2, 3, 4):
            a, s = self._verdict(Filters(address_bits=limit,
                                         cost_model="description"))
            self.assertEqual(a, s, f"limit {limit}")

    def test_all_three_are_directional_under_content_pricing(self):
        """The control again: the bias must still be there when the cost model
        is the one that produced it."""
        for f in (Filters(budget=5), Filters(certify_effort=5),
                  Filters(address_bits=3)):
            a, s = self._verdict(f)
            self.assertLess(a, s, f)


# ------------------------------------------------------ 4. the basin survives


class TestBasinSurvives(unittest.TestCase):

    def test_sideways_is_still_available(self):
        """The claim is that it stops being downhill, not that it disappears."""
        out = run_filtered(broaden, 30, filters=Filters(cost_model="description"))
        self.assertGreater(out["tally"]["sideways"], 0)

    def test_a_join_seeking_policy_still_goes_nowhere_if_it_wants_to(self):
        out = run_filtered(broaden, 60, filters=Filters(budget=10,
                                                        cost_model="description"))
        self.assertGreater(out["tally"]["sideways"], 40)
        self.assertEqual(out["tally"]["advancing"], 0)

    def test_but_a_rank_aware_policy_no_longer_falls_into_it(self):
        content = run_adaptive(60, filters=Filters(budget=10))
        descr = run_adaptive(60, filters=Filters(budget=10,
                                                 cost_model="description"))
        self.assertGreater(content["tally"]["sideways"], 40)
        self.assertEqual(descr["tally"]["sideways"], 0)


if __name__ == "__main__":
    unittest.main()
