"""Tests for the three filters restored over the reflection DAG.

The claim is uncomfortable — that the walls meant to make `G` meaningful
preferentially block the move that advances — so it is pinned hard:

1. **The direction must never reverse.** Across every setting of every one of
   the three filters, sideways must never be blocked more than advancing. A
   single reversal would turn "counterproductive" into "sometimes helps", which
   is a different claim.

2. **The asymmetry must be structural, not tuned.** Advancing reflects on the
   frontier's closure; sideways joins shallow nodes. The size gap is a fact
   about the moves, so it is asserted directly rather than inferred from the
   pass rates it causes.

3. **The falsifier must actually fire.** An arity cap has to block sideways, or
   the pessimistic reading is unfalsifiable in this family and worth much less.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.reflection_dag import (
    Filters, ReflectionGraph, broaden, deepen, filtered_step, reflect,
    run_filtered,
)


def warmed(roots=3, warmup=5):
    g = ReflectionGraph.base(roots=roots)
    for _ in range(warmup):
        g = reflect(g, deepen(g)).graph_after
    return g


def passes(policy, f, steps=30):
    out = run_filtered(policy, steps, filters=f)
    return out["tally"]["advancing"] + out["tally"]["sideways"]


# --------------------------------------------- 1. the asymmetry is structural


class TestCostAsymmetry(unittest.TestCase):

    def test_advancing_reflects_on_strictly_more(self):
        g = warmed()
        adv = frozenset().union(*(g.node(p).content for p in deepen(g)))
        side = frozenset().union(*(g.node(p).content for p in broaden(g)))
        self.assertGreater(len(adv), len(side))
        self.assertGreater(max(adv), max(side))

    def test_the_gap_widens_as_the_graph_deepens(self):
        gaps = []
        for warmup in (3, 6, 9):
            g = warmed(warmup=warmup)
            adv = frozenset().union(*(g.node(p).content for p in deepen(g)))
            side = frozenset().union(*(g.node(p).content for p in broaden(g)))
            gaps.append(len(adv) - len(side))
        self.assertEqual(gaps, sorted(gaps))
        self.assertGreater(gaps[-1], gaps[0])

    def test_cost_is_the_size_of_what_is_reflected_on(self):
        f = Filters()
        self.assertEqual(f.cost(frozenset({1, 2, 3})), 3)
        self.assertEqual(f.cost(frozenset()), 0)


# ------------------------------------------- 2. the direction never reverses


class TestDirection(unittest.TestCase):

    def _never_reverses(self, make_filter, limits):
        for limit in limits:
            f = make_filter(limit)
            adv = run_filtered(deepen, 30, filters=f)["tally"]["advancing"]
            side = run_filtered(broaden, 30, filters=f)["tally"]["sideways"]
            self.assertLessEqual(adv, side,
                                 f"reversed at limit {limit}: {adv} vs {side}")

    def test_economic_never_blocks_sideways_more(self):
        self._never_reverses(lambda L: Filters(budget=L),
                             [3, 4, 5, 6, 8, 12, 20, 40])

    def test_structural_never_blocks_sideways_more(self):
        self._never_reverses(lambda L: Filters(address_bits=L),
                             [2, 3, 4, 5, 6, 8, 10])

    def test_epistemic_never_blocks_sideways_more(self):
        self._never_reverses(lambda L: Filters(certify_effort=L),
                             [3, 4, 5, 6, 8, 12, 20, 40])

    def test_each_filter_actually_bites_somewhere(self):
        """A filter that never constrains anything would pass the test above
        vacuously."""
        for f in (Filters(budget=5), Filters(address_bits=3),
                  Filters(certify_effort=5)):
            adv = run_filtered(deepen, 30, filters=f)["tally"]["advancing"]
            self.assertLess(adv, 30, f)

    def test_at_the_tightest_setting_advancing_is_shut_out_entirely(self):
        for f in (Filters(budget=5), Filters(certify_effort=5)):
            adv = run_filtered(deepen, 30, filters=f)
            side = run_filtered(broaden, 30, filters=f)
            self.assertEqual(adv["tally"]["advancing"], 0)
            self.assertGreater(side["tally"]["sideways"], 20)


# ------------------------------------------------- 3. the falsifier can fire


class TestArityDefence(unittest.TestCase):

    def test_an_arity_cap_blocks_sideways_completely(self):
        f = Filters(max_arity=1)
        self.assertEqual(run_filtered(broaden, 30,
                                      filters=f)["tally"]["sideways"], 0)

    def test_and_leaves_advancing_untouched(self):
        f = Filters(max_arity=1)
        out = run_filtered(deepen, 30, filters=f)
        self.assertEqual(out["tally"]["advancing"], 30)
        self.assertEqual(out["refused"], 0)

    def test_it_is_the_only_one_that_reverses_the_direction(self):
        reversing = []
        for name, f in (("economic", Filters(budget=5)),
                        ("structural", Filters(address_bits=3)),
                        ("epistemic", Filters(certify_effort=5)),
                        ("arity", Filters(max_arity=1))):
            adv = run_filtered(deepen, 30, filters=f)["tally"]["advancing"]
            side = run_filtered(broaden, 30, filters=f)["tally"]["sideways"]
            if side < adv:
                reversing.append(name)
        self.assertEqual(reversing, ["arity"])

    def test_arity_is_not_a_function_of_the_reflected_object(self):
        """Which is why it behaves differently: the other three all read the
        key, this one reads the parent count."""
        f = Filters(max_arity=1)
        g = warmed()
        small, blocked_small = filtered_step(g, frozenset({0}), f)
        pair, blocked_pair = filtered_step(g, frozenset({0, 1}), f)
        self.assertIsNotNone(small)
        self.assertIsNone(pair)
        self.assertEqual(blocked_pair, "arity")


# ------------------------------------------------- 4. filters are wired right


class TestFilterMechanics(unittest.TestCase):

    def test_no_filters_admits_everything(self):
        out = run_filtered(deepen, 20, filters=Filters())
        self.assertEqual(out["refused"], 0)

    def test_a_blocked_step_changes_nothing(self):
        g = warmed()
        before = g.size
        step, blocked = filtered_step(g, deepen(g), Filters(budget=1))
        self.assertIsNone(step)
        self.assertEqual(blocked, "economic")
        self.assertEqual(g.size, before)

    def test_blocks_are_attributed_to_the_right_filter(self):
        g = warmed()
        for f, expected in ((Filters(budget=1), "economic"),
                            (Filters(address_bits=1), "structural"),
                            (Filters(certify_effort=1), "epistemic")):
            _, blocked = filtered_step(g, deepen(g), f)
            self.assertEqual(blocked, expected)

    def test_economic_and_epistemic_are_the_same_predicate(self):
        """Stated in the experiment's honest scope and asserted here: their
        agreement is arithmetic, not independent evidence."""
        for limit in (3, 5, 8, 12, 20):
            a = run_filtered(deepen, 20, filters=Filters(budget=limit))
            b = run_filtered(deepen, 20, filters=Filters(certify_effort=limit))
            self.assertEqual(a["tally"], b["tally"], limit)


if __name__ == "__main__":
    unittest.main()
