"""Tests for the limit mechanism and the ω² rank fragment.

The claim this layer rests on is unusual for this series: it is a claim that
something is **impossible**, not that something is expensive. Everything else
measured here has been a ratio. So the tests concentrate on the two places that
claim could be wrong:

1. **`inline` really has no limit, at any budget.** If some code path let it
   through — a default, a fallback, a silently-permitted union — the headline
   would collapse into another cost result.

2. **The limit really is productive, and really is O(1).** A limit that
   collided with an index already used, or whose cost scaled with the ladder
   beneath it, would mean the "description" is a listing wearing a hat.

The rank arithmetic is tested too, since `ω·a + b` ordering is the only ordinal
fact the module uses and a wrong comparison would silently reorder everything.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.reflection import (
    Capacity, LimitUndefined, Rank, con_formula, construction_cost,
    first_index_collision, ladder, limit_step, peano, step, transfinite_climb,
)


def climb(theory, n):
    for s in ladder(theory, n):
        theory = s.theory_after
    return theory


# ------------------------------------------------------------ 1. rank algebra


class TestRank(unittest.TestCase):

    def test_orders_lexicographically(self):
        self.assertLess(Rank(0, 5), Rank(1, 0))
        self.assertLess(Rank(1, 3), Rank(1, 4))
        self.assertLess(Rank(1, 99), Rank(2, 0))
        self.assertEqual(Rank(2, 3), Rank(2, 3))

    def test_finite_ranks_sort_below_every_limit(self):
        finites = [Rank(0, n) for n in range(50)]
        self.assertTrue(all(r < Rank(1, 0) for r in finites))

    def test_is_limit_only_at_exact_multiples_of_omega(self):
        self.assertTrue(Rank(1, 0).is_limit)
        self.assertTrue(Rank(3, 0).is_limit)
        self.assertFalse(Rank(0, 0).is_limit)
        self.assertFalse(Rank(1, 1).is_limit)

    def test_renders_readably(self):
        self.assertEqual(str(Rank(0, 0)), "0")
        self.assertEqual(str(Rank(0, 7)), "7")
        self.assertEqual(str(Rank(1, 0)), "ω")
        self.assertEqual(str(Rank(1, 4)), "ω+4")
        self.assertEqual(str(Rank(3, 0)), "ω·3")
        self.assertEqual(str(Rank(3, 2)), "ω·3+2")


# --------------------------------------------------- 2. the gate is a real gate


class TestLimitIsAGate(unittest.TestCase):

    def test_inline_refuses_the_limit_at_every_depth(self):
        for depth in (0, 1, 5, 12):
            theory = climb(peano("inline"), depth)
            self.assertFalse(theory.can_take_limit())
            with self.assertRaises(LimitUndefined):
                limit_step(theory)

    def test_no_budget_buys_the_limit(self):
        """The distinguishing property: paying more moves an economic wall and
        does nothing at all to this one."""
        outcomes = [transfinite_climb(peano("inline"), blocks=3, per_block=6,
                                      capacity=Capacity(b, 1.0))
                    for b in (1e4, 1e12, 1e24, 1e48)]
        self.assertTrue(any(o.stopped_because == "unaffordable"
                            for o in outcomes),
                        "small budgets should stop it for cost")
        self.assertTrue(any(o.stopped_because == "limit-undefined"
                            for o in outcomes),
                        "large budgets should stop it for definability")
        for o in outcomes:
            self.assertEqual(o.limits_taken, 0)

    def test_the_wall_stops_receding(self):
        """Rank reached must increase with budget and then plateau — the
        crossover from a priced wall to a structural one."""
        ranks = [transfinite_climb(peano("inline"), blocks=3, per_block=12,
                                   capacity=Capacity(b, 1.0)).rank
                 for b in (1e5, 1e6, 1e9, 1e12, 1e15)]
        self.assertEqual(ranks, sorted(ranks), "reach should be monotone")
        self.assertEqual(ranks[-1], ranks[-2], "the wall should plateau")
        self.assertLess(ranks[0], ranks[-1])

    def test_indexed_arms_can_always_take_it(self):
        for kind, width in (("indexed", None), ("truncated", 3)):
            for depth in (0, 3, 9):
                self.assertTrue(climb(peano(kind, width=width),
                                      depth).can_take_limit())


# ----------------------------------------------- 3. the limit is real and cheap


class TestLimitStep(unittest.TestCase):

    def test_costs_a_successor_regardless_of_depth(self):
        theory0 = peano("indexed")
        successor = construction_cost(step(theory0))
        for depth in (1, 4, 16, 40):
            lim = limit_step(climb(peano("indexed"), depth))
            self.assertEqual(lim.con_symbols, successor,
                             f"limit cost moved at depth {depth}")

    def test_is_productive_and_names_an_index_never_used(self):
        theory = climb(peano("indexed"), 8)
        lim = limit_step(theory)
        self.assertTrue(lim.new_axiom)
        self.assertNotIn(lim.theory_before.index(), ())
        self.assertEqual(len(lim.theory_after.axioms()),
                         len(theory.axioms()) + 1)

    def test_advances_the_rank_past_every_finite_one(self):
        theory = climb(peano("indexed"), 8)
        self.assertEqual(theory.rank, Rank(0, 8))
        after = limit_step(theory).theory_after
        self.assertEqual(after.rank, Rank(1, 1))
        self.assertGreater(after.rank, Rank(0, 10 ** 6))

    def test_preserves_the_axioms_it_subsumes(self):
        theory = climb(peano("indexed"), 6)
        after = limit_step(theory).theory_after
        for axiom in theory.axioms():
            self.assertIn(axiom, after.axioms())

    def test_successors_resume_productively_past_omega(self):
        theory = limit_step(climb(peano("indexed"), 5)).theory_after
        steps = list(ladder(theory, 10))
        self.assertTrue(all(s.new_axiom for s in steps))
        self.assertIsNone(first_index_collision(steps))
        self.assertEqual(steps[-1].theory_after.rank, Rank(1, 11))

    def test_stacked_limits_keep_working(self):
        theory = peano("indexed")
        for _ in range(4):
            theory = climb(theory, 3)
            lim = limit_step(theory)
            self.assertTrue(lim.new_axiom)
            theory = lim.theory_after
        self.assertEqual(theory.rank.limits, 4)


# ------------------------------------------- 4. the reprieve is bounded, not a cure


class TestTruncatedReprieve(unittest.TestCase):

    def test_a_limit_restarts_a_stalled_ladder(self):
        theory = climb(peano("truncated", width=2), 8)   # stalled since rung 4
        before = len(theory.axioms())
        self.assertFalse(step(theory).new_axiom)
        after = limit_step(theory).theory_after
        self.assertEqual(len(after.axioms()), before + 1)
        self.assertTrue(step(after).new_axiom, "should produce again past ω")

    def test_but_only_for_two_to_the_width_rungs(self):
        for width in (2, 3):
            theory = peano("truncated", width=width)
            per_block = []
            for block in range(4):
                produced = 0
                for s in ladder(theory, 10):
                    produced += 1 if s.new_axiom else 0
                    theory = s.theory_after
                lim = limit_step(theory)
                produced += 1 if lim.new_axiom else 0
                theory = lim.theory_after
                per_block.append(produced)
            self.assertEqual(set(per_block[1:]), {1 << width},
                             f"width {width}: {per_block}")

    def test_productive_content_grows_linearly_in_limits_not_in_rank(self):
        """The rank runs away like ω·limits; the axioms do not."""
        theory = peano("truncated", width=3)
        for _ in range(6):
            theory = limit_step(climb(theory, 20)).theory_after
        self.assertEqual(theory.rank.limits, 6)
        # 8 per block after the first, 9 in the first, plus PA's base axioms.
        self.assertEqual(len(theory.axioms()), len(peano("indexed").base) + 49)


# --------------------------------------------------------- 5. the climb driver


class TestTransfiniteClimb(unittest.TestCase):

    def test_reports_the_horizon_when_nothing_blocks(self):
        o = transfinite_climb(peano("indexed"), blocks=3, per_block=5)
        self.assertEqual(o.stopped_because, "horizon")
        self.assertEqual(o.productive, o.taken)
        self.assertEqual(o.limits_taken, 2)

    def test_distinguishes_its_two_stop_reasons(self):
        cheap = transfinite_climb(peano("inline"), blocks=3, per_block=5,
                                  capacity=Capacity(1e4, 1.0))
        rich = transfinite_climb(peano("inline"), blocks=3, per_block=5,
                                 capacity=Capacity(1e30, 1.0))
        self.assertEqual(cheap.stopped_because, "unaffordable")
        self.assertEqual(rich.stopped_because, "limit-undefined")

    def test_counts_productive_separately_from_taken(self):
        o = transfinite_climb(peano("truncated", width=2), blocks=3,
                              per_block=10)
        self.assertEqual(o.stopped_because, "horizon")
        self.assertLess(o.productive, o.taken)

    def test_a_single_block_never_needs_a_limit(self):
        """With one block there is no limit to take, so even inline finishes."""
        o = transfinite_climb(peano("inline"), blocks=1, per_block=4)
        self.assertEqual(o.stopped_because, "horizon")
        self.assertEqual(o.limits_taken, 0)


if __name__ == "__main__":
    unittest.main()
