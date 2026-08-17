"""Tests for the interior view: what a theory determines about its own walls.

This is the first layer whose claims are about a system's *self*-knowledge, so
the tests guard two things the conclusion cannot survive without:

1. **The prediction must not consult the run.** `predict_stop` uses only checks
   a theory can run on its own presentation. If it ever read the outcome of a
   climb, "the interior view is exact" would be circular and empty. The tests
   below pin the agreement at every arm and budget, *and* pin the ordering rule
   that made it exact — walls resolved as they would be met.

2. **The third wall must stay genuinely different.** The `searched` arm has to
   be unable to settle its own edge, while that edge is really there. If its
   verdict ever became decided, the third wall would collapse into one of the
   first two and the whole distinction would go with it.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.reflection import (
    Capacity, LimitUndefined, limit_step, peano, predict_stop,
    transfinite_climb, verify_searched_notation,
)

ARMS = (("inline", None), ("indexed", None), ("truncated", 3),
        ("searched", None))
BUDGETS = (None, 1e5, 1e8, 1e12)


def both(kind, width, budget, blocks=4, per_block=10):
    theory = peano(kind, width=width)
    cap = None if budget is None else Capacity(budget, 1.0)
    p = predict_stop(theory, blocks=blocks, per_block=per_block, capacity=cap)
    o = transfinite_climb(theory, blocks=blocks, per_block=per_block,
                          capacity=cap)
    return p, o


# ------------------------------------------------- 1. the interior is exact


class TestPredictionMatchesTheRun(unittest.TestCase):

    def test_where_is_exact_for_every_arm_and_budget(self):
        for kind, width in ARMS:
            for budget in BUDGETS:
                p, o = both(kind, width, budget)
                expected = None if o.stopped_because == "horizon" else o.taken
                self.assertEqual(p.stop_rung, expected,
                                 f"{kind} at {budget}: {p.detail}")

    def test_why_is_exact_wherever_it_is_decided(self):
        for kind, width in ARMS:
            for budget in BUDGETS:
                p, o = both(kind, width, budget)
                self.assertEqual(p.reason, o.stopped_because,
                                 f"{kind} at {budget}")

    def test_walls_are_resolved_in_the_order_they_would_be_met(self):
        """The bug this layer was written with. At a budget that binds before
        the first limit, `inline` must name the economic wall, not the
        structural one it would meet later."""
        tight, _ = both("inline", None, 1e5)
        loose, _ = both("inline", None, 1e12)
        self.assertEqual(tight.reason, "unaffordable")
        self.assertEqual(loose.reason, "limit-undefined")
        self.assertLess(tight.stop_rung, loose.stop_rung)

    def test_prediction_is_pure(self):
        """Calling it twice, and calling it after a climb, must not change it —
        the interior view cannot be quietly reading the exterior one."""
        theory = peano("inline")
        cap = Capacity(1e5, 1.0)
        first = predict_stop(theory, blocks=4, per_block=10, capacity=cap)
        transfinite_climb(theory, blocks=4, per_block=10, capacity=cap)
        second = predict_stop(theory, blocks=4, per_block=10, capacity=cap)
        self.assertEqual((first.stop_rung, first.reason),
                         (second.stop_rung, second.reason))


# ------------------------------------------ 2. the three walls stay distinct


class TestLimitStatus(unittest.TestCase):

    def test_each_arm_reports_the_expected_status(self):
        self.assertEqual(peano("inline").limit_status().status, "absent")
        self.assertEqual(peano("indexed").limit_status().status, "available")
        self.assertEqual(peano("truncated", width=3).limit_status().status,
                         "available")
        self.assertEqual(peano("searched").limit_status().status, "unknown")

    def test_only_the_searched_arm_is_undecided(self):
        for kind, width in ARMS:
            status = peano(kind, width=width).limit_status()
            self.assertEqual(status.decided, kind != "searched", kind)

    def test_can_take_limit_refuses_an_uncertified_edge(self):
        """`searched` must not be allowed through: an uncertified edge is not
        one the theory may take, even though it is really there."""
        self.assertFalse(peano("searched").can_take_limit())
        with self.assertRaises(LimitUndefined):
            limit_step(peano("searched"))

    def test_the_searched_arms_edge_is_actually_real(self):
        """The detail the whole finding rests on: its fundamental sequence is
        total, so it halts on a live continuation rather than an absent one."""
        from project_genesis.reflection import _opaque_sequence
        for n in range(200):
            self.assertIsNotNone(_opaque_sequence(n, 10 ** 6))
        v = verify_searched_notation(_opaque_sequence, bound=500, budget=10 ** 6)
        self.assertEqual(v.status, "verified-to")
        self.assertFalse(v.conclusive,
                         "a bounded certifier must never conclude validity")

    def test_the_third_wall_is_budget_invariant(self):
        answers = {(p.stop_rung, p.reason, p.wall_is_real)
                   for p in (both("searched", None, b)[0] for b in BUDGETS)}
        self.assertEqual(len(answers), 1)

    def test_the_climb_reports_undecidable_separately(self):
        _, o = both("searched", None, None)
        self.assertEqual(o.stopped_because, "undecidable")
        self.assertEqual(o.limits_taken, 0)


# --------------------------------- 3. complete on location, not on necessity


class TestLocationVersusNecessity(unittest.TestCase):

    def test_decidable_arms_settle_both_questions(self):
        for kind, width in (("inline", None), ("indexed", None),
                            ("truncated", 3)):
            for budget in BUDGETS:
                p, _ = both(kind, width, budget)
                self.assertTrue(p.certain, f"{kind} at {budget}")
                self.assertIsNotNone(p.wall_is_real)

    def test_the_searched_arm_settles_location_but_not_necessity(self):
        for budget in BUDGETS:
            p, o = both("searched", None, budget)
            self.assertIsNotNone(p.stop_rung, "location must be known")
            self.assertEqual(p.stop_rung, o.taken)
            self.assertIsNone(p.wall_is_real, "necessity must not be known")
            self.assertFalse(p.certain)

    def test_a_single_block_never_reaches_any_wall(self):
        """With no limit to take, even the two blocked arms run clean — so the
        walls measured above are the limit edge and nothing incidental."""
        for kind, width in ARMS:
            p, o = both(kind, width, None, blocks=1, per_block=6)
            self.assertEqual(o.stopped_because, "horizon")
            self.assertIsNone(p.stop_rung)
            self.assertTrue(p.certain)


if __name__ == "__main__":
    unittest.main()
