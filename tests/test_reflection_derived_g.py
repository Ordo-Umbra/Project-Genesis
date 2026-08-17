"""Tests for `G` as a derived quantity.

The claim is that `G` needs no independent existence — it falls out of four
measured dimensions. Two things have to hold for that to be worth anything:

1. **The algebra must be right, including the invariant.** `G_certified` can
   never exceed `G_actual`: a system must not be able to certify a move that
   is not there. That is the one direction of the interior/exterior gap which
   would be a soundness bug rather than a finding.

2. **All three verdicts must be realised, and by the right arms.** If `hidden`
   were empty the category would be unnecessary; if every arm had it, it would
   not be the epistemic wall it was attributed to.
"""

from __future__ import annotations

import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.reflection import (
    Capacity, Continuation, derive_continuation, ladder, peano,
    transfinite_climb,
)

ARMS = (("inline", None), ("indexed", None), ("truncated", 3),
        ("searched", None))


def climb(theory, n):
    for s in ladder(theory, n):
        theory = s.theory_after
    return theory


# ---------------------------------------------------------- 1. the algebra


class TestContinuationAlgebra(unittest.TestCase):

    def test_certified_never_exceeds_actual(self):
        """The soundness direction. Over every combination of the four
        dimensions, a system must never certify a move that is not there."""
        for s, a, p, c in itertools.product((True, False), (True, False),
                                            (True, False),
                                            (True, False, None)):
            k = Continuation(structural=s, affordable=a, productive=p,
                             certifiable=c)
            if c and not s:
                continue          # certifying a nonexistent edge is the bug
            self.assertLessEqual(k.g_certified, k.g_actual,
                                 f"{(s, a, p, c)}")

    def test_the_three_verdicts(self):
        live = dict(structural=True, affordable=True, productive=True)
        self.assertEqual(Continuation(**live, certifiable=True).verdict,
                         "recognised")
        self.assertEqual(Continuation(**live, certifiable=None).verdict,
                         "hidden")
        self.assertEqual(Continuation(structural=False, affordable=True,
                                      productive=False,
                                      certifiable=False).verdict, "terminal")

    def test_hidden_requires_a_real_edge_that_is_not_certified(self):
        hidden = Continuation(structural=True, affordable=True,
                              productive=True, certifiable=None)
        self.assertEqual((hidden.g_actual, hidden.g_certified), (1, 0))

    def test_blocked_by_names_the_first_failure_in_order(self):
        both = Continuation(structural=False, affordable=False,
                            productive=False, certifiable=False)
        self.assertEqual(both.blocked_by, "economic")
        struct = Continuation(structural=False, affordable=True,
                              productive=True, certifiable=False)
        self.assertEqual(struct.blocked_by, "structural")
        unprod = Continuation(structural=True, affordable=True,
                              productive=False, certifiable=True)
        self.assertEqual(unprod.blocked_by, "unproductive")
        epi = Continuation(structural=True, affordable=True, productive=True,
                           certifiable=None)
        self.assertEqual(epi.blocked_by, "epistemic")
        self.assertIsNone(Continuation(True, True, True, True).blocked_by)


# ------------------------------------------------- 2. derived from real arms


class TestDerivedFromArms(unittest.TestCase):

    def test_each_arm_derives_the_expected_limit_verdict(self):
        expected = {"inline": "terminal", "indexed": "recognised",
                    "truncated": "recognised", "searched": "hidden"}
        for kind, width in ARMS:
            theory = climb(peano(kind, width=width), 6)
            c = derive_continuation(theory, move="limit")
            self.assertEqual(c.verdict, expected[kind], kind)

    def test_hidden_belongs_to_exactly_one_arm(self):
        hidden = {kind for kind, width in ARMS
                  if derive_continuation(climb(peano(kind, width=width), 6),
                                         move="limit").verdict == "hidden"}
        self.assertEqual(hidden, {"searched"})

    def test_a_stalled_arm_derives_unproductive_on_the_successor(self):
        stalled = climb(peano("truncated", width=3), 9)
        c = derive_continuation(stalled, move="successor")
        self.assertFalse(c.productive)
        self.assertEqual(c.blocked_by, "unproductive")
        self.assertEqual(c.g_actual, 0)

    def test_unproductivity_does_not_stop_a_climb(self):
        """Q3: it is a fourth dimension, not a fourth wall."""
        outcome = transfinite_climb(peano("truncated", width=3), blocks=1,
                                    per_block=20)
        self.assertEqual(outcome.stopped_because, "horizon")
        self.assertLess(outcome.productive, outcome.taken)

    def test_a_successor_is_always_structural_and_certifiable(self):
        """The asymmetry with the limit, which is why two mechanisms exist."""
        for kind, width in ARMS:
            c = derive_continuation(peano(kind, width=width),
                                    move="successor")
            self.assertTrue(c.structural, kind)
            self.assertTrue(c.certifiable, kind)

    def test_a_tight_budget_blocks_economically_before_anything_else(self):
        c = derive_continuation(peano("indexed"), move="successor", kappa=10.0)
        self.assertFalse(c.affordable)
        self.assertEqual(c.blocked_by, "economic")
        self.assertEqual(c.g_actual, 0)

    def test_capacity_and_kappa_agree_as_budget_sources(self):
        a = derive_continuation(peano("indexed"), move="limit", kappa=1e12)
        b = derive_continuation(peano("indexed"), move="limit",
                                capacity=Capacity(1e12, 1.0))
        self.assertEqual(a, b)

    def test_rejects_an_unknown_move(self):
        with self.assertRaises(ValueError):
            derive_continuation(peano("indexed"), move="teleport")


if __name__ == "__main__":
    unittest.main()
