"""Tests for Cantor-normal-form ranks, limits of limits, and notation checking.

Two things here are load-bearing in a way the earlier layers were not:

1. **The ordinal arithmetic must be right.** Comparison is the only ordinal
   fact the module uses, and `ω^k` ordering is easy to get subtly wrong — a
   backwards comparison would silently reorder every result that mentions rank
   without breaking anything visibly.

2. **The decidable/searched distinction must be real.** The whole conclusion
   about Kleene's `O` rests on CNF notations being conclusively checkable and
   opaque ones not being. If `verify_searched_notation` ever returned a
   conclusive "valid", the distinction would collapse.

Backward compatibility is also asserted: `Rank(limits, successors)` and its
`.limits` / `.successors` accessors are what the ω² fragment used, and every
result recorded before CNF existed has to keep reading the same.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.reflection import (
    LimitUndefined, Rank, canonical_fundamental_sequence, construction_cost,
    ladder, limit_step, peano, step, verify_cnf_notation,
    verify_searched_notation,
)


def climb(theory, n):
    for s in ladder(theory, n):
        theory = s.theory_after
    return theory


# --------------------------------------------------------- 1. CNF arithmetic


class TestCantorNormalForm(unittest.TestCase):

    def test_backward_compatible_with_the_omega_squared_fragment(self):
        r = Rank(2, 3)
        self.assertEqual((r.limits, r.successors), (2, 3))
        self.assertEqual(str(r), "ω·2+3")
        self.assertEqual(Rank(0, 0).coeffs, ())

    def test_normalises_away_leading_zeros(self):
        self.assertEqual(Rank.from_levels({5: 0, 1: 2, 0: 1}).coeffs, (2, 1))
        self.assertEqual(Rank.from_levels({3: 0}).coeffs, ())

    def test_rejects_negative_coefficients(self):
        with self.assertRaises(ValueError):
            Rank.from_levels({1: -1})

    def test_orders_by_leading_exponent_first(self):
        self.assertLess(Rank(0, 10 ** 6), Rank.from_levels({1: 1}))
        self.assertLess(Rank.from_levels({1: 10 ** 6}),
                        Rank.from_levels({2: 1}))
        self.assertLess(Rank.from_levels({2: 10 ** 6}),
                        Rank.from_levels({3: 1}))

    def test_orders_lexicographically_within_a_degree(self):
        self.assertLess(Rank.from_levels({2: 1, 1: 5}),
                        Rank.from_levels({2: 1, 1: 6}))
        self.assertLess(Rank.from_levels({2: 1, 0: 9}),
                        Rank.from_levels({2: 1, 1: 1}))

    def test_ordering_is_a_total_order_on_a_sample(self):
        sample = [Rank.from_levels({e: c}) for e in range(4) for c in (1, 2, 7)]
        sample += [Rank(0, 0), Rank(1, 1), Rank.from_levels({3: 2, 1: 4})]
        for a in sample:
            for b in sample:
                self.assertEqual(sum((a < b, a == b, a > b)), 1, f"{a} vs {b}")

    def test_successor_and_limit_move_the_right_coefficient(self):
        r = Rank.from_levels({2: 1, 1: 3, 0: 5})
        self.assertEqual(r.successor(), Rank.from_levels({2: 1, 1: 3, 0: 6}))
        self.assertEqual(r.limit(1), Rank.from_levels({2: 1, 1: 4}))
        self.assertEqual(r.limit(2), Rank.from_levels({2: 2}))
        self.assertEqual(r.limit(3), Rank.from_levels({3: 1}))

    def test_a_limit_zeroes_everything_below_it(self):
        r = Rank.from_levels({3: 1, 2: 9, 1: 9, 0: 9}).limit(2)
        self.assertEqual(r, Rank.from_levels({3: 1, 2: 10}))
        self.assertEqual(r.successors, 0)

    def test_limit_rejects_level_zero(self):
        with self.assertRaises(ValueError):
            Rank(1, 1).limit(0)

    def test_is_limit_and_degree(self):
        self.assertTrue(Rank.from_levels({2: 1}).is_limit)
        self.assertFalse(Rank.from_levels({2: 1, 0: 1}).is_limit)
        self.assertFalse(Rank(0, 0).is_limit)
        self.assertEqual(Rank.from_levels({4: 1}).degree, 4)
        self.assertEqual(Rank(0, 0).degree, -1)

    def test_renders_readably(self):
        self.assertEqual(str(Rank.from_levels({2: 1})), "ω^2")
        self.assertEqual(str(Rank.from_levels({2: 3})), "ω^2·3")
        self.assertEqual(str(Rank.from_levels({2: 1, 1: 1, 0: 4})), "ω^2+ω+4")
        self.assertEqual(str(Rank(0, 0)), "0")

    def test_hashes_by_value(self):
        self.assertEqual(len({Rank(1, 2), Rank.from_levels({1: 1, 0: 2})}), 1)


# ------------------------------------------------------- 2. limits of limits


class TestHigherLimits(unittest.TestCase):

    def test_a_level_two_limit_reaches_omega_squared(self):
        theory = climb(peano("indexed"), 3)
        after = limit_step(theory, 2).theory_after
        self.assertEqual(after.rank, Rank.from_levels({2: 1, 0: 1}))
        self.assertGreater(after.rank, Rank.from_levels({1: 10 ** 6}))

    def test_costs_the_same_at_every_level(self):
        theory = climb(peano("indexed"), 4)
        successor = construction_cost(step(peano("indexed")))
        for level in range(1, 9):
            self.assertEqual(limit_step(theory, level).con_symbols, successor,
                             f"cost moved at level {level}")

    def test_is_productive_at_every_level(self):
        theory = climb(peano("indexed"), 4)
        for level in range(1, 9):
            self.assertTrue(limit_step(theory, level).new_axiom, level)

    def test_stacking_levels_raises_the_degree(self):
        theory = peano("indexed")
        for level in range(1, 6):
            theory = climb(theory, 2)
            theory = limit_step(theory, level).theory_after
            self.assertEqual(theory.rank.degree, level)

    def test_successors_still_work_above_omega_squared(self):
        theory = limit_step(climb(peano("indexed"), 2), 3).theory_after
        steps = list(ladder(theory, 6))
        self.assertTrue(all(s.new_axiom for s in steps))
        self.assertEqual(steps[-1].theory_after.rank.degree, 3)

    def test_the_gate_holds_at_every_level(self):
        theory = climb(peano("inline"), 4)
        for level in range(1, 9):
            with self.assertRaises(LimitUndefined, msg=f"level {level}"):
                limit_step(theory, level)

    def test_rank_round_trips_through_the_theory_fields(self):
        theory = peano("indexed")
        for level in (1, 3, 2, 4):
            theory = climb(theory, 2)
            theory = limit_step(theory, level).theory_after
            self.assertEqual(theory.rank.coefficient(0), theory.rung)
            self.assertEqual(theory.rank.coefficient(1), theory.limits)


# --------------------------------------------- 3. canonical vs searched checks


class TestNotationChecking(unittest.TestCase):

    def test_canonical_sequences_increase_and_stay_below_their_limit(self):
        for r in (Rank.from_levels({1: 1}), Rank.from_levels({1: 4}),
                  Rank.from_levels({2: 1}), Rank.from_levels({3: 2, 1: 5}),
                  Rank.from_levels({5: 1})):
            seq = canonical_fundamental_sequence(r)
            values = [seq(n) for n in range(6)]
            for a, b in zip(values, values[1:]):
                self.assertLess(a, b, f"{r}: not increasing")
            for v in values:
                self.assertLess(v, r, f"{r}: element {v} not below the limit")

    def test_known_closed_forms(self):
        omega = canonical_fundamental_sequence(Rank.from_levels({1: 1}))
        self.assertEqual([str(omega(n)) for n in range(3)], ["0", "1", "2"])
        sq = canonical_fundamental_sequence(Rank.from_levels({2: 1}))
        self.assertEqual([str(sq(n)) for n in range(3)], ["0", "ω", "ω·2"])

    def test_successors_have_no_fundamental_sequence(self):
        with self.assertRaises(ValueError):
            canonical_fundamental_sequence(Rank(1, 1))

    def test_cnf_validity_is_always_conclusive(self):
        ranks = [Rank(0, 0), Rank(0, 7), Rank.from_levels({1: 1}),
                 Rank.from_levels({2: 3, 1: 1}), Rank.from_levels({6: 2})]
        for r in ranks:
            self.assertTrue(verify_cnf_notation(r).conclusive, str(r))

    def test_a_searched_notation_is_never_conclusively_valid(self):
        """The claim the O conclusion rests on: running a sequence can refute
        it but can never confirm it."""
        total = lambda n, _b: n           # noqa: E731
        for bound in (1, 10, 1000, 10000):
            v = verify_searched_notation(total, bound=bound, budget=10)
            self.assertEqual(v.status, "verified-to")
            self.assertFalse(v.conclusive)

    def test_divergence_is_detected_only_once_the_bound_passes_it(self):
        partial = lambda n, _b, k=9: n if n < k else None   # noqa: E731
        self.assertFalse(
            verify_searched_notation(partial, bound=9, budget=10).conclusive)
        late = verify_searched_notation(partial, bound=20, budget=10)
        self.assertEqual((late.status, late.checked), ("diverges-at", 9))

    def test_total_and_diverging_are_indistinguishable_below_the_divergence(self):
        total = lambda n, _b: n                              # noqa: E731
        partial = lambda n, _b, k=9: n if n < k else None    # noqa: E731
        for bound in range(0, 10):
            a = verify_searched_notation(total, bound=bound, budget=10)
            b = verify_searched_notation(partial, bound=bound, budget=10)
            self.assertEqual((a.status, a.checked), (b.status, b.checked),
                             f"separated at bound {bound}")

    def test_a_non_increasing_sequence_is_refused(self):
        flat = lambda n, _b: 0            # noqa: E731
        v = verify_searched_notation(flat, bound=5, budget=10)
        self.assertEqual(v.status, "diverges-at")
        self.assertTrue(v.conclusive)


if __name__ == "__main__":
    unittest.main()
