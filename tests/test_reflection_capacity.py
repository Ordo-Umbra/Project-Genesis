"""Tests for cost-bounded accessibility on the reflection ladder.

The point of this layer is to make `G > 0` refutable, so the tests are mostly
about the bound actually biting:

1. **The capacity arithmetic is the field program's, discretised.** If `spend`
   does not implement "pay the load, heal a fraction `r` back toward the
   ceiling", then the analogy to `∂_t κ = r(κ₀−κ) − load` is decoration and the
   threshold below means nothing.

2. **The threshold is sharp and matches the closed form.** `r* = L/κ_max` is
   derived, not fitted. A bisection that disagreed with it would mean the
   numerics are measuring something other than the algebra — and the first
   version of the scan did exactly that, by using a horizon shorter than the
   budget's own decay time, so the horizon is asserted here too.

3. **Cost-bounding alone does not rule out the degenerate arm.** This is the
   experiment's headline negative result, so it is pinned down rather than left
   to a printed table.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.reflection import (
    Capacity, bounded_ladder, construction_cost, critical_recovery, ladder,
    measure_critical_recovery, peano, step, terminal_rung,
)


def unit_cost(kind: str, width: int | None = None) -> int:
    return construction_cost(step(peano(kind, width=width)))


# ------------------------------------------------------ 1. the arithmetic


class TestCapacity(unittest.TestCase):

    def test_validates_its_arguments(self):
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                Capacity(bad, 1.0)
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                Capacity(100.0, bad)

    def test_spend_pays_then_heals(self):
        cap = Capacity(kappa_max=100.0, recovery=0.5)
        # 80 - 20 = 60, then half way back to 100 -> 80
        self.assertAlmostEqual(cap.spend(80.0, 20.0), 80.0)

    def test_full_recovery_returns_to_the_ceiling(self):
        cap = Capacity(kappa_max=100.0, recovery=1.0)
        self.assertAlmostEqual(cap.spend(40.0, 30.0), 100.0)

    def test_converges_to_the_predicted_fixed_point(self):
        """κ* = κ_max − L(1−r)/r, which is what makes r* derivable."""
        kmax, r, load = 1000.0, 0.25, 100.0
        cap = Capacity(kmax, r)
        kappa = kmax
        for _ in range(400):
            kappa = cap.spend(kappa, load)
        self.assertAlmostEqual(kappa, kmax - load * (1 - r) / r, places=6)


# ---------------------------------------------- 2. the bound actually bites


class TestTermination(unittest.TestCase):

    def test_geometric_cost_terminates_at_every_budget(self):
        theory = peano("inline")
        for budget in (1e4, 1e5, 1e6, 1e7):
            self.assertIsNotNone(
                terminal_rung(theory, Capacity(budget, 1.0), horizon=64),
                f"inline survived a budget of {budget:g}")

    def test_flat_cost_survives_once_it_can_afford_one_rung(self):
        theory = peano("indexed")
        cost = unit_cost("indexed")
        self.assertIsNone(
            terminal_rung(theory, Capacity(cost * 2.0, 1.0), horizon=64))

    def test_a_budget_below_the_first_rung_terminates_immediately(self):
        theory = peano("indexed")
        cost = unit_cost("indexed")
        self.assertEqual(
            terminal_rung(theory, Capacity(cost / 2.0, 1.0), horizon=10), 0)

    def test_reach_is_monotone_in_budget(self):
        theory = peano("inline")
        reach = [terminal_rung(theory, Capacity(b, 1.0), horizon=64)
                 for b in (1e4, 1e5, 1e6, 1e7, 1e8)]
        self.assertEqual(reach, sorted(reach))

    def test_reach_is_logarithmic_not_linear(self):
        """A thousandfold budget must not buy a thousandfold reach."""
        theory = peano("inline")
        small = terminal_rung(theory, Capacity(1e4, 1.0), horizon=64)
        large = terminal_rung(theory, Capacity(1e7, 1.0), horizon=64)
        self.assertLess(large - small, 15)
        self.assertGreater(large, small)

    def test_the_final_step_is_reported_as_unaffordable(self):
        theory = peano("inline")
        steps = list(bounded_ladder(theory, 64, Capacity(1e5, 1.0)))
        self.assertIsNone(steps[-1].step)
        self.assertFalse(steps[-1].affordable)
        self.assertGreater(steps[-1].cost, steps[-1].kappa_before)
        for b in steps[:-1]:
            self.assertIsNotNone(b.step)
            self.assertTrue(b.affordable)


# ------------------------------------------- 3. the threshold is the algebra


class TestCriticalRecovery(unittest.TestCase):

    def test_matches_the_closed_form(self):
        budget = 2e4
        for kind, width in (("indexed", None), ("truncated", 3)):
            theory = peano(kind, width=width)
            closed = critical_recovery(unit_cost(kind, width), budget)
            measured = measure_critical_recovery(
                theory, budget, horizon=int(20 / closed))
            self.assertIsNotNone(measured)
            self.assertLess(abs(measured / closed - 1.0), 0.01,
                            f"{kind}: {measured} vs closed form {closed}")

    def test_the_threshold_is_sharp(self):
        budget = 2e4
        theory = peano("indexed")
        closed = critical_recovery(unit_cost("indexed"), budget)
        horizon = int(50 / closed)
        self.assertIsNone(terminal_rung(theory, Capacity(budget, closed * 1.2),
                                        horizon=horizon),
                          "above r* the ladder should never terminate")
        self.assertIsNotNone(terminal_rung(theory,
                                           Capacity(budget, closed * 0.8),
                                           horizon=horizon),
                             "below r* the ladder should run out")

    def test_a_horizon_shorter_than_the_decay_time_hides_the_threshold(self):
        """The bug this scan actually had. A short horizon makes even a
        starving budget look sustainable, because it has not had time to
        starve — so the horizon is part of the measurement, not a detail."""
        budget = 2e4
        theory = peano("indexed")
        closed = critical_recovery(unit_cost("indexed"), budget)
        starving = Capacity(budget, closed * 0.5)
        self.assertIsNone(terminal_rung(theory, starving, horizon=3))
        self.assertIsNotNone(terminal_rung(theory, starving,
                                           horizon=int(50 / closed)))

    def test_growing_cost_has_no_sustainable_rate(self):
        self.assertIsNone(
            measure_critical_recovery(peano("inline"), 2e4, horizon=200),
            "no fixed recovery rate can sustain an unbounded cost")


# -------------------------------- 4. a budget does not certify productivity


class TestBudgetDoesNotCertifyProductivity(unittest.TestCase):

    def test_the_degenerate_arm_is_indistinguishable_on_cost(self):
        """Q4: same reach, same threshold, and nothing produced."""
        budget = 2e4
        indexed, trunc = peano("indexed"), peano("truncated", width=3)
        cap = Capacity(budget, 0.5)
        self.assertEqual(terminal_rung(indexed, cap, horizon=64),
                         terminal_rung(trunc, cap, horizon=64))
        closed = critical_recovery(unit_cost("indexed"), budget)
        horizon = int(20 / closed)
        self.assertEqual(
            measure_critical_recovery(indexed, budget, horizon=horizon),
            measure_critical_recovery(trunc, budget, horizon=horizon))

    def test_and_yet_it_produces_nothing_past_the_wrap(self):
        cap = Capacity(2e4, 0.5)
        counts = {}
        for kind, width in (("indexed", None), ("truncated", 3)):
            taken = [b for b in bounded_ladder(peano(kind, width=width), 32, cap)
                     if b.step is not None]
            counts[kind] = (len(taken), sum(1 for b in taken if b.new_axiom))
        self.assertEqual(counts["indexed"], (32, 32))
        self.assertEqual(counts["truncated"], (32, 8))

    def test_requiring_productivity_terminates_the_degenerate_arm(self):
        """The corrected accessibility relation, at every budget and rate."""
        for budget in (2e4, 1e5, 1e6):
            for r in (0.25, 0.5, 1.0):
                cap = Capacity(budget, r)
                self.assertEqual(
                    terminal_rung(peano("truncated", width=3), cap,
                                  horizon=64, require_productive=True), 8,
                    f"budget {budget:g}, r {r}")
                self.assertIsNone(
                    terminal_rung(peano("indexed"), cap, horizon=64,
                                  require_productive=True))

    def test_productive_accessibility_tracks_the_wrap_width(self):
        for width in (2, 3, 4):
            self.assertEqual(
                terminal_rung(peano("truncated", width=width),
                              Capacity(1e5, 1.0), horizon=64,
                              require_productive=True), 1 << width)

    def test_requiring_productivity_does_not_change_the_honest_arms(self):
        cap = Capacity(1e5, 1.0)
        for kind in ("inline", "indexed"):
            plain = terminal_rung(peano(kind), cap, horizon=64)
            strict = terminal_rung(peano(kind), cap, horizon=64,
                                   require_productive=True)
            self.assertEqual(plain, strict, kind)


# --------------------------------------------------- 5. the cost model itself


class TestCostModel(unittest.TestCase):

    def test_flat_arms_are_flat_and_inline_is_not(self):
        for kind, width in (("indexed", None), ("truncated", 3)):
            costs = {construction_cost(s)
                     for s in ladder(peano(kind, width=width), 8)}
            self.assertEqual(len(costs), 1, kind)
        costs = [construction_cost(s) for s in ladder(peano("inline"), 8)]
        self.assertEqual(costs, sorted(costs))
        self.assertGreater(costs[-1] / costs[0], 50)

    def test_construction_cost_is_the_sentence_being_added(self):
        s = step(peano("indexed"))
        self.assertEqual(construction_cost(s), s.con_symbols)


if __name__ == "__main__":
    unittest.main()
