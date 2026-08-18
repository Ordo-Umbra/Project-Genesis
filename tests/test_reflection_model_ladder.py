"""Tests for the semantic ladder, where the ceiling is set by consistency.

This domain exists to check whether the box's tidiness was an artifact, so the
tests concentrate on the two places the box could not have gone wrong:

1. **Capacity must be emergent, not stipulated.** `C = 2^n - 1` has to come from
   "a theory with no models is inconsistent", and the ladder must actually be
   stopped by that and not by a counter. If it were possible to eliminate the
   last model, the whole domain would be as hand-set as the box.

2. **Cost must be separable from location.** Two adequate schemes reaching the
   same floor at different costs is the finding. If every arriving scheme
   arrived in `C` steps, this experiment would have shown nothing the box did
   not, and `efficiency` would be dead weight.

The `inline` freeze is pinned mechanically rather than by its stall value,
because "stops at 1" is a symptom and "its address stops moving" is the cause.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.model_ladder import (
    ModelTheory, model_climb, model_step, schemes,
)

ADEQUATE = ("indexed", "scattered")


def run(n, kind, width=None, factor=30):
    return model_climb(ModelTheory(variables=n, kind=kind, width=width),
                       (1 << n) * factor)


# ------------------------------------------------- 1. the ceiling is semantic


class TestEmergentCapacity(unittest.TestCase):

    def test_capacity_is_one_less_than_the_model_count(self):
        for n in (2, 3, 5, 8):
            t = ModelTheory(variables=n)
            self.assertEqual(t.capacity, (1 << n) - 1)
            self.assertEqual(t.models, 1 << n)

    def test_the_last_model_can_never_be_eliminated(self):
        """Consistency is the binding law, and it binds at exactly one model."""
        t = ModelTheory(variables=3, kind="indexed")
        for _ in range(7):
            t = model_step(t).after
        self.assertEqual(len(t.alive), 1)
        s = model_step(t)
        self.assertFalse(s.consistency_allows)
        self.assertFalse(s.productive)
        self.assertEqual(s.blocked_by, "exhausted")

    def test_a_well_addressed_ladder_reaches_exactly_c(self):
        for n in (3, 4, 6, 8):
            r = run(n, "indexed")
            self.assertTrue(r["reached_ceiling"], n)
            self.assertEqual(r["integration"], r["capacity"])
            self.assertEqual(r["reason"], "exhausted")

    def test_it_never_runs_past_c(self):
        for n in (3, 5, 7):
            for kind in ADEQUATE:
                r = run(n, kind, factor=12)
                self.assertLessEqual(r["integration"], r["capacity"])

    def test_validates_its_arguments(self):
        for bad in (dict(variables=0), dict(variables=3, kind="nonsense"),
                    dict(variables=3, kind="truncated")):
            with self.assertRaises(ValueError):
                ModelTheory(**bad)


# --------------------------------------- 2. location invariant, cost is not


class TestLocationVersusCost(unittest.TestCase):

    def test_every_arriving_scheme_arrives_at_the_same_floor(self):
        for n in (4, 6, 8):
            floors = {run(n, k, factor=12)["integration"] for k in ADEQUATE}
            self.assertEqual(len(floors), 1, f"n={n}: {floors}")
            self.assertEqual(floors.pop(), (1 << n) - 1)

    def test_an_adequate_scheme_can_be_substantially_more_expensive(self):
        """The finding. Existence is what is asserted — the multiple is a
        property of one hash and is deliberately not pinned."""
        gaps = []
        for n in (7, 8):
            a, b = run(n, "indexed", factor=12), run(n, "scattered", factor=12)
            self.assertTrue(a["reached_ceiling"] and b["reached_ceiling"])
            gaps.append(b["steps_to_floor"] / a["steps_to_floor"])
        self.assertTrue(all(g > 2.0 for g in gaps), gaps)

    def test_the_efficient_scheme_wastes_nothing(self):
        for n in (4, 6, 8):
            r = run(n, "indexed")
            self.assertEqual(r["steps_to_floor"], r["capacity"])
            self.assertAlmostEqual(r["efficiency"], 1.0)

    def test_efficiency_is_below_one_where_the_hash_revisits(self):
        r = run(8, "scattered", factor=12)
        self.assertTrue(r["reached_ceiling"])
        self.assertLess(r["efficiency"], 0.6)


# --------------------------------------------- 3. three stagnation mechanisms


class TestStagnationMechanisms(unittest.TestCase):

    def test_content_addressing_freezes_after_one_wasted_move(self):
        """Pinned by mechanism, not by the stall value: the address stops
        moving because only a successful elimination changes it."""
        t = ModelTheory(variables=4, kind="inline")
        first = model_step(t)
        self.assertTrue(first.productive)
        second = model_step(first.after)
        self.assertFalse(second.productive)
        self.assertFalse(second.target_was_alive)
        third = model_step(second.after)
        self.assertEqual(third.address, second.address)
        self.assertFalse(third.address_is_new)
        self.assertFalse(third.productive)

    def test_the_freeze_is_permanent_and_early(self):
        for n in (4, 6, 8):
            r = run(n, "inline")
            self.assertFalse(r["reached_ceiling"])
            self.assertEqual(r["integration"], 1)

    def test_coverage_failure_reaches_exactly_half(self):
        """`partial` steps by two, so it can only ever address half the models
        — its addresses never repeat and it still cannot exhaust."""
        for n in (4, 6, 8):
            r = run(n, "partial")
            self.assertFalse(r["reached_ceiling"])
            self.assertEqual(r["integration"], (1 << n) // 2)

    def test_collision_failure_stalls_at_the_address_space(self):
        for width in (2, 3, 4):
            r = run(8, "truncated", width)
            self.assertFalse(r["reached_ceiling"])
            self.assertEqual(r["integration"], 1 << width)

    def test_the_three_mechanisms_reach_different_places(self):
        n = 8
        reached = {k: run(n, k, w)["integration"]
                   for k, w in (("inline", None), ("partial", None),
                                ("truncated", 3))}
        self.assertEqual(len(set(reached.values())), 3, reached)

    def test_all_three_report_the_same_verdict_despite_differing(self):
        """Which is the point: `stagnant` was hiding structure."""
        n = 8
        for kind, w in (("inline", None), ("partial", None),
                        ("truncated", 3)):
            self.assertEqual(run(n, kind, w)["reason"], "stagnant")


# ------------------------------------------------------ 4. the arms are wired


class TestSchemes(unittest.TestCase):

    def test_every_scheme_is_exercised(self):
        kinds = {k for k, _ in schemes(3)}
        self.assertEqual(kinds, {"inline", "indexed", "scattered", "partial",
                                 "truncated"})

    def test_adequate_schemes_never_repeat_an_address(self):
        for kind in ADEQUATE:
            t, seen = ModelTheory(variables=5, kind=kind), set()
            for _ in range(60):
                s = model_step(t)
                self.assertNotIn(s.address, seen, kind)
                seen.add(s.address)
                t = s.after

    def test_room_left_reaches_zero_only_at_the_ceiling(self):
        t = ModelTheory(variables=4, kind="indexed")
        for _ in range(15):
            self.assertGreater(t.room_left, 0)
            t = model_step(t).after
        self.assertEqual(t.room_left, 0)
        self.assertEqual(t.integration, t.capacity)


if __name__ == "__main__":
    unittest.main()
