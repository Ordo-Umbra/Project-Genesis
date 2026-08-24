"""Tests for the interior view of a retracting wall.

The claim is that a system cannot tell, by climbing, that its foundation stopped
counting — and can tell by re-deriving it. Four things have to hold or the result
is an artifact:

1. **`retracts` must be narrower than `admits`.** Only unsettleability retracts.
   A move you cannot afford is still true. An earlier version of `certified_rank`
   tested `admits`, which made every economic wall look like a collapse and
   inflated the very quantity it was measuring; that conflation is pinned here so
   it cannot come back.

2. **The interior agent must actually be blind.** `blind_climb` may not consult
   the filter the way `rank_aware` is handed it — it may only learn by attempting
   — or Q1 measures nothing. A failed attempt must also cost a step, or probing
   would be free scepticism.

3. **The probe's inference must be sound, including where it declines.** A budget
   below every cost refuses the foundation too. If the verdict were `retracted`
   there, the probe would cry collapse every time a system merely ran out of
   money.

4. **Detection must cost something.** If probing were free there would be no
   trade to report and Q3 would be vacuous.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.reflection_dag import (
    Filters, ReflectionGraph, blind_climb, certified_rank, deepen, foundation,
    interior_verdict, probe, reflect,
)

FLAT = Filters(cost_model="description")


def tower(rungs, roots=3):
    g = ReflectionGraph.base(roots=roots)
    for _ in range(rungs):
        g = reflect(g, deepen(g)).graph_after
    return g


# ------------------------------------- 1. retraction is narrower than refusal


class TestWhatRetracts(unittest.TestCase):

    def test_only_unsettleability_retracts(self):
        key = frozenset({0, 1, 2})
        self.assertTrue(Filters(opaque=frozenset({1})).retracts(key))
        for f in (Filters(budget=0), Filters(address_bits=1),
                  Filters(certify_effort=0), Filters(max_arity=1),
                  Filters(opaque_form="join")):
            self.assertFalse(f.retracts(key), f)

    def test_an_unaffordable_tower_still_counts(self):
        """The conflation that an earlier version of `certified_rank` made."""
        g = tower(10)
        self.assertEqual(certified_rank(g, Filters(budget=0)), g.rank)
        self.assertEqual(certified_rank(g, Filters(address_bits=1)), g.rank)

    def test_an_unsettleable_base_takes_the_tower_with_it(self):
        g = tower(10)
        self.assertEqual(certified_rank(g, FLAT), g.rank)
        self.assertEqual(certified_rank(g, Filters(opaque=frozenset({0}))), 0)

    def test_only_the_lineage_that_rests_on_it(self):
        """Marking a root the tower does not stand on retracts nothing."""
        g = tower(10)
        self.assertEqual(certified_rank(g, Filters(opaque=frozenset({1}))),
                         g.rank)


# ----------------------------------------------- 2. the agent is really blind


class TestTheInteriorIsBlind(unittest.TestCase):

    def test_forward_records_are_identical_under_the_two_walls(self):
        """Q1, stated as an assertion. Same proposals, same outcomes, different
        truth."""
        g = tower(12)
        dear = Filters(budget=len(g.frontier()[0].content) - 1)
        opaque = Filters(opaque=frozenset({0}))
        a = blind_climb(g, opaque, 12)
        b = blind_climb(g, dear, 12)
        self.assertEqual(a["record"], b["record"])
        self.assertNotEqual(a["certified_rank"], b["certified_rank"])

    def test_a_refused_attempt_still_costs_a_step(self):
        """Otherwise probing would be free and Q3 would be vacuous."""
        g = tower(6)
        run = blind_climb(g, Filters(budget=0), 10)
        self.assertEqual(len(run["observations"]), 10)
        self.assertTrue(all(not o["admitted"] for o in run["observations"]))
        self.assertEqual(run["believed_rank"], g.rank)

    def test_probing_displaces_climbing_one_for_one(self):
        g = tower(6)
        free = blind_climb(g, FLAT, 20)["believed_rank"]
        probed = blind_climb(g, FLAT, 20, probe_every=2)["believed_rank"]
        self.assertEqual(free - probed, 10)

    def test_the_agent_never_reports_detection_without_probing(self):
        g = tower(12)
        run = blind_climb(g, Filters(opaque=frozenset({0})), 20)
        self.assertIsNone(run["detected_at"])


# ---------------------------------------- 3. the probe, including its silence


class TestTheProbe(unittest.TestCase):

    def test_the_foundation_is_the_cheapest_key_in_the_graph(self):
        g = tower(12)
        base = foundation(g)
        for n in g.nodes:
            self.assertLessEqual(len(base.content), len(n.content))

    def test_it_probes_a_root_the_frontier_actually_rests_on(self):
        g = tower(12)
        self.assertIn(foundation(g).ident, g.frontier()[0].content)

    def test_retraction_is_detected(self):
        g = tower(12)
        self.assertEqual(
            interior_verdict(probe(g, Filters(opaque=frozenset({0})))),
            "retracted")

    def test_an_affordability_wall_does_not_trigger_it(self):
        g = tower(12)
        dear = Filters(budget=len(g.frontier()[0].content) - 1)
        self.assertEqual(interior_verdict(probe(g, dear)), "no evidence")

    def test_a_total_halt_is_reported_as_a_non_inference(self):
        """The condition on the rule. Without it the probe would report a
        collapse every time a system merely ran out of money."""
        g = tower(12)
        result = probe(g, Filters(budget=0))
        self.assertFalse(result["admitted"])
        self.assertFalse(result["alternatives_admitted"])
        self.assertEqual(interior_verdict(result), "halted")

    def test_it_says_nothing_about_a_root_the_tower_does_not_rest_on(self):
        g = tower(12)
        self.assertEqual(
            interior_verdict(probe(g, Filters(opaque=frozenset({1})))),
            "no evidence")


# ------------------------------------ 4. the price, and the undiagnosed repair


class TestPriceAndRecovery(unittest.TestCase):

    AFTER = Filters(cost_model="description", opaque=frozenset({0}))

    def test_a_faster_probe_detects_sooner_and_climbs_less(self):
        g = tower(12)
        fast = blind_climb(g, FLAT, 60, probe_every=2, wall_at=21,
                           filters_after=self.AFTER)
        slow = blind_climb(g, FLAT, 60, probe_every=16, wall_at=21,
                           filters_after=self.AFTER)
        self.assertLess(fast["latency"], slow["latency"])
        self.assertLess(fast["believed_rank"], slow["believed_rank"])

    def test_no_probe_rate_detects_before_the_wall_exists(self):
        g = tower(12)
        for rate in (2, 4, 8):
            run = blind_climb(g, FLAT, 60, probe_every=rate, wall_at=25,
                              filters_after=self.AFTER)
            self.assertGreaterEqual(run["detected_at"], 25, rate)

    def test_the_gap_is_full_with_no_horizon_and_closes_with_plenty(self):
        g = tower(12)
        tight = blind_climb(g, FLAT, 20, wall_at=20, filters_after=self.AFTER)
        loose = blind_climb(g, FLAT, 60, wall_at=20, filters_after=self.AFTER)
        self.assertEqual(tight["certified_rank"], 0)
        self.assertGreater(tight["believed_rank"], 0)
        self.assertEqual(loose["believed_rank"], loose["certified_rank"])

    def test_the_repair_happens_without_the_diagnosis(self):
        """The finding, as an assertion: recovery does not require detection."""
        g = tower(12)
        run = blind_climb(g, FLAT, 60, wall_at=20, filters_after=self.AFTER)
        self.assertIsNone(run["detected_at"])
        self.assertEqual(run["believed_rank"], run["certified_rank"])

    def test_belief_never_moves_whatever_happens(self):
        g = tower(12)
        beliefs = {blind_climb(g, FLAT, 20 + hz, wall_at=20,
                               filters_after=self.AFTER)["believed_rank"]
                   for hz in (0, 2, 5, 10)}
        self.assertEqual(len(beliefs), 1)


if __name__ == "__main__":
    unittest.main()
