"""Tests for `G` as a derived quantity.

The claim is that `G` needs no independent existence — it falls out of four
measured dimensions. Two things have to hold for that to be worth anything:

1. **The algebra must be right, including the invariant.** `G_certified` can
   never exceed `G_actual`: a system must not be able to certify a move that
   is not there. That is the one direction of the interior/exterior gap which
   would be a soundness bug rather than a finding.

2. **All four verdicts must be realised, and by the right arms.** If `hidden`
   were empty the category would be unnecessary; if every arm had it, it would
   not be the epistemic wall it was attributed to. And `stagnant` must stay
   distinct from `terminal` — an earlier version merged them, which contradicted
   the measurement that found a stalled arm running to the horizon.

3. **Directional advance must be a second axis, not a sixth verdict.** A
   review asked for it in the verdict list. It cannot go there: an earlier
   version of this class merged `stagnant` into `terminal` by exactly that
   route, and folding `advancing` in would repeat it — a state that is
   productive, non-advancing *and* uncertifiable would report one fact and lose
   the other. So the anti-merge property is asserted directly, and so is the
   2x2 whose off-diagonal cells are the two dissociations this series found in
   two different domains.

4. **The classification must be checked exhaustively.** A review proposed that
   every terminal state is economic, structural, or epistemic. Over the whole
   sound predicate space that is false, and the way it fails is the finding, so
   it is pinned here rather than left to a run.
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

    def test_the_four_verdicts(self):
        live = dict(structural=True, affordable=True, productive=True)
        self.assertEqual(Continuation(**live, certifiable=True).verdict,
                         "recognised")
        self.assertEqual(Continuation(**live, certifiable=None).verdict,
                         "hidden")
        self.assertEqual(Continuation(structural=True, affordable=True,
                                      productive=False,
                                      certifiable=True).verdict, "stagnant")
        self.assertEqual(Continuation(structural=False, affordable=True,
                                      productive=False,
                                      certifiable=False).verdict, "terminal")

    def test_stagnant_is_not_terminal(self):
        """A review caught this: an earlier version returned `terminal`
        whenever `g_actual` was 0, merging a system with no move at all into a
        system whose moves achieve nothing. The second one does not halt."""
        stagnant = Continuation(structural=True, affordable=True,
                                productive=False, certifiable=True)
        terminal = Continuation(structural=False, affordable=True,
                                productive=True, certifiable=False)
        self.assertEqual(stagnant.g_actual, terminal.g_actual, 0)
        self.assertNotEqual(stagnant.verdict, terminal.verdict)
        self.assertTrue(stagnant.moves_exist)
        self.assertFalse(terminal.moves_exist)
        self.assertFalse(stagnant.halts)
        self.assertTrue(terminal.halts)

    def test_the_degenerate_case_is_not_also_recognised(self):
        """`G_actual = G_certified = 0` must read as terminal and nothing else.
        Stated as a table of conditions rather than an ordered rule, that case
        satisfies both `terminal` and `recognised` — which is the ambiguity the
        review flagged in the prose version."""
        dead = Continuation(structural=False, affordable=False,
                            productive=False, certifiable=False)
        self.assertEqual((dead.g_actual, dead.g_certified), (0, 0))
        self.assertEqual(dead.verdict, "terminal")

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


# --------------------------------- 3. the classification, checked exhaustively


def _sound_space():
    """Every combination where the certifier does not lie: it may decline to
    commit (`None`), but cannot certify an absent edge nor refute a present
    one. The rest describe a broken certifier, not a reachable state."""
    for s, a, p, c in itertools.product((True, False), (True, False),
                                        (True, False), (True, False, None)):
        if c is not None and c != s:
            continue
        yield Continuation(structural=s, affordable=a, productive=p,
                           certifiable=c)


class TestClassification(unittest.TestCase):
    """A review proposed: every terminal state is economic, structural, or
    epistemic. Checked over the whole sound space, that is false — and the way
    it fails is the finding, so it is pinned here rather than left to a run."""

    def test_the_sound_space_is_the_expected_size(self):
        self.assertEqual(len(list(_sound_space())), 16)

    def test_terminal_states_carry_only_two_walls(self):
        walls = {k.blocked_by for k in _sound_space() if k.verdict == "terminal"}
        self.assertEqual(walls, {"economic", "structural"})
        self.assertNotIn("epistemic", walls)

    def test_the_epistemic_case_is_hidden_not_terminal(self):
        verdicts = {k.verdict for k in _sound_space()
                    if k.blocked_by == "epistemic"}
        self.assertEqual(verdicts, {"hidden"})

    def test_hidden_has_a_live_move(self):
        for k in _sound_space():
            if k.verdict == "hidden":
                self.assertEqual(k.g_actual, 1)
                self.assertEqual(k.g_certified, 0)

    def test_halting_and_having_no_moves_are_different_cuts(self):
        halts = {k.verdict for k in _sound_space() if k.halts}
        runs = {k.verdict for k in _sound_space() if not k.halts}
        self.assertEqual(halts, {"terminal", "hidden"})
        self.assertEqual(runs, {"stagnant", "recognised"})

    def test_every_state_gets_exactly_one_verdict(self):
        names = {"terminal", "stagnant", "hidden", "recognised"}
        for k in _sound_space():
            self.assertIn(k.verdict, names)

    def test_no_soundness_violation_anywhere_in_the_space(self):
        for k in _sound_space():
            self.assertLessEqual(k.g_certified, k.g_actual)


# ------------------------------------------ 4. the advancing axis

from project_genesis.reflection_dag import (  # noqa: E402
    ReflectionGraph, as_continuation, broaden, deepen, reflect,
)


def _live(**kw):
    base = dict(structural=True, affordable=True, productive=True,
                certifiable=True)
    return Continuation(**(base | kw))


class TestAdvancingIsASecondAxis(unittest.TestCase):

    def test_it_defaults_to_true_so_no_earlier_verdict_moves(self):
        """Every result before this one was measured in a linear domain where
        every step increments the rung by construction."""
        self.assertTrue(Continuation(True, True, True, True).advancing)
        self.assertEqual(Continuation(True, True, True, True).verdict,
                         "recognised")

    def test_the_verdict_does_not_read_it(self):
        """The anti-merge property. Turning advance off must not change what the
        world/knowledge axis reports, or the two are not separable."""
        for c in (True, False, None):
            a = _live(certifiable=c, advancing=True)
            b = _live(certifiable=c, advancing=False)
            self.assertEqual(a.verdict, b.verdict, c)
            self.assertNotEqual(a.direction, b.direction, c)

    def test_a_circling_and_uncertifiable_state_keeps_both_facts(self):
        """Exactly the state a sixth verdict slot would have flattened."""
        k = _live(certifiable=None, advancing=False)
        self.assertEqual(k.verdict, "hidden")
        self.assertEqual(k.direction, "circling")

    def test_direction_is_halted_when_no_move_exists(self):
        dead = Continuation(structural=False, affordable=True, productive=True,
                            certifiable=False)
        self.assertEqual(dead.direction, "halted")
        self.assertFalse(dead.moves_exist)

    def test_g_advancing_never_exceeds_g_actual(self):
        for s, a, p, c, adv in itertools.product(
                (True, False), (True, False), (True, False),
                (True, False, None), (True, False)):
            if c and not s:
                continue
            k = Continuation(structural=s, affordable=a, productive=p,
                             certifiable=c, advancing=adv)
            self.assertLessEqual(k.g_advancing, k.g_actual)

    def test_advance_is_not_a_wall(self):
        """`blocked_by` names what stops a system. Circling does not stop it —
        that is the same argument that kept unproductivity off the wall list."""
        self.assertIsNone(_live(advancing=False).blocked_by)


class TestTheTwoDissociations(unittest.TestCase):
    """Both off-diagonal cells, each measured in the domain that produces it."""

    def test_the_arithmetic_ladder_advances_without_producing(self):
        stalled = climb(peano("truncated", width=3), 9)
        k = derive_continuation(stalled, move="successor")
        self.assertTrue(k.advancing)
        self.assertFalse(k.productive)
        self.assertEqual((k.verdict, k.direction), ("stagnant", "advancing"))

    def test_the_graph_domain_produces_without_advancing(self):
        g = ReflectionGraph.base(roots=3)
        for _ in range(5):
            g = reflect(g, deepen(g)).graph_after
        k = as_continuation(reflect(g, broaden(g)))
        self.assertTrue(k.productive)
        self.assertFalse(k.advancing)
        self.assertEqual((k.verdict, k.direction), ("recognised", "circling"))

    def test_the_two_are_different_states(self):
        stalled = derive_continuation(climb(peano("truncated", width=3), 9),
                                      move="successor")
        g = ReflectionGraph.base(roots=3)
        for _ in range(5):
            g = reflect(g, deepen(g)).graph_after
        sideways = as_continuation(reflect(g, broaden(g)))
        self.assertNotEqual((stalled.verdict, stalled.direction),
                            (sideways.verdict, sideways.direction))
        self.assertEqual(stalled.g_actual, 0)
        self.assertEqual(sideways.g_actual, 1)

    def test_an_advancing_graph_step_reads_as_the_real_thing(self):
        g = ReflectionGraph.base(roots=3)
        for _ in range(5):
            g = reflect(g, deepen(g)).graph_after
        k = as_continuation(reflect(g, deepen(g)))
        self.assertEqual((k.verdict, k.direction), ("recognised", "advancing"))
        self.assertEqual(k.g_advancing, 1)

    def test_all_four_cells_are_distinguishable(self):
        cells = {(p, a): (_live(productive=p, advancing=a).verdict,
                          _live(productive=p, advancing=a).direction)
                 for p in (True, False) for a in (True, False)}
        self.assertEqual(len(set(cells.values())), 4)


if __name__ == "__main__":
    unittest.main()
