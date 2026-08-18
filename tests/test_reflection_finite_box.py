"""Tests for the saturating domain and the fifth verdict.

The claim this layer rests on is that `exhausted` is a *genuine* category and
not stagnation renamed. One test decides that and everything else supports it:

    **naming-invariance.** Change the presentation and a stagnant ladder moves
    its stall point; an exhausted one does not, because there is nothing left to
    name. If that ever failed, the fifth category would collapse into the fourth
    and `Continuation.domain_exhausted` would be dead weight.

The second load-bearing claim is Q4 — that a naming defect is *invisible* once
the box binds first. That one has consequences outside the construction, so it
is pinned across a grid of boxes and widths rather than at a single point.
"""

from __future__ import annotations

import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.finite_ladder import (
    FiniteTheory, finite_climb, finite_step, naming_schemes,
    predict_finite_stall, stall_point,
)
from project_genesis.reflection import Continuation

ADEQUATE = (("inline", None), ("indexed", None), ("searched", None))
BOXES = (3, 4, 6, 8, 12)


# ------------------------------------------------------- 1. the box saturates


class TestSaturation(unittest.TestCase):

    def test_the_ladder_reaches_the_ceiling_exactly(self):
        for k in BOXES:
            for kind, w in ADEQUATE:
                rung, reason = stall_point(k, kind, w)
                self.assertEqual(rung, k, f"{kind} in a box of {k}")
                self.assertEqual(reason, "exhausted")

    def test_i_equals_c_at_the_stall(self):
        """The first time in the series that `I = C` is reachable at all."""
        theory = FiniteTheory(atoms=5, kind="indexed")
        for _ in range(5):
            theory = finite_step(theory).after
        self.assertEqual(theory.integration, theory.capacity)
        self.assertEqual(theory.room_left, 0)
        self.assertFalse(finite_step(theory).productive)

    def test_exhaustion_does_not_halt_the_ladder(self):
        """Like stagnation and unlike a wall: it arrives, having nothing left."""
        steps, rung, reason = finite_climb(
            FiniteTheory(atoms=4, kind="indexed"), 12)
        self.assertEqual(len(steps), 12)
        self.assertEqual((rung, reason), (4, "exhausted"))
        self.assertTrue(all(not s.productive for s in steps[4:]))

    def test_a_box_needs_at_least_one_atom(self):
        with self.assertRaises(ValueError):
            FiniteTheory(atoms=0)
        with self.assertRaises(ValueError):
            FiniteTheory(atoms=4, kind="nonsense")
        with self.assertRaises(ValueError):
            FiniteTheory(atoms=4, kind="truncated")


# --------------------------------------- 2. the discriminator: naming-invariance


class TestNamingInvariance(unittest.TestCase):

    def test_exhaustion_does_not_move_with_the_scheme(self):
        """The test that makes `exhausted` a category rather than a synonym."""
        for k in BOXES:
            stalls = {stall_point(k, kind, w) for kind, w in ADEQUATE}
            self.assertEqual(len(stalls), 1, f"box {k}: {stalls}")
            self.assertEqual(stalls.pop(), (k, "exhausted"))

    def test_stagnation_does_move_with_the_scheme(self):
        """The other side of the discriminator: a bound in the presentation
        comes off with the presentation."""
        k, width = 12, 2
        stagnant = stall_point(k, "truncated", width)
        self.assertEqual(stagnant, (1 << width, "stagnant"))
        for kind, w in ADEQUATE:
            self.assertEqual(stall_point(k, kind, w), (k, "exhausted"))

    def test_stagnation_stalls_at_its_address_space(self):
        for width in (2, 3, 4):
            rung, reason = stall_point(32, "truncated", width)
            self.assertEqual((rung, reason), (1 << width, "stagnant"))


# ------------------------------- 3. a defect is invisible when the box binds first


class TestDefectInvisibility(unittest.TestCase):

    def test_indistinguishable_exactly_when_the_box_binds_first(self):
        for k, width in itertools.product((4, 6, 8, 12, 16), (2, 3, 4)):
            truncated = stall_point(k, "truncated", width)
            indexed = stall_point(k, "indexed")
            box_binds_first = (1 << width) >= k
            self.assertEqual(truncated == indexed, box_binds_first,
                             f"box {k}, address space {1 << width}")

    def test_a_broken_scheme_in_a_small_box_matches_a_sound_one_exactly(self):
        """Same stall, same reason, same interior prediction — nothing separates
        them. This is the result with consequences outside the construction."""
        k, width = 6, 3                     # address space 8 >= box 6
        broken = FiniteTheory(atoms=k, kind="truncated", width=width)
        sound = FiniteTheory(atoms=k, kind="indexed")
        self.assertEqual(stall_point(k, "truncated", width),
                         stall_point(k, "indexed"))
        self.assertEqual(predict_finite_stall(broken),
                         predict_finite_stall(sound))

    def test_the_same_defect_is_visible_in_a_larger_box(self):
        """It was never fixed — the box was just too small to reveal it."""
        width = 3
        self.assertEqual(stall_point(6, "truncated", width),
                         stall_point(6, "indexed"))
        self.assertNotEqual(stall_point(32, "truncated", width),
                            stall_point(32, "indexed"))


# ------------------------------------------- 4. the interior sees it perfectly


class TestInteriorView(unittest.TestCase):

    def test_prediction_is_exact_everywhere(self):
        n = bad = 0
        for k in BOXES:
            for width in (2, 3, 4):
                for kind, w in naming_schemes(width):
                    theory = FiniteTheory(atoms=k, kind=kind, width=w)
                    n += 1
                    if predict_finite_stall(theory) != stall_point(k, kind, w):
                        bad += 1
        self.assertEqual(bad, 0, f"{bad}/{n} mismatched")
        self.assertGreater(n, 50)

    def test_lookahead_is_unbounded(self):
        """It knows at rung 0, before taking a single step — the bookend to the
        epistemic wall, which it could never see at all."""
        theory = FiniteTheory(atoms=9, kind="indexed")
        self.assertEqual(predict_finite_stall(theory), (9, "exhausted"))
        self.assertEqual(theory.rung, 0)

    def test_room_left_counts_down_to_zero(self):
        theory = FiniteTheory(atoms=5, kind="indexed")
        seen = []
        for _ in range(7):
            seen.append(theory.room_left)
            theory = finite_step(theory).after
        self.assertEqual(seen, [5, 4, 3, 2, 1, 0, 0])


# ------------------------------------------------ 5. the fifth verdict is real


def _sound_space():
    """Every sound combination of the now-five dimensions."""
    for s, a, p, c, x in itertools.product((True, False), (True, False),
                                           (True, False), (True, False, None),
                                           (True, False)):
        if c is not None and c != s:
            continue
        yield Continuation(structural=s, affordable=a, productive=p,
                           certifiable=c, domain_exhausted=x)


class TestFifthVerdict(unittest.TestCase):

    def test_exhausted_is_its_own_verdict(self):
        k = Continuation(structural=True, affordable=True, productive=False,
                         certifiable=True, domain_exhausted=True)
        self.assertEqual(k.verdict, "exhausted")
        self.assertEqual(k.blocked_by, "exhausted")

    def test_exhausted_does_not_halt(self):
        """Same as stagnant on this axis, different on rescuability."""
        for k in _sound_space():
            if k.verdict == "exhausted":
                self.assertFalse(k.halts)

    def test_the_arithmetic_setting_pins_it_false(self):
        """Which is why it took a new domain to find, rather than being an
        oversight in the first four."""
        default = Continuation(structural=True, affordable=True,
                               productive=True, certifiable=True)
        self.assertFalse(default.domain_exhausted)
        self.assertEqual(default.verdict, "recognised")

    def test_all_five_verdicts_are_realised(self):
        self.assertEqual({k.verdict for k in _sound_space()},
                         {"exhausted", "terminal", "stagnant", "hidden",
                          "recognised"})

    def test_soundness_still_holds_over_the_extended_space(self):
        for k in _sound_space():
            self.assertLessEqual(k.g_certified, k.g_actual)

    def test_the_extended_space_is_twice_the_old_one(self):
        self.assertEqual(len(list(_sound_space())), 32)


if __name__ == "__main__":
    unittest.main()
