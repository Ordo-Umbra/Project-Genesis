"""Tests for the "was this claim ever actually tested?" diagnostics.

These are meta-instruments: they judge other measurements. That makes their
failure mode nastier than usual, because a meta-instrument that is too lenient
does not produce a wrong number — it produces a *reassuring* one, on every claim
it is pointed at, forever. So the tests here are organised around the cases the
diagnostics exist to catch, each constructed with a known answer:

- a difference whose parts move together (real cancellation);
- a difference with one part pinned (arithmetic wearing a measurement's clothes);
- a ranking with a rival that nearly overtakes (a real robustness test);
- a ranking whose winner leads by so much that nothing could have flipped it;
- a ranking whose rivals are identically zero (categorical, not robust).

The last one matters most and is the easiest to get wrong. §2's `P = 3` cliff is
that case, and it is a *stronger* claim than robustness — so the diagnostic must
not report it as a weak result, only as a different one.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.robustness import (  # noqa: E402
    cancellation, ordinal_headroom, spread, threshold_headroom,
)


class TestSpread(unittest.TestCase):

    def test_it_is_max_minus_min(self):
        self.assertAlmostEqual(spread([1.0, 3.0, 2.0]), 2.0)

    def test_non_finite_entries_are_dropped(self):
        self.assertAlmostEqual(spread([1.0, np.nan, 3.0, np.inf]), 2.0)

    def test_nothing_finite_is_nan_not_zero(self):
        """Zero would read as 'perfectly stable', which is the opposite of
        'nothing was measured'. The distinction is the point of the module."""
        self.assertTrue(np.isnan(spread([np.nan, np.inf])))

    def test_a_single_value_has_zero_spread(self):
        self.assertEqual(spread([2.5]), 0.0)


class TestCancellation(unittest.TestCase):

    def test_parts_moving_together_cancel(self):
        shift = [0.0, 0.4, -0.6, 1.0, -0.5]
        a = [2.0 + s for s in shift]
        b = [1.0 + s + 0.01 * i for i, s in enumerate(shift)]
        c = cancellation(a, b)
        self.assertTrue(c["exercised"])
        self.assertTrue(c["cancels"])
        self.assertLess(c["ratio_vs_smaller"], 0.2)
        self.assertEqual(c["verdict"], "CANCELS")

    def test_one_part_pinned_is_subtraction_not_cancellation(self):
        """The bug that started this module. With `a` fixed, the difference's
        spread EQUALS `b`'s exactly — no measurement has occurred, and a
        diagnostic that passed it would be certifying arithmetic."""
        a = [2.0] * 4
        b = [1.0, 1.3, 0.8, 1.1]
        c = cancellation(a, b)
        self.assertAlmostEqual(c["spread_a"], 0.0, places=14)
        self.assertAlmostEqual(c["spread_difference"], c["spread_b"], places=14)
        self.assertFalse(c["exercised"])
        self.assertFalse(c["cancels"])

    def test_nothing_moving_is_not_cancellation(self):
        c = cancellation([2.0, 2.001, 1.999], [1.0, 1.001, 0.999])
        self.assertFalse(c["exercised"])
        self.assertFalse(c["cancels"])

    def test_anti_correlated_parts_are_exercised_but_do_not_cancel(self):
        """Both move, so the sweep had grip — and the difference moves *more*
        than either part. 'Exercised' must not be confused with 'passed'."""
        c = cancellation([2.0, 2.5, 1.5], [1.0, 0.5, 1.5])
        self.assertTrue(c["exercised"])
        self.assertFalse(c["cancels"])
        self.assertGreater(c["ratio_vs_smaller"], 1.0)
        self.assertEqual(c["verdict"], "no cancellation")

    def test_the_floor_is_reported_back_with_the_result(self):
        """A threshold that travels separately from its number is how the
        undeclared constants this programme keeps finding got there."""
        c = cancellation([2.0, 2.1], [1.0, 1.1], floor=0.05)
        self.assertEqual(c["floor"], 0.05)

    def test_the_floor_gates_exercised(self):
        a, b = [2.0, 2.1, 1.9], [1.0, 1.1, 0.9]
        self.assertTrue(cancellation(a, b, floor=0.05)["exercised"])
        self.assertFalse(cancellation(a, b, floor=0.5)["exercised"])

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            cancellation([1.0, 2.0], [1.0])


class TestOrdinalHeadroom(unittest.TestCase):
    """The new instrument: could the ranking have flipped at all?"""

    def test_a_near_miss_is_a_real_robustness_test(self):
        """The rival closes to within a fraction of its own motion and still
        loses. This is the only shape that earns 'the ranking is robust'."""
        sweep = [{"a": 1.00, "b": 0.90},
                 {"a": 1.00, "b": 0.98},
                 {"a": 1.00, "b": 0.85}]
        h = ordinal_headroom(sweep)
        self.assertEqual(h["winner"], "a")
        self.assertFalse(h["flips"])
        self.assertTrue(h["exercised"])
        self.assertLess(h["headroom"], 1.0)
        self.assertIn("EXERCISED", h["verdict"])

    def test_an_untouchable_lead_is_not_a_measurement(self):
        """The winner leads by 100x anything the sweep moved. It could not have
        lost, so its winning says nothing about the convention."""
        sweep = [{"a": 10.0, "b": 0.10},
                 {"a": 10.0, "b": 0.11},
                 {"a": 10.0, "b": 0.09}]
        h = ordinal_headroom(sweep)
        self.assertFalse(h["exercised"])
        self.assertGreater(h["headroom"], 100.0)
        self.assertIn("NOT EXERCISED", h["verdict"])

    def test_a_flip_is_reported_and_is_information(self):
        sweep = [{"a": 1.0, "b": 0.5}, {"a": 0.4, "b": 0.9}]
        h = ordinal_headroom(sweep)
        self.assertTrue(h["flips"])
        self.assertIn("FLIPS", h["verdict"])

    def test_identically_zero_rivals_are_structural_not_weak(self):
        """§2's P=3 cliff. The rivals are not merely behind, they are absent —
        a categorical claim, and a stronger one than robustness. The diagnostic
        must say which it is, and must not file it as a failure."""
        sweep = [{2: 0.0, 3: 0.41, 4: 0.0},
                 {2: 0.0, 3: 0.33, 4: 0.0},
                 {2: 0.0, 3: 0.28, 4: 0.0}]
        h = ordinal_headroom(sweep)
        self.assertEqual(h["winner"], 3)
        self.assertTrue(h["structural"])
        self.assertFalse(h["flips"])
        self.assertIn("STRUCTURAL", h["verdict"])

    def test_a_live_rival_outranks_a_pinned_one_and_the_sweep_is_exercised(self):
        """A zero rival alongside a live one must not mask the live one. The
        live rival is what constrains the flip, so it is reported and the
        verdict is a genuine robustness test — not 'structural'."""
        sweep = [{"a": 1.0, "zero": 0.0, "near": 0.8},
                 {"a": 1.0, "zero": 0.0, "near": 0.6}]
        h = ordinal_headroom(sweep)
        self.assertEqual(h["closest_rival"], "near")
        self.assertFalse(h["structural"])
        self.assertEqual(h["zero_rivals"], ["zero"])
        self.assertIn("EXERCISED", h["verdict"])

    def test_a_pinned_rival_never_registers_as_a_near_miss(self):
        """The bug this module exists to catch, in its ranking form. The winner
        swings widely, so the *lead* swings widely, and a headroom computed from
        that reads as 'a rival nearly overtook'. Nothing of the sort happened:
        the rival never moved, and the lead's motion is the winner's alone."""
        sweep = [{"a": 0.41, "z": 0.0}, {"a": 0.33, "z": 0.0},
                 {"a": 0.28, "z": 0.0}]
        h = ordinal_headroom(sweep)
        self.assertTrue(np.isinf(h["headroom"]))
        self.assertFalse(h["exercised"])
        self.assertTrue(h["structural"])

    def test_a_motionless_sweep_has_infinite_headroom(self):
        """Nothing moved at all: the lead never varied, so no amount of this
        convention could close it. Division by a zero spread must not crash."""
        sweep = [{"a": 1.0, "b": 0.5}, {"a": 1.0, "b": 0.5}]
        h = ordinal_headroom(sweep)
        self.assertTrue(np.isinf(h["headroom"]))
        self.assertFalse(h["exercised"])

    def test_the_tolerance_is_a_declared_convention(self):
        sweep = [{"a": 1.0, "b": 0.5}, {"a": 1.0, "b": 0.9}]
        loose = ordinal_headroom(sweep, headroom_tol=10.0)
        tight = ordinal_headroom(sweep, headroom_tol=0.1)
        self.assertTrue(loose["exercised"])
        self.assertFalse(tight["exercised"])
        self.assertEqual(loose["headroom_tol"], 10.0)
        self.assertAlmostEqual(loose["headroom"], tight["headroom"])

    def test_an_empty_sweep_is_handled(self):
        h = ordinal_headroom([])
        self.assertIsNone(h["winner"])
        self.assertFalse(h["exercised"])

    def test_a_single_option_cannot_be_a_ranking(self):
        h = ordinal_headroom([{"a": 1.0}, {"a": 2.0}])
        self.assertFalse(h["exercised"])
        self.assertIn("only one option", h["verdict"])

    def test_settings_must_score_the_same_options(self):
        with self.assertRaises(ValueError):
            ordinal_headroom([{"a": 1.0, "b": 0.5}, {"a": 1.0, "c": 0.5}])

    def test_linear_and_log_disagree_on_orders_of_magnitude(self):
        """§2's junction densities, as measured. At the tightest probe the
        rivals are exactly zero, so the *difference* is smallest there — a
        linear reading calls that a near miss even though the winner leads by
        infinity. In ratio terms the rival that is actually closing is a
        different one. Getting this backwards produced a wrong verdict once
        already, so both readings are pinned."""
        sweep = [{3: 0.006944, 4: 0.0, 5: 0.0},
                 {3: 0.029257, 4: 0.000643, 5: 0.0},
                 {3: 0.068416, 4: 0.004115, 5: 0.000707}]
        lin = ordinal_headroom(sweep, scale="linear")
        log = ordinal_headroom(sweep, scale="log")
        self.assertEqual(lin["closest_rival"], 5)     # smallest difference
        self.assertEqual(log["closest_rival"], 4)     # smallest ratio
        self.assertEqual(log["scale"], "log")

    def test_a_rival_absent_at_every_setting_is_infinitely_behind_in_ratio(self):
        sweep = [{"a": 0.1, "b": 0.0}, {"a": 0.9, "b": 0.0}]
        h = ordinal_headroom(sweep, scale="log")
        self.assertTrue(np.isinf(h["headroom"]))
        self.assertFalse(h["exercised"])
        self.assertTrue(h["structural"])

    def test_log_scale_tracks_a_collapsing_ratio(self):
        """A margin falling 100x -> 10x -> 2x is a rival genuinely closing, and
        must read as exercised however small the absolute numbers are."""
        sweep = [{"a": 1e-3, "b": 1e-5},
                 {"a": 1e-3, "b": 1e-4},
                 {"a": 1e-3, "b": 5e-4}]
        h = ordinal_headroom(sweep, scale="log")
        self.assertTrue(h["exercised"])
        self.assertLess(h["headroom"], 1.0)

    def test_an_unknown_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            ordinal_headroom([{"a": 1.0, "b": 0.5}], scale="sideways")

    def test_headroom_is_measured_in_units_of_the_leads_own_motion(self):
        """The scale-free property that makes the number comparable across
        experiments with different units: multiply every reading by a constant
        and the headroom is unchanged."""
        sweep = [{"a": 1.0, "b": 0.9}, {"a": 1.0, "b": 0.6}]
        base = ordinal_headroom(sweep)["headroom"]
        scaled = ordinal_headroom([{k: 37.0 * v for k, v in s.items()}
                                   for s in sweep])["headroom"]
        self.assertAlmostEqual(base, scaled, places=12)


class TestThresholdHeadroom(unittest.TestCase):
    """For a boolean claim: '16/16 arms held' counts settings, not closeness."""

    def test_a_comfortable_margin_is_not_a_measurement(self):
        """Every arm holds, and every arm holds by five times anything the
        sweep moved. The condition could not have failed, so the count of
        passing arms is a restatement of the setup."""
        t = threshold_headroom([0.50, 0.60, 0.55])
        self.assertTrue(t["holds"])
        self.assertFalse(t["exercised"])
        self.assertGreater(t["headroom"], 3.0)
        self.assertIn("NOT EXERCISED", t["verdict"])

    def test_a_near_approach_is_a_measurement(self):
        t = threshold_headroom([0.05, 0.60, 0.55])
        self.assertTrue(t["holds"])
        self.assertTrue(t["exercised"])
        self.assertLess(t["headroom"], 1.0)
        self.assertIn("EXERCISED", t["verdict"])

    def test_a_crossing_is_reported_as_failure_not_as_holding(self):
        t = threshold_headroom([-0.10, 0.60])
        self.assertFalse(t["holds"])
        self.assertIn("CROSSES", t["verdict"])

    def test_a_motionless_sweep_cannot_have_tested_anything(self):
        t = threshold_headroom([0.4, 0.4, 0.4])
        self.assertTrue(np.isinf(t["headroom"]))
        self.assertFalse(t["exercised"])

    def test_the_threshold_is_declared_by_the_caller(self):
        """Unlike a rival pinned at zero, this boundary is crossable in both
        directions, so no reachability is inferred on the caller's behalf."""
        margins = [1.05, 1.10, 1.15]
        far = threshold_headroom(margins, threshold=0.0)
        near = threshold_headroom(margins, threshold=1.0)
        self.assertFalse(far["exercised"])       # 1.05 above, motion 0.10
        self.assertTrue(near["exercised"])       # 0.05 above, same motion
        self.assertGreater(far["headroom"], near["headroom"])

    def test_missing_readings_are_dropped_but_infinities_are_not(self):
        """The three cases are genuinely different and were once conflated,
        which silently deleted the arm that decided the claim:

        - ``nan`` is a reading that was never taken — drop it;
        - ``+inf`` is the condition holding by an unbounded amount — a pass;
        - ``-inf`` is the condition failing as badly as it can — a failure,
          and the one that must never be filtered away.
        """
        self.assertTrue(threshold_headroom([0.5, np.nan, 0.6])["holds"])
        self.assertTrue(threshold_headroom([np.inf, 0.6])["holds"])
        failed = threshold_headroom([-np.inf, 0.6])
        self.assertFalse(failed["holds"])
        self.assertIn("CROSSES", failed["verdict"])

    def test_nothing_but_missing_readings_is_not_a_pass(self):
        empty = threshold_headroom([np.nan, np.nan])
        self.assertFalse(empty["holds"])
        self.assertFalse(empty["exercised"])

    def test_headroom_is_scale_free(self):
        base = threshold_headroom([0.2, 0.5, 0.9])["headroom"]
        scaled = threshold_headroom([2.0, 5.0, 9.0])["headroom"]
        self.assertAlmostEqual(base, scaled, places=12)


if __name__ == "__main__":
    unittest.main()
