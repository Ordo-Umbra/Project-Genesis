"""Tests for the compass deadband sweep.

The experiment's conclusion rests on two things being right. The band has to be
located correctly — a step sits *between* two scan points, and an off-by-one
here would move every reported interval by half a grid step and quietly change
which band the optimum falls inside. And `plateau` has to detect a setting
sitting on a boundary, since that is the entire content of Q3: a published
constant that survives a 20x change downward and none upward is knife-edge in
the direction that matters, and a plateau function that reported a symmetric
range would hide exactly the failure being looked for.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.label_stability import (  # noqa: E402
    band_extent,
    invariant_fraction,
    label_sequence,
    plateau,
)


def rows_from(dc, di, x_key="T"):
    return [{x_key: float(i), "delta_c": float(a), "delta_i": float(b)}
            for i, (a, b) in enumerate(zip(dc, di))]


class TestLabelSequence(unittest.TestCase):

    def test_one_label_per_step_not_per_point(self):
        r = rows_from([0, 1, 2, 3], [3, 2, 1, 0])
        self.assertEqual(len(label_sequence(r, 0.01)), 3)

    def test_rising_distinction_and_falling_integration_is_diverging(self):
        r = rows_from([0.0, 0.5], [1.0, 0.5])
        self.assertEqual(label_sequence(r, 0.01), ["diverging"])

    def test_falling_distinction_and_rising_integration_is_consolidating(self):
        r = rows_from([0.5, 0.0], [0.5, 1.0])
        self.assertEqual(label_sequence(r, 0.01), ["consolidating"])

    def test_a_wide_deadband_turns_small_steps_steady(self):
        r = rows_from([0.0, 0.005], [1.0, 0.995])
        self.assertEqual(label_sequence(r, 0.001), ["diverging"])
        self.assertEqual(label_sequence(r, 0.05), ["steady"])

    def test_widening_never_creates_a_diverging_step(self):
        """Q2 as a property: wider deadband only ever removes flags."""
        rng = np.random.default_rng(0)
        r = rows_from(rng.random(30), rng.random(30))
        counts = [label_sequence(r, d).count("diverging")
                  for d in (0.001, 0.01, 0.05, 0.2, 0.5)]
        self.assertTrue(all(a >= b for a, b in zip(counts, counts[1:])),
                        msg=str(counts))


class TestBandExtent(unittest.TestCase):

    def test_the_band_is_placed_at_step_midpoints(self):
        """A step lives between two scan points, not at one of them."""
        x = [0.0, 1.0, 2.0, 3.0]
        labels = ["contracting", "diverging", "contracting"]
        b = band_extent(labels, x)
        self.assertAlmostEqual(b["lo"], 1.5)
        self.assertAlmostEqual(b["hi"], 1.5)

    def test_a_band_spanning_several_steps_reports_its_extent(self):
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        b = band_extent(["diverging"] * 4, x)
        self.assertAlmostEqual(b["lo"], 0.5)
        self.assertAlmostEqual(b["hi"], 3.5)
        self.assertEqual(b["count"], 4)

    def test_an_absent_band_is_reported_as_absent_not_as_nan_width(self):
        b = band_extent(["steady", "steady"], [0.0, 1.0, 2.0])
        self.assertFalse(b["present"])
        self.assertEqual(b["count"], 0)
        self.assertEqual(b["width"], 0.0)

    def test_other_labels_can_be_asked_for(self):
        b = band_extent(["expanding", "steady"], [0.0, 1.0, 2.0],
                        name="expanding")
        self.assertTrue(b["present"])
        self.assertEqual(b["count"], 1)


class TestInvariantFraction(unittest.TestCase):

    def test_identical_sequences_are_fully_invariant(self):
        s = ["a", "b", "c"]
        self.assertEqual(invariant_fraction([s, list(s), list(s)]), 1.0)

    def test_one_differing_position_lowers_it(self):
        self.assertAlmostEqual(
            invariant_fraction([["a", "b", "c"], ["a", "x", "c"]]), 2 / 3)

    def test_nothing_in_common_gives_zero(self):
        self.assertEqual(invariant_fraction([["a", "a"], ["b", "b"]]), 0.0)


class TestPlateau(unittest.TestCase):
    """Q3's instrument: it has to find a one-sided boundary."""

    def band(self, lo, hi, present=True):
        return {"present": present, "lo": lo, "hi": hi,
                "width": hi - lo if present else 0.0, "count": 1}

    def test_a_setting_in_a_wide_flat_region_survives_both_ways(self):
        d = [0.001, 0.01, 0.1, 1.0]
        bands = [self.band(1.0, 2.0)] * 4
        pl = plateau(d, bands, 1)
        self.assertEqual(pl["n_settings"], 4)
        self.assertGreater(pl["factor_up"], 1.0)
        self.assertGreater(pl["factor_down"], 1.0)

    def test_a_setting_on_the_upper_boundary_reports_no_headroom(self):
        """The measured Kuramoto case: fine going down, on the edge going up."""
        d = [0.001, 0.005, 0.01, 0.02]
        bands = [self.band(1.0, 2.0), self.band(1.0, 2.0),
                 self.band(1.0, 2.0), self.band(1.5, 2.0)]
        pl = plateau(d, bands, 2)
        self.assertAlmostEqual(pl["factor_up"], 1.0)
        self.assertAlmostEqual(pl["factor_down"], 10.0)

    def test_a_setting_on_the_lower_boundary_is_caught_too(self):
        d = [0.001, 0.005, 0.01, 0.02]
        bands = [self.band(1.5, 2.0), self.band(1.0, 2.0),
                 self.band(1.0, 2.0), self.band(1.0, 2.0)]
        pl = plateau(d, bands, 1)
        self.assertAlmostEqual(pl["factor_down"], 1.0)
        self.assertAlmostEqual(pl["factor_up"], 4.0)

    def test_a_band_appearing_or_vanishing_breaks_the_plateau(self):
        d = [0.001, 0.01, 0.1]
        bands = [self.band(1.0, 2.0), self.band(1.0, 2.0),
                 self.band(0, 0, present=False)]
        pl = plateau(d, bands, 1)
        self.assertAlmostEqual(pl["factor_up"], 1.0)

    def test_an_all_absent_family_is_a_plateau_of_absence(self):
        d = [0.001, 0.01, 0.1]
        bands = [self.band(0, 0, present=False)] * 3
        self.assertEqual(plateau(d, bands, 1)["n_settings"], 3)


if __name__ == "__main__":
    unittest.main()
