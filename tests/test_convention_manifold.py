"""Tests for the convention-family dimension reading.

This instrument exists to answer "is the arbitrariness one axis or an open
set", and the calibration cases below are the reason its verdict can be
trusted: each is a synthetic family whose true answer is known by construction.
The one that matters most is the two-knob case where the second knob changes
the peak's *width* — the family is genuinely two-dimensional, but its peak
location is not, and the instrument has to report the consequence rather than
the family or the experiment's claim would be wrong in a way nobody would
notice.

Two degenerate cases are pinned because both bit during development: a family
of identical shapes at different levels leaves numerical dust that a naive
ratio turns into confident-looking structure, and a target quantised to the
scan grid produces a meaningless r-squared driven entirely by which side of a
tie one outlier fell.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.convention_manifold import (  # noqa: E402
    axis_is_monotone,
    curve_matrix,
    explained_by,
    principal_axes,
    shape_normalise,
)

X = np.linspace(0.0, 2.0, 20)
GRID = float(np.mean(np.diff(X)))


def sliding(positions, width=0.35):
    return [np.exp(-((X - p) / width) ** 2) for p in positions]


def peaks_of(curves):
    return [float(X[int(np.argmax(c))]) for c in curves]


def analyse(curves, grid=None):
    pa = principal_axes(curve_matrix(curves))
    return pa, explained_by(pa["scores"], peaks_of(curves), grid=grid)


class TestCalibration(unittest.TestCase):
    """Synthetic families whose dimension is known by construction."""

    def test_one_knob_family_is_functionally_one_dimensional(self):
        pa, ex = analyse(sliding(np.linspace(0.6, 1.4, 7)))
        self.assertGreater(ex["r2_pc1"], 0.90)
        self.assertLess(ex["gain"], 0.05)

    def test_independent_random_curves_are_not(self):
        rng = np.random.default_rng(0)
        _, ex = analyse([rng.random(X.size) for _ in range(7)])
        self.assertLess(ex["r2_pc1"], 0.60)
        self.assertGreater(ex["gain"], 0.20)

    def test_the_variance_ratio_alone_would_misjudge_the_one_knob_family(self):
        """Why the verdict is functional and not the PCA ratio: a curved
        one-dimensional family spends variance on several components."""
        pa, ex = analyse(sliding(np.linspace(0.6, 1.4, 7)))
        self.assertLess(pa["pc1"], 0.80)      # would fail a naive 0.8 bar
        self.assertGreater(ex["r2_pc1"], 0.90)   # while the truth is 1-D

    def test_a_second_knob_that_never_moves_the_peak_reads_as_one_dimensional(self):
        """Genuinely 2-D in shape, 1-D in consequence — the claim is about the
        consequence, and the instrument must not conflate them."""
        curves = [np.exp(-((X - (0.8 + 0.2 * i)) / (0.2 + 0.15 * j)) ** 2)
                  for i in range(3) for j in range(3)]
        _, ex = analyse(curves)
        self.assertGreater(ex["r2_pc1"], 0.80)
        self.assertLess(ex["gain"], 0.05)

    def test_two_knobs_along_one_axis_stay_one_dimensional(self):
        curves = sliding([0.7 + 0.1 * i + 0.05 * j
                          for i in range(3) for j in range(3)])
        _, ex = analyse(curves)
        self.assertGreater(ex["r2_pc1"], 0.80)


class TestDegenerateFamilies(unittest.TestCase):
    """Both of these produced confident nonsense before being handled."""

    def test_level_only_changes_are_reported_degenerate_not_scored(self):
        curves = [3.0 * np.exp(-((X - 1.0) / 0.35) ** 2) * a
                  for a in (0.2, 0.5, 1.0, 2.0, 5.0)]
        pa = principal_axes(curve_matrix(curves))
        self.assertTrue(pa["degenerate"])
        self.assertTrue(np.isnan(pa["pc1"]))

    def test_shape_normalise_makes_level_changes_identical(self):
        base = np.exp(-((X - 1.0) / 0.35) ** 2)
        m = shape_normalise(curve_matrix([base * a for a in (0.2, 1.0, 7.0)]))
        self.assertTrue(np.allclose(m[0], m[1]))
        self.assertTrue(np.allclose(m[1], m[2]))

    def test_a_flat_curve_normalises_to_zero_rather_than_noise(self):
        m = shape_normalise(curve_matrix([np.ones(X.size), np.arange(X.size)]))
        self.assertTrue(np.allclose(m[0], 0.0))

    def test_a_target_within_one_grid_step_is_flagged_invariant(self):
        """The Kuramoto case: r2 would report a confident failure on variance
        that does not exist."""
        curves = sliding([1.00, 1.00, 1.00, 1.00, 1.09])
        ex = explained_by(principal_axes(curve_matrix(curves))["scores"],
                          peaks_of(curves), grid=GRID)
        self.assertTrue(ex["invariant"])
        self.assertTrue(np.isnan(ex["r2_pc1"]))
        self.assertLessEqual(ex["spread"], GRID * 1.001)

    def test_a_target_that_moves_further_is_scored_normally(self):
        curves = sliding(np.linspace(0.6, 1.4, 7))
        ex = explained_by(principal_axes(curve_matrix(curves))["scores"],
                          peaks_of(curves), grid=GRID)
        self.assertFalse(ex["invariant"])
        self.assertTrue(np.isfinite(ex["r2_pc1"]))

    def test_a_constant_target_is_invariant_without_a_grid(self):
        curves = sliding([1.0] * 5)
        ex = explained_by(principal_axes(curve_matrix(curves))["scores"],
                          peaks_of(curves))
        self.assertTrue(ex["invariant"])


class TestMonotoneAxis(unittest.TestCase):

    def test_a_swept_knob_gives_a_monotone_axis(self):
        params = np.linspace(0.6, 1.4, 7)
        pa = principal_axes(curve_matrix(sliding(params)))
        self.assertTrue(axis_is_monotone(pa["scores"][:, 0], params)["monotone"])

    def test_a_shuffled_family_does_not(self):
        rng = np.random.default_rng(3)
        pa = principal_axes(curve_matrix([rng.random(X.size) for _ in range(9)]))
        self.assertFalse(
            axis_is_monotone(pa["scores"][:, 0], np.arange(9))["monotone"])

    def test_monotone_is_sign_agnostic(self):
        params = np.linspace(0.6, 1.4, 7)
        pa = principal_axes(curve_matrix(sliding(params)))
        flipped = axis_is_monotone(-pa["scores"][:, 0], params)
        self.assertTrue(flipped["monotone"])


class TestMatrixHandling(unittest.TestCase):

    def test_ragged_curves_are_rejected(self):
        with self.assertRaises(ValueError):
            curve_matrix([np.zeros(4), np.zeros(5)])

    def test_scores_have_one_row_per_convention(self):
        pa = principal_axes(curve_matrix(sliding(np.linspace(0.6, 1.4, 7))))
        self.assertEqual(pa["scores"].shape[0], 7)

    def test_explained_variance_sums_to_one(self):
        pa = principal_axes(curve_matrix(sliding(np.linspace(0.6, 1.4, 7))))
        self.assertAlmostEqual(float(np.sum(pa["explained"])), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
