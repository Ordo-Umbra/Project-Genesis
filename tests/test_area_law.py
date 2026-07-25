"""Checks for the scaling-dimension instrument (area_law)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.area_law import (  # noqa: E402
    correlation_length,
    cross_boundary_mi,
    ensemble_correlation,
    gaussian_control_field,
    mi_kernel,
    radial_profile,
    region_content,
    region_indicator,
    scaling_exponent,
)


class TestPrimitives(unittest.TestCase):

    def test_region_indicator_area(self):
        chi = region_indicator(64, 8)
        self.assertAlmostEqual(chi.sum(), (2 * 8) ** 2)

    def test_region_content_sums_inside_only(self):
        d = np.ones((64, 64))
        self.assertAlmostEqual(region_content(d, 10), (2 * 10) ** 2)

    def test_scaling_exponent_recovers_power(self):
        halves = [4, 8, 16, 32]
        for n in (1.0, 1.5, 2.0):
            vals = [h ** n for h in halves]
            self.assertAlmostEqual(scaling_exponent(halves, vals), n, places=6)

    def test_radial_profile_peaks_at_zero(self):
        a = np.zeros((32, 32))
        a[0, 0] = 1.0
        prof = radial_profile(a)
        self.assertAlmostEqual(prof[0], 1.0)
        self.assertAlmostEqual(prof[5], 0.0)


class TestCorrelation(unittest.TestCase):

    def test_control_field_has_requested_scale(self):
        rng = np.random.default_rng(0)
        narrow = ensemble_correlation(
            [gaussian_control_field(96, 1.5, rng) for _ in range(8)])
        wide = ensemble_correlation(
            [gaussian_control_field(96, 5.0, rng) for _ in range(8)])
        self.assertGreater(correlation_length(wide), correlation_length(narrow))

    def test_ensemble_averaging_lowers_the_noise_floor(self):
        # the floor is what fakes a volume law; averaging must suppress it
        rng = np.random.default_rng(1)
        _, floor_1 = mi_kernel(ensemble_correlation(
            [gaussian_control_field(96, 2.0, rng)]))
        _, floor_n = mi_kernel(ensemble_correlation(
            [gaussian_control_field(96, 2.0, rng) for _ in range(16)]))
        self.assertLess(floor_n, floor_1)

    def test_kernel_is_non_negative(self):
        rng = np.random.default_rng(2)
        kern, _ = mi_kernel(ensemble_correlation(
            [gaussian_control_field(64, 2.0, rng) for _ in range(4)]))
        self.assertGreaterEqual(kern.min(), 0.0)


class TestAreaLawOnControl(unittest.TestCase):
    """The calibration claim: a known area law must be recovered."""

    def _exponent(self, xi, seeds, n=128, halves=(18, 24, 30, 36)):
        rng = np.random.default_rng(3)
        g = ensemble_correlation(
            [gaussian_control_field(n, xi, rng) for _ in range(seeds)])
        kern, _ = mi_kernel(g)
        return scaling_exponent(halves, [cross_boundary_mi(kern, h)
                                         for h in halves])

    def test_ensemble_recovers_area_law_within_bias(self):
        # truth is 1.0; the estimator is biased high but must be close
        n_i = self._exponent(2.0, seeds=24)
        self.assertLess(abs(n_i - 1.0), 0.45)

    def test_more_seeds_reduce_the_bias(self):
        few = abs(self._exponent(2.0, seeds=2) - 1.0)
        many = abs(self._exponent(2.0, seeds=24) - 1.0)
        self.assertLess(many, few)

    def test_single_realisation_is_biased_high(self):
        # the trap this instrument exists to avoid: one seed fakes a volume term
        self.assertGreater(self._exponent(2.0, seeds=1), 1.3)


class TestVolumeLawControl(unittest.TestCase):

    def test_uniform_density_is_volume_law(self):
        # a space-filling density must read the volume exponent d = 2
        d = np.ones((128, 128))
        halves = [18, 24, 30, 36]
        n_c = scaling_exponent(halves, [region_content(d, h) for h in halves])
        self.assertAlmostEqual(n_c, 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
