"""Checks for the Gauss-law / boundary-reconstruction instrument."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.boundary_gravity import (  # noqa: E402
    disc_indicator, enclosed_flux, enclosed_mass, flatness, gauss_ratio_profile)
from project_genesis.capacity_gravity import (  # noqa: E402
    gaussian_load, relax_capacity, screening_length)

KW = dict(kappa_diffusion=1.0, kappa_consumption=2.0, kappa_baseline=1.0)


def _relax(n, recovery, amp=0.4, width=4.0):
    load = gaussian_load((n, n), [(n / 2, n / 2)], width, amp)
    return load, relax_capacity(load, kappa_recovery=recovery, dt=0.05,
                                max_iters=20000, **KW)


class TestPrimitives(unittest.TestCase):

    def test_disc_area(self):
        chi = disc_indicator((128, 128), 20.0)
        self.assertAlmostEqual(chi.sum() / (np.pi * 400), 1.0, delta=0.05)

    def test_flux_of_constant_field_is_zero(self):
        # a uniform field has no gradient, hence no flux through any surface
        self.assertAlmostEqual(enclosed_flux(np.full((64, 64), 0.7), 15.0), 0.0,
                               places=9)

    def test_enclosed_mass_grows_then_saturates(self):
        load, _ = _relax(96, 0.05)
        small = enclosed_mass(load, 6.0)
        big = enclosed_mass(load, 40.0)
        self.assertGreater(big, small)
        self.assertAlmostEqual(big, float(load.sum()), delta=0.05 * float(load.sum()))

    def test_flatness_is_one_for_constant(self):
        self.assertAlmostEqual(flatness([0.3, 0.3, 0.3]), 1.0, places=9)


class TestNoGaussLawWhenScreened(unittest.TestCase):
    """The headline negative: the boundary does not know the interior."""

    def test_ratio_decays_with_surface_radius(self):
        load, kappa = _relax(128, 0.2)          # xi ~ 2.2, strongly screened
        prof = gauss_ratio_profile(kappa, load, [10, 18, 26])
        self.assertGreater(prof[0], prof[1])
        self.assertGreater(prof[1], prof[2])
        self.assertGreater(flatness(prof), 5.0)   # nowhere near a Gauss law

    def test_arrangement_changes_the_boundary_reading(self):
        n = 128
        mid = n / 2
        one = gaussian_load((n, n), [(mid, mid)], 4.0, 0.4)
        four = sum(gaussian_load((n, n), [c], 4.0, 0.1) for c in
                   [(mid-10, mid-10), (mid+10, mid-10),
                    (mid-10, mid+10), (mid+10, mid+10)])
        self.assertAlmostEqual(float(one.sum()), float(four.sum()), delta=1e-6)
        k1 = relax_capacity(one, kappa_recovery=0.05, dt=0.05, max_iters=20000, **KW)
        k4 = relax_capacity(four, kappa_recovery=0.05, dt=0.05, max_iters=20000, **KW)
        r1 = enclosed_flux(k1, 30.0) / enclosed_mass(one, 30.0)
        r4 = enclosed_flux(k4, 30.0) / enclosed_mass(four, 30.0)
        self.assertGreater(abs(r4 / r1), 3.0)     # same mass, different reading


class TestGaussLawRestored(unittest.TestCase):
    """Removing the screening restores the boundary's knowledge."""

    def test_flatness_improves_as_screening_is_removed(self):
        radii = [10, 18, 26]
        load_s, k_s = _relax(160, 0.2)          # xi ~ 2
        load_w, k_w = _relax(160, 0.001)        # xi ~ 32
        tight = flatness(gauss_ratio_profile(k_s, load_s, radii))
        loose = flatness(gauss_ratio_profile(k_w, load_w, radii))
        self.assertLess(loose, tight / 10.0)

    def test_unscreened_ratio_is_nearly_flat(self):
        load, kappa = _relax(160, 0.001)        # xi >> surface radii
        prof = gauss_ratio_profile(kappa, load, [10, 18, 26])
        self.assertLess(flatness(prof), 2.0)    # a Gauss law, recovered

    def test_screening_length_matches_formula(self):
        self.assertAlmostEqual(screening_length(1.0, 0.01), 10.0, places=6)


if __name__ == "__main__":
    unittest.main()
