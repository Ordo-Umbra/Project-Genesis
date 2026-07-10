"""Checks for the capacity-as-gravity instrument (screened κ attraction)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import (  # noqa: E402
    capacity_free_energy,
    fit_yukawa_range,
    gaussian_load,
    relax_capacity,
    screening_length,
    well_range,
)

KW = dict(kappa_diffusion=1.0, kappa_recovery=0.05, kappa_consumption=2.0,
          kappa_baseline=1.0)


class TestScreening(unittest.TestCase):

    def test_screening_length_formula(self):
        self.assertAlmostEqual(screening_length(2.0, 0.5), 2.0, places=12)
        self.assertAlmostEqual(screening_length(1.0, 0.04), 5.0, places=12)

    def test_screening_length_no_recovery_infinite(self):
        self.assertEqual(screening_length(1.0, 0.0), float("inf"))

    def test_gaussian_load_peaks_at_centers(self):
        load = gaussian_load((16, 16, 16), (8, 4, 8), width=1.5, amplitude=0.7)
        self.assertAlmostEqual(load[8, 4, 8], 0.7, places=6)
        self.assertGreater(load[8, 4, 8], load[8, 10, 8])


class TestCapacityField(unittest.TestCase):

    def test_relax_digs_a_well(self):
        # a mass depletes capacity at its center, below the baseline
        load = gaussian_load((24, 24, 24), (12, 12, 12), 1.5, 0.5)
        k = relax_capacity(load, dt=0.1, max_iters=4000, **KW)
        self.assertLess(k[12, 12, 12], 0.9)          # depleted at the mass
        self.assertGreater(k[0, 0, 0], 0.99)         # recovered far away

    def test_relax_lowers_free_energy(self):
        # the κ flow is a gradient descent of F → relaxed F below the flat start
        load = gaussian_load((24, 24, 24), (12, 12, 12), 1.5, 0.5)
        flat = np.full((24, 24, 24), 1.0)
        ekw = {k: v for k, v in KW.items()}
        F0 = capacity_free_energy(flat, load, **ekw)
        k = relax_capacity(load, dt=0.1, max_iters=4000, **KW)
        self.assertLess(capacity_free_energy(k, load, **ekw), F0)

    def test_well_range_matches_prediction(self):
        # the well's decay length is √(D/r) (within lattice/finite-width offset)
        D, r = 1.0, 0.05
        load = gaussian_load((48, 48, 48), (24, 24, 24), 1.3, 0.5)
        k = relax_capacity(load, kappa_diffusion=D, kappa_recovery=r,
                           kappa_consumption=2.0, kappa_baseline=1.0,
                           dt=0.1, max_iters=6000)
        xi = well_range((1.0 - k)[24:, 24, 24])
        self.assertAlmostEqual(xi, screening_length(D, r), delta=0.8)


class TestTwoBodyAttraction(unittest.TestCase):

    def test_two_masses_attract(self):
        # bringing two masses closer lowers F (V(near) < V(far)): attraction
        shape = (40, 40, 40)
        c = 20
        ekw = {k: v for k, v in KW.items()}

        def F(sep):
            load = (gaussian_load(shape, (c, c - sep // 2, c), 1.3, 0.5)
                    + gaussian_load(shape, (c, c + sep - sep // 2, c), 1.3, 0.5))
            return capacity_free_energy(
                relax_capacity(load, dt=0.1, max_iters=5000, **KW), load, **ekw)

        self.assertLess(F(5) - F(16), 0.0)

    def test_fit_yukawa_range_recovers_xi(self):
        # synthetic V(r) = −A e^{−r/ξ}/r → the fit recovers ξ
        xi, A = 4.0, 3.0
        r = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
        V = -A * np.exp(-r / xi) / r
        self.assertAlmostEqual(fit_yukawa_range(r, V)["xi"], xi, places=6)

    def test_fit_yukawa_range_needs_attraction(self):
        with self.assertRaises(ValueError):
            fit_yukawa_range([3.0, 5.0], [0.1, 0.2])   # no V<0 points


if __name__ == "__main__":
    unittest.main()
