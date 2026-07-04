"""Fast checks for experiments/n3_potts_transition.py."""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_potts_transition import (  # noqa: E402
    POTTS_GAMMA_OVER_NU,
    binder_crossing,
    potts_magnetisation,
    run_point,
)


class TestPottsMagnetisation(unittest.TestCase):

    def test_fully_ordered_gives_one(self):
        amps = np.zeros((3, 8, 8))
        amps[1] = 1.0  # every site in sector 1
        self.assertAlmostEqual(potts_magnetisation(amps), 1.0, places=12)

    def test_symmetric_disorder_gives_near_zero(self):
        rng = np.random.default_rng(0)
        amps = rng.random((3, 64, 64))
        m = potts_magnetisation(amps)
        self.assertLess(abs(m), 0.05)

    def test_exact_reference_value(self):
        self.assertAlmostEqual(POTTS_GAMMA_OVER_NU, 26.0 / 15.0, places=15)


class TestBinderCrossing(unittest.TestCase):

    def test_finds_interpolated_crossing(self):
        temps = [0.1, 0.2, 0.3]
        results = {
            32: [{"binder": 0.6}, {"binder": 0.4}, {"binder": 0.2}],
            48: [{"binder": 0.65}, {"binder": 0.3}, {"binder": 0.1}],
        }
        # diff = u1 - u2: -0.05, +0.1 → crossing between 0.1 and 0.2
        t_c = binder_crossing(results, [32, 48], temps)
        self.assertGreater(t_c, 0.1)
        self.assertLess(t_c, 0.2)

    def test_no_crossing_gives_nan(self):
        temps = [0.1, 0.2]
        results = {
            32: [{"binder": 0.6}, {"binder": 0.5}],
            48: [{"binder": 0.5}, {"binder": 0.4}],
        }
        self.assertTrue(math.isnan(binder_crossing(results, [32, 48], temps)))


class TestRunPoint(unittest.TestCase):

    def test_cold_ordered_and_hot_disordered(self):
        """Micro version of the transition: m near 1 cold, near 0 hot."""
        common = dict(n_palette=3, size=10, beta_g=3.0, matter_coupling=4.0,
                      quartic_u=1.0, n_therm=60, n_meas=30, n_skip=2,
                      bin_size=5, step_scale=0.3)
        cold = run_point(temperature=0.03, seed=1, **common)
        hot = run_point(temperature=1.0, seed=2, **common)
        self.assertGreater(cold["magnetisation"], 0.5,
                           f"cold m={cold['magnetisation']:.3f}")
        self.assertLess(hot["magnetisation"], 0.3,
                        f"hot m={hot['magnetisation']:.3f}")
        for pt in (cold, hot):
            self.assertGreaterEqual(pt["susceptibility"], 0.0)
            self.assertTrue(np.isfinite(pt["binder"]))


if __name__ == "__main__":
    unittest.main()
