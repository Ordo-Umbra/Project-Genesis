"""Tests for gauge_mc v2."""

from __future__ import annotations

import unittest

import numpy as np

from project_genesis.gauge_mc import (
    metropolis_sweep,
    heatbath_sweep,
    overrelaxation_sweep,
    wilson_loop,
    polyakov_loop,
    creutz_ratio,
    thermalize_and_measure_pure_gauge,
)
from project_genesis.gauge import random_unitary, wilson_action


class TestMonteCarloV2(unittest.TestCase):
    def test_ndim2_metropolis_runs(self):
        rng = np.random.default_rng(0)
        links = random_unitary(rng, 2, (2, 6, 6), special=True)
        links2, acc = metropolis_sweep(links, 1.5, rng, n_sweeps=10)
        self.assertGreater(acc, 0.1)

    def test_heatbath_runs_su2_su3(self):
        rng = np.random.default_rng(1)
        for n in (2, 3):
            links = random_unitary(rng, n, (2, 4, 4), special=True)
            links2, _ = heatbath_sweep(links, 2.0, rng, n_sweeps=5)
            self.assertEqual(links2.shape, links.shape)

    def test_overrelax_runs(self):
        rng = np.random.default_rng(2)
        links = random_unitary(rng, 3, (2, 5, 5), special=True)
        links2, _ = overrelaxation_sweep(links, rng, n_sweeps=5)
        self.assertEqual(links2.shape, links.shape)

    def test_wilson_loop_ndim2(self):
        rng = np.random.default_rng(3)
        links = random_unitary(rng, 2, (2, 6, 6), special=True)
        w = wilson_loop(links, (2, 2))
        self.assertGreaterEqual(w, -1.0)
        self.assertLessEqual(w, 1.0)

    def test_driver_v2(self):
        rng = np.random.default_rng(4)
        for updater in ["metropolis", "heatbath", "overrelax"]:
            summary, _ = thermalize_and_measure_pure_gauge(
                size=5, n=2, beta_g=1.8, rng=rng, updater=updater,
                n_therm=10, n_meas=5
            )
            self.assertIn("updater", summary)
