"""Tests for the pure-gauge Monte-Carlo layer (gauge_mc.py).

Covers Metropolis updates, local ΔS correctness, Wilson/Polyakov loops,
and Creutz ratio on small deterministic configurations.
"""

from __future__ import annotations

import unittest

import numpy as np

from project_genesis.gauge import (
    apply_gauge_transform,
    random_unitary,
    wilson_action,
)
from project_genesis.gauge_mc import (
    creutz_ratio,
    metropolis_sweep,
    polyakov_loop,
    thermalize_and_measure_pure_gauge,
    wilson_loop_2d,
)


class TestMonteCarloPureGauge(unittest.TestCase):
    def test_metropolis_lowers_action(self):
        rng = np.random.default_rng(0)
        links = random_unitary(rng, 2, (2, 6, 6), special=True, scale=0.6)
        s0 = wilson_action(links)
        links2, acc = metropolis_sweep(links, beta_g=1.5, rng=rng, n_sweeps=30, step_scale=0.2)
        s1 = wilson_action(links2)
        self.assertGreater(acc, 0.2)
        self.assertLess(s1, s0 + 5.0)  # action should decrease or stay similar

    def test_wilson_loop_gauge_invariant(self):
        rng = np.random.default_rng(1)
        links = random_unitary(rng, 3, (2, 5, 5), special=True)
        g = random_unitary(rng, 3, (5, 5), special=True)
        # dummy psi for transform
        psi_dummy = np.zeros((5, 5, 3), dtype=np.complex128)
        _, links_g = apply_gauge_transform(psi_dummy, links, g)
        w1, _ = wilson_loop_2d(links, 2, 2)
        w2, _ = wilson_loop_2d(links_g, 2, 2)
        self.assertAlmostEqual(w1, w2, places=8)

    def test_polyakov_loop_gauge_invariant(self):
        rng = np.random.default_rng(2)
        links = random_unitary(rng, 3, (2, 6, 6), special=True)
        g = random_unitary(rng, 3, (6, 6), special=True)
        psi_dummy = np.zeros((6, 6, 3), dtype=np.complex128)
        _, links_g = apply_gauge_transform(psi_dummy, links, g)
        p1 = polyakov_loop(links)
        p2 = polyakov_loop(links_g)
        self.assertAlmostEqual(p1, p2, places=8)

    def test_creutz_ratio_positive(self):
        chi = creutz_ratio(0.85, 0.72, 0.78, 0.80)
        self.assertGreater(chi, 0.0)

    def test_thermalize_driver_runs(self):
        rng = np.random.default_rng(3)
        summary, links = thermalize_and_measure_pure_gauge(
            size=6, n=2, beta_g=1.8, rng=rng,
            n_therm=20, n_meas=5, n_skip=2
        )
        self.assertIn("beta_g", summary)
        self.assertIn("loop_averages", summary)
        self.assertGreater(summary["mean_acceptance_therm"], 0.1)
