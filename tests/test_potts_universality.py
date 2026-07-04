"""Fast checks for the ν-collapse and 3-D first-order analysis helpers."""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_potts_nu import collapse_residual, fit_collapse  # noqa: E402
from n3_potts_3d import binder_minima, hysteresis_summary  # noqa: E402


def _synthetic_results(t_c, nu, sizes, temps, noise=0.0, seed=0):
    """U₄ curves obeying exact single-parameter scaling f(x) = 0.6/(1+e^x)."""
    rng = np.random.default_rng(seed)
    results = {}
    for L in sizes:
        pts = []
        for t in temps:
            x = (t - t_c) * L ** (1.0 / nu)
            u = 0.6 / (1.0 + math.exp(x))
            pts.append({"temperature": t,
                        "binder": u + noise * rng.standard_normal(),
                        "binder_err": max(noise, 0.01)})
        results[L] = pts
    return results


class TestCollapse(unittest.TestCase):

    def test_exact_scaling_recovers_parameters(self):
        """A dense T grid recovers (T_c, ν); note the estimator carries an
        interpolation bias on coarse grids (~0.1 in ν for 5 points), which
        is why the experiments quote a band, not a point estimate."""
        t_c_true, nu_true = 0.11, 0.85
        sizes = [16, 24, 32, 48]
        temps = list(np.linspace(0.06, 0.16, 11))
        results = _synthetic_results(t_c_true, nu_true, sizes, temps)
        fit = fit_collapse(results, sizes,
                           t_c_grid=np.linspace(0.08, 0.14, 25),
                           nu_grid=np.linspace(0.4, 1.6, 49))
        self.assertAlmostEqual(fit["t_c"], t_c_true, delta=0.01)
        self.assertAlmostEqual(fit["nu"], nu_true, delta=0.1)

    def test_wrong_nu_has_larger_residual(self):
        sizes = [16, 32, 48]
        temps = [0.07, 0.09, 0.11, 0.13, 0.15]
        results = _synthetic_results(0.11, 0.85, sizes, temps)
        good = collapse_residual(results, sizes, 0.11, 0.85)
        bad = collapse_residual(results, sizes, 0.11, 2.5)
        self.assertLess(good, bad / 5.0)


class TestFirstOrderHelpers(unittest.TestCase):

    def _mk(self, m_cold, m_hot, binders):
        temps = [0.1, 0.2, 0.3]
        res = {"cold": {8: []}, "hot": {8: []}}
        for t, mc, mh, b in zip(temps, m_cold, m_hot, binders):
            res["cold"][8].append({"temperature": t, "magnetisation": mc,
                                   "magnetisation_err": 0.01, "binder": b,
                                   "binder_err": 0.02})
            res["hot"][8].append({"temperature": t, "magnetisation": mh,
                                  "magnetisation_err": 0.01, "binder": 0.4,
                                  "binder_err": 0.02})
        return res, temps

    def test_hysteresis_window_detected(self):
        res, temps = self._mk([0.9, 0.8, 0.05], [0.9, 0.05, 0.05],
                              [0.6, 0.1, 0.4])
        h = hysteresis_summary(res, [8], temps)
        self.assertEqual(h["8"]["n_significant"], 1)
        self.assertEqual(h["8"]["window"], [0.2, 0.2])

    def test_no_window_when_branches_agree(self):
        res, temps = self._mk([0.9, 0.5, 0.05], [0.89, 0.49, 0.05],
                              [0.6, 0.3, 0.4])
        h = hysteresis_summary(res, [8], temps)
        self.assertIsNone(h["8"]["window"])

    def test_binder_minimum_found(self):
        res, temps = self._mk([0.9, 0.8, 0.05], [0.9, 0.05, 0.05],
                              [0.6, -0.3, 0.4])
        mn = binder_minima(res, [8])
        self.assertAlmostEqual(mn["8"]["binder_min"], -0.3, places=12)
        self.assertEqual(mn["8"]["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
