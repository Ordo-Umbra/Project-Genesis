"""Checks for the dimensionless topological invariant of the one-κ obstruction."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.topological_charge import charge_variance_per_action  # noqa: E402


class TestChargeVariancePerAction(unittest.TestCase):

    def test_ratio_value(self):
        # ⟨Q²⟩ / ⟨S⟩ = mean(Q²) / mean(S)
        q = [1.0, -1.0, 2.0, 0.0]           # ⟨Q²⟩ = (1+1+4+0)/4 = 1.5
        s = [10.0, 10.0, 10.0, 10.0]        # ⟨S⟩ = 10
        self.assertAlmostEqual(charge_variance_per_action(q, s), 0.15, places=12)

    def test_zero_action_guard(self):
        self.assertEqual(charge_variance_per_action([1.0, 2.0], [0.0, 0.0]), 0.0)

    def test_dimensionless_scale_invariance_in_Q(self):
        # scaling all charges by c scales the invariant by c²
        rng = np.random.default_rng(0)
        q = rng.standard_normal(50); s = 5.0 + rng.random(50)
        base = charge_variance_per_action(q, s)
        scaled = charge_variance_per_action(3.0 * q, s)
        self.assertAlmostEqual(scaled, 9.0 * base, places=10)

    def test_uses_variance_not_mean(self):
        # symmetric charges: ⟨Q⟩ = 0 but ⟨Q²⟩ > 0, so the invariant is positive
        self.assertGreater(charge_variance_per_action([2.0, -2.0], [4.0, 4.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
