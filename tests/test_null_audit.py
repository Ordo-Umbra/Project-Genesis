"""Tests for the null audit of URP's measured claims.

Pins the facts the audit's readings rest on: the cliff's φ(3) = 1 is an
identity that holds on pure noise, φ(4) < 1 needs a tessellation, the smoothed
Voronoi null is a genuine partition of the box, and the eviction verdict
agrees with the ladder on random curves, which is what makes the transplants'
toggle a property of the functional rather than of the substrates.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from n3_null_audit import (  # noqa: E402
    eviction_verdicts, integrated_fraction, random_scan, smoothed_voronoi,
)


class CliffTests(unittest.TestCase):
    def test_phi_is_one_at_three_on_pure_noise(self):
        rng = np.random.default_rng(0)
        self.assertEqual(integrated_fraction(rng.integers(0, 3, (40, 40)), 3), 1.0)
        self.assertEqual(integrated_fraction(rng.integers(0, 3, (12, 12, 12)), 3), 1.0)

    def test_noise_does_not_collapse_at_four(self):
        rng = np.random.default_rng(1)
        self.assertGreater(integrated_fraction(rng.integers(0, 4, (60, 60)), 4), 0.5)


class SmoothedVoronoiTests(unittest.TestCase):
    def test_components_partition_the_box(self):
        f = smoothed_voronoi(48, 8, 1.0, 3, np.random.default_rng(2))
        self.assertEqual(f.shape, (3, 48, 48))
        np.testing.assert_allclose(f.sum(axis=0), 1.0, atol=1e-9)


class EvictionIdentityTests(unittest.TestCase):
    def test_driven_random_curves_are_always_evicted(self):
        rng = np.random.default_rng(3)
        for _ in range(15):
            rule, ladder = eviction_verdicts(random_scan(rng, free=False, weight=1.0), 1.0)
            self.assertTrue(rule)
            self.assertTrue(ladder)

    def test_free_random_curves_are_never_evicted(self):
        rng = np.random.default_rng(4)
        for _ in range(15):
            rule, ladder = eviction_verdicts(random_scan(rng, free=True, weight=1.0), 1.0)
            self.assertFalse(rule)
            self.assertFalse(ladder)


if __name__ == "__main__":
    unittest.main()
