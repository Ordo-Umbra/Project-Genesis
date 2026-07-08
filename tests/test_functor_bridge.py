"""Checks for the functor-bridge helpers."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_functor_bridge import (  # noqa: E402
    coherence,
    path_independence,
    reflection_ladder,
    summarise,
)


class TestCoherence(unittest.TestCase):

    def test_uniform_field_is_fully_coherent(self):
        psi = np.zeros((6, 6, 3), dtype=complex)
        psi[..., 0] = 1.0
        self.assertAlmostEqual(coherence(psi), 1.0, places=12)

    def test_bounded_in_unit_interval(self):
        rng = np.random.default_rng(0)
        psi = rng.standard_normal((8, 8, 3)) + 1j * rng.standard_normal((8, 8, 3))
        psi /= np.linalg.norm(psi, axis=-1, keepdims=True)
        c = coherence(psi)
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)


class TestReflectionLadder(unittest.TestCase):

    def _ladder(self):
        rng = np.random.default_rng(1)
        psi = rng.standard_normal((16, 16, 3)) + 1j * rng.standard_normal((16, 16, 3))
        psi /= np.linalg.norm(psi, axis=-1, keepdims=True)
        return reflection_ladder(psi, rate=0.4, levels=8, block=1)

    def test_integration_climbs_monotonically(self):
        lad = self._ladder()
        I = [r["integration"] for r in lad]
        self.assertTrue(all(I[i + 1] >= I[i] - 1e-9 for i in range(len(I) - 1)))
        self.assertGreater(I[-1], I[0])          # it really climbs

    def test_topology_falls_as_integration_climbs(self):
        lad = self._ladder()
        # contravariant: the instanton gas is annihilated by cooling
        self.assertLess(lad[-1]["inst_density"], lad[0]["inst_density"])


class TestPathIndependence(unittest.TestCase):

    def test_matched_I_agrees_across_rates(self):
        # T(I) should be nearly schedule-independent → small relative spread
        pind = path_independence(20, [0.25, 0.6], levels=12, block=1,
                                 trials=2, seed=5)
        self.assertLess(pind["relative_spread"], 0.15)
        self.assertIn("grid", pind)


class TestSummarise(unittest.TestCase):

    def test_verdict_reports_functor_and_contravariance(self):
        lad = [{"level": 0, "integration": 0.33, "inst_density": 0.22,
                "net_charge": 3.0, "action": 0.9},
               {"level": 1, "integration": 0.90, "inst_density": 0.02,
                "net_charge": 1.0, "action": 0.3}]
        pind = {"relative_spread": 0.03, "curves": {}, "grid": [], "resampled": {}}
        text = "\n".join(summarise(lad, pind, [0.2, 0.6]))
        self.assertIn("CONTRAVARIANT", text)
        self.assertIn("genuine functor", text)
        self.assertIn("naturality", text)


if __name__ == "__main__":
    unittest.main()
