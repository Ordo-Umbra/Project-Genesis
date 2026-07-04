"""Fast checks for experiments/n3_phase_boundary.py and the 3-D path.

Exercises the boundary-locating helpers on synthetic data and the 3-D
sector-network / junction-measure path at CI scale.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_phase_boundary import (  # noqa: E402
    half_value_temperature,
    peak_temperature,
)
from n3_thermal_selection import build_sector_network  # noqa: E402
from project_genesis.annealed_matter import (  # noqa: E402
    sector_amplitudes,
    thermal_junction_analysis,
)


class TestBoundaryHelpers(unittest.TestCase):

    def test_half_value_interpolates(self):
        temps = [0.1, 0.2, 0.3, 0.4]
        vals = [1.0, 0.8, 0.4, 0.0]
        # crosses 0.5 between 0.2 and 0.3: 0.2 + 0.1·(0.8−0.5)/(0.8−0.4)
        t = half_value_temperature(temps, vals)
        self.assertAlmostEqual(t, 0.275, places=10)

    def test_half_value_never_crossing_is_inf(self):
        self.assertTrue(math.isinf(
            half_value_temperature([0.1, 0.2], [1.0, 0.9])))

    def test_half_value_zero_reference_is_nan(self):
        self.assertTrue(math.isnan(
            half_value_temperature([0.1, 0.2], [0.0, 0.0])))

    def test_peak_temperature_flags_edges(self):
        temps = [0.1, 0.2, 0.3]
        t, h, edge = peak_temperature(temps, [1.0, 3.0, 2.0])
        self.assertEqual((t, h, edge), (0.2, 3.0, False))
        t, h, edge = peak_temperature(temps, [3.0, 2.0, 1.0])
        self.assertTrue(edge)
        t, h, edge = peak_temperature(temps, [1.0, 2.0, 3.0])
        self.assertTrue(edge)


class Test3DNetworkPath(unittest.TestCase):

    def test_3d_networks_and_junction_measure(self):
        """3-D: cold P=3 carries a dense junction-line signal; P=4 is
        strongly *suppressed* (not exactly zero — the 3-D result is
        dominance via sparse quadruple points, unlike 2-D exclusivity) and
        pure noise scores exactly zero.

        16³ is the smallest lattice where the conserved dynamics reliably
        forms a 3-D junction network (12³ collapses to layered slabs).
        """
        nets = {}
        for P in (3, 4):
            nets[P] = build_sector_network(
                P, size=16, steps=400, gamma=1.5, diffusion=1.0, dt=0.1,
                seed=7, beta=0.09, dim=3)
            self.assertEqual(nets[P]["psi"].shape, (16, 16, 16, P))
        tj3 = thermal_junction_analysis(
            sector_amplitudes(nets[3]["psi"].astype(complex)), 3)
        tj4 = thermal_junction_analysis(
            sector_amplitudes(nets[4]["psi"].astype(complex)), 4)
        self.assertGreater(tj3["neutrality"], 0.1,
                           "3-D triple lines should give a dense signal")
        self.assertLess(tj4["neutrality"], tj3["neutrality"] / 3.0,
                        "P=4 full-palette density should be strongly "
                        "suppressed relative to P=3 in 3-D")

        from project_genesis.gauge import random_psi
        rng = np.random.default_rng(0)
        noise = sector_amplitudes(random_psi(rng, 3, (16, 16, 16)))
        self.assertLess(
            thermal_junction_analysis(noise, 3)["neutrality"], 1e-3)


if __name__ == "__main__":
    unittest.main()
