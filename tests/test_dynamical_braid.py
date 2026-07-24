"""Checks for the dynamical-braid instruments (co-evolved κ-pinned defects)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.spin_statistics import (  # noqa: E402
    dynamical_braid,
    gaussian_wells,
    transport_defect,
)
from project_genesis.vortex_chiral import imprint_vortices, winding_number  # noqa: E402
from project_genesis.nematic_spinor import director_holonomy  # noqa: E402
from project_genesis.two_field import chiral_detuning, step_chiral_detuned  # noqa: E402


class TestGaussianWells(unittest.TestCase):

    def test_dip_at_centre_baseline_far(self):
        k = gaussian_wells((60, 60), [(30, 30)], width=4.0, depth=0.9)
        self.assertLess(k[30, 30], 0.2)          # deep hole at the well
        self.assertAlmostEqual(k[5, 5], 1.0, places=2)   # baseline far away

    def test_two_wells(self):
        k = gaussian_wells((60, 60), [(20, 30), (40, 30)], width=4.0, depth=0.9)
        self.assertLess(k[20, 30], 0.2)
        self.assertLess(k[40, 30], 0.2)
        self.assertGreater(k[30, 30], 0.5)       # midpoint mostly recovered


class TestStaticPinning(unittest.TestCase):
    """A pinned ½ keeps its winding and −1 holonomy under co-evolution."""

    def test_static_defect_survives(self):
        n = 90
        c = (n / 2, n / 2)
        g = chiral_detuning(gaussian_wells((n, n), [c], 5.0, 0.9),
                            detune_gamma=0.85)
        psi = imprint_vortices((n, n), [c], [1], core=3.0, detune=g)
        for _ in range(500):
            psi = step_chiral_detuned(psi, g, dt=0.08)
        self.assertAlmostEqual(abs(winding_number(psi, c, 6)), 1.0, delta=0.3)
        self.assertLess(director_holonomy(psi, c, radius=10)[0], -0.5)


class TestTransportThreshold(unittest.TestCase):
    """Adiabatic transport: slow keeps the winding, fast unwinds it."""

    def test_slow_transport_conserves_winding(self):
        n = 90
        out = transport_defect((n, n), (n / 2 - 12, n / 2), (n / 2 + 12, n / 2),
                               1, width=5.0, depth=0.9, speed=0.03,
                               relax=400, radius=6)
        self.assertTrue(out["conserved"])
        self.assertAlmostEqual(abs(out["w_end"]), 1.0, delta=0.4)

    def test_fast_transport_loses_winding(self):
        n = 90
        out = transport_defect((n, n), (n / 2 - 12, n / 2), (n / 2 + 12, n / 2),
                               1, width=5.0, depth=0.9, speed=0.8,
                               relax=400, radius=6)
        self.assertFalse(out["conserved"])


class TestDynamicalBraid(unittest.TestCase):

    def test_braid_runs_and_reports(self):
        # a short braid: both cores present at the seed, keys well-formed
        out = dynamical_braid((100, 100), (50, 50), 16.0, (1, 1), turns=0.5,
                              width=5.0, depth=0.9, delta=0.5, sub=40,
                              relax=300)
        for key in ("k", "sign", "survival", "survived_fraction", "steps"):
            self.assertIn(key, out)
        self.assertEqual(out["survival"][0], 2)   # both ½ cores seeded
        self.assertGreaterEqual(out["survived_fraction"], 0.0)
        self.assertLessEqual(out["survived_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
