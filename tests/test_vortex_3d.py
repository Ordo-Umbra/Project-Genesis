"""Checks for the 3-D vortex line: spin as a quantised axial vector."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import gaussian_load, relax_capacity  # noqa: E402
from project_genesis.two_field import chiral_detuning, step_chiral_detuned  # noqa: E402
from project_genesis.vortex_chiral_3d import (  # noqa: E402
    evolve_seeded_line,
    line_angular_momentum,
    line_winding,
    vortex_line,
)


class AxialVectorTests(unittest.TestCase):
    def test_L_aligns_with_the_line_any_orientation(self):
        n = (40, 40, 40)
        c = (20, 20, 20)
        for nrm in [(0, 0, 1), (1, 0, 0), (1, 1, 1), (2, -1, 1)]:
            L = line_angular_momentum(vortex_line(n, c, nrm, 1), c, radius=16)
            nh = np.asarray(nrm, float)
            nh = nh / np.linalg.norm(nh)
            self.assertGreater(abs(L @ nh) / np.linalg.norm(L), 0.99)

    def test_L_is_sign_locked_and_quantised(self):
        n = (40, 40, 40)
        c = (20, 20, 20)

        def Lz(q):
            return line_angular_momentum(vortex_line(n, c, (0, 0, 1), q), c, 16)[2]
        self.assertGreater(Lz(1), 0.0)
        self.assertLess(Lz(-1), 0.0)
        self.assertAlmostEqual(Lz(0), 0.0, delta=1.0)
        self.assertAlmostEqual(Lz(1), -Lz(-1), delta=0.02 * abs(Lz(1)))
        self.assertGreater(abs(Lz(2)), abs(Lz(1)))          # a ladder in |q|

    def test_magnitude_is_direction_independent(self):
        n = (40, 40, 40)
        c = (20, 20, 20)
        mags = [np.linalg.norm(line_angular_momentum(
            vortex_line(n, c, nrm, 1), c, 16))
            for nrm in [(0, 0, 1), (1, 1, 1), (2, -1, 1)]]
        self.assertLess((max(mags) - min(mags)) / max(mags), 0.02)


class LineWindingTests(unittest.TestCase):
    def test_reads_the_integer_charge(self):
        n = (40, 40, 40)
        c = (20, 20, 20)
        self.assertAlmostEqual(abs(line_winding(vortex_line(n, c, (0, 0, 1), 1),
                                                c, 2)), 1.0, delta=0.05)
        self.assertAlmostEqual(abs(line_winding(vortex_line(n, c, (0, 0, 1), 2),
                                                c, 2)), 2.0, delta=0.05)
        self.assertAlmostEqual(line_winding(vortex_line(n, c, (0, 0, 1), 0),
                                            c, 2), 0.0, delta=0.05)


class SelfSustainedLineTests(unittest.TestCase):
    def _kappa(self, n, c):
        return relax_capacity(gaussian_load(n, [c], 2.5, 0.6),
                              kappa_diffusion=1.0, kappa_recovery=0.02,
                              kappa_consumption=0.8)

    def test_winding_and_axis_conserved_no_reimprint(self):
        n = (36, 36, 36)
        c = (18, 18, 18)
        h = evolve_seeded_line(n, c, (0, 0, 1), 1, self._kappa(n, c),
                               chiral=0.0, steps=400, record_every=200,
                               noise=0.15, seed=1)
        w = np.asarray(h["winding"])
        self.assertLess(float(np.max(np.abs(w - w[0]))), 0.1)   # conserved
        self.assertAlmostEqual(abs(round(w[-1])), 1)            # integer, |w|=1
        self.assertGreater(min(h["align"]), 0.95)              # axis stable
        self.assertLess(h["amp_min"][-1], 0.2)                 # core survives

    def test_line_antiline_annihilate(self):
        n = (36, 36, 36)
        c0, c1 = (18, 13, 18), (18, 23, 18)
        kappa = relax_capacity(
            gaussian_load(n, [c0], 2.5, 0.6) + gaussian_load(n, [c1], 2.5, 0.6),
            kappa_diffusion=1.0, kappa_recovery=0.02, kappa_consumption=0.8)
        g = chiral_detuning(kappa, detune_gamma=0.8)
        psi = (vortex_line(n, c0, (0, 0, 1), 1)
               * vortex_line(n, c1, (0, 0, 1), -1)
               * np.sqrt(np.clip(1.0 - g, 0.0, 1.0)))
        for _ in range(400):
            psi = step_chiral_detuned(psi, g, chiral=0.0, dt=0.1)
        self.assertLess(abs(line_winding(psi, c0, 2)), 0.2)     # unwound
        self.assertLess(abs(line_winding(psi, c1, 2)), 0.2)
        self.assertGreater(np.abs(psi).min(), 0.4)             # healed


if __name__ == "__main__":
    unittest.main()
