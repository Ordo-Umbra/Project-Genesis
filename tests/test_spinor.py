"""Checks for the nematic ½-disclination: half-integer spin and its double cover."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import gaussian_load, relax_capacity  # noqa: E402
from project_genesis.nematic_spinor import (  # noqa: E402
    director_holonomy,
    disclination_strength,
)
from project_genesis.two_field import chiral_detuning, step_chiral_detuned  # noqa: E402
from project_genesis.vortex_chiral import imprint_vortices  # noqa: E402


class DisclinationStrengthTests(unittest.TestCase):
    def test_strength_is_half_the_charge(self):
        n = (96, 96)
        c = (48, 48)
        for q in (-2, -1, 0, 1, 2, 3):
            s = disclination_strength(imprint_vortices(n, [c], [q], core=3.0),
                                      c, radius=12)
            self.assertAlmostEqual(s, q / 2.0, delta=0.05)

    def test_elementary_defect_is_half_integer(self):
        n = (96, 96)
        c = (48, 48)
        s = disclination_strength(imprint_vortices(n, [c], [1], core=3.0), c, 12)
        self.assertAlmostEqual(abs(s), 0.5, delta=0.05)   # not an integer


class DoubleCoverTests(unittest.TestCase):
    def test_half_disclination_flips_at_2pi_restores_at_4pi(self):
        n = (96, 96)
        c = (48, 48)
        h2, h4 = director_holonomy(imprint_vortices(n, [c], [1], core=3.0), c, 12)
        self.assertLess(h2, -0.9)         # 2π: the spinor sign flip
        self.assertGreater(h4, 0.9)       # 4π: restored

    def test_integer_disclination_does_not_flip(self):
        n = (96, 96)
        c = (48, 48)
        h2, _ = director_holonomy(imprint_vortices(n, [c], [2], core=3.0), c, 12)
        self.assertGreater(h2, 0.9)       # integer defect: no flip


class FusionAndMatterTests(unittest.TestCase):
    def test_two_halves_fuse_to_an_integer(self):
        n = (96, 96)
        c0, c1 = (40, 48), (56, 48)
        psi = imprint_vortices(n, [c0, c1], [1, 1], core=3.0)
        s = disclination_strength(psi, (48, 48), radius=24)
        self.assertAlmostEqual(s, 1.0, delta=0.1)         # ½ + ½ = 1

    def test_half_disclination_pinned_and_conserved(self):
        n = (64, 64)
        c = (32, 32)
        kappa = relax_capacity(gaussian_load(n, [c], 2.5, 0.6),
                               kappa_diffusion=1.0, kappa_recovery=0.02,
                               kappa_consumption=0.8)
        g = chiral_detuning(kappa, detune_gamma=0.8)
        psi = imprint_vortices(n, [c], [1], core=3.0, detune=g)
        for _ in range(400):
            psi = step_chiral_detuned(psi, g, chiral=0.0, dt=0.1)
        self.assertAlmostEqual(disclination_strength(psi, c, 10), 0.5, delta=0.05)
        h2, h4 = director_holonomy(psi, c, 10)
        self.assertLess(h2, -0.9)         # the −1 holonomy survives evolution
        self.assertGreater(h4, 0.9)

    def test_composite_spin_is_additive_and_statistics_alternate(self):
        # n half-constituents: far-field s = n/2; odd n flips (-1), even doesn't
        n = (96, 96)
        mid = 48.0
        for n_c, want_flip in ((2, False), (3, True)):
            r = 9.0 / (2.0 * np.sin(np.pi / n_c))
            cs = [(mid + r * np.cos(2 * np.pi * k / n_c),
                   mid + r * np.sin(2 * np.pi * k / n_c)) for k in range(n_c)]
            psi = imprint_vortices(n, cs, [1] * n_c, core=3.0)
            self.assertAlmostEqual(
                disclination_strength(psi, (mid, mid), radius=28),
                n_c / 2.0, delta=0.1)
            h2, _ = director_holonomy(psi, (mid, mid), radius=28)
            if want_flip:
                self.assertLess(h2, -0.9)      # baryon-like: fermionic composite
            else:
                self.assertGreater(h2, 0.9)    # meson-like: bosonic composite

    def test_half_antihalf_annihilate(self):
        n = (64, 64)
        c0, c1 = (32, 27), (32, 37)
        kappa = relax_capacity(
            gaussian_load(n, [c0], 2.5, 0.6) + gaussian_load(n, [c1], 2.5, 0.6),
            kappa_diffusion=1.0, kappa_recovery=0.02, kappa_consumption=0.8)
        g = chiral_detuning(kappa, detune_gamma=0.8)
        psi = imprint_vortices(n, [c0, c1], [1, -1], core=3.0, detune=g)
        seed_min = np.abs(psi).min()
        for _ in range(600):
            psi = step_chiral_detuned(psi, g, chiral=0.0, dt=0.1)
        self.assertLess(abs(disclination_strength(psi, c0, 6)), 0.15)
        self.assertLess(abs(disclination_strength(psi, c1, 6)), 0.15)
        self.assertGreater(np.abs(psi).min(), seed_min + 0.1)   # field heals


if __name__ == "__main__":
    unittest.main()
