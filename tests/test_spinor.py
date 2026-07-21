"""Checks for the nematic ½-disclination: half-integer spin and its double cover."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import gaussian_load, relax_capacity  # noqa: E402
from project_genesis.chiral_field import evolve_chiral, phase_sector_labels  # noqa: E402
from project_genesis.nematic_spinor import (  # noqa: E402
    director_holonomy,
    disclination_strength,
    plaquette_winding,
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


class JunctionFermionTests(unittest.TestCase):
    """The emergent spin defect is a three-sector junction carrying ±½."""

    def test_plaquette_winding_finds_the_imprinted_pair(self):
        # a ± pair is the valid torus configuration (total winding must be 0);
        # a lone imprint would force compensating windings at the seam
        psi = imprint_vortices((64, 64), [(20, 32), (44, 32)], [1, -1], core=3.0)
        psi = psi * np.exp(0.1j)   # break exact-π edge ties of the clean imprint
        w = plaquette_winding(psi)
        self.assertEqual(int(np.sum(w)), 0)              # net zero on the torus
        # the two imprinted cores read at their centres with opposite signs
        # (the periodic imprint also leaves wrap-seam image windings far away —
        # a property of the periodic-distance phase, not of the detector)
        cores = list(zip(*np.nonzero(w)))
        near1 = [c for c in cores if np.hypot(c[0] - 20, c[1] - 32) < 2.0]
        near2 = [c for c in cores if np.hypot(c[0] - 44, c[1] - 32) < 2.0]
        self.assertEqual(len(near1), 1)
        self.assertEqual(len(near2), 1)
        self.assertEqual(int(w[near1[0]]), -int(w[near2[0]]))  # opposite pair

    def test_emergent_defects_are_three_sector_spin_halves(self):
        psi = evolve_chiral((64, 64), chiral=0.0, steps=800, dt=0.05, seed=5)
        w = plaquette_winding(psi)
        defects = [((x + 0.5, y + 0.5), int(w[x, y]))
                   for x, y in zip(*np.nonzero(w))]
        self.assertGreaterEqual(len(defects), 2)
        self.assertEqual(sum(q for _, q in defects), 0)   # ± pairs on the torus
        labels = phase_sector_labels(psi, 3)

        def sectors_near(c, r=2):
            vals = set()
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        vals.add(int(labels[int(c[0] + dx) % 64,
                                            int(c[1] + dy) % 64]))
            return vals

        def nn(c):
            return min((np.hypot(min(abs(c[0] - c2[0]), 64 - abs(c[0] - c2[0])),
                                 min(abs(c[1] - c2[1]), 64 - abs(c[1] - c2[1])))
                        for c2, _ in defects if c2 != c), default=99.0)

        for c, q in defects:
            self.assertEqual(len(sectors_near(c)), 3)     # a 3-sector junction
            if nn(c) >= 8.0:                              # resolvable by the loop
                s = disclination_strength(psi, c, radius=4)
                self.assertAlmostEqual(s, q / 2.0, delta=0.1)   # spin-½


if __name__ == "__main__":
    unittest.main()
