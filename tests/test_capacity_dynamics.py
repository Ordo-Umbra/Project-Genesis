"""Checks for self-gravitating capacity dynamics (infall + accretion)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_dynamics import (  # noqa: E402
    _merge_close,
    capacity_force,
    evolve,
)

FK = dict(kappa_diffusion=1.0, kappa_recovery=0.02, kappa_consumption=0.8,
          kappa_baseline=1.0, max_iters=3000)


class TestForce(unittest.TestCase):

    def test_force_is_attractive(self):
        # two masses along axis 1 pull toward each other (opposite x-forces)
        L = 64
        pos = [[L / 2, L / 2 - 6], [L / 2, L / 2 + 6]]
        F, _ = capacity_force(pos, [1.0, 1.0], shape=(L, L), width=2.5, **FK)
        self.assertGreater(F[0, 1], 0.0)      # left mass pushed right (+x)
        self.assertLess(F[1, 1], 0.0)         # right mass pushed left (−x)
        self.assertAlmostEqual(F[0, 1], -F[1, 1], places=6)  # equal & opposite

    def test_no_gravity_no_force(self):
        # capacity coupling off → no well → no force
        L = 64
        pos = [[L / 2, L / 2 - 6], [L / 2, L / 2 + 6]]
        off = dict(FK, kappa_consumption=0.0)
        F, _ = capacity_force(pos, [1.0, 1.0], shape=(L, L), width=2.5, **off)
        self.assertTrue(np.allclose(F, 0.0, atol=1e-9))


class TestMerge(unittest.TestCase):

    def test_merge_conserves_mass_and_fuses(self):
        pos, m = _merge_close([np.array([10.0, 10.0]), np.array([11.0, 10.0])],
                              [1.0, 3.0], merge_dist=3.0)
        self.assertEqual(len(m), 1)
        self.assertAlmostEqual(m[0], 4.0, places=9)
        # mass-weighted centre
        self.assertAlmostEqual(pos[0][0], (1 * 10 + 3 * 11) / 4, places=9)

    def test_merge_leaves_distant_masses(self):
        pos, m = _merge_close([np.array([0.0, 0.0]), np.array([20.0, 0.0])],
                              [1.0, 1.0], merge_dist=3.0)
        self.assertEqual(len(m), 2)


class TestEvolve(unittest.TestCase):

    def test_two_masses_fall_together(self):
        L = 64
        h = evolve([[L / 2, L / 2 - 7], [L / 2, L / 2 + 7]], [1.2, 1.2],
                   shape=(L, L), width=2.5, mobility=2.0, dt=1.0, steps=25,
                   merge_dist=3.0, **FK)
        self.assertLess(h["separation"][-1], h["separation"][0])
        self.assertEqual(len(h["final_masses"]), 1)      # merged

    def test_control_stays_put(self):
        L = 64
        off = dict(FK, kappa_consumption=0.0)
        h = evolve([[L / 2, L / 2 - 7], [L / 2, L / 2 + 7]], [1.2, 1.2],
                   shape=(L, L), width=2.5, mobility=2.0, dt=1.0, steps=12,
                   merge_dist=3.0, **off)
        self.assertAlmostEqual(h["separation"][0], h["separation"][-1], places=6)

    def test_nbody_accretes_conserving_mass(self):
        L = 80
        rng = np.random.default_rng(1)
        pos = [rng.uniform(24, L - 24, 2) for _ in range(6)]
        h = evolve(pos, [1.0] * 6, shape=(L, L), width=2.5, mobility=3.0,
                   dt=1.0, steps=50, merge_dist=4.0, **FK)
        self.assertLess(len(h["final_masses"]), 6)                 # clumped
        self.assertAlmostEqual(sum(h["final_masses"]), 6.0, places=6)  # conserved


if __name__ == "__main__":
    unittest.main()
