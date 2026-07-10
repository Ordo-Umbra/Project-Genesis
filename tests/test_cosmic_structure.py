"""Checks for κ-gravity against an expanding background (turnaround, FoF)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_dynamics import (  # noqa: E402
    evolve_inertial,
    fof_groups,
    hubble_flow,
)

FK = dict(kappa_diffusion=1.0, kappa_recovery=0.02, kappa_consumption=0.8,
          kappa_baseline=1.0, max_iters=1500)


class TestHubbleAndFoF(unittest.TestCase):

    def test_hubble_flow_is_outward_and_linear(self):
        ctr = np.array([0.0, 0.0])
        v = hubble_flow([[1.0, 0.0], [2.0, 0.0]], 0.1, center=ctr)
        # velocity points outward and scales with distance
        self.assertAlmostEqual(v[0, 0], 0.1, places=9)
        self.assertAlmostEqual(v[1, 0], 0.2, places=9)

    def test_hubble_flow_default_center_is_com(self):
        v = hubble_flow([[0.0, 0.0], [10.0, 0.0]], 0.1)  # com at (5,0)
        self.assertAlmostEqual(v[0, 0], -0.5, places=9)  # inward-left of com
        self.assertAlmostEqual(v[1, 0], +0.5, places=9)

    def test_fof_one_group_when_close(self):
        sizes = fof_groups([[0, 0], [1, 0], [2, 0]], link=1.5)
        self.assertEqual(sizes, [3])

    def test_fof_singletons_when_far(self):
        sizes = fof_groups([[0, 0], [20, 0], [40, 0]], link=1.5)
        self.assertEqual(sizes, [1, 1, 1])

    def test_fof_chains_transitively(self):
        # A–B close, B–C close, A–C far → still one group (friends of friends)
        sizes = fof_groups([[0, 0], [2, 0], [4, 0]], link=2.5)
        self.assertEqual(sizes, [3])


class TestExpansionVsGravity(unittest.TestCase):

    def _pair(self, H, steps):
        L, d = 64, 10.0
        pos = [[L / 2 - d / 2, L / 2], [L / 2 + d / 2, L / 2]]
        vH = H * d / 2.0
        vel = [[-vH, 0.0], [vH, 0.0]]
        return evolve_inertial(pos, vel, [1.5, 1.5], shape=(L, L), width=2.5,
                               dt=0.5, steps=steps, escape_radius=30.0, **FK)

    def test_slow_expansion_recollapses(self):
        # a slowly receding pair turns around and comes back (bound)
        h = self._pair(0.12, 60)
        sep = np.array(h["separation"])
        imax = int(np.argmax(sep))
        self.assertGreater(imax, 0)                 # it rose (turnaround)
        self.assertLess(sep[-1], sep[imax])         # then recollapsed

    def test_fast_expansion_escapes(self):
        # a fast kick unbinds the pair — it runs to the escape radius
        h = self._pair(0.30, 60)
        self.assertGreater(max(h["separation"]), 28.0)


if __name__ == "__main__":
    unittest.main()
