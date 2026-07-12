"""Checks for the matched-scale bridge instrument (instanton_scales)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.instanton_scales import (  # noqa: E402
    cp_flow_diffusion_constant,
    cp_flow_step,
    cp_gradient_flow,
    instanton_sizes_cp,
    instanton_sizes_su,
    local_maxima,
    matched_comparison,
    smoothing_radius,
)
from project_genesis.topological_charge import cp_action  # noqa: E402


def _periodic_r2(shape, center):
    """Squared periodic distance to ``center`` on a lattice of ``shape``."""
    grids = np.indices(shape).astype(float)
    r2 = np.zeros(shape)
    for g, c, L in zip(grids, center, shape):
        d = np.abs(g - c)
        r2 += np.minimum(d, L - d) ** 2
    return r2


def _cp_lump(shape, center, rho):
    """Exact CP instanton charge density ``q(x) = ρ²/(π(r²+ρ²)²)``."""
    r2 = _periodic_r2(shape, center)
    return rho ** 2 / (np.pi * (r2 + rho ** 2) ** 2)


def _bpst_lump(shape, center, rho):
    """Exact BPST charge density ``q(x) = 6ρ⁴/(π²(r²+ρ²)⁴)``."""
    r2 = _periodic_r2(shape, center)
    return 6.0 * rho ** 4 / (np.pi ** 2 * (r2 + rho ** 2) ** 4)


class TestLocalMaxima(unittest.TestCase):

    def test_finds_planted_peak(self):
        d = np.zeros((8, 8))
        d[3, 5] = 1.0
        peaks = local_maxima(d)
        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0][0], (3, 5))
        self.assertEqual(peaks[0][1], 1.0)

    def test_periodic_wraparound(self):
        # a lump centred on the lattice edge still yields exactly one peak
        q = _cp_lump((16, 16), (0, 15), rho=2.0)
        peaks = local_maxima(q)
        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0][0], (0, 15))

    def test_threshold_filters_minor_peaks(self):
        d = np.zeros((12, 12))
        d[2, 2] = 1.0
        d[8, 8] = 0.1                      # below the default 25% threshold
        self.assertEqual(len(local_maxima(d, min_frac=0.25)), 1)
        self.assertEqual(len(local_maxima(d, min_frac=0.05)), 2)

    def test_sorted_descending(self):
        d = np.zeros((12, 12))
        d[2, 2] = 0.5
        d[8, 8] = 1.0
        peaks = local_maxima(d, min_frac=0.1)
        self.assertEqual([p[0] for p in peaks], [(8, 8), (2, 2)])


class TestSizeEstimators(unittest.TestCase):

    def test_cp_size_recovered_exactly(self):
        # peak-height inversion is exact on a lattice-centred lump
        for rho in (2.0, 3.0, 5.0):
            q = _cp_lump((32, 32), (16, 16), rho)
            sizes = instanton_sizes_cp(q)
            self.assertEqual(sizes.size, 1)
            self.assertAlmostEqual(sizes[0], rho, places=10)

    def test_su_size_recovered_exactly(self):
        for rho in (2.0, 2.5):
            q = _bpst_lump((12, 12, 12, 12), (6, 6, 6, 6), rho)
            sizes = instanton_sizes_su(q)
            self.assertEqual(sizes.size, 1)
            self.assertAlmostEqual(sizes[0], rho, places=10)

    def test_two_lumps_two_sizes(self):
        q = _cp_lump((48, 48), (12, 12), 2.0) + _cp_lump((48, 48), (36, 36), 4.0)
        sizes = np.sort(instanton_sizes_cp(q, min_frac=0.05))
        self.assertEqual(sizes.size, 2)
        # overlap between the tails shifts peaks slightly; sizes near truth
        self.assertAlmostEqual(sizes[0], 2.0, delta=0.1)
        self.assertAlmostEqual(sizes[1], 4.0, delta=0.3)

    def test_sign_insensitive(self):
        # anti-instantons (q < 0) are sized identically
        q = -_cp_lump((32, 32), (16, 16), 3.0)
        sizes = instanton_sizes_cp(q)
        self.assertEqual(sizes.size, 1)
        self.assertAlmostEqual(sizes[0], 3.0, places=10)

    def test_empty_field_gives_no_sizes(self):
        self.assertEqual(instanton_sizes_cp(np.zeros((8, 8))).size, 0)


class TestCPGradientFlow(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(7)
        z = rng.standard_normal((16, 16, 3)) + 1j * rng.standard_normal((16, 16, 3))
        self.psi = z / np.linalg.norm(z, axis=-1, keepdims=True)

    def test_step_preserves_normalisation(self):
        z = cp_flow_step(self.psi, 0.1)
        np.testing.assert_allclose(np.linalg.norm(z, axis=-1), 1.0, atol=1e-12)

    def test_flow_descends_the_action(self):
        actions = [cp_action(self.psi)]
        z = self.psi
        for _ in range(5):
            z = cp_flow_step(z, 0.1)
            actions.append(cp_action(z))
        self.assertTrue(all(b < a for a, b in zip(actions, actions[1:])))

    def test_trajectory_records_observables(self):
        _, traj = cp_gradient_flow(self.psi, t_max=0.5, dt=0.1, measure_every=2)
        self.assertEqual(traj[0]["t"], 0.0)
        self.assertAlmostEqual(traj[-1]["t"], 0.5)
        for rec in traj:
            for key in ("t", "S", "Q", "kappa", "rho_mean", "n_lumps"):
                self.assertIn(key, rec)
        # the flow descends the action along the trajectory too
        self.assertLess(traj[-1]["S"], traj[0]["S"])

    def test_diffusion_constant_is_unity(self):
        # the linearised flow is the lattice heat equation with D = 1;
        # the calibration *measures* it and should land within a percent
        d = cp_flow_diffusion_constant(size=32, dt=0.1, n_steps=40)
        self.assertAlmostEqual(d, 1.0, delta=0.05)


class TestSmoothingRadius(unittest.TestCase):

    def test_luscher_convention_in_4d(self):
        # √(8t) for the Wilson flow
        self.assertAlmostEqual(smoothing_radius(2.0, dim=4), 4.0, places=12)

    def test_2d_with_measured_diffusion(self):
        self.assertAlmostEqual(smoothing_radius(1.0, dim=2, diffusion=2.0),
                               np.sqrt(8.0), places=12)

    def test_accepts_arrays(self):
        t = np.array([0.0, 1.0, 4.0])
        np.testing.assert_allclose(smoothing_radius(t, dim=2),
                                   [0.0, 2.0, 4.0])


class TestMatchedComparison(unittest.TestCase):

    def test_detects_crossing(self):
        s = np.linspace(0.0, 1.0, 21)
        out = matched_comparison(s, 0.1 + 0.4 * s, s, 0.5 - 0.4 * s)
        self.assertTrue(out["overlap"])
        self.assertTrue(out["crossing"])
        self.assertAlmostEqual(out["s_star"], 0.5, places=6)
        self.assertAlmostEqual(out["kappa_star"], 0.3, places=6)

    def test_reports_min_gap_without_crossing(self):
        s = np.linspace(0.0, 1.0, 21)
        out = matched_comparison(s, 0.5 + 0.1 * s, s, 0.1 * s)
        self.assertTrue(out["overlap"])
        self.assertFalse(out["crossing"])
        self.assertAlmostEqual(out["min_gap"], 0.5, places=6)

    def test_disjoint_ranges_reported_honestly(self):
        out = matched_comparison([0.0, 1.0], [0.1, 0.2], [2.0, 3.0], [0.3, 0.4])
        self.assertFalse(out["overlap"])
        self.assertIn("range_a", out)

    def test_nan_samples_dropped(self):
        s = np.array([0.0, np.nan, 0.5, 1.0])
        k = np.array([0.0, 1.0, 0.25, 0.5])
        out = matched_comparison(s, k, [0.0, 1.0], [0.5, 0.0])
        self.assertTrue(out["overlap"])
        self.assertTrue(out["crossing"])

    def test_unsorted_input_handled(self):
        out = matched_comparison([1.0, 0.0], [0.5, 0.1], [0.0, 1.0], [0.3, 0.3])
        self.assertTrue(out["overlap"])


if __name__ == "__main__":
    unittest.main()
