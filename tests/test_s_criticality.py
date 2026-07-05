"""Fast checks for the S-criticality and latent-heat experiment helpers."""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_latent_heat import bimodality, matter_energy_per_site  # noqa: E402
from n3_s_criticality import s_err_of, s_of  # noqa: E402

from project_genesis.gauge import identity_links, random_psi  # noqa: E402
from project_genesis.gauge import covariant_coherence  # noqa: E402


class TestSAssembly(unittest.TestCase):

    def test_s_of_and_error(self):
        pt = {"delta_c": 0.03, "delta_c_err": 0.001,
              "kappa": 0.8, "coherence": 0.5, "coherence_err": 0.01}
        self.assertAlmostEqual(s_of(pt, 0.1), 0.03 + 0.8 * 0.1 * 0.5,
                               places=12)
        expected = math.hypot(0.001, 0.8 * 0.1 * 0.01)
        self.assertAlmostEqual(s_err_of(pt, 0.1), expected, places=12)


class TestMatterEnergy(unittest.TestCase):

    def test_energy_matches_direct_computation(self):
        rng = np.random.default_rng(0)
        psi = random_psi(rng, 3, (5, 5))
        links = identity_links(3, (5, 5))
        g_m, u = 4.0, 1.0
        e = matter_energy_per_site(psi, links, g_m, u)
        direct = -(g_m * covariant_coherence(psi, links)
                   + u * float(np.sum(np.abs(psi) ** 4))) / 25
        self.assertAlmostEqual(e, direct, places=12)

    def test_ordered_field_has_lower_energy(self):
        """A basis-aligned field must sit below a random one in energy."""
        rng = np.random.default_rng(1)
        links = identity_links(3, (8, 8))
        hot = random_psi(rng, 3, (8, 8))
        cold = np.zeros((8, 8, 3), dtype=complex)
        cold[..., 0] = 1.0
        e_hot = matter_energy_per_site(hot, links, 4.0, 1.0)
        e_cold = matter_energy_per_site(cold, links, 4.0, 1.0)
        self.assertLess(e_cold, e_hot - 1.0)


class TestBimodality(unittest.TestCase):

    def test_two_separated_peaks_detected(self):
        rng = np.random.default_rng(2)
        pooled = np.concatenate([rng.normal(-1.0, 0.05, 500),
                                 rng.normal(1.0, 0.05, 500)])
        b = bimodality(pooled)
        self.assertTrue(b["bimodal"])
        self.assertAlmostEqual(b["peak_separation"], 2.0, delta=0.3)

    def test_single_peak_not_bimodal(self):
        rng = np.random.default_rng(3)
        b = bimodality(rng.normal(0.0, 0.3, 1000))
        self.assertFalse(b["bimodal"])

    def test_overlapping_peaks_not_bimodal(self):
        rng = np.random.default_rng(4)
        pooled = np.concatenate([rng.normal(-0.1, 0.3, 500),
                                 rng.normal(0.1, 0.3, 500)])
        b = bimodality(pooled)
        self.assertFalse(b["bimodal"])


if __name__ == "__main__":
    unittest.main()
