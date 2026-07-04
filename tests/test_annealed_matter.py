"""Tests for the annealed (ψ, U) joint ensemble.

Covers:
- Python vs Numba matter sweeps: same seed → same field (draw-order parity)
- free-theory limit (all couplings off): amplitudes equilibrate to uniform
- corner potential + fraction pinning: sectors form, fractions stay pinned,
  and a junction network exists in the joint equilibrium
- temperature melts sector order (the order parameter falls with T)
- input validation
"""

from __future__ import annotations

import unittest

import numpy as np

from project_genesis.gauge import identity_links, random_psi, random_unitary
from project_genesis.annealed_matter import (
    joint_sweep,
    matter_metropolis_sweep,
    phase_fractions,
    sector_amplitudes,
    thermal_junction_analysis,
)
from project_genesis.multiphase import (
    count_triple_junctions,
    sector_labels,
    step_multiphase_conserved,
)


class TestSweepEquivalence(unittest.TestCase):

    def test_python_numba_same_seed_same_field(self):
        rng = np.random.default_rng(0)
        psi = random_psi(rng, 3, (8, 8))
        links = random_unitary(rng, 3, (2, 8, 8), special=True)
        kw = dict(temperature=0.2, matter_coupling=2.0, quartic_u=1.0,
                  frac_penalty=10.0, step_scale=0.3)
        p_py, a_py = matter_metropolis_sweep(
            psi, links, np.random.default_rng(5), use_numba=False, **kw)
        p_nb, a_nb = matter_metropolis_sweep(
            psi, links, np.random.default_rng(5), use_numba=True, **kw)
        self.assertLess(float(np.max(np.abs(p_py - p_nb))), 1e-12)
        self.assertEqual(a_py, a_nb)

    def test_rejects_nonpositive_temperature(self):
        rng = np.random.default_rng(1)
        psi = random_psi(rng, 2, (4, 4))
        links = identity_links(2, (4, 4))
        with self.assertRaises(ValueError):
            matter_metropolis_sweep(psi, links, rng, temperature=0.0)

    def test_psi_stays_normalised(self):
        rng = np.random.default_rng(2)
        psi = random_psi(rng, 3, (6, 6))
        links = random_unitary(rng, 3, (2, 6, 6), special=True)
        for _ in range(10):
            psi, _ = matter_metropolis_sweep(
                psi, links, rng, temperature=0.3, matter_coupling=1.0)
        norms = np.linalg.norm(psi, axis=-1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-12))


class TestPhysicalLimits(unittest.TestCase):

    def test_free_theory_gives_uniform_amplitudes(self):
        """With every coupling off the weight is flat: ⟨|ψ_a|²⟩ → 1/P."""
        rng = np.random.default_rng(3)
        psi = random_psi(rng, 3, (10, 10))
        links = identity_links(3, (10, 10))
        for _ in range(120):
            psi, acc = matter_metropolis_sweep(
                psi, links, rng, temperature=1.0, matter_coupling=0.0,
                quartic_u=0.0, frac_penalty=0.0, step_scale=0.8)
        self.assertEqual(acc, 1.0, "flat weight must accept every move")
        fracs = phase_fractions(psi)
        self.assertTrue(np.allclose(fracs, 1.0 / 3.0, atol=0.06),
                        f"free-theory fractions {fracs} far from uniform")

    def test_sectors_form_and_fractions_pin_at_low_t(self):
        rng = np.random.default_rng(4)
        psi = random_psi(rng, 3, (10, 10))
        links = identity_links(3, (10, 10))
        for _ in range(250):
            psi, links, _ = joint_sweep(
                psi, links, rng, temperature=0.05, beta_g=3.0,
                matter_coupling=2.0, quartic_u=1.0, frac_penalty=20.0,
                step_scale=0.12)
        amps = sector_amplitudes(psi)
        fracs = phase_fractions(psi)
        self.assertTrue(np.allclose(fracs, 1.0 / 3.0, atol=0.05),
                        f"pinned fractions drifted: {fracs}")
        self.assertGreater(float(np.max(amps, axis=0).mean()), 0.7,
                           "sectors should be sharp at low temperature")
        labels = sector_labels(amps)
        self.assertEqual(len(np.unique(labels)), 3)
        self.assertGreater(count_triple_junctions(labels), 0,
                           "a pinned 3-phase system should carry junctions")

    def test_temperature_melts_sector_order(self):
        """The order parameter ⟨max_a |ψ_a|²⟩ must fall as T rises."""
        rng = np.random.default_rng(5)
        links = identity_links(3, (10, 10))

        def order_at(t, seed):
            r = np.random.default_rng(seed)
            psi = random_psi(r, 3, (10, 10))
            for _ in range(200):
                psi, _ = matter_metropolis_sweep(
                    psi, links, r, temperature=t, matter_coupling=0.0,
                    quartic_u=1.0, frac_penalty=20.0,
                    step_scale=min(0.6, max(0.1, 0.5 * t ** 0.5)))
            return float(np.max(sector_amplitudes(psi), axis=0).mean())

        cold = order_at(0.05, 6)
        hot = order_at(2.0, 7)
        self.assertGreater(cold, hot + 0.1,
                           f"order should melt: cold={cold:.3f} hot={hot:.3f}")


class TestThermalJunctionMeasure(unittest.TestCase):
    """Calibration of the noise-robust junction density."""

    def _cold_amps(self):
        rng = np.random.default_rng(7)
        fields = rng.random((3, 24, 24)) * 0.1
        for _ in range(300):
            fields, _ = step_multiphase_conserved(
                fields, diffusion=1.0, gamma=1.5, dt=0.1)
        from project_genesis.gauge import sector_field_to_psi
        return sector_amplitudes(sector_field_to_psi(fields))

    def test_cold_network_scores_nonzero(self):
        tj = thermal_junction_analysis(self._cold_amps(), 3)
        self.assertGreater(tj["neutrality"], 0.003)
        self.assertGreater(tj["ordered_fraction"], 0.8)

    def test_pure_noise_scores_zero(self):
        rng = np.random.default_rng(0)
        from project_genesis.gauge import random_psi
        noise = sector_amplitudes(random_psi(rng, 3, (24, 24)))
        tj = thermal_junction_analysis(noise, 3)
        self.assertLess(tj["neutrality"], 1e-3,
                        "molten label noise must not count as junctions")
        self.assertLess(tj["ordered_fraction"], 0.2)

    def test_two_palette_field_has_no_junctions(self):
        rng = np.random.default_rng(1)
        fields = rng.random((2, 20, 20)) * 0.1
        for _ in range(200):
            fields, _ = step_multiphase_conserved(
                fields, diffusion=1.0, gamma=1.5, dt=0.1)
        from project_genesis.gauge import sector_field_to_psi
        amps = sector_amplitudes(sector_field_to_psi(fields))
        tj = thermal_junction_analysis(amps, 2)
        self.assertEqual(tj["neutrality"], 0.0)


if __name__ == "__main__":
    unittest.main()
