"""Checks for the condensation instruments (the reaction force and the gas)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import gaussian_load, relax_capacity  # noqa: E402
from project_genesis.condensation import (  # noqa: E402
    amp_reaction_force,
    condensation_run,
    defect_positions,
    grow_defect_gas,
)
from project_genesis.nematic_spinor import (  # noqa: E402
    director_holonomy,
    disclination_strength,
)
from project_genesis.two_field import chiral_detuning, step_chiral_detuned  # noqa: E402
from project_genesis.vortex_chiral import imprint_vortices  # noqa: E402


def _grow(g, lam, seed, n, steps):
    rng = np.random.default_rng(seed)
    psi = 0.1 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    for _ in range(steps):
        psi = step_chiral_detuned(psi, g, chiral=lam, dt=0.1)
    return psi


class ReactionForceTests(unittest.TestCase):
    def test_raw_force_points_toward_a_core(self):
        # a mass at (32,32), a core at (40,32): the force should point +x
        n = (64, 64)
        psi = imprint_vortices(n, [(40, 32)], [1], core=3.0)
        load = gaussian_load(n, [(32, 32)], 2.5, 0.6)
        f = amp_reaction_force([load], psi, chi_amp=5.0)
        self.assertGreater(f[0][0], 0.0)                 # toward the hole
        self.assertLess(abs(f[0][1]), 0.2 * abs(f[0][0]))

    def test_normalized_force_ignores_an_empty_well(self):
        # an empty well (detuning envelope, no defect): the raw force pulls a
        # neighbouring mass toward it; the normalised force does not — the
        # selectivity that separates matter from fermion
        n = (64, 64)
        well = gaussian_load(n, [(40, 32)], 2.5, 1.0)
        kappa = relax_capacity(well, kappa_diffusion=1.0, kappa_recovery=0.02,
                               kappa_consumption=0.8)
        g = chiral_detuning(kappa, detune_gamma=0.8)
        psi = np.sqrt(np.clip(1.0 - g, 0.0, 1.0)).astype(complex)  # no core
        load = gaussian_load(n, [(32, 32)], 2.5, 0.6)
        raw = amp_reaction_force([load], psi, chi_amp=5.0)
        norm = amp_reaction_force([load], psi, chi_amp=5.0, detune=g,
                                  normalize=True)
        self.assertGreater(raw[0][0], 1e-4)              # raw: pulled to well
        self.assertLess(abs(norm[0][0]), 0.05 * abs(raw[0][0]))  # norm: blind

    def test_normalized_force_still_sees_a_core(self):
        # with a genuine core inside the well, the normalised force does pull
        n = (64, 64)
        well = gaussian_load(n, [(40, 32)], 2.5, 1.0)
        kappa = relax_capacity(well, kappa_diffusion=1.0, kappa_recovery=0.02,
                               kappa_consumption=0.8)
        g = chiral_detuning(kappa, detune_gamma=0.8)
        psi = imprint_vortices(n, [(40, 32)], [1], core=3.0, detune=g)
        load = gaussian_load(n, [(32, 32)], 2.5, 0.6)
        f = amp_reaction_force([load], psi, chi_amp=5.0, detune=g,
                               normalize=True)
        self.assertGreater(f[0][0], 1e-4)                # toward the core


class GasAndRunTests(unittest.TestCase):
    def test_gas_defects_come_in_balanced_pairs(self):
        psi = grow_defect_gas((64, 64), np.zeros((64, 64)), steps=600, seed=7)
        dfs = defect_positions(psi)
        self.assertGreaterEqual(len(dfs), 2)
        self.assertEqual(sum(q for _, q in dfs), 0)      # net zero on the torus

    def test_condensation_run_records_and_stays_finite(self):
        n = 48
        h = condensation_run(
            [[n / 2 - 4.5, n / 2], [n / 2 + 4.5, n / 2]],
            [[0.0, 0.2], [0.0, -0.2]], [0.6, 0.6], shape=(n, n),
            steps=200, record_every=20, pregrow_steps=200, seed=9,
            phase_chi=0.6, chi_amp=5.0, normalize=True)
        self.assertTrue(np.all(np.isfinite(np.asarray(h["separation"]))))
        self.assertEqual(len(h["time"]), len(h["mass_defect_dist"]))
        self.assertTrue(np.all(np.isfinite(np.asarray(h["final_positions"]))))


class SelfAssemblyTests(unittest.TestCase):
    """A molecule co-evolved with the gas: it binds, and ψ can be snapshotted."""

    def test_record_snapshots_returns_the_field_at_each_record(self):
        n = 48
        h = condensation_run(
            [[n / 2 - 5, n / 2], [n / 2 + 5, n / 2]],
            [[0.0, 0.0], [0.0, 0.0]], [0.6, 0.6], shape=(n, n),
            steps=120, record_every=40, pregrow_steps=100, seed=3,
            phase_chi=0.3, chi_amp=2.0, normalize=True, chiral_lambda=0.5,
            record_snapshots=True)
        snaps = h["psi_snapshots"]
        self.assertEqual(len(snaps), len(h["time"]))          # one per record
        self.assertEqual(snaps[0].shape, (n, n))
        self.assertTrue(np.iscomplexobj(snaps[0]))

    def test_masses_bind_near_the_derived_floor(self):
        # started beyond the floor, κ-gravity + the derived exclusion pull the
        # pair inward to a bound separation (neither escaped nor collapsed)
        n = 48
        sep0 = 12.0
        h = condensation_run(
            [[n / 2 - sep0 / 2, n / 2], [n / 2 + sep0 / 2, n / 2]],
            [[0.0, 0.0], [0.0, 0.0]], [0.6, 0.6], shape=(n, n),
            steps=500, record_every=50, pregrow_steps=60, seed=1,
            phase_chi=0.0, chi_amp=0.0, chiral_lambda=0.0)
        final = float(np.linalg.norm(h["final_positions"][0]
                                     - h["final_positions"][1]))
        self.assertLess(final, sep0 - 1.0)      # pulled inward from the start
        self.assertGreater(final, 3.0)          # bound, not collapsed


class TransportCurrentTests(unittest.TestCase):
    """The λ precession is the transport current that carries fermions to matter."""

    def _late_velocity(self, g, lam, seed, n=40, warm=600, meas=30):
        # the direct transport-current metric: a relaxational (λ=0) field settles
        # to rest, a λ>0 field keeps moving — mean ⟨|ψ_{t+1}−ψ_t|⟩ over a late window
        psi = _grow(g, lam, seed, n, warm)
        v = 0.0
        for _ in range(meas):
            nxt = step_chiral_detuned(psi, g, chiral=lam, dt=0.1)
            v += float(np.mean(np.abs(nxt - psi)))
            psi = nxt
        return v / meas

    def test_lambda_sustains_a_transport_current(self):
        # a relaxational (λ=0) detuned CGL relaxes to rest; turning on λ switches
        # on a persistent spiral current that grows with the drive
        g = np.zeros((40, 40))
        v0 = self._late_velocity(g, 0.0, 5)
        v_half = self._late_velocity(g, 0.5, 5)
        v1 = self._late_velocity(g, 1.0, 5)
        self.assertLess(v0, 0.01)              # λ=0 settles to rest
        self.assertGreater(v1, 0.02)           # λ=1 sustains a current
        self.assertGreater(v1, 5.0 * v0)       # the current is switched on by λ
        self.assertGreater(v_half, v0)         # and grows monotonically with λ
        self.assertGreater(v1, v_half)

    def test_a_trapped_core_reads_as_a_half_integer_fermion(self):
        # what condenses is a fermion: a grown core reads s=±½ with the −1
        # double-cover holonomy (the spinor signature), never imprinted
        psi = _grow(np.zeros((64, 64)), 1.0, 3, 64, 1500)
        dfs = defect_positions(psi)
        self.assertTrue(dfs)
        half = sum(abs(abs(disclination_strength(psi, p, radius=4)) - 0.5) <= 0.15
                   and director_holonomy(psi, p, radius=4)[0] < -0.5
                   for p, _ in dfs)
        self.assertGreaterEqual(half, max(1, (2 * len(dfs)) // 3))


if __name__ == "__main__":
    unittest.main()
