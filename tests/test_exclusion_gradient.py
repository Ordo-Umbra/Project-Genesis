"""Checks for the gradient-corrected exclusion term (Part II):
``E_x = 2E(ρ_dup) − E(2ρ_dup)`` with ``E(ρ)`` the relaxed minimum of the
full capacity free energy at fixed load — gradient energy included —
and ``ρ_dup = min(ρ₁, ρ₂)`` the duplicated (cloned) fraction of the
pair.  The binary instrument prices only the duplicated fraction,
vanishes for separated structures, and keeps the exact-gradient force
and the Lyapunov bookkeeping.  (Part IV generalizes the path to n
same-label masses — the n-copy sector; see
``tests/test_exclusion_ncopy.py``.)
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import gaussian_load  # noqa: E402
from project_genesis.capacity_waves import (  # noqa: E402
    _relax_functional,
    _solve_relaxed,
    duplicated_load,
    evolve_inertial_retarded,
    exclusion_gap_full,
    linear_response_exclusion_gap,
    screened_green_function,
)

R, C = 0.02, 0.8          # the exclusion-core operating point
FKW = dict(kappa_diffusion=1.0, kappa_recovery=R, kappa_consumption=C,
           kappa_baseline=1.0)


def pair(s, mass=0.6):
    """The mirrored equal binary at separation s (centres on lattice sites)."""
    l1 = gaussian_load((48, 48), [(24 - s / 2, 24)], 2.5, mass)
    l2 = gaussian_load((48, 48), [(24 + s / 2, 24)], 2.5, mass)
    return l1, l2


class ScreenedKernelTests(unittest.TestCase):
    def test_kernel_inverts_the_operator(self):
        # (r − D∇²)G = δ: applying the operator to the Green's function
        # returns the lattice delta (peak 1, sum 1)
        g = screened_green_function((48, 48), kappa_diffusion=1.0,
                                    kappa_recovery=R)
        self.assertAlmostEqual(float(g.sum()), 1.0 / R, places=6)
        self.assertGreater(g[0, 0], 0.0)
        # screening length ξ = √(D/r) ≈ 7.07: the kernel decays over cells
        self.assertLess(float(g[0, 30]), float(g[0, 5]))

    def test_linear_response_gap_is_positive_and_scales_quadratically(self):
        blob = gaussian_load((48, 48), [(24, 24)], 2.5, 0.01)
        g1 = linear_response_exclusion_gap(blob, **FKW)
        g2 = linear_response_exclusion_gap(2.0 * blob, **FKW)
        self.assertGreater(g1, 0.0)
        self.assertAlmostEqual(g2 / g1, 4.0, places=6)


class SolveRelaxedTests(unittest.TestCase):
    def test_matches_the_relaxer(self):
        # the conjugate-gradient fixed point IS the relaxer's steady state
        from project_genesis.capacity_gravity import relax_capacity
        load = sum(pair(6.0))
        k_ref = relax_capacity(load, **FKW)
        k_cg = _solve_relaxed(load, None, **FKW)
        self.assertLess(float(np.max(np.abs(k_cg - k_ref))), 1e-6)
        # warm-started from the answer it is one iteration
        k_warm = _solve_relaxed(load, k_ref, **fkw := FKW)
        self.assertLess(float(np.max(np.abs(k_warm - k_ref))), 1e-8)

    def test_relax_functional_minimum(self):
        # the solved field minimises _relax_functional: perturbing it
        # raises the functional (it is the functional the relaxer
        # actually descends)
        load = sum(pair(6.0))
        k = _solve_relaxed(load, None, **FKW)
        f0 = _relax_functional(k, load, **FKW)
        rng = np.random.default_rng(0)
        for eps in (1e-4, 1e-3):
            f1 = _relax_functional(k + eps * rng.random(k.shape), load,
                                   **FKW)
            self.assertGreater(f1, f0)


class FullGapTests(unittest.TestCase):
    def test_gap_is_nonnegative_and_zero_for_separated(self):
        l1, l2 = pair(6.0)
        self.assertGreater(exclusion_gap_full(duplicated_load(l1, l2),
                                              **FKW), 0.0)
        l1, l2 = pair(20.0)   # far apart: no duplicated fraction
        self.assertAlmostEqual(
            exclusion_gap_full(duplicated_load(l1, l2), **FKW), 0.0,
            places=8)

    def test_gap_exceeds_linear_response_at_operating_amplitude(self):
        # at mass 0.6 the homogeneous Part-I term underestimates: the
        # gradient energy of the overlapped core is priced in (the
        # derivation doc, Part II, M1)
        l1, l2 = pair(6.0)
        dup = duplicated_load(l1, l2)
        full = exclusion_gap_full(dup, **FKW)
        lr = linear_response_exclusion_gap(dup, **FKW)
        self.assertGreater(full, lr)

    def test_six_fold_split_gap(self):
        # split blobs at s = 6: the duplicated fraction is half of each
        # blob — the gap of the split configuration is the gap of the
        # shared component, not of the total load
        l1, l2 = pair(6.0)
        dup = duplicated_load(l1, l2)
        self.assertAlmostEqual(float(dup.sum()),
                               float(0.5 * (l1.sum() + l2.sum())),
                               places=1)


class ContactFullForceTests(unittest.TestCase):
    def _run(self, pos, vel, steps, **kw):
        base = dict(shape=(48, 48), width=2.5, tau=0.1, dt=0.1,
                    record_every=1, kappa_recovery=R, kappa_consumption=C)
        base.update(kw)
        return evolve_inertial_retarded(pos, vel, [0.6, 0.6], steps=steps,
                                        **base)

    def test_force_is_the_energy_gradient(self):
        # exclusion force vs finite-difference of the booked E_x with
        # separation: 1 part in 1e4 (exact-gradient claim)
        s = 6.0
        pos = [[24 - s / 2, 24], [24 + s / 2, 24]]
        common = dict(shape=(48, 48), width=2.5, tau=0.1, dt=0.1,
                      record_every=1, contact_full=True,
                      kappa_recovery=R, kappa_consumption=C)

        def booked_ex(sep):
            hist = evolve_inertial_retarded(
                [[24 - sep / 2, 24], [24 + sep / 2, 24]],
                [[0.0, 0.0], [0.0, 0.0]], [0.6, 0.6], steps=1, **common)
            return (hist["energy"][0] - hist["field_energy"][0]
                    - hist["kinetic"][0] - hist["field_kinetic"][0])

        d = 0.02
        de_ds = (booked_ex(s + d) - booked_ex(s - d)) / (2.0 * d)
        hist = self._run(pos, [[0.0, 0.0], [0.0, 0.0]], 3,
                         contact_full=True)
        ctrl = self._run(pos, [[0.0, 0.0], [0.0, 0.0]], 3)
        dv = (np.asarray(hist["velocities"])
              - np.asarray(ctrl["velocities"]))
        # each blob's exclusion force pulls it outward: blob 0 at
        # x = 21 feels −x, blob 1 at x = 27 feels +x; the pair force
        # F = m·(dv1 − dv0)/(2·steps·dt) opposes dE/ds
        f_pair = 0.6 * (dv[-1, 1, 0] - dv[-1, 0, 0]) / (2 * 0.1 * 3)
        self.assertLess(abs(f_pair / (-de_ds) - 1.0), 1e-4)

    def test_momentum_conservation_on_the_tie_plane(self):
        # the equal mirrored binary lands exactly on lattice sites: the
        # smoothed min splits the tie 50/50 and the pair's total
        # momentum stays zero to machine precision (the hard
        # min + indicator rule is kept OFF this path — it mis-splits
        # the tie and violates Newton's third law by up to ~35%)
        hist = self._run([[21.0, 24.0], [27.0, 24.0]],
                         [[0.0, 0.0], [0.0, 0.0]], 10, contact_full=True)
        vs = np.asarray(hist["velocities"])
        self.assertLess(np.max(np.abs(vs[:, 0, :] + vs[:, 1, :])), 1e-12)

    def test_lyapunov_with_contact_full(self):
        # the exclusion-booked energy is a Lyapunov function of the
        # retarded dynamics: non-increasing at the repo's 1e-6 bar,
        # net decrease, dissipation non-negative
        hist = self._run([[19.0, 24.0], [29.0, 24.0]],
                         [[0.0, 0.3], [0.0, -0.3]], 200,
                         contact_full=True)
        e_rec = np.asarray(hist["energy"])
        self.assertLess(e_rec[-1], e_rec[0])
        self.assertTrue(np.all(np.diff(e_rec) <= 1e-6))
        self.assertTrue(all(d >= 0.0 for d in hist["dissipation"]))

    def test_contact_full_generalizes_beyond_the_binary(self):
        # Part IV (the n-copy sector) generalizes the contact_full
        # path to n same-label masses: within a same-label group the
        # cloned component is the min over the group and the group
        # pays nE(m) − E(nm) — three masses no longer raise.  The
        # reduction to THIS binary instrument at n = 2 is pinned
        # bitwise in tests/test_exclusion_ncopy.py.  A single mass
        # still raises: it clones nothing.
        hist = evolve_inertial_retarded(
            [[10.0, 12.0], [14.0, 12.0], [12.0, 16.0]],
            [[0.0, 0.0]] * 3, [1.0, 1.0, 1.0], shape=(24, 24),
            steps=2, contact_full=True,
            kappa_recovery=R, kappa_consumption=C)
        self.assertTrue(np.all(np.isfinite(hist["energy"])))
        with self.assertRaises(ValueError):
            evolve_inertial_retarded(
                [[12.0, 12.0]], [[0.0, 0.0]], [1.0], shape=(24, 24),
                steps=2, contact_full=True,
                kappa_recovery=R, kappa_consumption=C)

    def test_contact_terms_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            self._run([[21.0, 24.0], [27.0, 24.0]],
                      [[0.0, 0.0], [0.0, 0.0]], 2,
                      contact_full=True, contact_b=0.1)


if __name__ == "__main__":
    unittest.main()
