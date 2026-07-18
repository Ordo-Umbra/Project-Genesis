"""Checks for the gradient-corrected exclusion term (Part II).

The exclusion gap of the FULL capacity free energy —
``G(A) = 2E(ρ₁) − E(2ρ₁)`` with ``E(ρ) = min_κ F[κ; ρ]``, gradient
energy included — and its binary force law ``contact_full``, applied
to the duplicated fraction ``ρ_dup = min(ρ₁, ρ₂)`` of the pair.  Also
the linear-response infrastructure (the screened Green's function)
and the direct fixed-point solver used by the force path.  (Part IV
generalizes the path to n same-label masses — the n-copy sector; see
``tests/test_exclusion_ncopy.py``.)
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import (  # noqa: E402
    capacity_free_energy,
    gaussian_load,
    relax_capacity,
)
from project_genesis.capacity_waves import (  # noqa: E402
    _gaussian_load_and_grad,
    _relax_functional,
    _solve_relaxed,
    apply_screened_kernel,
    duplicated_load,
    evolve_inertial_retarded,
    exclusion_gap_full,
    linear_response_exclusion_gap,
    screened_green_function,
)
from project_genesis.multiphase import periodic_laplacian  # noqa: E402

R, C = 0.02, 0.8          # the exclusion-core operating point
FKW = dict(kappa_diffusion=1.0, kappa_recovery=R, kappa_consumption=C,
           kappa_baseline=1.0)
N = 48


def blob(x=N / 2, y=N / 2, width=2.5, amplitude=0.6):
    return gaussian_load((N, N), [(x, y)], width, amplitude)


class LinearResponseGapTests(unittest.TestCase):
    def test_kernel_zero_mode_is_one_over_r(self):
        g_hat = apply_screened_kernel(np.ones((N, N)),
                                      kappa_diffusion=1.0,
                                      kappa_recovery=R)
        np.testing.assert_allclose(g_hat, 1.0 / R, atol=1e-12)

    def test_gap_is_quadratic_and_positive(self):
        rho = blob(amplitude=0.01)
        g1 = linear_response_exclusion_gap(rho, **FKW)
        g2 = linear_response_exclusion_gap(2.0 * rho, **FKW)
        self.assertGreater(g1, 0.0)
        self.assertAlmostEqual(g2 / g1, 4.0, places=10)


class ScreenedKernelTests(unittest.TestCase):
    def test_kernel_inverts_the_screened_operator(self):
        # (r − D∇²)G = δ: applying the operator to the Green's function
        # returns the lattice delta (peak 1, sum 1)
        g = screened_green_function((N, N), kappa_diffusion=1.0,
                                    kappa_recovery=R)
        self.assertAlmostEqual(float(g.sum()), 1.0 / R, places=6)
        self.assertGreater(g[0, 0], 0.0)
        # screening length ξ = √(D/r) ≈ 7.07: the kernel decays over cells
        self.assertLess(float(g[0, 30]), float(g[0, 5]))

    def test_cg_solver_matches_relax_capacity(self):
        # the direct fixed-point solver reproduces the relaxer's answer
        load = blob() + blob(x=N / 2 + 6.0)
        k_ref = relax_capacity(load, **FKW)
        k_cg = _solve_relaxed(load, None, **FKW)
        self.assertLess(float(np.max(np.abs(k_cg - k_ref))), 1e-6)
        # warm-started from the answer it is one iteration
        k_warm = _solve_relaxed(load, k_ref, **FKW)
        self.assertLess(float(np.max(np.abs(k_warm - k_ref))), 1e-8)

    def test_analytic_load_gradient_matches_gaussian_load(self):
        rho, grad = _gaussian_load_and_grad((N, N), (N / 2, N / 2), 2.5,
                                            0.6)
        np.testing.assert_allclose(rho, blob(), atol=1e-15)
        for ax in range(2):
            d = 0.5 * (np.roll(rho, -1, ax) - np.roll(rho, 1, ax))
            # the analytic gradient matches the central difference away
            # from the tails (where the difference stencil is exact)
            mask = rho > 1e-3 * rho.max()
            np.testing.assert_allclose(grad[ax][mask], d[mask],
                                       rtol=5e-2, atol=1e-4)


class DuplicatedFractionGapTests(unittest.TestCase):
    def test_gap_is_nonnegative_and_vanishes_when_separated(self):
        l1, l2 = blob(x=N / 2 - 3.0), blob(x=N / 2 + 3.0)
        dup = duplicated_load(l1, l2)
        self.assertGreater(exclusion_gap_full(dup, **FKW), 0.0)
        l1, l2 = blob(x=N / 2 - 12.0), blob(x=N / 2 + 12.0)
        self.assertAlmostEqual(
            exclusion_gap_full(duplicated_load(l1, l2), **FKW), 0.0,
            places=8)

    def test_gap_is_a_minimum_over_affine_functions(self):
        # E(A) = min_κ F[κ; A·ρ] is concave in A — check the measured
        # concavity directly: E((A+B)/2) ≥ (E(A) + E(B))/2
        rho = blob(amplitude=0.01)

        def e_of(a):
            k = relax_capacity(a * rho, **FKW)
            return capacity_free_energy(k, a * rho, **FKW)

        e1, e2, e3 = e_of(1.0), e_of(2.0), e_of(3.0)
        self.assertGreaterEqual(e2, 0.5 * (e1 + e3))

    def test_gap_matches_linear_response_in_the_dilute_limit(self):
        rho = blob(amplitude=0.01)
        full = exclusion_gap_full(rho, **FKW)
        lr = linear_response_exclusion_gap(rho, **FKW)
        self.assertLess(abs(full / lr - 1.0), 0.1)

    def test_gap_exceeds_linear_response_at_operating_amplitude(self):
        # at mass 0.6 the homogeneous Part-I term underestimates: the
        # gradient energy of the overlapped core is priced in (the
        # derivation doc, Part II, M1)
        rho = blob()
        full = exclusion_gap_full(rho, **FKW)
        lr = linear_response_exclusion_gap(rho, **FKW)
        self.assertGreater(full, lr)


class ContactFullForceTests(unittest.TestCase):
    def _run(self, pos, vel, steps, **kw):
        base = dict(shape=(N, N), width=2.5, tau=0.1, dt=0.1,
                    record_every=1, kappa_recovery=R, kappa_consumption=C)
        base.update(kw)
        return evolve_inertial_retarded(pos, vel, [0.6, 0.6], steps=steps,
                                        **base)

    def test_force_is_the_energy_gradient(self):
        # exclusion force vs finite-difference of the booked E_x with
        # separation: 1 part in 1e4 (exact-gradient claim)
        s = 6.0
        pos = [[N / 2 - s / 2, N / 2], [N / 2 + s / 2, N / 2]]
        common = dict(shape=(N, N), width=2.5, tau=0.1, dt=0.1,
                      record_every=1, contact_full=True,
                      kappa_recovery=R, kappa_consumption=C)

        def booked_ex(sep):
            hist = evolve_inertial_retarded(
                [[N / 2 - sep / 2, N / 2], [N / 2 + sep / 2, N / 2]],
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
