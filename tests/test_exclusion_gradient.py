"""Checks for the gradient-corrected exclusion term (Part II).

The exclusion gap of the FULL capacity free energy —
``G(A) = 2E(ρ₁) − E(2ρ₁)`` with ``E(ρ) = F[κ̄[ρ]]`` the relaxed minimum
(gradient energy included) — replaces the Part-I homogeneous form.  In
linear response ``G(ρ₁) = c²κ₀²(ρ₁, G_r ρ₁) > 0`` with the screened
kernel ``(r − D∇²)G_r = δ``; exactly, the gap is non-negative at every
amplitude (E is concave in amplitude as a minimum over affine
functions).  The binary instrument prices only the duplicated fraction
``ρ_dup = min(ρ₁, ρ₂)``: ``E_x = 2E(ρ_dup) − E(2ρ_dup)``, whose exact
position-gradient drives ``contact_full`` in
``evolve_inertial_retarded``.  These tests guard the kernel, the gap's
sign and limits, the force's sign and exactness, the statics floor,
and the Lyapunov bookkeeping.
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
N = 48                   # tests run at 48²; the experiment's record is 96²


def e_of(load):
    """E(ρ) = F[κ̄[ρ]] — the relaxed capacity free energy at fixed load."""
    kappa = relax_capacity(load, **FKW)
    return capacity_free_energy(kappa, load, **FKW)


def gap_of_amplitude(a, shape=(N, N), width=2.5):
    """G(A) = 2E(ρ₁) − E(2ρ₁) for one Gaussian blob of amplitude A."""
    rho = gaussian_load(shape, [(shape[0] / 2, shape[1] / 2)], width, a)
    return 2.0 * e_of(rho) - e_of(2.0 * rho)


class LinearResponseGapTests(unittest.TestCase):
    def test_gap_positive_at_small_amplitude(self):
        # pre-registered G1 check: G(A) > 0 for A = 0.01
        self.assertGreater(gap_of_amplitude(0.01), 0.0)

    def test_gap_positive_at_operating_amplitude(self):
        # G1's sign at the operating point — and the concavity theorem's
        # guarantee that the total gap can never flip sign
        self.assertGreater(gap_of_amplitude(0.6), 0.0)

    def test_gap_matches_kernel_benchmark_in_linear_response(self):
        # G(A)/G_LR(A) → 1 as A → 0 (M2): within 5% at A = 0.002
        a = 0.002
        rho = gaussian_load((N, N), [(N / 2, N / 2)], 2.5, a)
        g_lr = linear_response_exclusion_gap(rho, **FKW)
        ratio = gap_of_amplitude(a) / g_lr
        self.assertGreater(ratio, 0.9)
        self.assertLess(ratio, 1.05)


class ScreenedKernelTests(unittest.TestCase):
    def test_kernel_zero_mode_is_one_over_r(self):
        # Ĝ(0) = 1/r: a uniform load maps to a uniform 1/r
        ones = np.ones((N, N))
        out = apply_screened_kernel(ones, kappa_diffusion=1.0,
                                    kappa_recovery=R)
        self.assertTrue(np.allclose(out, 1.0 / R, rtol=1e-12))

    def test_kernel_inverts_the_screened_operator(self):
        # (r − D∇²)(G_r ρ) = ρ, with the repo's own 5-point Laplacian
        rho = gaussian_load((N, N), [(N / 2, N / 2)], 2.5, 0.6)
        gr = apply_screened_kernel(rho, kappa_diffusion=1.0,
                                   kappa_recovery=R)
        back = R * gr - 1.0 * periodic_laplacian(gr)
        self.assertLess(np.max(np.abs(back - rho)), 1e-10)

    def test_green_function_decays_over_screening_length(self):
        # G_r(x) at one screening length is well below its core value
        g = screened_green_function((N, N), kappa_diffusion=1.0,
                                    kappa_recovery=R)
        xi = np.sqrt(1.0 / R)
        core = g[0, 0]
        at_xi = g[int(round(xi)), 0]
        self.assertGreater(core, 0.0)
        self.assertLess(at_xi, 0.5 * core)

    def test_cg_solver_matches_relax_capacity(self):
        # _solve_relaxed (the contact_full force-path solver) converges to
        # the same fixed point as relax_capacity, to the relaxer's own
        # tolerance (~5e-6 at tol = 1e-8)
        load = gaussian_load((N, N), [(N / 2, N / 2)], 2.5, 0.6)
        k_ref = relax_capacity(load, **FKW)
        k_cg = _solve_relaxed(load, **FKW)
        self.assertLess(np.max(np.abs(k_cg - k_ref)), 1e-4)

    def test_analytic_load_gradient_matches_gaussian_load(self):
        # _gaussian_load_and_grad builds the same load as gaussian_load
        # (and its gradient integrates back to the load)
        l_ref = gaussian_load((N, N), [(20.3, 24.7)], 2.5, 0.6)
        l_new, _ = _gaussian_load_and_grad((N, N), (20.3, 24.7), 2.5, 0.6)
        self.assertLess(np.max(np.abs(l_new - l_ref)), 1e-12)


class DuplicatedFractionGapTests(unittest.TestCase):
    def test_ex_vanishes_at_infinite_separation(self):
        # E_x(∞) = 0: no common component, no exclusion energy
        s = 20.0
        l1 = gaussian_load((N, N), [(N / 2 - s / 2, N / 2)], 2.5, 0.6)
        l2 = gaussian_load((N, N), [(N / 2 + s / 2, N / 2)], 2.5, 0.6)
        ex = exclusion_gap_full(duplicated_load(l1, l2), **FKW)
        self.assertLess(abs(ex), 1e-6)

    def test_ex_at_zero_separation_is_the_full_gap(self):
        # E_x(0) = G(A): at coincidence the whole blob is duplicated
        l1 = gaussian_load((N, N), [(N / 2, N / 2)], 2.5, 0.6)
        ex = exclusion_gap_full(duplicated_load(l1, l1), **FKW)
        self.assertAlmostEqual(ex, gap_of_amplitude(0.6), places=9)

    def test_ex_is_nonnegative_and_ordered(self):
        # the exclusion energy is positive and fades with separation
        def ex_at(s):
            l1 = gaussian_load((N, N), [(N / 2 - s / 2, N / 2)], 2.5, 0.6)
            l2 = gaussian_load((N, N), [(N / 2 + s / 2, N / 2)], 2.5, 0.6)
            return exclusion_gap_full(duplicated_load(l1, l2), **FKW)
        e2, e8 = ex_at(2.0), ex_at(8.0)
        self.assertGreater(e2, e8)
        self.assertGreater(e8, 0.0)


class ContactFullForceTests(unittest.TestCase):
    def test_contact_paths_are_mutually_exclusive(self):
        base = dict(shape=(24, 24), steps=2, kappa_recovery=R,
                    kappa_consumption=C)
        with self.assertRaises(ValueError):
            evolve_inertial_retarded(
                [[10.0, 10.0], [14.0, 10.0]], [[0.0, 0.0], [0.0, 0.0]],
                [1.0, 1.0], contact_b=0.2, contact_full=True, **base)
        with self.assertRaises(ValueError):
            evolve_inertial_retarded(
                [[10.0, 10.0], [14.0, 10.0]], [[0.0, 0.0], [0.0, 0.0]],
                [1.0, 1.0], contact_derived=True, contact_full=True, **base)

    def test_contact_full_requires_at_least_two_masses(self):
        # Part IV generalizes contact_full to n >= 2 same-label masses
        # (the n-copy sector — see tests/test_exclusion_ncopy.py); the
        # guard that survives is that duplicated overlap needs a pair.
        with self.assertRaises(ValueError):
            evolve_inertial_retarded(
                [[10.0, 12.0]],
                [[0.0, 0.0]], [1.0], shape=(24, 24),
                steps=2, contact_full=True,
                kappa_recovery=R, kappa_consumption=C)

    def test_full_contact_repels_overlapping_masses(self):
        # Dilute pair (width 2.5, separation 3, amplitude 0.01 — the
        # linear-response regime where the gap is the kernel's positive
        # self-interaction): the contact_full contribution (full run
        # minus no-exclusion control) must push the blobs APART.
        mass = 0.01
        common = dict(shape=(48, 48), width=2.5, tau=1.0, dt=0.1,
                      steps=2, record_every=1, kappa_recovery=R,
                      kappa_consumption=C)
        pos = [[22.5, 24.0], [25.5, 24.0]]
        vel = [[0.0, 0.0], [0.0, 0.0]]
        hf = evolve_inertial_retarded(pos, vel, [mass, mass],
                                      contact_full=True, **common)
        hc = evolve_inertial_retarded(pos, vel, [mass, mass], **common)
        dv = (np.asarray(hf["velocities"])[1]
              - np.asarray(hc["velocities"])[1])
        self.assertLess(dv[0, 0], 0.0)   # left blob pushed further left
        self.assertGreater(dv[1, 0], 0.0)  # right blob pushed right

    def test_force_matches_static_gap_gradient(self):
        # The integrated contact_full force equals the exact position-
        # gradient of the static E_x = 2E(ρ_dup) − E(2ρ_dup): the
        # exclusion-only velocity kick over the first step matches
        # −∂E_x/∂R₁ (finite difference of exclusion_gap_full) to 5%.
        s, mass = 6.0, 0.6
        common = dict(shape=(N, N), width=2.5, tau=0.1, dt=0.1,
                      steps=2, record_every=1, kappa_recovery=R,
                      kappa_consumption=C)
        pos = [[N / 2 - s / 2, N / 2], [N / 2 + s / 2, N / 2]]
        vel = [[0.0, 0.0], [0.0, 0.0]]
        hf = evolve_inertial_retarded(pos, vel, [mass, mass],
                                      contact_full=True, **common)
        hc = evolve_inertial_retarded(pos, vel, [mass, mass], **common)
        dv1x = (np.asarray(hf["velocities"])[1, 0, 0]
                - np.asarray(hc["velocities"])[1, 0, 0])
        f1x = mass * dv1x / common["dt"]  # exclusion force on blob 1, x

        def ex_at(sep):
            l1 = gaussian_load((N, N), [(N / 2 - sep / 2, N / 2)], 2.5,
                               mass)
            l2 = gaussian_load((N, N), [(N / 2 + sep / 2, N / 2)], 2.5,
                               mass)
            return exclusion_gap_full(duplicated_load(l1, l2), **FKW)
        h = 0.02
        dex_ds = (ex_at(s + h) - ex_at(s - h)) / (2.0 * h)
        # F_1x = dE_x/ds for blob 1 on the left (repulsive: both < 0)
        self.assertLess(f1x, 0.0)
        self.assertLess(dex_ds, 0.0)
        self.assertLess(abs(f1x / dex_ds - 1.0), 0.05)

    def test_full_contact_lyapunov_still_holds(self):
        # The force is the exact gradient of the recorded E_x (smoothed
        # min, envelope pair), so the Lyapunov law survives the
        # nonlocal form — same bar as the repo's contact checks.
        hist = evolve_inertial_retarded(
            [[18.0, 24.0], [30.0, 24.0]], [[0.0, 0.3], [0.0, -0.3]],
            [0.6, 0.6], shape=(48, 48), width=2.5, tau=0.1, dt=0.1,
            steps=300, record_every=10, contact_full=True,
            kappa_recovery=R, kappa_consumption=C)
        e_rec = np.asarray(hist["energy"])
        self.assertLess(e_rec[-1], e_rec[0])           # net dissipation
        self.assertTrue(np.all(np.diff(e_rec) <= 1e-6))  # Lyapunov
        self.assertTrue(all(d >= 0.0 for d in hist["dissipation"]))


class OperatingPointStaticsTests(unittest.TestCase):
    """The G2 regression: the gradient-corrected term buys the floor."""

    def test_statics_have_interior_minimum_and_barrier(self):
        seps = [1.0, 4.0, 6.0, 8.0, 10.0, 16.0]
        e_full, e_ctl = [], []
        for s in seps:
            l1 = gaussian_load((N, N), [(N / 2 - s / 2, N / 2)], 2.5, 0.6)
            l2 = gaussian_load((N, N), [(N / 2 + s / 2, N / 2)], 2.5, 0.6)
            tot = l1 + l2
            e_ctl.append(e_of(tot))
            e_full.append(e_ctl[-1]
                          + exclusion_gap_full(duplicated_load(l1, l2),
                                               **FKW))
        e_full = np.array(e_full) - e_full[-1]
        e_ctl = np.array(e_ctl) - e_ctl[-1]
        i_min = int(np.argmin(e_full))
        s_star = seps[i_min]
        barrier = float(e_full[0] - e_full[i_min])
        self.assertTrue(2.0 < s_star < 10.0)
        self.assertGreaterEqual(barrier, 0.2)
        self.assertTrue(np.all(np.diff(e_ctl) >= -1e-9))  # control sinks


if __name__ == "__main__":
    unittest.main()
