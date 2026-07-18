"""Checks for the distinction-resolving (labelled) load (Part III).

The unlabelled instrument prices ALL overlap as duplication:
``ρ_dup = min(ρ₁, ρ₂)`` cannot tell same-distinction from
different-distinction stacking.  But the gap ``2E(ρ) − E(2ρ)`` exactly
cancels the concavity (sharing) discount for IDENTICAL overlap only —
different distinctions legitimately keep the discount (that discount IS
binding).  The labelled load makes exclusion identity-selective: each
mass carries a label vector over distinction types, the capacity field
responds to the TOTAL load as before (gravity is identity-blind), and
exclusion applies per SHARED type only,
``E_x = Σ_t [2E(ρ_dup^(t)) − E(2ρ_dup^(t))]`` with
``ρ_dup^(t) = min(w₁ₜρ₁, w₂ₜρ₂)``.  These tests guard the φ = 1
reproduction of the base branch, the φ = 0 control, the fractional-φ
ordering, the general label algebra, and the Lyapunov/momentum
bookkeeping of the labelled force.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import gaussian_load  # noqa: E402
from project_genesis.capacity_waves import (  # noqa: E402
    duplicated_load,
    evolve_inertial_retarded,
    exclusion_gap_full,
    exclusion_gap_labelled,
    shared_duplicated_components,
    shared_fraction_labels,
)

R, C = 0.02, 0.8          # the exclusion-core operating point
FKW = dict(kappa_diffusion=1.0, kappa_recovery=R, kappa_consumption=C,
           kappa_baseline=1.0)
N = 48                   # tests run at 48²; the experiment's record is 96²


def pair(s, mass=0.6):
    """The mirrored equal binary at separation s (centres on lattice sites)."""
    l1 = gaussian_load((N, N), [(N / 2 - s / 2, N / 2)], 2.5, mass)
    l2 = gaussian_load((N, N), [(N / 2 + s / 2, N / 2)], 2.5, mass)
    return l1, l2


def ex_at(s, share):
    """The labelled statics E_x of the equal binary at (s, φ)."""
    l1, l2 = pair(s)
    labels1, labels2 = shared_fraction_labels(share)
    return exclusion_gap_labelled(l1, l2, labels1, labels2, **FKW)


class LabelConstructionTests(unittest.TestCase):
    def test_share_is_a_probability(self):
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                shared_fraction_labels(bad)

    def test_labels_are_distributions(self):
        with self.assertRaises(ValueError):
            shared_duplicated_components(*pair(6.0), {"a": 1.2}, {"a": 1.0})
        with self.assertRaises(ValueError):
            shared_duplicated_components(*pair(6.0), {"a": 1.1, "b": -0.1},
                                         {"a": 1.0})

    def test_shared_types_are_the_types_both_blobs_carry(self):
        l1, l2 = pair(6.0)
        # fully private labels: nothing is shared, nothing is cloned
        self.assertEqual(shared_duplicated_components(
            l1, l2, {"a": 1.0}, {"b": 1.0}), [])
        # weight-zero entries do not make a type shared
        comps = shared_duplicated_components(
            l1, l2, {"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 0.0})
        self.assertEqual(len(comps), 1)
        np.testing.assert_array_equal(comps[0],
                                      np.minimum(0.5 * l1, 1.0 * l2))


class LabelledStaticsTests(unittest.TestCase):
    def test_share_one_reproduces_the_base_statics_exactly(self):
        # φ = 1 IS the unlabelled instrument: the labelled statics value
        # at s = 6 reproduces the base branch's min(ρ₁, ρ₂) gap bitwise.
        l1, l2 = pair(6.0)
        base = exclusion_gap_full(duplicated_load(l1, l2), **FKW)
        self.assertEqual(ex_at(6.0, 1.0), base)

    def test_share_zero_prices_nothing(self):
        # φ = 0 (orthogonal labels): zero exclusion energy even deep in
        # overlap — the different-distinction stack keeps the sharing
        # discount.  Exactly zero (no shared type is ever relaxed).
        self.assertEqual(ex_at(3.0, 0.0), 0.0)
        self.assertEqual(ex_at(0.0, 0.0), 0.0)

    def test_fractional_share_scales_between_the_endpoints(self):
        # at fixed geometry the shared component's amplitude is φ·ρ_dup,
        # and the gap grows with amplitude (G(A) increasing): E_x(φ)
        # sits strictly between the φ = 0 and φ = 1 endpoints.
        e0, e25, e5, e75, e1 = (ex_at(6.0, p)
                                for p in (0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertEqual(e0, 0.0)
        self.assertLess(e0, e25)
        self.assertLess(e25, e5)
        self.assertLess(e5, e75)
        self.assertLess(e75, e1)

    def test_general_labels_sum_over_shared_types(self):
        # the general label algebra: E_x is the sum of the per-type gaps
        l1, l2 = pair(6.0)
        labels1, labels2 = {"a": 0.6, "b": 0.4}, {"a": 0.5, "b": 0.5}
        total = exclusion_gap_labelled(l1, l2, labels1, labels2, **FKW)
        per_type = sum(
            exclusion_gap_full(comp, **FKW)
            for comp in (np.minimum(0.6 * l1, 0.5 * l2),
                         np.minimum(0.4 * l1, 0.5 * l2)))
        self.assertAlmostEqual(total, per_type, places=12)
        self.assertGreater(total, 0.0)


class LabelledDynamicsTests(unittest.TestCase):
    def test_labelled_options_require_contact_full(self):
        base = dict(shape=(24, 24), steps=2, kappa_recovery=R,
                    kappa_consumption=C)
        with self.assertRaises(ValueError):
            evolve_inertial_retarded(
                [[10.0, 12.0], [14.0, 12.0]], [[0.0, 0.0]] * 2,
                [1.0, 1.0], contact_full_share=0.5, **base)
        with self.assertRaises(ValueError):
            evolve_inertial_retarded(
                [[10.0, 12.0], [14.0, 12.0]], [[0.0, 0.0]] * 2,
                [1.0, 1.0],
                contact_full_labels=({"a": 1.0}, {"a": 1.0}), **base)

    def test_share_zero_force_is_the_no_exclusion_control(self):
        # φ = 0: the labelled run is the no-exclusion run — overlapping
        # blobs (s = 3) feel exactly the gravity-only force.
        common = dict(shape=(N, N), width=2.5, tau=0.1, dt=0.1,
                      steps=3, record_every=1, kappa_recovery=R,
                      kappa_consumption=C)
        pos = [[N / 2 - 1.5, N / 2], [N / 2 + 1.5, N / 2]]
        vel = [[0.0, 0.0], [0.0, 0.0]]
        h0 = evolve_inertial_retarded(pos, vel, [0.6, 0.6], **common)
        hz = evolve_inertial_retarded(pos, vel, [0.6, 0.6],
                                      contact_full=True,
                                      contact_full_share=0.0, **common)
        for a, b in zip(h0["velocities"], hz["velocities"]):
            self.assertLess(np.max(np.abs(a - b)), 1e-12)
        for a, b in zip(h0["energy"], hz["energy"]):
            self.assertLess(abs(a - b), 1e-12)

    def test_smoothed_min_labelled_force_conserves_momentum(self):
        # the equal mirrored binary at a fractional share: the smoothed
        # min splits the tie-plane 50/50 per shared type, so the pair's
        # total momentum stays at zero to machine precision.
        hist = evolve_inertial_retarded(
            [[21.0, 24.0], [27.0, 24.0]], [[0.0, 0.0], [0.0, 0.0]],
            [0.6, 0.6], shape=(N, N), width=2.5, tau=0.1, dt=0.1,
            steps=20, record_every=5, contact_full=True,
            contact_full_share=0.5, kappa_recovery=R, kappa_consumption=C)
        vs = np.asarray(hist["velocities"])
        self.assertLess(np.max(np.abs(vs[:, 0, :] + vs[:, 1, :])), 1e-12)

    def test_general_labels_force_is_the_energy_gradient(self):
        # the multi-shared-type path with UNEQUAL weights breaks the
        # mirror symmetry, so the pair force need not sum to zero to
        # machine precision — but it must remain the exact gradient of
        # the booked E_x: the residual pair force equals minus the
        # energy's common-translation derivative (the lattice artifact
        # of shifting sampled Gaussians), and the Lyapunov law holds.
        labels = ({"a": 0.6, "b": 0.4}, {"a": 0.5, "b": 0.5})
        common = dict(shape=(N, N), width=2.5, tau=0.1, dt=0.1,
                      record_every=1, contact_full=True,
                      contact_full_labels=labels,
                      kappa_recovery=R, kappa_consumption=C)

        def labelled_ex(pos):
            hist = evolve_inertial_retarded(pos, [[0.0, 0.0]] * 2,
                                            [0.6, 0.6], steps=1, **common)
            return (hist["energy"][0] - hist["field_energy"][0]
                    - hist["kinetic"][0] - hist["field_kinetic"][0])

        d = 0.02
        ex_p = labelled_ex([[21.0 + d, 24.0], [27.0 + d, 24.0]])
        ex_m = labelled_ex([[21.0 - d, 24.0], [27.0 - d, 24.0]])
        de_dd = (ex_p - ex_m) / (2.0 * d)

        pos = [[21.0, 24.0], [27.0, 24.0]]
        hist = evolve_inertial_retarded(pos, [[0.0, 0.0]] * 2, [0.6, 0.6],
                                        steps=3, **common)
        no_labels = {k: v for k, v in common.items()
                     if k not in ("contact_full", "contact_full_labels")}
        ctrl = evolve_inertial_retarded(pos, [[0.0, 0.0]] * 2, [0.6, 0.6],
                                        steps=3, **no_labels)
        v = np.asarray(hist["velocities"])
        v0 = np.asarray(ctrl["velocities"])
        self.assertTrue(np.all(np.isfinite(v)))
        dv = (v[-1] - v[0]) - (v0[-1] - v0[0])
        f_pair_x = 0.6 * (dv[0, 0] + dv[1, 0]) / (0.1 * (len(v) - 1))
        self.assertLess(abs(f_pair_x / (-de_dd) - 1.0), 0.05)

        long_run = evolve_inertial_retarded(
            [[18.0, 24.0], [30.0, 24.0]], [[0.0, 0.3], [0.0, -0.3]],
            [0.6, 0.6], shape=(N, N), width=2.5, tau=0.1, dt=0.1,
            steps=80, record_every=10, contact_full=True,
            contact_full_labels=labels, kappa_recovery=R,
            kappa_consumption=C)
        e_rec = np.asarray(long_run["energy"])
        self.assertLess(e_rec[-1], e_rec[0])
        self.assertTrue(np.all(np.diff(e_rec) <= 1e-6))

    def test_labelled_lyapunov_still_holds(self):
        # a short labelled run (φ = 0.5): the force is the exact
        # gradient of the recorded E_x per shared type, so the Lyapunov
        # law survives the labelling — same bar as the repo's checks.
        hist = evolve_inertial_retarded(
            [[18.0, 24.0], [30.0, 24.0]], [[0.0, 0.3], [0.0, -0.3]],
            [0.6, 0.6], shape=(N, N), width=2.5, tau=0.1, dt=0.1,
            steps=200, record_every=10, contact_full=True,
            contact_full_share=0.5, kappa_recovery=R, kappa_consumption=C)
        e_rec = np.asarray(hist["energy"])
        self.assertLess(e_rec[-1], e_rec[0])           # net dissipation
        self.assertTrue(np.all(np.diff(e_rec) <= 1e-6))  # Lyapunov
        self.assertTrue(all(d >= 0.0 for d in hist["dissipation"]))


if __name__ == "__main__":
    unittest.main()
