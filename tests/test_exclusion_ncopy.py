"""Checks for the n-copy sector (Part IV): three-and-more identical stacks.

The pairwise-min instrument prices each shared pair separately; the
general form for a group of ``n`` same-label masses is the n-copy gap
``E_x = n·E(m) − E(n·m)`` with ``m`` the min over the group's
(shared-type) loads.  At O(c²) the two are EQUAL — the pairwise-summed
exclusion and the true n-copy gap are both ``k·n(n−1)`` in linear
response — so their split at the operating amplitude is a pure
core-saturation measurement.  These tests guard the n = 2 reduction
(bitwise the base branch's pair form, both forms), the dilute N1
theorem, the group smoothed min's symmetry/gradient algebra, the
label routing with n > 2 present (a group of 2 + a singleton prices
only the pair), and the Lyapunov/momentum bookkeeping of the trimer.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import gaussian_load  # noqa: E402
from project_genesis.capacity_waves import (  # noqa: E402
    _smoothed_min,
    _smoothed_min_group,
    _smoothed_min_weight,
    duplicated_load,
    evolve_inertial_retarded,
    exclusion_gap_full,
    exclusion_gap_group,
    group_duplicated_component,
)

R, C = 0.02, 0.8          # the exclusion-core operating point
FKW = dict(kappa_diffusion=1.0, kappa_recovery=R, kappa_consumption=C,
           kappa_baseline=1.0)
N = 48                   # tests run at 48²; the experiment's record is 96²

# recorded on the base branch (kimi/exclusion-labelled-load, 5e2368e)
# before the generalization — the n-copy sector must reproduce the pair
# form exactly at n = 2 (spec bar 1e-8; measured bitwise).
PIN_STATICS_S6 = 0.73159256758897584
PIN_BOOKED_EX_S6 = 0.73120365796494458
PIN_DV_STEP1 = (-0.038222276336428745, 0.038222276336428732)


def pair(s, mass=0.6):
    """The mirrored equal binary at separation s (centres on lattice sites)."""
    l1 = gaussian_load((N, N), [(N / 2 - s / 2, N / 2)], 2.5, mass)
    l2 = gaussian_load((N, N), [(N / 2 + s / 2, N / 2)], 2.5, mass)
    return l1, l2


def trimer_pos():
    """A mirror-symmetric isosceles trimer on lattice sites (48²).

    Apex on the mirror axis x = 24, base pair mirrored about it — the
    lattice reflection maps the configuration to itself, so the
    x-channel of the total momentum is exact; the y-channel carries
    the sub-lattice translation artifact.
    """
    return [[24.0, 30.0], [18.0, 20.0], [30.0, 20.0]]


def booked_ex(hist):
    """The E_x booked in a run's energy record at the first record."""
    return (hist["energy"][0] - hist["field_energy"][0]
            - hist["kinetic"][0] - hist["field_kinetic"][0])


class NCopyReductionTests(unittest.TestCase):
    def test_statics_two_copies_reduce_to_the_pair_form_bitwise(self):
        # both forms of the group gap at n = 2 ARE the base branch's
        # 2E(ρ_dup) − E(2ρ_dup): the single "pair" of a 2-group and the
        # n-copy gap nE(m) − E(nm) at n = 2 are the same computation.
        l1, l2 = pair(6.0)
        base = exclusion_gap_full(duplicated_load(l1, l2), **FKW)
        self.assertEqual(exclusion_gap_group([l1, l2], n_copy=True, **FKW),
                         base)
        self.assertEqual(exclusion_gap_group([l1, l2], n_copy=False, **FKW),
                         base)
        # explicit unit weights are the default group
        self.assertEqual(
            exclusion_gap_group([l1, l2], [1.0, 1.0], n_copy=True, **FKW),
            base)
        # and the value is the base branch's own, digit for digit
        self.assertAlmostEqual(base, PIN_STATICS_S6, places=12)

    def test_dynamics_two_copies_reproduce_the_pair_form(self):
        # the generalized path with two masses reproduces the base pair
        # form's booked energy and forces exactly (spec bar 1e-8; the
        # implementation is bitwise): default labels, n_copy True and
        # False, and explicit identical labels all coincide.
        common = dict(shape=(N, N), width=2.5, tau=0.1, dt=0.1, steps=3,
                      record_every=1, contact_full=True,
                      kappa_recovery=R, kappa_consumption=C)
        pos = [[N / 2 - 3.0, N / 2], [N / 2 + 3.0, N / 2]]
        vel = [[0.0, 0.0], [0.0, 0.0]]
        h_def = evolve_inertial_retarded(pos, vel, [0.6, 0.6], **common)
        v_def = np.asarray(h_def["velocities"])
        e_def = np.asarray(h_def["energy"])
        for opts in (dict(contact_full_ncopy=True),
                     dict(contact_full_ncopy=False),
                     dict(contact_full_labels=({"a": 1.0}, {"a": 1.0}))):
            h = evolve_inertial_retarded(pos, vel, [0.6, 0.6], **common,
                                         **opts)
            self.assertLess(np.max(np.abs(np.asarray(h["velocities"])
                                          - v_def)), 1e-8)
            self.assertLess(np.max(np.abs(np.asarray(h["energy"])
                                          - e_def)), 1e-8)
        # the booked E_x and the exclusion force kicks are the base
        # branch's pinned values
        self.assertAlmostEqual(booked_ex(h_def), PIN_BOOKED_EX_S6,
                               places=12)
        no_contact = {k: v for k, v in common.items() if k != "contact_full"}
        ctrl = evolve_inertial_retarded(pos, vel, [0.6, 0.6], **no_contact)
        dv = (np.asarray(h_def["velocities"])[1]
              - np.asarray(ctrl["velocities"])[1])
        self.assertAlmostEqual(float(dv[0, 0]), PIN_DV_STEP1[0], places=12)
        self.assertAlmostEqual(float(dv[1, 0]), PIN_DV_STEP1[1], places=12)

    def test_ncopy_options_require_contact_full(self):
        base = dict(shape=(24, 24), steps=2, kappa_recovery=R,
                    kappa_consumption=C)
        pos = [[10.0, 12.0], [14.0, 12.0]]
        with self.assertRaises(ValueError):
            evolve_inertial_retarded(pos, [[0.0, 0.0]] * 2, [1.0, 1.0],
                                     contact_full_ncopy=False, **base)
        # a single mass clones nothing
        with self.assertRaises(ValueError):
            evolve_inertial_retarded([[12.0, 12.0]], [[0.0, 0.0]], [1.0],
                                     contact_full=True, **base)
        # one label dict per mass
        with self.assertRaises(ValueError):
            evolve_inertial_retarded(
                pos + [[12.0, 18.0]], [[0.0, 0.0]] * 3, [1.0] * 3,
                contact_full=True,
                contact_full_labels=({"a": 1.0}, {"a": 1.0}), **base)
        with self.assertRaises(ValueError):
            evolve_inertial_retarded(pos, [[0.0, 0.0]] * 2, [1.0, 1.0],
                                     contact_full=True,
                                     contact_full_share=1.5, **base)


class GroupMinTests(unittest.TestCase):
    def test_two_arguments_are_the_pair_smoothed_min_bitwise(self):
        rng = np.random.default_rng(1)
        a, b = rng.random((2, 16, 16))
        m, dws = _smoothed_min_group([a, b], 1e-4)
        np.testing.assert_array_equal(m, _smoothed_min(a, b, 1e-4))
        self.assertEqual(len(dws), 2)
        np.testing.assert_array_equal(
            dws[0], _smoothed_min_weight(a, b, 1e-4))
        np.testing.assert_array_equal(
            dws[1], _smoothed_min_weight(b, a, 1e-4))

    def test_group_min_algebra(self):
        # symmetric under permutations, bounded by the hard min within
        # (n−1)ε, weights the exact partials summing to 1 (the gradient
        # the force is built from).
        rng = np.random.default_rng(0)
        a, b, c = rng.random((3, 24, 24))
        m, dws = _smoothed_min_group([a, b, c], 1e-4)
        hard = group_duplicated_component([a, b, c])
        self.assertTrue(np.all(m <= hard + 1e-15))
        self.assertLess(float(np.max(hard - m)), 2.0e-4)
        self.assertLess(float(np.max(np.abs(dws[0] + dws[1] + dws[2]
                                            - 1.0))), 1e-12)
        m_p, dws_p = _smoothed_min_group([c, a, b], 1e-4)
        np.testing.assert_allclose(m_p, m, atol=1e-15)
        np.testing.assert_allclose(dws_p[1], dws[0], atol=1e-15)
        # at the exact triple tie the weights split 1/3-1/3-1/3 (the
        # equilateral trimer's tie point), and the value is the
        # symmetrized iteration's own: smin2(x, x) = x − ε, so
        # m = smin2(a − ε, a) = a − (ε/2)(1 + √5)
        m_t, dws_t = _smoothed_min_group([a, a, a], 1e-4)
        np.testing.assert_allclose(
            m_t, a - 0.5 * (1.0 + np.sqrt(5.0)) * 1e-4, atol=1e-12)
        for d in dws_t:
            np.testing.assert_allclose(d, 1.0 / 3.0, atol=1e-12)


class NCopyTheoremTests(unittest.TestCase):
    def test_dilute_triple_pairwise_equals_ncopy(self):
        # N1 at the test level: a fully-stacked DILUTE triple (amplitude
        # 0.01) — the pairwise-summed exclusion and the n-copy gap agree
        # within 5% (both are k·n(n−1) at O(c²); measured 4.5%, the
        # higher-order remainder at this amplitude).
        blob = gaussian_load((N, N), [(N / 2, N / 2)], 2.5, 0.01)
        loads = [blob, blob, blob]
        pw = exclusion_gap_group(loads, n_copy=False, **FKW)
        nc = exclusion_gap_group(loads, n_copy=True, **FKW)
        self.assertGreater(pw, 0.0)
        self.assertGreater(nc, 0.0)
        self.assertLess(abs(pw / nc - 1.0), 0.05)


class NCopyLabelRoutingTests(unittest.TestCase):
    def test_group_of_two_plus_singleton_prices_only_the_pair(self):
        # labels (a, a, b) — or (b, a, a), so the shared pair is not
        # always masses (0, 1): the shared type "a" groups the two
        # a-carriers; the singleton "b" clones nothing.  The booked
        # E_x is the pair's own (the 2-mass run's at the same geometry,
        # both forms), and the singleton feels exactly the
        # gravity-only force — the follow-up-#2 routing with n > 2
        # present.
        vel = [[0.0, 0.0]] * 3
        base = dict(shape=(N, N), width=2.5, tau=0.1, dt=0.1, steps=3,
                    record_every=1, contact_full=True, kappa_recovery=R,
                    kappa_consumption=C)
        # the singleton sits 24 cells from the pair — the far side of
        # the box, Gaussian tail ~e^-46 at the pair: its own exclusion
        # force is exactly zero (it is in no group), and the pair's
        # exclusion-driven motion can only reach it through the
        # RETARDED κ field — front speed √(D/τ) ≈ 3.2 cells per unit
        # time, i.e. ~1 cell over the 3 steps — so the singleton's
        # velocities are bitwise the no-exclusion control's here
        # (measured 6e-18; at 20 cells the tail's field echo is 7e-12).
        arrangements = (
            # (positions, labels, singleton index, pair indices)
            ([[N / 2 - 3.0, N / 2], [N / 2 + 3.0, N / 2],
              [N / 2, N / 2 + 24.0]],
             ({"a": 1.0}, {"a": 1.0}, {"b": 1.0}), 2, (0, 1)),
            ([[N / 2, N / 2 + 24.0], [N / 2 - 3.0, N / 2],
              [N / 2 + 3.0, N / 2]],
             ({"b": 1.0}, {"a": 1.0}, {"a": 1.0}), 0, (1, 2)),
        )
        for pos, labels, i_single, i_pair in arrangements:
            common = dict(base, contact_full_labels=labels)
            for opts in (dict(contact_full_ncopy=True),
                         dict(contact_full_ncopy=False)):
                h = evolve_inertial_retarded(pos, vel, [0.6] * 3,
                                             **common, **opts)
                h_pair = evolve_inertial_retarded(
                    [pos[i_pair[0]], pos[i_pair[1]]], vel[:2], [0.6] * 2,
                    **{**base,
                       "contact_full_labels": ({"a": 1.0}, {"a": 1.0})},
                    **opts)
                self.assertAlmostEqual(booked_ex(h), booked_ex(h_pair),
                                       places=12)
                no_contact = {k: v for k, v in common.items()
                              if k not in ("contact_full",
                                           "contact_full_labels")}
                ctrl = evolve_inertial_retarded(pos, vel, [0.6] * 3,
                                                **no_contact)
                dv_single = (np.asarray(h["velocities"])[:, i_single, :]
                             - np.asarray(ctrl["velocities"])
                             [:, i_single, :])
                self.assertLess(np.max(np.abs(dv_single)), 1e-12)
                # the pair members DO feel the exclusion: their
                # velocities differ from the control's
                dv_pair = (np.asarray(h["velocities"])[:, i_pair[0], :]
                           - np.asarray(ctrl["velocities"])
                           [:, i_pair[0], :])
                self.assertGreater(np.max(np.abs(dv_pair)), 1e-4)


class NCopyDynamicsTests(unittest.TestCase):
    def test_trimer_lyapunov_both_forms(self):
        # a moving same-label trimer: the force is the exact gradient
        # of the recorded E_x per group/pair, so the Lyapunov law holds
        # at the repo's 1e-6 bar for both forms.
        vel = [[0.0, -0.2], [0.15, 0.1], [-0.15, 0.1]]
        for opts in (dict(contact_full_ncopy=True),
                     dict(contact_full_ncopy=False)):
            hist = evolve_inertial_retarded(
                trimer_pos(), vel, [0.6] * 3, shape=(N, N), width=2.5,
                tau=0.1, dt=0.1, steps=200, record_every=10,
                contact_full=True, kappa_recovery=R, kappa_consumption=C,
                **opts)
            e_rec = np.asarray(hist["energy"])
            self.assertLess(e_rec[-1], e_rec[0])           # net dissipation
            self.assertTrue(np.all(np.diff(e_rec) <= 1e-6))  # Lyapunov
            self.assertTrue(all(d >= 0.0 for d in hist["dissipation"]))

    def test_trimer_mirror_momentum_is_machine_precision(self):
        # the isosceles trimer on lattice sites is mirror-symmetric:
        # the x-channel of the total momentum stays at zero to machine
        # precision for both forms (the symmetrized group min splits
        # the tie exactly; the pairwise sum mirrors pair for pair).
        for opts in (dict(contact_full_ncopy=True),
                     dict(contact_full_ncopy=False)):
            hist = evolve_inertial_retarded(
                trimer_pos(), [[0.0, 0.0]] * 3, [0.6] * 3, shape=(N, N),
                width=2.5, tau=0.1, dt=0.1, steps=20, record_every=5,
                contact_full=True, kappa_recovery=R, kappa_consumption=C,
                **opts)
            vs = np.asarray(hist["velocities"])
            self.assertLess(np.max(np.abs(vs.sum(axis=1)[:, 0])), 1e-12)

    def test_trimer_force_is_the_energy_gradient(self):
        # the mirror channel is exact, but the common-translation
        # channel is the lattice artifact (the analytic Gaussians do
        # not translate exactly on the lattice): the summed exclusion
        # force along y must equal −dE_x/d(common y shift) — the same
        # bookkeeping bar as the labelled pair's (5%).  Measured 0.4%.
        common = dict(shape=(N, N), width=2.5, tau=0.1, dt=0.1,
                      record_every=1, contact_full=True,
                      kappa_recovery=R, kappa_consumption=C)

        def ex_at(pos):
            hist = evolve_inertial_retarded(pos, [[0.0, 0.0]] * 3,
                                            [0.6] * 3, steps=1, **common)
            return booked_ex(hist)

        d = 0.02
        pos = trimer_pos()
        ex_p = ex_at([[x, y + d] for x, y in pos])
        ex_m = ex_at([[x, y - d] for x, y in pos])
        de_dd = (ex_p - ex_m) / (2.0 * d)
        hist = evolve_inertial_retarded(pos, [[0.0, 0.0]] * 3, [0.6] * 3,
                                        steps=3, **common)
        no_contact = {k: v for k, v in common.items()
                      if k != "contact_full"}
        ctrl = evolve_inertial_retarded(pos, [[0.0, 0.0]] * 3, [0.6] * 3,
                                        steps=3, **no_contact)
        v = np.asarray(hist["velocities"])
        v0 = np.asarray(ctrl["velocities"])
        dv = (v[-1] - v[0]) - (v0[-1] - v0[0])
        f_y = 0.6 * float(dv[:, 1].sum()) / (0.1 * (len(v) - 1))
        self.assertLess(abs(f_y / (-de_dd) - 1.0), 0.05)


if __name__ == "__main__":
    unittest.main()
