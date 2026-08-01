"""Tests for the RP2 nematic director and its homotopy invariants.

The load-bearing property is headlessness: the energy and the invariants must
be blind to flipping any director, because the whole difference between RP1 and
RP2 lives in that identification. A measure that quietly cares about the sign of
n is measuring S1 or S2 and would answer the wrong question entirely.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.nematic_rp2 import (  # noqa: E402
    core_escape,
    elastic_energy,
    imprint_disclination,
    loop_strength,
    loop_z2_class,
    relax,
    sample_director,
    tilted_core,
)


class TestNematicRp2(unittest.TestCase):

    # -------------------------------------------------------- headlessness

    def test_energy_is_blind_to_flipping_any_director(self):
        """n -> -n at arbitrary sites must not change the energy at all."""
        rng = np.random.default_rng(0)
        n = imprint_disclination(24, 0.5)
        flip = rng.random((24, 24)) < 0.5
        m = n.copy()
        m[:, flip] *= -1.0
        self.assertAlmostEqual(elastic_energy(n), elastic_energy(m), places=10)

    def test_z2_class_is_blind_to_flipping_directors(self):
        rng = np.random.default_rng(1)
        n = imprint_disclination(48, 0.5)
        flip = rng.random((48, 48)) < 0.5
        m = n.copy()
        m[:, flip] *= -1.0
        self.assertEqual(loop_z2_class(n, (24.0, 24.0), 12.0),
                         loop_z2_class(m, (24.0, 24.0), 12.0))

    def test_directors_stay_unit_length_through_relaxation(self):
        n = relax(imprint_disclination(24, 0.5), 40)
        norms = np.sqrt((n * n).sum(axis=0))
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-9))

    # --------------------------------------------------- the Z/2 invariant

    def test_z2_class_is_two_s_mod_two(self):
        """The whole of pi_1(RP2) = Z/2, on every texture we can build."""
        for s in (0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, 2.5):
            n = imprint_disclination(64, s)
            want = int(round(2 * abs(s))) % 2
            self.assertEqual(loop_z2_class(n, (32.0, 32.0), 14.0), want,
                             msg=f"strength {s}")

    def test_plus_and_minus_half_are_the_same_class(self):
        """In RP1 they are opposite charges; in RP2 they are one object."""
        a = imprint_disclination(64, 0.5)
        b = imprint_disclination(64, -0.5)
        self.assertEqual(loop_z2_class(a, (32.0, 32.0), 14.0),
                         loop_z2_class(b, (32.0, 32.0), 14.0))

    def test_uniform_field_is_the_trivial_class(self):
        n = np.zeros((3, 32, 32))
        n[0] = 1.0
        self.assertEqual(loop_z2_class(n, (16.0, 16.0), 8.0), 0)

    def test_z2_class_is_radius_independent_away_from_the_core(self):
        n = imprint_disclination(80, 0.5)
        classes = {loop_z2_class(n, (40.0, 40.0), r) for r in (10.0, 16.0, 24.0)}
        self.assertEqual(classes, {1})

    # ------------------------------------------------- the integer grading

    def test_strength_recovers_the_imprinted_value(self):
        for s in (0.5, -0.5, 1.0, -1.0, 1.5, 2.0):
            n = imprint_disclination(64, s)
            self.assertAlmostEqual(loop_strength(n, (32.0, 32.0), 14.0), s,
                                   delta=0.05)

    def test_strength_distinguishes_what_the_z2_class_cannot(self):
        """+1/2 and -1/2 share a Z/2 class but differ in the integer grading --
        which is exactly the information RP2 does not have."""
        a = imprint_disclination(64, 0.5)
        b = imprint_disclination(64, -0.5)
        self.assertAlmostEqual(loop_strength(a, (32.0, 32.0), 14.0), 0.5, delta=0.05)
        self.assertAlmostEqual(loop_strength(b, (32.0, 32.0), 14.0), -0.5, delta=0.05)

    # ------------------------------------------------------------- escape

    def test_tilted_core_really_is_tilted(self):
        n = tilted_core(64, 1.0)
        self.assertGreater(core_escape(n, (32.0, 32.0), 4.0), 0.5)

    def test_tilted_core_is_planar_far_away(self):
        n = tilted_core(64, 1.0, core=4.0)
        far = core_escape(n, (32.0, 32.0), 30.0)
        near = core_escape(n, (32.0, 32.0), 3.0)
        self.assertLess(far, near)

    def test_confined_relaxation_keeps_the_director_in_plane(self):
        n = relax(tilted_core(48, 1.0), 60, confine_plane=True)
        self.assertAlmostEqual(np.abs(n[2]).max(), 0.0, places=12)

    def test_integer_line_escapes_when_the_third_dimension_exists(self):
        """The trivial class is unprotected: seeded off the saddle it runs away."""
        n0 = tilted_core(64, 1.0)
        before = core_escape(n0, (32.0, 32.0), 4.0)
        n1 = relax(n0, 800, confine_plane=False, pin_radius=64 * 0.42)
        self.assertGreater(core_escape(n1, (32.0, 32.0), 4.0), before)

    def test_half_integer_line_cannot_escape(self):
        """The non-trivial class is protected: the same seed decays away."""
        n0 = tilted_core(64, 0.5)
        before = core_escape(n0, (32.0, 32.0), 4.0)
        n1 = relax(n0, 800, confine_plane=False, pin_radius=64 * 0.42)
        after = core_escape(n1, (32.0, 32.0), 4.0)
        self.assertLess(after, before)
        self.assertLess(after, 0.2)

    def test_the_two_classes_separate_under_identical_treatment(self):
        """The experiment's whole claim, as one assertion."""
        args = dict(confine_plane=False, pin_radius=64 * 0.42)
        one = relax(tilted_core(64, 1.0), 1200, **args)
        half = relax(tilted_core(64, 0.5), 1200, **args)
        self.assertGreater(core_escape(one, (32.0, 32.0), 4.0),
                           5 * core_escape(half, (32.0, 32.0), 4.0))

    # --------------------------------------------------------- relaxation

    def test_relaxation_does_not_increase_the_energy(self):
        n0 = imprint_disclination(32, 0.5)
        n1 = relax(n0, 200)
        self.assertLessEqual(elastic_energy(n1), elastic_energy(n0) + 1e-9)

    def test_pinned_boundary_is_actually_held_fixed(self):
        n0 = imprint_disclination(48, 0.5)
        n1 = relax(n0, 300, pin_radius=20.0)
        y, x = np.indices((48, 48)).astype(float)
        outside = np.hypot(x - 24.0, y - 24.0) >= 20.0
        self.assertTrue(np.allclose(n0[:, outside], n1[:, outside], atol=1e-9))

    def test_uniform_field_is_a_fixed_point(self):
        n = np.zeros((3, 24, 24))
        n[0] = 1.0
        self.assertTrue(np.allclose(relax(n, 100), n, atol=1e-9))

    def test_sample_director_returns_one_row_per_point(self):
        n = imprint_disclination(32, 0.5)
        d = sample_director(n, np.array([10.0, 20.0]), np.array([12.0, 22.0]))
        self.assertEqual(d.shape, (2, 3))
        self.assertTrue(np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-9))

    def test_mismatched_centres_and_strengths_are_rejected(self):
        with self.assertRaises(ValueError):
            imprint_disclination(32, [0.5, -0.5], centres=[(10.0, 10.0)])


if __name__ == "__main__":
    unittest.main()
