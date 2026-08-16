"""Tests for the coarse-graining flow.

The point of the module is to sort observables into ones that survive the choice
of cut and ones that do not. That only means something if the three schemes are
genuinely different operations — if they secretly agree, "scheme-independent" is
vacuous and every observable passes. So the first block proves they disagree on
a case built to separate them.

The second block pins the observables as *dimensionless*, which is what lets
levels be compared at all: a quantity carrying lattice units would flow purely
because the lattice shrank, and that flow would mean nothing.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.coarse_grain import (  # noqa: E402
    SCHEMES, block_decimate, block_majority, block_random,
    largest_component_fraction, observables, order_parameter, rg_flow,
    wall_density,
)


class TestBlockingShrinksTheLattice(unittest.TestCase):

    def test_every_scheme_divides_each_axis_by_the_block_factor(self):
        rng = np.random.default_rng(0)
        for d in (1, 2, 3):
            lab = rng.integers(0, 3, (16,) * d)
            for name, fn in SCHEMES.items():
                out = fn(lab, 2, 3, rng)
                self.assertEqual(out.shape, (8,) * d, msg=f"{name} {d}D")

    def test_a_non_divisible_axis_is_cropped_not_padded(self):
        """Padding would invent data at the edge; cropping only discards."""
        rng = np.random.default_rng(1)
        lab = rng.integers(0, 3, (17, 15))
        for name, fn in SCHEMES.items():
            self.assertEqual(fn(lab, 2, 3, rng).shape, (8, 7), msg=name)

    def test_a_block_factor_below_two_is_rejected(self):
        with self.assertRaises(ValueError):
            block_majority(np.zeros((8, 8), dtype=int), 1, 2)


class TestTheSchemesAreGenuinelyDifferent(unittest.TestCase):
    """Without this, 'scheme-independent' would be a free pass."""

    def test_majority_erases_a_minority_that_decimation_keeps(self):
        """One odd cell per block, placed at the sampled corner. Majority votes
        it away; decimation returns exactly it. If these ever agree here, the
        schemes are not probing different things and R2 is meaningless."""
        lab = np.zeros((8, 8), dtype=int)
        lab[::2, ::2] = 1                      # the decimated corner of each block
        maj = block_majority(lab, 2, 2)
        dec = block_decimate(lab, 2, 2)
        self.assertTrue(np.all(maj == 0))
        self.assertTrue(np.all(dec == 1))

    def test_random_picks_vary_with_the_generator(self):
        lab = np.arange(64).reshape(8, 8) % 4
        a = block_random(lab, 2, 4, np.random.default_rng(1))
        b = block_random(lab, 2, 4, np.random.default_rng(2))
        self.assertFalse(np.array_equal(a, b))

    def test_all_schemes_agree_on_a_uniform_field(self):
        """The sanity floor: where there is nothing to disagree about, they
        must not disagree."""
        lab = np.full((8, 8), 2, dtype=int)
        for name, fn in SCHEMES.items():
            self.assertTrue(np.all(fn(lab, 2, 3, np.random.default_rng(0)) == 2),
                            msg=name)


class TestObservablesAreDimensionless(unittest.TestCase):
    """Fractions, so two lattice sizes are comparable."""

    def test_a_uniform_field_has_no_walls_and_total_integration(self):
        lab = np.zeros((16, 16), dtype=int)
        self.assertEqual(wall_density(lab), 0.0)
        self.assertAlmostEqual(order_parameter(lab, 3), 1.0)
        self.assertAlmostEqual(largest_component_fraction(lab), 1.0)

    def test_noise_is_almost_all_wall(self):
        rng = np.random.default_rng(3)
        self.assertGreater(wall_density(rng.integers(0, 4, (32, 32))), 0.95)

    def test_wall_density_is_size_invariant_for_the_same_structure(self):
        """Two lattices, same stripe geometry scaled up. A quantity that
        changed here would flow under blocking for no reason at all."""
        small = np.zeros((16, 16), dtype=int)
        small[:, 8:] = 1
        big = np.zeros((32, 32), dtype=int)
        big[:, 16:] = 1
        self.assertLess(abs(wall_density(small) - 2 * wall_density(big)), 0.02)

    def test_order_and_integration_separate_islands_from_one_piece(self):
        """The two are not redundant: a sector can hold half the field as many
        islands (high order, low integration) or as one block (both high)."""
        islands = np.indices((16, 16)).sum(axis=0) % 2
        onepiece = np.zeros((16, 16), dtype=int)
        onepiece[:, 8:] = 1
        self.assertAlmostEqual(order_parameter(islands, 2),
                               order_parameter(onepiece, 2), places=6)
        self.assertLess(largest_component_fraction(islands),
                        largest_component_fraction(onepiece))

    def test_observables_reports_every_key_the_flow_uses(self):
        o = observables(np.zeros((8, 8), dtype=int), 2)
        for k in ("wall_density", "order_parameter", "largest_component",
                  "cells"):
            self.assertIn(k, o)


class TestTheFlow(unittest.TestCase):

    def test_it_records_the_original_field_as_level_zero(self):
        lab = np.zeros((32, 32), dtype=int)
        flow = rg_flow(lab, 2, "majority", levels=2)
        self.assertEqual(flow[0]["cells"], 32 * 32)

    def test_each_level_shrinks_by_the_block_factor(self):
        rng = np.random.default_rng(4)
        flow = rg_flow(rng.integers(0, 3, (64, 64)), 3, "decimate", levels=3)
        self.assertEqual([lv["cells"] for lv in flow],
                         [64 ** 2, 32 ** 2, 16 ** 2, 8 ** 2])

    def test_it_stops_before_the_lattice_is_too_small_to_read(self):
        """Below min_side the numbers are about the boundary, not the field."""
        rng = np.random.default_rng(5)
        flow = rg_flow(rng.integers(0, 3, (32, 32)), 3, "majority",
                       levels=10, min_side=8)
        self.assertLessEqual(min(lv["cells"] for lv in flow), 32 * 32)
        self.assertGreaterEqual(min(lv["cells"] for lv in flow), 8 * 8)

    def test_an_unknown_scheme_is_rejected(self):
        with self.assertRaises(ValueError):
            rg_flow(np.zeros((8, 8), dtype=int), 2, "wishful")

    def test_a_uniform_field_is_a_fixed_point_of_every_scheme(self):
        """Nothing to coarse-grain: the flow must not manufacture motion."""
        lab = np.full((32, 32), 1, dtype=int)
        for s in SCHEMES:
            flow = rg_flow(lab, 3, s, levels=2)
            self.assertTrue(all(lv["wall_density"] == 0.0 for lv in flow),
                            msg=s)
            self.assertTrue(all(abs(lv["order_parameter"] - 1.0) < 1e-12
                                for lv in flow), msg=s)


if __name__ == "__main__":
    unittest.main()
