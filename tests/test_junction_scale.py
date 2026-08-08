"""Tests for the neighbourhood-scale audit of the selection measure.

This audit only means something if radius 1 is *exactly* the published probe.
If the generalised neighbourhood differs from `multiphase`'s by even one offset,
then widening it is not widening the published measure and every conclusion
about §2 is about a different instrument. So the first block pins the
equivalence in both dimensions and at every palette size, rather than assuming
it from the code looking similar.

The second block pins the monotonicity the whole argument leans on — a bigger
window can only ever see more colours, never fewer — and the third pins the
magnitude/ranking split that the experiment scores, including the degenerate
cases where a margin is infinite or undefined.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.junction_scale import (  # noqa: E402
    full_palette_density,
    neighbourhood_label_bits,
    neighbourhood_offsets,
    selection_profile,
    valence_profile,
)
from project_genesis.multiphase import (  # noqa: E402
    _neighbour_offsets,
    domain_diameter,
    _neighbourhood_label_bits,
    full_palette_junction_density,
)


class TestRadiusOneIsThePublishedProbe(unittest.TestCase):
    """Without this the audit is of some other measure."""

    def test_offsets_match_the_published_ring(self):
        for ndim in (2, 3):
            self.assertEqual(sorted(neighbourhood_offsets(ndim, 1)),
                             sorted(_neighbour_offsets(ndim)))

    def test_label_bits_match_exactly(self):
        rng = np.random.default_rng(0)
        for ndim in (2, 3):
            for p in (2, 3, 5):
                lab = rng.integers(0, p, (12,) * ndim)
                self.assertTrue(np.array_equal(
                    neighbourhood_label_bits(lab, 1),
                    _neighbourhood_label_bits(lab)), msg=f"{ndim}D P={p}")

    def test_density_matches_exactly(self):
        rng = np.random.default_rng(1)
        for ndim in (2, 3):
            for p in (2, 3, 4, 5, 6):
                lab = rng.integers(0, p, (12,) * ndim)
                self.assertAlmostEqual(full_palette_density(lab, p, 1),
                                       full_palette_junction_density(lab, p),
                                       places=15, msg=f"{ndim}D P={p}")


class TestNeighbourhoodGrowth(unittest.TestCase):

    def test_offset_count_is_the_box_minus_the_centre(self):
        for ndim in (2, 3):
            for r in (1, 2, 3):
                self.assertEqual(len(neighbourhood_offsets(ndim, r)),
                                 (2 * r + 1) ** ndim - 1)

    def test_radius_zero_has_no_neighbours(self):
        self.assertEqual(neighbourhood_offsets(2, 0), [])

    def test_a_negative_radius_is_rejected(self):
        with self.assertRaises(ValueError):
            neighbourhood_offsets(2, -1)

    def test_bits_are_monotone_in_the_radius(self):
        """A bigger window sees everything a smaller one does. The whole
        argument — widening can only add colours — depends on this."""
        rng = np.random.default_rng(2)
        lab = rng.integers(0, 5, (16, 16))
        small = neighbourhood_label_bits(lab, 1)
        big = neighbourhood_label_bits(lab, 2)
        self.assertTrue(np.all((small & big) == small))

    def test_mean_valence_is_non_decreasing_and_ceilings_at_the_palette(self):
        """It rises, then saturates: once every window holds every colour there
        is nowhere further to go. Both halves matter — the rise is why widening
        degrades the measure, and the ceiling is why radius 3 is already
        measuring a region rather than a vertex."""
        rng = np.random.default_rng(3)
        lab = rng.integers(0, 4, (20, 20))
        vals = [valence_profile(lab, r)["mean"] for r in (1, 2, 3)]
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(vals, vals[1:])),
                        msg=str(vals))
        self.assertLess(vals[0], vals[1])
        self.assertLessEqual(vals[-1], 4.0 + 1e-12)
        self.assertAlmostEqual(vals[-1], 4.0, places=6)   # saturated

    def test_a_larger_palette_has_not_saturated_at_radius_three(self):
        rng = np.random.default_rng(4)
        lab = rng.integers(0, 9, (24, 24))
        vals = [valence_profile(lab, r)["mean"] for r in (1, 2, 3)]
        self.assertTrue(all(a < b for a, b in zip(vals, vals[1:])), msg=str(vals))

    def test_a_uniform_field_has_valence_one_at_every_radius(self):
        lab = np.zeros((10, 10), dtype=int)
        for r in (1, 2, 3):
            self.assertAlmostEqual(valence_profile(lab, r)["mean"], 1.0)

    def test_a_single_domain_never_carries_a_full_palette(self):
        lab = np.zeros((10, 10), dtype=int)
        for r in (1, 2, 3):
            self.assertEqual(full_palette_density(lab, 3, r), 0.0)


class TestFullPaletteConditions(unittest.TestCase):

    def test_a_two_colour_stripe_has_no_junction(self):
        """Two sectors meet along a wall, never at a vertex — density 0."""
        lab = np.zeros((12, 12), dtype=int)
        lab[:, 6:] = 1
        self.assertEqual(full_palette_density(lab, 2, 1), 0.0)

    def test_a_three_colour_vertex_registers(self):
        lab = np.zeros((12, 12), dtype=int)
        lab[:6, 6:] = 1
        lab[6:, 6:] = 2
        self.assertGreater(full_palette_density(lab, 3, 1), 0.0)

    def test_a_three_colour_field_cannot_fill_a_four_palette(self):
        """`bits == full` is exact: a missing colour disqualifies every cell."""
        lab = np.zeros((12, 12), dtype=int)
        lab[:6, 6:] = 1
        lab[6:, 6:] = 2
        self.assertEqual(full_palette_density(lab, 4, 1), 0.0)


class TestSelectionProfile(unittest.TestCase):
    """The magnitude/ranking split the experiment scores separately."""

    def test_the_winner_is_the_argmax(self):
        p = selection_profile({2: 0.0, 3: 0.5, 4: 0.1})
        self.assertEqual(p["argmax"], 3)
        self.assertTrue(p["selects"])

    def test_margin_is_winner_over_best_rival(self):
        p = selection_profile({2: 0.0, 3: 0.6, 4: 0.2})
        self.assertAlmostEqual(p["margin"], 3.0)

    def test_a_clean_sweep_gives_an_infinite_margin(self):
        p = selection_profile({2: 0.0, 3: 0.4, 4: 0.0})
        self.assertTrue(np.isinf(p["margin"]))
        self.assertTrue(p["selects"])

    def test_nothing_selected_when_every_reading_is_zero(self):
        p = selection_profile({2: 0.0, 3: 0.0, 4: 0.0})
        self.assertFalse(p["selects"])
        self.assertEqual(p["margin"], 0.0)

    def test_an_empty_sweep_is_handled(self):
        self.assertFalse(selection_profile({})["selects"])

    def test_the_ranking_survives_a_shared_rescaling_the_margin_does_not(self):
        """The structural claim, as a test: a convention that scales every
        reading leaves the winner alone. A convention that adds a floor to all
        of them leaves the winner alone too, but crushes the margin."""
        base = {2: 0.0, 3: 0.4, 4: 0.1}
        scaled = {k: v * 7.0 for k, v in base.items()}
        floored = {k: v + 0.3 for k, v in base.items()}
        self.assertEqual(selection_profile(scaled)["argmax"],
                         selection_profile(base)["argmax"])
        self.assertAlmostEqual(selection_profile(scaled)["margin"],
                               selection_profile(base)["margin"])
        self.assertEqual(selection_profile(floored)["argmax"],
                         selection_profile(base)["argmax"])
        self.assertLess(selection_profile(floored)["margin"],
                        selection_profile(base)["margin"])


class TestOneDimension(unittest.TestCase):
    """The d=1 arm, which is what turns d+1 from two points into a trend.

    1-D is the degenerate end and the place the law is most likely to break: a
    lattice has no junctions in the 2-D sense at all, so a "vertex" separating
    two sectors is simply a wall. If d+1 held in the plane and in space but not
    on a line, the honest claim would be `d >= 2` — so this is worth pinning
    rather than assuming.
    """

    def test_the_neighbourhood_is_the_two_adjacent_cells(self):
        self.assertEqual(sorted(neighbourhood_offsets(1, 1)), [(-1,), (1,)])

    def test_a_two_colour_wall_is_a_junction_at_min_valence_two(self):
        lab = np.zeros(40, dtype=int)
        lab[20:] = 1
        self.assertGreater(full_palette_density(lab, 2, 1, min_valence=2), 0.0)

    def test_the_published_2d_condition_finds_nothing_in_1d(self):
        """`distinct >= 3` cannot be met by two sectors on a line, which is why
        the dimension-blind constant hides the answer here exactly as it does
        in 3-D — in the opposite direction."""
        lab = np.zeros(40, dtype=int)
        lab[20:] = 1
        self.assertEqual(full_palette_density(lab, 2, 1, min_valence=3), 0.0)

    def test_three_colours_cannot_fill_a_two_cell_neighbourhood_plus_centre(self):
        """A 1-D radius-1 window holds three cells, so a 3-palette *can* just
        fit — the exclusion of P>2 in the measured fields is dynamical, not a
        counting artefact, and this test keeps the two explanations separate."""
        lab = np.array([0, 1, 2] * 12)
        self.assertGreater(full_palette_density(lab, 3, 1, min_valence=3), 0.0)

    def test_a_larger_palette_than_the_window_can_hold_is_always_zero(self):
        lab = np.array([0, 1, 2, 3] * 10)
        self.assertEqual(full_palette_density(lab, 4, 1, min_valence=2), 0.0)


class TestDimensionCorrectedTextureGuard(unittest.TestCase):
    """The second dimension-blind constant in this measure.

    `domain_scale` is compared against a fixed floor of 2.5 wherever it guards a
    junction reading, and that floor does not mean the same thing in every
    dimension: a cell is wall if any of its 3^d - 1 neighbours differs, and that
    ring grows exponentially, so a larger share of any domain sits within one
    step of its surface as d rises. Measured, a 2-D field at scale 5.14 and a
    4-D field at scale 2.31 hold domains of 19.5 and 15.1 lattice units — the
    same kind of structure, on opposite sides of the cut.

    `domain_diameter` inverts the relation so the guard means one thing
    everywhere. These tests pin the property that makes it worth having.
    """

    def test_noise_reads_one_in_every_dimension(self):
        """The whole point. Texture is texture in 1-D or 4-D, and the guard
        must say so with the same number, which a raw scale does not."""
        rng = np.random.default_rng(0)
        for d in (2, 3, 4):
            lab = rng.integers(0, 4, (12,) * d)
            self.assertAlmostEqual(domain_diameter(lab), 1.0, places=6,
                                   msg=f"{d}-D")

    def test_a_single_domain_has_no_width_to_report(self):
        """inf rather than a number, because any finite answer for a field with
        no walls is a fact about the box, not the structure."""
        self.assertTrue(np.isinf(domain_diameter(np.zeros((10, 10), dtype=int))))

    def test_it_recovers_a_known_compact_domain_size(self):
        """Checkerboard blocks of width w: the guard should land near w."""
        for w in (4, 8):
            n = 4 * w
            idx = np.arange(n) // w
            lab = ((idx[:, None] + idx[None, :]) % 2).astype(int)
            got = domain_diameter(lab)
            self.assertGreater(got, w * 0.5, msg=str(w))
            self.assertLess(got, w * 2.5, msg=str(w))

    def test_a_slab_over_reads_by_about_the_dimension(self):
        """The documented limit, pinned so it stays documented. Stripes vary on
        one axis and carry walls on one axis instead of d, so the compact-domain
        assumption over-reads — by about a factor of d."""
        lab = np.zeros((40, 40), dtype=int)
        lab[:, 20:] = 1
        got = domain_diameter(lab)
        self.assertGreater(got, 30.0)     # true slab width is 20
        self.assertLess(got, 50.0)

    def test_bigger_domains_read_bigger(self):
        """Monotonicity is what makes it usable as a guard at all."""
        sizes = []
        for w in (2, 4, 8, 16):
            n = 4 * w
            idx = np.arange(n) // w
            lab = ((idx[:, None] + idx[None, :]) % 2).astype(int)
            sizes.append(domain_diameter(lab))
        self.assertTrue(all(a < b for a, b in zip(sizes, sizes[1:])),
                        msg=str(sizes))


if __name__ == "__main__":
    unittest.main()
