"""Tests for the control that retracted `d+1`.

These pin a *negative* result, which needs saying because it inverts the usual
purpose of a test. Normally a test guards a finding against regression. Here the
finding is that the measure returns `d+1` no matter what it is shown, so these
guard the **retraction** against being quietly undone — if someone later restores
`d+1` as a claim about the field, one of these fails.

The three arms are not interchangeable and the distinction is the whole argument:

- **noise** demonstrates the construction argument at its starkest, but *fails the
  texture guard*, so on its own it does not refute a properly guarded claim;
- **Voronoi** passes every check the repository applies and still returns `d+1`
  with no dynamics — this is the arm that settles it;
- **relaxed** is the published field, included so the three are scored by
  identical code and the comparison cannot drift.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from project_genesis.junction_scale import full_palette_density  # noqa: E402
from project_genesis.multiphase import domain_diameter           # noqa: E402
from n3_junction_null import (                                   # noqa: E402
    noise_labels, score, voronoi_labels,
)


class TestTheArgmaxIsForcedByConstruction(unittest.TestCase):
    """`P < d+1` is zero by definition, not by measurement."""

    def test_a_palette_smaller_than_d_plus_1_cannot_score_at_all(self):
        """You cannot draw `d+1` distinct colours from fewer than `d+1`. No
        field, however arranged, can make this non-zero — which is why every
        such entry in the published tables carried no information."""
        rng = np.random.default_rng(0)
        for d, n in ((2, 24), (3, 12)):
            for P in range(2, d + 1):
                for lab in (noise_labels(n, d, P, rng),
                            voronoi_labels(n, d, 8, P, rng)):
                    self.assertEqual(score(lab, P, d), 0.0,
                                     msg=f"d={d} P={P}")

    def test_the_condition_is_monotone_harder_as_the_palette_grows(self):
        """Above `d+1` every colour must fit in one 3^d window, so the density
        falls. Zero below, falling above: the peak sits at `d+1` before any
        field exists."""
        rng = np.random.default_rng(1)
        d, n = 2, 64
        vals = [np.mean([score(noise_labels(n, d, P, rng), P, d)
                         for _ in range(3)]) for P in (3, 4, 5, 6)]
        self.assertTrue(all(a > b for a, b in zip(vals, vals[1:])),
                        msg=str(vals))


class TestVoronoiIsTheDecisiveControl(unittest.TestCase):
    """No dynamics, passes every guard, same answer."""

    def test_a_voronoi_diagram_selects_d_plus_1(self):
        rng = np.random.default_rng(17)
        for d, n, seeds in ((1, 2000, 90), (2, 64, 14)):
            dens = {P: float(np.mean([score(voronoi_labels(n, d, seeds, P, rng),
                                            P, d) for _ in range(3)]))
                    for P in (2, 3, 4, 5)}
            best = max(dens, key=lambda k: dens[k])
            self.assertEqual(best, d + 1, msg=f"d={d} {dens}")

    def test_a_voronoi_diagram_passes_the_texture_guard(self):
        """The reason this arm is decisive rather than merely suggestive: it
        clears the guard that (correctly) rejects noise."""
        rng = np.random.default_rng(3)
        lab = voronoi_labels(72, 2, 14, 3, rng)
        self.assertGreater(domain_diameter(lab), 5.0)


class TestNoiseIsStarkerButGuarded(unittest.TestCase):

    def test_noise_also_selects_d_plus_1(self):
        rng = np.random.default_rng(5)
        for d, n in ((2, 64), (3, 20)):
            dens = {P: float(np.mean([score(noise_labels(n, d, P, rng), P, d)
                                      for _ in range(3)]))
                    for P in (2, 3, 4, 5)}
            self.assertEqual(max(dens, key=lambda k: dens[k]), d + 1,
                             msg=f"d={d} {dens}")

    def test_noise_scores_far_higher_than_any_tessellation(self):
        """Two orders of magnitude. A measure whose value is maximised by
        structurelessness is not measuring structure."""
        rng = np.random.default_rng(7)
        n = float(np.mean([score(noise_labels(72, 2, 3, rng), 3, 2)
                           for _ in range(3)]))
        v = float(np.mean([score(voronoi_labels(72, 2, 14, 3, rng), 3, 2)
                           for _ in range(3)]))
        self.assertGreater(n, 50 * v, msg=f"noise {n}, voronoi {v}")

    def test_but_noise_fails_the_texture_guard(self):
        """Which is why noise alone does not refute a guarded claim, and why
        the Voronoi arm is the one that carries the retraction.

        Checked against the guard floor the experiment actually applies (5.0),
        not an invented number: noise reads 1.0-2.1 depending on palette size,
        a Voronoi reads 11-29, and the floor sits between them with room."""
        rng = np.random.default_rng(9)
        for P in (2, 3, 5):
            self.assertLess(domain_diameter(noise_labels(72, 2, P, rng)), 5.0,
                            msg=f"P={P}")


class TestTheMeasureStillDistinguishesSomething(unittest.TestCase):
    """The retraction is bounded: the instrument is not worthless."""

    def test_the_texture_guard_separates_tiling_from_noise_by_an_order(self):
        rng = np.random.default_rng(11)
        noise = domain_diameter(noise_labels(72, 2, 3, rng))
        vor = domain_diameter(voronoi_labels(72, 2, 14, 3, rng))
        self.assertGreater(vor, 10 * noise)

    def test_the_published_measure_is_unmodified_by_this_control(self):
        """The control has to score with the repo's own function or it proves
        nothing about the repo's own claim."""
        rng = np.random.default_rng(13)
        lab = voronoi_labels(48, 2, 10, 3, rng)
        self.assertAlmostEqual(
            score(lab, 3, 2),
            full_palette_density(lab, 3, radius=1, min_valence=3), places=15)


if __name__ == "__main__":
    unittest.main()
