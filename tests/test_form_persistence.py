"""Checks for the form-persistence texture guard.

P1 of `n3_form_persistence` claims that intermediate noise sustains *ordered*
light structure "without collapsing to single-domain or pure texture".  The
whole content of that claim sits in whatever separates structure from noise, so
that predicate is what these checks pin.

The guard it used to carry — `n_phases`, the count of distinct sector labels
present anywhere on the lattice — cannot make the separation at all: texture
carries every label too.  The regression test below is the one that matters:
it constructs a field that is *literally* texture and asserts that `n_phases`
passes it while the current guard does not.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from project_genesis.multiphase import (  # noqa: E402
    domain_diameter,
    sector_labels,
    step_multiphase_conserved,
)
from n3_form_persistence import (  # noqa: E402
    MIN_DOMAIN_DIAMETER,
    light_form_density,
    pick_size_scan_noise,
    tessellated,
)

PALETTE = 3


def _texture(size: int = 16, ndim: int = 3, seed: int = 0) -> np.ndarray:
    """Uncorrelated labels: every site independent of its neighbours."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, PALETTE, size=(size,) * ndim)


def _relaxed(size: int = 16, ndim: int = 3, steps: int = 140,
             seed: int = 0) -> np.ndarray:
    """A conserved multiphase field, coarsened into genuine domains."""
    rng = np.random.default_rng(seed)
    fields = 0.1 * rng.standard_normal((PALETTE,) + (size,) * ndim)
    fields = fields / (np.sqrt((fields * fields).sum(0, keepdims=True) + 1e-12) * 0.7)
    for _ in range(steps):
        fields, _ = step_multiphase_conserved(
            fields, None, diffusion=1.0, gamma=1.5, dt=0.1
        )
    return sector_labels(fields)


class TestTextureGuard(unittest.TestCase):

    def test_pure_noise_reads_unit_diameter(self):
        # The property that makes `domain_diameter` usable as a guard: noise
        # reads 1.0 — every site borders a different sector, so there is no
        # domain to have a width.
        for ndim in (2, 3, 4):
            size = 16 if ndim < 4 else 10
            self.assertAlmostEqual(
                domain_diameter(_texture(size=size, ndim=ndim)), 1.0, places=6,
                msg=f"noise should read unit width in {ndim}-D",
            )

    def test_relaxed_field_is_tessellated(self):
        labels = _relaxed()
        self.assertGreaterEqual(domain_diameter(labels), MIN_DOMAIN_DIAMETER)
        self.assertTrue(
            tessellated(len(np.unique(labels)), domain_diameter(labels), PALETTE)
        )

    def test_texture_is_rejected(self):
        labels = _texture()
        self.assertFalse(
            tessellated(len(np.unique(labels)), domain_diameter(labels), PALETTE)
        )

    def test_n_phases_alone_cannot_separate_them(self):
        """The regression this guard exists for.

        Both fields carry all three labels, so the old predicate — `n_phases >=
        palette - 1.1` — admits texture exactly as readily as structure.  If
        this assertion ever fails, `n_phases` has become informative and the
        argument for the current guard needs re-examining; it has not.
        """
        noise, structure = _texture(), _relaxed()
        self.assertEqual(len(np.unique(noise)), PALETTE)
        self.assertEqual(len(np.unique(structure)), PALETTE)
        # ...and the guard that does separate them, on the same two fields.
        self.assertLess(domain_diameter(noise), MIN_DOMAIN_DIAMETER)
        self.assertGreaterEqual(domain_diameter(structure), MIN_DOMAIN_DIAMETER)

    def test_single_domain_is_rejected_by_the_other_half(self):
        """Both halves of the guard are load-bearing.

        A collapsed field has no walls, so its diameter is `inf` and the width
        test passes vacuously.  `n_phases` is what catches it — which is why
        the guard is a conjunction rather than a replacement.
        """
        labels = np.zeros((16, 16, 16), dtype=int)
        self.assertEqual(domain_diameter(labels), float("inf"))
        self.assertFalse(tessellated(1.0, float("inf"), PALETTE))

    def test_texture_inflates_the_light_form_count(self):
        """Why this matters for P1 rather than being a tidiness point.

        The measured quantity is the light-form density, and texture does not
        merely fail to be structure — it reads *higher* than structure does,
        so an unguarded P1 is preferentially passed by the fields that have
        least to do with the claim.
        """
        self.assertGreater(
            light_form_density(_texture()), light_form_density(_relaxed())
        )


class TestSizeScanSelection(unittest.TestCase):
    """Which noise P2's size scan runs at.

    The selection reads the same measured quantity the guard exists to
    qualify, so it inherits the same failure: texture scores highest on
    light-form density, and an unguarded `max` therefore picks it every time.
    """

    # a sweep where the densest point is texture and the runner-up is not
    MIDS = [
        {"noise": 0.03, "late_mean": 0.045, "n_phases_mean": 3.0, "diameter_mean": 4.9},
        {"noise": 0.06, "late_mean": 0.084, "n_phases_mean": 3.0, "diameter_mean": 4.3},
        {"noise": 0.12, "late_mean": 0.525, "n_phases_mean": 3.0, "diameter_mean": 1.4},
    ]

    def test_picks_the_densest_tessellated_point(self):
        self.assertEqual(pick_size_scan_noise(self.MIDS, PALETTE), 0.06)

    def test_does_not_pick_the_densest_point_overall(self):
        # what the old selection did: max on late_mean, guarded by n_phases,
        # which is 3.0 on all three rows and so decides nothing.
        blind = max(self.MIDS, key=lambda r: r["late_mean"])["noise"]
        self.assertEqual(blind, 0.12)
        self.assertNotEqual(pick_size_scan_noise(self.MIDS, PALETTE), blind)

    def test_falls_back_when_nothing_is_tessellated(self):
        texture_only = [
            dict(r, diameter_mean=1.0) for r in self.MIDS
        ]
        # no tessellated point to prefer, so it returns the raw maximum and
        # leaves the refusal to P2's own guard rather than inventing one here.
        self.assertEqual(pick_size_scan_noise(texture_only, PALETTE), 0.12)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
