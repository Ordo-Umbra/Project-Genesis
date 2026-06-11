"""Tests for junction-resolving (conserved) dynamics and the topological
neutrality selector.

These cover the final step of the N⋆=3 narrowing: volume-conserving dynamics
that keep triple junctions alive, and a full-palette junction measure that is
non-collinear with ΔC and selects three.
"""

from __future__ import annotations

import unittest

import numpy as np

from project_genesis.multiphase import (
    count_triple_junctions,
    full_palette_junction_density,
    sector_labels,
    step_multiphase,
    step_multiphase_conserved,
    topological_s_functional,
)


def _init(p: int = 3, size: int = 48, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).random((p, size, size)) * 0.1


class TestConservedDynamics(unittest.TestCase):
    def test_conserves_component_totals(self) -> None:
        f = _init(size=32)
        totals0 = f.sum(axis=(1, 2))
        for _ in range(100):
            f, _ = step_multiphase_conserved(f, dt=0.1)
        np.testing.assert_allclose(f.sum(axis=(1, 2)), totals0, rtol=1e-6, atol=1e-6)

    def test_rejects_bad_kappa_shape(self) -> None:
        with self.assertRaises(ValueError):
            step_multiphase_conserved(_init(size=16), np.ones((8, 8)))

    def test_kappa_returned_only_when_supplied(self) -> None:
        f = _init(size=16)
        _, k_none = step_multiphase_conserved(f, dt=0.1)
        self.assertIsNone(k_none)
        _, k = step_multiphase_conserved(f, np.ones((16, 16)), dt=0.1)
        self.assertIsNotNone(k)

    def test_junctions_persist_unlike_plain_coarsening(self) -> None:
        # Conserved dynamics keep all phases, so triple junctions survive where
        # plain Allen–Cahn would coarsen them away.
        init = _init(p=3, size=64, seed=3)
        conserved = init.copy()
        for _ in range(500):
            conserved, _ = step_multiphase_conserved(conserved, dt=0.1)
        self.assertGreater(count_triple_junctions(sector_labels(conserved)), 0)
        # All three phases still present (none eliminated).
        self.assertEqual(len(np.unique(sector_labels(conserved))), 3)

    def test_deterministic(self) -> None:
        f1 = _init(size=24)
        f2 = f1.copy()
        for _ in range(20):
            f1, _ = step_multiphase_conserved(f1, dt=0.1)
            f2, _ = step_multiphase_conserved(f2, dt=0.1)
        self.assertTrue(np.array_equal(f1, f2))


class TestFullPaletteJunctions(unittest.TestCase):
    def test_three_colours_at_a_point_qualifies(self) -> None:
        labels = np.zeros((3, 3), dtype=int)
        labels[0, :] = 0
        labels[1, :] = 1
        labels[2, :] = 2
        self.assertGreater(full_palette_junction_density(labels, 3), 0.0)

    def test_two_palette_has_no_full_junctions(self) -> None:
        # A 2-colour interface forms no triple junction, so no "complete" one.
        labels = np.zeros((8, 8), dtype=int)
        labels[4:, :] = 1
        self.assertEqual(full_palette_junction_density(labels, 2), 0.0)

    def test_incomplete_palette_at_triple_junction(self) -> None:
        # Three colours meet, but the palette is four -> not full-palette.
        labels = np.zeros((3, 3), dtype=int)
        labels[0, :] = 0
        labels[1, :] = 1
        labels[2, :] = 2
        self.assertGreater(full_palette_junction_density(labels, 3), 0.0)
        self.assertEqual(full_palette_junction_density(labels, 4), 0.0)


class TestSelectsThree(unittest.TestCase):
    def test_neutrality_peaks_at_three_after_evolution(self) -> None:
        neut = {}
        for P in (2, 3, 4, 5):
            f = _init(p=P, size=56, seed=7)
            for _ in range(400):
                f, _ = step_multiphase_conserved(f, dt=0.1)
            neut[P] = full_palette_junction_density(sector_labels(f), P)
        self.assertGreater(neut[3], neut[2])
        self.assertGreater(neut[3], neut[4])
        self.assertGreater(neut[3], neut[5])

    def test_topological_s_interior_optimum_at_three(self) -> None:
        rows = {}
        for P in (2, 3, 4, 5):
            f = _init(p=P, size=56, seed=7)
            for _ in range(400):
                f, _ = step_multiphase_conserved(f, dt=0.1)
            rows[P] = topological_s_functional(f, weight=0.2)
        s = {P: rows[P]["s_increment"] for P in rows}
        self.assertEqual(max(s, key=lambda pp: s[pp]), 3)

    def test_topological_s_keys(self) -> None:
        f = _init(p=3, size=24)
        s = topological_s_functional(f, weight=0.1)
        for key in ("delta_c", "kappa", "neutrality", "s_increment"):
            self.assertIn(key, s)


class TestSelectsThreeIn3D(unittest.TestCase):
    """Three survives in 3-D: triple junctions are abundant *lines*, while the
    full palette for P≥4 needs sparse quadruple *points* — so a 3-colour palette
    still maximizes the full-palette junction density."""

    @staticmethod
    def _evolve3d(P: int, *, size: int = 20, steps: int = 150, seed: int = 7) -> np.ndarray:
        f = np.random.default_rng(seed).random((P, size, size, size)) * 0.1
        for _ in range(steps):
            f, _ = step_multiphase_conserved(f, dt=0.1)
        return f

    def test_neutrality_peaks_at_three(self) -> None:
        neut = {}
        for P in (2, 3, 4, 5):
            neut[P] = full_palette_junction_density(sector_labels(self._evolve3d(P)), P)
        self.assertGreater(neut[3], neut[2])
        self.assertGreater(neut[3], neut[4])
        self.assertGreater(neut[3], neut[5])

    def test_topological_s_optimum_at_three(self) -> None:
        s = {P: topological_s_functional(self._evolve3d(P), weight=0.2)["s_increment"]
             for P in (2, 3, 4, 5)}
        self.assertEqual(max(s, key=lambda pp: s[pp]), 3)


if __name__ == "__main__":
    unittest.main()
