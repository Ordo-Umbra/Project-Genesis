"""Tests for the three-component (Ψ∈ℂ³) sector field model.

These validate that the multi-phase model does what the scalar model cannot:
sustain three mutually-adjacent phases with genuine triple junctions, with the
S₃ relabelling symmetry the gauge picture expects.
"""

from __future__ import annotations

import json
import unittest

import numpy as np

from project_genesis.multiphase import (
    analyze_multiphase,
    count_triple_junctions,
    interface_mask,
    periodic_laplacian,
    sector_labels,
    step_multiphase,
)


def _coarsened_field(steps: int = 300, size: int = 64, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fields = rng.random((3, size, size)) * 0.1
    for _ in range(steps):
        fields = step_multiphase(fields, diffusion=1.0, gamma=1.5, dt=0.1)
    return fields


class TestLaplacian(unittest.TestCase):
    def test_constant_field_zero(self) -> None:
        f = np.full((6, 6), 0.4)
        self.assertTrue(np.allclose(periodic_laplacian(f), 0.0))

    def test_works_in_3d(self) -> None:
        f = np.zeros((4, 4, 4))
        f[1, 1, 1] = 1.0
        lap = periodic_laplacian(f)
        # Centre loses 6, each of 6 neighbours gains 1.
        self.assertAlmostEqual(lap[1, 1, 1], -6.0)
        self.assertAlmostEqual(lap[0, 1, 1], 1.0)


class TestStep(unittest.TestCase):
    def test_shape_preserved(self) -> None:
        f = np.random.default_rng(0).random((3, 16, 16))
        out = step_multiphase(f, dt=0.05)
        self.assertEqual(out.shape, f.shape)

    def test_rejects_bad_shape(self) -> None:
        with self.assertRaises(ValueError):
            step_multiphase(np.zeros(5))

    def test_drives_toward_single_dominant_phase(self) -> None:
        # A cell starting near a corner should sharpen toward it.
        f = np.zeros((3, 4, 4))
        f[0] = 0.6
        f[1] = 0.3
        f[2] = 0.3
        for _ in range(200):
            f = step_multiphase(f, diffusion=0.0, gamma=1.5, dt=0.05)
        self.assertGreater(f[0].mean(), f[1].mean())
        self.assertGreater(f[0].mean(), f[2].mean())


class TestLabelsAndInterfaces(unittest.TestCase):
    def test_sector_labels_argmax(self) -> None:
        f = np.zeros((3, 2, 2))
        f[2, 0, 0] = 1.0  # blue dominates here
        labels = sector_labels(f)
        self.assertEqual(labels[0, 0], 2)

    def test_interface_mask_flags_mixed_cells(self) -> None:
        f = np.zeros((3, 2, 2))
        f[:, 0, 0] = [0.5, 0.5, 0.0]   # top two tied -> interface
        f[:, 1, 1] = [1.0, 0.0, 0.0]   # clear dominant -> interior
        mask = interface_mask(f, margin=0.15)
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[1, 1])


class TestTripleJunctions(unittest.TestCase):
    def test_uniform_has_none(self) -> None:
        labels = np.ones((6, 6), dtype=int)
        self.assertEqual(count_triple_junctions(labels), 0)

    def test_detects_three_way_meeting(self) -> None:
        # Three sectors meeting around a point.
        labels = np.zeros((3, 3), dtype=int)
        labels[0, :] = 0
        labels[1, :] = 1
        labels[2, :] = 2
        self.assertGreater(count_triple_junctions(labels), 0)

    def test_s3_permutation_invariance(self) -> None:
        fields = _coarsened_field()
        labels = sector_labels(fields)
        j1 = count_triple_junctions(labels)
        # Relabel R->G->B->R; junction count must be unchanged.
        j2 = count_triple_junctions(sector_labels(fields[[1, 2, 0]]))
        self.assertEqual(j1, j2)
        self.assertGreater(j1, 0)


class TestAnalyze(unittest.TestCase):
    def test_three_phases_with_junctions_form(self) -> None:
        fields = _coarsened_field()
        report = analyze_multiphase(fields, beta=0.09)
        self.assertEqual(report["n_phases"], 3)
        self.assertGreater(report["triple_junctions"], 0)
        self.assertTrue(0.0 <= report["boundary_fraction"] <= 1.0)
        self.assertGreaterEqual(report["distinction"], 0.0)
        self.assertTrue(0.0 <= report["integration"] <= 1.0)

    def test_report_json_serializable(self) -> None:
        report = analyze_multiphase(_coarsened_field(steps=100), beta=0.09)
        json.loads(json.dumps(report))

    def test_works_in_3d(self) -> None:
        rng = np.random.default_rng(1)
        fields = rng.random((3, 24, 24, 24)) * 0.1
        for _ in range(80):
            fields = step_multiphase(fields, diffusion=1.0, gamma=1.5, dt=0.1)
        report = analyze_multiphase(fields)
        self.assertEqual(report["n_phases"], 3)
        # In 3-D triple "junctions" are lines, but the count should be positive.
        self.assertGreater(report["triple_junctions"], 0)


if __name__ == "__main__":
    unittest.main()
