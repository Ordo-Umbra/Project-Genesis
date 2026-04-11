"""Tests for the URP physics extensions and visualization.

Covers:
- Coherence potential V(x,t) via Poisson solver (∇²V = ρ)
- Gradient dot product ∇V·∇φ
- Nonlocal integration functional I[φ] via correlation kernels
- Full URP evolution kernel (v2) with all terms
- Engine integration with new config options
- Matplotlib visualization output
- CLI flags for new features
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from project_genesis import EngineConfig, GenesisEngine
from project_genesis.numba_kernels import (
    correlation_kernel_3d,
    gradient_dot_product_3d,
    gradient_squared_3d,
    jit_step,
    jit_step_v2,
    laplacian_3d,
    solve_poisson_jacobi,
)


# ------------------------------------------------------------------ #
# Poisson solver tests
# ------------------------------------------------------------------ #


class PoissonSolverTests(unittest.TestCase):
    def test_zero_source_yields_zero_potential(self) -> None:
        rho = np.zeros((8, 8, 8), dtype=np.float64)
        out = np.zeros_like(rho)
        solve_poisson_jacobi(rho, out, iterations=30)
        np.testing.assert_allclose(out, 0.0, atol=1e-15)

    def test_constant_source_converges(self) -> None:
        rho = np.ones((8, 8, 8), dtype=np.float64)
        out = np.zeros_like(rho)
        solve_poisson_jacobi(rho, out, iterations=100)
        self.assertTrue(np.isfinite(out).all())
        # For a periodic Poisson solve with constant source, V converges
        # to a constant.
        self.assertAlmostEqual(float(np.std(out)), 0.0, places=3)

    def test_potential_is_smooth(self) -> None:
        rng = np.random.default_rng(42)
        rho = rng.random((8, 8, 8)).astype(np.float64)
        out = np.zeros_like(rho)
        solve_poisson_jacobi(rho, out, iterations=30)
        self.assertLess(float(np.std(out)), float(np.std(rho)) * 5)
        self.assertTrue(np.isfinite(out).all())


# ------------------------------------------------------------------ #
# Gradient dot product tests
# ------------------------------------------------------------------ #


class GradientDotProductTests(unittest.TestCase):
    def test_dot_product_of_identical_fields_equals_gradient_squared(self) -> None:
        rng = np.random.default_rng(7)
        field = rng.random((8, 8, 8)).astype(np.float64)
        gsq = np.empty_like(field)
        dot = np.empty_like(field)
        gradient_squared_3d(field, gsq)
        gradient_dot_product_3d(field, field, dot)
        np.testing.assert_allclose(dot, gsq, atol=1e-12)

    def test_dot_product_is_symmetric(self) -> None:
        rng = np.random.default_rng(11)
        a = rng.random((8, 8, 8)).astype(np.float64)
        b = rng.random((8, 8, 8)).astype(np.float64)
        dot_ab = np.empty_like(a)
        dot_ba = np.empty_like(a)
        gradient_dot_product_3d(a, b, dot_ab)
        gradient_dot_product_3d(b, a, dot_ba)
        np.testing.assert_allclose(dot_ab, dot_ba, atol=1e-12)

    def test_constant_field_gives_zero(self) -> None:
        field = np.ones((8, 8, 8), dtype=np.float64)
        other = np.random.default_rng(0).random((8, 8, 8)).astype(np.float64)
        dot = np.empty_like(field)
        gradient_dot_product_3d(field, other, dot)
        np.testing.assert_allclose(dot, 0.0, atol=1e-12)


# ------------------------------------------------------------------ #
# Correlation kernel tests
# ------------------------------------------------------------------ #


class CorrelationKernelTests(unittest.TestCase):
    def test_constant_field_output(self) -> None:
        field = np.ones((8, 8, 8), dtype=np.float64) * 0.5
        out = np.empty_like(field)
        correlation_kernel_3d(field, out, radius=1, decay=1.0)
        self.assertAlmostEqual(float(np.std(out)), 0.0, places=10)
        self.assertGreater(float(np.mean(out)), 0.0)

    def test_output_increases_with_field_magnitude(self) -> None:
        low = np.ones((8, 8, 8), dtype=np.float64) * 0.1
        high = np.ones((8, 8, 8), dtype=np.float64) * 0.9
        out_low = np.empty_like(low)
        out_high = np.empty_like(high)
        correlation_kernel_3d(low, out_low, radius=2, decay=1.0)
        correlation_kernel_3d(high, out_high, radius=2, decay=1.0)
        self.assertGreater(float(np.mean(out_high)), float(np.mean(out_low)))

    def test_output_is_finite(self) -> None:
        rng = np.random.default_rng(42)
        field = rng.random((8, 8, 8)).astype(np.float64)
        out = np.empty_like(field)
        correlation_kernel_3d(field, out, radius=2, decay=0.5)
        self.assertTrue(np.isfinite(out).all())


# ------------------------------------------------------------------ #
# jit_step_v2 integration tests
# ------------------------------------------------------------------ #


class JitStepV2Tests(unittest.TestCase):
    def test_v2_produces_finite_field(self) -> None:
        rng = np.random.default_rng(42)
        field = rng.random((8, 8, 8)).astype(np.float64)
        new_field, lap, gsq = jit_step_v2(field, 0.09, 0.22, 0.01)
        self.assertTrue(np.isfinite(new_field).all())
        self.assertTrue(np.isfinite(lap).all())
        self.assertTrue(np.isfinite(gsq).all())

    def test_v2_differs_from_v1(self) -> None:
        rng = np.random.default_rng(42)
        field = rng.random((8, 8, 8)).astype(np.float64)
        v1, _, _ = jit_step(field, 0.09, 0.22, 0.01)
        v2, _, _ = jit_step_v2(field, 0.09, 0.22, 0.01)
        self.assertFalse(np.allclose(v1, v2))

    def test_v2_repeatable_with_same_input(self) -> None:
        rng = np.random.default_rng(42)
        field = rng.random((8, 8, 8)).astype(np.float64)
        r1, _, _ = jit_step_v2(field, 0.09, 0.22, 0.01)
        r2, _, _ = jit_step_v2(field, 0.09, 0.22, 0.01)
        np.testing.assert_allclose(r1, r2)


# ------------------------------------------------------------------ #
# Engine with new config options
# ------------------------------------------------------------------ #


class EngineCoherencePotentialTests(unittest.TestCase):
    def test_coherence_potential_produces_finite_field(self) -> None:
        config = EngineConfig(
            chunk_size=8,
            seed=42,
            default_steps=5,
            default_dt=0.01,
            use_coherence_potential=True,
        )
        engine = GenesisEngine(config=config)
        engine.evolve_field()
        self.assertTrue(np.isfinite(engine.field).all())

    def test_coherence_potential_changes_outcome(self) -> None:
        base_config = EngineConfig(chunk_size=8, seed=42, default_steps=5, default_dt=0.01)
        cp_config = EngineConfig(
            chunk_size=8, seed=42, default_steps=5, default_dt=0.01,
            use_coherence_potential=True,
        )
        base = GenesisEngine(config=base_config)
        cp = GenesisEngine(config=cp_config)
        base.evolve_field()
        cp.evolve_field()
        self.assertFalse(np.allclose(base.field, cp.field))


class EngineIntegrationFunctionalTests(unittest.TestCase):
    def test_integration_functional_produces_finite_field(self) -> None:
        config = EngineConfig(
            chunk_size=8,
            seed=42,
            default_steps=5,
            default_dt=0.01,
            use_integration_functional=True,
        )
        engine = GenesisEngine(config=config)
        engine.evolve_field()
        self.assertTrue(np.isfinite(engine.field).all())

    def test_both_features_together(self) -> None:
        config = EngineConfig(
            chunk_size=8,
            seed=42,
            default_steps=5,
            default_dt=0.01,
            use_coherence_potential=True,
            use_integration_functional=True,
            agent_count=2,
            agent_goal="s_functional",
        )
        engine = GenesisEngine(config=config)
        engine.evolve_field()
        self.assertTrue(np.isfinite(engine.field).all())
        voxels = set(np.unique(engine.quantize_to_voxels()).tolist())
        self.assertGreaterEqual(len(voxels), 2)

    def test_save_load_preserves_new_config(self) -> None:
        config = EngineConfig(
            chunk_size=8,
            seed=42,
            default_steps=3,
            default_dt=0.01,
            use_coherence_potential=True,
            poisson_iterations=20,
            use_integration_functional=True,
            integration_radius=3,
            integration_decay=0.5,
            integration_weight=0.02,
        )
        engine = GenesisEngine(config=config)
        engine.evolve_field(record_every=3)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "snap.npz"
            engine.save(path)
            loaded = GenesisEngine.load(path)

        np.testing.assert_allclose(engine.field, loaded.field)
        self.assertEqual(loaded.config.use_coherence_potential, True)
        self.assertEqual(loaded.config.poisson_iterations, 20)
        self.assertEqual(loaded.config.use_integration_functional, True)
        self.assertEqual(loaded.config.integration_radius, 3)
        self.assertAlmostEqual(loaded.config.integration_decay, 0.5)
        self.assertAlmostEqual(loaded.config.integration_weight, 0.02)


# ------------------------------------------------------------------ #
# Config validation tests
# ------------------------------------------------------------------ #


class NewConfigValidationTests(unittest.TestCase):
    def test_invalid_poisson_iterations(self) -> None:
        with self.assertRaises(ValueError):
            EngineConfig(poisson_iterations=0)

    def test_invalid_integration_radius(self) -> None:
        with self.assertRaises(ValueError):
            EngineConfig(integration_radius=0)

    def test_invalid_integration_decay(self) -> None:
        with self.assertRaises(ValueError):
            EngineConfig(integration_decay=-0.1)

    def test_invalid_integration_weight(self) -> None:
        with self.assertRaises(ValueError):
            EngineConfig(integration_weight=-1.0)


# ------------------------------------------------------------------ #
# Visualization tests
# ------------------------------------------------------------------ #


class VisualizationTests(unittest.TestCase):
    def test_render_voxels_3d(self) -> None:
        from project_genesis.visualize import render_voxels_3d

        rng = np.random.default_rng(42)
        voxels = rng.integers(0, 5, size=(8, 8, 8))
        fig = render_voxels_3d(voxels)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_render_field_slices(self) -> None:
        from project_genesis.visualize import render_field_slices

        field = np.random.default_rng(42).random((8, 8, 8))
        fig = render_field_slices(field)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_s_history(self) -> None:
        from project_genesis.visualize import plot_s_history

        history = [
            {"step": i, "delta_c": 0.01 * i, "delta_i": 0.005, "kappa": 0.9, "s_increment": 0.015}
            for i in range(10)
        ]
        fig = plot_s_history(history)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_save_visualization_writes_files(self) -> None:
        from project_genesis.visualize import save_visualization

        engine = GenesisEngine(config=EngineConfig(chunk_size=8, seed=42, default_steps=3, default_dt=0.01))
        engine.evolve_field(record_every=1)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "viz"
            paths = save_visualization(out, engine.quantize_to_voxels(), engine.field, engine.history)
            self.assertEqual(len(paths), 3)
            for p in paths:
                self.assertTrue(p.exists())
                self.assertGreater(p.stat().st_size, 0)


# ------------------------------------------------------------------ #
# CLI tests for new flags
# ------------------------------------------------------------------ #


class CLINewFlagsTests(unittest.TestCase):
    def test_cli_with_coherence_potential(self) -> None:
        from genesis_engine import main

        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td) / "cli-cp"
            argv = [
                "genesis_engine.py",
                "--chunk-size", "8",
                "--steps", "3",
                "--dt", "0.01",
                "--seed", "4",
                "--record-every", "1",
                "--coherence-potential",
                "--output-dir", str(output_dir),
            ]
            with patch("sys.argv", argv):
                main()
            self.assertTrue((output_dir / "engine_snapshot.npz").exists())
            self.assertTrue((output_dir / "run_summary.json").exists())

    def test_cli_with_visualize(self) -> None:
        from genesis_engine import main

        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td) / "cli-viz"
            argv = [
                "genesis_engine.py",
                "--chunk-size", "8",
                "--steps", "3",
                "--dt", "0.01",
                "--seed", "4",
                "--record-every", "1",
                "--visualize",
                "--output-dir", str(output_dir),
            ]
            with patch("sys.argv", argv):
                main()
            self.assertTrue((output_dir / "voxel_3d.png").exists())
            self.assertTrue((output_dir / "field_slices.png").exists())
            self.assertTrue((output_dir / "s_history.png").exists())


if __name__ == "__main__":
    unittest.main()
