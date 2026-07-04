"""Fast checks for experiments/n3_thermal_selection.py.

Exercises the experiment's machinery at CI scale: the sector-network
builder, the gauge-dressing measurement (retention bounded and increasing
with the matter coupling), and the dressed-selection assembly.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_thermal_selection import (  # noqa: E402
    build_sector_network,
    dressed_selection_table,
    measure_gauge_dressing,
)


def _tiny_network(P, seed=7, steps=300):
    return build_sector_network(
        P, size=16, steps=steps, gamma=1.5, diffusion=1.0, dt=0.1,
        seed=seed, beta=0.09,
    )


class TestSectorNetwork(unittest.TestCase):

    def test_structure_and_neutrality_selects_three(self):
        nets = {P: _tiny_network(P) for P in (2, 3, 4)}
        for P, net in nets.items():
            self.assertEqual(net["psi"].shape, (16, 16, P))
            self.assertGreater(net["delta_c"], 0.0)
            self.assertTrue(0.0 < net["kappa"] <= 1.0)
            # ψ rows are normalised
            norms = np.linalg.norm(net["psi"], axis=-1)
            self.assertTrue(np.allclose(norms, 1.0, atol=1e-12))
        # the 2-D structural result: full-palette junctions exist only at P=3
        self.assertEqual(nets[2]["neutrality"], 0.0)
        self.assertGreater(nets[3]["neutrality"], 0.0)
        self.assertEqual(nets[4]["neutrality"], 0.0)


class TestGaugeDressing(unittest.TestCase):

    def test_retention_bounded_and_grows_with_coupling(self):
        net = _tiny_network(3)
        results = []
        for g_m in (0.5, 4.0):
            d = measure_gauge_dressing(
                net["psi"], net["walls"], junctions=net["junctions"],
                beta_g=3.0, matter_coupling=g_m,
                n_therm=25, n_meas=20, n_skip=2, bin_size=5, seed=42,
            )
            self.assertTrue(-1.5 < d["retention"] < 1.5)
            self.assertFalse(math.isnan(d["retention"]))
            results.append(d["retention"])
        self.assertGreater(
            results[1], results[0] + 0.1,
            "retention should grow substantially with the matter coupling",
        )

    def test_no_curvature_localization_in_equilibrium(self):
        """Quenched matter constraints are integrable, so junction/wall
        curvature ratios stay ≈ 1 (the recorded negative verdict)."""
        net = _tiny_network(3)
        d = measure_gauge_dressing(
            net["psi"], net["walls"], junctions=net["junctions"],
            beta_g=5.0, matter_coupling=4.0,
            n_therm=30, n_meas=25, n_skip=2, bin_size=5, seed=43,
        )
        self.assertLess(abs(d["wall_curvature_ratio"] - 1.0), 0.15)
        if not math.isnan(d["junction_curvature_ratio"]):
            self.assertLess(abs(d["junction_curvature_ratio"] - 1.0), 0.25)


class TestSelectionAssembly(unittest.TestCase):

    def test_argmax_follows_retention_times_neutrality(self):
        networks = {
            2: {"delta_c": 0.0075, "kappa": 0.92, "neutrality": 0.0},
            3: {"delta_c": 0.0078, "kappa": 0.92, "neutrality": 0.012},
            4: {"delta_c": 0.0080, "kappa": 0.92, "neutrality": 0.0},
        }
        dressing = {
            P: [
                {"beta_g": 3.0, "matter_coupling": 0.5, "retention": 0.01},
                {"beta_g": 3.0, "matter_coupling": 4.0, "retention": 0.7},
            ]
            for P in networks
        }
        rows = dressed_selection_table(networks, dressing, [0.1])
        by_g = {r["matter_coupling"]: r["argmax"] for r in rows}
        # weak coupling: neutrality bonus (0.92·0.1·0.012·0.01 ≈ 1e-5) cannot
        # beat P=4's ΔC edge (2e-4) → washout; strong coupling: 3 wins
        self.assertEqual(by_g[0.5], 4)
        self.assertEqual(by_g[4.0], 3)


if __name__ == "__main__":
    unittest.main()
