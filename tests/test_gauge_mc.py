"""Tests for the Monte-Carlo lattice gauge sampler and confinement observables.

The decisive test validates the checkerboard Metropolis sampler against the
exactly-solvable 2-D SU(2) plaquette average; the rest cover observables and the
qualitative confinement signal.
"""

from __future__ import annotations

import collections
import unittest

import numpy as np

from project_genesis.gauge import is_unitary
from project_genesis.gauge_mc import (
    creutz_ratio,
    exact_plaquette_su2_2d,
    hot_start,
    mean_plaquette,
    metropolis_sweep,
    wilson_loop,
)


class TestObservables(unittest.TestCase):
    def test_wilson_1x1_equals_mean_plaquette(self) -> None:
        rng = np.random.default_rng(0)
        links = hot_start(rng, 2, (8, 8))
        self.assertAlmostEqual(wilson_loop(links, 1, 1), mean_plaquette(links), places=10)

    def test_mean_plaquette_identity_is_one(self) -> None:
        from project_genesis.gauge import identity_links
        links = identity_links(3, (6, 6))
        # identity_links has only one direction's worth shaped (ndim,*); build 2-dir.
        links = np.broadcast_to(np.eye(3, dtype=np.complex128), (2, 6, 6, 3, 3)).copy()
        self.assertAlmostEqual(mean_plaquette(links), 1.0, places=10)

    def test_creutz_ratio_positive_for_decaying_loops(self) -> None:
        # Synthetic pure area law W(r,t) = exp(-σ r t): Creutz ratio recovers σ.
        sigma = 0.3
        w = {(r, t): np.exp(-sigma * r * t) for r in range(1, 4) for t in range(1, 4)}
        self.assertAlmostEqual(creutz_ratio(w, 2, 2), sigma, places=10)


class TestSampler(unittest.TestCase):
    def test_sweep_preserves_su_n(self) -> None:
        rng = np.random.default_rng(0)
        links = hot_start(rng, 2, (8, 8))
        for _ in range(5):
            metropolis_sweep(links, 1.5, rng, step=0.4)
        for mu in range(links.shape[0]):
            self.assertTrue(is_unitary(links[mu]))
        np.testing.assert_allclose(np.linalg.det(links), 1.0, atol=1e-7)

    def test_plaquette_increases_with_beta(self) -> None:
        # Stronger coupling (larger β) => more ordered => larger ⟨P⟩.
        def equilibrated_plaquette(beta: float) -> float:
            rng = np.random.default_rng(3)
            links = hot_start(rng, 2, (12, 12))
            for _ in range(200):
                metropolis_sweep(links, beta, rng, step=0.4)
            return float(np.mean([
                (metropolis_sweep(links, beta, rng, step=0.4), mean_plaquette(links))[1]
                for _ in range(120)
            ]))
        self.assertGreater(equilibrated_plaquette(3.0), equilibrated_plaquette(0.7))

    def test_matches_exact_2d_su2_plaquette(self) -> None:
        # The decisive correctness gate: MC must reproduce I_2(β)/I_1(β).
        rng = np.random.default_rng(11)
        beta = 2.0
        links = hot_start(rng, 2, (16, 16))
        for _ in range(400):
            metropolis_sweep(links, beta, rng, step=0.4)
        samples = []
        for s in range(400):
            metropolis_sweep(links, beta, rng, step=0.4)
            if s % 4 == 0:
                samples.append(mean_plaquette(links))
        mc = float(np.mean(samples))
        self.assertAlmostEqual(mc, exact_plaquette_su2_2d(beta), delta=0.02)


class TestConfinement(unittest.TestCase):
    def test_2d_su2_shows_positive_string_tension(self) -> None:
        rng = np.random.default_rng(5)
        beta = 1.5
        links = hot_start(rng, 2, (16, 16))
        for _ in range(400):
            metropolis_sweep(links, beta, rng, step=0.4)
        loops = collections.defaultdict(list)
        plaq = []
        for s in range(600):
            metropolis_sweep(links, beta, rng, step=0.4)
            if s % 2 == 0:
                plaq.append(mean_plaquette(links))
                for r in (1, 2):
                    for t in (1, 2):
                        loops[(r, t)].append(wilson_loop(links, r, t))
        w = {k: float(np.mean(v)) for k, v in loops.items()}
        chi = creutz_ratio(w, 2, 2)
        sigma_exact = -np.log(float(np.mean(plaq)))
        self.assertGreater(chi, 0.0)
        # Creutz ratio should be near the exact 2-D σ = −log⟨P⟩.
        self.assertAlmostEqual(chi, sigma_exact, delta=0.4)


if __name__ == "__main__":
    unittest.main()
