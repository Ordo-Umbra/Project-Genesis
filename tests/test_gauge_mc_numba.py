"""Equivalence tests: Numba MC kernels vs the pure-Python reference path.

The JIT kernels in ``project_genesis.gauge_mc_kernels`` must implement
*exactly* the same update as the reference implementations in
``project_genesis.gauge_mc``: same staple convention, same SU(2) heat-bath
sampler, same Cabibbo–Marinari scheme, same random-draw order.  Numba's
``np.random.Generator`` support draws from the same bit stream as NumPy, so
most of these checks can demand near-bit-level agreement.

Covers:
- staple sum: kernel vs reference on random SU(2)/SU(3) configs
- a0 sampler: bit-identical draw sequences for the same seed
- one full heat-bath sweep: same seed → same links (float tolerance)
- overrelaxation: deterministic equality + action preservation via kernels
- Wilson / Polyakov loops: kernel vs reference equality
- unitarity after JIT sweeps (SU(2) and SU(3))
- ensemble agreement of the JIT chain with the exact SU(2) Bessel benchmark
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import i1, iv

from project_genesis.gauge import (
    random_unitary,
    wilson_action,
    is_unitary,
)
from project_genesis import gauge_mc as gm

try:
    from project_genesis import gauge_mc_kernels as nbk
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    nbk = None
    HAVE_NUMBA = False


@unittest.skipUnless(HAVE_NUMBA, "numba not available")
class TestStapleEquivalence(unittest.TestCase):

    def _check(self, n, shape):
        rng = np.random.default_rng(5)
        links = random_unitary(rng, n, shape, special=True)
        flat, fwd, bwd, spatial = nbk.flatten_links(links)
        staple = np.empty((n, n), dtype=np.complex128)
        t1 = np.empty((n, n), dtype=np.complex128)
        t2 = np.empty((n, n), dtype=np.complex128)
        nsites = flat.shape[1]
        for mu in range(shape[0]):
            for s in range(nsites):
                site = np.unravel_index(s, spatial)
                nbk._staple_sum_flat(flat, fwd, bwd, mu, s, staple, t1, t2)
                ref = gm._staple_sum(links, mu, tuple(int(c) for c in site))
                self.assertLess(
                    float(np.max(np.abs(staple - ref))), 1e-12,
                    f"staple mismatch at mu={mu}, site={site}",
                )

    def test_su2_2d(self):
        self._check(2, (2, 5, 5))

    def test_su3_3d(self):
        self._check(3, (3, 3, 3, 3))


@unittest.skipUnless(HAVE_NUMBA, "numba not available")
class TestSamplerStreamEquivalence(unittest.TestCase):

    def test_a0_sampler_bit_identical(self):
        """Same seed → identical a0 sequences across both implementations."""
        r_py = np.random.default_rng(9)
        r_nb = np.random.default_rng(9)
        cs = [0.0, 0.1, 0.5, 1.0, 1.9, 2.1, 3.0, 8.0, 25.0] * 20
        for c in cs:
            a_py = gm._sample_su2_a0(c, r_py)
            a_nb = float(nbk._sample_su2_a0_nb(c, r_nb))
            self.assertEqual(a_py, a_nb, f"a0 sampler diverged at c={c}")


@unittest.skipUnless(HAVE_NUMBA, "numba not available")
class TestSweepEquivalence(unittest.TestCase):

    def test_heatbath_single_sweep_su2(self):
        rng = np.random.default_rng(5)
        links = random_unitary(rng, 2, (2, 6, 6), special=True)
        lp, _ = gm.heatbath_sweep(links, 1.0, np.random.default_rng(11), use_numba=False)
        ln, _ = gm.heatbath_sweep(links, 1.0, np.random.default_rng(11), use_numba=True)
        self.assertLess(float(np.max(np.abs(lp - ln))), 1e-10)

    def test_heatbath_single_sweep_su3(self):
        rng = np.random.default_rng(6)
        links = random_unitary(rng, 3, (2, 4, 4), special=True)
        lp, _ = gm.heatbath_sweep(links, 1.5, np.random.default_rng(12), use_numba=False)
        ln, _ = gm.heatbath_sweep(links, 1.5, np.random.default_rng(12), use_numba=True)
        self.assertLess(float(np.max(np.abs(lp - ln))), 1e-10)

    def test_input_links_not_mutated(self):
        rng = np.random.default_rng(7)
        links = random_unitary(rng, 2, (2, 4, 4), special=True)
        before = links.copy()
        gm.heatbath_sweep(links, 1.0, rng, use_numba=True)
        self.assertTrue(np.array_equal(links, before))

    def test_unitarity_after_many_sweeps(self):
        rng = np.random.default_rng(8)
        links = random_unitary(rng, 2, (2, 6, 6), special=True)
        for _ in range(50):
            links, _ = gm.heatbath_sweep(links, 0.7, rng, use_numba=True)
        self.assertTrue(is_unitary(links.reshape(-1, 2, 2)))

    def test_unitarity_su3_after_sweeps(self):
        rng = np.random.default_rng(9)
        links = random_unitary(rng, 3, (3, 3, 3, 3), special=True)
        for _ in range(15):
            links, _ = gm.heatbath_sweep(links, 1.0, rng, use_numba=True)
        self.assertTrue(is_unitary(links.reshape(-1, 3, 3)))

    def test_overrelax_matches_reference(self):
        rng = np.random.default_rng(10)
        links = random_unitary(rng, 2, (2, 5, 5), special=True)
        lp, _ = gm.overrelaxation_sweep(links, rng, use_numba=False)
        ln, _ = gm.overrelaxation_sweep(links, rng, use_numba=True)
        self.assertLess(float(np.max(np.abs(lp - ln))), 1e-10)

    def test_overrelax_preserves_action_via_kernels(self):
        rng = np.random.default_rng(11)
        links = random_unitary(rng, 2, (2, 6, 6), special=True, scale=0.4)
        for _ in range(30):
            links, _ = gm.heatbath_sweep(links, 2.0, rng, use_numba=True)
        s0 = wilson_action(links)
        links2, _ = gm.overrelaxation_sweep(links, rng, use_numba=True)
        self.assertLess(abs(wilson_action(links2) - s0) / max(s0, 1e-12), 1e-10)


@unittest.skipUnless(HAVE_NUMBA, "numba not available")
class TestObservableEquivalence(unittest.TestCase):

    def test_wilson_loop_matches(self):
        rng = np.random.default_rng(20)
        for n, shape in ((2, (2, 6, 6)), (3, (3, 4, 4, 4))):
            links = random_unitary(rng, n, shape, special=True)
            for extents in ((1, 1), (2, 2), (2, 3)):
                w_p = gm.wilson_loop(links, extents, use_numba=False)
                w_n = gm.wilson_loop(links, extents, use_numba=True)
                self.assertAlmostEqual(w_p, w_n, places=12)

    def test_polyakov_matches(self):
        rng = np.random.default_rng(21)
        for n, shape in ((2, (2, 6, 6)), (3, (3, 4, 4, 4))):
            links = random_unitary(rng, n, shape, special=True)
            p_p = gm.polyakov_loop(links, use_numba=False)
            p_n = gm.polyakov_loop(links, use_numba=True)
            self.assertAlmostEqual(p_p, p_n, places=12)


@unittest.skipUnless(HAVE_NUMBA, "numba not available")
class TestJITEnsemblePhysics(unittest.TestCase):

    def test_jit_chain_hits_bessel_benchmark(self):
        """A JIT-driven ensemble must reproduce <W(1,1)> = I₂(2β)/I₁(2β)."""
        beta_g = 1.0
        rng = np.random.default_rng(313)
        links = random_unitary(rng, 2, (2, 8, 8), special=True)
        for _ in range(200):
            links, _ = gm.heatbath_sweep(links, beta_g, rng, use_numba=True)
        total = 0.0
        n_meas = 300
        for _ in range(n_meas):
            for _ in range(2):
                links, _ = gm.heatbath_sweep(links, beta_g, rng, use_numba=True)
            total += gm.wilson_loop(links, (1, 1))
        measured = total / n_meas
        exact = float(iv(2, 2 * beta_g) / i1(2 * beta_g))
        self.assertAlmostEqual(
            measured, exact, delta=0.03,
            msg=f"JIT ensemble <W(1,1)>={measured:.4f} vs exact {exact:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
