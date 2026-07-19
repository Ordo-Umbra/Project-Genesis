"""Checks for the chiral (complex Ginzburg–Landau) spin term."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.chiral_field import (  # noqa: E402
    evolve_chiral,
    net_vorticity,
    precession_rate,
    step_chiral,
    vorticity,
)


class ChiralTests(unittest.TestCase):
    def test_real_gl_does_not_precess(self):
        psi = evolve_chiral((48, 48), chiral=0.0, steps=300, seed=1)
        self.assertLess(abs(precession_rate(psi, chiral=0.0)), 0.02)

    def test_precession_equals_minus_lambda(self):
        for lam in (0.4, -0.6):
            psi = evolve_chiral((48, 48), chiral=lam, steps=400, seed=2)
            omega = precession_rate(psi, chiral=lam)
            self.assertAlmostEqual(omega, -lam, delta=0.1)

    def test_precession_sign_flips_with_coupling(self):
        pp = evolve_chiral((48, 48), chiral=0.6, steps=400, seed=3)
        pm = evolve_chiral((48, 48), chiral=-0.6, steps=400, seed=3)
        self.assertLess(precession_rate(pp, chiral=0.6)
                        * precession_rate(pm, chiral=-0.6), 0.0)

    def test_net_vorticity_is_topologically_neutral(self):
        # total vorticity ~ 0 on the torus regardless of chirality
        psi = evolve_chiral((48, 48), chiral=0.8, steps=400, seed=4)
        self.assertLess(abs(net_vorticity(psi)), 1e-2)

    def test_vorticity_is_parity_odd(self):
        # mirroring x sends omega -> -omega pointwise
        psi = evolve_chiral((48, 48), chiral=0.5, steps=200, seed=5)
        w = vorticity(psi)
        w_mirror = vorticity(psi[::-1, :])
        self.assertTrue(np.allclose(w_mirror, -w[::-1, :], atol=1e-9))

    def test_step_preserves_shape_and_complex(self):
        psi = (np.random.default_rng(0).standard_normal((16, 16))
               + 1j * np.random.default_rng(1).standard_normal((16, 16)))
        out = step_chiral(psi, chiral=0.3)
        self.assertEqual(out.shape, psi.shape)
        self.assertTrue(np.iscomplexobj(out))


if __name__ == "__main__":
    unittest.main()
