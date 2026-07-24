"""The chiral leaning imposes a handedness: sign(rotation) = sign(λ)·charge."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.nematic_spinor import plaquette_winding  # noqa: E402
from project_genesis.two_field import step_chiral_detuned  # noqa: E402
from project_genesis.vortex_chiral import imprint_vortices  # noqa: E402

N = 64
MID = N / 2.0


def _cores(psi, sign):
    w = plaquette_winding(psi)
    return [np.array([x + 0.5, y + 0.5]) for x, y in zip(*np.nonzero(w))
            if np.sign(w[x, y]) == sign]


def _pair_rotation(lam, charge, sep=10.0, steps=300):
    """Accumulated rotation (deg) of a same-charge pair, nearest-neighbour tracked."""
    c1 = np.array([MID - sep / 2, MID])
    c2 = np.array([MID + sep / 2, MID])
    psi = imprint_vortices((N, N), [tuple(c1), tuple(c2)], [charge, charge],
                           core=3.0)
    ang, prev = 0.0, c2 - c1
    for _ in range(steps):
        psi = step_chiral_detuned(psi, np.zeros((N, N)), chiral=lam, dt=0.1)
        cc = _cores(psi, np.sign(charge))
        if len(cc) < 2:
            break
        c1 = min(cc, key=lambda p: np.hypot(*(p - c1)))
        c2 = min(cc, key=lambda p: np.hypot(*(p - c2)))
        if np.allclose(c1, c2):
            break
        v = c2 - c1
        ang += np.arctan2(prev[0] * v[1] - prev[1] * v[0], prev @ v)
        prev = v
    return np.degrees(ang)


class ChiralHandednessTests(unittest.TestCase):
    def test_lambda_zero_is_parity_symmetric(self):
        self.assertLess(abs(_pair_rotation(0.0, 1)), 5.0)     # no handed rotation

    def test_handedness_flips_with_the_chiral_sign(self):
        # CP-like: λ → −λ reverses the winding sense
        rp = _pair_rotation(1.0, 1)
        rm = _pair_rotation(-1.0, 1)
        self.assertGreater(rp, 5.0)
        self.assertLess(rm, -5.0)

    def test_matter_and_antimatter_are_chirally_opposite(self):
        # the same leaning winds +½ and −½ in opposite senses
        self.assertLess(_pair_rotation(1.0, 1) * _pair_rotation(1.0, -1), 0.0)


if __name__ == "__main__":
    unittest.main()
