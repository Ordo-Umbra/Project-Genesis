"""Checks for the exchange-sign (spin–statistics) instrument."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.spin_statistics import (  # noqa: E402
    braid_positions,
    exchange_holonomy,
    self_rotation_sign,
)

SHAPE = (200, 200)
MID = (100.0, 100.0)
D = 30.0
KW = dict(core=3.0, n_steps=241)


class TestBraidGeometry(unittest.TestCase):

    def test_half_turn_swaps_the_pair(self):
        pos = list(braid_positions(MID, D, 0.5, 1.0, 51))
        a0, b0 = pos[0]
        a1, b1 = pos[-1]
        # after one exchange, A sits where B started and vice versa
        self.assertAlmostEqual(a1[0], b0[0], places=4)
        self.assertAlmostEqual(a1[1], b0[1], places=4)
        self.assertAlmostEqual(b1[0], a0[0], places=4)

    def test_full_braid_returns_home(self):
        pos = list(braid_positions(MID, D, 1.0, 1.0, 51))
        a0, b0 = pos[0]
        a1, b1 = pos[-1]
        self.assertAlmostEqual(a1[0], a0[0], places=4)
        self.assertAlmostEqual(b1[1], b0[1], places=4)


class TestExchangeSign(unittest.TestCase):
    """The heart: exchange sign = (−1)^{2s} = (−1)^q."""

    def test_half_disclination_pair_is_antisymmetric(self):
        out = exchange_holonomy(SHAPE, MID, D, (1, 1), turns=0.5, **KW)
        self.assertLess(out["sign"], -0.8)          # fermionic −1
        self.assertAlmostEqual(abs(out["k"]), 1.0, places=1)

    def test_integer_disclination_pair_is_symmetric(self):
        out = exchange_holonomy(SHAPE, MID, D, (2, 2), turns=0.5, **KW)
        self.assertGreater(out["sign"], 0.8)        # bosonic +1

    def test_three_halves_pair_is_antisymmetric(self):
        out = exchange_holonomy(SHAPE, MID, D, (3, 3), turns=0.5, **KW)
        self.assertLess(out["sign"], -0.8)          # (−1)^3 = −1

    def test_sign_is_braid_direction_independent(self):
        plus = exchange_holonomy(SHAPE, MID, D, (1, 1), turns=0.5,
                                 arc_sign=+1.0, **KW)
        minus = exchange_holonomy(SHAPE, MID, D, (1, 1), turns=0.5,
                                  arc_sign=-1.0, **KW)
        self.assertLess(plus["sign"], -0.8)
        self.assertLess(minus["sign"], -0.8)


class TestControls(unittest.TestCase):

    def test_double_braid_is_bosonic(self):
        # two exchanges = a boson, for a fermionic pair
        out = exchange_holonomy(SHAPE, MID, D, (1, 1), turns=1.0, **KW)
        self.assertGreater(out["sign"], 0.8)

    def test_far_field_sees_only_total_charge(self):
        # the exchange sign is internal; far away winds 0 (fusion, not exchange)
        out = exchange_holonomy(SHAPE, MID, D, (1, 1), turns=0.5,
                                probe=(MID[0], MID[1] + 64), **KW)
        self.assertAlmostEqual(out["k"], 0.0, places=1)
        self.assertGreater(out["sign"], 0.8)

    def test_particle_antiparticle_pair_has_no_exchange_sign(self):
        # (½, −½): windings cancel at the centre → +1 (the −1 needs identicals)
        out = exchange_holonomy(SHAPE, MID, D, (1, -1), turns=0.5, **KW)
        self.assertGreater(out["sign"], 0.8)


class TestConnection(unittest.TestCase):
    """The spin–statistics connection: exchange sign == 2π self-rotation."""

    def test_exchange_equals_self_rotation(self):
        for q in (1, 2, 3):
            rot = self_rotation_sign(SHAPE, MID, q, core=3.0, radius=12.0)
            exch = exchange_holonomy(SHAPE, MID, D, (q, q), turns=0.5, **KW)["sign"]
            self.assertAlmostEqual(rot, exch, delta=0.2,
                                   msg=f"q={q}: rot {rot} vs exch {exch}")

    def test_self_rotation_parity(self):
        self.assertLess(self_rotation_sign(SHAPE, MID, 1), -0.8)     # ½ → −1
        self.assertGreater(self_rotation_sign(SHAPE, MID, 2), 0.8)   # 1 → +1


if __name__ == "__main__":
    unittest.main()
