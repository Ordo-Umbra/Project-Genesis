"""Checks for the 3-D recovery-rescue analysis helpers."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_recovery_rescue_3d import reconnection_threshold, summarise  # noqa: E402


def _row(r, span):
    return {"recovery": r, "min_spanning": span}


class TestReconnectionThreshold(unittest.TestCase):

    def test_interpolates_upward_crossing(self):
        # crosses ½ between r=0.8 (0.15) and r=1.6 (1.0):
        # 0.8 + 0.8·(0.5−0.15)/(1.0−0.15) = 1.129...
        rows = [_row(0.4, 0.0), _row(0.8, 0.15), _row(1.6, 1.0)]
        self.assertAlmostEqual(reconnection_threshold(rows), 1.1294117, places=5)

    def test_already_connected_returns_first_rate(self):
        rows = [_row(0.4, 0.8), _row(0.8, 1.0)]
        self.assertEqual(reconnection_threshold(rows), 0.4)

    def test_never_reconnects_returns_none(self):
        rows = [_row(0.4, 0.0), _row(0.8, 0.1), _row(1.6, 0.3)]
        self.assertIsNone(reconnection_threshold(rows))

    def test_custom_level(self):
        rows = [_row(0.5, 0.2), _row(1.0, 0.8)]
        # level 0.5 crosses at 0.5 + 0.5·(0.5−0.2)/(0.8−0.2) = 0.75
        self.assertAlmostEqual(reconnection_threshold(rows, 0.5), 0.75, places=6)


class TestSummarise(unittest.TestCase):

    def _rows(self, spans):
        rs = [0.3, 0.5, 0.8, 1.6, 2.4]
        return [_row(r, s) for r, s in zip(rs, spans)]

    def test_3d_needs_more_than_2d(self):
        by_dim = {
            2: self._rows([0.4, 1.0, 1.0, 1.0, 1.0]),   # r*_2D ~0.36
            3: self._rows([0.0, 0.1, 0.6, 1.0, 1.0]),   # r*_3D ~0.75
        }
        r_star = {d: reconnection_threshold(by_dim[d]) for d in (2, 3)}
        text = "\n".join(summarise(by_dim, r_star))
        self.assertIn("faster recovery DOES rescue 3-D", text)
        self.assertIn("needs markedly more healing than 2-D", text)

    def test_3d_never_reconnects_branch(self):
        by_dim = {
            2: self._rows([0.6, 1.0, 1.0, 1.0, 1.0]),
            3: self._rows([0.0, 0.05, 0.1, 0.2, 0.3]),
        }
        r_star = {d: reconnection_threshold(by_dim[d]) for d in (2, 3)}
        text = "\n".join(summarise(by_dim, r_star))
        self.assertIn("does not lift the worst-case 3-D", text)


if __name__ == "__main__":
    unittest.main()
