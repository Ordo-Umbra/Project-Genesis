"""Checks for the distinction–integration separation helpers."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_capacity_separation import distinction_integration, summarise  # noqa: E402


class TestDistinctionIntegration(unittest.TestCase):

    def test_three_fold_junction_is_fully_integrated(self):
        # a small P=3 field with all three colours meeting: every triple
        # junction carries the whole palette, so I == C and φ == 1
        f = np.zeros((3, 6, 6))
        f[0, :3, :] = 1.0          # colour 0 fills the top
        f[1, 3:, :3] = 1.0         # colour 1 bottom-left
        f[2, 3:, 3:] = 1.0         # colour 2 bottom-right
        d = distinction_integration(f, 3)
        self.assertGreater(d["distinction"], 0.0)
        self.assertAlmostEqual(d["integration"], d["distinction"], places=12)
        self.assertAlmostEqual(d["phi"], 1.0, places=12)

    def test_two_colours_have_no_triple_junction(self):
        f = np.zeros((2, 6, 6))
        f[0, :, :3] = 1.0
        f[1, :, 3:] = 1.0
        d = distinction_integration(f, 2)
        self.assertEqual(d["distinction"], 0.0)  # no triple points
        self.assertEqual(d["integration"], 0.0)
        self.assertEqual(d["phi"], 0.0)

    def test_phi_is_zero_when_no_distinction(self):
        f = np.zeros((3, 4, 4))
        f[0] = 1.0  # single uniform domain
        d = distinction_integration(f, 3)
        self.assertEqual(d["distinction"], 0.0)
        self.assertEqual(d["phi"], 0.0)


class TestSummarise(unittest.TestCase):

    def _forms(self):
        return {
            2: {"distinction": 0.0, "integration": 0.0, "phi": 0.0, "gap": 0.0},
            3: {"distinction": 0.007, "integration": 0.007, "phi": 1.0, "gap": 0.0},
            4: {"distinction": 0.017, "integration": 0.0, "phi": 0.0, "gap": 0.017},
            5: {"distinction": 0.019, "integration": 0.0, "phi": 0.0, "gap": 0.019},
        }

    def test_reports_structural_cliff(self):
        forms = self._forms()
        sweep = {
            3: [{"kappa": 0.1, "distinction": 0.56, "phi": 1.0},
                {"kappa": 1.0, "distinction": 0.04, "phi": 1.0}],
            4: [{"kappa": 0.1, "distinction": 0.74, "phi": 0.33},
                {"kappa": 1.0, "distinction": 0.06, "phi": 0.04}],
        }
        text = "\n".join(summarise(forms, [2, 3, 4, 5], sweep, [3, 4]))
        self.assertIn("structural cliff", text)
        self.assertIn("stays complete", text)          # P=3 branch
        self.assertIn("accidental palette-completeness", text)  # P=4 branch
        self.assertIn("distinction dial", text)


if __name__ == "__main__":
    unittest.main()
