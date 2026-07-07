"""Checks for the Platonic-form selection helpers."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_form_selection import selected_form, summarise  # noqa: E402


# a corpus where neutrality is a P=3 monopoly and distinction rises with P
FORMS = {
    2: {"delta_c": 0.0052, "neutrality": 0.0000},
    3: {"delta_c": 0.0055, "neutrality": 0.0110},
    4: {"delta_c": 0.0053, "neutrality": 0.0009},
    5: {"delta_c": 0.0058, "neutrality": 0.0000},
    6: {"delta_c": 0.0061, "neutrality": 0.0000},
}
PHASES = [2, 3, 4, 5, 6]


class TestSelectedForm(unittest.TestCase):

    def test_abundant_capacity_selects_three(self):
        # high κ·w lifts the integration bonus above the ΔC gap → P=3
        P, _ = selected_form(FORMS, PHASES, kappa=0.95, w=1.6)
        self.assertEqual(P, 3)

    def test_scarce_capacity_selects_fragmentation(self):
        # κ·w → 0 kills integration; the max-distinction form wins
        P, _ = selected_form(FORMS, PHASES, kappa=0.02, w=0.05)
        self.assertEqual(P, 6)  # highest ΔC

    def test_zero_capacity_is_pure_distinction(self):
        P, _ = selected_form(FORMS, PHASES, kappa=0.0, w=1.0)
        self.assertEqual(P, 6)

    def test_boundary_monotone_in_capacity(self):
        # once P=3 is selected at some κ, it stays selected for larger κ (w fixed)
        w = 0.4
        selected3 = [selected_form(FORMS, PHASES, kappa=k, w=w)[0] == 3
                     for k in (0.02, 0.1, 0.2, 0.4, 0.7, 0.95)]
        # no True may appear before a False (monotone onset)
        first_true = selected3.index(True) if True in selected3 else len(selected3)
        self.assertTrue(all(selected3[i] for i in range(first_true, len(selected3))))


class TestSummarise(unittest.TestCase):

    def test_verdict_reports_three_monopoly_and_island(self):
        kappas = [0.02, 0.1, 0.4, 0.95]
        ws = [0.05, 0.4, 1.6]
        sel = {(ci, wi): selected_form(FORMS, PHASES, kappas[ci], ws[wi])[0]
               for ci in range(len(kappas)) for wi in range(len(ws))}
        thermal = [{"consumption": 0.5, "kappa_mean": 0.95, "neutrality": 0.0},
                   {"consumption": 32, "kappa_mean": 0.55, "neutrality": 0.0}]
        text = "\n".join(summarise(FORMS, PHASES, kappas, ws, sel, thermal))
        self.assertIn("peaks at P=3", text)
        self.assertIn("conditional", text)
        self.assertIn("fragmentation", text)


if __name__ == "__main__":
    unittest.main()
