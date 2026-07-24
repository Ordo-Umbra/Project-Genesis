"""Checks for phase-aware pinning of the dynamical braid."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.spin_statistics import dynamical_braid, _phase_mask  # noqa: E402

# small / short braids so the checks run fast
BRAID = dict(delta=0.5, sub=40, relax=300, width=5.0, depth=0.9)


class TestPhaseMask(unittest.TestCase):

    def test_mask_peaks_at_centres_and_decays(self):
        m = _phase_mask((60, 60), [(20, 30), (40, 30)], mask_width=5.0)
        self.assertAlmostEqual(m[20, 30], 1.0, places=2)
        self.assertAlmostEqual(m[40, 30], 1.0, places=2)
        self.assertLess(m[5, 5], 0.05)
        self.assertTrue((m >= 0).all() and (m <= 1.0 + 1e-9).all())


class TestPhasePinCompletesBraid(unittest.TestCase):

    def test_pin_improves_survival_and_sign(self):
        # amplitude-only vs phase-pinned on the same short ½,½ braid
        amp = dynamical_braid((110, 110), (55, 55), 16.0, (1, 1),
                              phase_pin=0.0, **BRAID)
        pin = dynamical_braid((110, 110), (55, 55), 16.0, (1, 1),
                              phase_pin=0.4, **BRAID)
        # the phase pin conserves the cores at least as well ...
        self.assertGreaterEqual(pin["survived_fraction"],
                                amp["survived_fraction"])
        # ... and drives the exchange sign toward the fermionic −1
        self.assertLess(pin["sign"], -0.7)
        self.assertLessEqual(pin["sign"], amp["sign"] + 1e-6)

    def test_pinned_fermion_is_negative(self):
        pin = dynamical_braid((110, 110), (55, 55), 16.0, (1, 1),
                              phase_pin=0.4, **BRAID)
        self.assertLess(pin["sign"], 0.0)


class TestPinTransportsNotFabricates(unittest.TestCase):
    """The honest control: the pin carries winding; the sign is the braid's."""

    def test_integer_pair_stays_bosonic_under_the_pin(self):
        # same pin as the fermion test, but an integer (s=1) pair → +1
        out = dynamical_braid((110, 110), (55, 55), 16.0, (2, 2),
                              phase_pin=0.4, **BRAID)
        self.assertGreater(out["sign"], 0.0)

    def test_double_braid_is_bosonic_under_the_pin(self):
        # two exchanges of the ½,½ pair → +1, even with the pin on
        out = dynamical_braid((110, 110), (55, 55), 16.0, (1, 1), turns=1.0,
                              phase_pin=0.4, **BRAID)
        self.assertGreater(out["sign"], 0.0)


class TestBackwardCompatible(unittest.TestCase):

    def test_default_is_amplitude_only(self):
        # phase_pin defaults to 0 → identical to a plain amplitude-only braid
        a = dynamical_braid((110, 110), (55, 55), 16.0, (1, 1), **BRAID)
        b = dynamical_braid((110, 110), (55, 55), 16.0, (1, 1),
                            phase_pin=0.0, **BRAID)
        self.assertAlmostEqual(a["k"], b["k"], places=6)


if __name__ == "__main__":
    unittest.main()
