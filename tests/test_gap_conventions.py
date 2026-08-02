"""Tests for the convention audit of §2's one-dimension gap.

Two things need pinning, and only one of them is the usual kind.

The first is ordinary: the scaled-floor kernel must reduce **exactly** to
`area_law.mi_kernel` at scale 1, or the post-hoc sweep is measuring a different
instrument than the published one and its numbers cannot be compared.

The second is the point of the module. :func:`cancellation` exists because the
bar used by five previous audits — "the difference moved less than the
tolerance" — cannot tell cancellation from nothing having moved. These tests
construct all three cases by hand, with known answers, so the diagnostic is
shown to separate them rather than assumed to. In particular the arithmetic
trap gets its own test: when one part is pinned, ``gap = n_C − n_I`` inherits
the other's spread identically, and a diagnostic that scored that as a pass
would be reporting a subtraction as a result.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from project_genesis.area_law import (  # noqa: E402
    ensemble_correlation, gaussian_control_field, mi_kernel,
)
from n3_gap_conventions import (  # noqa: E402
    FLOORS, LADDERS, PUBLISHED_FLOOR, PUBLISHED_HALVES,
    cancellation, exponents, exponents_scaled_floor, spread,
)


def _ensemble(n=48, xi=3.0, k=6, seed=0):
    rng = np.random.default_rng(seed)
    return [gaussian_control_field(n, xi, rng) for _ in range(k)]


class TestScaledFloorReducesToThePublishedKernel(unittest.TestCase):
    """Without this the post-hoc sweep is about some other instrument."""

    def test_scale_one_reproduces_the_published_floor(self):
        comps = _ensemble()
        grads = [np.abs(c) for c in comps]
        halves = [6, 8, 10, 12]
        pub = exponents(comps, grads, halves, PUBLISHED_FLOOR)
        one = exponents_scaled_floor(comps, grads, halves, 1.0)
        self.assertAlmostEqual(one["floor_value"], pub["floor_value"], places=14)
        self.assertAlmostEqual(one["n_I_raw"], pub["n_I_raw"], places=12)
        self.assertAlmostEqual(one["n_C"], pub["n_C"], places=12)

    def test_the_floor_scales_linearly(self):
        comps = _ensemble()
        grads = [np.abs(c) for c in comps]
        halves = [6, 8, 10, 12]
        base = exponents_scaled_floor(comps, grads, halves, 1.0)["floor_value"]
        for s in (0.0, 0.5, 2.0):
            got = exponents_scaled_floor(comps, grads, halves, s)["floor_value"]
            self.assertAlmostEqual(got, base * s, places=14)

    def test_scale_zero_subtracts_nothing(self):
        """The 'I don't believe in a floor' convention: the raw ρ² kernel."""
        comps = _ensemble()
        g = ensemble_correlation(comps)
        rho2 = (g / g[0, 0]) ** 2
        grads = [np.abs(c) for c in comps]
        zero = exponents_scaled_floor(comps, grads, [6, 8, 10, 12], 0.0)
        self.assertEqual(zero["floor_value"], 0.0)
        # and a subtracted kernel is pointwise no larger than the raw one
        kern, _ = mi_kernel(g)
        self.assertTrue(np.all(kern <= rho2 + 1e-15))

    def test_n_C_is_blind_to_the_floor(self):
        """The differential claim, as a test: n_C never touches the kernel."""
        comps = _ensemble()
        grads = [np.abs(c) for c in comps]
        vals = [exponents_scaled_floor(comps, grads, [6, 8, 10, 12], s)["n_C"]
                for s in (0.0, 1.0, 3.0)]
        self.assertAlmostEqual(spread(vals), 0.0, places=14)

    def test_a_bigger_floor_lowers_the_measured_exponent(self):
        """Subtracting more of the tail makes the kernel shorter-ranged, so
        the boundary term looks more like an area law. Monotone, which is why
        the sweep is interpretable at all."""
        comps = _ensemble()
        grads = [np.abs(c) for c in comps]
        vals = [exponents_scaled_floor(comps, grads, [6, 8, 10, 12], s)["n_I_raw"]
                for s in (0.0, 1.0, 2.0, 4.0)]
        self.assertTrue(all(a >= b - 1e-9 for a, b in zip(vals, vals[1:])),
                        msg=str(vals))


class TestCancellationSeparatesTheThreeCases(unittest.TestCase):
    """The whole reason the diagnostic exists."""

    @staticmethod
    def rows(nc, ni):
        return [{"n_C": c, "n_I_raw": i, "gap_raw": c - i}
                for c, i in zip(nc, ni)]

    def test_genuine_cancellation_is_reported(self):
        """Both parts swing by 1.0 together; the difference barely moves."""
        shift = [0.0, 0.4, -0.6, 1.0, -0.5]
        r = self.rows([2.0 + s for s in shift], [1.0 + s + 0.01 * i
                                                 for i, s in enumerate(shift)])
        c = cancellation(r)
        self.assertTrue(c["exercised"])
        self.assertTrue(c["cancels"])
        self.assertLess(c["ratio_vs_smaller"], 0.2)

    def test_one_part_pinned_is_not_cancellation_it_is_subtraction(self):
        """The arithmetic trap. n_C fixed, so gap spread EQUALS n_I spread
        exactly — and a diagnostic that called that a pass would be scoring a
        subtraction. It must report the sweep as not exercised."""
        r = self.rows([2.0] * 4, [1.0, 1.3, 0.8, 1.1])
        c = cancellation(r)
        self.assertAlmostEqual(c["n_C"], 0.0, places=14)
        self.assertAlmostEqual(c["gap"], c["n_I"], places=14)
        self.assertFalse(c["exercised"])
        self.assertFalse(c["cancels"])

    def test_nothing_moving_is_not_cancellation_either(self):
        r = self.rows([2.0, 2.001, 1.999], [1.0, 1.001, 0.999])
        c = cancellation(r)
        self.assertFalse(c["exercised"])
        self.assertFalse(c["cancels"])

    def test_both_parts_move_but_oppositely_is_reported_as_no_cancellation(self):
        """Anti-correlated parts: the difference moves more than either. The
        diagnostic must not confuse 'exercised' with 'passed'."""
        r = self.rows([2.0, 2.5, 1.5], [1.0, 0.5, 1.5])
        c = cancellation(r)
        self.assertTrue(c["exercised"])
        self.assertFalse(c["cancels"])
        self.assertGreater(c["ratio_vs_smaller"], 1.0)

    def test_the_floor_argument_gates_exercised(self):
        r = self.rows([2.0, 2.1, 1.9], [1.0, 1.1, 0.9])
        self.assertTrue(cancellation(r, floor=0.05)["exercised"])
        self.assertFalse(cancellation(r, floor=0.5)["exercised"])

    def test_a_single_row_has_no_spread_and_is_not_exercised(self):
        c = cancellation(self.rows([2.0], [1.0]))
        self.assertFalse(c["exercised"])


class TestSweepDefinitions(unittest.TestCase):
    """The sweeps have to be what the docstring says they are."""

    def test_the_published_ladder_is_in_the_sweep_unchanged(self):
        self.assertIn(PUBLISHED_HALVES, list(LADDERS.values()))

    def test_the_published_floor_is_in_the_sweep_unchanged(self):
        self.assertIn(PUBLISHED_FLOOR, list(FLOORS.values()))

    def test_every_ladder_has_at_least_three_points(self):
        """A two-point 'fit' is a chord, not a power law."""
        for name, h in LADDERS.items():
            self.assertGreaterEqual(len(h), 3, msg=name)

    def test_every_ladder_fits_inside_the_default_box(self):
        """half-width must stay under n/2 = 80, or the region is clipped and
        the exponent measures the boundary condition instead."""
        for name, h in LADDERS.items():
            self.assertLess(max(h), 80, msg=name)

    def test_every_floor_band_is_ordered_and_outside_the_correlated_core(self):
        for name, (lo, hi) in FLOORS.items():
            self.assertLess(lo, hi, msg=name)
            self.assertGreaterEqual(lo, 0.2, msg=name)
            self.assertLessEqual(hi, 0.5, msg=name)


class TestSpread(unittest.TestCase):

    def test_spread_is_max_minus_min(self):
        self.assertAlmostEqual(spread([1.0, 3.0, 2.0]), 2.0)

    def test_non_finite_values_are_dropped(self):
        self.assertAlmostEqual(spread([1.0, np.nan, 3.0, np.inf]), 2.0)

    def test_an_all_non_finite_list_is_nan_not_zero(self):
        """Zero would read as 'perfectly stable', which is the opposite of
        'nothing was measured'."""
        self.assertTrue(np.isnan(spread([np.nan, np.inf])))


if __name__ == "__main__":
    unittest.main()
