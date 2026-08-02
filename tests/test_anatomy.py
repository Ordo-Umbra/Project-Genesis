"""Tests for the chord-excess reading — the destination question as one scalar.

The claim `anatomy.py` exists to support is that ``sign(E)`` is exactly the
condition for a third convex-hull vertex, i.e. for the optimum stopping at an
intermediate state instead of crossing once. If that equivalence is not exact,
the whole experiment reduces to two loosely-correlated statistics and proves
nothing, so the central test here checks it against `optimum_walk` on random
point clouds rather than on a hand-picked fixture.

The rest pin the degenerate cases that would otherwise report silently wrong
answers: a cloud with nothing between the two ends, a peak that IS the ordered
point, and a scan where the order parameter never crosses the levels the width
reading interpolates between.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.anatomy import (  # noqa: E402
    anatomy_reading,
    chord_excess,
    hull_size,
    transition_width,
)
from project_genesis.crossing import optimum_walk  # noqa: E402


def rows_from(points, x_key="T", load=0.05):
    return [{x_key: float(i), "delta_i": float(di), "delta_c": float(dc),
             "activity": load} for i, (di, dc) in enumerate(points)]


# a spike peak (nothing above the chord) and a plateau peak (something is)
SPIKE = [(1.00, 0.00), (0.80, 0.05), (0.55, 0.18), (0.40, 0.50), (0.15, 0.04)]
PLATEAU = [(1.00, 0.00), (0.80, 0.04), (0.68, 0.30), (0.30, 0.32), (0.10, 0.05)]


class TestExcessIsExactlyTheHullCondition(unittest.TestCase):

    def test_sign_of_excess_matches_hull_size_on_random_clouds(self):
        """The equivalence the experiment rests on, checked without fixtures."""
        rng = np.random.default_rng(0)
        seen = {2: 0, 3: 0}
        for _ in range(400):
            n = int(rng.integers(4, 12))
            di = np.sort(rng.random(n))[::-1]          # ordered end first
            dc = rng.random(n)
            dc[0] = 0.0                                 # ordered point: no ΔC
            rows = rows_from(list(zip(di, dc)))
            e = chord_excess(rows)["excess"]
            n_hull = hull_size(rows)
            if not np.isfinite(e):
                continue
            self.assertEqual(e > 0, n_hull > 2,
                             msg=f"E={e} hull={n_hull} di={di} dc={dc}")
            seen[min(n_hull, 3)] = seen.get(min(n_hull, 3), 0) + 1
        self.assertGreater(seen[2], 10)                 # both branches exercised
        self.assertGreater(seen[3], 10)

    def test_a_spike_peak_gives_no_intermediate(self):
        rows = rows_from(SPIKE)
        self.assertLess(chord_excess(rows)["excess"], 0.0)
        self.assertEqual(hull_size(rows), 2)

    def test_a_plateau_peak_gives_one(self):
        rows = rows_from(PLATEAU)
        self.assertGreater(chord_excess(rows)["excess"], 0.0)
        self.assertEqual(hull_size(rows), 3)

    def test_the_excess_is_measured_against_the_chord_not_the_peak(self):
        rows = rows_from(PLATEAU)
        e = chord_excess(rows)
        di = [r["delta_i"] for r in rows]
        dc = [r["delta_c"] for r in rows]
        o, s, j = 0, int(np.argmax(dc)), e["index"]
        chord = dc[o] + (dc[s] - dc[o]) * (di[o] - di[j]) / (di[o] - di[s])
        self.assertAlmostEqual(e["excess"], dc[j] - chord, places=12)

    def test_excess_frac_is_relative_to_the_distinction_gap(self):
        e = chord_excess(rows_from(PLATEAU))
        self.assertAlmostEqual(e["excess_frac"], e["excess"] / e["gap"],
                               places=12)


class TestDegenerateClouds(unittest.TestCase):

    def test_two_points_only_reports_no_intermediate_rather_than_crashing(self):
        e = chord_excess(rows_from([(1.0, 0.0), (0.4, 0.5)]))
        self.assertEqual(e["excess"], float("-inf"))
        self.assertEqual(e["index"], -1)

    def test_a_peak_at_the_ordered_point_is_reported_as_undefined(self):
        """No span to lay a chord across."""
        e = chord_excess(rows_from([(1.0, 0.6), (0.5, 0.2), (0.2, 0.1)]))
        self.assertTrue(np.isnan(e["excess"]))

    def test_points_outside_the_two_ends_are_not_counted(self):
        """A point with ΔI below the peak's is past the end of the chord."""
        base = rows_from(PLATEAU)
        e0 = chord_excess(base)
        extra = base + [{"T": 9.0, "delta_i": 0.05, "delta_c": 0.31,
                         "activity": 0.05}]
        self.assertAlmostEqual(chord_excess(extra)["excess"], e0["excess"],
                               places=12)


class TestTransitionWidth(unittest.TestCase):

    def test_width_is_the_range_over_which_order_collapses(self):
        x = np.linspace(0.0, 2.0, 41)
        di = 1.0 / (1.0 + np.exp((x - 1.0) / 0.10))
        rows = [{"T": float(a), "delta_i": float(b), "delta_c": 0.0}
                for a, b in zip(x, di)]
        w = transition_width(rows, "T")
        self.assertLess(w["x_hi"], w["x_lo"])
        self.assertAlmostEqual(w["midpoint"], 1.0, delta=0.05)

    def test_a_sharper_transition_reads_narrower(self):
        def width(scale):
            x = np.linspace(0.0, 2.0, 81)
            di = 1.0 / (1.0 + np.exp((x - 1.0) / scale))
            rows = [{"T": float(a), "delta_i": float(b), "delta_c": 0.0}
                    for a, b in zip(x, di)]
            return transition_width(rows, "T")["relative_width"]
        vals = [width(s) for s in (0.05, 0.15, 0.35)]
        self.assertTrue(all(a < b for a, b in zip(vals, vals[1:])), msg=str(vals))

    def test_relative_width_is_scale_free(self):
        """Two substrates' control parameters are in different units; the
        reading has to be comparable across them or the sweep means nothing."""
        x = np.linspace(0.0, 2.0, 81)
        di = 1.0 / (1.0 + np.exp((x - 1.0) / 0.15))
        a = transition_width([{"T": float(p), "delta_i": float(q),
                               "delta_c": 0.0} for p, q in zip(x, di)], "T")
        b = transition_width([{"T": float(7 * p), "delta_i": float(q),
                               "delta_c": 0.0} for p, q in zip(x, di)], "T")
        self.assertAlmostEqual(a["relative_width"], b["relative_width"],
                               places=9)
        self.assertAlmostEqual(b["width"] / a["width"], 7.0, places=6)

    def test_a_scan_that_never_crosses_reports_nan(self):
        rows = [{"T": float(i), "delta_i": 0.95, "delta_c": 0.0}
                for i in range(5)]
        self.assertTrue(np.isnan(transition_width(rows, "T")["width"]))


class TestAnatomyReading(unittest.TestCase):

    def test_reading_bundles_the_pieces_consistently(self):
        rows = rows_from(PLATEAU)
        a = anatomy_reading(rows, "T")
        self.assertEqual(a["hull"], hull_size(rows))
        self.assertAlmostEqual(a["excess"], chord_excess(rows)["excess"],
                               places=12)

    def test_the_intermediate_state_is_identified_by_its_x_value(self):
        rows = rows_from(PLATEAU)
        a = anatomy_reading(rows, "T")
        self.assertAlmostEqual(a["x_intermediate"],
                               rows[chord_excess(rows)["index"]]["T"], places=12)

    def test_grid_spacing_is_reported_so_resolution_can_be_judged(self):
        rows = rows_from(PLATEAU)
        self.assertAlmostEqual(anatomy_reading(rows, "T")["grid"], 1.0,
                               places=12)


class TestExcessRespondsToPeakShape(unittest.TestCase):
    """Not a claim about substrates — a check that the scalar tracks the shape
    the experiment says it tracks, so a null result there would mean something."""

    def test_flattening_the_peak_raises_the_excess_monotonically(self):
        vals = []
        for dc_mid in (0.05, 0.15, 0.25, 0.31):
            pts = [(1.00, 0.00), (0.80, 0.04), (0.68, dc_mid),
                   (0.30, 0.32), (0.10, 0.05)]
            vals.append(chord_excess(rows_from(pts))["excess"])
        self.assertTrue(all(a < b for a, b in zip(vals, vals[1:])),
                        msg=str(vals))

    def test_a_peak_that_dominates_everything_leaves_no_intermediate(self):
        pts = [(1.00, 0.00), (0.80, 0.02), (0.68, 0.05), (0.30, 0.60),
               (0.10, 0.01)]
        self.assertLess(chord_excess(rows_from(pts))["excess"], 0.0)


if __name__ == "__main__":
    unittest.main()
