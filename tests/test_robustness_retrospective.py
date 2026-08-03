"""Tests for the re-scoring of earlier audits.

The diagnostics themselves are pinned in `test_robustness.py`. What is pinned
here is the reading of the published JSON, which is where this experiment can go
wrong quietly: a key renamed in one of the source experiments would make a claim
vanish from the report, and a report that silently drops a claim looks exactly
like a report where that claim passed.

So the cases below are mostly about absence. A missing file, a missing key, a
sweep too short to have a spread — each must come back empty and be surfaced as
UNAVAILABLE, never as a pass. The one thing worse than an unevidenced claim is
an unevidenced claim that no longer appears in the audit that was supposed to
catch it.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from n3_robustness_retrospective import (  # noqa: E402
    cliff_claim, eviction_claim, gap_claim, load,
)


class TestLoad(unittest.TestCase):

    def test_a_missing_file_is_none_not_an_exception(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(load(Path(d), "nope.json"))

    def test_a_present_file_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            p.write_text(json.dumps({"a": 1}))
            self.assertEqual(load(Path(d), "x.json"), {"a": 1})


class TestGapClaim(unittest.TestCase):

    @staticmethod
    def data():
        mk = lambda c, i: {"n_C": c, "n_I_raw": i, "gap_raw": c - i}  # noqa: E731
        return {"window": {"a": mk(2.0, 1.0), "b": mk(2.1, 1.2)},
                "floor": {"a": mk(2.0, 1.0), "b": mk(2.0, 1.3)},
                "floor_magnitude_posthoc": {"a": mk(2.0, 1.0),
                                            "b": mk(2.0, 1.4)}}

    def test_all_three_conventions_are_scored(self):
        g = gap_claim(self.data(), floor=0.02)
        self.assertEqual(len(g), 3)

    def test_a_pinned_n_C_reports_not_exercised(self):
        """The floor sweeps cannot move n_C, so the difference inherits n_I.
        This is the finding the whole module exists for; it must survive the
        round-trip through JSON."""
        g = gap_claim(self.data(), floor=0.02)
        for key in ("noise-floor band", "noise-floor magnitude"):
            self.assertFalse(g[key]["exercised"], msg=key)
            self.assertAlmostEqual(g[key]["spread_difference"],
                                   g[key]["spread_b"], places=14, msg=key)

    def test_absent_sections_are_skipped_not_faked(self):
        g = gap_claim({"window": self.data()["window"]}, floor=0.02)
        self.assertEqual(list(g), ["fitting window (common-mode)"])

    def test_empty_data_yields_no_claims(self):
        self.assertEqual(gap_claim({}, floor=0.02), {})


class TestCliffClaim(unittest.TestCase):

    @staticmethod
    def data(densities):
        # nested under "results", exactly as n3_junction_scale writes it —
        # the flat shape this test originally assumed is what let a wrong
        # loader pass, so the fixture mirrors the real file.
        return {"params": {"palettes": [2, 3, 4], "radii": [1, 2]},
                "results": {"2-D_density": densities}}

    def test_a_radius_sweep_becomes_a_ranking(self):
        d = self.data({"P2_r1": 0.0, "P3_r1": 0.4, "P4_r1": 0.0,
                       "P2_r2": 0.0, "P3_r2": 0.3, "P4_r2": 0.01})
        c = cliff_claim(d, tol=3.0)
        key = "2-D — argmax across radii [1, 2]"
        self.assertIn(key, c)
        self.assertEqual(c[key]["winner"], 3)
        self.assertFalse(c[key]["flips"])

    def test_identically_zero_rivals_come_back_structural(self):
        d = self.data({"P2_r1": 0.0, "P3_r1": 0.4, "P4_r1": 0.0,
                       "P2_r2": 0.0, "P3_r2": 0.3, "P4_r2": 0.0})
        c = cliff_claim(d, tol=3.0)
        h = c["2-D — argmax across radii [1, 2]"]
        self.assertTrue(h["structural"])
        self.assertFalse(h["exercised"])

    def test_an_incomplete_grid_is_dropped_rather_than_guessed(self):
        """A missing radius must not be silently backfilled — a partial sweep
        would score as a complete one and read as evidence."""
        d = self.data({"P2_r1": 0.0, "P3_r1": 0.4, "P4_r1": 0.0,
                       "P2_r2": 0.0, "P3_r2": 0.3})       # P4_r2 absent
        self.assertEqual(cliff_claim(d, tol=3.0), {})

    def test_a_missing_density_block_yields_nothing(self):
        self.assertEqual(cliff_claim({"params": {}, "results": {}}, tol=3.0), {})


class TestEvictionClaim(unittest.TestCase):
    """§5's condition after the ceiling was decided.

    `evicted` used to be `isfinite(c_evict)` on a ladder to c = 1e5, arguing
    with the run's own c_max = 50. That ambiguity is resolved in
    `crossing.evictable`: c is the wrong variable, and in the capacity form no
    ceiling survives. The condition became two margins — the ordered point
    starts ahead, and holding it costs capacity — and those are the primary
    rows here. The c-ceiling rows are kept as the sensitivity that motivated
    the decision, and must never be counted as claims.
    """

    @staticmethod
    def data(hop_c_evict, kur_c_evict, margin=0.3, load_ratio=0.5,
             c_max=50.0):
        """`c_evict` for the sensitivity rows; `crossing_margin` and the load
        ratio for the two primary ones."""
        def mk(ce):
            return {"c_evict": ce, "crossing_margin": margin,
                    "load_o": load_ratio, "load_s": 1.0}
        arms = {f"hopfield th={i}": mk(v) for i, v in enumerate(hop_c_evict)}
        arms.update({f"kuramoto tol={i}": mk(v)
                     for i, v in enumerate(kur_c_evict)})
        return {"arms": arms, "params": {"c_max": c_max}}

    @staticmethod
    def primary(e):
        return {k: v for k, v in e.items() if not v.get("sensitivity")}

    def test_the_two_ceiling_free_margins_are_the_primary_claims(self):
        e = eviction_claim(self.data([0.4, 0.5, 0.45], [0.3, 0.35, 0.32]),
                           tol=3.0)
        prim = self.primary(e)
        self.assertEqual(len(prim), 4)          # 2 margins x 2 families
        self.assertTrue(all("starts ahead" in k or "numerical zero" in k
                            for k in prim), msg=str(list(prim)))

    def test_every_consumption_row_is_a_sensitivity_never_a_claim(self):
        """The regression guard on the decision: if a c-ceiling row is ever
        counted as a claim again, the ambiguity is back in the tally."""
        e = eviction_claim(self.data([0.4, 0.5, 0.45], [0.3, 0.35, 0.32]),
                           tol=3.0)
        for k, v in e.items():
            is_c_row = "budget" in k or "ladder" in k
            self.assertEqual(bool(v.get("sensitivity")), is_c_row, msg=k)

    def test_a_free_ordered_phase_fails_the_load_margin(self):
        """L_o = 0 is the control. It must fail, or the condition is vacuous."""
        e = self.primary(eviction_claim(
            self.data([0.4, 0.5], [0.3, 0.35], load_ratio=0.0), tol=3.0))
        load_rows = [v for k, v in e.items() if "numerical zero" in k]
        self.assertTrue(load_rows)
        for h in load_rows:
            self.assertFalse(h["holds"])

    def test_machine_noise_also_fails_the_load_margin(self):
        """1.24e-15 is what an undriven scan actually measures. A literal
        `> 0` would pass it; the decades form must not."""
        e = self.primary(eviction_claim(
            self.data([0.4, 0.5], [0.3, 0.35], load_ratio=1.24e-15), tol=3.0))
        for k, h in e.items():
            if "numerical zero" in k:
                self.assertFalse(h["holds"], msg=k)

    def test_a_driven_load_clears_the_margin_by_many_decades(self):
        e = self.primary(eviction_claim(
            self.data([0.4, 0.5], [0.3, 0.35], load_ratio=1e-2), tol=3.0))
        for k, h in e.items():
            if "numerical zero" in k:
                self.assertTrue(h["holds"], msg=k)
                self.assertGreater(h["min_margin"], 6.0, msg=k)

    def test_an_ordered_point_starting_behind_fails(self):
        e = self.primary(eviction_claim(
            self.data([0.4, 0.5], [0.3, 0.35], margin=-0.2), tol=3.0))
        for k, h in e.items():
            if "starts ahead" in k:
                self.assertFalse(h["holds"], msg=k)

    def test_a_family_with_one_arm_is_not_scored(self):
        """One arm has no spread, so 'it held' says nothing at all."""
        e = self.primary(eviction_claim(self.data([0.4], [0.3, 0.35]),
                                        tol=3.0))
        self.assertTrue(all("kuramoto" in k for k in e), msg=str(list(e)))

    def test_arms_without_any_margin_key_are_dropped(self):
        """The published anatomy.json stored only the boolean. That must come
        back empty — reported as UNAVAILABLE — and never as a pass."""
        d = {"arms": {"hopfield th=0": {"evicted": True},
                      "hopfield th=1": {"evicted": True}}}
        self.assertEqual(eviction_claim(d, tol=3.0), {})

    def test_no_arms_yields_nothing(self):
        self.assertEqual(eviction_claim({}, tol=3.0), {})


if __name__ == "__main__":
    unittest.main()
