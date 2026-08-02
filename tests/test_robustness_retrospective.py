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

    @staticmethod
    def data(hop, kur, c_max=50.0):
        """`c_evict` — the consumption at which eviction happens. NOT
        `excess_frac`: that is the chord excess, it is negative in several arms
        that evict perfectly well, and scoring it against zero reports failures
        that did not occur. Getting this wrong is the specific mistake this
        fixture exists to prevent recurring."""
        arms = {f"hopfield th={i}": {"c_evict": v} for i, v in enumerate(hop)}
        arms.update({f"kuramoto tol={i}": {"c_evict": v}
                     for i, v in enumerate(kur)})
        return {"arms": arms, "params": {"c_max": c_max}}

    @staticmethod
    def primary(e):
        return {k: v for k, v in e.items() if not v.get("sensitivity")}

    def test_both_families_are_scored_separately(self):
        e = eviction_claim(self.data([0.4, 0.5, 0.45], [0.3, 0.35, 0.32]),
                           tol=3.0)
        self.assertEqual(len(self.primary(e)), 2)

    def test_the_ladder_ceiling_is_marked_as_a_sensitivity_not_a_claim(self):
        """Both ceilings are shown, but only the declared budget is a claim.
        Counting the sensitivity rows in the tally would inflate the score with
        a presentation choice."""
        e = eviction_claim(self.data([0.4, 0.5, 0.45], [0.3, 0.35, 0.32]),
                           tol=3.0)
        self.assertEqual(len(e), 4)
        self.assertEqual(sum(1 for v in e.values() if v.get("sensitivity")), 2)
        for k, v in e.items():
            self.assertEqual(v.get("sensitivity"), "search ladder" in k, msg=k)

    def test_evicting_far_inside_budget_is_not_exercised(self):
        """c_evict around 0.4 against a ceiling of 50 is two decades of room,
        and the arms barely differ. The condition could not have failed."""
        e = self.primary(eviction_claim(
            self.data([0.40, 0.42, 0.44], [0.30, 0.31, 0.32]), tol=3.0))
        for name, h in e.items():
            self.assertTrue(h["holds"], msg=name)
            self.assertFalse(h["exercised"], msg=name)

    def test_an_arm_that_never_evicts_fails_the_condition(self):
        """`c_evict = inf` is the real failure mode — the argmax never leaves
        the ordered point. It must read as CROSSES, not as a large margin."""
        e = self.primary(eviction_claim(
            self.data([0.4, float("inf")], [0.3, 0.35]), tol=3.0))
        hop = [v for k, v in e.items() if "hopfield" in k][0]
        self.assertFalse(hop["holds"])

    def test_evicting_only_above_budget_fails(self):
        e = self.primary(eviction_claim(self.data([0.4, 80.0], [0.3, 0.35]),
                                        tol=3.0))
        hop = [v for k, v in e.items() if "hopfield" in k][0]
        self.assertFalse(hop["holds"])

    def test_a_family_with_one_arm_is_not_scored(self):
        """One arm has no spread, so 'it held' says nothing at all."""
        e = self.primary(eviction_claim(self.data([0.4], [0.3, 0.35]), tol=3.0))
        self.assertEqual(len(e), 1)
        self.assertTrue(any("kuramoto" in k for k in e))

    def test_arms_without_the_margin_key_are_dropped(self):
        """The published anatomy.json stored only the boolean. That must come
        back empty — reported as UNAVAILABLE — and never as a pass."""
        d = {"arms": {"hopfield th=0": {"evicted": True},
                      "hopfield th=1": {"evicted": True}}}
        self.assertEqual(eviction_claim(d, tol=3.0), {})

    def test_no_arms_yields_nothing(self):
        self.assertEqual(eviction_claim({}, tol=3.0), {})


if __name__ == "__main__":
    unittest.main()
