"""Tests for the observer protocol's analysis.

Same posture as the survival tests: the instrument must not flatter. A null
must read as null, a planted effect must be found, and the sample-size table
must not quietly understate what a test can see.
"""

from __future__ import annotations

import json

import numpy as np

import os
import sys
import pathlib
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.n3_observer import (
    load_log,
    power_n,
    two_prop_permutation,
    _z,
)


# ------------------------------------------------------------------ power


class TestObserver(unittest.TestCase):

    def test_z_matches_known_quantiles(self):
        self.assertAlmostEqual(_z(0.975), 1.959964, delta=1e-4)
        self.assertAlmostEqual(_z(0.80), 0.841621, delta=1e-4)
        self.assertAlmostEqual(_z(0.50), 0.0, delta=1e-6)


    def test_smaller_effects_need_more_trials(self):
        ns = [power_n(0.40, d) for d in (0.30, 0.20, 0.15, 0.10, 0.05)]
        assert ns == sorted(ns)


    def test_power_scales_roughly_as_one_over_delta_squared(self):
        """Halving the effect should cost about four times the trials."""
        n10 = power_n(0.40, 0.10)
        n05 = power_n(0.40, 0.05)
        assert 3.0 < n05 / n10 < 5.0


    def test_power_rejects_an_impossible_shift(self):
        assert power_n(0.90, 0.20) == -1


    def test_power_is_high_enough_to_be_honest(self):
        """A +30 point shift at 40% baseline must need more than a casual handful —
        the table exists precisely to stop twenty trials being read as evidence."""
        assert power_n(0.40, 0.30) > 20


    # ------------------------------------------------------- permutation test


    def test_no_effect_reads_as_no_effect(self):
        rng = np.random.default_rng(0)
        n = 400
        cond = rng.random(n) < 0.5
        surv = rng.random(n) < 0.4          # independent of condition
        r = two_prop_permutation(surv, cond, n_perm=2000, seed=0)
        assert r["p_value"] > 0.05
        assert abs(r["diff"]) < 0.15


    def test_a_planted_effect_is_found(self):
        rng = np.random.default_rng(1)
        n = 400
        cond = rng.random(n) < 0.5
        surv = np.where(cond, rng.random(n) < 0.75, rng.random(n) < 0.35)
        r = two_prop_permutation(surv, cond, n_perm=2000, seed=0)
        assert r["p_value"] < 0.01
        assert r["diff"] > 0.25


    def test_direction_of_the_difference_is_attend_minus_ignore(self):
        cond = np.array([True, True, False, False])
        surv = np.array([True, True, False, False])
        r = two_prop_permutation(surv, cond, n_perm=200, seed=0)
        self.assertAlmostEqual(r["diff"], 1.0, places=7)
        self.assertAlmostEqual(r["p_attend"], 1.0, places=7)
        self.assertAlmostEqual(r["p_ignore"], 0.0, places=7)


    def test_single_arm_is_reported_not_crashed(self):
        cond = np.array([True, True, True])
        surv = np.array([True, False, True])
        r = two_prop_permutation(surv, cond, n_perm=100, seed=0)
        assert np.isnan(r["p_value"])
        assert r["n_attend"] == 3 and r["n_ignore"] == 0


    def test_permutation_p_is_never_zero(self):
        """A p of exactly zero would overstate certainty; the +1 correction matters."""
        cond = np.array([True] * 30 + [False] * 30)
        surv = np.array([True] * 30 + [False] * 30)
        r = two_prop_permutation(surv, cond, n_perm=500, seed=0)
        assert r["p_value"] > 0.0


    def test_permutation_is_two_sided(self):
        """An effect in either direction must be detected equally."""
        rng = np.random.default_rng(2)
        n = 300
        cond = rng.random(n) < 0.5
        up = np.where(cond, rng.random(n) < 0.75, rng.random(n) < 0.35)
        down = np.where(cond, rng.random(n) < 0.35, rng.random(n) < 0.75)
        a = two_prop_permutation(up, cond, n_perm=1000, seed=0)
        b = two_prop_permutation(down, cond, n_perm=1000, seed=0)
        assert a["p_value"] < 0.01 and b["p_value"] < 0.01
        assert a["diff"] > 0 and b["diff"] < 0


    def test_false_positive_rate_is_near_alpha(self):
        """Run the whole test repeatedly on null data: about 5% should trip."""
        trips = 0
        trials = 60
        for s in range(trials):
            rng = np.random.default_rng(1000 + s)
            n = 200
            cond = rng.random(n) < 0.5
            surv = rng.random(n) < 0.4
            r = two_prop_permutation(surv, cond, n_perm=400, seed=s)
            trips += int(r["p_value"] < 0.05)
        assert trips <= 0.20 * trials


    # --------------------------------------------------------------- log I/O


    def test_load_log_accepts_a_bare_list(self):
        import tempfile
        with tempfile.TemporaryDirectory() as _td:
            tmp_path = pathlib.Path(_td)
            p = tmp_path / "t.json"
            p.write_text(json.dumps([{"sham": False, "attend": True, "survived": True}]))
            out = load_log(p)
            assert out == [{"sham": False, "attend": True, "survived": True}]


    def test_load_log_accepts_a_wrapped_object_and_ignores_extras(self):
        import tempfile
        with tempfile.TemporaryDirectory() as _td:
            tmp_path = pathlib.Path(_td)
            p = tmp_path / "t.json"
            p.write_text(json.dumps({"meta": {"budget": 900},
                                     "trials": [{"sham": True, "attend": False,
                                                 "survived": False, "deathStep": 310}]}))
            out = load_log(p)
            assert out == [{"sham": True, "attend": False, "survived": False}]


if __name__ == "__main__":
    unittest.main()
