"""Tests for the survival instrument.

The point of this module is to keep an analyst honest, so the tests are mostly
about the instrument refusing to flatter: a null input must read as null, a
censored run must not be silently dropped, and the look-elsewhere correction
must actually cost something.
"""

from __future__ import annotations

import numpy as np

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.multiphase import step_multiphase
from project_genesis.survival import (
    FEATURE_NAMES,
    bootstrap_hazard_ratio,
    confirm_feature,
    held_out_auc,
    kaplan_meier,
    max_feature_permutation,
    multiplicity_batch,
    permutation_p,
    relaxation_time,
    roc_auc,
    run_ensemble,
    sector_labels_batch,
    seed_batch,
    spearman_r,
    split_hazard,
    step_batch,
    window_features,
)


# --------------------------------------------------------------------- kernel

def _exponential_deaths(seed, n=4000, scale=400.0, horizon=2000):
    rng = np.random.default_rng(seed)
    draws = rng.exponential(scale, n)
    death = np.where(draws < horizon, np.ceil(draws).astype(int), -1)
    return death, death < 0


class TestSurvival(unittest.TestCase):

    def test_batched_step_matches_repo_kernel_exactly(self):
        """One law, one code path: the batched step must be the repo's step."""
        rng = np.random.default_rng(0)
        fields = seed_batch(4, 3, 6, rng, ndim=3)
        batched = step_batch(fields, gamma=1.5, dt=0.06, ndim=3)
        for r in range(fields.shape[0]):
            single = step_multiphase(fields[r], gamma=1.5, dt=0.06)
            assert np.array_equal(batched[r], single)


    def test_batched_step_matches_repo_kernel_in_2d(self):
        rng = np.random.default_rng(1)
        fields = seed_batch(3, 4, 8, rng, ndim=2)
        batched = step_batch(fields, gamma=1.2, dt=0.1, ndim=2)
        for r in range(fields.shape[0]):
            assert np.array_equal(batched[r], step_multiphase(fields[r], gamma=1.2, dt=0.1))


    def test_seed_batch_is_on_the_unit_sphere(self):
        rng = np.random.default_rng(2)
        fields = seed_batch(5, 4, 6, rng, ndim=3)
        norms = np.sqrt((fields ** 2).sum(axis=1))
        assert np.allclose(norms, 1.0, atol=1e-6)


    def test_replicas_are_independent(self):
        """Different replicas must not be copies of one another."""
        rng = np.random.default_rng(3)
        fields = seed_batch(4, 3, 6, rng, ndim=3)
        for r in range(1, 4):
            assert not np.allclose(fields[0], fields[r])


    # ------------------------------------------------------------- classification


    def test_multiplicity_of_a_uniform_field_is_one(self):
        labels = np.zeros((2, 5, 5, 5), dtype=int)
        m = multiplicity_batch(labels, 3, 3)
        assert (m == 1).all()


    def test_multiplicity_counts_a_two_sector_interface(self):
        labels = np.zeros((1, 6, 6, 6), dtype=int)
        labels[0, 3:] = 1
        m = multiplicity_batch(labels, 2, 3)[0]
        # sites adjacent to the interface see both sectors, deep bulk sees one
        assert m.max() == 2
        assert m.min() == 1


    def test_multiplicity_never_exceeds_palette(self):
        rng = np.random.default_rng(4)
        labels = rng.integers(0, 4, (2, 6, 6, 6))
        m = multiplicity_batch(labels, 4, 3)
        assert m.max() <= 4
        assert m.min() >= 1


    def test_sector_labels_pick_the_largest_component(self):
        fields = np.zeros((1, 3, 2, 2, 2))
        fields[0, 2] = 5.0
        assert (sector_labels_batch(fields) == 2).all()


    def test_window_features_returns_all_names_per_replica(self):
        rng = np.random.default_rng(5)
        fields = seed_batch(6, 4, 6, rng, ndim=3)
        feats = window_features(fields, 4, 3)
        assert set(feats) == set(FEATURE_NAMES)
        for name in FEATURE_NAMES:
            assert feats[name].shape == (6,)
            assert np.isfinite(feats[name]).all()


    # -------------------------------------------------------------------- ensemble


    def test_run_ensemble_records_every_run(self):
        res = run_ensemble(6, 3, 6, 40, ndim=3, noise=0.0, seed=0, early=10, sample=10)
        assert len(res["death"]) == 6
        assert len(res["censored"]) == 6
        # censored is exactly "no death recorded"
        assert np.array_equal(res["censored"], res["death"] < 0)


    def test_censored_runs_are_kept_not_dropped(self):
        """A run that outlives the horizon is evidence, not missing data."""
        res = run_ensemble(4, 4, 6, 20, ndim=3, noise=0.0, seed=1, early=10, sample=10)
        assert len(res["death"]) == 4          # nothing removed regardless of outcome


    def test_features_captured_at_the_requested_step(self):
        res = run_ensemble(3, 3, 6, 50, ndim=3, noise=0.0, seed=2, early=20, sample=10)
        for name in FEATURE_NAMES:
            assert res["features"][name].shape == (3,)


    def test_a_uniform_start_dies_immediately(self):
        """Sanity on the death rule: a field already dominated by one sector is dead."""
        res = run_ensemble(2, 3, 6, 30, ndim=3, noise=0.0, seed=3, early=5, sample=5,
                           death_frac=0.0)
        assert (res["death"] > 0).all()


    # ------------------------------------------------------------------- survival


    def test_kaplan_meier_starts_at_one_and_decreases(self):
        death = np.array([10, 20, 30, -1, -1])
        cens = death < 0
        t, s = kaplan_meier(death, cens, 100)
        self.assertAlmostEqual(s[0], 1.0, places=7)
        assert np.all(np.diff(s) <= 1e-12)


    def test_kaplan_meier_with_no_deaths_is_flat(self):
        death = np.array([-1, -1, -1])
        t, s = kaplan_meier(death, death < 0, 100)
        assert np.allclose(s, 1.0)


    def test_split_hazard_is_unbiased_for_a_constant_rate_process(self):
        """A memoryless process must read as memoryless — the key null."""
        ratios = [split_hazard(*_exponential_deaths(s), 400, 2000)["ratio"]
                  for s in range(24)]
        self.assertAlmostEqual(np.mean(ratios), 1.0, delta=0.05)


    def test_hazard_ratio_ci_covers_one_for_a_memoryless_process(self):
        """A 95% CI promises coverage over repeated draws, not on any single one,
        so that is what is asserted — checking one draw would be flaky by design."""
        covered = 0
        trials = 20
        for s in range(trials):
            death, cens = _exponential_deaths(s)
            lo, hi = bootstrap_hazard_ratio(death, cens, 400, 2000, n_boot=200, seed=s)
            covered += int(lo <= 1.0 <= hi)
        assert covered >= 0.80 * trials


    def test_split_hazard_detects_a_falling_rate(self):
        """Everything dies early, nothing dies late -> ratio well below one."""
        death = np.concatenate([np.full(60, 50), np.full(40, -1)])
        cens = death < 0
        h = split_hazard(death, cens, 200, 2000)
        assert h["late_deaths"] == 0
        self.assertAlmostEqual(h["ratio"], 0.0, places=7)


    def test_hazard_exposure_only_counts_time_actually_at_risk(self):
        death = np.array([100, -1])
        h = split_hazard(death, death < 0, 500, 1000)
        # run 1 contributes 100 steps to the early window, run 2 contributes 500
        self.assertAlmostEqual(h["early_exposure"], 600.0, places=7)


    # ----------------------------------------------------------------- statistics


    def test_spearman_of_a_monotone_map_is_one(self):
        x = np.arange(20, dtype=float)
        self.assertAlmostEqual(spearman_r(x, x ** 3), 1.0, places=7)
        self.assertAlmostEqual(spearman_r(x, -x), -1.0, places=7)


    def test_spearman_handles_ties_without_blowing_up(self):
        x = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = spearman_r(x, y)
        assert np.isfinite(r) and 0.0 < r < 1.0


    def test_roc_auc_endpoints(self):
        scores = np.arange(10, dtype=float)
        labels = np.arange(10) >= 5
        self.assertAlmostEqual(roc_auc(scores, labels), 1.0, places=7)
        self.assertAlmostEqual(roc_auc(-scores, labels), 0.0, places=7)


    def test_roc_auc_of_a_constant_score_is_one_half(self):
        self.assertAlmostEqual(roc_auc(np.ones(20), np.arange(20) < 10), 0.5, places=7)


    def test_held_out_auc_on_pure_noise_sits_near_chance(self):
        """The instrument must not manufacture signal from noise."""
        rng = np.random.default_rng(11)
        n = 240
        feats = {name: rng.standard_normal(n) for name in FEATURE_NAMES}
        labels = rng.random(n) < 0.5
        auc, per = held_out_auc(feats, labels, seed=0)
        assert 0.30 < auc < 0.70


    def test_held_out_auc_finds_a_planted_signal(self):
        rng = np.random.default_rng(12)
        n = 240
        labels = rng.random(n) < 0.5
        feats = {name: rng.standard_normal(n) for name in FEATURE_NAMES}
        feats["maxfrac"] = labels * 3.0 + rng.standard_normal(n) * 0.5
        auc, per = held_out_auc(feats, labels, seed=0)
        assert auc > 0.85
        assert per["maxfrac"] > 0.85


    def test_permutation_p_is_uninteresting_for_noise(self):
        rng = np.random.default_rng(13)
        n = 160
        feats = {name: rng.standard_normal(n) for name in FEATURE_NAMES}
        labels = rng.random(n) < 0.5
        obs, null_mean, null_95, p = permutation_p(feats, labels, n_perm=200, seed=0)
        assert p > 0.05
        self.assertAlmostEqual(null_mean, 0.5, delta=0.12)


    def test_look_elsewhere_null_is_stricter_than_a_single_feature_null(self):
        """The best of five must be held to a higher bar than one chosen in advance."""
        rng = np.random.default_rng(14)
        n = 200
        feats = {name: rng.standard_normal(n) for name in FEATURE_NAMES}
        labels = rng.random(n) < 0.5
        _, _, max_null_95, _ = max_feature_permutation(feats, labels, n_perm=300, seed=0)
        _, single_null_95, _ = confirm_feature(feats["ell"], labels, n_perm=300, seed=0)
        assert max_null_95 > single_null_95


    def test_max_feature_permutation_reports_the_actual_winner(self):
        rng = np.random.default_rng(15)
        n = 200
        labels = rng.random(n) < 0.5
        feats = {name: rng.standard_normal(n) for name in FEATURE_NAMES}
        feats["junction"] = labels * 4.0 + rng.standard_normal(n) * 0.3
        best, winner, _, p = max_feature_permutation(feats, labels, n_perm=200, seed=0)
        assert winner == "junction"
        assert best > 0.9
        assert p < 0.01


    def test_confirm_feature_rejects_a_null_feature(self):
        rng = np.random.default_rng(16)
        n = 200
        labels = rng.random(n) < 0.5
        obs, null95, p = confirm_feature(rng.standard_normal(n), labels,
                                         n_perm=400, seed=0)
        assert p > 0.05


    def test_confirm_feature_is_direction_free(self):
        """A perfectly anti-correlated predictor is just as informative."""
        labels = np.arange(40) < 20
        values = labels.astype(float) * -1.0
        obs, _, p = confirm_feature(values, labels, n_perm=200, seed=0)
        self.assertAlmostEqual(obs, 1.0, places=7)


    # ------------------------------------------------------- when is it decided


    def test_relaxation_time_finds_the_knee_of_a_rising_curve(self):
        amp = 1.0 - np.exp(-np.arange(600) / 100.0)
        tau = relaxation_time(amp, frac=0.95)
        assert 250 < tau < 350          # -ln(0.05) * 100 ~= 300


    def test_relaxation_time_of_a_flat_curve_is_immediate(self):
        assert relaxation_time(np.ones(200)) == 1


    def test_relaxation_time_handles_an_empty_trace(self):
        assert relaxation_time(np.array([])) == 0


    def test_capture_returns_features_at_each_requested_step(self):
        res = run_ensemble(4, 3, 6, 40, ndim=3, noise=0.0, seed=0, early=10,
                           sample=10, capture=(5, 15, 30))
        assert sorted(res["captured"]) == [5, 15, 30]
        for t, feats in res["captured"].items():
            assert set(feats) == set(FEATURE_NAMES)
            assert feats["maxfrac"].shape == (4,)


    def test_capture_does_not_disturb_the_default_feature_window(self):
        common = dict(ndim=3, noise=0.0, seed=5, early=20, sample=10)
        plain = run_ensemble(4, 3, 6, 40, **common)
        with_cap = run_ensemble(4, 3, 6, 40, capture=(5, 15), **common)
        for name in FEATURE_NAMES:
            assert np.allclose(plain["features"][name], with_cap["features"][name])


    def test_amplitude_tracking_is_opt_in_and_one_value_per_step(self):
        off = run_ensemble(3, 3, 6, 25, ndim=3, noise=0.0, seed=1, early=10, sample=10)
        assert off["amplitude"].size == 0
        on = run_ensemble(3, 3, 6, 25, ndim=3, noise=0.0, seed=1, early=10,
                          sample=10, track_amplitude=True)
        assert on["amplitude"].shape == (25,)
        assert np.isfinite(on["amplitude"]).all()


    def test_early_exit_waits_for_the_last_capture_step(self):
        """All-dead used to break the loop; captures scheduled later must still land."""
        res = run_ensemble(3, 2, 6, 60, ndim=3, noise=0.0, seed=2, early=5,
                           sample=5, death_frac=0.0, capture=(5, 50))
        assert (res["death"] > 0).all()          # everything died immediately
        assert sorted(res["captured"]) == [5, 50]


if __name__ == "__main__":
    unittest.main()
