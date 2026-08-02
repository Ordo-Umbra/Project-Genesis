"""Tests for the dictionary-free distinction reading.

The construction claims its resolution comes from the system's own noise rather
than from a chosen cut. That is checkable against ground truth: build a system
with a known number of genuinely distinct causal states and see whether the
count comes back. The first block does that, and it is the test that matters —
if the count is wrong on a system whose answer we know, nothing measured on a
real substrate means anything.

The second block pins the two properties the experiment had to disclose rather
than fix: more replicates resolve finer distinctions, and the horizon changes
the answer. Both are recorded here as known behaviour so that a later change
which quietly removes them shows up as a failing test rather than as a
surprising result.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.causal_state import (  # noqa: E402
    causal_delta_c,
    causal_distinction,
    causal_partition,
    future_ensemble,
    null_calibration,
    separation,
)


def attractors(k, noise, n_states=24, dim=8, spread=3.0):
    """`k` well-separated attractors, `n_states` samples cycling through them.

    Ground truth: samples sharing an attractor ARE the same causal state, so a
    correct construction returns exactly `k` classes whatever the noise.
    """
    centres = np.eye(k, dim) * spread
    assign = np.arange(n_states) % k
    states = [centres[a] for a in assign]

    def evolve(s, steps, rng):
        return s + noise * rng.standard_normal(s.size)

    return states, evolve, (lambda v: v), assign


class TestRecoversKnownCausalStates(unittest.TestCase):

    def test_the_count_is_exact_for_one_through_five_attractors(self):
        for k in (1, 2, 3, 5):
            s, ev, em, _ = attractors(k, 0.3)
            r = causal_delta_c(s, ev, em, np.random.default_rng(0), tau=1,
                               n_replicates=12)
            self.assertEqual(r["n_causal"], k, msg=f"k={k}")

    def test_the_count_is_unchanged_by_the_noise_amplitude(self):
        """The property the threshold reading lacked: a uniform noise floor
        must not move the answer."""
        for noise in (0.1, 0.3, 1.0, 2.0):
            s, ev, em, _ = attractors(3, noise)
            r = causal_delta_c(s, ev, em, np.random.default_rng(1), tau=1,
                               n_replicates=12)
            self.assertEqual(r["n_causal"], 3, msg=f"noise={noise}")

    def test_delta_c_matches_the_true_partition_entropy(self):
        for k in (2, 3, 5):
            s, ev, em, truth = attractors(k, 0.5)
            r = causal_delta_c(s, ev, em, np.random.default_rng(2), tau=1,
                               n_replicates=12)
            p = np.bincount(truth) / truth.size
            want = float(-(p * np.log2(p)).sum() / np.log2(truth.size))
            self.assertAlmostEqual(r["delta_c"], want, places=6, msg=f"k={k}")

    def test_indistinguishable_states_are_merged(self):
        """Centres far closer than the noise resolves: one causal state."""
        s, ev, em, _ = attractors(4, 1.0, spread=0.05)
        r = causal_delta_c(s, ev, em, np.random.default_rng(3), tau=1,
                           n_replicates=12)
        self.assertEqual(r["n_causal"], 1)
        self.assertAlmostEqual(r["delta_c"], 0.0, places=9)


class TestSeparationStatistic(unittest.TestCase):

    def test_separation_is_dimensionless(self):
        """Rescaling the state space must not change distinguishability."""
        rng = np.random.default_rng(4)
        a = rng.standard_normal((12, 6))
        b = rng.standard_normal((12, 6)) + 2.0
        self.assertAlmostEqual(separation(a, b), separation(100 * a, 100 * b),
                               places=9)

    def test_separation_grows_with_the_gap(self):
        rng = np.random.default_rng(5)
        a = rng.standard_normal((16, 4))
        vals = [separation(a, rng.standard_normal((16, 4)) + d)
                for d in (0.0, 1.0, 4.0)]
        self.assertTrue(all(x < y for x, y in zip(vals, vals[1:])), msg=str(vals))

    def test_separation_needs_at_least_two_replicates(self):
        self.assertTrue(np.isnan(separation(np.zeros((1, 3)), np.ones((4, 3)))))

    def test_the_null_is_centred_where_same_state_pairs_land(self):
        rng = np.random.default_rng(6)
        ens = [rng.standard_normal((16, 5)) + 5.0 * i for i in range(4)]
        null = null_calibration(ens, rng, n_draws=200)
        self.assertGreater(null.size, 100)
        # a same-state split should look nothing like a genuine 5-sigma gap
        self.assertLess(float(np.median(null)), separation(ens[0], ens[1]))

    def test_future_ensemble_has_one_row_per_replicate(self):
        e = future_ensemble(np.zeros(4), lambda s, n, r: s + r.standard_normal(4),
                            lambda v: v, 1, 9, np.random.default_rng(7))
        self.assertEqual(e.shape, (9, 4))


class TestPartitionMechanics(unittest.TestCase):

    def test_an_infinite_cutoff_merges_everything(self):
        rng = np.random.default_rng(8)
        ens = [rng.standard_normal((10, 3)) + 9.0 * i for i in range(5)]
        self.assertEqual(len(set(causal_partition(ens, np.inf).tolist())), 1)

    def test_a_zero_cutoff_merges_nothing(self):
        rng = np.random.default_rng(9)
        ens = [rng.standard_normal((10, 3)) for _ in range(5)]
        self.assertEqual(len(set(causal_partition(ens, 0.0).tolist())), 5)

    def test_distinction_is_zero_for_one_class_and_one_for_all_distinct(self):
        self.assertAlmostEqual(
            causal_distinction(np.zeros(16, dtype=int))["delta_c"], 0.0,
            places=12)
        self.assertAlmostEqual(
            causal_distinction(np.arange(16))["delta_c"], 1.0, places=12)

    def test_average_linkage_does_not_chain_two_classes_together(self):
        """Single linkage would collapse these; average linkage must not."""
        rng = np.random.default_rng(10)
        left = [rng.standard_normal((14, 2)) * 0.1 + [0.0, 0] for _ in range(3)]
        right = [rng.standard_normal((14, 2)) * 0.1 + [6.0, 0] for _ in range(3)]
        bridge = [rng.standard_normal((14, 2)) * 0.1 + [3.0, 0]]
        labels = causal_partition(left + bridge + right, cutoff=8.0)
        self.assertGreater(len(set(labels.tolist())), 1)


class TestDisclosedLimitations(unittest.TestCase):
    """Behaviour the experiment reports rather than fixes, pinned so a later
    change that removes it is visible instead of silent."""

    def test_more_replicates_resolve_finer_distinctions(self):
        counts = []
        for rep in (6, 12, 30):
            s, ev, em, _ = attractors(4, 1.0, spread=0.35)
            counts.append(causal_delta_c(s, ev, em, np.random.default_rng(11),
                                         tau=1, n_replicates=rep)["n_causal"])
        self.assertLessEqual(counts[0], counts[-1])
        self.assertGreater(counts[-1], counts[0])

    def test_the_horizon_changes_the_answer(self):
        """The finding: causal ΔC has no horizon-independent value. Here a
        diffusing system forgets its starting point, so distinctions vanish as
        the horizon grows."""
        def drift(s, steps, rng):
            out = s.copy()
            for _ in range(steps):
                out = out + 1.2 * rng.standard_normal(out.size)
            return out
        centres = np.eye(3, 5) * 2.0
        states = [centres[i % 3] for i in range(18)]
        counts = [causal_delta_c(states, drift, lambda v: v,
                                 np.random.default_rng(12), tau=t,
                                 n_replicates=12)["n_causal"]
                  for t in (1, 20, 200)]
        self.assertGreater(counts[0], counts[-1], msg=str(counts))


if __name__ == "__main__":
    unittest.main()
