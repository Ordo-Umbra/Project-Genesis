"""Checks for the continual-learning substrate (continual_learning)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.continual_learning import (  # noqa: E402
    ConstructiveKappaSGD,
    KappaSGD,
    MLP,
    accuracy,
    make_concept_tasks,
    make_split_pair,
    make_task,
    make_teacher,
    permute_task,
    train_task,
)


def _setup(seed=0, dim=16, n=400):
    rng = np.random.default_rng(seed)
    teacher = make_teacher(rng, dim)
    x, y = make_task(rng, teacher, n, dim)
    model = MLP(dim, hidden=24, rng=rng)
    return rng, model, x, y


class TestModel(unittest.TestCase):

    def test_gradients_match_finite_differences(self):
        rng, model, x, y = _setup(seed=1, n=20)
        _, grads = model.loss_and_grads(x, y)
        eps = 1e-6
        for name in ("w1", "b1", "w2", "b2"):
            p = model.params[name]
            idx = tuple(0 for _ in p.shape)
            p[idx] += eps
            up, _ = model.loss_and_grads(x, y)
            p[idx] -= 2 * eps
            down, _ = model.loss_and_grads(x, y)
            p[idx] += eps
            numeric = (up - down) / (2 * eps)
            self.assertAlmostEqual(grads[name][idx], numeric, places=6,
                                   msg=f"gradient mismatch in {name}")

    def test_learns_a_single_task(self):
        rng, model, x, y = _setup(seed=2)
        opt = KappaSGD(model, lr=0.5)          # plain SGD path
        train_task(model, opt, x, y, epochs=30, batch=32, rng=rng)
        self.assertGreater(accuracy(model, x, y), 0.9)

    def test_labels_roughly_balanced(self):
        rng = np.random.default_rng(3)
        teacher = make_teacher(rng, 16)
        _, y = make_task(rng, teacher, 500, 16)
        self.assertGreater(y.mean(), 0.4)
        self.assertLess(y.mean(), 0.6)

    def test_permuted_task_same_labels_new_geometry(self):
        rng = np.random.default_rng(4)
        teacher = make_teacher(rng, 16)
        x, y = make_task(rng, teacher, 100, 16)
        xp, yp = permute_task(x, y, rng)
        np.testing.assert_array_equal(y, yp)
        self.assertFalse(np.allclose(x, xp))
        np.testing.assert_allclose(np.sort(x, axis=1), np.sort(xp, axis=1))


class TestKappaSGD(unittest.TestCase):

    def test_disabled_capacity_is_plain_sgd(self):
        rng, model, x, y = _setup(seed=5)
        twin = MLP(16, hidden=24, rng=np.random.default_rng(99))
        for k in model.params:
            twin.params[k] = model.params[k].copy()
        _, grads = model.loss_and_grads(x, y)
        KappaSGD(model, lr=0.3).step(grads)
        for k, g in grads.items():
            twin.params[k] -= 0.3 * g
        for k in model.params:
            np.testing.assert_allclose(model.params[k], twin.params[k])

    def test_training_consumes_capacity(self):
        rng, model, x, y = _setup(seed=6)
        opt = KappaSGD(model, lr=0.5, recovery=0.0, consumption=2.0)
        self.assertEqual(opt.mean_kappa(), 1.0)
        train_task(model, opt, x, y, epochs=10, batch=32, rng=rng)
        self.assertLess(opt.mean_kappa(), 0.6)

    def test_recovery_restores_capacity_between_tasks(self):
        rng, model, x, y = _setup(seed=7)
        opt = KappaSGD(model, lr=0.5, recovery=0.5, consumption=2.0)
        train_task(model, opt, x, y, epochs=10, batch=32, rng=rng)
        drained = opt.mean_kappa()
        # idle steps: zero gradients → pure recovery
        zero = {k: np.zeros_like(v) for k, v in model.params.items()}
        for _ in range(30):
            opt.step(zero)
        self.assertGreater(opt.mean_kappa(), drained)

    def test_write_once_limit_blocks_new_learning(self):
        # r = 0: task A drains the soil for good; task B cannot root
        rng, model, x, y = _setup(seed=8)
        opt = KappaSGD(model, lr=0.5, recovery=0.0, consumption=8.0)
        train_task(model, opt, x, y, epochs=20, batch=32, rng=rng)
        acc_a = accuracy(model, x, y)
        xb, yb = permute_task(x, y, rng)
        train_task(model, opt, xb, yb, epochs=20, batch=32, rng=rng)
        self.assertGreater(accuracy(model, x, y), acc_a - 0.1)   # A retained
        self.assertLess(accuracy(model, xb, yb), 0.8)            # B blocked

    def test_fast_recovery_allows_overwrite(self):
        rng, model, x, y = _setup(seed=8)
        opt = KappaSGD(model, lr=0.5, recovery=2.0, consumption=8.0)
        train_task(model, opt, x, y, epochs=20, batch=32, rng=rng)
        xb, yb = permute_task(x, y, rng)
        train_task(model, opt, xb, yb, epochs=20, batch=32, rng=rng)
        self.assertGreater(accuracy(model, xb, yb), 0.85)        # B learned

    def test_split_tasks_drain_soil_selectively(self):
        # structured interference: training A must drain the w1 rows of A's
        # informative half far more than the noise half — the heterogeneity
        # that lets capacity gating do what a global learning rate cannot.
        # (r > 0: selectivity lives in the steady state κ_ss = r/(r+c·load);
        # at r = 0 anything with nonzero load eventually drains to zero.)
        rng = np.random.default_rng(11)
        dim = 16
        (xa, ya), _ = make_split_pair(rng, dim, 400)
        model = MLP(dim, hidden=24, rng=rng)
        opt = KappaSGD(model, lr=0.5, recovery=0.3, consumption=4.0)
        train_task(model, opt, xa, ya, epochs=15, batch=32, rng=rng)
        half = dim // 2
        kappa_active = float(opt.kappa["w1"][:half].mean())
        kappa_idle = float(opt.kappa["w1"][half:].mean())
        self.assertLess(kappa_active, kappa_idle - 0.1)

    def test_split_pair_labels_balanced_and_learnable(self):
        rng = np.random.default_rng(12)
        (xa, ya), (xb, yb) = make_split_pair(rng, 16, 500)
        for y in (ya, yb):
            self.assertGreater(y.mean(), 0.4)
            self.assertLess(y.mean(), 0.6)
        model = MLP(16, hidden=24, rng=rng)
        opt = KappaSGD(model, lr=0.5)
        train_task(model, opt, xa, ya, epochs=25, batch=32, rng=rng)
        self.assertGreater(accuracy(model, xa, ya), 0.85)

    def test_concept_tasks_are_compositional(self):
        rng = np.random.default_rng(13)
        x, y_a, y_b, y_c = make_concept_tasks(rng, 16, 600)
        np.testing.assert_array_equal(y_c, y_a ^ y_b)          # C is XOR(A,B)
        for y in (y_a, y_b, y_c):
            self.assertGreater(y.mean(), 0.4)
            self.assertLess(y.mean(), 0.6)
        # the concepts are genuinely distinct labelings
        self.assertGreater(np.mean(y_a != y_b), 0.3)

    def test_foundations_help_the_composite(self):
        # transfer sanity: C after A,B beats C from scratch at matched epochs
        rng = np.random.default_rng(14)
        x, y_a, y_b, y_c = make_concept_tasks(rng, 16, 800)
        xtr, xte = x[:600], x[600:]

        cold = MLP(16, hidden=32, rng=np.random.default_rng(20))
        opt_cold = KappaSGD(cold, lr=0.5)
        train_task(cold, opt_cold, xtr, y_c[:600], epochs=15, batch=32, rng=rng)

        warm = MLP(16, hidden=32, rng=np.random.default_rng(20))
        opt_warm = KappaSGD(warm, lr=0.5)
        train_task(warm, opt_warm, xtr, y_a[:600], epochs=15, batch=32, rng=rng)
        train_task(warm, opt_warm, xtr, y_b[:600], epochs=15, batch=32, rng=rng)
        train_task(warm, opt_warm, xtr, y_c[:600], epochs=15, batch=32, rng=rng)

        self.assertGreater(accuracy(warm, xte, y_c[600:]),
                           accuracy(cold, xte, y_c[600:]) - 0.02)

    def test_constructive_first_task_is_plain_sgd(self):
        # fresh net: d = 0 everywhere → nothing destructive → exact SGD step
        rng, model, x, y = _setup(seed=15)
        twin = MLP(16, hidden=24, rng=np.random.default_rng(99))
        for k in model.params:
            twin.params[k] = model.params[k].copy()
        _, grads = model.loss_and_grads(x, y)
        opt = ConstructiveKappaSGD(model, lr=0.3, recovery=0.1, consumption=4.0)
        opt.step(grads)
        for k, g in grads.items():
            twin.params[k] -= 0.3 * g
            np.testing.assert_allclose(model.params[k], twin.params[k])
        self.assertEqual(opt.mean_kappa(), 1.0)   # no destructive load yet

    def test_destructive_updates_are_gated_and_consume(self):
        rng, model, x, y = _setup(seed=16)
        opt = ConstructiveKappaSGD(model, lr=0.3, recovery=0.0, consumption=8.0)
        # commit: displace w1 positively from anchor
        opt.anchors = {k: v.copy() for k, v in model.params.items()}
        model.params["w1"] += 1.0
        # a gradient pushing w1 back toward the anchor is destructive
        grads = {k: np.zeros_like(v) for k, v in model.params.items()}
        grads["w1"] = np.ones_like(model.params["w1"])   # g·d > 0
        before = model.params["w1"].copy()
        opt.step(grads)
        first_move = before - model.params["w1"]
        self.assertTrue(np.all(first_move > 0))          # still moves (κ=1 first)
        for _ in range(5):
            opt.step(grads)
        drained = opt.kappa["w1"].mean()
        self.assertLess(drained, 0.2)                     # destructive load consumed κ
        before = model.params["w1"].copy()
        opt.step(grads)
        late_move = before - model.params["w1"]
        self.assertTrue(np.all(late_move < 0.2 * first_move))  # gated hard

    def test_constructive_updates_stay_free_while_committed(self):
        rng, model, x, y = _setup(seed=17)
        opt = ConstructiveKappaSGD(model, lr=0.3, recovery=0.0, consumption=8.0)
        opt.anchors = {k: v.copy() for k, v in model.params.items()}
        model.params["w1"] += 1.0
        # a gradient pushing w1 FURTHER from anchor is constructive: g·d < 0
        grads = {k: np.zeros_like(v) for k, v in model.params.items()}
        grads["w1"] = -np.ones_like(model.params["w1"])
        for _ in range(6):
            opt.step(grads)
        self.assertAlmostEqual(float(opt.kappa["w1"].mean()), 1.0, places=6)

    def test_task_boundaries_do_not_erase_commitments(self):
        # the v1 failure mode: new_task() must NOT move the anchor
        rng, model, x, y = _setup(seed=18)
        opt = ConstructiveKappaSGD(model, lr=0.5, recovery=0.1, consumption=4.0)
        init_anchor = opt.anchors["w1"].copy()
        train_task(model, opt, x, y, epochs=5, batch=32, rng=rng)
        opt.new_task()
        np.testing.assert_allclose(opt.anchors["w1"], init_anchor)
        # explicit reanchor is available for deliberate resets
        opt.reanchor()
        np.testing.assert_allclose(opt.anchors["w1"], model.params["w1"])

    def test_kappa_stays_in_range(self):
        rng, model, x, y = _setup(seed=9)
        opt = KappaSGD(model, lr=0.5, recovery=0.3, consumption=3.0)
        train_task(model, opt, x, y, epochs=5, batch=32, rng=rng)
        for arr in opt.kappa.values():
            self.assertTrue(np.all(arr >= 0.0))
            self.assertTrue(np.all(arr <= 1.0 + 1e-12))


if __name__ == "__main__":
    unittest.main()
