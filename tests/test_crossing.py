"""Tests for the scale-free crossing reduction.

The whole point of `project_genesis/crossing.py` is that one quantity survives a
change of load units and another does not.  If the tests do not pin that, the
module is just an algebraic restatement of the ladder and proves nothing.  So
the central test here is an **invariance** test: multiply the measured load by
an arbitrary constant — exactly what changing the Kuramoto integrator's time
step does — and check that the predicted crossing capacity is unmoved while the
predicted consumption moves by precisely that factor.

The second thing that must be right is the failure mode.  A free ordered phase
must give a capacity floor of exactly 1 and no eviction at any consumption; if
that came out even slightly soft, the module would report the mechanism's
condition as satisfied everywhere and the Q1 separation would be vacuous.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.crossing import (  # noqa: E402
    capacity,
    capacity_floor,
    consumption_at_crossing,
    crossing_exists,
    kappa_at_crossing,
    kappa_partner,
    measured_crossing,
    optimum_walk,
    two_point_reduction,
)

W, R = 1.5, 0.1


def make_rows(scale: float = 1.0, load_o: float = 0.02, x_key: str = "T"):
    """A scan with the anatomy both transplants report: ordered end high in ΔI
    and low in ΔC, a ΔC peak at the transition, load rising with disorder."""
    x = np.linspace(0.0, 2.0, 21)
    di = 1.0 / (1.0 + np.exp((x - 1.0) / 0.15))
    dc = 0.45 * np.exp(-((x - 1.0) / 0.25) ** 2)
    # the bump is normalised to vanish exactly at the ordered end, so
    # `load_o = 0` really is a free ordered phase rather than a small one
    bump = np.exp(-((x - 1.1) / 0.5) ** 2)
    bump = np.clip((bump - bump[0]) / (1.0 - bump[0]), 0.0, None)
    load = load_o + 0.3 * bump
    return [{x_key: float(a), "delta_c": float(b), "delta_i": float(c),
             "activity": float(d) * scale}
            for a, b, c, d in zip(x, dc, di, load)]


# (ΔI, ΔC) point clouds with a known hull, used where the hull length is the
# thing under test.  FLAT_TWO has no viable intermediate state; FLAT_THREE adds
# one that is well inside the ordered-to-critical chord, which is the shape the
# driven Kuramoto scan turns out to have.
FLAT_TWO = [(1.00, 0.00), (0.80, 0.02), (0.55, 0.10), (0.43, 0.48), (0.20, 0.05)]
FLAT_THREE = [(1.00, 0.00), (0.80, 0.02), (0.68, 0.28), (0.29, 0.44), (0.10, 0.05)]


def flat_rows(points, load: float = 0.05, x_key: str = "T"):
    """A scan in which every point pays the same load — the uniform-tax limit."""
    return [{x_key: float(i), "delta_i": float(di), "delta_c": float(dc),
             "activity": float(load)} for i, (di, dc) in enumerate(points)]


class TestCapacityLaw(unittest.TestCase):

    def test_capacity_is_one_when_nothing_is_consumed(self):
        self.assertAlmostEqual(float(capacity(0.4, 0.0)), 1.0)

    def test_capacity_falls_monotonically_in_u(self):
        k = capacity(0.3, np.array([0.0, 1.0, 10.0, 1000.0]))
        self.assertTrue(np.all(np.diff(k) < 0))

    def test_capacity_matches_the_engine_law_through_u_equals_c_over_r(self):
        """r and c are one parameter, not two — the whole audit rests on this."""
        from project_genesis.hopfield_substrate import steady_state_kappa
        for c, r in ((0.05, 0.1), (3.0, 0.1), (50.0, 0.7), (0.2, 2.0)):
            self.assertAlmostEqual(float(capacity(0.37, c / r)),
                                   steady_state_kappa(0.37, c, r), places=12,
                                   msg=f"c={c} r={r}")

    def test_kappa_partner_is_identity_at_equal_loads(self):
        self.assertAlmostEqual(float(kappa_partner(0.4, 1.0)), 0.4, places=12)

    def test_kappa_partner_vanishes_for_a_free_ordered_phase(self):
        self.assertAlmostEqual(float(kappa_partner(0.4, np.inf)), 0.0)

    def test_kappa_partner_agrees_with_evaluating_the_law_directly(self):
        l_o, l_s, u = 0.02, 0.31, 40.0
        k_o = float(capacity(l_o, u))
        self.assertAlmostEqual(float(kappa_partner(k_o, l_s / l_o)),
                               float(capacity(l_s, u)), places=12)


class TestReduction(unittest.TestCase):

    def test_the_two_points_are_the_ordered_end_and_the_dc_peak(self):
        red = two_point_reduction(make_rows(), x_key="T")
        self.assertEqual(red["i_ordered"], 0)
        self.assertAlmostEqual(red["x_critical"], 1.0, places=6)

    def test_the_gap_is_the_distinction_the_critical_point_offers(self):
        red = two_point_reduction(make_rows(), x_key="T")
        self.assertAlmostEqual(red["gap"], red["dc_s"] - red["dc_o"], places=12)

    def test_rho_is_infinite_when_the_ordered_phase_is_free(self):
        rows = make_rows(load_o=0.0)
        self.assertFalse(np.isfinite(two_point_reduction(rows, x_key="T")["rho"]))

    def test_reduction_reads_whichever_control_parameter_the_scan_used(self):
        red = two_point_reduction(make_rows(x_key="gamma"), x_key="gamma")
        self.assertAlmostEqual(red["x_critical"], 1.0, places=6)


class TestPrecondition(unittest.TestCase):

    def test_the_ordered_point_leading_at_zero_consumption_is_the_precondition(self):
        rows = make_rows()
        red = two_point_reduction(rows, x_key="T")
        self.assertTrue(crossing_exists(red, W))
        # ...and that is exactly "the ordered point is the argmax at c -> 0"
        s = np.array([r["delta_c"] + W * r["delta_i"] for r in rows])
        self.assertEqual(int(np.argmax(s)), red["i_ordered"])

    def test_precondition_fails_when_the_critical_point_already_leads(self):
        red = two_point_reduction(make_rows(), x_key="T")
        self.assertFalse(crossing_exists(red, 0.2))   # w too small to fund order

    def test_no_crossing_is_reported_when_the_precondition_fails(self):
        self.assertTrue(np.isnan(kappa_at_crossing(
            two_point_reduction(make_rows(), x_key="T"), 0.2)))


class TestCrossingLocation(unittest.TestCase):

    def test_predicted_capacity_matches_the_full_argmax(self):
        rows = make_rows()
        pred = kappa_at_crossing(two_point_reduction(rows, x_key="T"), W)
        meas = measured_crossing(rows, x_key="T", weight=W, recovery=R)
        self.assertAlmostEqual(pred, meas["kappa_arrive"], delta=0.01)

    def test_the_optimum_arrives_at_the_dc_peak_in_one_hop(self):
        m = measured_crossing(make_rows(), x_key="T", weight=W, recovery=R)
        self.assertEqual(m["hops"], 1)
        self.assertAlmostEqual(m["x_end"], m["x_critical"], places=9)

    def test_the_crossing_condition_actually_balances_there(self):
        """S_ordered = S_critical at the predicted capacity — the defining equation."""
        red = two_point_reduction(make_rows(), x_key="T")
        k_o = kappa_at_crossing(red, W)
        k_s = float(kappa_partner(k_o, red["rho"]))
        self.assertAlmostEqual(red["dc_o"] + W * k_o * red["di_o"],
                               red["dc_s"] + W * k_s * red["di_s"], places=6)

    def test_a_heavier_integration_weight_delays_the_crossing(self):
        red = two_point_reduction(make_rows(), x_key="T")
        # all above the fixture's precondition threshold w > 0.902; below it
        # there is no crossing at all, which a separate test covers
        ks = [kappa_at_crossing(red, w) for w in (1.0, 2.0, 4.0)]
        self.assertTrue(all(a > b for a, b in zip(ks, ks[1:])),
                        msg=f"expected falling kappa_star, got {ks}")

    def test_the_root_is_unique_so_the_scan_direction_cannot_matter(self):
        """F is concave for ρ>1 and convex for ρ<1, and the precondition puts
        F(0) < 0 < F(1) — so exactly one sign change, always. Checked over a
        wide random sweep because the module relies on it."""
        rng = np.random.default_rng(0)
        k = np.linspace(1e-9, 1.0, 20001)
        for _ in range(2000):
            di_o = rng.uniform(0.05, 1.0)
            di_s = rng.uniform(0.0, di_o)
            w = rng.uniform(0.1, 10.0)
            rho = float(np.exp(rng.uniform(np.log(1e-3), np.log(1e6))))
            gap = rng.uniform(0.0, w * (di_o - di_s))   # precondition holds
            f = w * (k * di_o - kappa_partner(k, rho) * di_s) - gap
            n = int(np.count_nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0))
            self.assertEqual(n, 1, msg=f"di_o={di_o} di_s={di_s} w={w} "
                                       f"rho={rho} gap={gap} -> {n} roots")

    def test_consumption_conversion_inverts_the_capacity_law(self):
        red = two_point_reduction(make_rows(), x_key="T")
        k = kappa_at_crossing(red, W)
        c = consumption_at_crossing(k, red["load_o"], R)
        self.assertAlmostEqual(float(capacity(red["load_o"], c / R)), k, places=10)


class TestUnitInvariance(unittest.TestCase):
    """The finding, as a test: one quantity survives a change of load units."""

    def test_predicted_capacity_is_invariant_under_load_rescaling(self):
        base = kappa_at_crossing(two_point_reduction(make_rows(), x_key="T"), W)
        for scale in (0.01, 0.1, 10.0, 100.0):
            k = kappa_at_crossing(
                two_point_reduction(make_rows(scale=scale), x_key="T"), W)
            self.assertAlmostEqual(k, base, places=10, msg=f"scale {scale}")

    def test_predicted_consumption_moves_by_exactly_the_rescaling_factor(self):
        red0 = two_point_reduction(make_rows(), x_key="T")
        c0 = consumption_at_crossing(kappa_at_crossing(red0, W), red0["load_o"], R)
        for scale in (0.01, 0.1, 10.0, 100.0):
            red = two_point_reduction(make_rows(scale=scale), x_key="T")
            c = consumption_at_crossing(kappa_at_crossing(red, W), red["load_o"], R)
            self.assertAlmostEqual(c * scale, c0, delta=1e-9 + 1e-6 * c0,
                                   msg=f"scale {scale}")

    def test_the_measured_crossing_shows_the_same_split(self):
        """Not just the formula — the dense-ladder argmax behaves the same way."""
        base = measured_crossing(make_rows(), x_key="T", weight=W, recovery=R)
        m = measured_crossing(make_rows(scale=100.0), x_key="T", weight=W,
                              recovery=R)
        self.assertAlmostEqual(m["kappa_arrive"], base["kappa_arrive"], delta=0.005)
        self.assertAlmostEqual(m["c_arrive"] * 100.0, base["c_arrive"],
                               delta=0.02 * base["c_arrive"])

    def test_recovery_rate_only_rescales_the_consumption_axis(self):
        """r is a unit, not a knob: kappa is untouched, c moves in proportion."""
        rows = make_rows()
        a = measured_crossing(rows, x_key="T", weight=W, recovery=0.1)
        b = measured_crossing(rows, x_key="T", weight=W, recovery=0.7)
        self.assertAlmostEqual(a["kappa_arrive"], b["kappa_arrive"], delta=0.005)
        self.assertAlmostEqual(b["c_arrive"] / a["c_arrive"], 7.0, delta=0.05)


class TestFreeOrderedPhase(unittest.TestCase):
    """The mechanism's condition, and it has to be exact, not approximate."""

    def test_a_free_ordered_phase_has_a_capacity_floor_of_exactly_one(self):
        red = two_point_reduction(make_rows(load_o=0.0), x_key="T")
        self.assertEqual(capacity_floor(red, 1e12), 1.0)

    def test_a_free_ordered_phase_is_never_evicted_at_any_consumption(self):
        m = measured_crossing(make_rows(load_o=0.0), x_key="T", weight=W,
                              recovery=R, c_hi=1e12)
        self.assertTrue(np.isnan(m["c_evict"]))
        self.assertEqual(m["hops"], 0)

    def test_a_paying_ordered_phase_is_evicted(self):
        m = measured_crossing(make_rows(load_o=0.02), x_key="T", weight=W,
                              recovery=R)
        self.assertTrue(np.isfinite(m["c_evict"]))

    def test_eviction_happens_exactly_when_the_floor_drops_below_the_crossing(self):
        rows = make_rows(load_o=0.02)
        red = two_point_reduction(rows, x_key="T")
        k_star = kappa_at_crossing(red, W)
        for c_max in (0.01, 0.1, 1.0, 10.0, 100.0):
            floor = capacity_floor(red, c_max / R)
            evicted = np.isfinite(measured_crossing(
                rows, x_key="T", weight=W, recovery=R, c_hi=c_max)["c_arrive"])
            self.assertEqual(floor < k_star, evicted, msg=f"c_max {c_max}")

    def test_the_floor_falls_as_the_ladder_is_extended(self):
        red = two_point_reduction(make_rows(), x_key="T")
        floors = [capacity_floor(red, u) for u in (1.0, 10.0, 100.0, 1e4)]
        self.assertTrue(all(a > b for a, b in zip(floors, floors[1:])))


class TestOptimumWalk(unittest.TestCase):
    """The hull is what the two-point reduction is competing against."""

    def test_hull_matches_a_brute_force_argmax_sweep(self):
        """The definition, checked directly: every argmax of ΔC + λ·ΔI."""
        rng = np.random.default_rng(0)
        lams = np.concatenate([[0.0], np.geomspace(1e-5, 1e5, 20000)])
        for _ in range(40):
            n = int(rng.integers(3, 14))
            di, dc = rng.random(n), rng.random(n)
            rows = [{"delta_i": float(a), "delta_c": float(b)}
                    for a, b in zip(di, dc)]
            hull = [i for i, _, _ in optimum_walk(rows)]
            j = (dc[None, :] + lams[:, None] * di[None, :]).argmax(axis=1)
            brute = [int(j[0])] + [int(b) for a, b in zip(j, j[1:]) if a != b]
            self.assertEqual(hull, brute[::-1])

    def test_hull_runs_from_most_integrated_to_most_distinguished(self):
        rows = make_rows()
        hull = optimum_walk(rows)
        di = [r["delta_i"] for r in rows]
        dc = [r["delta_c"] for r in rows]
        self.assertEqual(hull[0][0], int(np.argmax(di)))
        self.assertEqual(hull[-1][0], int(np.argmax(dc)))

    def test_hull_is_strictly_increasing_in_distinction(self):
        vals = [c for _, _, c in optimum_walk(make_rows())]
        self.assertTrue(all(a < b for a, b in zip(vals, vals[1:])))

    def test_dominated_points_are_never_visited(self):
        """A point with less of both can never be optimal at any capacity."""
        rows = make_rows() + [{"T": 9.0, "delta_c": 0.01, "delta_i": 0.01,
                               "activity": 0.05}]
        self.assertNotIn(len(rows) - 1, [i for i, _, _ in optimum_walk(rows)])

    def test_a_flat_load_makes_the_measured_walk_exactly_the_hull(self):
        """With every point paying the same, κ is one number and the argmax
        traces the hull by definition. This is the only case where the two
        coincide as a theorem rather than as an observation."""
        rows = flat_rows(FLAT_THREE)
        hull_x = [rows[i]["T"] for i, _, _ in optimum_walk(rows)]
        walk = measured_crossing(rows, x_key="T", weight=W, recovery=R)["walk"]
        self.assertEqual([round(v, 6) for v in walk],
                         [round(v, 6) for v in hull_x])

    def test_two_point_reduction_is_exact_when_the_hull_has_two_vertices(self):
        rows = flat_rows(FLAT_TWO)
        self.assertEqual(len(optimum_walk(rows)), 2)
        pred = kappa_at_crossing(two_point_reduction(rows, x_key="T"), W)
        meas = measured_crossing(rows, x_key="T", weight=W, recovery=R)
        self.assertEqual(meas["hops"], 1)
        self.assertAlmostEqual(pred, meas["kappa_arrive"], delta=0.005)

    def test_two_point_reduction_misses_when_the_hull_has_three(self):
        """An intermediate vertex takes the optimum before the ΔC peak does, so
        the ordered-vs-peak crossing is no longer when the peak actually wins —
        the driven-Kuramoto case."""
        rows = flat_rows(FLAT_THREE)
        self.assertEqual(len(optimum_walk(rows)), 3)
        meas = measured_crossing(rows, x_key="T", weight=W, recovery=R)
        self.assertEqual(meas["hops"], 2)
        pred = kappa_at_crossing(two_point_reduction(rows, x_key="T"), W)
        self.assertGreater(abs(pred - meas["kappa_arrive"]) / meas["kappa_arrive"],
                           0.20)

    def test_a_rising_load_can_skip_hull_vertices(self):
        """The hull is the UNIFORM-load prediction, not a bound on the route.
        When the ordered point is taxed harder than the rest it jumps straight
        past intermediate vertices, so hull length and hop count come apart."""
        rows = make_rows()
        self.assertGreater(len(optimum_walk(rows)), 2)
        self.assertEqual(measured_crossing(rows, x_key="T", weight=W,
                                           recovery=R)["hops"], 1)


class TestKuramotoLoadCarriesTheIntegratorScale(unittest.TestCase):
    """The empirical fact the audit is about, pinned so it cannot silently change."""

    def test_kuramoto_load_scales_as_sqrt_dt_while_the_readings_do_not(self):
        from project_genesis.kuramoto_substrate import (
            natural_frequencies, simulate, window_metrics)
        duration, out = 30.0, []
        for dt in (0.1, 0.05, 0.025):
            steps = int(duration / dt)
            rng = np.random.default_rng(11)
            omega = natural_frequencies(200, 0.3, rng, distribution="gaussian")
            phases = rng.uniform(-np.pi, np.pi, 200)
            sim = simulate(phases, omega, 2.0, dt, steps, 0.6, rng,
                           burn_in=steps // 2)
            out.append((dt, window_metrics(sim, freq_tol=0.3)))
        # the load is a per-step quantity: it carries sqrt(dt)
        scaled = [m["activity"] / np.sqrt(dt) for dt, m in out]
        self.assertLess((max(scaled) - min(scaled)) / np.mean(scaled), 0.10)
        # the raw load therefore is NOT invariant — that is the problem
        raw = [m["activity"] for _, m in out]
        self.assertGreater(max(raw) / min(raw), 1.5)
        # while the order parameter is unmoved by the integrator's step
        dis = [m["delta_i"] for _, m in out]
        self.assertLess(max(dis) - min(dis), 0.02)


if __name__ == "__main__":
    unittest.main()
