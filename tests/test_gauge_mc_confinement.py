"""Extended confinement tests for the Monte Carlo gauge layer.

This file targets gaps *not* covered by ``tests/test_gauge_mc.py``:

1. **SU(3) confinement** — area-law ordering and σ > 0 for the physical
   gauge group of the URP emergent-SU(3) derivation.
2. **Thermalization convergence** — Wilson action must stabilise within the
   declared thermalisation budget; a diverging or flat action indicates the
   sampler is broken, not just slow.
3. **Hot / cold start equivalence** — both hot (random) and cold (identity)
   starts must converge to the same ensemble average for W(1,1); this is the
   lattice equivalent of the URP statement that the Boltzmann weight is
   independent of initial sector labelling.
4. **Ensemble-averaged gauge invariance** — ensemble-mean Wilson loops must
   be unchanged under a global re-gauging of the thermalised ensemble.  This
   is the Monte Carlo version of the URP covariant-derivative proof.
5. **Action autocorrelation decay** — the integrated autocorrelation time of
   S_W must be finite and short (< half the measurement window), confirming
   the chain is ergodic enough to sample independent configurations.
6. **Multi-beta Polyakov order-parameter scan** — |<P>| must be monotonically
   non-decreasing as β_g increases across the confined → deconfined crossover,
   matching the URP β-sectorisation prediction that larger β_g corresponds to
   a less-confined (more ordered) gauge sector.

Parameter choices
-----------------
- β_g = 0.5  →  firmly strong-coupling for 2D SU(2) and SU(3): area law
  dominates, σ > 0, <|P|> ≈ 0.
- β_g = 3.0  →  intermediate-to-weak coupling in 2D: area law weakens,
  <|P|> rises.
- Lattice: 6×6 (2D) for speed; 4×4×4 (3D) for the SU(3) test only.
- Sweep counts are reduced for CI speed while preserving qualitative signals.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from project_genesis.gauge import (
    identity_links,
    random_unitary,
    apply_gauge_transform,
    random_psi,
    wilson_action,
)
from project_genesis.gauge_mc import (
    heatbath_sweep,
    metropolis_sweep,
    wilson_loop,
    polyakov_loop,
    fit_area_law,
    thermalize_and_measure_pure_gauge,
    deconfinement_scan,
)


def _run_ensemble(
    n: int = 2,
    beta_g: float = 0.5,
    size: int = 6,
    ndim: int = 2,
    n_therm: int = 40,
    n_meas: int = 40,
    n_skip: int = 2,
    seed: int = 9001,
    updater: str = "heatbath",
    loop_sizes: list | None = None,
) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    if loop_sizes is None:
        loop_sizes = [(1, 1), (2, 2), (3, 3)]
    return thermalize_and_measure_pure_gauge(
        size=size,
        n=n,
        beta_g=beta_g,
        rng=rng,
        ndim=ndim,
        n_therm=n_therm,
        n_meas=n_meas,
        n_skip=n_skip,
        updater=updater,
        loop_sizes=loop_sizes,
    )


class TestSU3Confinement(unittest.TestCase):
    """SU(3) confinement signals on a 4×4×4 3D lattice.

    Two ensembles are used, because not every observable is statistically
    resolvable at the same coupling with a CI-scale budget:

    - **Strong coupling (β_g = 0.5)** — W(1,1) ≈ 0.08, so W(3,3) ≈ 10⁻⁵ is
      pure noise here.  This ensemble supports the qualitative signals:
      area-law ordering, confined Polyakov loop, W(1,1) range.
    - **Moderate coupling (β_g = 2.0)** — 3D SU(3) is still confining, and
      W(2,2) is a strong signal, so the *quantitative* estimators (fitted
      σ > 0, Creutz ratio χ(2,2) > 0) are measured here, with loops
      restricted to R,T ≤ 2 to keep noise out of the log-fit.
    """

    @classmethod
    def setUpClass(cls):
        summary, cls._links = _run_ensemble(
            n=3, beta_g=0.5, size=4, ndim=3,
            n_therm=40, n_meas=30, n_skip=2,
            seed=30031,
        )
        cls._summary = summary
        cls._loops = summary["loop_averages"]

        moderate, _ = _run_ensemble(
            n=3, beta_g=2.0, size=4, ndim=3,
            n_therm=40, n_meas=60, n_skip=2,
            seed=30032,
            loop_sizes=[(1, 1), (2, 2)],
        )
        cls._fit = moderate["area_law_fit"]

    def test_su3_area_law_ordering(self):
        w11 = self._loops["W_1_1"]
        w33 = self._loops["W_3_3"]
        self.assertLess(w33, w11)

    def test_su3_positive_string_tension(self):
        sigma = self._fit["sigma"]
        self.assertGreater(sigma, 0.0)

    def test_su3_polyakov_near_zero(self):
        p = self._summary["polyakov_mean"]
        self.assertLess(abs(p), 0.4)

    def test_su3_creutz_ratio_positive(self):
        chi = self._fit["creutz_ratios"].get("chi_2_2", float("nan"))
        self.assertFalse(math.isnan(chi))
        self.assertGreater(chi, 0.0)

    def test_su3_w11_in_reasonable_range(self):
        w11 = self._loops["W_1_1"]
        self.assertGreater(w11, 0.0)
        self.assertLess(w11, 1.0)


class TestThermalizationConvergence(unittest.TestCase):
    def _action_trajectory(self, beta_g: float, n_sweeps: int, seed: int) -> list[float]:
        rng = np.random.default_rng(seed)
        links = random_unitary(rng, 2, (2, 6, 6), special=True, scale=1.0)
        trajectory = []
        for _ in range(n_sweeps):
            links, _ = heatbath_sweep(links, beta_g, rng, n_sweeps=1)
            trajectory.append(float(wilson_action(links)))
        return trajectory

    def test_action_stabilises_su2_strong_coupling(self):
        traj = self._action_trajectory(beta_g=0.5, n_sweeps=60, seed=7711)
        early_mean = np.mean(traj[30:45])
        late_mean = np.mean(traj[45:60])
        if early_mean > 1e-6:
            rel_drift = abs(late_mean - early_mean) / abs(early_mean)
            self.assertLess(rel_drift, 0.15)

    def test_action_decreases_from_hot_start_strong_coupling(self):
        traj = self._action_trajectory(beta_g=2.0, n_sweeps=40, seed=7712)
        self.assertLess(traj[-1], traj[0])

    def test_cold_start_action_rises_then_stabilises(self):
        rng = np.random.default_rng(7713)
        links = identity_links(2, (6, 6))
        trajectory = []
        for _ in range(60):
            links, _ = heatbath_sweep(links, 1.0, rng, n_sweeps=1)
            trajectory.append(float(wilson_action(links)))
        self.assertGreater(trajectory[-1], trajectory[0])
        late_mean = np.mean(trajectory[45:60])
        early_mean = np.mean(trajectory[30:45])
        if early_mean > 1e-6:
            rel_drift = abs(late_mean - early_mean) / abs(early_mean)
            self.assertLess(rel_drift, 0.15)


class TestHotColdStartEquivalence(unittest.TestCase):
    BETA_G = 1.0
    SIZE = 6
    N_THERM = 80
    N_MEAS = 40
    N_SKIP = 2
    EQUIV_TOL = 0.15

    @classmethod
    def setUpClass(cls):
        rng_hot = np.random.default_rng(55001)
        cls._hot_summary, _ = thermalize_and_measure_pure_gauge(
            size=cls.SIZE, n=2, beta_g=cls.BETA_G,
            rng=rng_hot, ndim=2,
            n_therm=cls.N_THERM, n_meas=cls.N_MEAS, n_skip=cls.N_SKIP,
            updater="heatbath",
            loop_sizes=[(1, 1)],
        )
        rng_cold = np.random.default_rng(55002)
        links_cold = identity_links(2, (cls.SIZE, cls.SIZE))
        for _ in range(cls.N_THERM):
            links_cold, _ = heatbath_sweep(links_cold, cls.BETA_G, rng_cold)
        w11_accum = 0.0
        for _ in range(cls.N_MEAS):
            for _ in range(cls.N_SKIP):
                links_cold, _ = heatbath_sweep(links_cold, cls.BETA_G, rng_cold)
            w11_accum += wilson_loop(links_cold, (1, 1))
        cls._cold_w11 = w11_accum / cls.N_MEAS
        cls._hot_w11 = cls._hot_summary["loop_averages"]["W_1_1"]

    def test_w11_hot_cold_agree(self):
        diff = abs(self._hot_w11 - self._cold_w11)
        self.assertLess(diff, self.EQUIV_TOL)

    def test_both_starts_give_positive_w11(self):
        self.assertGreater(self._hot_w11, 0.0)
        self.assertGreater(self._cold_w11, 0.0)


class TestEnsembleGaugeInvariance(unittest.TestCase):
    BETA_G = 1.5
    SIZE = 6
    N_THERM = 60
    TOL = 1e-7

    def test_w22_invariant_under_gauge_transform(self):
        rng = np.random.default_rng(66001)
        spatial = (self.SIZE, self.SIZE)
        links = random_unitary(rng, 2, (2, *spatial), special=True)
        for _ in range(self.N_THERM):
            links, _ = heatbath_sweep(links, self.BETA_G, rng)
        w_before = wilson_loop(links, (2, 2))
        g = random_unitary(rng, 2, spatial, special=True)
        psi = random_psi(rng, 2, spatial)
        _, links_gauged = apply_gauge_transform(psi, links, g)
        w_after = wilson_loop(links_gauged, (2, 2))
        self.assertAlmostEqual(w_before, w_after, delta=self.TOL)

    def test_w11_invariant_under_gauge_transform(self):
        rng = np.random.default_rng(66002)
        spatial = (self.SIZE, self.SIZE)
        links = random_unitary(rng, 2, (2, *spatial), special=True)
        for _ in range(self.N_THERM):
            links, _ = heatbath_sweep(links, self.BETA_G, rng)
        w_before = wilson_loop(links, (1, 1))
        g = random_unitary(rng, 2, spatial, special=True)
        psi = random_psi(rng, 2, spatial)
        _, links_gauged = apply_gauge_transform(psi, links, g)
        w_after = wilson_loop(links_gauged, (1, 1))
        self.assertAlmostEqual(w_before, w_after, delta=self.TOL)

    def test_polyakov_invariant_under_gauge_transform(self):
        rng = np.random.default_rng(66003)
        spatial = (self.SIZE, self.SIZE)
        links = random_unitary(rng, 2, (2, *spatial), special=True)
        for _ in range(self.N_THERM):
            links, _ = heatbath_sweep(links, self.BETA_G, rng)
        p_before = polyakov_loop(links)
        g = random_unitary(rng, 2, spatial, special=True)
        psi = random_psi(rng, 2, spatial)
        _, links_gauged = apply_gauge_transform(psi, links, g)
        p_after = polyakov_loop(links_gauged)
        self.assertAlmostEqual(p_before, p_after, delta=self.TOL)


class TestAutocorrelationDecay(unittest.TestCase):
    N_THERM = 60
    N_MEAS = 100
    BETA_G = 1.0
    SIZE = 6
    MAX_TAU_FRACTION = 0.4

    @staticmethod
    def _integrated_autocorr(series: list[float], t_max: int = 20) -> float:
        arr = np.array(series, dtype=np.float64)
        arr -= arr.mean()
        var = np.var(arr)
        if var < 1e-14:
            return 0.0
        tau = 0.5
        for t in range(1, min(t_max + 1, len(arr))):
            c_t = float(np.mean(arr[:-t] * arr[t:])) / var
            if c_t <= 0.0:
                break
            tau += c_t
        return tau

    def test_heatbath_autocorr_finite(self):
        rng = np.random.default_rng(77001)
        links = random_unitary(rng, 2, (2, self.SIZE, self.SIZE), special=True)
        for _ in range(self.N_THERM):
            links, _ = heatbath_sweep(links, self.BETA_G, rng)
        actions = []
        for _ in range(self.N_MEAS):
            links, _ = heatbath_sweep(links, self.BETA_G, rng)
            actions.append(float(wilson_action(links)))
        tau = self._integrated_autocorr(actions)
        threshold = self.MAX_TAU_FRACTION * self.N_MEAS
        self.assertLess(tau, threshold)

    def test_metropolis_autocorr_finite(self):
        rng = np.random.default_rng(77002)
        links = random_unitary(rng, 2, (2, self.SIZE, self.SIZE), special=True)
        for _ in range(self.N_THERM):
            links, _ = metropolis_sweep(links, self.BETA_G, rng, n_sweeps=1)
        actions = []
        for _ in range(self.N_MEAS):
            links, _ = metropolis_sweep(links, self.BETA_G, rng, n_sweeps=1)
            actions.append(float(wilson_action(links)))
        tau = self._integrated_autocorr(actions)
        threshold = self.MAX_TAU_FRACTION * self.N_MEAS
        self.assertLess(tau, threshold)


class TestPolyakovOrderParameterScan(unittest.TestCase):
    BETAS = [0.5, 1.5, 3.0]
    SIZE = 6
    N_THERM = 40
    N_MEAS = 50
    N_SKIP = 2
    MONO_TOL = 0.08

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(88001)
        # Loops restricted to R,T ≤ 2: at β_g = 0.5 the true W(3,3) is
        # ≈ 6·10⁻⁶ — pure noise at this measurement budget — and feeding
        # noise into the log-fit randomises the fitted σ.
        results = deconfinement_scan(
            size=cls.SIZE,
            n=2,
            beta_values=cls.BETAS,
            rng=rng,
            ndim=2,
            n_therm=cls.N_THERM,
            n_meas=cls.N_MEAS,
            n_skip=cls.N_SKIP,
            updater="heatbath",
            loop_sizes=[(1, 1), (2, 2)],
        )
        cls._results = results
        cls._poly = [abs(r["polyakov_mean"]) for r in results]
        cls._sigma = [r["area_law_fit"]["sigma"] for r in results]

    def test_polyakov_increases_with_beta(self):
        for i in range(len(self._poly) - 1):
            p_lo = self._poly[i]
            p_hi = self._poly[i + 1]
            self.assertGreaterEqual(p_hi + self.MONO_TOL, p_lo)

    def test_lowest_beta_has_smallest_polyakov(self):
        self.assertLessEqual(self._poly[0], self._poly[-1] + self.MONO_TOL)

    def test_sigma_non_negative_at_all_betas(self):
        for s in self._sigma:
            self.assertGreaterEqual(s, -0.01)

    def test_strong_coupling_sigma_largest(self):
        self.assertGreaterEqual(self._sigma[0] + 0.02, self._sigma[-1])


if __name__ == "__main__":
    unittest.main()
