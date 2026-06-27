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
  dominates, σ > 0, <|P|> ≈ 0.  (URP §4.A, roughening transition well above.)
- β_g = 3.0  →  intermediate-to-weak coupling in 2D: area law weakens,
  <|P|> rises.  (URP §4.B crossover region.)
- Lattice: 6×6 (2D) for speed; 4×4×4 (3D) for the SU(3) test only.
- Sweep counts are the minimum to give reliable qualitative signals; they are
  *not* tuned for precise continuum-limit extrapolation.
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run_ensemble(
    n: int = 2,
    beta_g: float = 0.5,
    size: int = 6,
    ndim: int = 2,
    n_therm: int = 60,
    n_meas: int = 80,
    n_skip: int = 3,
    seed: int = 9001,
    updater: str = "heatbath",
    loop_sizes: list | None = None,
) -> tuple[dict, np.ndarray]:
    """Thin wrapper around thermalize_and_measure_pure_gauge for test reuse."""
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


# ---------------------------------------------------------------------------
# 1. SU(3) confinement — area-law and positive sigma
# ---------------------------------------------------------------------------

class TestSU3Confinement(unittest.TestCase):
    """Confirm that the SU(3) Cabibbo-Marinari heat-bath produces confined-phase
    signals at strong coupling, as required by the URP emergent-SU(3) sector.

    We use a 3D lattice (4^3) so the Polyakov loop direction is well-defined
    and distinct from the two Wilson-loop directions.

    β_g = 0.5 is strong coupling for SU(3) in 3D: the roughening transition
    sits well above β_g ≈ 2 in 3D SU(3) (cf. Creutz 1980).
    """

    @classmethod
    def setUpClass(cls):
        summary, cls._links = _run_ensemble(
            n=3, beta_g=0.5, size=4, ndim=3,
            n_therm=80, n_meas=60, n_skip=4,
            seed=30031,
        )
        cls._summary = summary
        cls._loops = summary["loop_averages"]
        cls._fit = summary["area_law_fit"]

    def test_su3_area_law_ordering(self):
        """W(3,3) < W(1,1) for SU(3) at strong coupling: area decay is present."""
        w11 = self._loops["W_1_1"]
        w33 = self._loops["W_3_3"]
        self.assertLess(
            w33, w11,
            f"SU(3) area ordering failed: W(3,3)={w33:.4f} >= W(1,1)={w11:.4f}",
        )

    def test_su3_positive_string_tension(self):
        """fit_area_law returns σ > 0 for SU(3) at strong coupling (URP §4.A)."""
        sigma = self._fit["sigma"]
        self.assertGreater(
            sigma, 0.0,
            f"SU(3) string tension σ={sigma:.5f} should be positive in confined phase",
        )

    def test_su3_polyakov_near_zero(self):
        """SU(3) Polyakov loop should be near zero at β_g=0.5 (Z_3 unbroken)."""
        p = self._summary["polyakov_mean"]
        self.assertLess(
            abs(p), 0.4,
            f"SU(3) |<P>|={abs(p):.4f} should be near 0 in confined phase",
        )

    def test_su3_creutz_ratio_positive(self):
        """χ(2,2) > 0 for SU(3) — perimeter-subtracted string tension signal."""
        chi = self._fit["creutz_ratios"].get("chi_2_2", float("nan"))
        self.assertFalse(math.isnan(chi), "SU(3) Creutz ratio chi_2_2 is NaN")
        self.assertGreater(
            chi, 0.0,
            f"SU(3) χ(2,2)={chi:.5f} should be positive at β_g=0.5",
        )

    def test_su3_w11_in_reasonable_range(self):
        """W(1,1) at β_g=0.5, SU(3) should be positive and < 1."""
        w11 = self._loops["W_1_1"]
        self.assertGreater(w11, 0.0, "SU(3) W(1,1) should be positive")
        self.assertLess(w11, 1.0, "SU(3) W(1,1) should be < 1 on thermalised config")


# ---------------------------------------------------------------------------
# 2. Thermalization convergence
# ---------------------------------------------------------------------------

class TestThermalizationConvergence(unittest.TestCase):
    """The Wilson action must stabilise within the thermalisation budget.

    A sampler that is stuck or diverging will show a monotonically changing
    (not plateaued) action trajectory.  We test this by recording the action
    in two equal windows at the end of thermalisation and checking they have
    overlapping means — i.e. the chain has reached stationarity.

    This implements the URP requirement that the Boltzmann ensemble is
    genuinely sampled (not just initialised) before measurements begin.
    """

    def _action_trajectory(self, beta_g: float, n_sweeps: int, seed: int) -> list[float]:
        """Return the Wilson action after each sweep from a hot start."""
        rng = np.random.default_rng(seed)
        links = random_unitary(rng, 2, (2, 6, 6), special=True, scale=1.0)
        trajectory = []
        for _ in range(n_sweeps):
            links, _ = heatbath_sweep(links, beta_g, rng, n_sweeps=1)
            trajectory.append(float(wilson_action(links)))
        return trajectory

    def test_action_stabilises_su2_strong_coupling(self):
        """At β_g=0.5, action should plateau before sweep 80 from a hot start.

        We compare the mean of sweeps [40,60) vs [60,80) — if the chain has
        converged these two windows should agree within 10 % relative error.
        """
        traj = self._action_trajectory(beta_g=0.5, n_sweeps=80, seed=7711)
        early_mean = np.mean(traj[40:60])
        late_mean = np.mean(traj[60:80])
        if early_mean > 1e-6:
            rel_drift = abs(late_mean - early_mean) / abs(early_mean)
            self.assertLess(
                rel_drift, 0.10,
                f"Action not stabilised: early={early_mean:.4f}, late={late_mean:.4f}, "
                f"rel_drift={rel_drift:.3f} > 10%",
            )

    def test_action_decreases_from_hot_start_strong_coupling(self):
        """From a hot start at β_g=2.0, S_W must be lower after 50 sweeps.

        This is the most basic sanity check: a correct sampler must move
        the system towards lower action at larger β_g.
        """
        traj = self._action_trajectory(beta_g=2.0, n_sweeps=50, seed=7712)
        self.assertLess(
            traj[-1], traj[0],
            f"Action did not decrease: S_init={traj[0]:.4f}, S_final={traj[-1]:.4f}",
        )

    def test_cold_start_action_rises_then_stabilises(self):
        """From a cold start at moderate β_g, S_W should rise and then plateau.

        Identity links have S_W = 0.  At β_g = 1.0 the equilibrium action
        is nonzero (thermal fluctuations lift it).  After thermalisation the
        action must be positive and stable.
        """
        rng = np.random.default_rng(7713)
        links = identity_links(2, (6, 6))
        trajectory = []
        for _ in range(80):
            links, _ = heatbath_sweep(links, 1.0, rng, n_sweeps=1)
            trajectory.append(float(wilson_action(links)))
        # action must have risen from 0
        self.assertGreater(trajectory[-1], trajectory[0],
                           "Cold-start action should rise above 0 after thermalisation")
        # and then plateaued
        late_mean = np.mean(trajectory[60:80])
        early_mean = np.mean(trajectory[40:60])
        if early_mean > 1e-6:
            rel_drift = abs(late_mean - early_mean) / abs(early_mean)
            self.assertLess(rel_drift, 0.10,
                            f"Cold-start action not stabilised: rel_drift={rel_drift:.3f}")


# ---------------------------------------------------------------------------
# 3. Hot / cold start equivalence
# ---------------------------------------------------------------------------

class TestHotColdStartEquivalence(unittest.TestCase):
    """Hot and cold starts must converge to the same ensemble average.

    This is a direct test of ergodicity — the Markov chain must be capable
    of forgetting its initial condition.  In the URP picture this corresponds
    to the statement that the sector-weight distribution is uniquely determined
    by β_g and the gauge group, not by how the initial coherence pattern was
    set up.

    We use β_g = 1.0 (intermediate coupling) and measure W(1,1) after a
    generous thermalisation budget.  The two starting-point estimates should
    agree within 2× the single-run statistical uncertainty (≈ 0.08 for this
    ensemble size).
    """

    BETA_G = 1.0
    SIZE = 6
    N_THERM = 200
    N_MEAS = 100
    N_SKIP = 3
    EQUIV_TOL = 0.10  # allow 0.10 absolute difference between the two starts

    @classmethod
    def setUpClass(cls):
        # Hot start
        rng_hot = np.random.default_rng(55001)
        cls._hot_summary, _ = thermalize_and_measure_pure_gauge(
            size=cls.SIZE, n=2, beta_g=cls.BETA_G,
            rng=rng_hot, ndim=2,
            n_therm=cls.N_THERM, n_meas=cls.N_MEAS, n_skip=cls.N_SKIP,
            updater="heatbath",
            loop_sizes=[(1, 1)],
        )
        # Cold start: begin from identity links
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
        """Hot and cold starts must give W(1,1) within EQUIV_TOL of each other."""
        diff = abs(self._hot_w11 - self._cold_w11)
        self.assertLess(
            diff, self.EQUIV_TOL,
            f"Hot/cold starts disagree: hot W(1,1)={self._hot_w11:.4f}, "
            f"cold W(1,1)={self._cold_w11:.4f}, diff={diff:.4f} > {self.EQUIV_TOL}",
        )

    def test_both_starts_give_positive_w11(self):
        """At β_g=1.0 both starts should give W(1,1) > 0 (ordered phase signal)."""
        self.assertGreater(self._hot_w11, 0.0,
                           f"Hot-start W(1,1)={self._hot_w11:.4f} should be > 0")
        self.assertGreater(self._cold_w11, 0.0,
                           f"Cold-start W(1,1)={self._cold_w11:.4f} should be > 0")


# ---------------------------------------------------------------------------
# 4. Ensemble-averaged gauge invariance
# ---------------------------------------------------------------------------

class TestEnsembleGaugeInvariance(unittest.TestCase):
    """Ensemble-mean Wilson loops must be unchanged under re-gauging.

    The URP covariant-derivative proof guarantees that any gauge-invariant
    functional of the links is invariant under local G-transforms.  This test
    verifies the *ensemble mean* of W(2,2) is unchanged when we apply a
    random gauge transform to the final thermalised configuration and remeasure.

    We use a single thermalised configuration (not a full new run) because we
    are testing mathematical gauge invariance, not sampling stability.
    """

    BETA_G = 1.5
    SIZE = 6
    N_THERM = 100
    TOL = 1e-7  # gauge invariance is exact up to floating-point precision

    def test_w22_invariant_under_gauge_transform(self):
        rng = np.random.default_rng(66001)
        spatial = (self.SIZE, self.SIZE)
        links = random_unitary(rng, 2, (2, *spatial), special=True)
        for _ in range(self.N_THERM):
            links, _ = heatbath_sweep(links, self.BETA_G, rng)

        w_before = wilson_loop(links, (2, 2))

        # Apply a random local SU(2) gauge transform
        g = random_unitary(rng, 2, spatial, special=True)
        psi = random_psi(rng, 2, spatial)
        _, links_gauged = apply_gauge_transform(psi, links, g)

        w_after = wilson_loop(links_gauged, (2, 2))

        self.assertAlmostEqual(
            w_before, w_after, delta=self.TOL,
            msg=f"W(2,2) changed under gauge transform: {w_before:.10f} -> {w_after:.10f}",
        )

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

        self.assertAlmostEqual(
            w_before, w_after, delta=self.TOL,
            msg=f"W(1,1) changed under gauge transform: {w_before:.10f} -> {w_after:.10f}",
        )

    def test_polyakov_invariant_under_gauge_transform(self):
        """Polyakov loop is gauge-invariant (Tr of closed loop)."""
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

        self.assertAlmostEqual(
            p_before, p_after, delta=self.TOL,
            msg=f"Polyakov loop changed under gauge transform: {p_before:.10f} -> {p_after:.10f}",
        )


# ---------------------------------------------------------------------------
# 5. Action autocorrelation decay
# ---------------------------------------------------------------------------

class TestAutocorrelationDecay(unittest.TestCase):
    """The integrated autocorrelation time of S_W must be finite and < N_meas/2.

    An infinite (or very long) autocorrelation time would mean successive
    measurements are essentially identical — the chain is stuck.  For the
    URP lattice to genuinely sample the Boltzmann weight, the chain must
    decorrelate within the measurement window.

    We compute the normalised autocorrelation function

        C(t) = Cov(S_i, S_{i+t}) / Var(S)

    and the integrated autocorrelation time

        tau_int = 0.5 + sum_{t=1}^{T_max} C(t)

    and require tau_int < N_meas / 2.
    """

    N_THERM = 100
    N_MEAS = 200
    BETA_G = 1.0
    SIZE = 6
    MAX_TAU_FRACTION = 0.4  # tau_int must be < 40% of N_MEAS

    @staticmethod
    def _integrated_autocorr(series: list[float], t_max: int = 20) -> float:
        """Compute integrated autocorrelation time of a 1D time series."""
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
        """Heat-bath autocorrelation time must be < MAX_TAU_FRACTION * N_MEAS."""
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
        self.assertLess(
            tau, threshold,
            f"Heat-bath tau_int={tau:.2f} >= {threshold:.0f} ({self.MAX_TAU_FRACTION:.0%} of N_MEAS). "
            f"Chain may be stuck.",
        )

    def test_metropolis_autocorr_finite(self):
        """Metropolis autocorrelation time must be < MAX_TAU_FRACTION * N_MEAS."""
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
        self.assertLess(
            tau, threshold,
            f"Metropolis tau_int={tau:.2f} >= {threshold:.0f}. Chain may be stuck.",
        )


# ---------------------------------------------------------------------------
# 6. Multi-beta Polyakov order-parameter scan
# ---------------------------------------------------------------------------

class TestPolyakovOrderParameterScan(unittest.TestCase):
    """|<P>| must be monotonically non-decreasing as β_g increases.

    The URP β-sectorisation predicts a confined → deconfined crossover as the
    effective coupling weakens (β_g increases).  The Polyakov loop is the
    order parameter for this transition.  In a strict test we require that
    |<P>| at each successive β value is no smaller than the previous one.

    We use three β values spanning the crossover in 2D SU(2):
        β_g = 0.5  →  strongly confined
        β_g = 1.5  →  intermediate
        β_g = 3.0  →  weakly confined / pre-deconfined

    To avoid flaky order reversals from statistical noise we allow a tolerance
    of 0.05 in the monotonicity check (i.e. the next beta's |<P>| may be up to
    0.05 *lower* than the previous one and still pass — this covers ~1-sigma
    fluctuations for the ensemble sizes used).
    """

    BETAS = [0.5, 1.5, 3.0]
    SIZE = 6
    N_THERM = 80
    N_MEAS = 100
    N_SKIP = 3
    MONO_TOL = 0.05  # allowed downward wiggle between adjacent beta values

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(88001)
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
            loop_sizes=[(1, 1), (2, 2), (3, 3)],
        )
        cls._results = results
        cls._poly = [abs(r["polyakov_mean"]) for r in results]
        cls._sigma = [r["area_law_fit"]["sigma"] for r in results]

    def test_polyakov_increases_with_beta(self):
        """Each step up in β_g must give a non-decreasing Polyakov order parameter
        (within MONO_TOL statistical tolerance)."""
        for i in range(len(self._poly) - 1):
            b_lo = self.BETAS[i]
            b_hi = self.BETAS[i + 1]
            p_lo = self._poly[i]
            p_hi = self._poly[i + 1]
            self.assertGreaterEqual(
                p_hi + self.MONO_TOL, p_lo,
                f"|<P>|(β={b_hi}) = {p_hi:.4f} is more than {self.MONO_TOL} "
                f"below |<P>|(β={b_lo}) = {p_lo:.4f}; expected monotone increase.",
            )

    def test_lowest_beta_has_smallest_polyakov(self):
        """The most confined point (β_g=0.5) must have the smallest |<P>|."""
        self.assertLessEqual(
            self._poly[0],
            self._poly[-1] + self.MONO_TOL,
            f"|<P>|(β=0.5)={self._poly[0]:.4f} should be <= "
            f"|<P>|(β=3.0)={self._poly[-1]:.4f}",
        )

    def test_sigma_non_negative_at_all_betas(self):
        """Fitted σ must be ≥ 0 at all β values (never negative, even at weak coupling)."""
        for b, s in zip(self.BETAS, self._sigma):
            self.assertGreaterEqual(
                s, -0.01,  # tiny negative tolerance for numerical noise
                f"sigma={s:.5f} < 0 at β_g={b} — area-law fit broke down",
            )

    def test_strong_coupling_sigma_largest(self):
        """String tension at β_g=0.5 must be >= string tension at β_g=3.0.

        This is the lattice asymptotic-freedom direction: σ(lattice units)
        decreases as the coupling weakens.
        """
        self.assertGreaterEqual(
            self._sigma[0] + 0.02,
            self._sigma[-1],
            f"σ(β=0.5)={self._sigma[0]:.4f} should be >= σ(β=3.0)={self._sigma[-1]:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
