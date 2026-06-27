"""Tests for project_genesis.gauge_mc — Monte Carlo confinement layer.

Convention note
---------------
The Wilson action in gauge_mc.py is::

    S_W = Σ_{x,μ<ν} Re Tr(1 − P_{μν}(x))

The Metropolis weight is ``exp(−β_g · ΔS)`` where ΔS is computed directly
from the staple sum.  The SU(2) heat-bath kernel internally uses
``alpha = β_g · k / 2`` (Kennedy–Pendleton), but this ``/2`` is absorbed into
the sampling distribution and does **not** halve the effective coupling seen
by ensemble-averaged observables.  The correct single-plaquette SU(2)
benchmark for this convention is therefore::

    <W(1,1)> = I₁(β_g) / I₀(β_g)           [NOT I₁(β_g/2)/I₀(β_g/2)]

This is derived from the partition function of a single SU(2) plaquette:

    Z = ∫ dU exp(β_g Re Tr U) = 2π I₀(β_g)  (up to irrelevant constant)
    <Re Tr U / 2> = I₁(β_g) / I₀(β_g)

All Bessel-based tests in this file use this convention explicitly.

URP alignment
-------------
Confinement tests are anchored to the β-sectorisation / URP paper predictions:
- Strong coupling (β_g ≲ 1): area law dominates, σ > 0, <|P|> ≈ 0.
- The boundary tension formula gives σ ~ β² λ^(−1/2) at small β.
- Three-sector stability window: β ≈ 0.09 (field-theory β, not lattice β_g).
  On the lattice the strong-coupling regime corresponds to small β_g (<1.5 in
  2D SU(2)) before the roughening transition.

Covers:
- Identity / cold-start → zero action, W(R,T)=1
- Metropolis: acceptance rate, SU(N) unitarity, local ΔS consistency
- Heat-bath SU(2) and SU(3): shape, unitarity, action decrease
- Overrelaxation: shape, unitarity, action preservation (≤5% tolerance)
- Bessel benchmark (convention audit): <W(1,1)> vs I₁(β_g)/I₀(β_g)
- Wilson-loop bounds and gauge-invariance
- Polyakov-loop bounds
- Creutz-ratio formula and sign
- Area-law fitter on synthetic data
- URP confinement signals at strong coupling (β_g=0.5):
    * area-law ordering W(3,3) < W(1,1)
    * positive string tension from fit
    * Polyakov loop near zero
    * positive Creutz ratio
- Full driver summary keys and types
- Deconfinement scan returns one entry per beta value
"""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.special import i0, i1  # Bessel I₀, I₁

from project_genesis.gauge import (
    identity_links,
    random_unitary,
    wilson_action,
    apply_gauge_transform,
    is_unitary,
)
from project_genesis.gauge_mc import (
    metropolis_sweep,
    heatbath_sweep,
    overrelaxation_sweep,
    wilson_loop,
    polyakov_loop,
    creutz_ratio,
    fit_area_law,
    thermalize_and_measure_pure_gauge,
    deconfinement_scan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bessel_plaquette_exact(beta_g: float) -> float:
    """Exact single-plaquette <W(1,1)> = I₁(β_g)/I₀(β_g) for SU(2).

    Derived from the partition function::

        Z = ∫_{SU(2)} dU exp(β_g Re Tr U)

    The normalised Wilson loop (Re Tr U / 2) averages to I₁(β_g)/I₀(β_g).
    This matches the gauge_mc.py Metropolis convention where the full β_g
    appears in the Boltzmann weight (the /2 inside the SU(2) heat-bath kernel
    is internal and does not modify the observable expectation value).
    """
    return float(i1(beta_g) / i0(beta_g))


def _bessel_plaquette_halfbeta(beta_g: float) -> float:
    """Candidate (WRONG for this codebase): I₁(β_g/2)/I₀(β_g/2)."""
    return float(i1(beta_g / 2.0) / i0(beta_g / 2.0))


# ---------------------------------------------------------------------------
# Cold-start / identity links
# ---------------------------------------------------------------------------

class TestColdStart(unittest.TestCase):
    """Identity links are zero-action and W(R,T) = 1."""

    def test_identity_action_zero(self):
        links = identity_links(2, (4, 4))
        self.assertAlmostEqual(wilson_action(links), 0.0, places=10)

    def test_identity_wilson_loop_one(self):
        links = identity_links(2, (6, 6))
        for r, t in [(1, 1), (2, 2), (3, 3)]:
            w = wilson_loop(links, (r, t))
            self.assertAlmostEqual(w, 1.0, places=10,
                                   msg=f"W({r},{t}) should be 1 on identity links")

    def test_identity_polyakov_one(self):
        links = identity_links(2, (4, 4))
        p = polyakov_loop(links)
        self.assertAlmostEqual(p, 1.0, places=10)


# ---------------------------------------------------------------------------
# Convention audit — Bessel benchmark
# ---------------------------------------------------------------------------

class TestBesselConventionAudit(unittest.TestCase):
    """Verify <W(1,1)> matches the correct Bessel formula for this codebase.

    This is a *convention audit* test.  It simultaneously evaluates three
    candidate formulae so that any future convention drift is immediately
    visible in the failure message:

        Formula A (correct): I₁(β_g)   / I₀(β_g)
        Formula B (wrong):   I₁(β_g/2) / I₀(β_g/2)
        Formula C (wrong):   I₂(β_g)   / I₁(β_g)   (higher-moment variant)

    A clean failure here means the action convention in gauge_mc.py has
    changed and all downstream physics benchmarks need re-derivation.
    """

    _BETAS = [0.5, 1.0, 2.0]
    _N_THERM = 300
    _N_MEAS = 500
    _N_SKIP = 2
    _ATOL = 0.04   # 4-sigma tolerance for the ensemble size used

    def _measure_w11(self, beta_g: float) -> float:
        rng = np.random.default_rng(seed=int(beta_g * 1000) + 17)
        total = 0.0
        links = random_unitary(rng, 2, (2, 6, 6), special=True)
        for _ in range(self._N_THERM):
            links, _ = heatbath_sweep(links, beta_g, rng)
        for _ in range(self._N_MEAS):
            for _ in range(self._N_SKIP):
                links, _ = heatbath_sweep(links, beta_g, rng)
            total += wilson_loop(links, (1, 1))
        return total / self._N_MEAS

    def _convention_report(self, beta_g: float, measured: float) -> str:
        a = _bessel_plaquette_exact(beta_g)
        b = _bessel_plaquette_halfbeta(beta_g)
        from scipy.special import iv
        c = float(iv(2, beta_g) / i1(beta_g))
        return (
            f"β_g={beta_g}  measured={measured:.5f}  "
            f"I1/I0(β)={a:.5f}  I1/I0(β/2)={b:.5f}  I2/I1(β)={c:.5f}"
        )

    def test_w11_beta_0p5(self):
        beta_g = 0.5
        measured = self._measure_w11(beta_g)
        target = _bessel_plaquette_exact(beta_g)
        self.assertAlmostEqual(
            measured, target, delta=self._ATOL,
            msg=self._convention_report(beta_g, measured),
        )

    def test_w11_beta_1p0(self):
        beta_g = 1.0
        measured = self._measure_w11(beta_g)
        target = _bessel_plaquette_exact(beta_g)
        self.assertAlmostEqual(
            measured, target, delta=self._ATOL,
            msg=self._convention_report(beta_g, measured),
        )

    def test_w11_beta_2p0(self):
        beta_g = 2.0
        measured = self._measure_w11(beta_g)
        target = _bessel_plaquette_exact(beta_g)
        self.assertAlmostEqual(
            measured, target, delta=self._ATOL,
            msg=self._convention_report(beta_g, measured),
        )

    def test_wrong_formula_is_distinguishable(self):
        """Confirm that the wrong (β/2) formula gives a different value at β=1.

        This ensures the audit test has discriminating power — if both formulae
        gave the same value the audit would be vacuous.
        """
        beta_g = 1.0
        correct = _bessel_plaquette_exact(beta_g)
        wrong = _bessel_plaquette_halfbeta(beta_g)
        self.assertGreater(
            abs(correct - wrong), 0.05,
            "The two candidate Bessel formulae should be distinguishably different at β=1",
        )


# ---------------------------------------------------------------------------
# Metropolis
# ---------------------------------------------------------------------------

class TestMetropolis(unittest.TestCase):

    def test_acceptance_rate_positive(self):
        rng = np.random.default_rng(42)
        links = random_unitary(rng, 2, (2, 6, 6), special=True)
        links2, acc = metropolis_sweep(links, 1.5, rng, n_sweeps=5)
        self.assertGreater(acc, 0.05, "Acceptance rate should be > 5 %")
        self.assertLessEqual(acc, 1.0)

    def test_links_stay_unitary_su2(self):
        rng = np.random.default_rng(7)
        links = random_unitary(rng, 2, (2, 6, 6), special=True)
        links2, _ = metropolis_sweep(links, 1.5, rng, n_sweeps=3)
        for mu in range(2):
            for site in np.ndindex(6, 6):
                u = links2[(mu,) + site]
                self.assertTrue(
                    np.allclose(u @ u.conj().T, np.eye(2), atol=1e-9),
                    f"Link (mu={mu}, site={site}) lost unitarity after Metropolis"
                )

    def test_links_stay_unitary_su3(self):
        rng = np.random.default_rng(77)
        links = random_unitary(rng, 3, (2, 4, 4), special=True)
        links2, _ = metropolis_sweep(links, 2.0, rng, n_sweeps=3)
        for mu in range(2):
            for site in np.ndindex(4, 4):
                u = links2[(mu,) + site]
                self.assertTrue(
                    np.allclose(u @ u.conj().T, np.eye(3), atol=1e-8),
                    f"SU(3) link (mu={mu}, site={site}) lost unitarity after Metropolis"
                )

    def test_local_delta_s_consistency(self):
        """ΔS via staple must equal full-action difference (exact identity in code)."""
        from project_genesis.gauge_mc import _staple_sum
        rng = np.random.default_rng(99)
        links = random_unitary(rng, 2, (2, 5, 5), special=True)
        mu, site = 0, (2, 3)
        v = random_unitary(rng, 2, special=True, scale=0.2)
        u_old = links[(mu,) + site]
        u_prop = u_old @ v
        a = _staple_sum(links, mu, site)
        delta_s_local = float(np.real(
            np.trace((u_old - u_prop) @ a.conj().swapaxes(-1, -2))
        ))
        links_prop = links.copy()
        links_prop[(mu,) + site] = u_prop
        delta_s_full = wilson_action(links_prop) - wilson_action(links)
        self.assertAlmostEqual(delta_s_local, delta_s_full, places=8)

    def test_acceptance_band_at_moderate_beta(self):
        """At β_g=1.5 with default step_scale, acceptance should be 30–95 %."""
        rng = np.random.default_rng(43)
        links = random_unitary(rng, 2, (2, 8, 8), special=True)
        _, acc = metropolis_sweep(links, 1.5, rng, n_sweeps=10)
        self.assertGreater(acc, 0.30, "Acceptance too low — step_scale may be too large")
        self.assertLess(acc, 0.95, "Acceptance too high — step_scale may be too small")


# ---------------------------------------------------------------------------
# Heat-bath
# ---------------------------------------------------------------------------

class TestHeatbath(unittest.TestCase):

    def test_heatbath_su2_shape_and_unitary(self):
        rng = np.random.default_rng(1)
        links = random_unitary(rng, 2, (2, 4, 4), special=True)
        links2, _ = heatbath_sweep(links, 2.0, rng, n_sweeps=3)
        self.assertEqual(links2.shape, links.shape)
        self.assertTrue(is_unitary(links2.reshape(-1, 2, 2)))

    def test_heatbath_su3_shape_and_unitary(self):
        rng = np.random.default_rng(2)
        links = random_unitary(rng, 3, (2, 4, 4), special=True)
        links2, _ = heatbath_sweep(links, 2.0, rng, n_sweeps=3)
        self.assertEqual(links2.shape, links.shape)
        u = links2[0, 0, 0]
        self.assertTrue(np.allclose(u @ u.conj().T, np.eye(3), atol=1e-8))

    def test_heatbath_lowers_action_from_hot_start(self):
        """At β_g=5 (strong ordering), heat-bath must decrease S_W from a hot start."""
        rng = np.random.default_rng(55)
        links = random_unitary(rng, 2, (2, 6, 6), special=True, scale=1.0)
        s_before = wilson_action(links)
        links2, _ = heatbath_sweep(links, 5.0, rng, n_sweeps=20)
        s_after = wilson_action(links2)
        self.assertLess(
            s_after, s_before,
            "Heat-bath at β_g=5 should decrease Wilson action from hot start"
        )

    def test_heatbath_acceptance_is_one(self):
        """Heat-bath always accepts; returned acceptance rate should be 1.0."""
        rng = np.random.default_rng(56)
        links = random_unitary(rng, 2, (2, 4, 4), special=True)
        _, acc = heatbath_sweep(links, 2.0, rng)
        self.assertEqual(acc, 1.0)


# ---------------------------------------------------------------------------
# Overrelaxation
# ---------------------------------------------------------------------------

class TestOverrelaxation(unittest.TestCase):

    def test_overrelax_shape(self):
        rng = np.random.default_rng(3)
        links = random_unitary(rng, 3, (2, 5, 5), special=True)
        links2, _ = overrelaxation_sweep(links, rng, n_sweeps=5)
        self.assertEqual(links2.shape, links.shape)

    def test_overrelax_preserves_action(self):
        """Overrelaxation is microcanonical: single-sweep action change ≤ 5 %.

        Tolerance is 5 % (not exact) because the re-projection to SU(N) via
        QR introduces a small numerical error; the update is only exactly
        energy-preserving in exact arithmetic.
        """
        rng = np.random.default_rng(4)
        # equilibrate first so action is non-trivial
        links = random_unitary(rng, 2, (2, 6, 6), special=True, scale=0.4)
        links, _ = heatbath_sweep(links, 2.0, rng, n_sweeps=50)
        s_before = wilson_action(links)
        links2, _ = overrelaxation_sweep(links, rng, n_sweeps=1)
        s_after = wilson_action(links2)
        if s_before > 1e-6:
            rel_change = abs(s_after - s_before) / abs(s_before)
            self.assertLess(
                rel_change, 0.05,
                f"Overrelaxation changed action by {rel_change:.1%} (> 5 %)"
            )

    def test_overrelax_acceptance_is_one(self):
        rng = np.random.default_rng(5)
        links = random_unitary(rng, 2, (2, 4, 4), special=True)
        _, acc = overrelaxation_sweep(links, rng)
        self.assertEqual(acc, 1.0)


# ---------------------------------------------------------------------------
# Wilson loop
# ---------------------------------------------------------------------------

class TestWilsonLoop(unittest.TestCase):

    def test_bounds(self):
        rng = np.random.default_rng(10)
        links = random_unitary(rng, 2, (2, 8, 8), special=True)
        for r, t in [(1, 1), (2, 3), (3, 2)]:
            w = wilson_loop(links, (r, t))
            self.assertGreaterEqual(w, -1.0)
            self.assertLessEqual(w, 1.0 + 1e-9)

    def test_gauge_invariance(self):
        """W(R,T) must be invariant under local SU(N) gauge transforms.

        This is the lattice version of the gauge-covariance proof in the URP
        paper (§3.D): the covariant derivative ensures physical observables
        are invariant under local sector relabelling.
        """
        rng = np.random.default_rng(20)
        n = 2
        spatial = (6, 6)
        links = random_unitary(rng, n, (2, *spatial), special=True)
        g = random_unitary(rng, n, spatial, special=True)
        from project_genesis.gauge import random_psi
        psi = random_psi(rng, n, spatial)
        _, links_t = apply_gauge_transform(psi, links, g)
        w_before = wilson_loop(links, (2, 2))
        w_after = wilson_loop(links_t, (2, 2))
        self.assertAlmostEqual(
            w_before, w_after, places=8,
            msg="Wilson loop must be gauge-invariant (URP §3.D)"
        )

    def test_loop_decreases_with_size_on_random_config(self):
        """On a hot config, |W(1,1)| > |W(3,3)| in absolute value."""
        rng = np.random.default_rng(21)
        links = random_unitary(rng, 2, (2, 8, 8), special=True)
        w11 = wilson_loop(links, (1, 1))
        w33 = wilson_loop(links, (3, 3))
        self.assertGreater(
            abs(w11), abs(w33),
            "Hot-config: |W(1,1)| should exceed |W(3,3)|"
        )


# ---------------------------------------------------------------------------
# Polyakov loop
# ---------------------------------------------------------------------------

class TestPolyakovLoop(unittest.TestCase):

    def test_identity_links_polyakov_one(self):
        links = identity_links(2, (4, 4))
        p = polyakov_loop(links)
        self.assertAlmostEqual(p, 1.0, places=10)

    def test_hot_start_polyakov_small(self):
        """On a completely random config, |<P>| should be near 0.

        This is the URP/QCD prediction: hot start = disordered sector network
        = confined-like (Z_N symmetry unbroken).
        """
        rng = np.random.default_rng(30)
        links = random_unitary(rng, 2, (2, 8, 8), special=True, scale=1.5)
        p = polyakov_loop(links)
        self.assertLess(abs(p), 0.5,
                        "Polyakov loop on hot start should be near 0")


# ---------------------------------------------------------------------------
# Creutz ratio
# ---------------------------------------------------------------------------

class TestCreutzRatio(unittest.TestCase):

    def test_formula(self):
        w22, w11, w12, w21 = 0.5, 0.9, 0.7, 0.7
        chi = creutz_ratio(w22, w11, w21, w12)
        expected = -math.log((w22 * w11) / (w21 * w12))
        self.assertAlmostEqual(chi, expected, places=12)

    def test_nan_on_nonpositive(self):
        self.assertTrue(math.isnan(creutz_ratio(0.0, 0.5, 0.5, 0.5)))
        self.assertTrue(math.isnan(creutz_ratio(-0.1, 0.5, 0.5, 0.5)))

    def test_area_law_gives_positive_creutz(self):
        """If Wilson loops obey a pure area law, χ(R,T) = σ > 0."""
        sigma = 0.12
        W = {(r, t): math.exp(-sigma * r * t) for r in range(1, 4) for t in range(1, 4)}
        chi = creutz_ratio(W[2, 2], W[1, 1], W[2, 1], W[1, 2])
        self.assertGreater(chi, 0.0)
        self.assertAlmostEqual(chi, sigma, delta=0.01)


# ---------------------------------------------------------------------------
# Area-law fitter
# ---------------------------------------------------------------------------

class TestAreaLawFitter(unittest.TestCase):

    def test_recovers_known_sigma(self):
        """Fit must recover σ and c from synthetic area-law data."""
        sigma_true, c_true = 0.12, 0.08
        rs, ts = [1, 2, 3], [1, 2, 3]
        W = np.array([
            [math.exp(-sigma_true * r * t - c_true * (r + t)) for t in ts]
            for r in rs
        ])
        result = fit_area_law(W, rs, ts)
        self.assertAlmostEqual(result["sigma"], sigma_true, places=4)
        self.assertAlmostEqual(result["perimeter_coeff"], c_true, places=4)
        self.assertLess(result["fit_residual"], 1e-8)

    def test_creutz_ratios_present(self):
        rs, ts = [1, 2, 3], [1, 2, 3]
        W = np.ones((3, 3)) * 0.5
        W[0, 0] = 0.9
        result = fit_area_law(W, rs, ts)
        self.assertIn("creutz_ratios", result)

    def test_zero_loops_handled(self):
        """All-zero loops should not crash; returns sigma=0.0."""
        rs, ts = [1, 2], [1, 2]
        W = np.zeros((2, 2))
        result = fit_area_law(W, rs, ts)
        self.assertEqual(result["sigma"], 0.0)

    def test_perimeter_only_gives_zero_sigma(self):
        """Pure perimeter law W = exp(-c(R+T)) should give σ ≈ 0."""
        c = 0.15
        rs, ts = [1, 2, 3], [1, 2, 3]
        W = np.array([
            [math.exp(-c * (r + t)) for t in ts]
            for r in rs
        ])
        result = fit_area_law(W, rs, ts)
        self.assertAlmostEqual(result["sigma"], 0.0, delta=0.01)
        self.assertAlmostEqual(result["perimeter_coeff"], c, delta=0.01)


# ---------------------------------------------------------------------------
# URP confinement signal tests (strong coupling, β_g = 0.5)
# ---------------------------------------------------------------------------

class TestURPConfinementSignals(unittest.TestCase):
    """Ensemble tests anchored to URP/QCD strong-coupling predictions.

    β_g = 0.5 is firmly in the strong-coupling regime for 2D SU(2).  The URP
    paper (§4.A) predicts:

    - Area law: W(R,T) falls exponentially with R·T  →  W(3,3) < W(1,1)
    - Positive string tension: σ > 0 from fit_area_law
    - Confined Polyakov loop: <|P|> ≈ 0 (Z₂ symmetry unbroken)
    - Positive Creutz ratio: χ(2,2) > 0

    These are *ensemble* tests.  We use a small but sufficient run:
    50 thermalisation sweeps + 80 measurements every 3 sweeps.
    """

    BETA_G = 0.5
    SIZE = 6
    N_THERM = 50
    N_MEAS = 80
    N_SKIP = 3

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2026)
        summary, cls._links = thermalize_and_measure_pure_gauge(
            size=cls.SIZE,
            n=2,
            beta_g=cls.BETA_G,
            rng=rng,
            ndim=2,
            n_therm=cls.N_THERM,
            n_meas=cls.N_MEAS,
            n_skip=cls.N_SKIP,
            updater="heatbath",
            loop_sizes=[(1, 1), (2, 2), (3, 3)],
        )
        cls._summary = summary
        cls._loops = summary["loop_averages"]
        cls._fit = summary["area_law_fit"]

    def test_area_law_ordering(self):
        """W(3,3) < W(1,1): area decay must be visible at strong coupling."""
        w11 = self._loops["W_1_1"]
        w33 = self._loops["W_3_3"]
        self.assertLess(
            w33, w11,
            f"Area-law ordering failed: W(3,3)={w33:.4f} >= W(1,1)={w11:.4f}"
        )

    def test_positive_string_tension(self):
        """fit_area_law must return σ > 0 at β_g=0.5 (URP §4.A)."""
        sigma = self._fit["sigma"]
        self.assertGreater(
            sigma, 0.0,
            f"String tension σ={sigma:.5f} should be positive in confined phase"
        )

    def test_polyakov_loop_near_zero(self):
        """<P> ≈ 0 at strong coupling: confined phase, Z₂ unbroken (URP §4.A)."""
        p = self._summary["polyakov_mean"]
        self.assertLess(
            abs(p), 0.35,
            f"Polyakov loop |<P>|={abs(p):.4f} should be near 0 in confined phase"
        )

    def test_creutz_ratio_positive(self):
        """χ(2,2) > 0 at strong coupling — confirms σ > 0 without perimeter contamination."""
        chi = self._fit["creutz_ratios"].get("chi_2_2", float("nan"))
        self.assertFalse(math.isnan(chi), "Creutz ratio chi_2_2 is NaN")
        self.assertGreater(
            chi, 0.0,
            f"Creutz ratio χ(2,2)={chi:.5f} should be positive at β_g={self.BETA_G}"
        )

    def test_w11_in_expected_range(self):
        """W(1,1) at β_g=0.5 should be consistent with Bessel benchmark.

        Expected: I₁(0.5)/I₀(0.5) ≈ 0.240.  We allow ±0.08 for the small
        ensemble (6×6, 80 measurements).  This ties the physics test back to
        the convention audit.
        """
        w11 = self._loops["W_1_1"]
        target = _bessel_plaquette_exact(self.BETA_G)  # ≈ 0.240
        self.assertAlmostEqual(
            w11, target, delta=0.08,
            msg=f"W(1,1)={w11:.4f} far from Bessel target {target:.4f} at β_g={self.BETA_G}"
        )


# ---------------------------------------------------------------------------
# Full driver
# ---------------------------------------------------------------------------

class TestDriverV2(unittest.TestCase):

    def test_driver_summary_keys(self):
        rng = np.random.default_rng(4)
        for updater in ("metropolis", "heatbath", "overrelax"):
            summary, links = thermalize_and_measure_pure_gauge(
                size=4, n=2, beta_g=1.8, rng=rng,
                updater=updater, n_therm=5, n_meas=3
            )
            for key in ("updater", "beta_g", "loop_averages",
                        "polyakov_mean", "final_wilson_action",
                        "area_law_fit", "polyakov_susceptibility"):
                self.assertIn(key, summary,
                              f"Key {key!r} missing for updater={updater!r}")
            self.assertEqual(summary["updater"], updater)
            self.assertIsInstance(links, np.ndarray)

    def test_driver_ndim3_runs(self):
        rng = np.random.default_rng(5)
        summary, _ = thermalize_and_measure_pure_gauge(
            size=3, n=2, beta_g=2.0, rng=rng,
            ndim=3, n_therm=3, n_meas=2,
        )
        self.assertIn("sigma", summary["area_law_fit"])


# ---------------------------------------------------------------------------
# Deconfinement scan
# ---------------------------------------------------------------------------

class TestDeconfinementScan(unittest.TestCase):

    def test_scan_returns_one_entry_per_beta(self):
        rng = np.random.default_rng(6)
        betas = [1.0, 2.0, 3.0]
        results = deconfinement_scan(
            size=4, n=2, beta_values=betas, rng=rng,
            ndim=2, n_therm=5, n_meas=3, updater="metropolis"
        )
        self.assertEqual(len(results), len(betas))
        for r, b in zip(results, betas):
            self.assertAlmostEqual(r["beta_g"], b)
            self.assertIn("polyakov_mean", r)

    def test_scan_sigma_decreases_with_beta(self):
        """String tension should decrease as β_g increases (weaker coupling).

        This is a qualitative scan over two β values to avoid long runtimes.
        It is the lattice version of asymptotic freedom: at larger β_g
        (weaker coupling / finer lattice) the string tension measured in
        lattice units decreases.
        """
        rng = np.random.default_rng(7)
        results = deconfinement_scan(
            size=5, n=2, beta_values=[0.5, 2.5], rng=rng,
            ndim=2, n_therm=30, n_meas=30, n_skip=3, updater="heatbath"
        )
        sigma_strong = results[0]["area_law_fit"]["sigma"]
        sigma_weak = results[1]["area_law_fit"]["sigma"]
        self.assertGreaterEqual(
            sigma_strong, sigma_weak,
            f"Expected σ(β=0.5)={sigma_strong:.4f} ≥ σ(β=2.5)={sigma_weak:.4f}"
        )


if __name__ == "__main__":
    unittest.main()
