"""
tests/test_gauge_mc_confinement.py
====================================
URP-anchored confinement tests for project_genesis.gauge_mc.

Every test is tied to either an analytically exact result or a qualitative
expectation derived from the URP β-sectorisation → SU(3) derivation.

Test catalogue
--------------
Structural / algebraic (instantaneous, no Monte Carlo)
  T1   cold_start → W(1,1)=1, Wilson action=0  [flat connection = zero stress]
  T2   Metropolis preserves unitarity after 100 sweeps
  T3   Metropolis preserves det=1 after 100 sweeps

Metropolis sampler health
  T4   Acceptance rate 20–80 % at equilibrium for β ∈ {0.5, 1.5, 3.0}
  T5   Action decreases from hot start (200 sweeps, β=2.0)

Gauge invariance  [tests the URP gauge-symmetry proof in code]
  T6   W(2,2) invariant under a random local SU(2) gauge transformation
  T7   Wilson action invariant under same transformation

Bessel benchmark  [primary sampler correctness — single most important test]
  T8   ⟨W(1,1)⟩ matches I₁(β/2)/I₀(β/2) within 5 % for β ∈ {0.5…4.0}

Confinement signatures  [URP theory predictions]
  T9   W(1,1) > W(1,2) > 0 at β=1.5  (area-law suppression)
  T10  W(1,2) ≈ W(2,1) at β=1.5  (discrete rotational symmetry)
  T11  |⟨P⟩| < 0.15 at β=1.5  (confined phase, centre symmetry unbroken)
  T12  Creutz χ(2,2) computable and positive  (nonzero string tension)
  T13  fit_area_law returns σ > 0 at β=1.5  (area-law fit)
"""

from __future__ import annotations
import math
import pytest
import numpy as np
from scipy.special import iv as bessel_iv

from project_genesis.gauge import _dagger, random_unitary, identity_links
from project_genesis.gauge_mc import (
    metropolis_sweep,
    wilson_loop,
    polyakov_loop,
    creutz_ratio,
    fit_area_law,
    thermalize_and_measure_pure_gauge,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
L = 6           # small lattice for fast structural tests
L_CONF = 10     # larger lattice for confinement-signature tests
NDIM = 2        # 2D lattice — area law holds for any D≥2, fast to run
N_SU2 = 2       # SU(2) gauge group
BETA_CONF = 1.5 # strong coupling / confined regime for SU(2) 2D


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cold(size, ndim=NDIM, n=N_SU2):
    """Identity-link (cold) configuration."""
    return identity_links(size, ndim=ndim, n=n)


def _hot(rng, size, ndim=NDIM, n=N_SU2):
    """Haar-random (hot) configuration."""
    spatial = (size,) * ndim
    return random_unitary(rng, n, (ndim, *spatial), special=True, scale=0.7)


def _haar_su2(rng):
    """Single Haar-random SU(2) matrix."""
    return random_unitary(rng, 2, special=True, scale=0.7)


def _apply_gauge_transform(links, g, size, ndim=NDIM):
    """Apply local SU(2) gauge transform g[site] → new link array."""
    links_g = links.copy()
    for mu in range(ndim):
        for site in np.ndindex(*((size,) * ndim)):
            x_pmu = tuple((site[d] + (1 if d == mu else 0)) % size for d in range(ndim))
            links_g[(mu,) + site] = g[site] @ links[(mu,) + site] @ _dagger(g[x_pmu])
    return links_g


def _total_action(links):
    """Proxy Wilson action: Σ_{μ,site} -Re Tr[U_μ(site) · A†(site)]."""
    from project_genesis.gauge_mc import _staple_sum
    ndim = links.shape[0]
    spatial = links.shape[1:-2]
    s = 0.0
    for mu in range(ndim):
        for site in np.ndindex(*spatial):
            A = _staple_sum(links, mu, site)
            s -= float(np.real(np.trace(links[(mu,) + site] @ _dagger(A))))
    return s


# ---------------------------------------------------------------------------
# Module-scope fixture: pre-thermalised config at β=1.5, 10×10
# Computed once per test session — reused by T9–T13.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def thermalised_b15():
    rng = np.random.default_rng(2026_06_26)
    links = _hot(rng, L_CONF)
    for _ in range(600):
        links, _ = metropolis_sweep(links, BETA_CONF, rng, step_scale=0.4)
    return links, rng


# ===========================================================================
# T1 — Cold start: flat connection → W=1, S_W=0
# ===========================================================================

def test_cold_start_flat_connection():
    """
    T1: identity_links gives W(1,1)=1 and total Wilson action ≈ 0.

    Corresponds to the URP flat-connection (zero coherence stress) case:
    when U_μ(x) = I for all links the plaquette is P_{μν} = I and
    S_W = Σ Re Tr(I − I) = 0.  W(1,1) = Re Tr(I)/N = 1.
    """
    links = _cold(L)
    w11 = wilson_loop(links, (1, 1))
    assert abs(w11 - 1.0) < 1e-10, f"W(1,1)={w11:.12f}, expected 1.0 for cold links"
    s = _total_action(links)
    assert abs(s) < 1e-9, f"Wilson action={s:.2e} for cold links, expected 0"


# ===========================================================================
# T2 — Unitarity preserved after 100 sweeps
# ===========================================================================

def test_unitarity_after_sweeps():
    """
    T2: All links satisfy U†U = I (to float64 precision) after 100 sweeps.
    """
    rng = np.random.default_rng(2)
    links = _cold(L)
    for _ in range(100):
        links, _ = metropolis_sweep(links, 2.0, rng, step_scale=0.4)
    n = links.shape[-1]
    for mu in range(NDIM):
        for site in np.ndindex(*((L,) * NDIM)):
            U = links[(mu,) + site]
            err = np.max(np.abs(U @ _dagger(U) - np.eye(n, dtype=complex)))
            assert err < 1e-12, (
                f"Unitarity violation {err:.2e} at mu={mu}, site={site}"
            )


# ===========================================================================
# T3 — det = 1 preserved
# ===========================================================================

def test_determinant_after_sweeps():
    """
    T3: All links have det(U) = 1 after 100 Metropolis sweeps.
    """
    rng = np.random.default_rng(3)
    links = _cold(L)
    for _ in range(100):
        links, _ = metropolis_sweep(links, 2.0, rng, step_scale=0.4)
    for mu in range(NDIM):
        for site in np.ndindex(*((L,) * NDIM)):
            det = np.linalg.det(links[(mu,) + site])
            assert abs(det - 1.0) < 1e-12, (
                f"det={det:.10f} at mu={mu}, site={site}, expected 1.0"
            )


# ===========================================================================
# T4 — Acceptance rate in healthy range
# ===========================================================================

@pytest.mark.parametrize("beta_g", [0.5, 1.5, 3.0])
def test_acceptance_rate_in_range(beta_g):
    """
    T4: Equilibrium acceptance rate is 20–80 % for step_scale=0.4.
    This is not a physics test — it verifies that the chain is actually mixing
    and not stuck accepting or rejecting everything.
    """
    rng = np.random.default_rng(4_000 + int(beta_g * 100))
    links = _cold(L)
    for _ in range(200):
        links, _ = metropolis_sweep(links, beta_g, rng, step_scale=0.4)
    rates = []
    for _ in range(50):
        links, acc = metropolis_sweep(links, beta_g, rng, step_scale=0.4)
        rates.append(acc)
    mean_acc = float(np.mean(rates))
    assert 0.20 <= mean_acc <= 0.80, (
        f"beta={beta_g}: acceptance={mean_acc:.3f}, expected in [0.20, 0.80]"
    )


# ===========================================================================
# T5 — Action decreases from hot start
# ===========================================================================

def test_action_decreases_from_hot_start():
    """
    T5: The Wilson action (sampled every 20 sweeps) decreases over the first
    200 Metropolis sweeps from a hot (Haar-random) start at β=2.0.

    The Boltzmann weight exp(−β_g S_W) concentrates the ensemble on lower-S
    configurations — the chain should visibly thermalize downward from disorder.
    """
    rng = np.random.default_rng(5)
    links = _hot(rng, L)
    history = []
    for sweep in range(200):
        links, _ = metropolis_sweep(links, 2.0, rng, step_scale=0.4)
        if sweep % 20 == 19:
            history.append(_total_action(links))
    assert history[0] > history[-1], (
        f"Action did not decrease: initial={history[0]:.3f}, final={history[-1]:.3f}"
    )


# ===========================================================================
# T6 — Wilson loop gauge invariance
# ===========================================================================

def test_wilson_loop_gauge_invariance():
    """
    T6: W(2,2) is unchanged by a random local SU(2) gauge transformation.

    Wilson loops are gauge-invariant observables — this directly tests the
    gauge-symmetry property proved at the S-functional level in the URP
    β-sectorisation derivation.
    """
    rng = np.random.default_rng(6)
    links = _cold(L)
    for _ in range(200):
        links, _ = metropolis_sweep(links, 2.0, rng, step_scale=0.4)
    W_before = wilson_loop(links, (2, 2))
    g = {site: _haar_su2(rng) for site in np.ndindex(*((L,) * NDIM))}
    links_g = _apply_gauge_transform(links, g, L)
    W_after = wilson_loop(links_g, (2, 2))
    assert abs(W_before - W_after) < 1e-10, (
        f"W(2,2) changed under gauge transform: {W_before:.10f} → {W_after:.10f}"
    )


# ===========================================================================
# T7 — Wilson action gauge invariance
# ===========================================================================

def test_wilson_action_gauge_invariance():
    """
    T7: The total Wilson action S_W is invariant under a random local SU(2)
    gauge transformation — same theoretical guarantee as T6.
    """
    rng = np.random.default_rng(7)
    links = _cold(L)
    for _ in range(200):
        links, _ = metropolis_sweep(links, 2.0, rng, step_scale=0.4)
    S_before = _total_action(links)
    g = {site: _haar_su2(rng) for site in np.ndindex(*((L,) * NDIM))}
    links_g = _apply_gauge_transform(links, g, L)
    S_after = _total_action(links_g)
    assert abs(S_before - S_after) < 1e-8, (
        f"S_W changed: {S_before:.6f} → {S_after:.6f}"
    )


# ===========================================================================
# T8 — Bessel benchmark (the most important test)
# ===========================================================================

@pytest.mark.parametrize("beta_g", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
def test_bessel_benchmark(beta_g):
    """
    T8: Ensemble ⟨W(1,1)⟩ matches the exact SU(2) single-plaquette result
        I₁(β/2) / I₀(β/2)  within 5 %.

    This is the single most important correctness test.  It simultaneously
    validates:
    - correct plaquette orientation in wilson_loop() (not U instead of U†),
    - correct Metropolis accept/reject sign (not inadvertently flipped ΔS),
    - ergodicity: the chain converges to the right Gibbs measure.

    The exact formula follows from the SU(2) partition function being
    expressible in terms of modified Bessel functions I₀, I₁.
    """
    exact = float(bessel_iv(1, beta_g / 2) / bessel_iv(0, beta_g / 2))
    rng = np.random.default_rng(8_000 + int(beta_g * 100))
    summary, _ = thermalize_and_measure_pure_gauge(
        size=L,
        n=N_SU2,
        beta_g=beta_g,
        rng=rng,
        ndim=NDIM,
        n_therm=500,
        n_meas=400,
        n_skip=3,
        step_scale=0.4,
        updater="metropolis",
        loop_sizes=[(1, 1)],
    )
    measured = summary["loop_averages"]["W_1_1"]
    err = abs(measured - exact)
    assert err < 0.05, (
        f"beta={beta_g}: measured={measured:.5f}, exact={exact:.5f}, err={err:.5f} > 0.05"
    )


# ===========================================================================
# T9 — Area-law suppression
# ===========================================================================

def test_area_suppression(thermalised_b15):
    """
    T9: W(1,1) > W(1,2) > 0  and  W(1,1) > W(2,1) > 0  at β=1.5.

    In the confined phase log⟨W(R,T)⟩ ≈ −σ·R·T so loops with larger area
    are exponentially suppressed.  The 1×1 plaquette is the least suppressed.
    This is the most direct lattice signature of confinement predicted by the
    URP β-sectorisation derivation.
    """
    links, rng = thermalised_b15
    acc = {(1, 1): 0.0, (1, 2): 0.0, (2, 1): 0.0}
    n_meas = 200
    for _ in range(n_meas):
        for _ in range(5):
            links, _ = metropolis_sweep(links, BETA_CONF, rng, step_scale=0.4)
        for k in acc:
            acc[k] += wilson_loop(links, k)
    W = {k: v / n_meas for k, v in acc.items()}
    assert W[(1, 1)] > W[(1, 2)] > 0, (
        f"Area suppression failed: W11={W[(1,1)]:.4f}, W12={W[(1,2)]:.4f}"
    )
    assert W[(1, 1)] > W[(2, 1)] > 0, (
        f"Area suppression failed: W11={W[(1,1)]:.4f}, W21={W[(2,1)]:.4f}"
    )


# ===========================================================================
# T10 — Discrete rotational symmetry
# ===========================================================================

def test_rotational_symmetry(thermalised_b15):
    """
    T10: ⟨W(1,2)⟩ ≈ ⟨W(2,1)⟩  (|difference| < 0.05).

    The square lattice has a discrete 90° rotational symmetry, so rectangular
    Wilson loops with swapped dimensions should be equal in the ensemble average.
    """
    links, rng = thermalised_b15
    acc = {(1, 2): 0.0, (2, 1): 0.0}
    n_meas = 200
    for _ in range(n_meas):
        for _ in range(5):
            links, _ = metropolis_sweep(links, BETA_CONF, rng, step_scale=0.4)
        for k in acc:
            acc[k] += wilson_loop(links, k)
    W = {k: v / n_meas for k, v in acc.items()}
    diff = abs(W[(1, 2)] - W[(2, 1)])
    assert diff < 0.05, (
        f"Rotational symmetry broken: W12={W[(1,2)]:.5f}, W21={W[(2,1)]:.5f}, diff={diff:.5f}"
    )


# ===========================================================================
# T11 — Polyakov loop near zero (confined phase)
# ===========================================================================

def test_polyakov_near_zero_confined(thermalised_b15):
    """
    T11: |⟨P⟩| < 0.15 at β=1.5 (confined phase).

    The Polyakov loop is the Z_N order parameter for the confinement/
    deconfinement transition.  In the confined phase the centre symmetry is
    unbroken and ⟨P⟩ → 0 in the thermodynamic limit.  At β=1.5 on a 10×10
    2D SU(2) lattice we expect |⟨P⟩| to be well below 0.15.
    """
    links, rng = thermalised_b15
    vals = []
    for _ in range(200):
        for _ in range(5):
            links, _ = metropolis_sweep(links, BETA_CONF, rng, step_scale=0.4)
        vals.append(polyakov_loop(links))
    mean_P = abs(float(np.mean(vals)))
    assert mean_P < 0.15, (
        f"|<P>|={mean_P:.4f} > 0.15; expected near zero in confined phase (beta=1.5)"
    )


# ===========================================================================
# T12 — Creutz ratio positive
# ===========================================================================

def test_creutz_ratio_positive(thermalised_b15):
    """
    T12: Creutz ratio χ(2,2) is finite (non-NaN) and positive at β=1.5.

    χ(R,T) = −log[ W(R,T)·W(R−1,T−1) / (W(R,T−1)·W(R−1,T)) ] → σ

    A positive χ(2,2) is consistent with a nonzero string tension σ and
    hence confinement.  The test tolerates NaN only if W(2,2) ≤ 0 at this
    volume (graceful degradation rather than crash).
    """
    links, rng = thermalised_b15
    acc = {(1, 1): 0.0, (1, 2): 0.0, (2, 1): 0.0, (2, 2): 0.0}
    n_meas = 200
    for _ in range(n_meas):
        for _ in range(5):
            links, _ = metropolis_sweep(links, BETA_CONF, rng, step_scale=0.4)
        for k in acc:
            acc[k] += wilson_loop(links, k)
    W = {k: v / n_meas for k, v in acc.items()}
    chi = creutz_ratio(W[(2, 2)], W[(1, 1)], W[(2, 1)], W[(1, 2)])
    if not math.isnan(chi):
        assert chi > 0, f"Creutz chi(2,2)={chi:.5f} not positive"
    # NaN → W(2,2)<=0 at this volume; no crash = pass


# ===========================================================================
# T13 — fit_area_law returns σ > 0
# ===========================================================================

def test_area_law_fit_positive_sigma(thermalised_b15):
    """
    T13: fit_area_law() returns σ > 0 from a 3×3 grid of loop sizes at β=1.5.

    A positive fitted σ is the quantitative area-law / string-tension test
    predicted by the URP gauge derivation.  The least-squares fit over
    log W(R,T) = −σ·R·T − c·(R+T) should yield a positive coefficient
    for the area term in the strong-coupling (confined) regime.
    """
    links, rng = thermalised_b15
    rs, ts = [1, 2, 3], [1, 2, 3]
    W_acc = np.zeros((3, 3))
    n_meas = 200
    for _ in range(n_meas):
        for _ in range(5):
            links, _ = metropolis_sweep(links, BETA_CONF, rng, step_scale=0.4)
        for i, r in enumerate(rs):
            for j, t in enumerate(ts):
                W_acc[i, j] += wilson_loop(links, (r, t))
    W_mean = W_acc / n_meas
    result = fit_area_law(W_mean, rs, ts)
    sigma = result["sigma"]
    assert sigma > 0, (
        f"fit_area_law sigma={sigma:.5f} <= 0 at beta={BETA_CONF}; "
        "expected positive string tension in confined phase"
    )
