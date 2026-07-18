"""Finite-speed capacity dynamics: the κ field with its own update rate.

Everywhere else in the framework the capacity field is **parabolic** —
``∂_t κ = D∇²κ + r(κ₀ − κ) − c·load·κ`` — and in the gravity experiments it
is relaxed adiabatically, so κ-gravity acts *instantaneously*: a named piece
of missing physics.  This module gives κ the minimal honest extension — a
finite update latency ``τ`` (the telegrapher form):

    τ·∂²_t κ + ∂_t κ = D∇²κ + r(κ₀ − κ) − c·load·κ .

Three consequences are derivable and testable:

- **A causal cone.**  High-frequency disturbances propagate at the finite
  front speed ``c_κ = √(D/τ)`` — the field's own speed limit, set by its
  update rate; ``τ → 0`` recovers the parabolic (instantaneous) field.
- **The Debye mass propagates.**  Linearising about the loaded homogeneous
  steady state ``κ̄ = r·κ₀/(r + c·ρ)`` gives
  ``τ·δκ̈ + δκ̇ = D∇²δκ − (r + c·ρ)·δκ``: plane waves oscillate at

      ω²(k) = (D·k² + r + c·ρ)/τ − 1/(4τ²) ,   amplitude ∝ e^{−t/2τ} ,

  i.e. the κ-wave carries a **mass** ``m² = r + c·ρ`` — the *same* matter
  term that screens static κ-gravity (`capacity_gravity`, the screening-knee
  and local-screening experiments).  Waves are massive (slow, gapped) inside
  matter; the massless channel exists exactly where ``r + c·ρ → 0``.
- **Overdamping.**  Modes with ``4τ(Dk² + r + cρ) < 1`` do not oscillate —
  the parabolic regime survives inside the wave theory at long wavelength
  and small τ.

The integrator is explicit (kick–drift on ``(κ, κ̇)``), unclipped — the wave
sector is a *linear-response* instrument; keep amplitudes small.  Numerical
causality is bounded by one cell per step regardless of parameters, so keep
``c_κ·dt`` below the CFL bound (checked in ``step_capacity_wave``).
"""

from __future__ import annotations

import numpy as np

from .capacity_gravity import (capacity_free_energy, gaussian_load,
                               relax_capacity)
from .multiphase import periodic_laplacian


def wave_speed(kappa_diffusion: float, tau: float) -> float:
    """The κ front speed ``c_κ = √(D/τ)`` — the field's own update rate."""
    return float(np.sqrt(kappa_diffusion / tau))


def wave_mass2(kappa_recovery: float, kappa_consumption: float,
               rho: float) -> float:
    """The κ-wave mass ``m² = r + c·ρ`` — the Debye term, propagating."""
    return float(kappa_recovery + kappa_consumption * rho)


def dispersion_omega(k: float, *, tau: float, kappa_diffusion: float = 1.0,
                     kappa_recovery: float = 0.05,
                     kappa_consumption: float = 2.0,
                     rho: float = 0.0) -> float:
    """Oscillation frequency ``ω(k)`` of the linearised loaded mode.

    ``ω² = (D·k² + m²)/τ − 1/(4τ²)``; returns NaN where the mode is
    overdamped (no oscillation).
    """
    m2 = wave_mass2(kappa_recovery, kappa_consumption, rho)
    w2 = (kappa_diffusion * k * k + m2) / tau - 1.0 / (4.0 * tau * tau)
    return float(np.sqrt(w2)) if w2 > 0 else float("nan")


def steady_kappa(kappa_recovery: float, kappa_consumption: float, rho: float,
                 kappa_baseline: float = 1.0) -> float:
    """Homogeneous loaded steady state ``κ̄ = r·κ₀/(r + c·ρ)``."""
    return float(kappa_recovery * kappa_baseline
                 / (kappa_recovery + kappa_consumption * rho))


def exclusion_energy_density(rho, *, kappa_recovery: float,
                             kappa_consumption: float):
    """e(ρ) = c²rρ²/((r+cρ)(r+2cρ)) — the DERIVED exclusion energy density.

    The homogeneous steady-state capacity free energy density at load ρ
    (κ₀ = 1, gradient terms neglected) is ``F(ρ) = r·c·ρ/(2(r + cρ))`` —
    concave in ρ, which is exactly why merging is cheap.  The exclusion
    (no-cloning) principle prices a stack of two identical copies at the
    *extensive* cost ``2F(ρ)`` rather than the concave ``F(2ρ)``; the
    exclusion energy is the gap ``E_x(ρ) = 2F(ρ) − F(2ρ)``, which closes
    to the form above.  No free parameters: dilute limit ``e ≈ (b/2)ρ²``
    with ``b = 2c²/r``; saturates at ``r/2`` per site as ``ρ → ∞``;
    ``E_x(2ρ) > 2E_x(ρ)`` (clone refusal) exactly for ``ρ < r/(2c)``.
    Equivalently ``b(ρ) = 2c²ℓ²(ρ)/D`` with the loaded screening length
    ``ℓ² = D/(r + cρ)`` — degeneracy stiffness set by the local capacity
    range.  ``rho`` may be a numpy array.
    """
    rho = np.asarray(rho, dtype=float)
    r, c = kappa_recovery, kappa_consumption
    return (c * c * r * rho ** 2
            / ((r + c * rho) * (r + 2.0 * c * rho)))


def exclusion_energy_derivative(rho, *, kappa_recovery: float,
                                kappa_consumption: float):
    """e'(ρ) = c²r²ρ(3cρ+2r)/((r+cρ)²(r+2cρ)²) — exact gradient of e(ρ).

    Positive for all ``ρ > 0`` — the per-site gradient always pushes
    overlap apart.  Note it peaks inside the refusal window
    (``ρ ~ r/(2c)``) and fades as ``3r²/(4cρ²)`` in the saturated
    regime, so the net force on a dense blob is skirt-weighted:
    measured to INVERT (net attraction) at operating-point densities
    above the window — see ``Docs/Deriving_The_Exclusion_Coefficient.md``.
    ``rho`` may be a numpy array.
    """
    rho = np.asarray(rho, dtype=float)
    r, c = kappa_recovery, kappa_consumption
    return (c * c * r * r * rho * (3.0 * c * rho + 2.0 * r)
            / ((r + c * rho) ** 2 * (r + 2.0 * c * rho) ** 2))


# ---------------------------------------------------------------------------
# Part II — the gradient terms: the exclusion gap of the FULL F[κ]
#
# Part I derived the exclusion term from the HOMOGENEOUS F(ρ), dropping the
# gradient energy (D/2)|∇κ|².  With gradients kept, the relaxed minimum of
# the capacity free energy at fixed load ρ is, in linear response
# (cρκ₀ ≪ r),
#
#     E(ρ) = (cκ₀²/2)·∫ρ  −  (c²κ₀²/2)·(ρ, G_r ρ)  +  O(c³) ,
#
# where (r − D∇²)G_r = δ is the screened (Yukawa) kernel of range
# ξ = √(D/r) — the same kernel that mediates κ-gravity.  The full-overlap
# gap of two identical copies is then a POSITIVE, NONLOCAL self-interaction
#
#     G(ρ₁) := 2E(ρ₁) − E(2ρ₁) = c²κ₀²·(ρ₁, G_r ρ₁)  >  0 .
#
# Exactly (beyond linear response) E(A) = min_κ F[κ; A·ρ₁] is a minimum
# over functions AFFINE in A, hence concave in A — so G(A) ≥ 0 at every
# amplitude: no sign flip of the total gap is possible.  The functions
# below measure this gap with the framework's own instruments.
# ---------------------------------------------------------------------------


def _screened_kernel_hat(shape, *, kappa_diffusion: float,
                         kappa_recovery: float) -> np.ndarray:
    """``Ĝ(k) = 1/(r + D|k|²)`` on the periodic lattice.

    ``|k|²`` is the symbol ``4Σ_ax sin²(k_ax/2)`` of the repo's 5-point
    ``periodic_laplacian``, so the kernel inverts exactly the operator the
    capacity relaxer descends.  The k = 0 mode is finite because r > 0:
    ``Ĝ(0) = 1/r``.
    """
    ks = [2.0 * np.pi * np.fft.fftfreq(int(n)) for n in shape]
    grids = np.meshgrid(*ks, indexing="ij")
    k2 = sum(4.0 * np.sin(g / 2.0) ** 2 for g in grids)
    return 1.0 / (kappa_recovery + kappa_diffusion * k2)


def screened_green_function(shape, *, kappa_diffusion: float,
                            kappa_recovery: float) -> np.ndarray:
    """The lattice Green's function ``G_r`` of ``(r − D∇²)`` in real space.

    ``(r − D∇²)G_r = δ`` on the periodic lattice — the screened/Yukawa
    kernel of range ``ξ = √(D/r)`` that mediates both κ-gravity and, in
    linear response, the exclusion self-interaction.  Constructed by FFT.
    """
    return np.fft.ifftn(_screened_kernel_hat(
        shape, kappa_diffusion=kappa_diffusion,
        kappa_recovery=kappa_recovery)).real


def apply_screened_kernel(rho, *, kappa_diffusion: float,
                          kappa_recovery: float) -> np.ndarray:
    """``(G_r ρ)(x)`` — the screened kernel applied to a load, by FFT."""
    rho = np.asarray(rho, dtype=float)
    return np.fft.ifftn(np.fft.fftn(rho) * _screened_kernel_hat(
        rho.shape, kappa_diffusion=kappa_diffusion,
        kappa_recovery=kappa_recovery)).real


def linear_response_exclusion_gap(rho, *, kappa_diffusion: float,
                                  kappa_recovery: float,
                                  kappa_consumption: float,
                                  kappa_baseline: float = 1.0) -> float:
    """``G_LR(ρ) = c²κ₀²·(ρ, G_r ρ)`` — the linear-response full-overlap gap.

    The gap ``2E(ρ) − E(2ρ)`` of the relaxed capacity free energy, to
    leading (quadratic) order in the load: positive for every nonzero ρ —
    a nonlocal exclusion self-interaction through the screened kernel.
    """
    c, k0 = kappa_consumption, kappa_baseline
    gr = apply_screened_kernel(rho, kappa_diffusion=kappa_diffusion,
                               kappa_recovery=kappa_recovery)
    return float(c * c * k0 * k0 * np.sum(np.asarray(rho, dtype=float) * gr))


def duplicated_load(load1, load2):
    """``ρ_dup(x) = min(ρ₁, ρ₂)(x)`` — the cloned (common) component of a pair.

    For two IDENTICAL copies, the pointwise minimum is exactly the load
    that is present twice — the duplicated fraction whose stack the
    exclusion principle prices at the extensive cost.  (Exact for
    identical copies only; for non-identical loads it is the maximal
    common component — the same maximal-identicality convention as the
    Part-I term.)
    """
    return np.minimum(load1, load2)


def exclusion_gap_full(rho_dup, *, kappa_diffusion: float = 1.0,
                       kappa_recovery: float = 0.02,
                       kappa_consumption: float = 0.8,
                       kappa_baseline: float = 1.0) -> float:
    """``E_x = 2E(ρ_dup) − E(2ρ_dup)`` — the full-functional exclusion gap.

    ``E(ρ) = F[κ̄[ρ]]`` is the RELAXED capacity free energy at fixed load
    (``relax_capacity`` + ``capacity_free_energy``, the repo's own
    instruments) — gradient energy included, i.e. the gradient-corrected
    counterpart of the Part-I homogeneous gap ``∫e(ρ)``.  Non-negative for
    every load (E is concave in the load's amplitude as a minimum over
    affine functions), zero when ``ρ_dup = 0``.
    """
    fkw = dict(kappa_diffusion=kappa_diffusion,
               kappa_recovery=kappa_recovery,
               kappa_consumption=kappa_consumption,
               kappa_baseline=kappa_baseline)

    def _e(load):
        kappa = relax_capacity(load, **fkw)
        return capacity_free_energy(kappa, load, **fkw)

    return 2.0 * _e(rho_dup) - _e(2.0 * rho_dup)


# ---------------------------------------------------------------------------
# Part III — the load that can tell same from different (labelled loads)
#
# min(ρ₁, ρ₂) prices ALL overlap as duplication: the load field cannot
# tell same-distinction from different-distinction stacking.  But the gap
# 2F(ρ) − F(2ρ) exactly cancels the concavity (sharing) discount — for
# IDENTICAL overlap only; different distinctions legitimately keep the
# discount (that discount IS binding).  The labelled load makes exclusion
# identity-selective: each mass carries a label vector w_i over
# distinction types (weights ≥ 0, sum 1), the capacity field responds to
# the TOTAL load as before (gravity is identity-blind), and exclusion
# applies per SHARED type only: ρ_dup^(t) = min(w₁ₜρ₁, w₂ₜρ₂),
# E_x = Σ_t [2E(ρ_dup^(t)) − E(2ρ_dup^(t))].  Identical labels recover
# min(ρ₁, ρ₂); orthogonal labels price nothing (the no-exclusion
# control); a shared fraction φ splits each blob φ·ρ_i common +
# (1−φ)·ρ_i private.
# ---------------------------------------------------------------------------


def shared_fraction_labels(share: float):
    """The scalar special case of the labelled load: the (labels₁, labels₂)
    pair for shared fraction ``share`` (φ).

    Each blob's load splits ``φ·ρ_i`` on the COMMON type plus
    ``(1−φ)·ρ_i`` on its OWN private type::

        labels₁ = {"shared": φ, "private_1": 1 − φ}
        labels₂ = {"shared": φ, "private_2": 1 − φ}

    φ = 1 is the identical-label case (the whole pair is one shared type
    — the base ``contact_full`` form); φ = 0 the orthogonal-label case
    (no shared type, no exclusion).  Label vectors are dicts mapping a
    distinction type to its weight; weights are ≥ 0 and sum to 1.
    """
    share = float(share)
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"share must lie in [0, 1], got {share}")
    return ({"shared": share, "private_1": 1.0 - share},
            {"shared": share, "private_2": 1.0 - share})


def _check_labels(labels, name: str) -> None:
    """Label-vector sanity: weights ≥ 0, sum 1 (a distribution over types)."""
    total = 0.0
    for t, w in labels.items():
        w = float(w)
        if w < 0.0:
            raise ValueError(f"{name} has a negative weight on type {t!r}: "
                             f"{w} (label weights are ≥ 0).")
        total += w
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{name} weights sum to {total}, not 1 — a label "
                         "vector is a distribution over distinction types.")


def shared_duplicated_components(load1, load2, labels1, labels2):
    """``[ρ_dup^(t)]`` — the cloned component of each SHARED distinction type.

    For every type ``t`` both blobs carry with positive weight,
    ``ρ_dup^(t) = min(w₁ₜ·ρ₁, w₂ₜ·ρ₂)`` (pointwise, exact min — the
    statics convention, matching ``duplicated_load``).  Types carried by
    only one blob are private: nothing is duplicated, nothing is priced.
    Returns the list of shared-type components (empty for orthogonal
    labels).
    """
    _check_labels(labels1, "labels1")
    _check_labels(labels2, "labels2")
    load1 = np.asarray(load1, dtype=float)
    load2 = np.asarray(load2, dtype=float)
    comps = []
    for t, w1 in labels1.items():
        w2 = labels2.get(t, 0.0)
        if w1 > 0.0 and w2 > 0.0:
            comps.append(np.minimum(float(w1) * load1, float(w2) * load2))
    return comps


def exclusion_gap_labelled(load1, load2, labels1, labels2, *,
                           kappa_diffusion: float = 1.0,
                           kappa_recovery: float = 0.02,
                           kappa_consumption: float = 0.8,
                           kappa_baseline: float = 1.0) -> float:
    """``E_x = Σ_t [2E(ρ_dup^(t)) − E(2ρ_dup^(t))]`` — the labelled gap.

    The gradient-corrected exclusion energy of a labelled pair: the
    full-functional gap (``exclusion_gap_full``) of each shared-type
    cloned component, summed over shared types.  Identical labels
    (φ = 1) recover the base ``2E(min(ρ₁, ρ₂)) − E(2·min(ρ₁, ρ₂))``
    bitwise; orthogonal labels give exactly 0 — same-distinction overlap
    loses the sharing discount, different-distinction overlap keeps it.
    """
    fkw = dict(kappa_diffusion=kappa_diffusion,
               kappa_recovery=kappa_recovery,
               kappa_consumption=kappa_consumption,
               kappa_baseline=kappa_baseline)
    return float(sum(exclusion_gap_full(comp, **fkw)
                     for comp in shared_duplicated_components(
                         load1, load2, labels1, labels2)))


def _relax_functional(kappa: np.ndarray, load: np.ndarray, *,
                      kappa_diffusion: float = 1.0,
                      kappa_recovery: float = 0.02,
                      kappa_consumption: float = 0.8,
                      kappa_baseline: float = 1.0) -> float:
    """``F*[κ] = Σ_x [(D/2)κ(−∇²)κ + (r/2)(κ−κ₀)² + (c/2)·load·κ²]``.

    The functional the capacity relaxer ACTUALLY descends: integration by
    parts of the gradient term against the repo's 5-point
    ``periodic_laplacian``.  It differs from ``capacity_free_energy``
    (central-difference gradient) only at O(stencil²) — but the relaxed
    field is EXACTLY its minimum, so forces derived from it through the
    envelope theorem are exact to solver tolerance (measured: force vs
    finite-difference energy agree to 1 part in 1e4; against the
    central-difference functional the residual is ~1%).  Used for the
    ``contact_full`` energy bookkeeping.
    """
    quad = 0.5 * kappa_diffusion * float(
        np.sum(kappa * (-periodic_laplacian(kappa))))
    rec = 0.5 * kappa_recovery * float(
        np.sum((kappa - kappa_baseline) ** 2))
    cons = 0.5 * kappa_consumption * float(np.sum(load * kappa * kappa))
    return quad + rec + cons


def _solve_relaxed(load: np.ndarray, kappa_init: np.ndarray | None = None,
                   *, kappa_diffusion: float = 1.0,
                   kappa_recovery: float = 0.02,
                   kappa_consumption: float = 0.8,
                   kappa_baseline: float = 1.0,
                   rtol: float = 1e-12, max_iters: int = 5000) -> np.ndarray:
    """Solve ``(r + c·load − D∇²)κ = rκ₀`` — the relaxed capacity fixed point.

    The relaxer's steady state is the solution of a LINEAR system in κ
    (the load is fixed), so it can be solved directly: conjugate gradient
    on the symmetric positive-definite operator (bounded below by r > 0),
    optionally warm-started from ``kappa_init``.  Same fixed point as
    ``relax_capacity`` (validated against it to its own tolerance in
    ``tests/test_exclusion_gradient.py``), ~50× faster per solve — the
    ``contact_full`` force path needs two solves per timestep.  The
    solution lies in (0, κ₀), so the relaxer's clip is inactive there.
    """
    r, c, k0 = kappa_recovery, kappa_consumption, float(kappa_baseline)
    src = c * load

    def _apply(k):
        return r * k + src * k - kappa_diffusion * periodic_laplacian(k)

    b = np.full_like(load, r * k0)
    x = np.full_like(load, k0) if kappa_init is None else kappa_init.copy()
    res = b - _apply(x)
    p = res.copy()
    rs = float(np.sum(res * res))
    bnorm = float(np.sum(b * b))
    for _ in range(max_iters):
        if rs <= rtol * rtol * bnorm:
            break
        ap = _apply(p)
        alpha = rs / float(np.sum(p * ap))
        x = x + alpha * p
        res = res - alpha * ap
        rs_new = float(np.sum(res * res))
        p = res + (rs_new / rs) * p
        rs = rs_new
    return x


def _smoothed_min(a, b, eps: float):
    """``(a+b)/2 − ½√((a−b)² + 4ε²)`` — a smooth min (smooth-abs form).

    Converges uniformly to ``min(a, b)`` as ε → 0 (error ≤ ε); smooth
    everywhere, so the ``contact_full`` force is the exact gradient of the
    smoothed functional it is booked against.  At exact ties (a = b — the
    symmetry plane of an equal binary, which LANDS on lattice sites) it
    assigns the symmetric 50/50 subgradient, which the hard
    ``min + indicator`` rule does not (measured: the hard rule breaks
    Newton's third law on the tie plane by up to ~35% of the force).
    """
    return 0.5 * (a + b) - 0.5 * np.sqrt((a - b) ** 2 + 4.0 * eps * eps)


def _smoothed_min_weight(a, b, eps: float):
    """``∂/∂a smoothed_min(a, b) = ½ − (a−b)/(2√((a−b)² + 4ε²)) ∈ (0, 1)``."""
    return 0.5 - 0.5 * (a - b) / np.sqrt((a - b) ** 2 + 4.0 * eps * eps)


def _gaussian_load_and_grad(shape, center, width: float, amplitude: float):
    """``(ρ, ∇ρ)`` for one periodic Gaussian blob — analytic load gradient.

    Same periodic-distance construction as
    ``capacity_gravity.gaussian_load``; the gradient is the exact
    ``∂ρ/∂x_ax = −ρ·(x_ax − R_ax)/width²`` (not a finite difference), so
    forces built from it are the exact gradient of the load-parameterised
    energy to solver tolerance.
    """
    shape = tuple(int(s) for s in shape)
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    disp = [(((g - c0 + s / 2.0) % s) - s / 2.0)
            for g, c0, s in zip(grids, center, shape)]
    r2 = sum(d * d for d in disp)
    rho = amplitude * np.exp(-r2 / (2.0 * width ** 2))
    return rho, [rho * (-d / (width * width)) for d in disp]


def step_capacity_wave(kappa: np.ndarray, kappa_dot: np.ndarray,
                       load: np.ndarray | float, *, tau: float = 1.0,
                       kappa_diffusion: float = 1.0,
                       kappa_recovery: float = 0.05,
                       kappa_consumption: float = 2.0,
                       kappa_baseline: float = 1.0,
                       dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """One kick–drift step of the telegrapher κ dynamics; returns (κ, κ̇).

    Explicit and unclipped (linear-response instrument).  Raises if the CFL
    bound ``c_κ·dt ≤ 0.5`` (unit cells) is violated.
    """
    if wave_speed(kappa_diffusion, tau) * dt > 0.5:
        raise ValueError("CFL violated: reduce dt or raise tau "
                         f"(c_kappa*dt = {wave_speed(kappa_diffusion, tau) * dt:.3f})")
    force = (kappa_diffusion * periodic_laplacian(kappa)
             + kappa_recovery * (kappa_baseline - kappa)
             - kappa_consumption * load * kappa)
    # exponential integrator for the stiff damping part: exact for
    # τ·κ̈ = force − κ̇ with force frozen over the step, unconditionally
    # stable for any dt/τ, and τ → 0 recovers the parabolic flow
    # (κ̇ → force).  A plain kick is unstable/degenerate for dt ≳ τ.
    decay = np.exp(-dt / tau)
    kappa_dot = force + (kappa_dot - force) * decay
    kappa = kappa + dt * kappa_dot
    return kappa, kappa_dot


def step_capacity_parabolic(kappa: np.ndarray, load: np.ndarray | float, *,
                            kappa_diffusion: float = 1.0,
                            kappa_recovery: float = 0.05,
                            kappa_consumption: float = 2.0,
                            kappa_baseline: float = 1.0,
                            dt: float = 0.1) -> np.ndarray:
    """The τ → 0 control: one unclipped step of the parabolic κ flow."""
    return kappa + dt * (kappa_diffusion * periodic_laplacian(kappa)
                         + kappa_recovery * (kappa_baseline - kappa)
                         - kappa_consumption * load * kappa)


def evolve_inertial_retarded(positions, velocities, masses, *, shape,
                             width: float = 2.5, tau: float = 1.0,
                             dt: float = 0.1, steps: int = 1000,
                             record_every: int = 10,
                             probes=None, kappa0=None,
                             contact_b: float = 0.0,
                             contact_derived: bool = False,
                             contact_full: bool = False,
                             contact_full_eps: float = 1e-4,
                             contact_full_share: float = 1.0,
                             contact_full_labels=None,
                             kappa_diffusion: float = 1.0,
                             kappa_recovery: float = 0.05,
                             kappa_consumption: float = 2.0,
                             kappa_baseline: float = 1.0) -> dict:
    """Inertial masses coupled to the finite-speed (telegrapher) κ field.

    The retarded counterpart of ``capacity_dynamics.evolve_inertial``: the
    field is *not* relaxed adiabatically — ``(κ, κ̇)`` co-evolve with the
    masses, which feel the same envelope force ``F_i = −c·Σ load_i·κ·∇κ``
    from the field **as it currently is** (lagging, wavy, retarded).  The
    field starts from the parabolic steady state of the initial positions
    (κ̇ = 0), so all early dissipation is source motion, not transient.

    Exact energy bookkeeping (multiply the field equation by κ̇):

        d/dt [ T_mass + F[κ] + ∫(τ/2)κ̇² ] = −∫κ̇²  ≤ 0 ,

    so the recorded ``energy`` is a Lyapunov function — mechanical energy
    lost by the masses is carried by the field and dissipated at exactly
    the recorded rate ``dissipation``.  Adiabatic κ-gravity conserves
    ``T + F``; retardation makes moving masses **drag and radiate**.

    ``probes``: optional list of integer grid cells ``(i, j)``; κ at each
    is recorded **every step** into ``probe_kappa`` (shape steps × n) —
    the waveform channel for spectroscopy of the emitted disturbance.

    ``contact_b``: the **exclusion (no-cloning) term** — an energy
    ``E_x = (b/2)·Σ_x load_tot²`` added to the matter sector.  Zero for
    separated structures, quadratic where identical distinction stacks:
    duplicating a distinction in place costs capacity, so overlapping
    loads repel below the footprint scale — degeneracy pressure.  The
    force is the exact gradient ``F_i = −∂E_x/∂R_i``, and ``E_x`` is
    included in ``energy``, so the Lyapunov law is preserved.

    ``contact_derived``: the framework-DERIVED exclusion term — replace
    the hand-picked constant ``b`` with the density-dependent form the
    framework's own counting implies,
    ``E_x = Σ_x e(load_tot)`` with ``e(ρ) = 2F(ρ) − F(2ρ)`` the
    extensivity gap of the homogeneous capacity free energy (see
    ``exclusion_energy_density``).  The force is the exact gradient
    ``F_i = Σ_x e'(load_tot)·∇load_i`` — the same sign convention and
    bookkeeping as ``contact_b`` (which is the special case
    ``e'(ρ) = b·ρ``) — and ``E_x`` is included in ``energy``, so the
    Lyapunov law is preserved by the same argument.  Mutually exclusive
    with a nonzero ``contact_b``.

    ``contact_full``: the gradient-corrected derived term (Part II) —
    the exclusion energy of the FULL functional, applied to the
    duplicated (cloned) fraction of the pair:

        E_x = 2E(ρ_dup) − E(2ρ_dup) ,   ρ_dup = min(ρ₁, ρ₂) ,

    with ``E(ρ)`` the relaxed minimum of the capacity free energy at
    fixed load — gradient energy included, so the κ-field's own
    (D/2)|∇κ|² cost of the overlap is priced in.  This is the exact
    binary instrument: it vanishes for separated structures
    (ρ_dup → 0), equals the full-overlap gap ``2E(ρ₁) − E(2ρ₁)`` at
    coincidence, and is non-negative everywhere.  The force is the exact
    gradient w.r.t. the blob positions by the envelope theorem:
    ``δE/δρ(x) = (c/2)κ̄(x)²`` at the relaxed field, so with
    ``w = c·(κ̄[ρ_dup]² − κ̄[2ρ_dup]²)``,

        F_i = Σ_x w·(∂ρ_dup/∂ρ_i)·∇ρ_i ,

    computed with ANALYTIC load gradients and the two auxiliary relaxed
    fields solved directly (conjugate gradient on the relaxer's linear
    fixed point, ``_solve_relaxed``; the exclusion sector is adiabatic,
    like ``contact_b``'s instantaneous term — the retarded co-evolving κ
    still mediates the gravity part).  The min is taken in its smoothed
    form (``_smoothed_min`` with ``contact_full_eps``) so the force is
    everywhere the exact gradient of the recorded E_x — the hard
    min + indicator rule mis-splits the symmetry tie-plane, which lands
    on lattice sites for an equal binary (measured: up to ~35%
    third-law violation; the smoothed form conserves momentum to machine
    precision and matches finite-difference E_x to 1 part in 1e4).
    ``E_x`` is booked in ``energy`` through ``_relax_functional`` — the
    functional the relaxer actually descends — so the Lyapunov law is
    preserved exactly as for ``contact_b``.  Binary instrument:
    requires exactly two masses, and is mutually exclusive with
    ``contact_b`` and ``contact_derived``.

    ``contact_full_share`` / ``contact_full_labels``: the labelled load
    (Part III) — the load that can tell same-distinction from
    different-distinction stacking.  Each mass carries a label vector
    over distinction types; the capacity field responds to the TOTAL
    load as before (gravity is identity-blind), while exclusion prices
    only the SHARED-type components:

        E_x = Σ_t [2E(ρ_dup^(t)) − E(2ρ_dup^(t))] ,
        ρ_dup^(t) = smoothed_min(w₁ₜ·ρ₁, w₂ₜ·ρ₂) ,

    summed over the types t both blobs carry (``shared_fraction_labels``
    / ``shared_duplicated_components`` / ``exclusion_gap_labelled``).
    ``contact_full_share`` is the scalar special case: each blob splits
    φ·ρ_i common-type + (1−φ)·ρ_i private-type.  The default φ = 1 is
    exactly the unlabelled ``contact_full`` (one shared type of weight
    1 — bitwise the same path); φ = 0 is the no-exclusion control (no
    shared type — E_x = 0 and zero exclusion force, identically).
    ``contact_full_labels`` takes a general ``(labels₁, labels₂)`` pair
    of dicts {type: weight} (weights ≥ 0, sum 1) and overrides
    ``contact_full_share``.  The force stays the exact gradient of the
    recorded E_x per shared type (same envelope pair, same smoothed min,
    same Lyapunov bookkeeping — each shared type contributes its own two
    auxiliary relaxed fields).
    """
    n_contact = ((1 if contact_b else 0) + int(contact_derived)
                 + int(contact_full))
    if n_contact > 1:
        raise ValueError("contact_b, contact_derived and contact_full "
                         "are mutually exclusive: pick one exclusion "
                         "bookkeeping (the derived and full forms have "
                         "no free b).")

    pos = np.asarray(positions, dtype=float).copy()
    vel = np.asarray(velocities, dtype=float).copy()
    m = np.asarray(masses, dtype=float)
    c = kappa_consumption
    if contact_full and len(m) != 2:
        raise ValueError("contact_full is the binary instrument: the "
                         "duplicated fraction ρ_dup = min(ρ₁, ρ₂) is "
                         "defined for exactly two masses.")
    if not contact_full and (contact_full_labels is not None
                             or float(contact_full_share) != 1.0):
        raise ValueError("contact_full_share / contact_full_labels modify "
                         "the contact_full exclusion term; pass "
                         "contact_full=True to use them.")
    # the shared-type weights (w₁ₜ, w₂ₜ) of the labelled load: the types
    # BOTH blobs carry with positive weight — exclusion prices only
    # these.  Default share 1 is exactly the unlabelled instrument.
    shared_w = []
    if contact_full:
        if contact_full_labels is not None:
            labels1, labels2 = contact_full_labels
        else:
            labels1, labels2 = shared_fraction_labels(contact_full_share)
        _check_labels(labels1, "contact_full_labels[0]")
        _check_labels(labels2, "contact_full_labels[1]")
        shared_w = [(float(w1), float(labels2.get(t, 0.0)))
                    for t, w1 in labels1.items()
                    if w1 > 0.0 and labels2.get(t, 0.0) > 0.0]
    fkw = dict(kappa_diffusion=kappa_diffusion, kappa_recovery=kappa_recovery,
               kappa_consumption=kappa_consumption,
               kappa_baseline=kappa_baseline)
    # warm-started auxiliary relaxed fields (one pair per shared type) +
    # the E_x of the last forces_of call (reused by the energy record at
    # the same positions)
    full_state = {"k1": [None] * len(shared_w),
                  "k2": [None] * len(shared_w), "ex": 0.0}

    def loads_of(p):
        return [gaussian_load(shape, [tuple(pi)], width, mi)
                for pi, mi in zip(p, m)]

    def forces_of(p, kappa):
        if contact_full:
            lg = [_gaussian_load_and_grad(shape, tuple(pi), width, mi)
                  for pi, mi in zip(p, m)]
            loads = [x[0] for x in lg]
        else:
            loads = loads_of(p)
        grad = [0.5 * (np.roll(kappa, -1, ax) - np.roll(kappa, 1, ax))
                for ax in range(kappa.ndim)]
        f = np.array([[-c * float(np.sum(li * kappa * g)) for g in grad]
                      for li in loads])
        if contact_full:
            ex_total = 0.0
            for ti, (w1t, w2t) in enumerate(shared_w):
                s1, s2 = w1t * loads[0], w2t * loads[1]
                dup = _smoothed_min(s1, s2, contact_full_eps)
                k1 = _solve_relaxed(dup, full_state["k1"][ti], **fkw)
                k2 = _solve_relaxed(2.0 * dup, full_state["k2"][ti],
                                    **fkw)
                full_state["k1"][ti], full_state["k2"][ti] = k1, k2
                ex_total += (2.0 * _relax_functional(k1, dup, **fkw)
                             - _relax_functional(k2, 2.0 * dup, **fkw))
                w = c * (k1 * k1 - k2 * k2)
                for i, (si, sj, wit) in enumerate(((s1, s2, w1t),
                                                   (s2, s1, w2t))):
                    wi = w * _smoothed_min_weight(si, sj,
                                                  contact_full_eps)
                    for ax in range(kappa.ndim):
                        f[i, ax] += wit * float(np.sum(wi * lg[i][1][ax]))
            full_state["ex"] = ex_total
        elif contact_b or contact_derived:
            tot = np.sum(loads, axis=0)
            eprime = (exclusion_energy_derivative(
                          tot, kappa_recovery=kappa_recovery,
                          kappa_consumption=kappa_consumption)
                      if contact_derived else None)
            for i, li in enumerate(loads):
                for ax in range(kappa.ndim):
                    dli = 0.5 * (np.roll(li, -1, ax) - np.roll(li, 1, ax))
                    if contact_derived:
                        f[i, ax] += float(np.sum(eprime * dli))
                    else:
                        f[i, ax] += contact_b * float(np.sum(tot * dli))
        return f

    if kappa0 is not None:
        kappa = np.asarray(kappa0, dtype=float).copy()
    else:
        kappa = relax_capacity(np.sum(loads_of(pos), axis=0), **fkw)
    kdot = np.zeros_like(kappa)
    forces = forces_of(pos, kappa)
    hist = {k: [] for k in ("time", "positions", "velocities", "kinetic",
                            "field_energy", "field_kinetic", "energy",
                            "separation", "dissipation")}
    if probes is not None:
        hist["probe_kappa"] = []
    for i in range(steps):
        if probes is not None:
            hist["probe_kappa"].append([float(kappa[pi, pj])
                                        for pi, pj in probes])
        if i % record_every == 0:
            ke = 0.5 * float(np.sum(m[:, None] * vel ** 2))
            total_load = np.sum(loads_of(pos), axis=0)
            fe = capacity_free_energy(kappa, total_load, **fkw)
            if contact_full:
                ex = float(full_state["ex"])
            elif contact_derived:
                ex = float(np.sum(exclusion_energy_density(
                    total_load, kappa_recovery=kappa_recovery,
                    kappa_consumption=kappa_consumption)))
            else:
                ex = 0.5 * contact_b * float(np.sum(total_load ** 2))
            fk = 0.5 * tau * float(np.sum(kdot ** 2))
            hist["time"].append(i * dt)
            hist["positions"].append(pos.copy())
            hist["velocities"].append(vel.copy())
            hist["kinetic"].append(ke)
            hist["field_energy"].append(float(fe))
            hist["field_kinetic"].append(fk)
            hist["energy"].append(ke + float(fe) + fk + ex)
            hist["separation"].append(
                float(np.linalg.norm(pos[0] - pos[1])) if len(m) == 2
                else float("nan"))
            hist["dissipation"].append(float(np.sum(kdot ** 2)))
        acc = forces / m[:, None]
        vel_half = vel + 0.5 * acc * dt
        pos = pos + vel_half * dt
        kappa, kdot = step_capacity_wave(
            kappa, kdot, np.sum(loads_of(pos), axis=0), tau=tau, dt=dt, **fkw)
        forces = forces_of(pos, kappa)
        acc = forces / m[:, None]
        vel = vel_half + 0.5 * acc * dt
    hist["final_positions"] = pos.copy()
    hist["kappa"] = kappa
    return hist


def front_radius(delta: np.ndarray, center, threshold: float) -> float:
    """Largest periodic distance from ``center`` where ``|δ| > threshold``."""
    shape = delta.shape
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    d2 = sum((((g - c + s / 2) % s) - s / 2) ** 2
             for g, c, s in zip(grids, center, shape))
    mask = np.abs(delta) > threshold
    if not mask.any():
        return 0.0
    return float(np.sqrt(d2[mask].max()))
