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
    kappa_dot = kappa_dot + dt * (force - kappa_dot) / tau
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
