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
    """
    from .capacity_gravity import (capacity_free_energy, gaussian_load,
                                   relax_capacity)

    if contact_b and contact_derived:
        raise ValueError("contact_b and contact_derived are mutually "
                         "exclusive: the derived form has no free b.")

    pos = np.asarray(positions, dtype=float).copy()
    vel = np.asarray(velocities, dtype=float).copy()
    m = np.asarray(masses, dtype=float)
    c = kappa_consumption
    fkw = dict(kappa_diffusion=kappa_diffusion, kappa_recovery=kappa_recovery,
               kappa_consumption=kappa_consumption,
               kappa_baseline=kappa_baseline)

    def loads_of(p):
        return [gaussian_load(shape, [tuple(pi)], width, mi)
                for pi, mi in zip(p, m)]

    def forces_of(p, kappa):
        loads = loads_of(p)
        grad = [0.5 * (np.roll(kappa, -1, ax) - np.roll(kappa, 1, ax))
                for ax in range(kappa.ndim)]
        f = np.array([[-c * float(np.sum(li * kappa * g)) for g in grad]
                      for li in loads])
        if contact_b or contact_derived:
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
            if contact_derived:
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
