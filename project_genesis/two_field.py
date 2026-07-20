"""The derived two-field coupling: the chiral field ψ co-evolves with κ.

The spinning molecule (`experiments/n3_spinning_molecule.py`) held its spin
under a chiral drive — but the drive was *imposed*: a rigid background
rotation ``v_bg = Ω_bg·ẑ×(x − COM)`` with ``Ω_bg`` a free parameter, merely
*motivated* by the CGL precession ``Ω = −λ`` of `chiral_field`.  This module
closes that gap: the complex chiral field ``ψ`` is a dynamical actor on the
same lattice as the telegrapher κ field, coupled both ways, and the
molecule's spin is slaved to ψ's **own measured precession** — no ``Ω_bg``
parameter anywhere.

The couplings (all local, all derived from the two fields):

- **κ → ψ (detuning).**  The same capacity wells that bind the pair hole
  the chiral field:

      ∂_t ψ = (1 − g)·ψ + (1 + iλ)∇²ψ − (1 + iλ)|ψ|²ψ ,
      g(x) = γ·(1 − κ(x)/κ₀) ,

  so where matter sits (κ depleted) the local amplitude and precession
  rate drop — the 0D forms of the κ sector are amplitude holes of ψ.
- **ψ → matter (the phase-current force).**  A precessing field with
  amplitude holes carries a phase current ``j = Im(ψ*∇ψ)`` around each
  well (the detuning sources a static phase dip; the chiral term shears
  it), and each mass feels the footprint-averaged current as a force
  ``F_i = χ·Σ_x ρ̂_i·j``.  Measured (`n3_two_field_chiral`, C2): the force
  on a floor-bound pair is *radial* — a push or pull along the bond —
  exactly odd in λ, zero at λ = 0, with no tangential component.  The
  chiral field presses on the bond but cannot itself turn it: a uniformly
  precessing bulk carries no mechanical current (``j = 0`` there), so no
  purely local coupling of this form converts internal precession into
  orbital torque.
- **ψ → matter (the self-consistent circulation).**  The drive that spins
  the molecule is retained, but its rate is now the field's own: the bulk
  precession ``Ω_field`` is *measured* from ψ each step (the mean phase
  advance of the unloaded, ordered sites — where ``g ≈ 0`` and
  ``|ψ| ≈ 1``), and the masses relax toward rigid co-rotation at
  ``Ω_field`` about the centre of mass.  ``λ = 0`` gives ``Ω_field = 0``
  automatically — the achiral molecule (Z2) recovered with no special
  casing.  Honest scope: the *rate and handedness* are derived (the CGL's
  ``Ω = −λ``, read off the co-evolving field); the rigid-rotation *flow
  profile* remains a modelling ansatz — eliminating it needs a field that
  carries mechanical angular momentum (vorticity), the next rung.

With every chiral coupling off (``chiral_lambda = detune_gamma =
phase_chi = circulation_coupling = 0``… ψ still evolves but is inert) the
mass/κ trajectory is **bitwise identical** to
`capacity_waves.evolve_inertial_retarded` (regression-tested), so the
achiral baseline is exactly the κ-molecule of record.

Note on bookkeeping: the recorded ``energy`` keeps the same channels as
the retarded instrument, but once the chiral sector is active it is *not*
a Lyapunov function — the CGL reaction continuously injects and removes
energy (it is a driven, non-equilibrium medium), which is precisely what
allows a persistent spin where the achiral molecule drained.
"""

from __future__ import annotations

import numpy as np

from .capacity_gravity import (capacity_free_energy, gaussian_load,
                               relax_capacity)
from .capacity_waves import (_gaussian_load_and_grad, _relax_functional,
                             _smoothed_min, _smoothed_min_weight,
                             _solve_relaxed, step_capacity_wave)
from .multiphase import periodic_laplacian


def chiral_detuning(kappa: np.ndarray, *, detune_gamma: float = 0.8,
                    kappa_baseline: float = 1.0,
                    g_max: float = 0.95) -> np.ndarray:
    """``g(x) = γ·(1 − κ/κ₀)`` clipped to ``[0, g_max]`` — the well depth map.

    Where the capacity field is depleted by matter the chiral field's local
    growth rate (and hence amplitude ``√(1−g)`` and precession magnitude
    ``|λ|(1−g)``) is reduced: the κ wells are holes of the chiral field.
    """
    g = detune_gamma * (1.0 - np.asarray(kappa, dtype=float)
                        / float(kappa_baseline))
    return np.clip(g, 0.0, g_max)


def step_chiral_detuned(psi: np.ndarray, g: np.ndarray, *,
                        chiral: float = 0.0, dt: float = 0.1) -> np.ndarray:
    """One step of the κ-detuned CGL: ``∂_tψ = (1−g)ψ + (1+iλ)(∇²ψ − |ψ|²ψ)``.

    ``g = 0`` everywhere is exactly `chiral_field.step_chiral` (the
    uncoupled field); ``λ = 0`` is real, detuned Ginzburg–Landau (parity
    holds).  Explicit Euler; keep ``dt`` small (the reaction is stiff).
    """
    lap = periodic_laplacian(psi.real) + 1j * periodic_laplacian(psi.imag)
    coeff = 1.0 + 1j * chiral
    dpsi = ((1.0 - g) * psi + coeff * lap
            - coeff * (np.abs(psi) ** 2) * psi)
    return psi + dt * dpsi


def phase_current(psi: np.ndarray) -> list:
    """``j = Im(ψ*·∇ψ)`` — the chiral field's phase current (per axis).

    Smooth everywhere (vanishes at amplitude cores, where the phase is
    undefined), zero in a uniformly precessing bulk, and nonzero around
    the detuning holes of a loaded field when ``λ ≠ 0``: the detuning
    sources a static phase dip at each well and the chiral term shears it
    into a current.  Odd in λ (parity).
    """
    def dx(f):
        return 0.5 * (np.roll(f, -1, 0) - np.roll(f, 1, 0))

    def dy(f):
        return 0.5 * (np.roll(f, -1, 1) - np.roll(f, 1, 1))

    c = np.conj(psi)
    return [np.imag(c * dx(psi)), np.imag(c * dy(psi))]


def bulk_precession(psi: np.ndarray, dpsi: np.ndarray, *,
                    dt: float = 0.1, amp_quantile: float = 0.9) -> float:
    """The field's own spin rate ``Ω_field``, measured — not set.

    Mean phase-advance rate ``Im(ψ*·dψ)/(|ψ|²·dt)`` over the *ordered
    bulk* — the sites whose amplitude sits in the top decile (the wells
    hole the field to ``|ψ| ~ 0.5``, so amplitude is the robust bulk
    marker on any lattice, including small boxes where κ never recovers
    all the way to κ₀): the medium the molecule swims in.  For the CGL
    this is the analytic ``Ω = −λ`` (to the lattice's dispersion
    correction, ~1% at the operating point); ``λ = 0`` gives 0.  Returns
    0.0 if no site qualifies.
    """
    amp = np.abs(psi)
    floor = np.quantile(amp, amp_quantile)
    mask = amp >= floor
    if not mask.any():
        return 0.0
    p = psi[mask]
    return float(np.mean(np.imag(np.conj(p) * dpsi[mask])
                         / (np.abs(p) ** 2 * dt)))


def static_phase_force(loads, kappa: np.ndarray, *, chiral_lambda: float,
                       detune_gamma: float = 0.8, steps: int = 1500,
                       dt: float = 0.1, seed: int = 0,
                       kappa_baseline: float = 1.0):
    """The chiral force on a *fixed* configuration: co-evolve ψ, read j.

    The capacity field is held (the adiabatic statics of the exclusion
    derivations); ψ relaxes from a uniform ordered start under the
    κ-detuning, and each load's force is the footprint average of the
    steady phase current, ``F_i = Σ_x ρ̂_i·j`` (per unit ``phase_chi``).
    Returns ``(forces, psi)`` with ``forces[i][ax]``.
    """
    g = chiral_detuning(kappa, detune_gamma=detune_gamma,
                        kappa_baseline=kappa_baseline)
    rng = np.random.default_rng(seed)
    psi = (np.ones(kappa.shape, dtype=complex)
           + 0.01 * (rng.standard_normal(kappa.shape)
                     + 1j * rng.standard_normal(kappa.shape)))
    for _ in range(int(steps)):
        psi = step_chiral_detuned(psi, g, chiral=chiral_lambda, dt=dt)
    j = phase_current(psi)
    forces = []
    for rho in loads:
        rhon = np.asarray(rho, dtype=float)
        rhon = rhon / rhon.sum()
        forces.append([float(np.sum(rhon * j[ax]))
                       for ax in range(kappa.ndim)])
    return forces, psi


def evolve_two_field(positions, velocities, masses, *, shape,
                     width: float = 2.5, tau: float = 1.0,
                     dt: float = 0.1, steps: int = 1000,
                     record_every: int = 20,
                     contact_full: bool = False,
                     contact_full_eps: float = 1e-4,
                     chiral_lambda: float = 0.0,
                     detune_gamma: float = 0.8,
                     phase_chi: float = 0.0,
                     circulation_coupling: float = 0.0,
                     psi0=None, seed: int = 0,
                     kappa_diffusion: float = 1.0,
                     kappa_recovery: float = 0.05,
                     kappa_consumption: float = 2.0,
                     kappa_baseline: float = 1.0) -> dict:
    """Inertial masses, the telegrapher κ field, and the chiral ψ field.

    The integration loop is `capacity_waves.evolve_inertial_retarded`'s
    (same leapfrog ordering, same ``contact_full`` pair exclusion with its
    two CG-solved auxiliary relaxed fields and smoothed min), extended by
    the co-evolving ψ:

    - after each κ step, ψ advances under the κ-detuning
      (``step_chiral_detuned``) and the bulk precession ``Ω_field`` is
      measured (``bulk_precession``);
    - the phase-current force ``F_i = χ·Σ_x ρ̂_i·j`` is added to the force
      evaluation (``phase_chi``);
    - the velocity kick relaxes each mass toward rigid co-rotation at the
      *measured* ``Ω_field`` about the centre of mass
      (``circulation_coupling``) — the spinning molecule's drive with its
      rate slaved to the field, so ``λ = 0`` is automatically the achiral
      molecule.

    With ``phase_chi = circulation_coupling = 0`` the mass/κ trajectory is
    bitwise identical to ``evolve_inertial_retarded`` (ψ evolves but is
    inert) — regression-tested in ``tests/test_two_field.py``.  Extra
    recorded channels: ``psi`` (final), ``omega_field`` (per record),
    ``psi_amp_min`` / ``psi_amp_max`` (per record).
    """
    pos = np.asarray(positions, dtype=float).copy()
    vel = np.asarray(velocities, dtype=float).copy()
    m = np.asarray(masses, dtype=float)
    c = kappa_consumption
    if contact_full and len(m) < 2:
        raise ValueError("contact_full prices duplicated overlap: pass "
                         "at least two masses (a single mass clones "
                         "nothing).")

    fkw = dict(kappa_diffusion=kappa_diffusion,
               kappa_recovery=kappa_recovery,
               kappa_consumption=kappa_consumption,
               kappa_baseline=kappa_baseline)
    full_state = {"k1": None, "k2": None, "ex": 0.0}

    def loads_of(p):
        return [gaussian_load(shape, [tuple(pi)], width, mi)
                for pi, mi in zip(p, m)]

    def forces_of(p, kappa, j_cur):
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
            s1, s2 = loads[0], loads[1]
            dup = _smoothed_min(s1, s2, contact_full_eps)
            k1 = _solve_relaxed(dup, full_state["k1"], **fkw)
            k2 = _solve_relaxed(2.0 * dup, full_state["k2"], **fkw)
            full_state["k1"], full_state["k2"] = k1, k2
            full_state["ex"] = (2.0 * _relax_functional(k1, dup, **fkw)
                                - _relax_functional(k2, 2.0 * dup, **fkw))
            w = c * (k1 * k1 - k2 * k2)
            for i, (si, sj) in ((0, (s1, s2)), (1, (s2, s1))):
                wi = w * _smoothed_min_weight(si, sj, contact_full_eps)
                for ax in range(kappa.ndim):
                    f[i, ax] += float(np.sum(wi * lg[i][1][ax]))
        if phase_chi and j_cur is not None:
            for i, li in enumerate(loads):
                tot = float(np.sum(li))
                if tot == 0.0:
                    continue
                for ax in range(kappa.ndim):
                    f[i, ax] += phase_chi * float(
                        np.sum(li * j_cur[ax])) / tot
        return f

    kappa = relax_capacity(np.sum(loads_of(pos), axis=0), **fkw)
    kdot = np.zeros_like(kappa)
    if psi0 is not None:
        psi = np.asarray(psi0, dtype=complex).copy()
    else:
        rng = np.random.default_rng(seed)
        psi = (np.ones(kappa.shape, dtype=complex)
               + 0.01 * (rng.standard_normal(kappa.shape)
                         + 1j * rng.standard_normal(kappa.shape)))
    omega_field = 0.0

    def circulation_kick(p, v):
        """Drag toward co-rotation at the field's OWN measured precession.

        The spinning molecule's drive with its rate slaved: ``Ω_field`` is
        measured from ψ each step (``bulk_precession``), so the only
        chiral parameter is λ itself and ``λ = 0`` is automatically the
        undriven (achiral) molecule.  Self-limiting (a drag, not a pump).
        """
        if not circulation_coupling or p.shape[0] < 2:
            return v
        com = np.average(p, axis=0, weights=m)
        rel = p - com
        v_bg = omega_field * np.stack([-rel[:, 1], rel[:, 0]], axis=1)
        return v + dt * circulation_coupling * (v_bg - v)

    j_cur = phase_current(psi) if phase_chi else None
    forces = forces_of(pos, kappa, j_cur)
    hist = {k: [] for k in ("time", "positions", "velocities", "kinetic",
                            "field_energy", "field_kinetic", "energy",
                            "separation", "dissipation", "omega_field",
                            "psi_amp_min", "psi_amp_max")}
    for i in range(steps):
        if i % record_every == 0:
            ke = 0.5 * float(np.sum(m[:, None] * vel ** 2))
            total_load = np.sum(loads_of(pos), axis=0)
            fe = capacity_free_energy(kappa, total_load, **fkw)
            ex = float(full_state["ex"]) if contact_full else 0.0
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
            hist["omega_field"].append(float(omega_field))
            amp = np.abs(psi)
            hist["psi_amp_min"].append(float(amp.min()))
            hist["psi_amp_max"].append(float(amp.max()))
        acc = forces / m[:, None]
        vel_half = vel + 0.5 * acc * dt
        pos = pos + vel_half * dt
        kappa, kdot = step_capacity_wave(
            kappa, kdot, np.sum(loads_of(pos), axis=0), tau=tau, dt=dt,
            **fkw)
        g = chiral_detuning(kappa, detune_gamma=detune_gamma,
                            kappa_baseline=kappa_baseline)
        psi_new = step_chiral_detuned(psi, g, chiral=chiral_lambda, dt=dt)
        omega_field = bulk_precession(psi, psi_new - psi, dt=dt)
        psi = psi_new
        j_cur = phase_current(psi) if phase_chi else None
        forces = forces_of(pos, kappa, j_cur)
        acc = forces / m[:, None]
        vel = vel_half + 0.5 * acc * dt
        vel = circulation_kick(pos, vel)
    hist["final_positions"] = pos.copy()
    hist["kappa"] = kappa
    hist["psi"] = psi
    return hist
