"""Self-gravitating forms: masses that move in the κ-field they source.

`capacity_gravity` measured the force between *rigid* masses; `stable_forms`
showed a form's gravitational mass equals its structural mass.  This module
lets the masses **move** — the natural next step, structure formation from
first principles.

The dynamics is the adiabatic (Born–Oppenheimer) separation appropriate to a
light mediating field and heavy sources: the capacity field κ relaxes to its
steady state for the instantaneous mass positions, and the masses then drift
down the gradient of the resulting capacity free energy ``F``.  By the
envelope theorem — at the relaxed κ, ``δF/δκ = 0`` — the force on mass ``i`` is
the *direct* coupling gradient, with no implicit-κ term:

    F_i = −∂F/∂R_i = −c · Σ_x load_i(x) · κ(x) · ∇κ(x) ,

i.e. each mass feels the κ-gradient of the well the whole ensemble digs.  In
the overdamped (dissipative) regime the URP program lives in, positions follow

    dR_i/dt = μ · F_i .

Two masses fall together and merge; many masses **accrete** into clusters —
the framework growing its own bound structure out of the same κ-gravity whose
strength equals the forms' mass.
"""

from __future__ import annotations

import numpy as np

from .capacity_gravity import (
    capacity_free_energy,
    gaussian_load,
    relax_capacity,
)

# capacity-field relaxation params forwarded to relax_capacity. NB: the
# relaxation's own internal step is left at relax_capacity's default — it is
# deliberately NOT exposed here, so ``dt`` unambiguously means evolve's
# position-integration timestep (no name collision).
_FIELD_KEYS = ("kappa_diffusion", "kappa_recovery", "kappa_consumption",
               "kappa_baseline", "max_iters", "tol")


def _relax(positions, masses, shape, width, field_kw):
    loads = [gaussian_load(shape, [tuple(p)], width, m)
             for p, m in zip(positions, masses)]
    total = np.sum(loads, axis=0) if loads else np.zeros(shape)
    kappa = relax_capacity(total, **{k: v for k, v in field_kw.items()
                                     if k in _FIELD_KEYS})
    return loads, kappa


def capacity_force(positions, masses, *, shape, width=2.5, **field_kw):
    """Envelope-theorem κ force on each mass; returns ``(forces, kappa)``.

    ``forces`` is ``(n, ndim)``; ``kappa`` the relaxed capacity field.  A mass
    of zero capacity coupling (``kappa_consumption = 0``) sources no well, so
    all forces vanish — the "gravity off" control.
    """
    positions = np.asarray(positions, dtype=float)
    c = field_kw.get("kappa_consumption", 0.2)
    loads, kappa = _relax(positions, masses, shape, width, field_kw)
    grad = [0.5 * (np.roll(kappa, -1, ax) - np.roll(kappa, 1, ax))
            for ax in range(kappa.ndim)]
    forces = np.array([[-c * float(np.sum(li * kappa * g)) for g in grad]
                       for li in loads])
    return forces, kappa


def _merge_close(positions, masses, merge_dist):
    """Fuse mass pairs closer than ``merge_dist`` into one (mass-weighted centre)."""
    positions = [np.asarray(p, float) for p in positions]
    masses = list(masses)
    changed = True
    while changed and len(masses) > 1:
        changed = False
        n = len(masses)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(positions[i] - positions[j]) < merge_dist:
                    m = masses[i] + masses[j]
                    positions[i] = (masses[i] * positions[i]
                                    + masses[j] * positions[j]) / m
                    masses[i] = m
                    del positions[j], masses[j]
                    changed = True
                    break
            if changed:
                break
    return positions, masses


def evolve(positions, masses, *, shape, width=2.5, mobility=2.0, dt=1.0,
           steps=60, merge_dist=3.0, merge=True, **field_kw):
    """Overdamped self-gravitating evolution; returns a trajectory record.

    Each step relaxes κ, moves every mass by ``μ·F·dt``, and (if ``merge``)
    fuses masses that touch — modelling accretion.  Returns a dict with the
    per-step ``positions``, ``masses``, mean pairwise ``separation`` (two-body)
    and ``n_bodies`` count, plus the final ``kappa`` field.
    """
    positions = [np.asarray(p, float) for p in positions]
    masses = list(masses)
    hist = {"positions": [], "masses": [], "separation": [], "n_bodies": []}
    kappa = None
    for _ in range(steps):
        forces, kappa = capacity_force(positions, masses, shape=shape,
                                       width=width, **field_kw)
        hist["positions"].append([p.copy() for p in positions])
        hist["masses"].append(list(masses))
        hist["n_bodies"].append(len(masses))
        hist["separation"].append(_mean_separation(positions))
        positions = [p + mobility * f * dt for p, f in zip(positions, forces)]
        for p in positions:                      # wrap into the periodic box
            p %= np.asarray(shape, float)
        if merge:
            positions, masses = _merge_close(positions, masses, merge_dist)
        if len(masses) == 1:
            break
    hist["kappa"] = kappa
    hist["final_positions"] = [p.copy() for p in positions]
    hist["final_masses"] = list(masses)
    return hist


def _mean_separation(positions):
    if len(positions) < 2:
        return 0.0
    d = [np.linalg.norm(positions[i] - positions[j])
         for i in range(len(positions)) for j in range(i + 1, len(positions))]
    return float(np.mean(d))


def hubble_flow(positions, hubble, center=None):
    """Hubble-law recession velocities ``v_i = H·(r_i − r_centre)``.

    The initial-condition form of a (coasting, Newtonian) expanding background:
    every mass recedes from the centre at a rate proportional to its distance.
    Gravity then competes with this outflow — dense regions decelerate, turn
    around and collapse; the rest is carried apart.
    """
    pos = np.asarray(positions, dtype=float)
    ctr = pos.mean(axis=0) if center is None else np.asarray(center, float)
    return hubble * (pos - ctr)


def fof_groups(positions, link):
    """Friends-of-friends grouping (the cosmologist's halo finder).

    Masses within ``link`` of one another are joined (union-find); returns the
    group sizes, largest first.  A single large group means a bound structure
    formed; all-singletons means the masses dispersed.
    """
    pos = [np.asarray(p, float) for p in positions]
    n = len(pos)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[i] - pos[j]) <= link:
                parent[find(i)] = find(j)
    sizes = {}
    for i in range(n):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1
    return sorted(sizes.values(), reverse=True)


# --------------------------------------------------------------------------
# Inertial dynamics: give the forms momentum, so they orbit and virialise
# --------------------------------------------------------------------------
#
# The overdamped ``evolve`` makes masses *fall* (dissipative, first order).
# Give them inertia — ``M·d²R/dt² = F`` — and they *orbit*: the same envelope
# κ-force now competes with angular momentum.  Because the mediator is
# screened, the potential is Yukawa rather than 1/r, so bound orbits do not
# close — they **precess** (a massive-graviton signature).  A symplectic
# velocity-Verlet integrator conserves the total energy ``T + F[κ]`` (with κ
# relaxed each step); a small velocity damping lets an N-body cloud shed
# energy and **virialise** into a bound cluster.


def _force_and_energy(positions, masses, shape, width, field_kw):
    """Envelope force on each mass, the relaxed κ, and the potential energy F[κ]."""
    c = field_kw.get("kappa_consumption", 0.2)
    loads, kappa = _relax(positions, masses, shape, width, field_kw)
    grad = [0.5 * (np.roll(kappa, -1, ax) - np.roll(kappa, 1, ax))
            for ax in range(kappa.ndim)]
    forces = np.array([[-c * float(np.sum(li * kappa * g)) for g in grad]
                       for li in loads])
    total = np.sum(loads, axis=0) if loads else np.zeros(shape)
    ekw = {k: v for k, v in field_kw.items()
           if k in ("kappa_diffusion", "kappa_recovery", "kappa_consumption",
                    "kappa_baseline")}
    return forces, kappa, capacity_free_energy(kappa, total, **ekw)


def evolve_inertial(positions, velocities, masses, *, shape, width=2.5,
                    dt=0.5, steps=140, damping=0.0, escape_radius=None,
                    **field_kw):
    """Symplectic (velocity-Verlet) inertial evolution under κ-gravity.

    ``M·d²R/dt² = F`` with the envelope κ-force, integrated by velocity
    Verlet so the total energy ``T + F[κ]`` is conserved when ``damping = 0``.
    A nonzero ``damping`` multiplies velocities by ``(1 − damping)`` each step
    (dissipation → virialisation).  Returns per-step positions, velocities,
    kinetic / potential / total energy, mean separation, and the Clausius
    virial ``W = Σ_i (R_i − R_cm)·F_i`` (so ``2⟨T⟩ + ⟨W⟩ → 0`` at equilibrium).
    """
    pos = np.asarray(positions, dtype=float).copy()
    vel = np.asarray(velocities, dtype=float).copy()
    m = np.asarray(masses, dtype=float)
    forces, kappa, pe = _force_and_energy(pos, m, shape, width, field_kw)
    hist = {k: [] for k in ("positions", "velocities", "kinetic", "potential",
                            "energy", "separation", "virial")}
    for _ in range(steps):
        ke = 0.5 * float(np.sum(m[:, None] * vel ** 2))
        com = np.average(pos, axis=0, weights=m)
        virial = float(np.sum([(pos[i] - com) @ forces[i] for i in range(len(m))]))
        hist["positions"].append(pos.copy())
        hist["velocities"].append(vel.copy())
        hist["kinetic"].append(ke)
        hist["potential"].append(float(pe))
        hist["energy"].append(ke + float(pe))
        hist["separation"].append(_mean_separation(list(pos)))
        hist["virial"].append(virial)

        acc = forces / m[:, None]
        vel_half = vel + 0.5 * acc * dt
        pos = pos + vel_half * dt
        forces, kappa, pe = _force_and_energy(pos, m, shape, width, field_kw)
        acc = forces / m[:, None]
        vel = vel_half + 0.5 * acc * dt
        if damping:
            vel *= (1.0 - damping)
        if escape_radius is not None and _mean_separation(list(pos)) > escape_radius:
            break
    hist["kappa"] = kappa
    hist["final_positions"] = pos.copy()
    return hist


# --------------------------------------------------------------------------
# FLRW-like background: an evolving scale factor a(t), Hubble drag, dark energy
# --------------------------------------------------------------------------
#
# The cosmic-structure model carried the expansion in the initial velocities
# only (a coasting background).  Here the background is a genuine, evolving
# **scale factor** ``a(t)`` obeying a Friedmann-like law
#
#     (ȧ/a)² = H₀² [ Ω_m a^{-p} + Ω_Λ ] ,
#
# with a matter component (density diluting as ``a^{-p}``) and a cosmological-
# constant / dark-energy component ``Ω_Λ``.  Working in physical coordinates
# about a fixed comoving origin, each mass obeys
#
#     r̈_i = F_i/M + (ä/a)·(r_i − c) ,   ä/a = H₀²[(1 − p/2) Ω_m a^{-p} + Ω_Λ] ,
#
# the peculiar κ-force plus the background (de)celeration.  Two FLRW effects
# follow for free: a peculiar velocity **redshifts as 1/a** (Hubble drag), and
# a dark-energy-dominated (accelerating) background **freezes structure growth**
# — a faster, accelerating expansion assembles less.


def friedmann_rates(a, hubble0, omega_m, omega_lambda, p=3.0):
    """``(H, ä/a)`` at scale factor ``a`` for a matter+Λ Friedmann background.

    ``H = H₀√(Ω_m a^{-p} + Ω_Λ)`` and ``ä/a = H₀²[(1 − p/2)Ω_m a^{-p} + Ω_Λ]``
    (so matter decelerates for ``p > 2`` and Λ accelerates).
    """
    matter = omega_m * a ** (-p)
    H = hubble0 * np.sqrt(matter + omega_lambda)
    accel = hubble0 ** 2 * ((1.0 - p / 2.0) * matter + omega_lambda)
    return float(H), float(accel)


def evolve_cosmological(positions, masses, *, shape, width=2.5,
                        hubble0=0.05, omega_m=1.0, omega_lambda=0.0, p=3.0,
                        dt=0.5, steps=110, center=None, peculiar=None,
                        **field_kw):
    """Inertial κ-gravity in an FLRW-like background with scale factor ``a(t)``.

    Masses start in the Hubble flow ``v = H₀·(r − c)`` (plus optional
    ``peculiar`` velocities), integrated by velocity-Verlet under the peculiar
    κ-force and the background ``(ä/a)(r − c)``; ``a`` co-evolves via
    ``ȧ = a·H(a)``.  Returns per-step scale factor ``a``, Hubble rate ``H``,
    positions, and the mean physical separation.
    """
    pos = np.asarray(positions, dtype=float).copy()
    m = np.asarray(masses, dtype=float)
    c = pos.mean(axis=0) if center is None else np.asarray(center, float)
    a = 1.0
    H, accel = friedmann_rates(a, hubble0, omega_m, omega_lambda, p)
    vel = hubble0 * (pos - c)
    if peculiar is not None:
        vel = vel + np.asarray(peculiar, float)
    forces, _, _ = _force_and_energy(pos, m, shape, width, field_kw)
    hist = {k: [] for k in ("a", "H", "positions", "velocities", "separation")}
    for _ in range(steps):
        hist["a"].append(a)
        hist["H"].append(H)
        hist["positions"].append(pos.copy())
        hist["velocities"].append(vel.copy())
        hist["separation"].append(_mean_separation(list(pos)))
        acc = forces / m[:, None] + accel * (pos - c)
        vel_half = vel + 0.5 * acc * dt
        pos = pos + vel_half * dt
        a = a + a * H * dt                       # ȧ = a·H
        H, accel = friedmann_rates(a, hubble0, omega_m, omega_lambda, p)
        forces, _, _ = _force_and_energy(pos, m, shape, width, field_kw)
        acc = forces / m[:, None] + accel * (pos - c)
        vel = vel_half + 0.5 * acc * dt
    hist["final_positions"] = pos.copy()
    hist["a_final"] = a
    hist["center"] = c
    return hist


# --------------------------------------------------------------------------
# Closing the loop: the expansion driven by the κ-field's own energy content
# --------------------------------------------------------------------------
#
# In the FLRW model above, ``Ω_Λ`` was dialled by hand.  But the capacity
# field supplies a dark-energy component for free: the recovery term
# ``r·(κ₀ − κ)`` continually heals the field back to baseline, an energy the
# field spends **maintaining itself** that does *not* dilute as the universe
# expands — exactly a cosmological constant.  So the dark-energy density is not
# an input but a property of the capacity field,
#
#     ρ_Λ = coeff · r · κ₀² ,
#
# while matter dilutes as ``ρ_m(a) = ρ_m0 · a^{-dim}``.  The Friedmann equation
# ``H² = ρ_m + ρ_Λ`` (units ``8πG/3 = 1``) and ``ä/a = −½ρ_m + ρ_Λ`` then make
# the whole expansion history — matter-dominated deceleration giving way to
# Λ-dominated acceleration — a *prediction* of the field content: the same κ
# that is gravity also drives the cosmos.


def capacity_vacuum_density(kappa_recovery, kappa_baseline=1.0, coeff=0.5):
    """Emergent dark-energy density ``ρ_Λ = coeff · r · κ₀²``.

    The energy the capacity field spends maintaining itself against dilution
    (the recovery term) — a constant, non-diluting density that acts as a
    cosmological constant.  Set by the field parameters, not dialled.
    """
    return float(coeff * kappa_recovery * kappa_baseline ** 2)


def deceleration_parameter(a, rho_m0, rho_lambda, dim=3):
    """The deceleration parameter ``q = −ä a / ȧ²`` of the emergent background.

    ``q > 0`` decelerating (matter), ``q < 0`` accelerating (Λ); it crosses
    zero at the acceleration onset.
    """
    rho_m = rho_m0 * a ** (-dim)
    accel = -0.5 * rho_m + rho_lambda            # ä/a
    return float(-accel / (rho_m + rho_lambda))


def acceleration_onset(rho_m0, rho_lambda, dim=3):
    """Scale factor ``a_acc`` where the expansion turns from decel to accel.

    ``ä/a = 0`` at ``ρ_m(a) = 2 ρ_Λ`` → ``a_acc = (ρ_m0 / 2ρ_Λ)^{1/dim}`` —
    derived from the matter/vacuum balance, not an input.
    """
    if rho_lambda <= 0:
        return float("inf")
    return float((rho_m0 / (2.0 * rho_lambda)) ** (1.0 / dim))


def integrate_scale_factor(rho_m0, rho_lambda, *, dim=3, dt=0.5, steps=300):
    """Integrate the emergent Friedmann equation; return ``a, t, H, q`` histories.

    ``H = √(ρ_m0 a^{-dim} + ρ_Λ)`` (units ``8πG/3 = 1``), ``ȧ = aH``.
    """
    a, t = 1.0, 0.0
    hist = {"a": [], "t": [], "H": [], "q": []}
    for _ in range(steps):
        rho_m = rho_m0 * a ** (-dim)
        H = np.sqrt(rho_m + rho_lambda)
        hist["a"].append(a)
        hist["t"].append(t)
        hist["H"].append(float(H))
        hist["q"].append(deceleration_parameter(a, rho_m0, rho_lambda, dim))
        a = a + a * H * dt
        t += dt
    return hist


def matter_energy_density(masses, comoving_volume):
    """Present-day matter density ``ρ_m0 = Σ E_i / V`` from a corpus of forms.

    Closes the Friedmann source back onto the stable-form spectrum: each form
    contributes its structural (rest) energy ``E_i`` — which the Bogomolny
    bound fixes to ``E_i ∝ |Q_i|`` — and the sum over the comoving volume is
    the matter density that seeds ``ρ_m(a) = ρ_m0·a^{-dim}``.  So ``ρ_m0`` is
    read off the topological content of the field, not dialled.
    """
    volume = float(comoving_volume)
    if volume <= 0:
        raise ValueError("comoving_volume must be positive")
    return float(np.sum(np.asarray(masses, dtype=float)) / volume)


# --------------------------------------------------------------------------
# The equation of state of the matter: what kind of stuff the forms are
# --------------------------------------------------------------------------
#
# The Friedmann source now comes entirely from the field, but with an *assumed*
# character: matter as pressureless dust (``ρ_m ∝ a^{-dim}``) and the capacity
# vacuum as a cosmological constant (``ρ_Λ = const``).  Those characters are
# equations of state ``p = w·ρ``.  In ``dim`` spatial dimensions the continuity
# equation ``dρ/dt = −dim·H·(ρ + p)`` integrates to ``ρ ∝ a^{-dim(1+w)}``, so the
# dilution exponent *is* the equation of state:
#
#     ρ ∝ a^{-n}  ⇒  w = n/dim − 1 .
#
# Dust (``n = dim``) ⇒ ``w = 0``; a cosmological constant (``n = 0``) ⇒
# ``w = −1``; radiation (``n = dim + 1``) ⇒ ``w = 1/dim``.  And a *gas of forms*
# with peculiar velocities has a directly computable ``w = p/ρ``: at rest it is
# dust (``w → 0``), ultra-relativistic it is radiation (``w → 1/dim``).  These
# are the pieces a relativistic stress-energy tensor ``T_{μν}`` will need.


def equation_of_state_from_dilution(dilution_exponent, dim=3):
    """The equation of state ``w`` implied by a dilution law ``ρ ∝ a^{-n}``.

    From ``ρ ∝ a^{-dim(1+w)}``: ``w = n/dim − 1``.  So the ``a^{-dim}`` matter
    law means ``w = 0`` (dust), a constant vacuum (``n = 0``) means ``w = −1``,
    and radiation (``n = dim + 1``) means ``w = 1/dim``.
    """
    return float(dilution_exponent) / dim - 1.0


def gas_equation_of_state(masses, velocities, *, volume=1.0):
    """Energy density, pressure and ``w = p/ρ`` of a relativistic gas of forms.

    Point forms of rest energy ``m_i`` moving at velocity ``v_i`` (units
    ``c = 1``) have energy density ``ρ = Σ γ_i m_i / V`` and isotropic pressure
    ``p = Σ γ_i m_i |v_i|² / (dim·V)`` with ``γ = 1/√(1 − v²)``.  At rest the
    gas is pressureless dust (``w → 0``); ultra-relativistic it is radiation
    (``w → 1/dim``).  ``velocities`` is ``(N,)`` or ``(N, dim)``; speeds must be
    ``< 1``.
    """
    m = np.asarray(masses, dtype=float)
    v = np.asarray(velocities, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    dim = v.shape[1]
    speed2 = np.sum(v ** 2, axis=1)
    if np.any(speed2 >= 1.0):
        raise ValueError("form speeds must be < 1 (units c = 1)")
    gamma = 1.0 / np.sqrt(1.0 - speed2)
    vol = float(volume)
    if vol <= 0:
        raise ValueError("volume must be positive")
    rho = float(np.sum(gamma * m) / vol)
    pressure = float(np.sum(gamma * m * speed2) / (dim * vol))
    w = pressure / rho if rho > 0 else 0.0
    return {"rho": rho, "pressure": pressure, "w": w}


# --------------------------------------------------------------------------
# The relativistic closure: a stress-energy tensor T^μ_ν, and expansion as
# its consequence rather than an imposed law
# --------------------------------------------------------------------------
#
# The pieces are now all measured: the matter density and its dilution (Links
# 7–8) and the equation of state of each component (Link 9 — forms are dust
# ``w = 0``, the capacity vacuum is ``w = −1``).  This assembles them into a
# perfect-fluid stress-energy tensor in a 3+1 FLRW background (signature
# ``(−,+,+,+)``, comoving/rest frame),
#
#     T^μ_ν = diag(−ρ, p, p, p) ,   p_i = w_i·ρ_i ,   ρ = Σ ρ_i ,  p = Σ p_i .
#
# Its covariant conservation ``∇_μ T^{μν} = 0`` has, in FLRW, the single
# non-trivial (time) component
#
#     ρ̇_i + 3 H (ρ_i + p_i) = 0   ⇒   ρ_i ∝ a^{−3(1+w_i)} ,
#
# so the matter law ``a^{-3}`` and the constant vacuum are *derived* from
# conservation + the measured equations of state, not imposed.  Closing with the
# Einstein/Friedmann relations (units ``8πG/3 = 1``, so ``H² = ρ``)
#
#     H² = ρ ,   ä/a = −½(ρ + 3p) ,   q = −ä a/ȧ² = ½(1 + 3 w_eff) ,
#
# makes the whole expansion history a *consequence* of the field's own
# stress-energy — the Friedmann equation turned from input into output.


def stress_energy_tensor(densities, eos):
    """The perfect-fluid ``T^μ_ν = diag(−ρ, p, p, p)`` of a component mixture.

    ``densities`` are the current component energy densities ``ρ_i`` and ``eos``
    their equations of state ``w_i`` (so ``p_i = w_i ρ_i``).  Returns the 4×4
    mixed tensor together with the totals ``ρ = Σρ_i``, ``p = Σp_i`` and the
    effective equation of state ``w_eff = p/ρ`` (which runs from ~0 when matter
    dominates to −1 when the vacuum does).
    """
    rho_i = np.asarray(densities, dtype=float)
    w_i = np.asarray(eos, dtype=float)
    if rho_i.shape != w_i.shape:
        raise ValueError("densities and eos must have the same shape")
    rho = float(rho_i.sum())
    p = float((w_i * rho_i).sum())
    T = np.diag([-rho, p, p, p]).astype(float)
    w_eff = p / rho if rho != 0 else 0.0
    return {"T": T, "rho": rho, "pressure": p, "w_eff": float(w_eff)}


def covariant_conservation_rate(rho, w, hubble):
    """The FLRW continuity rate ``ρ̇ = −3 H (ρ + p) = −3 H ρ (1 + w)``.

    The single non-trivial component of ``∇_μ T^{μν} = 0`` in a 3+1 FLRW
    background — the law that makes each component dilute as ``a^{−3(1+w)}``.
    """
    return float(-3.0 * hubble * rho * (1.0 + w))


def friedmann_acceleration(rho, pressure):
    """The acceleration ``ä/a = −½(ρ + 3p)`` (the second Friedmann/Einstein eq)."""
    return float(-0.5 * (rho + 3.0 * pressure))


def integrate_stress_energy(densities0, eos, *, dt=0.5, steps=300):
    """Evolve the coupled ``T^μ_ν`` + Friedmann system — nothing imposed.

    Each component density evolves by covariant conservation
    ``ρ̇_i = −3 H (ρ_i + p_i)``; the Hubble rate closes the system through
    ``H = √(ρ_tot)`` (units ``8πG/3 = 1``) and ``ȧ = a H``.  So the dilution
    laws and the whole expansion history ``a(t)`` follow from the measured
    equations of state and conservation, not from an assumed ``a^{-3}``.
    Returns ``a, t, H, q``, the total density/pressure/``w_eff`` and the
    per-component densities.  Integrated with RK4 so the conservation-derived
    dilution matches ``a^{−3(1+w)}`` to high precision.
    """
    rho = np.asarray(densities0, dtype=float).copy()
    w = np.asarray(eos, dtype=float)

    def deriv(state):
        r = state[:-1]
        H = np.sqrt(max(float(r.sum()), 0.0))
        drho = -3.0 * H * r * (1.0 + w)          # covariant conservation
        da = state[-1] * H                       # ȧ = a H
        return np.append(drho, da)

    state = np.append(rho, 1.0)                  # [ρ_i..., a]
    t = 0.0
    hist = {k: [] for k in ("a", "t", "H", "q", "rho_tot", "p_tot", "w_eff")}
    hist["rho_components"] = []
    for _ in range(steps):
        r = state[:-1]
        rho_tot = float(r.sum())
        p_tot = float((w * r).sum())
        H = np.sqrt(max(rho_tot, 0.0))
        accel = friedmann_acceleration(rho_tot, p_tot)
        hist["a"].append(float(state[-1])); hist["t"].append(t)
        hist["H"].append(float(H))
        hist["q"].append(float(-accel / rho_tot) if rho_tot > 0 else float("nan"))
        hist["rho_tot"].append(rho_tot); hist["p_tot"].append(p_tot)
        hist["w_eff"].append(p_tot / rho_tot if rho_tot > 0 else 0.0)
        hist["rho_components"].append(r.copy())
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return hist


# --------------------------------------------------------------------------
# The variational closure: the Friedmann relations from an action, not by hand
# --------------------------------------------------------------------------
#
# Link 10 still *put in by hand* the Einstein/Friedmann relations that turn the
# stress-energy tensor into expansion.  Those relations are the content of a
# minisuperspace variational principle.  For a flat FLRW background with lapse
# ``N`` and scale factor ``a`` (units ``8πG/3 = 1``, comoving volume dropped),
#
#     S = ∫ dt [ −a ȧ²/N − N a³ ρ(a) ] ,
#
# the gravitational kinetic term ``−a ȧ²`` is the dynamical geometry's own
# energy (the capacity field's free energy setting the global scale) and
# ``ρ(a)`` is the matter/vacuum content of the stress-energy tensor.  Two
# variations give the two laws:
#
#   • ``∂S/∂N = 0`` (the lapse / Hamiltonian constraint) ⇒  H² = ρ   — Friedmann.
#   • the ``a`` Euler–Lagrange equation, with conservation, ⇒
#         ä/a = −½(ρ + 3p)   — the acceleration equation.
#
# Crucially the Friedmann equation is not an independent input: along the
# acceleration equation the quantity ``C = H² − ρ`` obeys ``Ċ = −2 H C`` — it is
# a **first integral** (``C = 0`` is preserved) and, since ``H > 0``, an
# **attractor** (a wrong initial expansion rate relaxes onto it).  So ``H² = ρ``
# is a *consequence* of the variational dynamics, not a postulate.


def minisuperspace_lagrangian(a, adot, rho, lapse=1.0):
    """The flat-FLRW minisuperspace Lagrangian ``L = −a ȧ²/N − N a³ ρ``.

    The gravitational kinetic term ``−a ȧ²`` (the geometry's own free energy)
    plus the matter/vacuum content ``a³ ρ``, whose lapse variation gives the
    Friedmann constraint and whose ``a``-variation gives the acceleration
    equation.  Units ``8πG/3 = 1``.
    """
    return float(-a * adot ** 2 / lapse - lapse * a ** 3 * rho)


def hamiltonian_constraint(a, adot, rho):
    """The Friedmann/Hamiltonian constraint value ``C = (ȧ/a)² − ρ = H² − ρ``.

    The lapse variation of the minisuperspace action requires this to vanish
    (equivalently the minisuperspace Hamiltonian ``H_ham = −a³·C`` is zero — the
    "zero total energy" of a reparametrization-invariant system).  Along the
    acceleration equation it obeys ``Ċ = −2 H C``, so ``C = 0`` is preserved and
    attracting: the Friedmann equation as a first integral, not an input.
    """
    H = adot / a
    return float(H * H - rho)


def integrate_friedmann_action(densities0, eos, *, dt=0.1, steps=700,
                               adot0=None):
    """Evolve the minisuperspace ``a``-Euler–Lagrange equation — ``H²=ρ`` not imposed.

    Integrates the second-order acceleration equation ``ä = −½ a (ρ + 3p)`` (the
    ``a``-variation of the action) together with covariant conservation
    ``ρ̇_i = −3 (ȧ/a)(ρ_i + p_i)``, using the *kinematic* Hubble rate ``H = ȧ/a``
    — the Friedmann relation ``H² = ρ`` is **never** substituted.  With the
    constraint-satisfying initial rate ``ȧ₀ = √ρ₀`` (the default) the constraint
    ``C = H² − ρ`` stays ~0; a violating ``adot0`` relaxes onto it.  Returns
    ``a, t, adot, H, q`` and the constraint history ``constraint``.
    """
    rho = np.asarray(densities0, dtype=float).copy()
    w = np.asarray(eos, dtype=float)
    a = 1.0
    adot = float(np.sqrt(rho.sum())) if adot0 is None else float(adot0)

    def deriv(state):
        a = state[0]; adot = state[1]; r = state[2:]
        rho_tot = float(r.sum()); p_tot = float((w * r).sum())
        H = adot / a
        addot = -0.5 * a * (rho_tot + 3.0 * p_tot)     # a-Euler–Lagrange
        drho = -3.0 * H * (r + w * r)                  # conservation, H = ȧ/a
        return np.concatenate([[adot], [addot], drho])

    state = np.concatenate([[a], [adot], rho])
    t = 0.0
    hist = {k: [] for k in ("a", "t", "adot", "H", "q", "constraint",
                            "rho_tot")}
    for _ in range(steps):
        a = state[0]; adot = state[1]; r = state[2:]
        rho_tot = float(r.sum()); p_tot = float((w * r).sum())
        H = adot / a
        addot = -0.5 * a * (rho_tot + 3.0 * p_tot)
        hist["a"].append(float(a)); hist["t"].append(t)
        hist["adot"].append(float(adot)); hist["H"].append(float(H))
        hist["q"].append(float(-addot * a / adot ** 2) if adot != 0
                         else float("nan"))
        hist["constraint"].append(float(H * H - rho_tot))
        hist["rho_tot"].append(rho_tot)
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return hist
