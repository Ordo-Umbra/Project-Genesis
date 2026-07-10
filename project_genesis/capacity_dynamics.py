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
