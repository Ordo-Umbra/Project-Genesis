"""The vorticity-bearing chiral field: spin from real mechanical circulation.

The two-field coupling (`two_field.py`) derived the molecule's spin *rate* and
*handedness* from a co-evolving chiral field — but its force on the bond was
**radial and torque-free**, because a uniformly precessing bulk carries no
mechanical current (``j = Im(ψ*∇ψ) = 0`` there).  So converting internal
precession into orbital motion still passed through one imposed modelling
ansatz: a rigid-rotation flow profile.  That ansatz's own diagnosis names its
cure — *a field that carries mechanical angular momentum, i.e. vorticity.*

This module supplies it.  The chiral field carries a **vortex** pinned to each
form (spin lives on the 0D forms, Act III §5): ``ψ = A(x)·exp(i·Σ_k q_k·arg(x −
x_k))``, an integer winding ``q_k`` about each mass, with the amplitude ``A``
holed at the cores and by the κ wells (the same detuning as `two_field`).  Now
the phase current is a genuine azimuthal **circulation** around each core, the
field carries real angular momentum ``L = Σ (x − x_cm) × j ≠ 0``, and — the key
fact, measured — the phase-current force on the bond has a **tangential**
component: two same-sign vortices co-rotate (a torque), a vortex–antivortex
pair translates (no torque), exactly the point-vortex law.

So the molecule spins with **no imposed rotation** at all: the masses are bound
by κ-gravity and the derived exclusion floor, each carries a pinned vortex, and
the field's own circulation torques the pair through the phase-current force
(``circulation_coupling`` is gone).  The handedness is the vortex charge's sign;
``q = 0`` is the achiral molecule (Z2) — no vortex, no circulation, no spin.
The winding is an integer, so the field's angular momentum is **quantised**.

The default vortex is *pinned* to its form (imprinted each step, the amplitude
co-evolving) rather than self-sustained by the bare CGL dynamics — a topological
binding of a defect to matter, which the κ wells physically anchor (a vortex
core sits at an amplitude minimum).  What that mode removes is the
rigid-rotation flow profile — the flow is now the real vortex circulation, and
the torque the real phase-current force.

The deeper, *self-sustained* version is here too: seed the vortices once and let
the field co-evolve under the κ-detuned CGL with no re-imprinting
(`evolve_seeded_field`, or `evolve_vortex_molecule(reimprint=False)`).  Then the
integer winding is a **dynamically-conserved** topological charge (not imposed
each step), the κ wells **pin** the core (a control in flat κ drifts), and a
vortex–antivortex pair **annihilates** while like charges survive
(`winding_number` reads the enclosed charge, tracker-free).  Measured honestly,
that self-sustained field keeps its topology and stays matter-pinned, but its
mechanical torque on the free masses is an order weaker than the re-imprinted
drive (the field's circulation drains without the continual re-sharpening) — so
a *strong* persistent orbital spin still needs the pinned mode.
"""

from __future__ import annotations

import numpy as np

from .capacity_gravity import capacity_free_energy, gaussian_load, relax_capacity
from .capacity_waves import (_gaussian_load_and_grad, _relax_functional,
                             _smoothed_min, _smoothed_min_weight,
                             _solve_relaxed, step_capacity_wave)
from .two_field import chiral_detuning, phase_current, step_chiral_detuned


def imprint_vortices(shape, centers, charges, *, core: float = 3.0,
                     detune: np.ndarray | None = None) -> np.ndarray:
    """``ψ = A(x)·exp(i·Σ q_k arg(x − c_k))`` — vortices pinned at ``centers``.

    ``A`` is the product of ``tanh(r_k/core)`` core-holes (times the optional
    ``√(1−detune)`` well envelope); the phase winds ``q_k`` times about each
    centre (periodic distance).  A single ``+1`` vortex carries unit winding
    and a right-handed azimuthal current.
    """
    shape = tuple(int(s) for s in shape)
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    amp = np.ones(shape, dtype=float)
    phase = np.zeros(shape, dtype=float)
    for (cx, cy), q in zip(centers, charges):
        dx = ((grids[0] - cx + shape[0] / 2) % shape[0]) - shape[0] / 2
        dy = ((grids[1] - cy + shape[1] / 2) % shape[1]) - shape[1] / 2
        amp *= np.tanh(np.sqrt(dx * dx + dy * dy) / core)
        phase = phase + q * np.arctan2(dy, dx)
    if detune is not None:
        amp = amp * np.sqrt(np.clip(1.0 - detune, 0.0, 1.0))
    return amp * np.exp(1j * phase)


def field_angular_momentum(psi: np.ndarray, center) -> float:
    """``L = Σ_x (x − c) × j``, ``j = Im(ψ*∇ψ)`` — the field's circulation."""
    jx, jy = phase_current(psi)
    gx = np.arange(psi.shape[0])[:, None] - center[0]
    gy = np.arange(psi.shape[1])[None, :] - center[1]
    return float(np.sum(gx * jy - gy * jx))


def winding_number(psi: np.ndarray, center, radius: int = 6) -> float:
    """Integer phase winding on a square loop of half-side ``radius``.

    The tracker-free topological charge enclosed by the loop: ``+1`` for a unit
    vortex at the centre, ``0`` once the defect drifts out of (or annihilates
    inside) the loop.  Robust where an amplitude-minimum tracker is not, because
    it counts a conserved integer rather than locating a moving core.
    """
    n0, n1 = psi.shape
    cx, cy = int(round(center[0])), int(round(center[1]))
    r = int(radius)
    pts = ([(cx - r, cy + t) for t in range(-r, r)]
           + [(cx + t, cy + r) for t in range(-r, r)]
           + [(cx + r, cy + t) for t in range(r, -r, -1)]
           + [(cx + t, cy - r) for t in range(r, -r, -1)])
    ph = [np.angle(psi[x % n0, y % n1]) for x, y in pts]
    return float(np.sum(np.diff(np.unwrap(ph + [ph[0]]))) / (2.0 * np.pi))


def evolve_seeded_field(shape, centers, charges, kappa, *,
                        chiral: float = 0.0, detune_gamma: float = 0.8,
                        kappa_baseline: float = 1.0, core: float = 3.0,
                        dt: float = 0.1, steps: int = 2000,
                        record_every: int = 100,
                        winding_radius: int = 6) -> dict:
    """Seed vortices ONCE, then co-evolve the field under the κ-detuned CGL.

    No re-imprinting: after the single seed the field evolves only under
    `step_chiral_detuned` (the detuning ``g`` frozen from the passed ``kappa``,
    i.e. static matter).  This is the self-sustained test — the vortices are
    kept (or lost) by the dynamics alone.  Pass ``kappa`` with wells to anchor
    them, or a flat ``kappa`` (``= kappa_baseline``) as the un-pinned control.
    Records, per centre, the enclosed winding; and the field's ``L`` about the
    centroid of ``centers``, plus ``amp_min``.
    """
    shape = tuple(int(s) for s in shape)
    g = chiral_detuning(kappa, detune_gamma=detune_gamma,
                        kappa_baseline=kappa_baseline)
    psi = imprint_vortices(shape, centers, charges, core=core, detune=g)
    com = np.mean(np.asarray(centers, dtype=float), axis=0)
    hist = {"time": [], "winding": [], "L_field": [], "amp_min": []}
    for i in range(steps + 1):
        if i % record_every == 0:
            hist["time"].append(i * dt)
            hist["winding"].append(
                [winding_number(psi, c, winding_radius) for c in centers])
            hist["L_field"].append(field_angular_momentum(psi, com))
            hist["amp_min"].append(float(np.abs(psi).min()))
        psi = step_chiral_detuned(psi, g, chiral=chiral, dt=dt)
    hist["psi"] = psi
    return hist


def vortex_phase_force(loads, psi: np.ndarray, *, phase_chi: float) -> np.ndarray:
    """Footprint-averaged phase-current force ``F_i = χ·Σ ρ̂_i·j`` per mass."""
    j = phase_current(psi)
    out = np.zeros((len(loads), psi.ndim))
    for i, li in enumerate(loads):
        tot = float(np.sum(li))
        if tot == 0.0:
            continue
        for ax in range(psi.ndim):
            out[i, ax] = phase_chi * float(np.sum(li * j[ax])) / tot
    return out


def evolve_vortex_molecule(positions, velocities, masses, *, shape,
                           width: float = 2.5, tau: float = 0.1,
                           dt: float = 0.1, steps: int = 2500,
                           record_every: int = 20,
                           vortex_charge: int = 0, phase_chi: float = 0.6,
                           core: float = 3.0, detune_gamma: float = 0.8,
                           reimprint: bool = True, chiral_lambda: float = 0.0,
                           kappa_diffusion: float = 1.0,
                           kappa_recovery: float = 0.02,
                           kappa_consumption: float = 0.8,
                           kappa_baseline: float = 1.0) -> dict:
    """Bound pair + a vortex per mass; spin from the field's circulation.

    κ-gravity and the ``contact_full`` exclusion floor bind the pair (bitwise
    the retarded molecule when ``vortex_charge = 0`` or ``phase_chi = 0``); the
    pair feels only the phase-current force — **no imposed rotation**.  Records
    positions, separation, ``omega`` (the pair's rotation), ``L_field`` (the
    field's angular momentum), ``winding`` (the enclosed charge at each mass),
    and amplitude bounds.

    Two modes for the vortex-bearing ψ:

    - ``reimprint=True`` (default): ψ is re-imprinted at the masses every step
      (amplitude holed by the κ detuning) — the *pinned* vortex, its circulation
      continually re-sharpened.
    - ``reimprint=False``: ψ is seeded ONCE and then co-evolves under the
      κ-detuned CGL (`step_chiral_detuned`, precession ``chiral_lambda``) with
      no re-imprinting — the *self-sustained* vortex, kept only by the dynamics
      and anchored by the κ wells.
    """
    pos = np.asarray(positions, dtype=float).copy()
    vel = np.asarray(velocities, dtype=float).copy()
    m = np.asarray(masses, dtype=float)
    c = kappa_consumption
    fkw = dict(kappa_diffusion=kappa_diffusion, kappa_recovery=kappa_recovery,
               kappa_consumption=kappa_consumption, kappa_baseline=kappa_baseline)
    st = {"k1": None, "k2": None, "ex": 0.0}

    def loads_of(p):
        return [gaussian_load(shape, [tuple(pi)], width, mi)
                for pi, mi in zip(p, m)]

    def make_psi(p, kappa):
        if not vortex_charge:
            return None
        g = chiral_detuning(kappa, detune_gamma=detune_gamma,
                            kappa_baseline=kappa_baseline)
        return imprint_vortices(shape, [tuple(pi) for pi in p],
                                [vortex_charge] * len(m), core=core, detune=g)

    def forces_of(p, kappa, psi):
        lg = [_gaussian_load_and_grad(shape, tuple(pi), width, mi)
              for pi, mi in zip(p, m)]
        loads = [x[0] for x in lg]
        grad = [0.5 * (np.roll(kappa, -1, ax) - np.roll(kappa, 1, ax))
                for ax in range(kappa.ndim)]
        f = np.array([[-c * float(np.sum(li * kappa * gg)) for gg in grad]
                      for li in loads])
        s1, s2 = loads[0], loads[1]
        dup = _smoothed_min(s1, s2, 1e-4)
        k1 = _solve_relaxed(dup, st["k1"], **fkw)
        k2 = _solve_relaxed(2.0 * dup, st["k2"], **fkw)
        st["k1"], st["k2"] = k1, k2
        st["ex"] = (2.0 * _relax_functional(k1, dup, **fkw)
                    - _relax_functional(k2, 2.0 * dup, **fkw))
        w = c * (k1 * k1 - k2 * k2)
        for i, (si, sj) in ((0, (s1, s2)), (1, (s2, s1))):
            wi = w * _smoothed_min_weight(si, sj, 1e-4)
            for ax in range(kappa.ndim):
                f[i, ax] += float(np.sum(wi * lg[i][1][ax]))
        if phase_chi and psi is not None:
            f = f + vortex_phase_force(loads, psi, phase_chi=phase_chi)
        return f

    def advance_psi(p, kappa, psi):
        # re-imprint the pinned vortex, or step the self-sustained one
        if reimprint or psi is None:
            return make_psi(p, kappa)
        g = chiral_detuning(kappa, detune_gamma=detune_gamma,
                            kappa_baseline=kappa_baseline)
        return step_chiral_detuned(psi, g, chiral=chiral_lambda, dt=dt)

    kappa = relax_capacity(np.sum(loads_of(pos), axis=0), **fkw)
    kdot = np.zeros_like(kappa)
    psi = make_psi(pos, kappa)
    forces = forces_of(pos, kappa, psi)
    hist = {k: [] for k in ("time", "positions", "separation", "omega",
                            "L_field", "amp_min", "winding")}
    for i in range(steps):
        if i % record_every == 0:
            com = np.average(pos, axis=0, weights=m)
            hist["time"].append(i * dt)
            hist["positions"].append(pos.copy())
            hist["separation"].append(float(np.linalg.norm(pos[0] - pos[1])))
            hist["L_field"].append(
                field_angular_momentum(psi, com) if psi is not None else 0.0)
            hist["amp_min"].append(float(np.abs(psi).min())
                                   if psi is not None else 1.0)
            hist["winding"].append(
                [winding_number(psi, pi) for pi in pos] if psi is not None
                else [0.0] * len(m))
        acc = forces / m[:, None]
        vel_half = vel + 0.5 * acc * dt
        pos = pos + vel_half * dt
        kappa, kdot = step_capacity_wave(
            kappa, kdot, np.sum(loads_of(pos), axis=0), tau=tau, dt=dt, **fkw)
        psi = advance_psi(pos, kappa, psi)
        forces = forces_of(pos, kappa, psi)
        acc = forces / m[:, None]
        vel = vel_half + 0.5 * acc * dt
    # per-record rotation rate from the unwrapped bond angle
    t = np.asarray(hist["time"])
    P = np.asarray(hist["positions"])
    sv = P[:, 0, :] - P[:, 1, :]
    phi = np.unwrap(np.arctan2(sv[:, 1], sv[:, 0]))
    hist["omega"] = np.gradient(phi, t).tolist()
    hist["final_positions"] = pos.copy()
    hist["psi"] = psi
    return hist
