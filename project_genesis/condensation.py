"""The condensation boundary: can matter gather its fermions from noise?

The hadron composites (`n3_hadron_spin`) were assembled — wells placed, defects
imprinted.  The dynamical version — matter *collecting* its spin-½ defects out
of a noise-grown field — turns out to have a sharp, measurable boundary, and
this module supplies the instruments that map it:

- `grow_defect_gas` — coarsen ψ from complex noise over a (possibly detuned)
  landscape: an emergent defect gas, never imprinted.
- `amp_reaction_force` — the detuning's *reaction* force on a mass.  The
  coupling energy (the well suppresses ``|ψ|²`` — measured in `two_field`)
  implies, by Newton's third law, a pull on the mass toward amplitude holes:
  ``F = −χ_a·⟨∇|ψ|²⟩`` over the footprint.  **Raw**, this force cannot tell a
  fermion from a fellow mass — every well is an amplitude hole — so it clumps
  matter through the exclusion floor.  **Envelope-normalised** (``normalize=
  True``: react to ``|ψ|²/(1−g)``, the amplitude with the wells' own envelope
  divided out), it responds only to the *topological* hole — the defect core —
  and becomes selective.
- `condensation_run` — the full co-evolution: masses under κ-gravity + the
  derived exclusion + optional phase-current and reaction forces, ψ grown from
  noise and co-evolved with the moving masses' detuning (no re-imprinting).
  Records mass→nearest-defect distances, separations, and the defect census.

The measured boundary (the `n3_condensation_boundary` experiment): passive
traps do not condense (the dilute defect gas is frozen — nothing transports a
defect to a well); the raw reaction force clumps rather than condenses; the
normalised reaction force *retains* a fermion at a mass — capture-and-hold —
while gathering from afar still fails for want of defect transport (the open
rung; the ``λ ≠ 0`` precession that regenerated the driven spin also mobilises
vortex cores, and is the named candidate).
"""

from __future__ import annotations

import numpy as np

from .capacity_gravity import gaussian_load, relax_capacity
from .capacity_waves import (_gaussian_load_and_grad, _smoothed_min,
                             _smoothed_min_weight, _solve_relaxed,
                             step_capacity_wave)
from .nematic_spinor import plaquette_winding
from .two_field import chiral_detuning, step_chiral_detuned
from .vortex_chiral import vortex_phase_force


def grow_defect_gas(shape, g, *, steps: int = 800, dt: float = 0.1,
                    chiral: float = 0.0, seed: int = 0,
                    init_amp: float = 0.1) -> np.ndarray:
    """Coarsen ψ from complex noise under the detuning ``g`` — an emergent gas."""
    rng = np.random.default_rng(seed)
    psi = init_amp * (rng.standard_normal(shape)
                      + 1j * rng.standard_normal(shape))
    for _ in range(steps):
        psi = step_chiral_detuned(psi, g, chiral=chiral, dt=dt)
    return psi


def defect_positions(psi: np.ndarray) -> list:
    """``[(position, charge), …]`` from the plaquette winding (core centres)."""
    w = plaquette_winding(psi)
    return [((x + 0.5, y + 0.5), int(w[x, y])) for x, y in zip(*np.nonzero(w))]


def amp_reaction_force(loads, psi: np.ndarray, *, chi_amp: float,
                       detune: np.ndarray | None = None,
                       normalize: bool = False) -> np.ndarray:
    """The detuning's reaction force: ``F_i = −χ_a·⟨∇A⟩`` over the footprint.

    ``A = |ψ|²`` raw (the naive third-law force — clumps: every well is an
    amplitude hole), or ``A = |ψ|²/(1−g)`` with ``normalize=True`` (the wells'
    own envelope divided out — reacts only to the topological core hole).
    """
    a2 = np.abs(psi) ** 2
    if normalize and detune is not None:
        a2 = a2 / np.clip(1.0 - detune, 0.05, 1.0)
    ga = [0.5 * (np.roll(a2, -1, ax) - np.roll(a2, 1, ax))
          for ax in range(a2.ndim)]
    out = np.zeros((len(loads), a2.ndim))
    for i, li in enumerate(loads):
        tot = float(np.sum(li))
        if tot == 0.0:
            continue
        for ax in range(a2.ndim):
            out[i, ax] = -chi_amp * float(np.sum(li * ga[ax])) / tot
    return out


def condensation_run(positions, velocities, masses, *, shape,
                     width: float = 2.5, tau: float = 0.1, dt: float = 0.1,
                     steps: int = 2200, record_every: int = 20,
                     pregrow_steps: int = 800, seed: int = 0,
                     phase_chi: float = 0.6, chi_amp: float = 0.0,
                     normalize: bool = False, chiral_lambda: float = 0.0,
                     detune_gamma: float = 0.8,
                     kappa_diffusion: float = 1.0, kappa_recovery: float = 0.02,
                     kappa_consumption: float = 0.8,
                     kappa_baseline: float = 1.0) -> dict:
    """Masses + a noise-grown defect gas, fully co-evolved (no re-imprinting).

    κ-gravity + the derived exclusion bind/repel the masses as ever; ψ is grown
    from noise under the masses' initial detuning, then co-evolves with their
    motion; the masses optionally feel the phase-current force and the
    (raw or normalised) amplitude-reaction force.  Records per-record times,
    positions, separations, mass→nearest-defect distances, and defect counts.
    """
    shape = tuple(int(s) for s in shape)
    pos = np.asarray(positions, dtype=float).copy()
    vel = np.asarray(velocities, dtype=float).copy()
    m = np.asarray(masses, dtype=float)
    c = kappa_consumption
    fkw = dict(kappa_diffusion=kappa_diffusion, kappa_recovery=kappa_recovery,
               kappa_consumption=kappa_consumption,
               kappa_baseline=kappa_baseline)
    st = {"k1": None, "k2": None}

    def loads_of(p):
        return [gaussian_load(shape, [tuple(pi)], width, mi)
                for pi, mi in zip(p, m)]

    def pdist(a, b):
        d = [abs(a[k] - b[k]) for k in range(2)]
        return float(np.hypot(min(d[0], shape[0] - d[0]),
                              min(d[1], shape[1] - d[1])))

    def nearest(p, dfs):
        return min((pdist(tuple(p), q) for q, _ in dfs), default=99.0)

    def forces_of(p, kappa, g, psi):
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
        w = c * (k1 * k1 - k2 * k2)
        for i, (si, sj) in ((0, (s1, s2)), (1, (s2, s1))):
            wi = w * _smoothed_min_weight(si, sj, 1e-4)
            for ax in range(kappa.ndim):
                f[i, ax] += float(np.sum(wi * lg[i][1][ax]))
        if phase_chi and psi is not None:
            f = f + vortex_phase_force(loads, psi, phase_chi=phase_chi)
        if chi_amp and psi is not None:
            f = f + amp_reaction_force(loads, psi, chi_amp=chi_amp,
                                       detune=g, normalize=normalize)
        return f

    kappa = relax_capacity(np.sum(loads_of(pos), axis=0), **fkw)
    kdot = np.zeros_like(kappa)
    g = chiral_detuning(kappa, detune_gamma=detune_gamma,
                        kappa_baseline=kappa_baseline)
    psi = grow_defect_gas(shape, g, steps=pregrow_steps, dt=dt,
                          chiral=chiral_lambda, seed=seed)
    forces = forces_of(pos, kappa, g, psi)
    hist = {k: [] for k in ("time", "positions", "separation",
                            "mass_defect_dist", "n_defects")}
    for i in range(steps):
        if i % record_every == 0:
            dfs = defect_positions(psi)
            hist["time"].append(i * dt)
            hist["positions"].append(pos.copy())
            hist["separation"].append(float(np.linalg.norm(pos[0] - pos[1])))
            hist["mass_defect_dist"].append([nearest(pi, dfs) for pi in pos])
            hist["n_defects"].append(len(dfs))
        acc = forces / m[:, None]
        vel_half = vel + 0.5 * acc * dt
        pos = pos + vel_half * dt
        kappa, kdot = step_capacity_wave(
            kappa, kdot, np.sum(loads_of(pos), axis=0), tau=tau, dt=dt, **fkw)
        g = chiral_detuning(kappa, detune_gamma=detune_gamma,
                            kappa_baseline=kappa_baseline)
        psi = step_chiral_detuned(psi, g, chiral=chiral_lambda, dt=dt)
        forces = forces_of(pos, kappa, g, psi)
        acc = forces / m[:, None]
        vel = vel_half + 0.5 * acc * dt
    hist["final_positions"] = pos.copy()
    hist["psi"] = psi
    return hist
