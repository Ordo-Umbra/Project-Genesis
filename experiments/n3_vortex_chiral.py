"""The vorticity-bearing chiral field: spin from real mechanical circulation.

The two-field molecule (`n3_two_field_chiral`) made the spin's *rate* and
*handedness* the co-evolving chiral field's own — but named one caveat it could
not retire: the circulation's **rigid-rotation flow profile**.  A uniformly
precessing bulk carries no mechanical current (``j = Im(ψ*∇ψ) = 0`` there),
which is exactly why that experiment's honest local force (C2) was *radial* and
torque-free; converting internal precession into orbital motion still passed
through one imposed profile.  That caveat named its own cure: *a chiral field
that carries mechanical angular momentum — vorticity.*

This experiment supplies it (`project_genesis/vortex_chiral.py`).  Each form
carries a **vortex** pinned at its core, ``ψ = A(x)·exp(i·Σ q_k arg(x − x_k))``,
an integer winding ``q_k``, amplitude holed at the cores and by the κ wells (the
same detuning as the two-field run).  Now the phase current is a genuine
azimuthal circulation, the field carries real angular momentum, and — the fact
that removes the ansatz — the phase-current force on the bond is **tangential**:
two same-sign vortices co-rotate (a torque), a vortex–antivortex pair
translates (no torque), the point-vortex law.  The molecule spins with **no
imposed rotation**: bound by κ-gravity and the derived exclusion floor, torqued
only by its own vortices' circulation.

Pre-registered predictions:

W1. **The field carries mechanical angular momentum, quantised by the
    winding.**  For a vortex imprinted at a mass, ``L = Σ (x − c) × j`` is
    zero at ``q = 0``, flips sign with ``q`` (``sign L = sign q``), is
    antisymmetric (``|L(+1) + L(−1)| ≤ 0.02·|L(+1)|``), and climbs a
    monotone ladder in ``|q|`` (``|L(±2)| > |L(±1)| > 0``) — an integer
    winding, so the circulation is quantised, not a tunable dial.
W2. **The vortex force is a torque — the ansatz removed.**  For a pair held
    on the floor, two same-sign vortices give a phase-current force that is
    **tangential** to the bond (``|F_t| ≥ 5·|F_r|``), producing a net torque
    ``Στ = Σ (x − x_cm) × F`` whose sign follows ``q``; the
    vortex–antivortex control (``+q, −q``) instead gives a **translation** —
    the two forces near-parallel, the torque collapsing (``|Στ(±)| ≤
    0.2·|Στ(same)|``).  This is the mechanical current the uniformly
    precessing bulk lacked.
W3. **The molecule spins from the field's own circulation — no imposed
    drive.**  Releasing the bound pair with a pinned vortex per mass and no
    rotation drive at all (``circulation_coupling`` is gone), each nonzero
    ``q`` holds a persistent late spin ``|Ω_late| ≥ 0.004`` that is ≥ 5× the
    ``q = 0`` value, with ``sign(Ω_late) = sign(q)`` (same-sign vortices
    co-rotate), the Ω-vs-q slope positive through zero, and every run bound;
    ``q = 0`` is the achiral molecule (Z2) — no vortex, no circulation, it
    drains.

Usage::

    python experiments/n3_vortex_chiral.py --output-dir artifacts/n3_vortex
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import (
    capacity_free_energy,
    gaussian_load,
    relax_capacity,
)
from project_genesis.capacity_waves import (
    duplicated_load,
    exclusion_gap_full,
)
from project_genesis.two_field import chiral_detuning
from project_genesis.vortex_chiral import (
    evolve_vortex_molecule,
    field_angular_momentum,
    imprint_vortices,
    vortex_phase_force,
)


def field_kwargs(args):
    return dict(kappa_diffusion=1.0, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0)


def derived_floor(args) -> float:
    """The exclusion-floor separation s⋆ (derived, parameter-free)."""
    n = int(args.size)
    fkw = field_kwargs(args)

    def energy(s):
        l1 = gaussian_load((n, n), [(n / 2 - s / 2, n / 2)], args.width,
                           args.mass)
        l2 = gaussian_load((n, n), [(n / 2 + s / 2, n / 2)], args.width,
                           args.mass)
        kappa = relax_capacity(l1 + l2, **fkw)
        return (capacity_free_energy(kappa, l1 + l2, **fkw)
                + exclusion_gap_full(duplicated_load(l1, l2), **fkw))

    es = [(s, energy(s)) for s in args.seps]
    return float(min(es, key=lambda t: t[1])[0])


def angular_momentum_ladder(args) -> list:
    """W1: L of a single pinned vortex, across the integer winding ladder."""
    n = int(args.size)
    c = (n / 2, n / 2)
    out = []
    for q in args.charges:
        psi = imprint_vortices((n, n), [c], [q], core=args.core)
        out.append({"q": q, "L": field_angular_momentum(psi, c)})
        print(f"  winding q={q:+d}: L_field={out[-1]['L']:+.1f}", flush=True)
    return out


def static_torque(args, s_star: float, charges) -> dict:
    """W2: the phase-current force / torque on a floor-bound vortex pair."""
    n = int(args.size)
    fkw = field_kwargs(args)
    centers = [(n / 2 - s_star / 2, n / 2), (n / 2 + s_star / 2, n / 2)]
    loads = [gaussian_load((n, n), [ci], args.width, args.mass)
             for ci in centers]
    kappa = relax_capacity(loads[0] + loads[1], **fkw)
    g = chiral_detuning(kappa, detune_gamma=args.gamma, kappa_baseline=1.0)
    psi = imprint_vortices((n, n), centers, list(charges), core=args.core,
                           detune=g)
    forces = vortex_phase_force(loads, psi, phase_chi=args.chi)
    com = np.average(np.array(centers), axis=0)
    torque = 0.0
    for i, ci in enumerate(centers):
        rx, ry = np.array(ci) - com
        torque += float(rx * forces[i][1] - ry * forces[i][0])
    # bond is the x-axis: radial = x, tangential = y
    f_rad = float(np.max(np.abs(forces[:, 0])))
    f_tan = float(np.max(np.abs(forces[:, 1])))
    return {"charges": list(charges), "forces": forces.tolist(),
            "torque": torque, "f_radial": f_rad, "f_tangential": f_tan}


def dynamic_run(args, s_star: float, q: int) -> dict:
    """W3: release the bound pair with a pinned vortex per mass, no drive."""
    L = int(args.size)
    pos0 = [[L / 2 - s_star / 2, L / 2], [L / 2 + s_star / 2, L / 2]]
    vel0 = [[0.0, +args.kick], [0.0, -args.kick]]
    h = evolve_vortex_molecule(
        pos0, vel0, [args.mass] * 2, shape=(L, L), width=args.width,
        tau=args.tau, dt=args.dt, steps=int(args.t_max / args.dt),
        record_every=20, vortex_charge=q, phase_chi=args.chi,
        core=args.core, detune_gamma=args.gamma, **field_kwargs(args))
    t = np.asarray(h["time"])
    sep = np.asarray(h["separation"])
    omega = np.asarray(h["omega"])
    late = t > 0.6 * args.t_max
    return {"q": q,
            "omega_late": float(omega[late].mean()),
            "sep_late": float(sep[late].mean()),
            "L_field": float(np.mean(h["L_field"][-5:])),
            "amp_min": float(np.mean(h["amp_min"][-5:])),
            "bounded": bool(np.all(np.isfinite(sep))
                            and sep[late].max() < 1.5 * s_star)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=float, default=96.0)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--mass", type=float, default=0.6)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--chi", type=float, default=0.6)
    p.add_argument("--core", type=float, default=3.0)
    p.add_argument("--charges", type=int, nargs="+", default=[-2, -1, 0, 1, 2])
    p.add_argument("--dyn-charges", type=int, nargs="+", default=[-1, 0, 1])
    p.add_argument("--seps", type=float, nargs="+",
                   default=[6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 16.0])
    p.add_argument("--kick", type=float, default=0.2)
    p.add_argument("--t-max", type=float, default=220.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_vortex")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.charges = [-1, 0, 1, 2]
        args.dyn_charges = [0, 1]
        args.t_max = 150.0
        args.seps = [7.0, 9.0, 12.0]

    os.makedirs(args.output_dir, exist_ok=True)
    print("the vortex molecule: spin from the chiral field's own circulation",
          flush=True)
    s_star = derived_floor(args)
    print(f"  derived exclusion floor s⋆ = {s_star:g}", flush=True)

    print("angular-momentum ladder (W1):", flush=True)
    ladder = angular_momentum_ladder(args)

    print("static torque (W2):", flush=True)
    same = static_torque(args, s_star, (1, 1))
    opp = static_torque(args, s_star, (1, -1))
    print(f"  same-sign (+1,+1): torque={same['torque']:+.4f}  "
          f"F_t={same['f_tangential']:.4f}  F_r={same['f_radial']:.4f}",
          flush=True)
    print(f"  vortex–antivortex (+1,−1): torque={opp['torque']:+.4f}  "
          f"F_t={opp['f_tangential']:.4f}  F_r={opp['f_radial']:.4f}",
          flush=True)

    print("dynamic scan (W3):", flush=True)
    dyn = [dynamic_run(args, s_star, q) for q in args.dyn_charges]
    for r in dyn:
        print(f"  q={r['q']:+d}: late Ω={r['omega_late']:+.5f}  "
              f"sep={r['sep_late']:.1f}  L_field={r['L_field']:+.1f}  "
              f"|ψ|min={r['amp_min']:.2f}  bounded={r['bounded']}", flush=True)

    # --- W1: quantised angular-momentum ladder
    Lof = {r["q"]: r["L"] for r in ladder}
    w1_zero = abs(Lof.get(0, 0.0)) <= 0.02 * abs(Lof.get(1, 1.0))
    w1_sign = all(np.sign(r["L"]) == np.sign(r["q"])
                  for r in ladder if r["q"] != 0)
    w1_anti = (abs(Lof[1] + Lof[-1]) <= 0.02 * abs(Lof[1])
               if 1 in Lof and -1 in Lof else False)
    w1_ladder = True
    for q in (1, 2):
        if q in Lof and (q - 1) in Lof:
            w1_ladder = w1_ladder and abs(Lof[q]) > abs(Lof[q - 1]) > -1e-9
    w1 = bool(w1_zero and w1_sign and w1_anti and w1_ladder)

    # --- W2: same-sign is a torque, vortex–antivortex a translation
    w2_torque = (same["f_tangential"] >= 5.0 * same["f_radial"]
                 and abs(same["torque"]) > 1e-4)
    w2_control = abs(opp["torque"]) <= 0.2 * abs(same["torque"])
    w2 = bool(w2_torque and w2_control)

    # --- W3: the molecule spins from its own circulation, no drive
    achiral = next(r for r in dyn if r["q"] == 0)
    chiral = [r for r in dyn if r["q"] != 0]
    w3_hold = all(abs(r["omega_late"]) >= 0.004
                  and abs(r["omega_late"]) >= 5.0 * abs(achiral["omega_late"])
                  and r["bounded"] for r in chiral)
    w3_sign = all(np.sign(r["omega_late"]) == np.sign(r["q"]) for r in chiral)
    slope_o = float(np.polyfit([r["q"] for r in dyn],
                               [r["omega_late"] for r in dyn], 1)[0])
    w3 = bool(w3_hold and w3_sign and slope_o > 0 and achiral["bounded"])

    lines = ["The vorticity-bearing chiral field — verdict", "=" * 74, ""]
    lines.append(
        "W1 (the field carries mechanical angular momentum, quantised): L = "
        + ", ".join(f"{r['L']:+.0f}(q={r['q']:+d})" for r in ladder)
        + " — " + ("✓ zero at q=0, sign-locked to the winding, antisymmetric, "
                   "and a monotone ladder in |q| — an integer winding, so the "
                   "circulation is quantised." if w1 else
                   "✗ the angular momentum is not the quantised, sign-locked "
                   "ladder."))
    lines.append(
        "W2 (the vortex force is a torque — the ansatz removed): same-sign "
        f"(+1,+1) torque={same['torque']:+.4f} (F_t={same['f_tangential']:.3f} "
        f"vs F_r={same['f_radial']:.3f}); vortex–antivortex (+1,−1) "
        f"torque={opp['torque']:+.4f} — "
        + ("✓ two like vortices co-rotate — the phase force is tangential, a "
           "real torque — while the antivortex control translates with no "
           "torque: the mechanical current the precessing bulk lacked."
           if w2 else
           "✗ the vortex force is not the clean torque / translation pair."))
    lines.append(
        "W3 (the molecule spins from its own circulation, no drive): late Ω = "
        + ", ".join(f"{r['omega_late']:+.4f}(q={r['q']:+d})" for r in dyn)
        + f", slope vs q {slope_o:+.4f} — "
        + ("✓ each pinned vortex torques its bound pair to a persistent, "
           "signed spin with NO imposed rotation, sign following the winding; "
           "q=0 is the achiral molecule and it drains." if w3 else
           "✗ the field's own circulation does not hold a signed, persistent "
           "spin without a drive."))
    lines.append("")
    lines.append(f"score: {int(w1)+int(w2)+int(w3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: this removes the two-field run's last modelling ansatz "
        "— the rigid-rotation flow profile.  The spin is now driven by a real "
        "mechanical circulation: the field carries genuine angular momentum "
        "(W1), and the phase-current force on the bond is a genuine torque "
        "(W2), so the pair turns from its own vortices with no imposed "
        "rotation (W3).  The remaining idealisation is that the vortex is "
        "PINNED to its form — imprinted at the core each step, its amplitude "
        "co-evolving with (holed by) the κ wells — rather than self-sustained "
        "by the bare CGL dynamics; this is a topological binding of a defect "
        "to matter, which the κ wells physically anchor (a core sits at an "
        "amplitude minimum).  A fully emergent, CGL-sustained vortex is the "
        "deeper version and the open frontier.  The phase force is booked as "
        "a force, not an energy gradient; one operating point, the derived "
        "exclusion floor; equal masses; a small initial kick so q=0 shows the "
        "drain.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_vortex_chiral.json"),
              "w") as fh:
        json.dump({"params": vars(args), "s_star": s_star,
                   "ladder": ladder, "static_same": same, "static_opp": opp,
                   "dynamic": dyn,
                   "analysis": {"w1": bool(w1), "w2": bool(w2),
                                "w3": bool(w3), "slope_omega": slope_o}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
