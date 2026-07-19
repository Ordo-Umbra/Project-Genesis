"""The derived two-field coupling: the chiral field co-evolves with κ-gravity.

The spinning molecule (`n3_spinning_molecule`) closed Act III's §5→§4 loop
but left a named caveat: its chiral drive was *imposed* — a rigid background
rotation at a free parameter Ω_bg, motivated by the CGL precession Ω = −λ
but not derived from a living field.  This experiment retires that caveat:
the complex chiral field ψ co-evolves on the same lattice as the telegrapher
κ field (`project_genesis/two_field.py`), coupled both ways — the κ wells
detune ψ (matter holes the chiral field), ψ's phase current pushes on the
masses (the field presses on matter), and the molecule's spin is slaved to
ψ's OWN measured bulk precession Ω_field.  There is no Ω_bg parameter; the
only chiral input is λ, and λ = 0 is automatically the achiral molecule.

Pre-registered predictions (same operating point as the spinning molecule:
the derived exclusion floor, a small initial tangential kick, late-time
rotation):

C1. **The two fields co-evolve, healthy**: at every λ of the scan the
    chiral field stays bounded (late max|ψ| ≤ 1.05) yet genuinely holed by
    the κ wells (late min|ψ| ≤ 0.75); the measured bulk precession matches
    the CGL law, |Ω_field + λ| ≤ 0.05·|λ| + 0.01; and the molecule stays
    bound on its floor (late sep < 1.5·s⋆) at every λ — carrying the
    molecule does not break the field, and carrying the field does not
    break the molecule.
C2. **The chiral field presses on matter — radially, odd in λ**: for a
    pair held static on the floor, the phase-current force is a bond-axis
    push (pull) that flips sign with the field's handedness: F_r monotone
    through zero across the λ scan, antisymmetric
    (|F(λ) + F(−λ)| ≤ 10% of max|F|), real (|F_r| at the scan edge ≥ 10×
    the λ = 0 residual), and torque-free (tangential component ≤ 5% of the
    radial).  The field pushes on the bond but cannot itself turn it — a
    uniformly precessing bulk carries no mechanical current.
C3. **The molecule spins, slaved to the field**: with the circulation
    coupling driven by the measured Ω_field (rate AND handedness now the
    field's own), at every nonzero λ the pair holds a persistent late spin
    |Ω_late| ≥ 0.003 that is ≥ 5× the λ = 0 late value, with
    sign(Ω_late) = −sign(λ) = sign(Ω_field), the Ω_late-vs-λ slope
    negative (monotone through zero), and all runs bound.

Usage::

    python experiments/n3_two_field_chiral.py --output-dir artifacts/n3_two_field
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
from project_genesis.two_field import (
    evolve_two_field,
    static_phase_force,
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


def static_forces(args, s_star: float) -> list:
    """C2: the phase-current force on a floor-bound pair, across the scan."""
    n = int(args.size)
    fkw = field_kwargs(args)
    loads = [gaussian_load((n, n), [(n / 2 - s_star / 2, n / 2)],
                           args.width, args.mass),
             gaussian_load((n, n), [(n / 2 + s_star / 2, n / 2)],
                           args.width, args.mass)]
    kappa = relax_capacity(loads[0] + loads[1], **fkw)
    out = []
    for lam in args.lambdas:
        forces, _ = static_phase_force(
            loads, kappa, chiral_lambda=lam,
            detune_gamma=args.gamma, steps=args.static_steps, dt=args.dt,
            seed=args.seed)
        # the bond is the x-axis; radial = x, tangential = y (mass 1)
        out.append({"lambda": lam,
                    "f_radial": forces[0][0], "f_tangential": forces[0][1],
                    "f_partner_radial": forces[1][0]})
        print(f"  static λ={lam:+.2f}: F_r={forces[0][0]:+.3e}  "
              f"F_t={forces[0][1]:+.3e}  F_r(partner)={forces[1][0]:+.3e}",
              flush=True)
    return out


def dynamic_run(args, s_star: float, lam: float) -> dict:
    """C1/C3: release the pair into the co-evolving two-field medium."""
    L = int(args.size)
    pos0 = [[L / 2 - s_star / 2, L / 2], [L / 2 + s_star / 2, L / 2]]
    vel0 = [[0.0, +args.kick], [0.0, -args.kick]]
    h = evolve_two_field(
        pos0, vel0, [args.mass] * 2, shape=(L, L), width=args.width,
        tau=args.tau, dt=args.dt, steps=int(args.t_max / args.dt),
        record_every=20, contact_full=True,
        chiral_lambda=lam, detune_gamma=args.gamma,
        phase_chi=args.chi, circulation_coupling=args.circulation,
        seed=args.seed, **field_kwargs(args))
    t = np.asarray(h["time"])
    pos = np.asarray(h["positions"])
    sep = np.asarray(h["separation"])
    sv = pos[:, 0, :] - pos[:, 1, :]
    phi = np.unwrap(np.arctan2(sv[:, 1], sv[:, 0]))
    omega = np.gradient(phi, t)
    late = t > 0.6 * args.t_max
    return {"lambda": lam,
            "omega_field": float(np.mean(h["omega_field"][-5:])),
            "omega_late": float(omega[late].mean()),
            "sep_late": float(sep[late].mean()),
            "psi_amp_min": float(np.mean(h["psi_amp_min"][-5:])),
            "psi_amp_max": float(np.mean(h["psi_amp_max"][-5:])),
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
    p.add_argument("--chi", type=float, default=0.1)
    p.add_argument("--circulation", type=float, default=0.05)
    p.add_argument("--kick", type=float, default=0.2)
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[-0.2, -0.1, 0.0, 0.1, 0.2])
    p.add_argument("--seps", type=float, nargs="+",
                   default=[6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 16.0])
    p.add_argument("--t-max", type=float, default=220.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--static-steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_two_field")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.lambdas = [0.0, 0.2]
        args.t_max = 160.0
        args.seps = [7.0, 9.0, 12.0]
        args.static_steps = 1200

    os.makedirs(args.output_dir, exist_ok=True)
    print("the two-field molecule: ψ co-evolves with κ, and the spin is "
          "the field's own", flush=True)
    s_star = derived_floor(args)
    print(f"  derived exclusion floor s⋆ = {s_star:g}", flush=True)

    print("static force scan (C2):", flush=True)
    statics = static_forces(args, s_star)

    print("dynamic scan (C1, C3):", flush=True)
    dyn = [dynamic_run(args, s_star, lam) for lam in args.lambdas]
    for r in dyn:
        print(f"  λ={r['lambda']:+.2f}: Ω_field={r['omega_field']:+.4f}  "
              f"late Ω={r['omega_late']:+.5f}  sep={r['sep_late']:.1f}  "
              f"|ψ|∈[{r['psi_amp_min']:.2f},{r['psi_amp_max']:.2f}]  "
              f"bounded={r['bounded']}", flush=True)

    # --- C1: two-field health
    c1_parts = []
    for r in dyn:
        lam = r["lambda"]
        c1_parts.append(
            r["psi_amp_max"] <= 1.05 and r["psi_amp_min"] <= 0.75
            and abs(r["omega_field"] + lam) <= 0.05 * abs(lam) + 0.01
            and r["bounded"])
    c1 = all(c1_parts)

    # --- C2: the static chiral force is radial, odd in λ
    lams = np.array([s["lambda"] for s in statics])
    fr = np.array([s["f_radial"] for s in statics])
    ft = np.array([s["f_tangential"] for s in statics])
    fmax = float(np.max(np.abs(fr)))
    antisym = True
    for lam in set(lams[lams != 0.0]):
        if (lams == -lam).any():  # only where the scan carries both signs
            antisym = antisym and (abs(fr[lams == lam][0]
                                       + fr[lams == -lam][0])
                                   <= 0.1 * fmax)
    edge = abs(fr[np.argmax(np.abs(lams))])
    zero = abs(fr[lams == 0.0][0]) if (lams == 0.0).any() else 0.0
    slope_f = float(np.polyfit(lams, fr, 1)[0])
    nz = lams != 0.0  # at λ = 0 both components are numerical noise;
    # the torque-free bar is only meaningful where the force is real
    c2 = bool(slope_f > 0 and antisym and edge >= 10.0 * max(zero, 1e-12)
              and np.all(np.abs(ft[nz])
                         <= 0.05 * np.abs(fr[nz])))

    # --- C3: the molecule spins, slaved to the field
    achiral = next(r for r in dyn if r["lambda"] == 0.0)
    chiral = [r for r in dyn if r["lambda"] != 0.0]
    c3_hold = all(abs(r["omega_late"]) >= 0.003
                  and abs(r["omega_late"]) >= 5.0 * abs(achiral["omega_late"])
                  and r["bounded"] for r in chiral)
    c3_sign = all(np.sign(r["omega_late"]) == -np.sign(r["lambda"])
                  for r in chiral)
    slope_o = float(np.polyfit([r["lambda"] for r in dyn],
                               [r["omega_late"] for r in dyn], 1)[0])
    c3 = bool(c3_hold and c3_sign and slope_o < 0)

    lines = ["The derived two-field coupling — verdict", "=" * 74, ""]
    lines.append(
        "C1 (the two fields co-evolve, healthy): "
        + "; ".join(f"λ={r['lambda']:+.2f}: Ω_field={r['omega_field']:+.3f} "
                    f"(vs −λ={-r['lambda']:+.3f}), |ψ|∈"
                    f"[{r['psi_amp_min']:.2f},{r['psi_amp_max']:.2f}], "
                    f"bound={r['bounded']}" for r in dyn)
        + " — " + ("✓ the chiral field carries its spin (Ω ≈ −λ) while the "
                   "κ wells hole it, and the molecule rides the medium "
                   "without breaking either field." if c1 else
                   "✗ the co-evolution is unhealthy somewhere in the scan."))
    lines.append(
        "C2 (the field presses on matter, radially, odd in λ): F_r = "
        + ", ".join(f"{s['f_radial']:+.2e}" for s in statics)
        + f" vs λ (slope {slope_f:+.2e}), tangential ≤ "
        + f"{float(np.max(np.abs(ft))):.1e} — "
        + ("✓ the phase current pushes the bond apart (or pulls it shut) "
           "with the field's handedness, and carries no torque — the "
           "chiral field's own force on matter, parity-exact." if c2 else
           "✗ the static chiral force is not the clean radial, λ-odd push."))
    lines.append(
        "C3 (the molecule spins, slaved to the field): late Ω = "
        + ", ".join(f"{r['omega_late']:+.4f}" for r in chiral)
        + f" vs achiral {achiral['omega_late']:+.4f}, slope vs λ "
        + f"{slope_o:+.4f}, sign(Ω) = −sign(λ) everywhere: {c3_sign} — "
        + ("✓ no Ω_bg anywhere: the pair turns at the rate and handedness "
           "it reads off the co-evolving field, and λ = 0 is the achiral "
           "molecule with no special casing." if c3 else
           "✗ the self-consistent drive does not hold a signed, persistent "
           "spin."))
    lines.append("")
    lines.append(f"score: {int(c1)+int(c2)+int(c3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: the spin's RATE and HANDEDNESS are now the field's "
        "own (the measured bulk precession of the co-evolving ψ), and the "
        "field exerts its own measured force on the matter (C2) — but the "
        "circulation's rigid-rotation FLOW PROFILE remains a modelling "
        "ansatz: a uniformly precessing bulk carries no mechanical current "
        "(j = 0 there, which is exactly why C2's honest local force is "
        "radial), so converting internal precession into orbital motion "
        "still passes through the one imposed profile.  Eliminating it "
        "needs a chiral field that carries mechanical angular momentum "
        "(vorticity) — the next rung.  The phase force is booked as a "
        "force, not an energy gradient, and the CGL reaction drives the "
        "medium, so the recorded energy is no longer a Lyapunov function "
        "once the chiral sector is active (that driven-ness is what a "
        "persistent spin costs).  One operating point, the derived "
        "exclusion floor; equal masses; a small initial kick so the λ = 0 "
        "run shows the drain, not merely rest.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_two_field_chiral.json"),
              "w") as fh:
        json.dump({"params": vars(args), "s_star": s_star,
                   "static": statics, "dynamic": dyn,
                   "analysis": {"c1": bool(c1), "c2": bool(c2),
                                "c3": bool(c3), "slope_force": slope_f,
                                "slope_omega": slope_o}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
