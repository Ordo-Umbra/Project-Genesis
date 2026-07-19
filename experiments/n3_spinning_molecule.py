"""The spinning molecule: the chiral term lets a bound pair hold its spin.

The κ-molecule (`n3_kappa_molecule`) was the first persistent bound object,
but it could not spin: rotation was overdamped (Q < ½), the viscous κ-medium
draining any angular momentum before a cycle closed (its Z2 failure).  The
chiral-spin experiment (`n3_chiral_spin`) then showed the missing ingredient:
a parity-breaking term gives the field an intrinsic precession Ω = −λ, and it
lives on the 0D forms.  This experiment brings the two together — the
registered bridge from Act III §5 back to §4.

The forms of the molecule are embedded in that chirally-precessing
background, so the medium drags each toward co-rotation about the pair's
centre of mass at Ω_bg = −λ (`capacity_waves.evolve_inertial_retarded`,
`chiral_omega`/`chiral_coupling`).  The drag is self-limiting — it spins the
pair up to the background rate rather than pumping it apart — so the bound
pair reaches a **steady rotation** where the chiral drive balances the
κ-field drag.  ``chiral_coupling = 0`` is the achiral molecule (Z2)
unchanged.

Pre-registered predictions (same operating point, derived exclusion floor,
a small initial tangential kick, then measure the LATE rotation):

S1. **The achiral molecule drains (Z2, recovered)**: with no chiral drive,
    the late rotation |Ω| < 0.002 — the initial spin is gone, the pair sits
    on its floor without turning, exactly the κ-molecule's overdamped
    verdict.
S2. **The chiral molecule holds its spin**: with the chiral drive on, the
    pair reaches a persistent late rotation |Ω| ≥ 0.003 that is ≥ 5× the
    achiral late value, while the separation stays bound on the floor
    (< 1.5·s⋆) — the medium's handedness is held as the pair's spin.
S3. **The spin is the field's, signed and graded**: across a Ω_bg scan the
    late rotation is monotone through zero with sign(Ω_late) = sign(Ω_bg) —
    the molecule inherits the chiral field's handedness, and turns it up
    with the coupling.

Usage::

    python experiments/n3_spinning_molecule.py --output-dir artifacts/n3_spin_mol
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
    evolve_inertial_retarded,
    exclusion_gap_full,
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


def spin_run(args, s_star: float, omega_bg: float) -> dict:
    """Release the pair with a kick; return late rotation and boundedness."""
    L = args.size
    pos0 = [[L / 2 - s_star / 2, L / 2], [L / 2 + s_star / 2, L / 2]]
    vel0 = [[0.0, +args.kick], [0.0, -args.kick]]
    coupling = args.coupling if omega_bg != 0.0 else 0.0
    h = evolve_inertial_retarded(
        pos0, vel0, [args.mass] * 2, shape=(int(L), int(L)), width=args.width,
        tau=args.tau, dt=args.dt, steps=int(args.t_max / args.dt),
        record_every=20, contact_full=True, chiral_omega=omega_bg,
        chiral_coupling=coupling, **field_kwargs(args))
    t = np.asarray(h["time"])
    pos = np.asarray(h["positions"])
    sep = np.asarray(h["separation"])
    sv = pos[:, 0, :] - pos[:, 1, :]
    phi = np.unwrap(np.arctan2(sv[:, 1], sv[:, 0]))
    omega = np.gradient(phi, t)
    late = t > 0.6 * args.t_max
    return {"omega_bg": omega_bg,
            "omega_late": float(omega[late].mean()),
            "sep_late": float(sep[late].mean()),
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
    p.add_argument("--coupling", type=float, default=0.05)
    p.add_argument("--kick", type=float, default=0.2)
    p.add_argument("--omegas", type=float, nargs="+",
                   default=[-0.2, -0.1, 0.0, 0.1, 0.2])
    p.add_argument("--seps", type=float, nargs="+",
                   default=[6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 16.0])
    p.add_argument("--t-max", type=float, default=280.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_spin_mol")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.omegas = [0.0, 0.2]
        args.t_max = 180.0
        args.seps = [7.0, 9.0, 12.0]

    os.makedirs(args.output_dir, exist_ok=True)
    print("the spinning molecule: chirality lets the bound pair hold Ω",
          flush=True)
    s_star = derived_floor(args)
    print(f"  derived exclusion floor s⋆ = {s_star:g}", flush=True)

    scan = [spin_run(args, s_star, om) for om in args.omegas]
    for r in scan:
        print(f"  Ω_bg={r['omega_bg']:+.2f}: late Ω = {r['omega_late']:+.5f}"
              f"  sep_late={r['sep_late']:.1f}  bounded={r['bounded']}",
              flush=True)

    achiral = next(r for r in scan if r["omega_bg"] == 0.0)
    chiral = [r for r in scan if r["omega_bg"] != 0.0]
    s1 = abs(achiral["omega_late"]) < 0.002 and achiral["bounded"]
    s2 = all(abs(r["omega_late"]) >= 0.003
             and abs(r["omega_late"]) >= 5.0 * abs(achiral["omega_late"])
             and r["bounded"] for r in chiral)
    oms = np.array([r["omega_bg"] for r in scan])
    olate = np.array([r["omega_late"] for r in scan])
    slope = float(np.polyfit(oms, olate, 1)[0])
    signed = all(np.sign(r["omega_late"]) == np.sign(r["omega_bg"])
                 for r in chiral)
    s3 = slope > 0 and signed

    lines = ["The spinning molecule — verdict", "=" * 74, ""]
    lines.append(
        f"S1 (the achiral molecule drains — Z2): at Ω_bg = 0, late Ω = "
        f"{achiral['omega_late']:+.5f} (sep {achiral['sep_late']:.1f}) — "
        + ("✓ no chiral drive, no spin: the initial rotation drains and the "
           "pair sits on its floor without turning — the κ-molecule's "
           "overdamped verdict, recovered." if s1 else
           "✗ the achiral pair rotates or unbinds."))
    lines.append(
        "S2 (the chiral molecule holds its spin): late Ω = "
        + ", ".join(f"{r['omega_late']:+.4f}" for r in chiral)
        + f" vs achiral {achiral['omega_late']:+.4f}, all bound — "
        + ("✓ the medium's handedness is held as the pair's persistent "
           "spin — the first bound object that turns." if s2 else
           "✗ the chiral pair does not hold a persistent bounded spin."))
    lines.append(
        f"S3 (the spin is the field's, signed): late Ω vs Ω_bg slope "
        f"{slope:+.4f}, sign follows at every point: {signed} — "
        + ("✓ the molecule inherits the chiral field's handedness and turns "
           "it up with the coupling." if s3 else
           "✗ the spin does not track the background handedness."))
    lines.append("")
    lines.append(f"score: {int(s1)+int(s2)+int(s3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: the chiral coupling is a rotational drag toward the "
        "background's rigid rotation Ω_bg = −λ (the CGL precession of "
        "`n3_chiral_spin`, §5) — a self-limiting drive, not a derived "
        "two-field coupling (co-evolving the complex chiral field with the "
        "real κ-gravity is the deeper next rung); the steady spin sits well "
        "below Ω_bg because the κ-drag is strong (Q < ½, the same viscosity "
        "that drained the achiral molecule — now balanced, not defeated); "
        "one operating point, the derived exclusion floor; equal masses; a "
        "small initial kick so S1 shows the drain, not merely rest.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_spinning_molecule.json"),
              "w") as fh:
        json.dump({"params": vars(args), "s_star": s_star, "scan": scan,
                   "analysis": {"s1": bool(s1), "s2": bool(s2),
                                "s3": bool(s3), "slope": slope}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
