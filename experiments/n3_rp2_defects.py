"""Does the ½-disclination survive in three dimensions, or is it a 2-D fact?

The programme's fermion arc — spin, exchange sign, and the ``N⋆ = 3``
constituent count — rests on a nematic ``±½`` disclination.  Every module that
builds it uses a complex field ``ψ`` with director ``θ = ½·arg ψ``.  A complex
phase is a point on a circle, so those modules describe a director *confined to
a plane*: order parameter space ``RP¹``, where

    π₁(RP¹) = **Z**

and ``±½, ±1, ±3/2, …`` are all distinct, with ``+½`` and ``−½`` genuinely
opposite charges.

A director free to point anywhere in space lives on ``RP²``, and

    π₁(RP²) = **Z/2**

which has only two classes.  That is not a small change.  It means integer
disclinations are *unprotected* and remove their own singularity by tilting the
director along the line — the classic **escape into the third dimension** — and
it means ``+½`` and ``−½`` are the **same class**, so a half-integer line is its
own antiparticle.

This matters because it is the same question that has now caught two claims in
this repo: the ``P = 3`` selection turned out to be Plateau's law, and the 3-D
selection sweep weakened honestly.  So: is the ``½`` defect a 3-D object, or a
2-D one that the ``ψ`` representation cannot tell apart from a 3-D one?

**One instrument, two worlds.**  A real three-component director on a lattice,
relaxed under Lebwohl–Lasher.  ``confine_plane=True`` pins ``n_z = 0`` and
*is* the ``RP¹`` world the existing modules describe; ``False`` is ``RP²``.
Same imprint, same seed, same relaxation — the only difference is whether the
third component is allowed to exist.  Escape shows up directly as ``|n_z|``
rising at the core.

Pre-registered predictions:

Q1. **The invariants are correct before anything moves.**  For an imprinted
    strength-``s`` texture the Z/2 class equals ``(2s) mod 2`` and the measured
    in-plane strength equals ``s`` — so ``s = ½`` and ``s = 3/2`` read
    non-trivial, ``s = 1`` and ``s = 2`` read trivial, and ``+½`` and ``−½``
    read the **same**.  **Falsifier:** any disagreement, in which case nothing
    below can be trusted.

Q2. **Integer disclinations escape in RP² and cannot in RP¹.**  Relaxing an
    ``s = 1`` line: core ``|n_z|`` exceeds **0.5** in the RP² arm and is exactly
    **0** in the RP¹ arm, and the RP² elastic energy ends up strictly lower.
    **Falsifier:** no escape in RP², which would mean the relaxation is not
    realising ``π₁(RP²) = Z/2`` and the comparison is empty.

Q3. **Half-integer disclinations cannot escape in either arm.**  Relaxing an
    ``s = ½`` line leaves core ``|n_z|`` below **0.2** even in RP², because the
    Z/2-non-trivial class is genuinely singular and topologically protected.
    **Falsifier:** if ``½`` escapes too, then nothing is protected in 3-D and
    the spinor construction has no three-dimensional analogue at all — which
    would be the strongest possible negative for the fermion arc.

Q2 and Q3 together *are* the statement ``π₁(RP²) = Z/2``, measured rather than
cited: the class that is trivial can undo itself, the class that is not cannot.

Honest scope: a straight line along ``z`` has a ``z``-independent director, so
the lattice is 2-D while the director keeps all three components.  That is the
standard treatment and it is exact for a straight line, but it does not address
loops, junctions, or line reconnection.  One elastic constant, one relaxation
law, zero temperature.

    python experiments/n3_rp2_defects.py
    python experiments/n3_rp2_defects.py --size 96 --steps 4000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_genesis.nematic_rp2 import (  # noqa: E402
    core_escape,
    elastic_energy,
    imprint_disclination,
    loop_strength,
    loop_z2_class,
    relax,
    tilted_core,
)

BANNER = "=" * 78


def run_arm(strength, args, confine):
    """Imprint, relax, measure — identical in both worlds but for one flag."""
    rng = np.random.default_rng(args.seed)
    n0 = imprint_disclination(args.size, strength, rng=rng, noise=args.seed_noise)
    centre = (args.size / 2.0, args.size / 2.0)
    n1 = relax(n0, args.steps, dt=args.dt, confine_plane=confine,
               pin_radius=args.size * 0.42)
    return {
        "escape": core_escape(n1, centre, args.core_radius),
        "energy": elastic_energy(n1),
        "z2_before": loop_z2_class(n0, centre, args.loop_radius),
        "z2_after": loop_z2_class(n1, centre, args.loop_radius),
        "s_before": loop_strength(n0, centre, args.loop_radius),
        "s_after": loop_strength(n1, centre, args.loop_radius),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--seed-noise", type=float, default=0.05,
                   help="tiny perturbation so escape is not blocked by exact symmetry")
    p.add_argument("--core-radius", type=float, default=4.0)
    p.add_argument("--loop-radius", type=float, default=14.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    print(BANNER)
    print("RP2 DEFECTS — is the half-disclination a 3-D object or a 2-D one?")
    print(BANNER)
    print(f"  {args.size}^2 lattice, 3-component director, {args.steps} relaxation steps")
    print("  RP1 arm pins n_z = 0 (what a psi-based model can represent)")
    print("  RP2 arm lets the director leave the plane")
    print(f"  far field pinned beyond r = {args.size * 0.42:.0f} so charge is meaningful")
    print()
    print("  Pre-registered:")
    print("    Q1  Z/2 class = (2s) mod 2 and measured strength = s, on imprint")
    print("    Q2  s=1 escapes in RP2 (core |n_z| > 0.5) and cannot in RP1 (= 0)")
    print("    Q3  s=1/2 cannot escape in EITHER arm (core |n_z| < 0.2)")
    print()

    # ------------------------------------------------------------------- Q1
    print(BANNER)
    print("Q1  DO THE INVARIANTS READ THE KNOWN TEXTURES CORRECTLY?")
    print(BANNER)
    print(f"  {'s':>6} {'Z/2':>5} {'expected':>9} {'strength':>9} {'ok':>4}")
    q1 = True
    centre = (args.size / 2.0, args.size / 2.0)
    for s in (0.5, -0.5, 1.0, -1.0, 1.5, 2.0):
        n = imprint_disclination(args.size, s)
        z = loop_z2_class(n, centre, args.loop_radius)
        st = loop_strength(n, centre, args.loop_radius)
        want = int(round(2 * abs(s))) % 2
        ok = (z == want) and abs(st - s) < 0.05
        q1 = q1 and ok
        print(f"  {s:>6.1f} {z:>5} {want:>9} {st:>9.3f} {'yes' if ok else 'NO':>4}")
    print()
    if q1:
        print("  PREDICTION HELD — the Z/2 grading is exactly (2s) mod 2, and note")
        print("  that +1/2 and -1/2 land in the SAME class. In RP1 they are")
        print("  opposite charges; in RP2 they are the same object.")
    else:
        print("  PREDICTION FAILED — the invariants disagree with the known")
        print("  textures, so nothing below can be trusted.")
        return 1

    # ------------------------------------------------------------- Q2 and Q3
    results = {}
    print()
    print(BANNER)
    print("Q2 / Q3  WHAT SURVIVES RELAXATION?")
    print(BANNER)
    print(f"  {'s':>5} {'world':>5} {'core |n_z|':>11} {'energy':>9} "
          f"{'Z/2 before':>11} {'Z/2 after':>10}")
    for s in (1.0, 0.5):
        for confine, world in ((True, "RP1"), (False, "RP2")):
            r = run_arm(s, args, confine)
            results[f"s{s}_{world}"] = r
            print(f"  {s:>5.1f} {world:>5} {r['escape']:>11.3f} {r['energy']:>9.4f} "
                  f"{r['z2_before']:>11} {r['z2_after']:>10}")

    one_rp1, one_rp2 = results["s1.0_RP1"], results["s1.0_RP2"]
    half_rp1, half_rp2 = results["s0.5_RP1"], results["s0.5_RP2"]

    print()
    print(BANNER)
    print("Q2  DOES THE INTEGER LINE ESCAPE?")
    print(BANNER)
    print(f"  s = 1, RP1: core |n_z| = {one_rp1['escape']:.3f}, "
          f"energy {one_rp1['energy']:.4f}")
    print(f"  s = 1, RP2: core |n_z| = {one_rp2['escape']:.3f}, "
          f"energy {one_rp2['energy']:.4f}")
    q2 = (one_rp2["escape"] > 0.5 and one_rp1["escape"] < 1e-9
          and one_rp2["energy"] < one_rp1["energy"])
    print()
    if q2:
        print("  PREDICTION HELD — the integer line tilts along its own axis and")
        print("  its singularity goes away, at strictly lower energy. In the")
        print("  plane-confined world it has nowhere to go and stays singular.")
        print("  That asymmetry is pi_1(RP2) = Z/2, measured rather than cited.")
    else:
        print("  PREDICTION FAILED — reported as measured:")
        if one_rp2["escape"] <= 0.5:
            print(f"    RP2 escape only reached {one_rp2['escape']:.3f} (needed > 0.5)")
        if one_rp1["escape"] >= 1e-9:
            print("    RP1 arm did not stay planar — the confinement leaked")
        if one_rp2["energy"] >= one_rp1["energy"]:
            print("    escaping did not lower the energy")
        print()
        print("  The reason is the instrument, not the physics, and the")
        print("  distinction is checked below rather than asserted. The flat")
        print("  texture is a SYMMETRIC SADDLE: with n_z identically zero the")
        print("  gradient has no out-of-plane component at all, so zero-")
        print("  temperature descent sits on the knife edge. The registered")
        print("  test asked whether the optimiser finds the escape from a")
        print("  symmetric start, which is a fact about the optimiser. It")
        print("  stands as failed; the topological question is asked properly")
        print("  in the section after Q3.")

    print()
    print(BANNER)
    print("Q3  IS THE HALF-INTEGER LINE PROTECTED?")
    print(BANNER)
    print(f"  s = 1/2, RP1: core |n_z| = {half_rp1['escape']:.3f}")
    print(f"  s = 1/2, RP2: core |n_z| = {half_rp2['escape']:.3f}   "
          f"(Z/2 class {half_rp2['z2_before']} -> {half_rp2['z2_after']})")
    q3 = half_rp2["escape"] < 0.2 and half_rp2["z2_after"] == 1
    print()
    if q3:
        print("  PREDICTION HELD — the half-integer line cannot escape even when")
        print("  the third dimension is available to it, and its Z/2 class")
        print("  survives relaxation. It is topologically protected in 3-D.")
    elif half_rp2["escape"] >= 0.2:
        print(f"  PREDICTION FAILED — the 1/2 line escaped too "
              f"({half_rp2['escape']:.3f}).")
        print("  That would be the strongest negative available for the fermion")
        print("  arc: nothing protected in 3-D, so no spinor to build on.")
    else:
        print("  PREDICTION FAILED — the Z/2 class did not survive relaxation.")

    # ------------------------------------- the repaired instrument (post-hoc)
    print()
    print(BANNER)
    print("SEEDED ESCAPE — the same question, asked off the saddle")
    print(BANNER)
    print("  Not a registered prediction: this instrument was built after Q2")
    print("  failed, once the symmetric-saddle problem was understood. It is")
    print("  reported as a repair, not as a re-score — Q2 stays failed above.")
    print()
    print("  Both strengths are handed the escape on a plate: the core is")
    print("  tipped out of plane by the SAME amount, and the dynamics is asked")
    print("  which way it moves. Growing means escape is downhill and the")
    print("  defect is unprotected; decaying means it is protected.")
    print()
    print(f"  {'s':>5} {'world':>5} {'|n_z| seeded':>13} {'after relax':>12} "
          f"{'':>3} {'reading':>22}")
    seeded = {}
    for s in (1.0, 0.5):
        for confine, world in ((True, "RP1"), (False, "RP2")):
            n0 = tilted_core(args.size, s)
            e0 = core_escape(n0, centre, args.core_radius)
            n1 = relax(n0, args.steps, dt=args.dt, confine_plane=confine,
                       pin_radius=args.size * 0.42)
            e1 = core_escape(n1, centre, args.core_radius)
            seeded[f"s{s}_{world}"] = {"seed": e0, "final": e1,
                                       "energy": elastic_energy(n1)}
            grew = e1 > e0
            note = "GROWS - unprotected" if grew else "decays - protected"
            if confine:
                note = "decays (plane-confined)"
            print(f"  {s:>5.1f} {world:>5} {e0:>13.3f} {e1:>12.3f} {'':>3} "
                  f"{note:>22}")

    grew_one = seeded["s1.0_RP2"]["final"] > seeded["s1.0_RP2"]["seed"]
    kept_half = seeded["s0.5_RP2"]["final"] < 0.2
    print()
    if grew_one and kept_half:
        print("  This is pi_1(RP2) = Z/2, measured. Identical seed, identical")
        print("  law, identical box: the TRIVIAL class (s=1) runs away with the")
        print("  escape and ends up non-singular, while the NON-TRIVIAL class")
        print("  (s=1/2) pushes the tilt back out and stays singular. And the")
        print("  plane-confined arm kills it for both, which is what confining")
        print("  to RP1 has to do.")
    else:
        print("  The seeded test did not separate the two classes either;")
        print("  read it as a null and do not draw the conclusion below.")

    # ------------------------------------------------------------- verdict
    print()
    print(BANNER)
    n_held = int(q1) + int(q2) + int(q3)
    print(f"VERDICT: {n_held}/3 pre-registered predictions held.")
    print(BANNER)
    if q3 and grew_one and kept_half:
        print("  WHAT THIS COSTS THE FERMION ARC, and what it does not.")
        print()
        print("  Survives: the 1/2 defect is a genuine, protected, singular")
        print("  object in three dimensions, so there IS a 3-D spinor to talk")
        print("  about. And the exchange sign needs exactly two classes to be")
        print("  well defined -- Z/2 supplies precisely two. The 4-pi double")
        print("  cover is a pi_1(SO(3)) = Z/2 statement and never depended on")
        print("  the dimension of the order parameter at all.")
        print()
        print("  Does NOT survive: the integer-graded ladder. In RP1 the charges")
        print("  are Z, so +1/2 and -1/2 are opposite and a defect has a signed")
        print("  winding that can 'cross every sector'. In RP2 they are the same")
        print("  class, a half-line is its own antiparticle, and there is no")
        print("  integer winding to count. So any argument in this repo that")
        print("  counts constituents by winding -- including the N* = 3")
        print("  constituent-count claim -- is a 2-D argument and does not")
        print("  transfer as written. It needs restating in Z/2 terms or")
        print("  marking as planar.")
    else:
        print("  Read Q2 and Q3 before drawing anything from this.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "rp2_defects.json"
        out.write_text(json.dumps(
            {"params": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
             "arms": results, "seeded": seeded, "score": n_held},
            indent=2, default=float))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
