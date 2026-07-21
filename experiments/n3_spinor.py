"""Half-integer spin: the nematic ½-disclination and its 4π double cover.

Every spin the program has measured was **integer** — a ``U(1)`` complex order
parameter, a boson.  A quark is a **half-integer spinor**: a ``2π`` rotation
gives ``−1``, only ``4π`` returns it — the ``SU(2)`` double cover of ``SO(3)``.
That is a *different* order parameter: a **nematic**, a headless director
``n̂ ≡ −n̂`` (space ``RP¹``), whose defects are **±½ disclinations** — the
director winds by only ``π``, which no vector can do.  Carried by
``ψ = e^{2iθ}`` (the doubled angle), the director is ``θ = ½·arg(ψ)`` and a
``ψ``-vortex of charge ``q`` is a disclination of strength ``s = q/2``
(`project_genesis/nematic_spinor.py`) — so the ``U(1)`` vortex machinery applies,
but the observable director is double-valued in the lift, and that is the spinor.

Pre-registered predictions:

S1. **The elementary defect is a half-integer disclination.**  For the nematic
    director ``θ = ½·arg(ψ)``, a ``ψ``-charge-``q`` defect has strength
    ``s = q/2`` (measured across the charge ladder): the elementary ``q = ±1``
    defect is a ``±½`` disclination — the director winds by ``±π``, **half-
    integer**, impossible for a vector (integer-winding) field.  Odd ``q`` gives
    a half-integer, even ``q`` an integer; ``s(q) = q/2`` exactly.
S2. **The double cover — the spinor's ``−1``.**  Transporting the *oriented*
    director once (``2π``) around a ``+½`` disclination **flips** it (holonomy
    ``n̂·n̂ = −1``); only a second loop (``4π``) restores it (``+1``).  An
    *integer* disclination (``s = 1``, from ``q = 2``) shows **no** flip
    (``+1`` already at ``2π``).  This is the ``SU(2)`` double cover of ``SO(3)``
    — the defining property of half-integer spin — measured directly.
S3. **Conserved, fused, and bound to matter.**  Pinned to a κ well and
    co-evolved under the CGL with **no re-imprinting**, a ``½`` disclination
    keeps its strength *and* its ``−1`` holonomy (a conserved topological charge
    on matter).  Two ``½`` disclinations **fuse** to a net *integer* (``s = 1``
    around a loop enclosing both); a ``½`` and a ``−½`` **annihilate**
    (``s → 0``, the field heals).  The half-integer is additive and protected.

Honest scope: this is the **order-parameter (topological)** realisation of
half-integer spin — a genuine ``Z₂/RP¹`` double cover with ``π``-winding
disclinations, the topological essence of a spinor.  It is **not** the full
quantum Dirac field: no spin–statistics (anticommutation), no Dirac equation, no
dynamical fermion — those are the deeper frontier.  What is measured is the
half-integer winding, the ``4π`` double cover, and the fusion rules.  2-D; the
elementary disclination is ``±½``; one lattice.

Usage::

    python experiments/n3_spinor.py --output-dir artifacts/n3_spinor
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import gaussian_load, relax_capacity
from project_genesis.nematic_spinor import (
    director_holonomy,
    disclination_strength,
)
from project_genesis.two_field import chiral_detuning, step_chiral_detuned
from project_genesis.vortex_chiral import imprint_vortices


def field_kwargs(args):
    return dict(kappa_diffusion=1.0, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0)


def s1_ladder(args):
    n = int(args.size)
    c = (n / 2, n / 2)
    out = {}
    for q in args.charges:
        psi = imprint_vortices((n, n), [c], [q], core=args.core)
        out[q] = disclination_strength(psi, c, radius=args.radius)
    return out


def s2_double_cover(args):
    n = int(args.size)
    c = (n / 2, n / 2)
    half = director_holonomy(imprint_vortices((n, n), [c], [1], core=args.core),
                             c, radius=args.radius)
    integer = director_holonomy(imprint_vortices((n, n), [c], [2], core=args.core),
                                c, radius=args.radius)
    return {"half_2pi": half[0], "half_4pi": half[1],
            "int_2pi": integer[0], "int_4pi": integer[1]}


def s3_matter(args):
    n = int(args.size)
    fkw = field_kwargs(args)
    # (a) a 1/2 disclination pinned to a well, co-evolved (no re-imprint)
    c = (n / 2, n / 2)
    kappa = relax_capacity(gaussian_load((n, n), [c], args.width, args.mass), **fkw)
    g = chiral_detuning(kappa, detune_gamma=args.gamma)
    psi = imprint_vortices((n, n), [c], [1], core=args.core, detune=g)
    for _ in range(args.steps):
        psi = step_chiral_detuned(psi, g, chiral=0.0, dt=args.dt)
    s_pin = disclination_strength(psi, c, radius=args.radius)
    h_pin = director_holonomy(psi, c, radius=args.radius)
    # (b) fusion + (c) annihilation, two wells
    s = args.sep
    c0, c1 = (n / 2 - s / 2, n / 2), (n / 2 + s / 2, n / 2)
    k2 = relax_capacity(gaussian_load((n, n), [c0], args.width, args.mass)
                        + gaussian_load((n, n), [c1], args.width, args.mass), **fkw)
    g2 = chiral_detuning(k2, detune_gamma=args.gamma)
    psi_fuse = imprint_vortices((n, n), [c0, c1], [1, 1], core=args.core, detune=g2)
    s_fuse = disclination_strength(psi_fuse, (n / 2, n / 2),
                                   radius=args.radius + args.sep)
    psi_ann = imprint_vortices((n, n), [c0, c1], [1, -1], core=args.core, detune=g2)
    amp0 = float(np.abs(psi_ann).min())
    for _ in range(args.steps):
        psi_ann = step_chiral_detuned(psi_ann, g2, chiral=0.0, dt=args.dt)
    return {"s_pinned": s_pin, "holo_pinned": list(h_pin),
            "s_fused": s_fuse,
            "s_ann0": disclination_strength(psi_ann, c0, radius=6),
            "s_ann1": disclination_strength(psi_ann, c1, radius=6),
            "amp_seed": amp0, "amp_ann": float(np.abs(psi_ann).min())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--mass", type=float, default=0.6)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--core", type=float, default=3.0)
    p.add_argument("--radius", type=float, default=12.0)
    p.add_argument("--charges", type=int, nargs="+", default=[-2, -1, 0, 1, 2, 3])
    p.add_argument("--sep", type=float, default=16.0)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_spinor")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.charges = [-1, 0, 1, 2]
        args.steps = 400

    os.makedirs(args.output_dir, exist_ok=True)
    print("half-integer spin: the nematic ½-disclination and its 4π double cover",
          flush=True)

    ladder = s1_ladder(args)
    print("S1 — disclination strength s vs ψ-charge q:", flush=True)
    print("  " + ", ".join(f"s={v:+.2f}(q={q:+d})" for q, v in ladder.items()),
          flush=True)
    dc = s2_double_cover(args)
    print("S2 — double cover (oriented-director holonomy):", flush=True)
    print(f"  ½ (q=1): n·n after 2π = {dc['half_2pi']:+.2f}, after 4π = "
          f"{dc['half_4pi']:+.2f}", flush=True)
    print(f"  integer (q=2): n·n after 2π = {dc['int_2pi']:+.2f}", flush=True)
    m = s3_matter(args)
    print("S3 — conserved, fused, bound:", flush=True)
    print(f"  pinned & evolved: s={m['s_pinned']:+.2f}, holonomy="
          f"({m['holo_pinned'][0]:+.2f},{m['holo_pinned'][1]:+.2f})", flush=True)
    print(f"  fusion (½+½): s(big loop)={m['s_fused']:+.2f}", flush=True)
    print(f"  annihilation (½−½): s=({m['s_ann0']:+.2f},{m['s_ann1']:+.2f}), "
          f"|ψ|min {m['amp_seed']:.2f}→{m['amp_ann']:.2f}", flush=True)

    # --- S1: s = q/2, elementary defect half-integer
    s1_exact = all(abs(v - q / 2.0) <= 0.05 for q, v in ladder.items())
    s1_half = (1 in ladder and abs(abs(ladder[1]) - 0.5) <= 0.05
               and -1 in ladder and abs(abs(ladder[-1]) - 0.5) <= 0.05)
    s1 = bool(s1_exact and s1_half)

    # --- S2: 2π flips the ½, 4π restores; integer no flip
    s2 = bool(dc["half_2pi"] < -0.9 and dc["half_4pi"] > 0.9
              and dc["int_2pi"] > 0.9)

    # --- S3: conserved (strength + holonomy), fused, annihilated & healed
    s3_pin = (abs(m["s_pinned"] - 0.5) <= 0.05 and m["holo_pinned"][0] < -0.9
              and m["holo_pinned"][1] > 0.9)
    s3_fuse = abs(m["s_fused"] - 1.0) <= 0.1
    s3_ann = (abs(m["s_ann0"]) <= 0.15 and abs(m["s_ann1"]) <= 0.15
              and m["amp_ann"] > m["amp_seed"] + 0.15)
    s3 = bool(s3_pin and s3_fuse and s3_ann)

    lines = ["Half-integer spin: the nematic ½-disclination — verdict",
             "=" * 74, ""]
    lines.append(
        "S1 (the elementary defect is a half-integer disclination): s(q) = "
        + ", ".join(f"{v:+.2f}(q={q:+d})" for q, v in ladder.items()) + " — "
        + ("✓ s = q/2 exactly: the elementary q=±1 defect is a ±½ disclination "
           "(director winds by ±π, half-integer), impossible for a vector field."
           if s1 else "✗ the strengths are not the half-integer s = q/2 ladder."))
    lines.append(
        "S2 (the double cover — the spinor's −1): oriented director n·n after "
        f"2π = {dc['half_2pi']:+.2f}, after 4π = {dc['half_4pi']:+.2f} (½); "
        f"{dc['int_2pi']:+.2f} at 2π (integer) — "
        + ("✓ a 2π loop around the ½ disclination flips the director (−1) and "
           "4π restores it, while the integer defect never flips — the SU(2) "
           "double cover of SO(3), the signature of half-integer spin." if s2
           else "✗ the 2π/4π double-cover holonomy does not hold."))
    lines.append(
        "S3 (conserved, fused, bound to matter): pinned+evolved s="
        f"{m['s_pinned']:+.2f} holonomy=({m['holo_pinned'][0]:+.2f},"
        f"{m['holo_pinned'][1]:+.2f}); fusion ½+½ → s={m['s_fused']:+.2f}; "
        f"annihilation ½−½ → s=({m['s_ann0']:+.2f},{m['s_ann1']:+.2f}), heals "
        f"{m['amp_seed']:.2f}→{m['amp_ann']:.2f} — "
        + ("✓ the ½ charge is conserved (strength and −1 holonomy) on matter "
           "under the self-sustained CGL, two ½'s fuse to an integer, and a "
           "½/−½ pair annihilates — additive and protected." if s3 else
           "✗ conservation, fusion, or annihilation does not hold cleanly."))
    lines.append("")
    lines.append(f"score: {int(s1)+int(s2)+int(s3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: this is the ORDER-PARAMETER (topological) realisation of "
        "half-integer spin — a genuine Z₂/RP¹ double cover with π-winding "
        "disclinations, the topological essence of a spinor (a 2π rotation gives "
        "−1, 4π restores).  It is NOT the full quantum Dirac field: no spin–"
        "statistics (anticommutation), no Dirac equation, no dynamical fermion — "
        "the deeper frontier toward genuinely fermionic matter.  What is measured "
        "is the half-integer winding (s = q/2), the 4π double cover, and the "
        "disclination fusion rules.  The nematic is carried by ψ = e^{2iθ}, so "
        "the U(1) vortex machinery applies but the observable director is double-"
        "valued in the lift; 2-D, one lattice, the elementary disclination ±½.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_spinor.json"), "w") as fh:
        json.dump({"params": vars(args),
                   "s1_ladder": {str(k): v for k, v in ladder.items()},
                   "s2_double_cover": dc, "s3_matter": m,
                   "analysis": {"s1": bool(s1), "s2": bool(s2), "s3": bool(s3)}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
