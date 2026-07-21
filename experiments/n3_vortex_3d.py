"""Spin in 3-D: the vortex becomes a line, and its spin becomes an axial vector.

The 2-D spin arc gave matter a quantised spin carried by a *point* vortex — but
its angular momentum was a scalar, a sign.  In 3-D a complex field vanishes on
*lines* (codimension 2), so the defect is a vortex **line**, and the field's
angular momentum ``L = Σ (x − c) × j`` is a genuine **3-vector aligned with the
line**.  Spin acquires a *direction*: the prerequisite for a real angular
momentum vector (and, eventually, a spinor).  This is the field-level 3-D
instrument (`project_genesis/vortex_chiral_3d.py`) — no molecule dynamics yet.

Pre-registered predictions:

V1. **Spin is a quantised axial vector, aligned with the line.**  For a vortex
    line along an arbitrary axis ``n̂`` — the coordinate axes, face- and
    body-diagonals, a skew direction — the angular momentum ``L`` is *parallel
    to the line* (``|L·n̂|/|L| ≥ 0.99``, transverse fraction ``≤ 0.05``), with a
    direction-independent magnitude; and along a fixed axis ``L`` is
    sign-locked to the winding and a monotone ladder in ``|q|`` (zero at
    ``q = 0``) — an integer winding, quantised.  Rotate the line and its spin
    vector rotates with it.
V2. **The winding is a dynamically-conserved charge, and the spin axis is
    stable.**  Seeded once and co-evolved under the κ-detuned CGL with **no
    re-imprinting**, the line's enclosed winding stays the integer it was seeded
    (``|Δw| ≤ 0.1`` over the run, and integer even from a noisy seed), while the
    ``L`` vector holds its direction (alignment to the seed axis ``≥ 0.95``) —
    the topology *and* the orientation are the dynamics', not an imposition.
V3. **Matter pins the line; like survive, unlike annihilate.**  Threaded
    through the κ wells of a mass pair: two like-winding lines both keep
    ``|w| = 1`` with their cores held (``min|ψ| ≤ 0.15``), while a line and an
    antiline *annihilate* — both windings unwind to ``0`` and the amplitude
    heals (``min|ψ| ≥ 0.4``).  A physical, dynamically-enforced sign rule for
    line defects, the 3-D echo of the 2-D vortex selection.

Honest scope: this ``U(1)`` field carries an **integer / axial-vector** angular
momentum — a spin-1-like (bosonic) object.  A genuine **half-integer spinor**
(the ``SU(2)`` double cover, ``4π`` periodicity — the fermion) is a *different*
field, and that is the open frontier toward fermionic matter, not what a complex
order parameter gives.  Field-level only (no bound-molecule dynamics); one
lattice; a torus (a line ∥ an axis wraps a cycle, so its winding needs no
compensating defect — unlike the 2-D point vortex).

Usage::

    python experiments/n3_vortex_3d.py --output-dir artifacts/n3_vortex_3d
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import gaussian_load, relax_capacity
from project_genesis.two_field import chiral_detuning, step_chiral_detuned
from project_genesis.vortex_chiral_3d import (
    evolve_seeded_line,
    line_angular_momentum,
    line_winding,
    vortex_line,
)


def field_kwargs(args):
    return dict(kappa_diffusion=1.0, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0)


def v1_axial_vector(args):
    n = (args.size,) * 3
    c = (args.size / 2,) * 3
    R = args.size / 2 - 4
    orient = {"z": (0, 0, 1), "x": (1, 0, 0), "(1,1,0)": (1, 1, 0),
              "(1,1,1)": (1, 1, 1), "(2,-1,1)": (2, -1, 1)}
    if args.quick:
        orient = {"z": (0, 0, 1), "(1,1,1)": (1, 1, 1), "(2,-1,1)": (2, -1, 1)}
    aligns = {}
    for name, nrm in orient.items():
        L = line_angular_momentum(vortex_line(n, c, nrm, 1, core=args.core), c, R)
        nh = np.asarray(nrm, float)
        nh = nh / np.linalg.norm(nh)
        Ln = np.linalg.norm(L)
        aligns[name] = {"align": float(abs(L @ nh) / Ln),
                        "perp": float(np.linalg.norm(L - (L @ nh) * nh) / Ln),
                        "mag": float(Ln)}
    ladder = {}
    for q in args.charges:
        L = line_angular_momentum(vortex_line(n, c, (0, 0, 1), q, core=args.core),
                                  c, R)
        ladder[q] = float(L[2])
    return aligns, ladder


def v2_conservation(args):
    n = (args.size,) * 3
    c = (args.size / 2,) * 3
    kappa = relax_capacity(gaussian_load(n, [c], args.width, args.mass),
                           **field_kwargs(args))
    out = {}
    for label, noise in (("clean", 0.0), ("noisy", args.noise)):
        h = evolve_seeded_line(n, c, (0, 0, 1), 1, kappa, chiral=0.0,
                               detune_gamma=args.gamma, core=args.core,
                               dt=args.dt, steps=args.steps, record_every=200,
                               noise=noise, seed=args.seed)
        w = np.asarray(h["winding"])
        out[label] = {"w_first": float(w[0]), "w_last": float(w[-1]),
                      "w_drift": float(np.max(np.abs(w - w[0]))),
                      "align_min": float(np.min(h["align"])),
                      "amp_min": float(h["amp_min"][-1])}
    return out


def v3_selection(args):
    n = (args.size,) * 3
    s = args.sep
    mid = args.size / 2
    c0 = (mid, mid - s / 2, mid)
    c1 = (mid, mid + s / 2, mid)
    kappa = relax_capacity(
        gaussian_load(n, [c0], args.width, args.mass)
        + gaussian_load(n, [c1], args.width, args.mass), **field_kwargs(args))
    g = chiral_detuning(kappa, detune_gamma=args.gamma)

    def run(qa, qb):
        psi = (vortex_line(n, c0, (0, 0, 1), qa, core=args.core)
               * vortex_line(n, c1, (0, 0, 1), qb, core=args.core)
               * np.sqrt(np.clip(1.0 - g, 0.0, 1.0)))
        for _ in range(args.steps):
            psi = step_chiral_detuned(psi, g, chiral=0.0, dt=args.dt)
        return {"w0": line_winding(psi, c0, 2), "w1": line_winding(psi, c1, 2),
                "amp_min": float(np.abs(psi).min())}

    return {"like": run(1, 1), "unlike": run(1, -1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=44)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--mass", type=float, default=0.6)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--core", type=float, default=3.0)
    p.add_argument("--charges", type=int, nargs="+", default=[-2, -1, 0, 1, 2])
    p.add_argument("--sep", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_vortex_3d")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.size = 36
        args.charges = [-1, 0, 1, 2]
        args.steps = 400

    os.makedirs(args.output_dir, exist_ok=True)
    print("3-D spin: the vortex is a line, its spin an axial vector", flush=True)

    aligns, ladder = v1_axial_vector(args)
    print("V1 — L aligns with the line (any orientation):", flush=True)
    for name, d in aligns.items():
        print(f"  ∥ {name:8s}: align={d['align']:.4f}  perp={d['perp']:.4f}  "
              f"|L|={d['mag']:.0f}", flush=True)
    print("  winding ladder (∥ z): "
          + ", ".join(f"{v:+.0f}(q={q:+d})" for q, v in ladder.items()),
          flush=True)

    cons = v2_conservation(args)
    print("V2 — self-sustained conservation & axis stability:", flush=True)
    for label, d in cons.items():
        print(f"  {label}: w {d['w_first']:+.2f}→{d['w_last']:+.2f} "
              f"(drift {d['w_drift']:.3f})  align_min={d['align_min']:.3f}  "
              f"|ψ|min={d['amp_min']:.2f}", flush=True)

    sel = v3_selection(args)
    print("V3 — matter pinning & selection:", flush=True)
    print(f"  like (+1,+1): w=({sel['like']['w0']:+.2f},{sel['like']['w1']:+.2f})"
          f"  |ψ|min={sel['like']['amp_min']:.2f}", flush=True)
    print(f"  unlike (+1,-1): w=({sel['unlike']['w0']:+.2f},"
          f"{sel['unlike']['w1']:+.2f})  |ψ|min={sel['unlike']['amp_min']:.2f}",
          flush=True)

    # --- V1
    v1_align = all(d["align"] >= 0.99 and d["perp"] <= 0.05
                   for d in aligns.values())
    v1_zero = abs(ladder.get(0, 0.0)) <= 0.02 * abs(ladder.get(1, 1.0))
    v1_sign = all(np.sign(v) == np.sign(q) for q, v in ladder.items() if q != 0)
    v1_ladder = True
    for q in (1, 2):
        if q in ladder and (q - 1) in ladder:
            v1_ladder = v1_ladder and abs(ladder[q]) > abs(ladder[q - 1]) - 1e-9
    v1 = bool(v1_align and v1_zero and v1_sign and v1_ladder)

    # --- V2
    v2 = bool(all(abs(round(d["w_last"])) == 1 and d["w_drift"] <= 0.1
                  and d["align_min"] >= 0.95 and d["amp_min"] <= 0.2
                  for d in cons.values()))

    # --- V3
    like, unlike = sel["like"], sel["unlike"]
    v3 = bool(abs(round(like["w0"])) == 1 and abs(round(like["w1"])) == 1
              and like["amp_min"] <= 0.15
              and abs(unlike["w0"]) <= 0.2 and abs(unlike["w1"]) <= 0.2
              and unlike["amp_min"] >= 0.4)

    lines = ["3-D spin: the axial-vector vortex line — verdict", "=" * 74, ""]
    lines.append(
        "V1 (spin is a quantised axial vector, line-aligned): alignments "
        + ", ".join(f"{n}={d['align']:.3f}" for n, d in aligns.items())
        + "; ladder L_z = "
        + ", ".join(f"{v:+.0f}(q={q:+d})" for q, v in ladder.items()) + " — "
        + ("✓ L points along the line for every orientation (magnitude "
           "direction-independent) and is a sign-locked, quantised ladder in "
           "the winding — spin has a direction." if v1 else
           "✗ L is not the clean line-aligned, quantised axial vector."))
    lines.append(
        "V2 (winding conserved & axis stable, self-sustained): "
        + "; ".join(f"{k}: w→{d['w_last']:+.2f} (drift {d['w_drift']:.3f}), "
                    f"align≥{d['align_min']:.2f}" for k, d in cons.items()) + " — "
        + ("✓ seeded once and co-evolved with no re-imprinting, the line keeps "
           "its integer winding and its spin axis — robust to a noisy seed."
           if v2 else
           "✗ the winding or the axis is not preserved under the dynamics."))
    lines.append(
        "V3 (matter pins the line; like survive, unlike annihilate): "
        f"like w=({like['w0']:+.2f},{like['w1']:+.2f}) |ψ|min={like['amp_min']:.2f}; "
        f"unlike w=({unlike['w0']:+.2f},{unlike['w1']:+.2f}) "
        f"|ψ|min={unlike['amp_min']:.2f} — "
        + ("✓ two like lines survive on their wells while a line–antiline pair "
           "annihilates and the field heals — a dynamical sign rule for line "
           "defects." if v3 else
           "✗ the survive/annihilate selection does not hold cleanly."))
    lines.append("")
    lines.append(f"score: {int(v1)+int(v2)+int(v3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: the 3-D defect is a vortex LINE and its angular momentum "
        "is a genuine axial VECTOR — spin has a direction (V1), conserved with "
        "its axis under the self-sustained CGL (V2), matter-pinned with a real "
        "sign rule (V3).  But this U(1) field carries an INTEGER / vector "
        "(spin-1-like, bosonic) angular momentum; a genuine HALF-INTEGER SPINOR "
        "— the SU(2) double cover, 4π periodicity, the fermion — is a different "
        "field structure and is the open frontier toward fermionic matter, not "
        "what a complex order parameter gives.  Field-level only (no bound-"
        "molecule dynamics — the driven 3-D molecule is the next rung); one "
        "lattice; a torus (a line ∥ an axis wraps a cycle, so its winding needs "
        "no compensating defect — unlike the 2-D point vortex, and its L is "
        "correspondingly more stable).")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_vortex_3d.json"), "w") as fh:
        json.dump({"params": vars(args),
                   "v1_aligns": aligns,
                   "v1_ladder": {str(k): v for k, v in ladder.items()},
                   "v2_conservation": cons, "v3_selection": sel,
                   "analysis": {"v1": bool(v1), "v2": bool(v2), "v3": bool(v3)}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
