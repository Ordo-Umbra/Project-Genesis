"""The junction is the fermion: N⋆ = 3 selects the spin-½ constituent count.

The hadron experiment (`n3_hadron_spin`) built meson- and baryon-like composites
but named its own gap: the constituent count was put in by hand — nothing
selected 2 or 3.  This experiment tests the program's standing candidate for
that selection, the ``N⋆ = 3`` sector structure — and the mechanism is exact:
**a spin defect's ``2π`` phase winding must pass through *every* phase sector
once**, so with three sectors the elementary spin-½ defect *is* a trivalent,
three-sector junction — the colour-singlet triple.  The count 3 is not a
modelling choice; it is the sector count.

Everything here is **emergent**: the field is grown from noise (defects formed
by the CGL dynamics, never imprinted), the phase tessellation is read off the
field (`phase_sector_labels`, the ``N⋆ = 3`` binning of Act III §5), and the
defects are found by plaquette winding — instrument meets instrument.

Pre-registered predictions:

J1. **The spin defect and the three-sector junction are the same object.**
    In noise-grown fields (several seeds), every defect sits on a junction
    where all 3 phase sectors meet (within 1.5 sites), every junction site
    sits on a defect (within 2.5), and *all three sectors are present* on a
    loop around every **resolvable** defect (each fraction ``≥ 0.10`` — the
    winding must cross every sector; resolvable = nearest neighbour beyond
    two loop radii, since a tighter ``±`` dipole — annihilation in flight,
    the ``½/−½`` channel — makes the loop read the *pair's* sum); for
    **isolated** defects (nearest neighbour ``≥ 10``) the composition is the
    singlet ``1:1:1`` (each fraction in ``[0.22, 0.45]``, ≥ 3 such defects
    measured); the net winding is 0 (defects in ``±`` pairs).
J2. **That junction is the fermion.**  Read as a nematic, every resolvable
    emergent defect carries ``s = q/2 = ±½`` with the ``−1``/``+1``
    (``2π``/``4π``) double-cover holonomy — while two-sector **walls** carry
    ``s = 0`` and no flip.  The elementary half-integer object and the
    three-sector singlet are one and the same; the "3" of the baryon-like
    triple is ``N⋆``.
J3. **Only at ``N⋆ = 3`` do the matter tessellation and the spin defect
    agree.**  The defect forces all ``P`` phase sectors to meet (measured
    sector-valence ``≈ P`` for ``P = 3, 4, 5``), but the *matter* tessellation
    (multiphase Allen–Cahn) caps its generic junctions at the Plateau valence
    ``d + 1 = 3`` regardless of ``P`` (measured ``≈ 3.0`` for all ``P``).
    The two structures coincide **only at ``P = 3``**: the three-sector world
    is the unique one whose natural junction can carry the elementary spin-½.

Honest scope: this ties the *counts* together — the elementary fermionic
defect is intrinsically 3-sector, and 3 is the unique sector number where the
matter tessellation's own junctions have that structure — but the phase-sector
binning inherits ``N⋆ = 3`` from the program's sector results rather than
deriving it afresh here, and the *composites* of `n3_hadron_spin` (2 and 3
wells) are still assembled, not condensed out of the tessellation.  The full
dynamical story — hadrons crystallising at the junctions of a living
tessellation — is the open rung.  2-D; the wall/junction language is the Act
III census.

Usage::

    python experiments/n3_junction_fermion.py --output-dir artifacts/n3_junction
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.chiral_field import evolve_chiral, phase_sector_labels
from project_genesis.dimensional_forms import dimensional_census
from project_genesis.multiphase import sector_labels, step_multiphase
from project_genesis.nematic_spinor import (
    director_holonomy,
    disclination_strength,
    plaquette_winding,
)


def pdist(a, b, n):
    """Periodic point distance on the n×n torus."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return float(np.hypot(min(dx, n - dx), min(dy, n - dy)))


def defect_list(psi):
    w = plaquette_winding(psi)
    return [((x + 0.5, y + 0.5), int(w[x, y])) for x, y in zip(*np.nonzero(w))]


def junction_sites(labels, n_sectors, r=1):
    """Sites whose radius-r disk contains all ``n_sectors`` sectors."""
    shifts = [np.roll(np.roll(labels, dx, 0), dy, 1)
              for dx in range(-r, r + 1) for dy in range(-r, r + 1)
              if dx * dx + dy * dy <= r * r]
    A = np.stack(shifts)
    m = np.ones(labels.shape, dtype=bool)
    for s in range(n_sectors):
        m &= np.any(A == s, axis=0)
    return m


def sector_fracs(labels, c, n_sectors, radius=4.0, npts=360):
    n = labels.shape[0]
    th = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    xs = np.round(c[0] + radius * np.cos(th)).astype(int) % n
    ys = np.round(c[1] + radius * np.sin(th)).astype(int) % n
    seq = labels[xs, ys]
    return [float(np.mean(seq == s)) for s in range(n_sectors)]


def sector_valence(labels, c, r=3):
    n = labels.shape[0]
    vals = set()
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                vals.add(int(labels[int(c[0] + dx) % n, int(c[1] + dy) % n]))
    return len(vals)


def grow_field(args, seed):
    return evolve_chiral((args.size, args.size), chiral=0.0,
                         steps=args.steps, dt=0.05, seed=seed)


def scan_seeds(args):
    """J1 + J2 raw data across seeds."""
    rows = []
    for seed in range(args.n_seeds):
        psi = grow_field(args, args.seed + 7 * seed)
        labels = phase_sector_labels(psi, 3)
        defects = defect_list(psi)
        jm = junction_sites(labels, 3, r=1)
        jsites = list(zip(*np.nonzero(jm)))
        d2j = [min(pdist(c, j, args.size) for j in jsites) if jsites else 99.0
               for c, _ in defects]
        j2d = [min(pdist(j, c, args.size) for c, _ in defects) if defects
               else 99.0 for j in jsites]
        per_defect = []
        for c, q in defects:
            fr = sector_fracs(labels, c, 3)
            s = disclination_strength(psi, c, radius=4)
            h2, h4 = director_holonomy(psi, c, radius=4)
            nn = min((pdist(c, c2, args.size) for c2, _ in defects
                      if c2 != c), default=99.0)
            per_defect.append({"charge": q, "fracs": fr, "s": s,
                               "h2": h2, "h4": h4, "nn": nn,
                               # the radius-4 loop measures ONE defect only if
                               # the nearest other core is > 2 loop-radii out;
                               # tighter pairs are annihilating ± dipoles (the
                               # ½/−½ channel), whose loop reads the pair's sum
                               "resolvable": bool(nn >= 8.0),
                               "isolated": bool(nn >= 10.0)})
        # wall controls: 2 sectors nearby, far from every defect
        walls = []
        for x in range(0, args.size, 3):
            for y in range(0, args.size, 3):
                if len(walls) >= 6:
                    break
                if jm[x, y]:
                    continue
                if sector_valence(labels, (x, y), r=2) == 2 and \
                        min(pdist((x, y), c, args.size)
                            for c, _ in defects) > 10:
                    s = disclination_strength(psi, (x, y), radius=3)
                    h2, _ = director_holonomy(psi, (x, y), radius=3)
                    walls.append({"s": s, "h2": h2})
        rows.append({"seed": seed, "n_defects": len(defects),
                     "net": sum(q for _, q in defects),
                     "d2j_max": float(max(d2j)) if d2j else 0.0,
                     "j2d_max": float(max(j2d)) if j2d else 0.0,
                     "defects": per_defect, "walls": walls,
                     "psi": psi, "positions": [c for c, _ in defects]})
        print(f"  seed {seed}: {len(defects)} defects (net "
              f"{rows[-1]['net']:+d})  d→junction ≤ {rows[-1]['d2j_max']:.1f}  "
              f"junction→d ≤ {rows[-1]['j2d_max']:.1f}  s = "
              + ", ".join(f"{d['s']:+.2f}" for d in per_defect)
              + f"  walls s = "
              + ", ".join(f"{w['s']:+.2f}" for w in walls), flush=True)
    return rows


def p_scan(args, rows):
    """J3: defect sector-valence vs the matter tessellation's, across P."""
    out = []
    for P in args.palettes:
        dv = []
        for row in rows:
            labP = phase_sector_labels(row["psi"], P)
            dv += [sector_valence(labP, c, r=3) for c in row["positions"]]
        rng = np.random.default_rng(args.seed)
        f = 0.1 * rng.standard_normal((P, args.size, args.size))
        for _ in range(args.matter_steps):
            f = step_multiphase(f, diffusion=1.0, gamma=1.5, dt=0.1)
        mv = float(dimensional_census(sector_labels(f))["mean_valence"])
        out.append({"P": P, "defect_valence": float(np.mean(dv)),
                    "matter_valence": mv})
        print(f"  P={P}: defect sector-valence = {np.mean(dv):.2f}  "
              f"matter junction valence = {mv:.2f}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--palettes", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--matter-steps", type=int, default=250)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_junction")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_seeds = 2
        args.steps = 1000

    os.makedirs(args.output_dir, exist_ok=True)
    print("the junction is the fermion: N⋆=3 selects the spin-½ count",
          flush=True)
    print("emergent defect scan (J1, J2):", flush=True)
    rows = scan_seeds(args)
    print("sector-count scan (J3):", flush=True)
    pscan = p_scan(args, rows)

    all_defects = [d for r in rows for d in r["defects"]]
    all_walls = [w for r in rows for w in r["walls"]]

    # --- J1: bijection + all-sectors-present + isolated-singlet + pairing.
    # Loop-based reads apply to RESOLVABLE defects (nearest neighbour beyond
    # two loop radii); tighter ± dipoles — annihilation in flight, the ½/−½
    # channel — are counted and reported, their loops reading the pair's sum.
    resolvable = [d for d in all_defects if d["resolvable"]]
    isolated = [d for d in all_defects if d["isolated"]]
    dipoles = len(all_defects) - len(resolvable)
    j1 = bool(all(r["net"] == 0 for r in rows)
              and all(r["d2j_max"] <= 1.5 for r in rows)
              and all(r["j2d_max"] <= 2.5 for r in rows)
              and len(resolvable) >= 4
              and all(f >= 0.10 for d in resolvable for f in d["fracs"])
              and len(isolated) >= 3
              and all(0.22 <= f <= 0.45 for d in isolated
                      for f in d["fracs"]))

    # --- J2: every resolvable defect is a spin-1/2; walls are not
    j2 = bool(len(resolvable) >= 6
              and all(abs(d["s"] - d["charge"] / 2.0) <= 0.1
                      and d["h2"] < -0.9 and d["h4"] > 0.9
                      for d in resolvable)
              and len(all_walls) >= 5
              and all(abs(w["s"]) <= 0.1 and w["h2"] > 0.9
                      for w in all_walls))

    # --- J3: valences match only at P = 3
    j3_parts = {r["P"]: abs(r["defect_valence"] - r["matter_valence"])
                for r in pscan}
    j3 = bool(all(abs(r["matter_valence"] - 3.0) <= 0.15 for r in pscan)
              and all(abs(r["defect_valence"] - r["P"]) <= 0.4 for r in pscan)
              and j3_parts.get(3, 9) <= 0.5
              and all(v >= 0.6 for P, v in j3_parts.items() if P != 3))

    lines = ["The junction is the fermion — verdict", "=" * 74, ""]
    lines.append(
        "J1 (the spin defect IS the three-sector junction): "
        + "; ".join(f"seed {r['seed']}: {r['n_defects']} defects, net "
                    f"{r['net']:+d}, d→j ≤ {r['d2j_max']:.1f}, j→d ≤ "
                    f"{r['j2d_max']:.1f}" for r in rows)
        + f"; {len(resolvable)} resolvable, {len(isolated)} isolated, "
        f"{dipoles} in annihilating ± dipoles — "
        + ("✓ grown from noise, every defect sits on a junction of all three "
           "phase sectors and every junction on a defect (all three sectors "
           "on every resolvable defect's loop), with the singlet 1:1:1 "
           "composition for every isolated defect — one object, in ± pairs."
           if j1 else
           "✗ the defect–junction bijection or the singlet composition fails."))
    lines.append(
        "J2 (that junction is the fermion): s = "
        + ", ".join(f"{d['s']:+.2f}" for d in resolvable)
        + "; walls s = " + ", ".join(f"{w['s']:+.2f}" for w in all_walls)
        + " — "
        + ("✓ every emergent three-sector junction carries s = ±½ with the "
           "−1/+1 double cover, and two-sector walls carry none — the "
           "elementary fermion and the colour-singlet triple are the same "
           "object, so the baryon's '3' is N⋆." if j2 else
           "✗ some junction is not the clean ±½ / some wall carries spin."))
    lines.append(
        "J3 (only at N⋆ = 3 do matter and spin agree): "
        + "; ".join(f"P={r['P']}: defect {r['defect_valence']:.2f} vs matter "
                    f"{r['matter_valence']:.2f}" for r in pscan)
        + " — "
        + ("✓ the defect needs all P sectors, the matter tessellation caps at "
           "the Plateau 3 — they coincide only at P = 3: the three-sector "
           "world is the unique one whose own junctions can carry the "
           "elementary spin-½." if j3 else
           "✗ the two valence structures do not single out P = 3."))
    lines.append("")
    lines.append(f"score: {int(j1)+int(j2)+int(j3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: this ties the COUNTS together — the elementary fermionic "
        "defect is intrinsically three-sector (its 2π winding must cross every "
        "sector), and 3 is the unique sector number where the matter "
        "tessellation's own generic junctions (Plateau, d+1) have exactly that "
        "structure — but the phase binning inherits N⋆ = 3 from the program's "
        "sector results rather than deriving it afresh, and the composites of "
        "n3_hadron_spin are still assembled at wells, not condensed out of the "
        "tessellation.  The dynamical version — hadrons crystallising at the "
        "junctions of a living tessellation — is the open rung.  2-D, emergent "
        "fields (grown from noise, never imprinted), several seeds; the "
        "wall/junction language is the Act III census.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_junction_fermion.json"),
              "w") as fh:
        json.dump({"params": vars(args),
                   "seeds": [{k: v for k, v in r.items()
                              if k not in ("psi", "positions")}
                             for r in rows],
                   "p_scan": pscan,
                   "analysis": {"j1": bool(j1), "j2": bool(j2),
                                "j3": bool(j3)}}, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
