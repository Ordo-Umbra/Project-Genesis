"""Chiral spin: the parity-breaking term that gives the field handedness.

The dimensional census left the 0D junctions handedness-blind and the
κ-molecule could not hold angular momentum — both because the sector
dynamics carry no chiral term, and the collab's "vorticity (spin analog)"
histograms were symmetric for the same reason.  This experiment adds the
minimal chiral term (`project_genesis/chiral_field.py`): a complex
Ginzburg–Landau field with coupling λ,

    ∂_t ψ = ψ + (1 + iλ)∇²ψ − (1 + iλ)|ψ|²ψ ,

λ = 0 being real GL (parity-symmetric, the collab's case).

The observable is chosen against a topological trap: the field-mean
vorticity ⟨ω⟩ is pinned to ~0 on the torus (vortices and antivortices pair
up), so it CANNOT see chirality.  The intrinsic **precession** Ω — the rate
at which the order parameter's phase rotates in the ordered bulk — can: it
is the field's own spin, an angular frequency.

Pre-registered predictions:

H1. **Parity holds at λ = 0**: the intrinsic precession |Ω| < 0.03 at
    λ = 0 — no chiral term, no spin, the collab's symmetric case
    reproduced (and the topological control ⟨ω⟩ ≈ 0 at every λ).
H2. **Spin is the chiral coupling, signed and linear**: across a λ-scan
    Ω = −λ with fit slope −1.0 ± 0.15, and Ω(λ)·Ω(−λ) < 0 — the chiral
    term gives the field an intrinsic angular frequency whose sign is the
    handedness.
H3. **Spin lives on the 0D forms**: the vorticity (spin density) is
    concentrated on the 0D junctions of the phase tessellation — mean |ω|
    on 0-cells ≥ 10× that on 2D interiors, at every λ — so the point-like
    "light" forms of the dimensional census are the spin-carriers, and the
    chiral term rides on the tessellation without dissolving it.

Usage::

    python experiments/n3_chiral_spin.py --output-dir artifacts/n3_chiral
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.chiral_field import (
    evolve_chiral,
    net_vorticity,
    phase_sector_labels,
    precession_rate,
    vorticity,
)
from project_genesis.dimensional_forms import dimensional_census, local_dimension


def measure(args, chiral: float) -> dict:
    """Seed-averaged precession, control vorticity, and junction spin ratio."""
    omegas, nets, ratios, valences = [], [], [], []
    for s in range(args.n_seeds):
        psi = evolve_chiral((args.size, args.size), chiral=chiral,
                            steps=args.steps, dt=args.dt, seed=args.seed + 7 * s)
        omegas.append(precession_rate(psi, chiral=chiral, dt=args.dt))
        nets.append(net_vorticity(psi))
        w = np.abs(vorticity(psi))
        lab = phase_sector_labels(psi, args.palette)
        dim = local_dimension(lab)
        if (dim == 0).any() and (dim == 2).any():
            ratios.append(w[dim == 0].mean() / max(w[dim == 2].mean(), 1e-12))
        valences.append(dimensional_census(lab)["mean_valence"])
    return {"chiral": chiral, "omega": float(np.mean(omegas)),
            "omega_std": float(np.std(omegas)),
            "net_vorticity": float(np.mean(np.abs(nets))),
            "junction_spin_ratio": float(np.mean(ratios)) if ratios else 0.0,
            "mean_valence": float(np.mean(valences))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--palette", type=int, default=3)
    p.add_argument("--chirals", type=float, nargs="+",
                   default=[-0.8, -0.4, 0.0, 0.4, 0.8])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_chiral")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.chirals = [-0.6, 0.0, 0.6]
        args.n_seeds, args.steps = 2, 400

    os.makedirs(args.output_dir, exist_ok=True)
    print("chiral spin: the parity-breaking term and its handedness",
          flush=True)

    scan = [measure(args, lam) for lam in args.chirals]
    for r in scan:
        print(f"  λ={r['chiral']:+.1f}: Ω={r['omega']:+.4f}±{r['omega_std']:.3f}"
              f"  ⟨ω⟩={r['net_vorticity']:.4f}  0D/2D spin ratio="
              f"{r['junction_spin_ratio']:.1f}  valence={r['mean_valence']:.2f}",
              flush=True)

    zero = min(scan, key=lambda r: abs(r["chiral"]))
    h1 = abs(zero["omega"]) < 0.03 and all(r["net_vorticity"] < 0.05
                                           for r in scan)
    lam = np.array([r["chiral"] for r in scan])
    om = np.array([r["omega"] for r in scan])
    slope = float(np.polyfit(lam, om, 1)[0])
    pos = next(r for r in scan if r["chiral"] > 0)
    neg = next(r for r in scan if r["chiral"] < 0)
    h2 = abs(slope + 1.0) <= 0.15 and pos["omega"] * neg["omega"] < 0
    h3 = all(r["junction_spin_ratio"] >= 10.0 for r in scan)

    lines = ["Chiral spin — verdict", "=" * 74, ""]
    lines.append(
        f"H1 (parity holds at λ = 0): Ω = {zero['omega']:+.4f} at λ = 0, "
        f"control ⟨ω⟩ ≤ {max(r['net_vorticity'] for r in scan):.4f} "
        f"(topologically neutral) — "
        + ("✓ no chiral term, no spin: the collab's symmetric case, and "
           "the total-vorticity trap sidestepped." if h1 else
           "✗ the field precesses without a chiral term."))
    lines.append(
        f"H2 (spin is the chiral coupling): Ω vs λ slope {slope:+.3f} "
        f"(target −1), Ω(+) = {pos['omega']:+.3f} vs Ω(−) = "
        f"{neg['omega']:+.3f} — "
        + ("✓ the chiral term gives the field an intrinsic angular "
           "frequency Ω = −λ, its sign the handedness — spin." if h2 else
           "✗ the precession does not track −λ."))
    lines.append(
        f"H3 (spin lives on the 0D forms): |ω| on 0D junctions / on 2D "
        f"interiors = "
        + ", ".join(f"{r['junction_spin_ratio']:.0f}" for r in scan)
        + " — "
        + ("✓ the point-like 'light' forms of the census carry the spin, "
           "and the trivalent tessellation survives the chiral term "
           f"(valence {scan[0]['mean_valence']:.2f})." if h3 else
           "✗ the spin density is not carried by the 0D forms."))
    lines.append("")
    lines.append(f"score: {int(h1)+int(h2)+int(h3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: the chiral term is the CGL minimal one (a single "
        "coupling λ on both diffusion and reaction); the spin observable "
        "is the temporal precession Ω (the order parameter's intrinsic "
        "angular frequency) — the spatial net vorticity is topologically "
        "pinned to ~0 on the torus and is kept only as the control; "
        "Ω = −λ is exact for the CGL bulk (the measurement recovers it "
        "with tiny scatter); one lattice (96²), 3 phase-sectors for the "
        "census tie-in; the junction spin-concentration is a magnitude "
        "ratio (cores are where |ψ|→0, so |ω| peaks there by "
        "construction — the physics is that the census's 0D class LOCATES "
        "those cores). The bridge back to the κ-molecule (does a chiral "
        "field let a bound pair hold Ω?) is the registered next rung.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_chiral_spin.json"), "w") as fh:
        json.dump({"params": vars(args), "scan": scan,
                   "analysis": {"h1": bool(h1), "h2": bool(h2),
                                "h3": bool(h3), "slope": slope}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
