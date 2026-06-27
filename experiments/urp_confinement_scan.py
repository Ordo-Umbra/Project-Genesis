"""
experiments/urp_confinement_scan.py
=====================================
CLI experiment: scan β_g across the confinement/deconfinement transition
and record string tension σ, Creutz ratios, and Polyakov-loop order parameter.

Usage
-----
    python experiments/urp_confinement_scan.py
    python experiments/urp_confinement_scan.py --size 12 --ndim 2 --n 2
    python experiments/urp_confinement_scan.py --beta-min 0.5 --beta-max 4.0 --steps 15

Theory anchors
--------------
The scan is designed around the SU(2) β-sectorisation derivation in the URP
framework.  Key qualitative expectations:

  β_g ≲ 2  (strong coupling):
    - ⟨W(R,T)⟩ ~ exp(−σ RT)   →   σ > 0   (area law, confinement)
    - |⟨P⟩| ≈ 0               →   Z_2 centre symmetry unbroken

  β_g ≳ 3  (weak coupling, 2D SU(2)):
    - ⟨W(R,T)⟩ ~ exp(−c(R+T)) →   σ → 0   (perimeter law)
    - |⟨P⟩| > 0               →   centre symmetry broken (deconfined)

The transition β_c depends on the lattice size and dimension; this script
makes it visible by plotting σ and ⟨|P|⟩ vs β.
"""

from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from project_genesis.gauge_mc import deconfinement_scan


def parse_args():
    p = argparse.ArgumentParser(description="URP confinement scan")
    p.add_argument("--size",      type=int,   default=10,  help="Lattice side length")
    p.add_argument("--ndim",      type=int,   default=2,   help="Spatial dimensions")
    p.add_argument("--n",         type=int,   default=2,   help="SU(N) gauge group")
    p.add_argument("--beta-min",  type=float, default=0.5, help="Minimum β_g")
    p.add_argument("--beta-max",  type=float, default=4.5, help="Maximum β_g")
    p.add_argument("--steps",     type=int,   default=12,  help="Number of β values")
    p.add_argument("--n-therm",   type=int,   default=300, help="Thermalisation sweeps")
    p.add_argument("--n-meas",    type=int,   default=80,  help="Measurement sweeps")
    p.add_argument("--n-skip",    type=int,   default=5,   help="Sweeps between measurements")
    p.add_argument("--updater",   type=str,   default="metropolis",
                   choices=["metropolis", "heatbath", "overrelax"])
    p.add_argument("--seed",      type=int,   default=42,  help="RNG seed")
    p.add_argument("--out",       type=str,   default=None,
                   help="Optional JSON output file path")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    beta_values = np.linspace(args.beta_min, args.beta_max, args.steps).tolist()
    loop_sizes = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)]

    print(f"\nURP Confinement Scan — SU({args.n}) on {args.size}^{args.ndim} lattice")
    print(f"Updater: {args.updater} | β ∈ [{args.beta_min}, {args.beta_max}], {args.steps} steps")
    print(f"Thermalise {args.n_therm} sweeps, measure {args.n_meas}×{args.n_skip}\n")
    print(f"{'β':>6}  {'σ (string tension)':>20}  {'|<P>|':>8}  {'W(1,1)':>8}  {'W(2,2)':>8}  {'χ(2,2)':>8}")
    print("-" * 70)

    results = deconfinement_scan(
        size=args.size,
        n=args.n,
        beta_values=beta_values,
        rng=rng,
        ndim=args.ndim,
        n_therm=args.n_therm,
        n_meas=args.n_meas,
        n_skip=args.n_skip,
        updater=args.updater,
        loop_sizes=loop_sizes,
    )

    for r in results:
        beta  = r["beta_g"]
        sigma = r["area_law_fit"]["sigma"]
        poly  = abs(r["polyakov_mean"])
        w11   = r["loop_averages"].get("W_1_1", float("nan"))
        w22   = r["loop_averages"].get("W_2_2", float("nan"))
        chi   = r["area_law_fit"]["creutz_ratios"].get("chi_2_2", float("nan"))
        print(f"{beta:6.3f}  {sigma:>20.5f}  {poly:8.4f}  {w11:8.4f}  {w22:8.4f}  {chi:8.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {out_path}")

    # Summary: identify approximate transition
    confined   = [(r["beta_g"], r["area_law_fit"]["sigma"]) for r in results
                  if r["area_law_fit"]["sigma"] > 0]
    deconfined = [(r["beta_g"], r["area_law_fit"]["sigma"]) for r in results
                  if r["area_law_fit"]["sigma"] <= 0]
    if confined:
        print(f"\nLast confined β (σ>0):   {confined[-1][0]:.3f}  (σ={confined[-1][1]:.4f})")
    if deconfined:
        print(f"First deconfined β (σ≤0): {deconfined[0][0]:.3f}  (σ={deconfined[0][1]:.4f})")
    print()


if __name__ == "__main__":
    main()
