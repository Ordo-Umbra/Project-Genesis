#!/usr/bin/env python3
"""Monte-Carlo measurement of pure-gauge confinement signatures (Wilson area law,
string tension via Creutz ratios, Polyakov loop).

This experiment implements the top item on the Project Genesis roadmap:
Monte-Carlo sampling of the pure Wilson action to extract thermodynamic
confinement observables.

Usage examples:
    python experiments/mc_confinement.py --size 12 --beta 2.3 --n-meas 60
    python experiments/mc_confinement.py --betas 1.5,2.0,2.5,3.0 --quick
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from project_genesis.gauge_mc import thermalize_and_measure_pure_gauge


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--n", type=int, default=3, help="SU(N) (3 for colour)")
    p.add_argument("--beta", type=float, default=2.5)
    p.add_argument("--betas", type=str, default=None, help="Comma-separated list for sweep")
    p.add_argument("--n-therm", type=int, default=150)
    p.add_argument("--n-meas", type=int, default=40)
    p.add_argument("--n-skip", type=int, default=5)
    p.add_argument("--step-scale", type=float, default=0.18)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    print("Pure-gauge Monte-Carlo confinement signatures (Wilson action)\n")

    betas = [float(b) for b in args.betas.split(",")] if args.betas else [args.beta]
    if args.quick:
        args.n_therm = min(args.n_therm, 60)
        args.n_meas = min(args.n_meas, 15)

    for beta in betas:
        print(f"=== β_g = {beta} ===")
        summary, _ = thermalize_and_measure_pure_gauge(
            size=args.size, n=args.n, beta_g=beta, rng=rng,
            n_therm=args.n_therm, n_meas=args.n_meas, n_skip=args.n_skip,
            step_scale=args.step_scale
        )
        print(f"  ⟨acc⟩ therm = {summary['mean_acceptance_therm']:.3f}")
        print(f"  Polyakov     = {summary['polyakov_mean']:.4f}")
        for k, v in summary["loop_averages"].items():
            print(f"  {k} = {v:.4f}")
        if summary["creutz_ratios_sample"]:
            print(f"  sample Creutz χ(2,2) ≈ {np.nanmean(summary['creutz_ratios_sample']):.4f}")
        print()

    print("Verdict: Wilson loops, Polyakov loop and Creutz ratios measured on ")
    print("thermalised pure-gauge ensembles. String tension extraction is now executable.")


if __name__ == "__main__":
    main()
