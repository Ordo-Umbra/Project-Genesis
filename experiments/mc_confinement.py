#!/usr/bin/env python3
"""Monte-Carlo measurement of pure-gauge confinement signatures (v2).

Supports Metropolis, heat-bath, and over-relaxation updaters,
and works in 2-D and 3-D.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from project_genesis.gauge_mc import thermalize_and_measure_pure_gauge


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--ndim", type=int, default=2)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--beta", type=float, default=2.5)
    p.add_argument("--updater", choices=["metropolis", "heatbath", "overrelax"], default="metropolis")
    p.add_argument("--n-therm", type=int, default=80)
    p.add_argument("--n-meas", type=int, default=20)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    if args.quick:
        args.n_therm = min(args.n_therm, 40)
        args.n_meas = min(args.n_meas, 10)

    rng = np.random.default_rng(args.seed)
    print(f"Monte-Carlo confinement (ndim={args.ndim}, updater={args.updater})\n")

    summary, _ = thermalize_and_measure_pure_gauge(
        size=args.size, n=args.n, beta_g=args.beta, rng=rng,
        ndim=args.ndim, n_therm=args.n_therm, n_meas=args.n_meas,
        updater=args.updater
    )

    print(f"  Polyakov mean = {summary['polyakov_mean']:.4f}")
    for k, v in summary["loop_averages"].items():
        print(f"  {k} = {v:.4f}")
    print(f"  Final action  = {summary['final_wilson_action']:.2f}")


if __name__ == "__main__":
    main()
