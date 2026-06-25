#!/usr/bin/env python3
"""Monte-Carlo measurement of pure-gauge confinement signatures.

Measures the Wilson-loop area law, string tension (sigma), Creutz ratios,
and the Polyakov-loop order parameter for a pure SU(N) lattice gauge theory
using the thermodynamic ensemble sampler in project_genesis.gauge_mc.

Usage examples
--------------
Single beta point (default)::

    python experiments/mc_confinement.py --beta 2.5 --size 8 --n 2

Full beta scan to observe the confinement/deconfinement crossover::

    python experiments/mc_confinement.py --scan \\
        --beta-min 1.0 --beta-max 5.0 --beta-steps 9 \\
        --size 6 --n 2 --updater heatbath

Quick smoke test::

    python experiments/mc_confinement.py --quick

Output
------
For a single-beta run the script prints:
  * Ensemble parameters (beta, size, N, updater, sweeps)
  * Wilson loop table W(R,T) for each measured (R,T) pair
  * Area-law fit: string tension sigma, perimeter coefficient c, residual
  * Creutz ratios chi(R,T) for all adjacent loop pairs
  * Polyakov loop mean and susceptibility
  * Confinement verdict (sigma > 0 consistent with confined phase)

For --scan mode the script prints a compact table with one row per beta.

All output is plain text and suitable for piping into a log file.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from project_genesis.gauge_mc import (
    thermalize_and_measure_pure_gauge,
    deconfinement_scan,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _separator(width: int = 72, char: str = "-") -> str:
    return char * width


def _print_header(args: argparse.Namespace) -> None:
    print()
    print(_separator(72, "="))
    print("  Project Genesis — Pure-Gauge Monte Carlo Confinement Study")
    print(_separator(72, "="))
    print(f"  Gauge group : SU({args.n})")
    print(f"  Lattice     : {args.size}^{args.ndim}  (ndim={args.ndim})")
    print(f"  Updater     : {args.updater}")
    print(f"  Therm sweeps: {args.n_therm}   Meas: {args.n_meas}   Skip: {args.n_skip}")
    print(f"  Seed        : {args.seed}")
    print(_separator(72, "="))
    print()


def _print_single_result(summary: dict, loop_sizes: list[tuple[int, int]]) -> None:
    """Pretty-print a single-beta measurement summary."""
    beta = summary["beta_g"]
    fit = summary["area_law_fit"]

    print(f"  beta_g = {beta:.4f}")
    print()

    # Wilson loop table
    la = summary["loop_averages"]
    rs = sorted({s[0] for s in loop_sizes})
    ts = sorted({s[1] for s in loop_sizes})
    col_w = 10
    header = "  W(R,T)  " + "".join(f"  T={t:<{col_w-4}}" for t in ts)
    print(header)
    print("  " + _separator(len(header) - 2))
    for r in rs:
        row = f"  R={r:<7}"
        for t in ts:
            key = f"W_{r}_{t}"
            val = la.get(key, float("nan"))
            row += f"  {val:<{col_w}.5f}"
        print(row)
    print()

    # Area-law fit
    sigma = fit["sigma"]
    c = fit["perimeter_coeff"]
    resid = fit["fit_residual"]
    print("  Area-law fit:  W(R,T) = exp(-sigma*R*T - c*(R+T))")
    print(f"    sigma (string tension)  = {sigma:+.6f}")
    print(f"    c     (perimeter coeff) = {c:+.6f}")
    if not math.isnan(resid):
        print(f"    fit residual (mean |log err|) = {resid:.2e}")
    print()

    # Creutz ratios
    creutz = fit.get("creutz_ratios", {})
    if creutz:
        print("  Creutz ratios chi(R,T) [perimeter-subtracted sigma estimator]:")
        for key, val in sorted(creutz.items()):
            marker = "  <-- nan (loop <= 0)" if math.isnan(val) else ""
            print(f"    {key:12s} = {val:+.6f}{marker}")
        print()

    # Polyakov loop
    p_mean = summary["polyakov_mean"]
    p_susc = summary.get("polyakov_susceptibility", float("nan"))
    print(f"  Polyakov loop  <|P|> = {p_mean:.5f}   susc = {p_susc:.5f}")
    print()

    # Final action
    print(f"  Final Wilson action  S = {summary['final_wilson_action']:.3f}")
    print()

    # Confinement verdict
    print("  " + _separator(60))
    if sigma > 0.005:
        verdict = f"CONFINED  (sigma = {sigma:.4f} > 0 — area law active)"
    elif sigma < -0.005:
        verdict = f"DECONFINED  (sigma = {sigma:.4f} < 0 — area law absent)"
    else:
        verdict = f"AMBIGUOUS  (sigma = {sigma:.4f} ~ 0 — near transition or low stats)"
    print(f"  Verdict: {verdict}")
    print("  " + _separator(60))
    print()


def _print_scan_table(results: list[dict]) -> None:
    """Print a compact table of sigma, Polyakov loop vs beta."""
    print()
    print(_separator(72, "="))
    print("  Beta scan — confinement/deconfinement crossover")
    print(_separator(72, "="))
    print(f"  {'beta':>8}  {'sigma':>10}  {'perimeter_c':>12}  "
          f"{'<|P|>':>8}  {'P_susc':>8}  {'verdict'}")
    print("  " + _separator(70))
    for r in results:
        beta = r["beta_g"]
        fit = r["area_law_fit"]
        sigma = fit["sigma"]
        c = fit["perimeter_coeff"]
        p_mean = r["polyakov_mean"]
        p_susc = r.get("polyakov_susceptibility", float("nan"))
        if sigma > 0.005:
            v = "confined"
        elif sigma < -0.005:
            v = "deconfined"
        else:
            v = "crossover?"
        print(f"  {beta:>8.3f}  {sigma:>+10.5f}  {c:>+12.5f}  "
              f"{p_mean:>8.5f}  {p_susc:>8.5f}  {v}")
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # lattice
    p.add_argument("--size", type=int, default=8,
                   help="Lattice side length (default: 8)")
    p.add_argument("--ndim", type=int, default=2,
                   help="Spatial dimensions (default: 2)")
    p.add_argument("--n", type=int, default=2,
                   help="Gauge group SU(N) (default: 2)")
    # single-beta mode
    p.add_argument("--beta", type=float, default=2.5,
                   help="Inverse gauge coupling beta_g (default: 2.5)")
    # scan mode
    p.add_argument("--scan", action="store_true",
                   help="Run a beta scan instead of a single-beta measurement")
    p.add_argument("--beta-min", type=float, default=1.0,
                   help="Minimum beta for scan (default: 1.0)")
    p.add_argument("--beta-max", type=float, default=5.0,
                   help="Maximum beta for scan (default: 5.0)")
    p.add_argument("--beta-steps", type=int, default=9,
                   help="Number of beta values in scan (default: 9)")
    # MC parameters
    p.add_argument("--updater",
                   choices=["metropolis", "heatbath", "overrelax"],
                   default="metropolis",
                   help="Update algorithm (default: metropolis)")
    p.add_argument("--n-therm", type=int, default=200,
                   help="Thermalisation sweeps (default: 200)")
    p.add_argument("--n-meas", type=int, default=50,
                   help="Measurement sweeps (default: 50)")
    p.add_argument("--n-skip", type=int, default=5,
                   help="Sweeps between measurements (default: 5)")
    p.add_argument("--step-scale", type=float, default=0.18,
                   help="Metropolis step scale — tune for ~50%% acceptance")
    # loop sizes
    p.add_argument("--loop-sizes", type=str, default="1x1,2x2,3x3,2x3,3x2",
                   help="Comma-separated R×T loop sizes, e.g. '1x1,2x2,3x3' (default: 1x1,2x2,3x3,2x3,3x2)")
    # misc
    p.add_argument("--quick", action="store_true",
                   help="Quick smoke-test mode: small lattice, few sweeps")
    p.add_argument("--seed", type=int, default=7,
                   help="RNG seed (default: 7)")
    return p


def _parse_loop_sizes(raw: str) -> list[tuple[int, int]]:
    sizes = []
    for token in raw.split(","):
        token = token.strip()
        if "x" in token:
            r, t = token.split("x", 1)
            sizes.append((int(r), int(t)))
        else:
            v = int(token)
            sizes.append((v, v))
    return sizes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.size = min(args.size, 5)
        args.n_therm = min(args.n_therm, 20)
        args.n_meas = min(args.n_meas, 8)
        args.n_skip = 2
        args.beta_steps = min(args.beta_steps, 3)
        print("[quick mode: reduced lattice and sweep counts]")

    loop_sizes = _parse_loop_sizes(args.loop_sizes)

    # Clamp loop sizes to fit within the lattice
    max_extent = args.size - 1
    loop_sizes = [(min(r, max_extent), min(t, max_extent))
                  for r, t in loop_sizes if r >= 1 and t >= 1]
    if not loop_sizes:
        loop_sizes = [(1, 1)]

    rng = np.random.default_rng(args.seed)
    _print_header(args)

    if args.scan:
        # -------------------------------------------------------------------
        # Beta scan mode
        # -------------------------------------------------------------------
        beta_values = np.linspace(args.beta_min, args.beta_max,
                                  args.beta_steps).tolist()
        print(f"  Scanning {len(beta_values)} beta values: "
              f"{beta_values[0]:.2f} .. {beta_values[-1]:.2f}")
        print(f"  Loop sizes: {loop_sizes}")
        print()

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
        _print_scan_table(results)

    else:
        # -------------------------------------------------------------------
        # Single-beta mode
        # -------------------------------------------------------------------
        print(f"  Loop sizes  : {loop_sizes}")
        print()

        summary, _ = thermalize_and_measure_pure_gauge(
            size=args.size,
            n=args.n,
            beta_g=args.beta,
            rng=rng,
            ndim=args.ndim,
            n_therm=args.n_therm,
            n_meas=args.n_meas,
            n_skip=args.n_skip,
            step_scale=args.step_scale,
            updater=args.updater,
            loop_sizes=loop_sizes,
        )
        _print_single_result(summary, loop_sizes)


if __name__ == "__main__":
    main()
