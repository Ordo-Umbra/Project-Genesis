"""Monte Carlo confinement experiment.

Measures the confinement signatures of the pure SU(N) Wilson lattice gauge
theory using the Monte Carlo samplers in ``project_genesis.gauge_mc``.

Physical observables measured
------------------------------
- **Wilson loops W(R,T)** — averaged over all origins; area-law fit extracts
  the **string tension σ** and perimeter coefficient *c*.
- **Creutz ratios χ(R,T)** — perimeter-subtracted string-tension estimators.
- **Polyakov loop ⟨|P|⟩** — order parameter for the deconfinement transition:
  ≈ 0 (confined), > 0 (deconfined).
- **Polyakov susceptibility** — peaks at the deconfinement transition.
- (Optional) **β-scan** over a range of inverse couplings to map the transition.

Usage examples
--------------
# Quick run (SU(2), 2-D, Metropolis, 6^2 lattice)
python experiments/monte_carlo_confinement.py --size 6 --n 2 --beta 1.8

# SU(3) heat-bath, 3-D lattice
python experiments/monte_carlo_confinement.py --size 5 --n 3 --beta 5.0 \
    --ndim 3 --updater heatbath --n-therm 200 --n-meas 50

# Deconfinement β-scan
python experiments/monte_carlo_confinement.py --size 6 --n 2 --beta-scan \
    --beta-min 0.5 --beta-max 3.5 --beta-steps 7 --updater heatbath

Output
------
Results are printed to stdout as a JSON summary and written to
``artifacts/monte_carlo_confinement/`` if ``--output-dir`` is specified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from project_genesis.gauge_mc import (
    thermalize_and_measure_pure_gauge,
    deconfinement_scan,
    fit_area_law,
    wilson_loop,
    polyakov_loop,
)


def _build_loop_sizes(max_r: int, max_t: int) -> list[tuple[int, int]]:
    return [(r, t) for r in range(1, max_r + 1) for t in range(1, max_t + 1)]


def _print_findings(summary: dict) -> None:
    beta = summary["beta_g"]
    fit = summary["area_law_fit"]
    sigma = fit["sigma"]
    c = fit["perimeter_coeff"]
    poly = summary["polyakov_mean"]
    action = summary["final_wilson_action"]

    print(f"\n{'─'*60}")
    print(f"  β_g = {beta:.3f}   SU({summary['n']})   {summary['ndim']}-D   "
          f"L={summary['size']}   updater={summary['updater']}")
    print(f"{'─'*60}")
    print(f"  String tension σ     = {sigma:+.6f}")
    print(f"  Perimeter coeff c    = {c:+.6f}")
    print(f"  Fit residual         = {fit['fit_residual']:.2e}")
    print(f"  Polyakov ⟨P⟩         = {poly:+.6f}")
    print(f"  Polyakov suscept.    = {summary['polyakov_susceptibility']:.6f}")
    print(f"  Final Wilson action  = {action:.4f}")
    print()
    print("  Wilson loop averages:")
    for k, v in sorted(summary["loop_averages"].items()):
        print(f"    {k:10s} = {v:.6f}")
    print()
    print("  Creutz ratios χ(R,T):")
    cr = fit.get("creutz_ratios", {})
    if cr:
        for k, v in sorted(cr.items()):
            tag = "  ← σ estimate" if not ("nan" in str(v)) else ""
            print(f"    {k:12s} = {v!s:.8}{tag}")
    else:
        print("    (none — need R,T ≥ 2)")
    print()

    # Interpretation
    if sigma > 0.02:
        verdict = "CONFINED — positive string tension detected."
    elif sigma < -0.01:
        verdict = "ANTICONFINED / DECONFINED — negative or zero σ."
    else:
        verdict = "UNCLEAR — σ near zero; try larger lattice or more sweeps."
    print(f"  Verdict: {verdict}")
    print(f"{'─'*60}\n")


def run_single(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    loop_sizes = _build_loop_sizes(args.max_r, args.max_t)

    print(f"Thermalising ({args.n_therm} sweeps) …", flush=True)
    t0 = time.perf_counter()
    summary, links = thermalize_and_measure_pure_gauge(
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
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f} s.")
    _print_findings(summary)
    return summary


def run_scan(args: argparse.Namespace) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    betas = np.linspace(args.beta_min, args.beta_max, args.beta_steps).tolist()
    loop_sizes = _build_loop_sizes(args.max_r, args.max_t)

    print(f"Deconfinement scan: β_g ∈ [{args.beta_min:.2f}, {args.beta_max:.2f}], "
          f"{args.beta_steps} points …", flush=True)
    t0 = time.perf_counter()
    results = deconfinement_scan(
        size=args.size,
        n=args.n,
        beta_values=betas,
        rng=rng,
        ndim=args.ndim,
        n_therm=args.n_therm,
        n_meas=args.n_meas,
        n_skip=args.n_skip,
        updater=args.updater,
        loop_sizes=loop_sizes,
    )
    elapsed = time.perf_counter() - t0
    print(f"Scan done in {elapsed:.1f} s.")
    print()
    print(f"{'β_g':>8}  {'σ':>10}  {'⟨P⟩':>10}  {'Action':>12}")
    print("-" * 46)
    for r in results:
        sigma = r["area_law_fit"]["sigma"]
        poly = r["polyakov_mean"]
        action = r["final_wilson_action"]
        print(f"{r['beta_g']:8.3f}  {sigma:+10.5f}  {poly:+10.5f}  {action:12.4f}")
    print()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure-gauge Monte Carlo confinement experiment for Project Genesis."
    )
    parser.add_argument("--size", type=int, default=6,
                        help="Lattice side length (default: 6)")
    parser.add_argument("--n", type=int, default=2,
                        help="SU(N) gauge group (default: 2)")
    parser.add_argument("--beta", type=float, default=1.8,
                        help="Inverse coupling β_g (default: 1.8)")
    parser.add_argument("--ndim", type=int, default=2,
                        help="Spatial dimensions (default: 2)")
    parser.add_argument("--updater", choices=["metropolis", "heatbath", "overrelax"],
                        default="metropolis",
                        help="MC update algorithm (default: metropolis)")
    parser.add_argument("--n-therm", type=int, default=150,
                        help="Thermalisation sweeps (default: 150)")
    parser.add_argument("--n-meas", type=int, default=30,
                        help="Measurement sweeps (default: 30)")
    parser.add_argument("--n-skip", type=int, default=5,
                        help="Sweeps between measurements (default: 5)")
    parser.add_argument("--step-scale", type=float, default=0.18,
                        help="Metropolis step scale (default: 0.18)")
    parser.add_argument("--max-r", type=int, default=3,
                        help="Maximum spatial loop extent R (default: 3)")
    parser.add_argument("--max-t", type=int, default=3,
                        help="Maximum temporal loop extent T (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to write JSON results (optional)")
    # Deconfinement scan options
    parser.add_argument("--beta-scan", action="store_true",
                        help="Run a β-scan instead of a single-β measurement")
    parser.add_argument("--beta-min", type=float, default=0.5,
                        help="Minimum β for scan (default: 0.5)")
    parser.add_argument("--beta-max", type=float, default=3.5,
                        help="Maximum β for scan (default: 3.5)")
    parser.add_argument("--beta-steps", type=int, default=7,
                        help="Number of β values in scan (default: 7)")

    args = parser.parse_args()

    if args.beta_scan:
        results = run_scan(args)
        output_name = "scan_results.json"
        payload = results
    else:
        results = run_single(args)
        output_name = "single_results.json"
        payload = results

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / output_name
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results written to {path}")


if __name__ == "__main__":
    main()
