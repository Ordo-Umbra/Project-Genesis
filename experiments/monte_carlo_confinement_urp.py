"""URP-aligned Monte Carlo confinement experiments.

This script runs `thermalize_and_measure_pure_gauge` and `deconfinement_scan`
from `project_genesis.gauge_mc` to explore confinement signatures in a way
that mirrors the qualitative picture in the β-sectorisation / emergent SU(3)
paper:

- Confinement: nonzero string tension σ (area-law suppression of Wilson loops)
  and Polyakov loop ⟨P⟩ ≈ 0.
- Deconfinement: σ → 0 and ⟨P⟩ > 0 as boundaries "melt" at higher effective
  coupling.

The numerical values here are not meant to reproduce continuum QCD; they are
lattice-level diagnostics that the Monte Carlo layer behaves in the expected
qualitative regimes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from project_genesis.gauge_mc import (
    thermalize_and_measure_pure_gauge,
    deconfinement_scan,
)


@dataclass
class ConfinementSummary:
    beta_g: float
    ndim: int
    size: int
    n: int
    updater: str
    loop_averages: dict
    polyakov_mean: float
    polyakov_susceptibility: float
    sigma: float
    perimeter_coeff: float


def run_single_point(
    beta_g: float,
    size: int = 8,
    n: int = 2,
    ndim: int = 2,
    updater: str = "metropolis",
    n_therm: int = 200,
    n_meas: int = 40,
    n_skip: int = 5,
    seed: int = 42,
) -> ConfinementSummary:
    """Run a single (β_g, lattice) point and extract Wilson-loop + Polyakov data.

    Parameters are chosen to give reasonably decorrelated measurements on
    modest lattices; they can be tuned further once we inspect the outputs.
    """
    rng = np.random.default_rng(seed)

    summary, _ = thermalize_and_measure_pure_gauge(
        size=size,
        n=n,
        beta_g=beta_g,
        rng=rng,
        ndim=ndim,
        n_therm=n_therm,
        n_meas=n_meas,
        n_skip=n_skip,
        updater=updater,
        loop_sizes=[(1, 1), (2, 2), (3, 3)],
    )

    area = summary["area_law_fit"]
    return ConfinementSummary(
        beta_g=summary["beta_g"],
        ndim=summary["ndim"],
        size=summary["size"],
        n=summary["n"],
        updater=summary["updater"],
        loop_averages=summary["loop_averages"],
        polyakov_mean=summary["polyakov_mean"],
        polyakov_susceptibility=summary["polyakov_susceptibility"],
        sigma=area["sigma"],
        perimeter_coeff=area["perimeter_coeff"],
    )


def run_scan(
    beta_values: Sequence[float],
    size: int = 8,
    n: int = 2,
    ndim: int = 3,
    updater: str = "heatbath",
    n_therm: int = 200,
    n_meas: int = 40,
    n_skip: int = 5,
    seed: int = 123,
) -> list[dict]:
    """Run a β_g scan using `deconfinement_scan` and return the raw summaries.

    This is a convenience wrapper that keeps parameter choices together and
    provides a JSON-serialisable list of dicts for downstream analysis.
    """
    rng = np.random.default_rng(seed)
    results = deconfinement_scan(
        size=size,
        n=n,
        beta_values=beta_values,
        rng=rng,
        ndim=ndim,
        n_therm=n_therm,
        n_meas=n_meas,
        n_skip=n_skip,
        updater=updater,
        loop_sizes=[(1, 1), (2, 2), (3, 3)],
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="URP-aligned Monte Carlo confinement experiments",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Single β_g value to probe (if omitted, runs a scan)",
    )
    parser.add_argument(
        "--scan",
        nargs="*",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0],
        help="β_g values for a deconfinement scan",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=8,
        help="Lattice size (per dimension)",
    )
    parser.add_argument(
        "--ndim",
        type=int,
        default=2,
        help="Number of spatial dimensions for single-point run",
    )
    parser.add_argument(
        "--updater",
        type=str,
        default="metropolis",
        choices=["metropolis", "heatbath", "overrelax"],
        help="Update algorithm for single-point run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON results",
    )

    args = parser.parse_args()

    if args.beta is not None:
        conf = run_single_point(
            beta_g=args.beta,
            size=args.size,
            ndim=args.ndim,
            updater=args.updater,
        )
        payload = asdict(conf)
    else:
        payload = run_scan(args.scan, size=args.size)

    text = json.dumps(payload, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":  # pragma: no cover
    main()
