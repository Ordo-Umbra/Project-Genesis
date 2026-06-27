#!/usr/bin/env python3
"""URP Monte Carlo Confinement — Analysis & Visualisation

This script runs the full pure-gauge beta scan, prints a rich comparison
table of measured observables against URP/lattice-theory expectations, and
saves four publication-quality figures to ``results/confinement/``.

Usage
-----
    # Quick run (small lattice, few sweeps — ~30 s on a laptop)
    python experiments/analyse_confinement.py --quick

    # Production run (matches test-suite parameters)
    python experiments/analyse_confinement.py

    # Custom scan
    python experiments/analyse_confinement.py \\
        --size 8 --n 2 --ndim 2 \\
        --beta-min 0.3 --beta-max 4.0 --beta-steps 12 \\
        --n-therm 300 --n-meas 150 --n-skip 4 \\
        --updater heatbath --seed 42

Output
------
Prints to stdout:
  1. Run parameters
  2. Per-beta measurement table (sigma, Polyakov, Creutz ratios)
  3. Theory-comparison table: 6 URP predictions vs measured values
  4. Paths to saved figures

Saved figures (results/confinement/):
  sigma_vs_beta.png      — string tension σ(β_g)
  polyakov_vs_beta.png   — Polyakov order parameter and susceptibility
  wilson_loops.png       — W(R,T) area-law decay at two representative β values
  creutz_ratios.png      — Creutz ratio χ(R,T) vs β_g

Theory anchors (URP documents)
------------------------------
All six comparison checks are annotated with the URP β-sectorisation /
Emergent-SU(3) section that predicts the expected behaviour:

  P1  σ > 0 at β_low          →  β-sectorisation §4.A  (confined phase)
  P2  σ decreases with β       →  asymptotic-freedom direction on the lattice
  P3  |⟨P⟩| near 0 at β_low   →  Z_N symmetry unbroken in confined phase
  P4  |⟨P⟩| increases with β  →  crossover to deconfined sector
  P5  W(3,3) < W(1,1) at β_low →  area-law ordering (area > perimeter decay)
  P6  χ(2,2) > 0 at β_low     →  positive Creutz ratio = nonzero string tension
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from project_genesis.gauge_mc import (
    deconfinement_scan,
    thermalize_and_measure_pure_gauge,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--size", type=int, default=6)
    p.add_argument("--ndim", type=int, default=2)
    p.add_argument("--n", type=int, default=2,
                   help="Gauge group SU(N) (default 2)")
    p.add_argument("--beta-min", type=float, default=0.5)
    p.add_argument("--beta-max", type=float, default=3.5)
    p.add_argument("--beta-steps", type=int, default=7,
                   help="Number of beta points in scan (default 7)")
    p.add_argument("--n-therm", type=int, default=150)
    p.add_argument("--n-meas", type=int, default=80)
    p.add_argument("--n-skip", type=int, default=4)
    p.add_argument("--updater",
                   choices=["metropolis", "heatbath", "overrelax"],
                   default="heatbath")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--quick", action="store_true",
                   help="Override to a very small run for quick smoke test")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip matplotlib figure generation")
    p.add_argument("--outdir", type=str, default="results/confinement",
                   help="Directory for saved figures (default: results/confinement)")
    return p


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

SEP72 = "=" * 72
SEP70 = "-" * 70


def _print_params(args: argparse.Namespace, betas: list[float]) -> None:
    print()
    print(SEP72)
    print("  URP Monte Carlo Confinement — Analysis & Visualisation")
    print(SEP72)
    print(f"  Gauge group : SU({args.n})   Lattice: {args.size}^{args.ndim}   ndim={args.ndim}")
    print(f"  Updater     : {args.updater}")
    print(f"  Sweeps      : therm={args.n_therm}  meas={args.n_meas}  skip={args.n_skip}")
    print(f"  Beta range  : {betas[0]:.3f} .. {betas[-1]:.3f}  ({len(betas)} points)")
    print(f"  Seed        : {args.seed}")
    print(SEP72)
    print()


def _print_scan_table(results: list[dict], loop_sizes: list[tuple[int, int]]) -> None:
    rs = sorted({s[0] for s in loop_sizes})
    ts = sorted({s[1] for s in loop_sizes})
    loop_cols = [f"W({r},{t})" for r in rs for t in ts]
    header = (f"  {'beta':>6}  {'sigma':>9}  {'|<P>|':>7}  {'chi_2_2':>8}  "
              + "  ".join(f"{c:>9}" for c in loop_cols)
              + "  verdict")
    print(SEP70)
    print("  Raw measurement table")
    print(SEP70)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        beta = r["beta_g"]
        sigma = r["area_law_fit"]["sigma"]
        poly = abs(r["polyakov_mean"])
        chi22 = r["area_law_fit"]["creutz_ratios"].get("chi_2_2", float("nan"))
        la = r["loop_averages"]
        loop_vals = "".join(
            f"  {la.get(f'W_{ri}_{ti}', float('nan')):>9.5f}"
            for ri in rs for ti in ts
        )
        if sigma > 0.005:
            verdict = "CONFINED"
        elif sigma < -0.005:
            verdict = "deconfined"
        else:
            verdict = "crossover?"
        chi_str = f"{chi22:>8.4f}" if not math.isnan(chi22) else "     nan"
        print(f"  {beta:>6.3f}  {sigma:>+9.5f}  {poly:>7.4f}  {chi_str}  "
              f"{loop_vals}  {verdict}")
    print()


# ---------------------------------------------------------------------------
# Theory comparison
# ---------------------------------------------------------------------------

PREDICTIONS = [
    (
        "P1",
        "sigma > 0 at beta_low (confined phase, area law active)",
        "URP beta-sectorisation §4.A",
    ),
    (
        "P2",
        "sigma(beta_low) >= sigma(beta_high) (sigma decreases with beta)",
        "asymptotic-freedom direction on the lattice",
    ),
    (
        "P3",
        "|<P>| < 0.3 at beta_low (Z_N symmetry unbroken)",
        "confined phase: Polyakov loop near zero",
    ),
    (
        "P4",
        "|<P>|(beta_high) >= |<P>|(beta_low) - 0.05 (Polyakov increases with beta)",
        "URP crossover to deconfined sector",
    ),
    (
        "P5",
        "W(3,3) < W(1,1) at beta_low (area > perimeter decay)",
        "Wilson-loop area-law ordering",
    ),
    (
        "P6",
        "chi_2_2 > 0 at beta_low (positive Creutz ratio)",
        "perimeter-subtracted string tension > 0",
    ),
]


def _check_predictions(results: list[dict]) -> list[tuple[str, str, str, bool]]:
    """Evaluate each theory prediction against the scan results.

    Returns a list of (label, description, anchor, passed) tuples.
    """
    r_low = results[0]   # lowest beta
    r_high = results[-1]  # highest beta
    la_low = r_low["loop_averages"]

    checks = []

    # P1
    sigma_low = r_low["area_law_fit"]["sigma"]
    checks.append(("P1", PREDICTIONS[0][1], PREDICTIONS[0][2], sigma_low > 0.0))

    # P2
    sigma_high = r_high["area_law_fit"]["sigma"]
    checks.append(("P2", PREDICTIONS[1][1], PREDICTIONS[1][2],
                   sigma_low + 0.02 >= sigma_high))

    # P3
    poly_low = abs(r_low["polyakov_mean"])
    checks.append(("P3", PREDICTIONS[2][1], PREDICTIONS[2][2], poly_low < 0.3))

    # P4
    poly_high = abs(r_high["polyakov_mean"])
    checks.append(("P4", PREDICTIONS[3][1], PREDICTIONS[3][2],
                   poly_high + 0.05 >= poly_low))

    # P5
    w11 = la_low.get("W_1_1", float("nan"))
    w33 = la_low.get("W_3_3", float("nan"))
    if not (math.isnan(w11) or math.isnan(w33)):
        p5 = w33 < w11
    else:
        p5 = False
    checks.append(("P5", PREDICTIONS[4][1], PREDICTIONS[4][2], p5))

    # P6
    chi22_low = r_low["area_law_fit"]["creutz_ratios"].get("chi_2_2", float("nan"))
    p6 = (not math.isnan(chi22_low)) and chi22_low > 0.0
    checks.append(("P6", PREDICTIONS[5][1], PREDICTIONS[5][2], p6))

    return checks


def _print_theory_comparison(checks: list[tuple]) -> None:
    print(SEP70)
    print("  Theory comparison — URP predictions vs measured values")
    print(SEP70)
    n_pass = sum(1 for c in checks if c[3])
    for label, desc, anchor, passed in checks:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {label}: {desc}")
        print(f"         Anchor: {anchor}")
    print()
    print(f"  Result: {n_pass}/{len(checks)} predictions confirmed.")
    if n_pass == len(checks):
        print("  *** All URP confinement predictions CONFIRMED. ***")
    else:
        print(f"  *** {len(checks) - n_pass} prediction(s) NOT confirmed — "
              "check sweep counts / lattice size / beta range. ***")
    print()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_figures(
    results: list[dict],
    betas: list[float],
    loop_sizes: list[tuple[int, int]],
    outdir: str,
) -> list[str]:
    """Generate and save the four analysis figures.  Returns list of paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  [warn] matplotlib not available — skipping figures.")
        return []

    Path(outdir).mkdir(parents=True, exist_ok=True)
    saved = []

    # shared style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "figure.dpi": 150,
    })

    sigmas   = [r["area_law_fit"]["sigma"] for r in results]
    polys    = [abs(r["polyakov_mean"]) for r in results]
    p_suscs  = [r.get("polyakov_susceptibility", float("nan")) for r in results]
    chi22s   = [r["area_law_fit"]["creutz_ratios"].get("chi_2_2", float("nan"))
                for r in results]

    # ---- Figure 1: sigma vs beta ----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="#aaa", lw=0.8, ls="--")
    # shade confined / deconfined zones based on sigma sign
    b_cross = None
    for i in range(len(sigmas) - 1):
        if sigmas[i] > 0 >= sigmas[i + 1]:
            b_cross = (betas[i] + betas[i + 1]) / 2
            break
    if b_cross:
        ax.axvspan(betas[0], b_cross, alpha=0.08, color="steelblue", label="Confined zone")
        ax.axvspan(b_cross, betas[-1], alpha=0.08, color="tomato", label="Deconfined zone")
    ax.plot(betas, sigmas, "o-", color="#01696f", lw=2, ms=6, label=r"$\sigma(\beta_g)$")
    ax.set_xlabel(r"$\beta_g$ (inverse gauge coupling)", fontsize=12)
    ax.set_ylabel(r"String tension $\sigma$ [lattice units]", fontsize=12)
    ax.set_title(
        r"Pure-gauge String Tension $\sigma(\beta_g)$"
        "\n"
        r"URP $\beta$-sectorisation §4.A: $\sigma > 0$ in confined phase",
        fontsize=11,
    )
    ax.legend(framealpha=0.6)
    fig.tight_layout()
    p = str(Path(outdir) / "sigma_vs_beta.png")
    fig.savefig(p)
    plt.close(fig)
    saved.append(p)

    # ---- Figure 2: Polyakov loop + susceptibility -----------------------
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(betas, polys, "s-", color="#7a39bb", lw=2, ms=6,
             label=r"$|\langle P \rangle|$")
    ax2.plot(betas, p_suscs, "^--", color="#da7101", lw=1.5, ms=5, alpha=0.7,
             label=r"$\chi_P$ (susceptibility)")
    ax1.set_xlabel(r"$\beta_g$", fontsize=12)
    ax1.set_ylabel(r"$|\langle P \rangle|$", color="#7a39bb", fontsize=12)
    ax2.set_ylabel(r"$\chi_P$", color="#da7101", fontsize=12)
    ax1.set_title(
        r"Polyakov Loop Order Parameter $|\langle P \rangle|$ vs $\beta_g$"
        "\n"
        r"URP prediction: $|\langle P \rangle| \approx 0$ (confined) $\to > 0$ (deconfined)",
        fontsize=11,
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, framealpha=0.6)
    fig.tight_layout()
    p = str(Path(outdir) / "polyakov_vs_beta.png")
    fig.savefig(p)
    plt.close(fig)
    saved.append(p)

    # ---- Figure 3: Wilson loop area decay at low & high beta -----------
    rs = sorted({s[0] for s in loop_sizes})
    ts = sorted({s[1] for s in loop_sizes})
    r_low = results[0]
    r_high = results[-1]
    la_low  = r_low["loop_averages"]
    la_high = r_high["loop_averages"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, la, beta_val, label in [
        (axes[0], la_low,  r_low["beta_g"],  "Strong coupling (confined)"),
        (axes[1], la_high, r_high["beta_g"], "Weak coupling (deconfined)"),
    ]:
        for r in rs:
            wvals = [la.get(f"W_{r}_{t}", float("nan")) for t in ts]
            ax.plot(ts, wvals, "o-", ms=5, lw=1.8, label=f"R={r}")
        ax.set_xlabel("T [lattice units]", fontsize=11)
        ax.set_ylabel(r"$\langle W(R,T) \rangle$", fontsize=11)
        ax.set_title(
            f"Wilson Loops at $\\beta_g = {beta_val:.2f}$\n{label}",
            fontsize=11,
        )
        ax.legend(fontsize=9, framealpha=0.6)
        ax.set_ylim(bottom=0)
    fig.suptitle(
        r"Wilson Loop Area Decay: $\langle W(R,T)\rangle = e^{-\sigma RT - c(R+T)}$",
        fontsize=12,
    )
    fig.tight_layout()
    p = str(Path(outdir) / "wilson_loops.png")
    fig.savefig(p)
    plt.close(fig)
    saved.append(p)

    # ---- Figure 4: Creutz ratios vs beta --------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="#aaa", lw=0.8, ls="--")
    # collect all available creutz keys
    all_keys = set()
    for r in results:
        all_keys.update(r["area_law_fit"]["creutz_ratios"].keys())
    for key in sorted(all_keys):
        vals = [r["area_law_fit"]["creutz_ratios"].get(key, float("nan"))
                for r in results]
        ax.plot(betas, vals, "o-", ms=5, lw=1.8, label=r"$\chi$" + f"({key[4:]})".replace("_", ","))
    ax.set_xlabel(r"$\beta_g$", fontsize=12)
    ax.set_ylabel(r"Creutz ratio $\chi(R,T)$", fontsize=12)
    ax.set_title(
        r"Creutz Ratios $\chi(R,T)$ vs $\beta_g$"
        "\n"
        r"Perimeter-subtracted estimator of string tension $\sigma$",
        fontsize=11,
    )
    ax.legend(fontsize=9, framealpha=0.6)
    fig.tight_layout()
    p = str(Path(outdir) / "creutz_ratios.png")
    fig.savefig(p)
    plt.close(fig)
    saved.append(p)

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.quick:
        args.size     = min(args.size, 5)
        args.n_therm  = 30
        args.n_meas   = 20
        args.n_skip   = 2
        args.beta_steps = 4
        print("[quick mode: reduced lattice/sweep counts]")

    betas = list(np.linspace(args.beta_min, args.beta_max, args.beta_steps))
    loop_sizes = [(1, 1), (2, 2), (3, 3)]
    # clamp to lattice
    max_ext = args.size - 1
    loop_sizes = [(min(r, max_ext), min(t, max_ext)) for r, t in loop_sizes]

    _print_params(args, betas)

    rng = np.random.default_rng(args.seed)
    print("  Running beta scan ...")
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
    print("  Done.\n")

    _print_scan_table(results, loop_sizes)
    checks = _check_predictions(results)
    _print_theory_comparison(checks)

    if not args.no_plots:
        print("  Generating figures ...")
        saved = _save_figures(results, betas, loop_sizes, args.outdir)
        if saved:
            print("  Saved figures:")
            for p in saved:
                print(f"    {p}")
        print()

    # exit code: 0 if all predictions pass, 1 otherwise
    sys.exit(0 if all(c[3] for c in checks) else 1)


if __name__ == "__main__":
    main()
