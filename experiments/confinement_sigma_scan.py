"""σ(β_g) confinement scan — the thermodynamic area-law measurement.

This is the Monte Carlo study the README's Yang–Mills section lists as the
missing piece: the deterministic S-ascent flow finds stationary points of
the action but cannot see *ensemble* signatures; confinement (the area law,
a positive string tension σ) is a statement about the Boltzmann ensemble
exp(−β_g·S_W).  With the corrected heat-bath sampler and the Numba kernels,
proper ensembles are cheap enough to measure σ(β_g) honestly, with errors.

What is measured
----------------
For each β_g in the scan, for SU(2) (2-D) and SU(3) (3-D):

- Wilson loops W(R,T) for R,T ≤ R_max, ensemble-averaged with jackknife
  errors over measurement bins.
- Creutz ratios χ(R,R) = −log[W(R,R)·W(R−1,R−1)/W(R,R−1)²-style], the
  discretisation-robust string-tension estimator, with jackknife errors.
- The Polyakov loop magnitude |⟨P⟩| (confinement order parameter).

For SU(2) in 2-D the lattice theory is exactly solvable — plaquettes
decouple and

    σ_exact(β_g) = −ln[ I₂(2β_g) / I₁(2β_g) ]

so the SU(2) scan is a *calibration*: the measured χ(2,2) must track the
exact curve or the instrument is broken.  The SU(3) 3-D scan is then the
physics claim: σ > 0 across the coupling range (2+1-D Yang–Mills confines
at all couplings), decreasing as β_g grows.

Update schedule: 1 heat-bath + ``n_or`` overrelaxation sweeps per compound
sweep (overrelaxation decorrelates; heat-bath keeps it ergodic).

Usage::

    python experiments/confinement_sigma_scan.py --output-dir artifacts/sigma_scan
    python experiments/confinement_sigma_scan.py --group su2 --quick   # smoke run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.gauge import random_unitary  # noqa: E402
from project_genesis.gauge_mc import (  # noqa: E402
    heatbath_sweep,
    overrelaxation_sweep,
    wilson_loop,
    polyakov_loop,
    creutz_ratio,
)

try:
    from scipy.special import i1, iv
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False


def sigma_exact_su2_2d(beta_g: float) -> float:
    """Exact 2-D SU(2) string tension for the e^{β_g·Re Tr P} weight."""
    w = float(iv(2, 2.0 * beta_g) / i1(2.0 * beta_g))
    return -math.log(w)


def jackknife(samples: np.ndarray, estimator, bin_size: int = 1) -> tuple[float, float]:
    """Binned-jackknife mean and error of ``estimator`` over axis-0 samples.

    ``estimator`` maps the sample mean (an array over observables) to a
    scalar; jackknife handles the nonlinearity of e.g. Creutz ratios.
    ``bin_size > 1`` averages consecutive measurements into bins first, the
    standard guard against residual autocorrelation biasing the error low.
    """
    if bin_size > 1 and samples.shape[0] >= 2 * bin_size:
        nbins = samples.shape[0] // bin_size
        samples = samples[: nbins * bin_size].reshape(
            nbins, bin_size, *samples.shape[1:]
        ).mean(axis=1)
    n = samples.shape[0]
    full = estimator(samples.mean(axis=0))
    if n < 2:
        return full, float("nan")
    partial = np.empty(n)
    total = samples.sum(axis=0)
    for k in range(n):
        partial[k] = estimator((total - samples[k]) / (n - 1))
    err = math.sqrt((n - 1) / n * np.nansum((partial - full) ** 2))
    return full, err


def run_point(
    *,
    n: int,
    ndim: int,
    size: int,
    beta_g: float,
    r_max: int,
    n_therm: int,
    n_meas: int,
    n_skip: int,
    n_or: int,
    seed: int,
    bin_size: int = 10,
) -> dict:
    """Thermalise and measure one β_g point; returns a result dict."""
    rng = np.random.default_rng(seed)
    spatial = (size,) * ndim
    links = random_unitary(rng, n, (ndim, *spatial), special=(n > 1), scale=0.7)

    def compound_sweep(lk):
        lk, _ = heatbath_sweep(lk, beta_g, rng)
        if n_or:
            lk, _ = overrelaxation_sweep(lk, rng, n_sweeps=n_or)
        return lk

    for _ in range(n_therm):
        links = compound_sweep(links)

    # per-measurement samples: all W(R,T) with R,T ≤ r_max, plus |P|
    extents = [(r, t) for r in range(1, r_max + 1) for t in range(1, r_max + 1)]
    w_samples = np.empty((n_meas, len(extents)))
    p_samples = np.empty(n_meas)
    for m in range(n_meas):
        for _ in range(n_skip):
            links = compound_sweep(links)
        for e, (r, t) in enumerate(extents):
            w_samples[m, e] = wilson_loop(links, (r, t))
        p_samples[m] = polyakov_loop(links)

    def w_index(r, t):
        return extents.index((r, t))

    def chi_estimator(r):
        def est(w_mean):
            return creutz_ratio(
                w_mean[w_index(r, r)],
                w_mean[w_index(r - 1, r - 1)],
                w_mean[w_index(r, r - 1)],
                w_mean[w_index(r - 1, r)],
            )
        return est

    w_mean = w_samples.mean(axis=0)
    w_err = w_samples.std(axis=0, ddof=1) / math.sqrt(n_meas)

    creutz = {}
    for r in range(2, r_max + 1):
        chi, chi_err = jackknife(w_samples, chi_estimator(r), bin_size=bin_size)
        creutz[f"chi_{r}_{r}"] = {"value": chi, "error": chi_err}

    return {
        "n": n,
        "ndim": ndim,
        "size": size,
        "beta_g": beta_g,
        "n_therm": n_therm,
        "n_meas": n_meas,
        "n_skip": n_skip,
        "n_or": n_or,
        "seed": seed,
        "bin_size": bin_size,
        "loops": {
            f"W_{r}_{t}": {"value": float(w_mean[e]), "error": float(w_err[e])}
            for e, (r, t) in enumerate(extents)
        },
        "creutz": creutz,
        "polyakov_abs": float(abs(p_samples.mean())),
        "polyakov_err": float(p_samples.std(ddof=1) / math.sqrt(n_meas)),
    }


def run_scan(group: str, betas: list[float], args) -> list[dict]:
    """Run a full β_g scan for one gauge group."""
    if group == "su2":
        n, ndim, size = 2, 2, args.size_su2
    else:
        n, ndim, size = 3, 3, args.size_su3
    results = []
    for i, beta in enumerate(betas):
        print(f"[{group}] beta_g={beta:.3f} "
              f"({i + 1}/{len(betas)}, {size}^{ndim} lattice)...", flush=True)
        point = run_point(
            n=n, ndim=ndim, size=size, beta_g=beta, r_max=args.r_max,
            n_therm=args.n_therm, n_meas=args.n_meas, n_skip=args.n_skip,
            n_or=args.n_or, seed=args.seed + i, bin_size=args.bin_size,
        )
        if group == "su2" and HAVE_SCIPY:
            point["sigma_exact_2d"] = sigma_exact_su2_2d(beta)
        chi = point["creutz"].get(f"chi_2_2", {})
        print(f"    chi(2,2) = {chi.get('value', float('nan')):+.4f}"
              f" ± {chi.get('error', float('nan')):.4f}"
              + (f"   exact = {point['sigma_exact_2d']:.4f}"
                 if "sigma_exact_2d" in point else "")
              + f"   |<P>| = {point['polyakov_abs']:.4f}", flush=True)
        results.append(point)
    return results


def make_figure(all_results: dict[str, list[dict]], path: str) -> None:
    """σ(β_g) and |P|(β_g) panels.  Okabe–Ito colours (CVD-safe)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax in (ax1, ax2):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    if "su2" in all_results:
        res = all_results["su2"]
        betas = [r["beta_g"] for r in res]
        chi = [r["creutz"]["chi_2_2"]["value"] for r in res]
        err = [r["creutz"]["chi_2_2"]["error"] for r in res]
        ax1.errorbar(betas, chi, yerr=err, color=BLUE, marker="o", ms=5,
                     lw=1.6, capsize=3, label="SU(2) 2-D  χ(2,2)", zorder=3)
        if "sigma_exact_2d" in res[0]:
            bb = np.linspace(min(betas), max(betas), 200)
            ax1.plot(bb, [sigma_exact_su2_2d(b) for b in bb], color=BLUE,
                     ls="--", lw=1.2, label="SU(2) 2-D exact", zorder=2)
        ax2.errorbar(betas, [r["polyakov_abs"] for r in res],
                     yerr=[r["polyakov_err"] for r in res], color=BLUE,
                     marker="o", ms=5, lw=1.6, capsize=3,
                     label="SU(2) 2-D", zorder=3)

    if "su3" in all_results:
        res = all_results["su3"]
        betas = [r["beta_g"] for r in res]
        chi = [r["creutz"]["chi_2_2"]["value"] for r in res]
        err = [r["creutz"]["chi_2_2"]["error"] for r in res]
        ax1.errorbar(betas, chi, yerr=err, color=ORANGE, marker="s", ms=5,
                     lw=1.6, capsize=3, label="SU(3) 3-D  χ(2,2)", zorder=3)
        ax2.errorbar(betas, [r["polyakov_abs"] for r in res],
                     yerr=[r["polyakov_err"] for r in res], color=ORANGE,
                     marker="s", ms=5, lw=1.6, capsize=3,
                     label="SU(3) 3-D", zorder=3)

    ax1.axhline(0.0, color="#999999", lw=0.8)
    ax1.set_xlabel("β_g")
    ax1.set_ylabel("string tension σ  [lattice units]")
    ax1.set_title("Area law: Creutz ratio χ(2,2) vs coupling")
    ax1.legend(frameon=False)
    ax2.set_xlabel("β_g")
    ax2.set_ylabel("|⟨P⟩|")
    ax2.set_title("Polyakov loop (confinement order parameter)")
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(all_results: dict[str, list[dict]], path: str) -> None:
    lines = ["sigma(beta_g) confinement scan — summary", "=" * 44, ""]
    for group, res in all_results.items():
        lines.append(f"[{group}]  lattice {res[0]['size']}^{res[0]['ndim']}"
                     f"  n_meas={res[0]['n_meas']}  n_or={res[0]['n_or']}")
        for r in res:
            chi = r["creutz"]["chi_2_2"]
            line = (f"  beta_g={r['beta_g']:6.3f}  chi22={chi['value']:+8.4f}"
                    f" +- {chi['error']:.4f}")
            if "sigma_exact_2d" in r:
                dev = ((chi["value"] - r["sigma_exact_2d"]) / chi["error"]
                       if chi["error"] else float("nan"))
                line += (f"  exact={r['sigma_exact_2d']:7.4f}"
                         f"  pull={dev:+.1f}σ")
            line += f"  |P|={r['polyakov_abs']:.4f}"
            lines.append(line)
        # verdict per group
        chis = [(r["creutz"]["chi_2_2"]["value"], r["creutz"]["chi_2_2"]["error"])
                for r in res]
        all_positive = all(v - 2 * e > 0 for v, e in chis)
        decreasing = all(chis[i][0] >= chis[i + 1][0] - 2 * (chis[i][1] + chis[i + 1][1])
                         for i in range(len(chis) - 1))
        lines.append(f"  verdict: sigma > 0 at every beta (2σ): {all_positive};"
                     f" sigma decreasing with beta (2σ): {decreasing}")
        if group == "su2" and "sigma_exact_2d" in res[0]:
            pulls = [abs(r["creutz"]["chi_2_2"]["value"] - r["sigma_exact_2d"])
                     / r["creutz"]["chi_2_2"]["error"] for r in res
                     if r["creutz"]["chi_2_2"]["error"]]
            lines.append(f"  calibration: max |pull| vs exact 2-D curve ="
                         f" {max(pulls):.1f}σ over {len(pulls)} points")
        lines.append("")
    text = "\n".join(lines)
    with open(path, "w") as fh:
        fh.write(text + "\n")
    print("\n" + text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--group", choices=["su2", "su3", "both"], default="both")
    p.add_argument("--betas-su2", type=str,
                   default="0.75,1.0,1.5,2.0,2.5,3.0,4.0")
    p.add_argument("--betas-su3", type=str,
                   default="1.0,1.5,2.0,2.5,3.0,4.0,5.0")
    p.add_argument("--size-su2", type=int, default=16)
    p.add_argument("--size-su3", type=int, default=8)
    p.add_argument("--r-max", type=int, default=3)
    p.add_argument("--n-therm", type=int, default=200)
    p.add_argument("--n-meas", type=int, default=400)
    p.add_argument("--n-skip", type=int, default=2)
    p.add_argument("--n-or", type=int, default=2)
    p.add_argument("--bin-size", type=int, default=10,
                   help="measurements per jackknife bin (autocorrelation guard)")
    p.add_argument("--seed", type=int, default=20260703)
    p.add_argument("--output-dir", type=str, default="artifacts/sigma_scan")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke-test run (small lattices, few sweeps)")
    args = p.parse_args()

    if args.quick:
        args.size_su2, args.size_su3 = 8, 4
        args.n_therm, args.n_meas = 30, 40
        args.betas_su2 = "1.0,2.5"
        args.betas_su3 = "1.5,3.0"

    os.makedirs(args.output_dir, exist_ok=True)
    groups = ["su2", "su3"] if args.group == "both" else [args.group]
    all_results: dict[str, list[dict]] = {}
    for group in groups:
        betas = [float(b) for b in
                 (args.betas_su2 if group == "su2" else args.betas_su3).split(",")]
        all_results[group] = run_scan(group, betas, args)

    with open(os.path.join(args.output_dir, "sigma_scan.json"), "w") as fh:
        json.dump(all_results, fh, indent=2)
    write_summary(all_results, os.path.join(args.output_dir, "summary.txt"))
    if not args.no_figure:
        make_figure(all_results, os.path.join(args.output_dir, "sigma_scan.png"))
        print(f"figure: {os.path.join(args.output_dir, 'sigma_scan.png')}")


if __name__ == "__main__":
    main()
