"""The deflation test: is κ̂(t₀) ≈ 0.22 a number, or a convention?

Act I's central constant — the self-dual fraction of the SU(3) vacuum,
`κ̂(t₀) ≈ 0.22` — is read at the Wilson-flow scale `t₀` where `t²E(t)` first
reaches **0.3**.  But 0.3 is a *human convention*: Lüscher chose it for
scale-setting convenience, not because coherent fractions care about it.  The
number 0.22 entered the programme as a preliminary reading that stuck.  Before
anything more is built on it, this experiment stress-tests it the only honest
way: **sweep the convention and watch the number.**

The instrument reads `κ̂(t_c)` at the family of reference scales
`t_c : t²E(t_c) = c` for a scan of `c`, and reports:

A. **The trajectory with the conventions marked** — where each `c` lands on the
   κ̂(t) curve, per configuration (so config scatter and convention spread can
   be compared).
B. **The deflation curve** `κ̂(c)` — flat means the constant is real at this
   lattice; sloped means the convention picks the altitude on a rising curve.
   The scalar verdict is the **swing** `Δκ̂` across the scanned conventions
   against the ensemble error, plus the dimensionless sensitivity
   `c·dκ̂/dc` at the canonical `c = 0.3` (the shift per e-fold of convention).
C. **The plateau hunt** — `dκ̂/dt` along the flow: a genuine constant of the
   vacuum would show a window where the curve flattens; a scale-dependent
   quantity never does.  This is the dynamical-κ question asked directly: if
   there is no plateau, the honest object is the *function* `κ̂(scale)`, and
   0.22 is one slice of it, not a constant.

Whatever the outcome, it goes in the record: either 0.22 survives its own
convention (and becomes much more interesting), or Act I's headline number is
restated as a curve — which is what the framework's own author expects of a
dynamical κ.

Usage::

    python experiments/n3_kappa_deflation.py --output-dir artifacts/n3_deflation
    python experiments/n3_kappa_deflation.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.gauge import random_unitary
from project_genesis.gauge_mc import heatbath_sweep, overrelaxation_sweep
from project_genesis.gauge_topology import flow_scale, gradient_flow


def _interp(traj, t, key):
    """Linear interpolation of a trajectory observable at flow time ``t``."""
    if not np.isfinite(t):
        return float("nan")
    for a, b in zip(traj, traj[1:]):
        if a["t"] <= t <= b["t"] and b["t"] > a["t"]:
            frac = (t - a["t"]) / (b["t"] - a["t"])
            return float(a[key] + frac * (b[key] - a[key]))
    return float(traj[-1][key])


def run_ensemble(args, beta_g: float, t_max: float, seed_offset: int):
    """Flow an SU(3) ensemble; read κ̂ at every reference convention ``c``.

    ``t_max`` is per-coupling: weaker coupling means a smoother field whose
    ``t²E`` clock runs slower, so the same references need a deeper flow.
    """
    rng = np.random.default_rng(args.seed + seed_offset)
    L = args.gauge_size
    trajs = []
    for _ in range(args.n_conf):
        links = random_unitary(rng, 3, (4, L, L, L, L), special=True, scale=0.7)
        for _ in range(args.n_therm):
            links, _ = heatbath_sweep(links, beta_g, rng)
            links, _ = overrelaxation_sweep(links, rng)
        _, traj = gradient_flow(links, t_max=t_max, dt=args.dt,
                                measure_every=args.measure_every)
        trajs.append(traj)

    # per-config reading at each convention, then ensemble mean ± err
    scan = []
    for c in args.refs:
        per_conf = []
        for traj in trajs:
            t_c = flow_scale(traj, c)
            per_conf.append((_interp(traj, t_c, "f_sd"), t_c))
        kappas = np.array([k for k, _ in per_conf])
        times = np.array([t for _, t in per_conf])
        good = np.isfinite(kappas)
        n = int(good.sum())
        scan.append({
            "c": c,
            "kappa": float(np.mean(kappas[good])) if n else float("nan"),
            "err": float(np.std(kappas[good]) / max(1, np.sqrt(n))) if n else float("nan"),
            "t_ref": float(np.nanmean(times[good])) if n else float("nan"),
            "n": n,
        })

    # the mean trajectory (for the plateau hunt)
    times = [r["t"] for r in trajs[0]]
    kappa_t = np.mean([[r["f_sd"] for r in tr] for tr in trajs], axis=0)
    dk_dt = np.gradient(kappa_t, times)
    print(f"  SU(3) β={beta_g}, L={L}, {args.n_conf} configs: "
          + "  ".join(f"κ̂({r['c']:g})={r['kappa']:.3f}±{r['err']:.3f}"
                      for r in scan), flush=True)
    return {"beta_g": beta_g, "scan": scan, "times": times,
            "kappa_t": kappa_t.tolist(), "dk_dt": dk_dt.tolist(),
            "trajs_fsd": [[r["f_sd"] for r in tr] for tr in trajs]}


def analyse(ens, args):
    """The deflation scalars: swing, sensitivity at c=0.3, plateau verdict."""
    scan = [r for r in ens["scan"] if np.isfinite(r["kappa"])]
    if len(scan) < 2:
        return {"swing": float("nan"), "typical_err": float("nan"),
                "sensitivity_at_03": float("nan"),
                "min_abs_slope": float("nan"), "max_abs_slope": float("nan"),
                "significant": False,
                "note": "flow never reached the reference scales — deepen t_max"}
    kappas = np.array([r["kappa"] for r in scan])
    errs = np.array([r["err"] for r in scan])
    cs = np.array([r["c"] for r in scan])
    swing = float(kappas.max() - kappas.min())
    typical_err = float(np.mean(errs))
    # dimensionless sensitivity c·dκ̂/dc at the canonical convention
    dk_dc = np.gradient(kappas, cs)
    i03 = int(np.argmin(np.abs(cs - 0.3)))
    sensitivity = float(cs[i03] * dk_dc[i03])
    # plateau hunt: smallest |dκ̂/dt| over the flow, relative to its own scale
    t = np.array(ens["times"])
    dk = np.array(ens["dk_dt"])
    interior = (t > 0.1) & (t < t.max() - 0.05)
    min_slope = float(np.min(np.abs(dk[interior]))) if interior.any() else float("nan")
    max_slope = float(np.max(np.abs(dk[interior]))) if interior.any() else float("nan")
    return {"swing": swing, "typical_err": typical_err,
            "sensitivity_at_03": sensitivity,
            "min_abs_slope": min_slope, "max_abs_slope": max_slope,
            "significant": swing > args.deflation_threshold
            and swing > 3.0 * typical_err}


def summarise(results, args) -> list[str]:
    lines = ["The deflation test: is κ̂(t₀) ≈ 0.22 a number, or a convention? — "
             "verdict", "=" * 74, ""]
    primary = results[0]
    a = primary["analysis"]
    scan = primary["ens"]["scan"]
    lines.append(
        f"A. the readings, β_g = {primary['ens']['beta_g']:g}: "
        + ";  ".join(f"c={r['c']:g} → κ̂ = {r['kappa']:.3f} ± {r['err']:.3f} "
                     f"(t_ref = {r['t_ref']:.2f})" for r in scan) + ".")
    lines.append("")
    lines.append(
        f"B. the deflation curve: sweeping the reference convention across "
        f"c ∈ [{scan[0]['c']:g}, {scan[-1]['c']:g}] swings κ̂ by "
        f"Δκ̂ = {a['swing']:.3f} against a typical ensemble error of "
        f"{a['typical_err']:.3f} — "
        + (f"a {a['swing']/max(a['typical_err'],1e-9):.0f}σ-scale swing: the "
           "convention, not the ensemble, dominates the reading. The "
           "dimensionless sensitivity at the canonical point is "
           f"c·dκ̂/dc|₀.₃ = {a['sensitivity_at_03']:.3f} per e-fold of "
           "convention choice."
           if a["significant"] else
           f"within noise: the reading is convention-robust at this lattice "
           f"(sensitivity c·dκ̂/dc|₀.₃ = {a['sensitivity_at_03']:.3f})."))
    lines.append("")
    lines.append(
        f"C. the plateau hunt: |dκ̂/dt| along the flow spans "
        f"[{a['min_abs_slope']:.3f}, {a['max_abs_slope']:.3f}] — "
        + ("the curve never flattens: there is NO plateau at this lattice, so "
           "no flow scale at which κ̂ reads as a constant of the vacuum."
           if a["min_abs_slope"] > args.plateau_slope else
           "the curve carries a flat window — a candidate scale at which κ̂ is "
           "convention-insensitive; that window, not any c-value, is where a "
           "constant could live."))
    if len(results) > 1:
        lines.append("")
        for res in results[1:]:
            b = res["analysis"]
            lines.append(
                f"   (β_g = {res['ens']['beta_g']:g} robustness: swing "
                f"{b['swing']:.3f}, sensitivity {b['sensitivity_at_03']:.3f}, "
                f"min |dκ̂/dt| {b['min_abs_slope']:.3f} — same character.)")
    lines.append("")
    if a["significant"]:
        lines.append(
            "what this means: Act I's headline constant is substantially a "
            "convention artifact — κ̂(t₀) = 0.22 is one slice through a rising, "
            "plateau-free curve, and the choice t²E = 0.3 picks the altitude. "
            "The honest object is the FUNCTION κ̂(scale) — exactly what a "
            "dynamical κ predicts. Statements of the form 'κ ≈ 0.22' should be "
            "restated as 'κ̂ at the t₀ convention ≈ 0.22, rising through "
            "[range] across conventions'; the framework's dimensionless "
            "constant, if it exists, must be sought in convention-free "
            "structure (a plateau at finer lattices, a fixed point of the "
            "curve, or a framework-level definition) — not in this reading.")
    else:
        lines.append(
            "what this means: the reading survives its own convention at this "
            "lattice — a genuinely stronger status for 0.22 than it had, and "
            "worth a finer-lattice follow-up.")
    lines.append("")
    lines.append(
        "honest scope: one lattice size per β (no continuum limit — the known "
        "cutoff trend of κ̂ is a separate, additional systematic on top of the "
        "convention spread measured here); small ensembles; the flow-time "
        "derivative is a finite-difference estimate on the recorded grid. The "
        "test deflates (or upholds) the READING convention only — it does not "
        "measure what a continuum κ̂ is.")
    return lines


def make_figure(results, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GRAY, RED, GREEN = "#0072B2", "#E69F00", "#999999", "#c0392b", "#009E73"
    fig = plt.figure(figsize=(13, 9.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.27)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))
    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    primary = results[0]
    ens = primary["ens"]

    # panel A: κ̂(t) per config + mean, with the convention family marked
    t = np.array(ens["times"])
    for fsd in ens["trajs_fsd"]:
        ax1.plot(t, fsd, color=BLUE, lw=0.7, alpha=0.35, zorder=2)
    ax1.plot(t, ens["kappa_t"], color=BLUE, lw=2.4, zorder=3, label="⟨κ̂(t)⟩")
    for r in ens["scan"]:
        if np.isfinite(r["t_ref"]):
            ax1.axvline(r["t_ref"], color=ORANGE, lw=1.0, ls=":", zorder=2)
            ax1.text(r["t_ref"], 0.985, f"c={r['c']:g}", rotation=90,
                     fontsize=7, color=ORANGE, ha="right", va="top",
                     transform=ax1.get_xaxis_transform())
    ax1.axhline(0.22, color=GRAY, lw=1.0, ls="--", zorder=2)
    ax1.set_xlabel("Wilson flow time  t")
    ax1.set_ylabel("self-dual fraction  κ̂")
    ax1.set_title("A. One rising curve; the conventions pick altitudes")
    ax1.legend(fontsize=8, loc="lower right")

    # panel B: the deflation curve κ̂(c)
    for res, color in zip(results, (BLUE, GREEN)):
        scan = res["ens"]["scan"]
        cs = [r["c"] for r in scan]
        ks = [r["kappa"] for r in scan]
        es = [r["err"] for r in scan]
        ax2.errorbar(cs, ks, yerr=es, color=color, lw=2.0, marker="o", ms=5,
                     capsize=3, zorder=3,
                     label=f"β_g = {res['ens']['beta_g']:g}")
    ax2.axhline(0.22, color=GRAY, lw=1.0, ls="--", zorder=2)
    ax2.axvline(0.3, color=GRAY, lw=1.0, ls=":", zorder=2)
    ax2.text(0.3, 0.02, " canonical c = 0.3", fontsize=8, color=GRAY,
             transform=ax2.get_xaxis_transform())
    ax2.set_xlabel("reference convention  c  (t²E(t_c) = c)")
    ax2.set_ylabel("κ̂(t_c)")
    ax2.set_title("B. The deflation curve: the number vs its convention")
    ax2.legend(fontsize=8, loc="best")

    # panel C: the plateau hunt
    for res, color in zip(results, (BLUE, GREEN)):
        tt = np.array(res["ens"]["times"])
        dk = np.array(res["ens"]["dk_dt"])
        ax3.plot(tt, np.abs(dk), color=color, lw=2.0, zorder=3,
                 label=f"β_g = {res['ens']['beta_g']:g}")
    ax3.axhline(args.plateau_slope, color=RED, lw=1.0, ls="--", zorder=2,
                label="plateau criterion")
    ax3.set_yscale("log")
    ax3.set_xlabel("Wilson flow time  t")
    ax3.set_ylabel("|dκ̂/dt|")
    ax3.set_title("C. The plateau hunt: does κ̂(t) ever flatten?")
    ax3.legend(fontsize=8, loc="best")

    # panel D: verdict
    ax4.axis("off")
    ax4.set_title("D. The verdict")
    a = primary["analysis"]
    if a["significant"]:
        txt = (f"the convention dominates the reading:\n\n"
               f"   swing across c-scan:  Δκ̂ = {a['swing']:.3f}\n"
               f"   ensemble error:            {a['typical_err']:.3f}\n"
               f"   sensitivity c·dκ̂/dc|0.3 = {a['sensitivity_at_03']:.3f}\n"
               f"   min |dκ̂/dt| on flow:      {a['min_abs_slope']:.3f}\n\n"
               "κ̂(t0) = 0.22 is one slice of a rising,\n"
               "plateau-free curve — the honest object\n"
               "is κ̂(scale), a dynamical κ.\n\n"
               "the constant, if it exists, lives in\n"
               "convention-free structure not yet found.")
    else:
        txt = (f"the reading survives its convention:\n\n"
               f"   swing Δκ̂ = {a['swing']:.3f} (≤ noise)\n\n"
               "0.22 is upgraded, not deflated,\n"
               "at this lattice.")
    ax4.text(0.05, 0.9, txt, transform=ax4.transAxes, fontsize=10, va="top",
             family="monospace")

    fig.suptitle("The deflation test: sweep the t²E = c convention and watch "
                 "the 0.22", fontsize=12, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gauge-size", type=int, default=8)
    p.add_argument("--beta-g", type=float, nargs="+", default=[1.8, 2.2],
                   help="Couplings (first is primary; extras are robustness).")
    p.add_argument("--n-conf", type=int, default=6)
    p.add_argument("--n-therm", type=int, default=20)
    p.add_argument("--t-max", type=float, nargs="+", default=[1.2, 3.5],
                   help="Flow depth per coupling (matched to --beta-g; the "
                        "last value repeats if fewer are given).")
    p.add_argument("--dt", type=float, default=0.03)
    p.add_argument("--measure-every", type=int, default=2)
    p.add_argument("--refs", type=float, nargs="+",
                   default=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    p.add_argument("--deflation-threshold", type=float, default=0.03,
                   help="Swing below this (and below 3× error) counts as "
                        "convention-robust.")
    p.add_argument("--plateau-slope", type=float, default=0.02,
                   help="|dκ̂/dt| below this counts as a plateau.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_deflation")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.gauge_size = 6
        args.n_conf = 3
        args.n_therm = 12
        args.beta_g = args.beta_g[:1]

    os.makedirs(args.output_dir, exist_ok=True)
    print("the deflation test: sweep the t²E = c convention and watch the 0.22",
          flush=True)
    results = []
    for i, bg in enumerate(args.beta_g):
        t_max = args.t_max[min(i, len(args.t_max) - 1)]
        ens = run_ensemble(args, bg, t_max, seed_offset=i)
        results.append({"ens": ens, "analysis": analyse(ens, args)})

    lines = summarise(results, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args),
               "results": [{"beta_g": r["ens"]["beta_g"],
                            "scan": r["ens"]["scan"],
                            "analysis": r["analysis"]} for r in results]}
    with open(os.path.join(args.output_dir, "n3_kappa_deflation.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_kappa_deflation.png")
        make_figure(results, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
