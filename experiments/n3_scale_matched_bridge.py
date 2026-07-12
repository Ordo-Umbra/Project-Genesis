"""The scale-matched bridge: the one-κ comparison after the renormalisation condition.

The one-κ obstruction (`n3_kappa_obstruction.py`) proved the naive route dead:
the two acts' instantons renormalise oppositely under the flow (4-D SU(3)
instantons survive and emerge; 2-D CP² instantons shrink and annihilate), so any
bare `q(x), e(x)` estimator compares numbers pinned to *different, moving*
scales.  It also named the one concrete route left: **match the physical
instanton scales first** — a renormalisation condition — and only then compare.
This experiment walks that route.

The renormalisation condition (dimensionless, sector-local):

    s(t)  :=  λ(t) / ρ̄(t)

— the flow's heat-kernel smoothing radius `λ_d(t) = √(2·d·D·t)` (Lüscher's
`√(8t)` in 4-D; the CP flow's `D` is *measured*, not assumed, by
`cp_flow_diffusion_constant`) in units of the sector's own **mean instanton
size** `ρ̄(t)`, read from charge-density peak heights via the exact classical
profiles (BPST in 4-D, the CP lump in 2-D).  Two sectors observed at equal `s`
have been smoothed by the same fraction of their own topological scale — the
matched-RG condition the obstruction called for.

The instrument then interpolates both sectors' Bogomolny coherent fractions
`κ̂ = Σ|q|/Σe` onto a common grid of `s` and reports:

A. **The scales themselves** — `ρ̄(t)` per sector along its flow (the mechanism
   of the obstruction, now quantified: whether SU(3) sizes hold/grow while CP
   sizes shrink).
B. **The unmatched recap** — `κ̂` against raw flow time, the divergence the
   frontier found.
C. **The matched comparison** — `κ̂_I(s)` vs `κ̂_II(s)`: do the curves meet at
   any matched scale, and if so at what `κ⋆`?  How does it sit against Act I's
   `κ̂(t₀) ≈ 0.22`?
D. **The verdict**, data-driven: matched / crossing found / still-open gap —
   whatever the computation actually says.

Usage::

    python experiments/n3_scale_matched_bridge.py --output-dir artifacts/n3_smbridge
    python experiments/n3_scale_matched_bridge.py --quick
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
from project_genesis.gauge_topology import (
    flow_energy,
    flow_scale,
    flow_step,
    self_dual_fraction,
    topological_charge_density as gauge_qdensity,
)
from project_genesis.instanton_scales import (
    cp_flow_diffusion_constant,
    cp_gradient_flow,
    instanton_sizes_su,
    matched_comparison,
    smoothing_radius,
)
from project_genesis.topological_charge import cp_metropolis_sweep


def gauge_sector(args):
    """SU(3): Wilson flow recording κ̂(t) and the mean instanton size ρ̄(t)."""
    rng = np.random.default_rng(args.seed)
    L = args.gauge_size
    n_steps = max(1, int(round(args.t_max / args.dt)))
    rec_steps = [0] + [s for s in range(1, n_steps + 1)
                       if s % args.measure_every == 0 or s == n_steps]
    times = [s * args.dt for s in rec_steps]
    kappa_cfg, rho_cfg, t0s = [], [], []
    for _ in range(args.n_conf):
        links = random_unitary(rng, 3, (4, L, L, L, L), special=True, scale=0.7)
        for _ in range(args.n_therm):
            links, _ = heatbath_sweep(links, args.beta_g, rng)
            links, _ = overrelaxation_sweep(links, rng)
        kap, rho, traj = [], [], []
        z = links

        def record(t):
            e = flow_energy(z)
            traj.append({"t": t, "t2E": t * t * e})
            kap.append(self_dual_fraction(z))
            sizes = instanton_sizes_su(gauge_qdensity(z), min_frac=args.min_frac)
            rho.append(float(sizes.mean()) if sizes.size else float("nan"))

        record(0.0)
        for step in range(1, n_steps + 1):
            z = flow_step(z, args.dt)
            if step % args.measure_every == 0 or step == n_steps:
                record(step * args.dt)
        kappa_cfg.append(kap)
        rho_cfg.append(rho)
        t0s.append(flow_scale(traj, args.ref))
    kappa = np.nanmean(np.array(kappa_cfg), axis=0)
    rho = np.nanmean(np.array(rho_cfg), axis=0)
    lam = smoothing_radius(np.array(times), dim=4)
    s = lam / rho
    t0 = float(np.nanmean(t0s))
    print(f"  SU(3) β={args.beta_g}, L={L}, {args.n_conf} configs: "
          f"ρ̄ {rho[0]:.2f} → {rho[-1]:.2f},  κ̂ {kappa[0]:.3f} → {kappa[-1]:.3f},  "
          f"s = λ/ρ̄ up to {np.nanmax(s):.2f}  (t₀ ≈ {t0:.3f})", flush=True)
    return {"times": times, "kappa": kappa.tolist(), "rho": rho.tolist(),
            "lambda": lam.tolist(), "s": s.tolist(), "t0": t0}


def cp_sector(args, beta: float, seed_offset: int = 1):
    """CP²: gradient flow (measured D) recording κ̂(τ) and ρ̄(τ) at coupling ``beta``."""
    rng = np.random.default_rng(args.seed + seed_offset)
    L = args.cp_size
    diffusion = cp_flow_diffusion_constant()
    trajs = []
    for _ in range(args.n_conf_cp):
        z = rng.standard_normal((L, L, 3)) + 1j * rng.standard_normal((L, L, 3))
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        for _ in range(args.n_therm_cp):
            z = cp_metropolis_sweep(z, beta, rng, args.eps_cp)
        _, traj = cp_gradient_flow(z, t_max=args.tau_max, dt=args.dtau,
                                   measure_every=args.measure_every_cp,
                                   min_frac=args.min_frac)
        trajs.append(traj)
    times = [r["t"] for r in trajs[0]]
    kappa = np.nanmean([[r["kappa"] for r in tr] for tr in trajs], axis=0)
    with np.errstate(invalid="ignore"):
        rho = np.nanmean([[r["rho_mean"] for r in tr] for tr in trajs], axis=0)
    lam = smoothing_radius(np.array(times), dim=2, diffusion=diffusion)
    s = lam / rho
    print(f"  CP²   β={beta}, L={L}, {args.n_conf_cp} configs "
          f"(measured D = {diffusion:.3f}): "
          f"ρ̄ {rho[0]:.2f} → {rho[-1]:.2f},  κ̂ {kappa[0]:.3f} → {kappa[-1]:.3f},  "
          f"s = λ/ρ̄ up to {np.nanmax(s):.2f}", flush=True)
    return {"beta": beta, "times": times, "kappa": kappa.tolist(),
            "rho": rho.tolist(), "lambda": lam.tolist(), "s": s.tolist(),
            "diffusion": diffusion}


def summarise(gauge, cp, match, args, robust=None, cp2=None) -> list[str]:
    lines = ["The scale-matched bridge: the one-κ comparison after the "
             "renormalisation condition — verdict", "=" * 74, ""]
    lines.append(
        f"A. the scales, measured: along its flow the SU(3) mean instanton size "
        f"moves {gauge['rho'][0]:.2f} → {gauge['rho'][-1]:.2f} (lattice units) "
        f"while the CP² size moves {cp['rho'][0]:.2f} → {cp['rho'][-1]:.2f} — the "
        "obstruction's mechanism (opposite RG fates) is now a measured pair of "
        "size trajectories, and the matched coordinate s = λ/ρ̄ absorbs it.")
    lines.append("")
    lines.append(
        f"B. the unmatched recap: against raw flow time the coherent fractions "
        f"still diverge (SU(3) {gauge['kappa'][0]:.3f} → {gauge['kappa'][-1]:.3f}; "
        f"CP² {cp['kappa'][0]:.3f} → {cp['kappa'][-1]:.3f}) — reproducing the "
        "frontier.")
    lines.append("")
    if not match["overlap"]:
        lines.append(
            "C. the matched comparison: the two sectors' matched-coordinate "
            f"ranges do not overlap ({match.get('reason', '')}; "
            f"SU(3) s ∈ {match.get('range_a')}, CP² s ∈ {match.get('range_b')}). "
            "Even after the renormalisation condition the two flows never see "
            "their topology at a common relative scale in this window — the "
            "obstruction survives in sharpened form.")
        verdict = "no-overlap"
    elif match["crossing"]:
        lines.append(
            f"C. the matched comparison: at matched scale s⋆ = {match['s_star']:.2f} "
            f"(smoothing radius = {match['s_star']:.2f}·ρ̄ in both sectors) the two "
            f"coherent fractions CROSS at κ⋆ = {match['kappa_star']:.3f}. After the "
            "renormalisation condition the two acts' κ̂ do meet — the bridge has a "
            "matched point, where the raw comparison had none.")
        verdict = "crossing"
    else:
        lines.append(
            f"C. the matched comparison: the curves approach but do not cross — "
            f"minimum gap |κ̂_I − κ̂_II| = {match['min_gap']:.3f} at "
            f"s = {match['s_star']:.2f} (κ̂_I = {match['kappa_star']:.3f} there). "
            "The renormalisation condition narrows the frontier but does not close "
            "it in this window.")
        verdict = "open-gap"
    lines.append("")
    if match["overlap"]:
        kstar = match["kappa_star"]
        lines.append(
            f"D. against Act I's number: the matched point sits at κ ≈ {kstar:.3f} "
            f"vs the RG-clean SU(3) reading κ̂(t₀) ≈ 0.22 — "
            + ("consistent to within the coarse-lattice systematics."
               if abs(kstar - 0.22) < 0.06 else
               "a different number; the matched point and the t₀ reading are "
               "distinct observables and must not be conflated."))
        lines.append("")
    if robust is not None and robust.get("overlap"):
        spread = float(np.max(np.abs(robust["gap"])))
        lines.append(
            f"E. coupling robustness: the CP² matched curve κ̂_II(s) at "
            f"β = {cp['beta']:g} vs β = {cp2['beta']:g} differs by at most "
            f"{spread:.3f} over the shared window — "
            + ("the matched coordinate absorbs the bare coupling to this "
               "precision, so the comparison is not an artefact of one β."
               if spread < 0.05 else
               "a genuine coupling dependence survives the matching, so the "
               "matched κ̂_II is NOT yet scheme-free — the comparison inherits "
               "this systematic."))
        lines.append("")
    lines.append(
        "honest scope: peak-height sizes assume well-separated classical lumps — "
        "early-flow readings are UV-noise-dominated and enter only through the "
        "matched coordinate, and the clover/geometric Z ≲ 1 under-normalisation "
        "over-reads sizes on coarse lattices (fading as the flow deepens). The "
        "matching condition s = λ/ρ̄ is *a* renormalisation condition (the natural "
        "dimensionless one), not the unique choice; a crossing under it is a "
        "matched point, not yet a scheme-independent identity. The comparison "
        f"admits only λ ≥ {args.lambda_min:g} lattice units (a sub-lattice "
        "smoothing radius is not a scale), and at the window's deep end the "
        "smoothing radius approaches L/2, where finite-volume contamination "
        "grows. Small ensembles, one (β_g, β_cp) point each.")
    return lines


def make_figure(gauge, cp, match, args, path, cp2=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, GREEN, GRAY, RED = "#0072B2", "#009E73", "#999999", "#c0392b"
    fig = plt.figure(figsize=(13, 9.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.27)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))
    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # panel A: the measured instanton-size trajectories
    ax1.plot(gauge["times"], gauge["rho"], color=BLUE, lw=2.2, marker="o",
             ms=3, zorder=3, label="SU(3) (4-D)")
    ax1b = ax1.twiny()
    ax1b.plot(cp["times"], cp["rho"], color=GREEN, lw=2.2, marker="s",
              ms=3, zorder=3, label="CP² (2-D)")
    ax1b.set_xlabel("CP flow time  τ", color=GREEN)
    ax1.set_xlabel("Wilson flow time  t", color=BLUE)
    ax1.set_ylabel("mean instanton size  ρ̄  (lattice units)")
    ax1.set_title("A. The scales, measured: ρ̄ along each flow")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")

    # panel B: unmatched recap — κ̂ vs raw flow time
    ax2.plot(gauge["times"], gauge["kappa"], color=BLUE, lw=2.2, marker="o",
             ms=3, zorder=3, label="SU(3)")
    ax2b = ax2.twiny()
    ax2b.plot(cp["times"], cp["kappa"], color=GREEN, lw=2.2, marker="s",
              ms=3, zorder=3, label="CP²")
    ax2.axhline(0.22, color=GRAY, lw=1.0, ls="--", zorder=2)
    ax2.text(0.02, 0.225, "κ̂(t₀) ≈ 0.22", fontsize=8, color=GRAY,
             transform=ax2.get_yaxis_transform())
    ax2.set_xlabel("Wilson flow time  t", color=BLUE)
    ax2b.set_xlabel("CP flow time  τ", color=GREEN)
    ax2.set_ylabel("coherent fraction  κ̂ = Σ|q|/Σe")
    ax2.set_title("B. Unmatched (the frontier): κ̂ vs raw flow time")

    # panel C: the matched comparison
    if match["overlap"]:
        sg = np.asarray(match["s_grid"])
        ax3.plot(sg, match["kappa_a"], color=BLUE, lw=2.2, zorder=3,
                 label="SU(3)  κ̂_I(s)")
        ax3.plot(sg, match["kappa_b"], color=GREEN, lw=2.2, zorder=3,
                 label=f"CP²  κ̂_II(s)  (β={cp['beta']:g})")
        if cp2 is not None:
            lam2 = np.asarray(cp2["lambda"])
            good = lam2 >= args.lambda_min
            ax3.plot(np.asarray(cp2["s"])[good], np.asarray(cp2["kappa"])[good],
                     color=GREEN, lw=1.4, ls="--", zorder=3,
                     label=f"CP²  β={cp2['beta']:g} (robustness)")
        ax3.axhline(0.22, color=GRAY, lw=1.0, ls="--", zorder=2)
        marker = "crossing" if match["crossing"] else "closest approach"
        ax3.plot([match["s_star"]], [match["kappa_star"]], "o", color=RED,
                 ms=9, zorder=4, label=f"{marker}: κ = {match['kappa_star']:.3f}")
        ax3.set_xlabel("matched scale  s = λ(t) / ρ̄(t)")
    else:
        ax3.text(0.5, 0.5, "matched ranges do not overlap",
                 ha="center", va="center", fontsize=11, color=RED)
    ax3.set_ylabel("coherent fraction  κ̂")
    ax3.set_title("C. Matched: κ̂ at equal smoothing-per-instanton-size")
    ax3.legend(fontsize=8, loc="best")

    # panel D: the verdict, in words
    ax4.axis("off")
    ax4.set_title("D. The verdict")
    if not match["overlap"]:
        txt = ("the renormalisation condition s = λ/ρ̄\n"
               "cannot even be imposed: the two sectors'\n"
               "matched ranges are disjoint in this window.\n\n"
               "the obstruction survives, sharpened.")
    elif match["crossing"]:
        txt = (f"matched point found:\n\n"
               f"   s⋆ = {match['s_star']:.2f}   κ⋆ = {match['kappa_star']:.3f}\n\n"
               "after matching the physical instanton\n"
               "scales — the obstruction's own condition —\n"
               "the two acts' coherent fractions MEET.\n\n"
               "one matching condition, one number.\n"
               "(a matched point under this condition;\n"
               "scheme-independence is the next test.)")
    else:
        txt = (f"no crossing in the matched window:\n\n"
               f"   min |κ̂_I − κ̂_II| = {match['min_gap']:.3f}\n"
               f"   at s = {match['s_star']:.2f}\n\n"
               "the condition narrows the frontier\n"
               "but does not close it here.")
    ax4.text(0.05, 0.85, txt, transform=ax4.transAxes, fontsize=10,
             va="top", family="monospace")

    fig.suptitle("The scale-matched bridge: κ̂ compared at equal smoothing "
                 "radius per instanton size (s = λ/ρ̄)", fontsize=12, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gauge-size", type=int, default=8)
    p.add_argument("--beta-g", type=float, default=1.8)
    p.add_argument("--n-conf", type=int, default=6)
    p.add_argument("--n-therm", type=int, default=20)
    p.add_argument("--t-max", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=0.03)
    p.add_argument("--measure-every", type=int, default=5)
    p.add_argument("--ref", type=float, default=0.3)
    p.add_argument("--cp-size", type=int, default=40)
    p.add_argument("--beta-cp", type=float, default=1.7)
    p.add_argument("--beta-cp2", type=float, default=2.2,
                   help="Second CP coupling for the robustness check "
                        "(0 disables).")
    p.add_argument("--eps-cp", type=float, default=0.5)
    p.add_argument("--n-conf-cp", type=int, default=12)
    p.add_argument("--n-therm-cp", type=int, default=30)
    p.add_argument("--tau-max", type=float, default=12.0)
    p.add_argument("--dtau", type=float, default=0.1)
    p.add_argument("--measure-every-cp", type=int, default=2)
    p.add_argument("--min-frac", type=float, default=0.25)
    p.add_argument("--lambda-min", type=float, default=1.0,
                   help="Smallest smoothing radius (lattice units) admitted "
                        "to the matched comparison.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_smbridge")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.gauge_size = 6
        args.n_conf = 3
        args.n_therm = 12
        args.t_max = 1.5
        args.cp_size = 32
        args.n_conf_cp = 6
        args.n_therm_cp = 20
        args.tau_max = 8.0
        args.measure_every_cp = 1

    os.makedirs(args.output_dir, exist_ok=True)
    print("the scale-matched bridge: κ̂ compared after matching the physical "
          "instanton scales (s = λ/ρ̄)", flush=True)
    print("A/B — SU(3) sector (Act I):")
    gauge = gauge_sector(args)
    print("A/B — CP² sector (Act II substrate):")
    cp = cp_sector(args, args.beta_cp)
    cp2 = cp_sector(args, args.beta_cp2, seed_offset=2) if args.beta_cp2 > 0 else None

    def resolved(sector):
        # a smoothing radius below one lattice spacing is not a scale at all —
        # the matched comparison only starts once the flow resolves the lattice
        lam = np.asarray(sector["lambda"])
        s = np.asarray(sector["s"])
        k = np.asarray(sector["kappa"])
        good = lam >= args.lambda_min
        return s[good], k[good]

    match = matched_comparison(*resolved(gauge), *resolved(cp))
    # coupling-robustness: does the CP matched curve depend on the bare coupling?
    robust = (matched_comparison(*resolved(cp), *resolved(cp2))
              if cp2 is not None else None)

    lines = summarise(gauge, cp, match, args, robust=robust, cp2=cp2)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    def plain(d):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in d.items()} if d is not None else None

    payload = {"params": vars(args), "gauge": gauge, "cp": cp, "cp2": cp2,
               "match": plain(match), "robust": plain(robust)}
    with open(os.path.join(args.output_dir, "n3_scale_matched_bridge.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_scale_matched_bridge.png")
        make_figure(gauge, cp, match, args, fig_path, cp2=cp2)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
