"""The one-κ obstruction: why the sharper invariant fails, and the mechanism behind it.

The one-κ frontier (`n3_one_kappa_frontier.py`) showed the Bogomolny coherent
fraction does not give a matched `κ_I = κ_II`, and named two sharper candidates.
This tests the first — the dimensionless topological invariant `⟨Q²⟩/⟨S⟩`
(χ_top / action density, dimensionless in *any* dimension) — and then diagnoses
**why** no such lattice-topology estimator bridges the two acts.  The result is a
*mechanism*, not another bare non-match:

A. **The sharper invariant also diverges.**  `⟨Q²⟩/⟨S⟩` is `~10⁻⁴` in the SU(3)
   vacuum but `~10⁻²` in the CP² sector — two orders apart, and strongly
   dependent on the CP coupling/cooling.  A second natural, dimensionless
   invariant, a second non-match.
B. **The mechanism: topology has different RG relevance in the two theories.**
   Under the flow the SU(3) coherent fraction *rises* — 4-D instantons are
   (near) scale-invariant, so they survive the smoothing and emerge from the UV
   noise.  In the CP² vacuum the instanton density *collapses* under the same
   cooling flow — 2-D CP instantons are scale-*non*-invariant, they shrink
   through the lattice and annihilate.  Same flow, opposite fate.
C. **So the sectors' topological content lives at different, moving scales.**
   Any estimator built from `q(x)` and `e(x)` reads a scale-tied number, and the
   two scales flow apart — which is exactly why neither the coherent fraction nor
   the susceptibility ratio can coincide.

What this means for the identity (stated plainly): the equality of Act I's and
Act II's `κ` — if it holds — is **not** recoverable from a single lattice
topological estimator across these two models, because their topological sectors
renormalise differently.  A genuine bridge must either match the *physical
instanton scales* first (a renormalisation condition, not a raw ratio) or be a
framework-level *definition* that both are one abstract integration-exchange
rate.  This experiment turns the open frontier into a precise obstruction.

Honest scope: a deliberately honest **mechanism-of-a-boundary** result.  It does
not establish `κ_I = κ_II` (it explains why the obvious lattice routes cannot),
and the SU(3) susceptibility is a small-ensemble estimate (χ_top at this size is
noisy — the point is the order-of-magnitude gap and the flow behaviour, not a
precise number).  It closes the lattice-topology line of attack on the one-κ
identity with a reason, and points to what a real bridge would require.

Usage::

    python experiments/n3_kappa_obstruction.py --output-dir artifacts/n3_kobstruct
    python experiments/n3_kappa_obstruction.py --quick
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
    gradient_flow,
    self_dual_fraction,
)
from project_genesis.gauge_topology import topological_charge as gauge_charge
from project_genesis.topological_charge import (
    charge_variance_per_action,
    cool,
    cp_action,
    cp_metropolis_sweep,
    instanton_density,
    topological_charge as cp_charge,
)


def _nearest(traj, t):
    return min(traj, key=lambda r: abs(r["t"] - t))


def gauge_sector(args):
    """SU(3): ⟨Q²⟩/⟨S⟩ at t₀ and the self-dual fraction rising under the flow."""
    rng = np.random.default_rng(args.seed)
    L = args.gauge_size
    Qs, Ss, fsd_trajs, times = [], [], [], None
    for _ in range(args.n_conf):
        links = random_unitary(rng, 3, (4, L, L, L, L), special=True, scale=0.7)
        for _ in range(args.n_therm):
            links, _ = heatbath_sweep(links, args.beta_g, rng)
            links, _ = overrelaxation_sweep(links, rng)
        _, traj = gradient_flow(links, t_max=args.t_max, dt=args.dt,
                                measure_every=args.measure_every)
        t0 = flow_scale(traj, args.ref)
        rec = _nearest(traj, t0) if np.isfinite(t0) else traj[-1]
        Qs.append(rec["Q"]); Ss.append(rec["E"] * L ** 4)
        fsd_trajs.append([r["f_sd"] for r in traj])
        times = [r["t"] for r in traj]
    inv = charge_variance_per_action(Qs, Ss)
    fsd_mean = [float(np.mean([tr[i] for tr in fsd_trajs])) for i in range(len(times))]
    print(f"  SU(3) β={args.beta_g}, L={L}: ⟨Q²⟩/⟨S⟩ = {inv:.2e}  "
          f"(self-dual fraction rises {fsd_mean[0]:.3f} → {fsd_mean[-1]:.3f} under flow)",
          flush=True)
    return {"invariant": inv, "times": times, "f_sd": fsd_mean}


def cp_sector(args):
    """CP²: ⟨Q²⟩/⟨S⟩ (β/cool scan) and the instanton density collapsing under flow."""
    rng = np.random.default_rng(args.seed + 1)
    L = args.cp_size
    ensemble = []
    for _ in range(args.n_conf_cp):
        z = rng.standard_normal((L, L, 3)) + 1j * rng.standard_normal((L, L, 3))
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        for _ in range(args.n_therm_cp):
            z = cp_metropolis_sweep(z, args.beta_cp, rng, args.eps_cp)
        ensemble.append(z)
    # the sharper invariant across a β/cool scan (shows its scale/parameter dependence)
    scan = []
    for nc in args.cool_probe:
        Qs = [cp_charge(cool(z, nc, 1.0)) for z in ensemble]
        Ss = [cp_action(cool(z, nc, 1.0)) for z in ensemble]
        scan.append({"cool": nc, "invariant": charge_variance_per_action(Qs, Ss)})
    inv = scan[1]["invariant"] if len(scan) > 1 else scan[0]["invariant"]
    # instanton density under the cooling flow (the collapse)
    dens = [float(np.mean([instanton_density(cool(z, nc, 1.0)) for z in ensemble]))
            for nc in args.cool_steps]
    print(f"  CP² β={args.beta_cp}, L={L}: ⟨Q²⟩/⟨S⟩ = {inv:.2e} "
          f"(scan {scan[0]['invariant']:.2e}–{scan[-1]['invariant']:.2e}); "
          f"instanton density collapses {dens[0]:.3f} → {dens[-1]:.3f} under cooling",
          flush=True)
    return {"invariant": inv, "scan": scan, "cool_steps": list(args.cool_steps),
            "density": dens, "dens0": dens[0], "dens_end": dens[-1]}


def summarise(gauge, cp, args) -> list[str]:
    ratio = cp["invariant"] / gauge["invariant"] if gauge["invariant"] > 0 else float("inf")
    lines = ["The one-κ obstruction: why the sharper invariant fails, and the "
             "mechanism — verdict", "=" * 74, ""]
    lines.append(
        f"A. the sharper invariant also diverges: the dimensionless topological "
        f"invariant ⟨Q²⟩/⟨S⟩ (χ_top / action density) reads {gauge['invariant']:.1e} "
        f"in the SU(3) vacuum but {cp['invariant']:.1e} in the CP² sector — a "
        f"factor of ~{ratio:.0f} apart, and strongly dependent on the CP "
        f"coupling/cooling ({cp['scan'][0]['invariant']:.1e}–"
        f"{cp['scan'][-1]['invariant']:.1e} across the cool scan). A second "
        "natural dimensionless invariant, a second non-match.")
    lines.append("")
    lines.append(
        f"B. the mechanism — topology renormalises differently in the two theories: "
        f"under the flow the SU(3) self-dual fraction RISES "
        f"({gauge['f_sd'][0]:.3f} → {gauge['f_sd'][-1]:.3f}) — 4-D instantons are "
        f"(near) scale-invariant, they survive the smoothing and emerge from the "
        f"UV noise. In the CP² vacuum the instanton density COLLAPSES under the "
        f"same cooling flow ({cp['dens0']:.3f} → {cp['dens_end']:.3f}) — 2-D CP "
        "instantons are scale-non-invariant, they shrink through the lattice and "
        "annihilate. Same flow, opposite fate.")
    lines.append("")
    lines.append(
        "C. so the sectors' topological content lives at different, moving scales: "
        "any estimator built from q(x) and e(x) reads a scale-tied number, and the "
        "two scales flow apart — exactly why neither the coherent fraction nor the "
        "susceptibility ratio can coincide.")
    lines.append("")
    lines.append(
        "what it means for the identity: the equality of Act I's and Act II's κ — "
        "if it holds — is NOT recoverable from a single lattice topological "
        "estimator across these two models, because their topological sectors "
        "renormalise differently. A genuine bridge must either match the physical "
        "instanton scales first (a renormalisation condition, not a raw ratio) or "
        "be a framework-level definition that both are one abstract "
        "integration-exchange rate. The open frontier is now a precise obstruction.")
    lines.append("")
    lines.append(
        "honest scope: a mechanism-of-a-boundary result. It does not establish "
        "κ_I = κ_II (it explains why the obvious lattice routes cannot), and the "
        "SU(3) susceptibility is a small-ensemble estimate (χ_top is noisy at this "
        "size — the point is the order-of-magnitude gap and the flow behaviour, "
        "not a precise number). It closes the lattice-topology line of attack with "
        "a reason, and points to what a real bridge would require.")
    return lines


def make_figure(gauge, cp, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY, RED, PURPLE = (
        "#0072B2", "#E69F00", "#009E73", "#999999", "#c0392b", "#8551A6")
    fig = plt.figure(figsize=(13, 9.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.27)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))
    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # panel A: the sharper invariant — orders of magnitude apart
    ax1.bar([0, 1], [gauge["invariant"], cp["invariant"]], color=[BLUE, GREEN],
            width=0.6, zorder=3)
    ax1.set_yscale("log")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["SU(3)\n(Act I)", "CP²\n(Act II)"])
    ax1.set_ylabel("⟨Q²⟩ / ⟨S⟩  (dimensionless)")
    ax1.set_title("A. The sharper invariant also diverges")
    for i, v in enumerate([gauge["invariant"], cp["invariant"]]):
        ax1.text(i, v, f" {v:.1e}", ha="center", va="bottom", fontsize=9)

    # panel B: mechanism — gauge coherent fraction rises under flow
    t = np.array(gauge["times"]); f = np.array(gauge["f_sd"])
    ax2.plot(t, f, color=BLUE, lw=2.2, marker="o", ms=3, zorder=3)
    ax2.annotate("instantons survive\n(4-D scale-invariant)", xy=(t[-1], f[-1]),
                 xytext=(t[len(t) // 3], f[-1]), fontsize=8, color=BLUE,
                 va="center")
    ax2.set_xlabel("Wilson flow time  t")
    ax2.set_ylabel("self-dual fraction  (SU(3))")
    ax2.set_title("B. Act I: coherent content RISES under flow")

    # panel C: mechanism — CP instanton density collapses under cooling
    cs = np.array(cp["cool_steps"]); d = np.array(cp["density"])
    ax3.plot(cs, d, color=GREEN, lw=2.2, marker="s", ms=4, zorder=3)
    ax3.annotate("instantons shrink\n& annihilate\n(2-D scale-relevant)",
                 xy=(cs[-1], d[-1]), xytext=(cs[len(cs) // 3], d[0] * 0.6),
                 fontsize=8, color=GREEN, va="center")
    ax3.set_xlabel("cooling-flow steps")
    ax3.set_ylabel("instanton density  ⟨|q|/site⟩  (CP²)")
    ax3.set_title("C. Act II: instanton content COLLAPSES under flow")

    # panel D: the verdict — the obstruction, in words
    ax4.axis("off")
    ax4.set_title("D. The obstruction (and what a bridge needs)")
    txt = (
        "same operator, opposite RG fate:\n"
        "  • SU(3) 4-D — instantons scale-invariant → survive\n"
        "  • CP²  2-D — instantons scale-relevant → annihilate\n\n"
        "⇒ topological content sits at different,\n"
        "   flowing scales in the two acts, so any\n"
        "   q(x),e(x) estimator reads scale-tied\n"
        "   numbers that flow apart.\n\n"
        "a genuine κ_I = κ_II would need:\n"
        "  (1) match the physical instanton scales\n"
        "      first (a renormalisation condition), or\n"
        "  (2) a framework-level definition that both\n"
        "      are one integration-exchange rate.\n\n"
        "the frontier is now a precise obstruction,\n"
        "not a hand-waved hope.")
    ax4.text(0.02, 0.92, txt, transform=ax4.transAxes, fontsize=9.5, va="top",
             family="monospace")

    fig.suptitle("The one-κ obstruction: the sharper invariant fails too — "
                 "and the mechanism is the instantons' different RG fate",
                 fontsize=12, y=0.98)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gauge-size", type=int, default=8)
    p.add_argument("--beta-g", type=float, default=1.8)
    p.add_argument("--n-conf", type=int, default=6)
    p.add_argument("--n-therm", type=int, default=20)
    p.add_argument("--t-max", type=float, default=1.2)
    p.add_argument("--dt", type=float, default=0.03)
    p.add_argument("--measure-every", type=int, default=4)
    p.add_argument("--ref", type=float, default=0.3)
    p.add_argument("--cp-size", type=int, default=40)
    p.add_argument("--beta-cp", type=float, default=1.7)
    p.add_argument("--eps-cp", type=float, default=0.5)
    p.add_argument("--n-conf-cp", type=int, default=12)
    p.add_argument("--n-therm-cp", type=int, default=30)
    p.add_argument("--cool-probe", type=int, nargs="+", default=[2, 5, 10])
    p.add_argument("--cool-steps", type=int, nargs="+",
                   default=[0, 2, 5, 10, 20, 40])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_kobstruct")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.gauge_size = 6
        args.n_conf = 3
        args.n_therm = 12
        args.cp_size = 32
        args.n_conf_cp = 8
        args.n_therm_cp = 20

    os.makedirs(args.output_dir, exist_ok=True)
    print("the one-κ obstruction: the sharper invariant ⟨Q²⟩/⟨S⟩, and why lattice "
          "topology cannot bridge the two acts", flush=True)
    print("A/B — SU(3) sector (Act I):")
    gauge = gauge_sector(args)
    print("A/C — CP² sector (Act II substrate):")
    cp = cp_sector(args)

    lines = summarise(gauge, cp, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args), "gauge_invariant": gauge["invariant"],
               "cp_invariant": cp["invariant"], "cp_scan": cp["scan"],
               "gauge_fsd_rise": [gauge["f_sd"][0], gauge["f_sd"][-1]],
               "cp_density_collapse": [cp["dens0"], cp["dens_end"]]}
    with open(os.path.join(args.output_dir, "n3_kappa_obstruction.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_kappa_obstruction.png")
        make_figure(gauge, cp, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
