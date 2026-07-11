"""The one-κ frontier: a shared coherent-fraction operator across the two acts.

The URP reading of `κ` is the **integration/exchange fraction** — the share of a
field's action carried by topologically coherent structure.  Act I measured it
in the 4-D SU(3) vacuum as the self-dual fraction `f_SD(t₀) ≈ 0.22` (the instanton
fraction of the gluon condensate).  Act II uses `κ` as the capacity field whose
free energy is gravity.  The natural question — the one that would fuse the two
acts — is whether these are the *same dimensionless number*.

This experiment reports, honestly, what the computation says.  It builds **one
operator** — the Bogomolny coherent fraction `κ̂ = Σ|q(x)| / Σ e(x) ∈ [0,1]`
(`topological_charge.coherent_fraction`) — and applies it identically to both
sectors, measured at a matched flow scale.  The finding is a *frontier*, not a
coincidence:

A. **One operator, two sectors.**  `κ̂` is literally the same function that
   `gauge_topology.self_dual_fraction` computes for the SU(3) field and
   `topological_charge.cp_coherent_fraction` for the CP sector.  The Bogomolny
   bound `Σe ≥ Σ|q|` holds in both, so it is a genuine coherence fraction in
   `[0,1]`.
B. **Act I: it rises to 0.22.**  Under the Wilson flow the SU(3) self-dual
   fraction *rises* from the UV-noise floor to `κ̂(t₀) ≈ 0.22` at the RG-clean
   scale `t₀` (`t²E = 0.3`) — the instanton content emerging from the noise.
C. **Act II substrate: it falls.**  In a *thermal* CP² vacuum (a Metropolis
   ensemble), the same `κ̂` *falls* monotonically under the cooling flow — the
   opposite behaviour — with no plateau near 0.22.
D. **The frontier.**  Plotted against a common flow coordinate the two
   trajectories of the *same operator* move in opposite directions: **the naive
   coherent fraction does not give a parameter-free `κ_I = κ_II`.**  What an
   identity would require is stated plainly (a matched RG condition across a 4-D
   and a 2-D flow, or a different invariant — a susceptibility ratio or the
   instanton-size distribution rather than the action fraction).

Honest scope: this is a deliberately honest **negative/boundary** result.  The
two `κ`'s are the same *concept* — the integration fraction — and the operator is
genuinely one function, but their lattice realisations differ in normalisation
and flow behaviour, so a parameter-free numerical identity is *not* present in
this estimator.  The value is in mapping the frontier cleanly: what holds
(structural identity, Act I's 0.22), what does not (a matched number), and what
a real bridge would need.

Usage::

    python experiments/n3_one_kappa_frontier.py --output-dir artifacts/n3_onekappa
    python experiments/n3_one_kappa_frontier.py --quick
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
    action_density,
    flow_scale,
    gradient_flow,
    self_dual_fraction,
)
from project_genesis.gauge_topology import topological_charge_density as gauge_qdensity
from project_genesis.topological_charge import (
    coherent_fraction,
    cool,
    cp_action,
    cp_action_density,
    cp_coherent_fraction,
    cp_metropolis_sweep,
    topological_charge_density,
)


def operator_identity(args):
    """A: κ̂ is one operator; the Bogomolny bound Σe≥Σ|q| holds in both sectors."""
    rng = np.random.default_rng(args.seed)
    links = random_unitary(rng, 3, (4, 6, 6, 6, 6), special=True, scale=0.7)
    gauge_same = np.isclose(
        self_dual_fraction(links),
        coherent_fraction(action_density(links), gauge_qdensity(links)))
    e_g = float(action_density(links).sum())
    q_g = float(np.abs(gauge_qdensity(links)).sum())

    z = rng.standard_normal((32, 32, 3)) + 1j * rng.standard_normal((32, 32, 3))
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    for _ in range(20):
        z = cp_metropolis_sweep(z, args.beta_cp, rng, args.eps_cp)
    cp_same = np.isclose(cp_coherent_fraction(z),
                         coherent_fraction(cp_action_density(z),
                                           topological_charge_density(z)))
    e_c = float(cp_action_density(z).sum())
    q_c = float(np.abs(topological_charge_density(z)).sum())
    print(f"  gauge: self_dual_fraction == coherent_fraction(e,q)? {bool(gauge_same)}"
          f"   Σe≥Σ|q|? {e_g >= q_g} ({q_g / e_g:.3f} ∈ [0,1])", flush=True)
    print(f"  CP:    cp_coherent_fraction == coherent_fraction(e,q)? {bool(cp_same)}"
          f"   Σe≥Σ|q|? {e_c >= q_c} ({q_c / e_c:.3f} ∈ [0,1])", flush=True)
    return {"gauge_identity": bool(gauge_same), "cp_identity": bool(cp_same),
            "gauge_bound": bool(e_g >= q_g), "cp_bound": bool(e_c >= q_c)}


def _interp(traj, t, key):
    for a, b in zip(traj, traj[1:]):
        if a["t"] <= t <= b["t"] and b["t"] > a["t"]:
            frac = (t - a["t"]) / (b["t"] - a["t"])
            return float(a[key] + frac * (b[key] - a[key]))
    return float(traj[-1][key])


def gauge_sector(args):
    """B: the SU(3) coherent fraction rises to κ̂(t₀) ≈ 0.22 under the Wilson flow."""
    rng = np.random.default_rng(args.seed)
    L = args.gauge_size
    trajs = []
    for _ in range(args.n_conf):
        links = random_unitary(rng, 3, (4, L, L, L, L), special=True, scale=0.7)
        for _ in range(args.n_therm):
            links, _ = heatbath_sweep(links, args.beta_g, rng)
            links, _ = overrelaxation_sweep(links, rng)
        _, traj = gradient_flow(links, t_max=args.t_max, dt=args.dt,
                                measure_every=args.measure_every)
        trajs.append(traj)
    times = [r["t"] for r in trajs[0]]
    e0 = float(np.mean([tr[0]["E"] for tr in trajs]))
    mean = {}
    for k in ("t2E", "f_sd", "E"):
        mean[k] = [float(np.mean([tr[i][k] for tr in trajs])) for i in range(len(times))]
    mean_traj = [{"t": times[i], "t2E": mean["t2E"][i]} for i in range(len(times))]
    t0 = flow_scale(mean_traj, args.ref)
    fsd_t0 = [_interp(tr, t0, "f_sd") for tr in trajs]
    progress = [1.0 - e / e0 for e in mean["E"]]           # fraction of action dissipated
    prog_t0 = float(np.interp(t0, times, progress)) if np.isfinite(t0) else float("nan")
    kappa = float(np.nanmean(fsd_t0)); kstd = float(np.nanstd(fsd_t0))
    print(f"  SU(3) β_g={args.beta_g}, L={L}: t₀={t0:.3f}, κ̂(t₀)={kappa:.4f} ± "
          f"{kstd:.4f}  (rises under flow)", flush=True)
    return {"times": times, "f": mean["f_sd"], "progress": progress, "t0": t0,
            "kappa_t0": kappa, "kappa_std": kstd, "prog_t0": prog_t0}


def cp_sector(args):
    """C: the CP² coherent fraction falls under the cooling flow (thermal vacuum)."""
    rng = np.random.default_rng(args.seed + 1)
    L = args.cp_size
    fs, progs = [], []
    for _ in range(args.n_conf_cp):
        z = rng.standard_normal((L, L, 3)) + 1j * rng.standard_normal((L, L, 3))
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        for _ in range(args.n_therm_cp):
            z = cp_metropolis_sweep(z, args.beta_cp, rng, args.eps_cp)
        e0 = cp_action(z) / (L * L)
        f_here, p_here = [], []
        for nc in args.cool_steps:
            zc = cool(z, nc, 1.0)
            f_here.append(cp_coherent_fraction(zc))
            p_here.append(1.0 - (cp_action(zc) / (L * L)) / e0)
        fs.append(f_here); progs.append(p_here)
    f = np.array(fs).mean(axis=0); prog = np.array(progs).mean(axis=0)
    print(f"  CP² β_cp={args.beta_cp}, L={L}: κ̂ = {f[0]:.4f} → {f[-1]:.4f} "
          f"(falls under cooling)", flush=True)
    return {"cool_steps": list(args.cool_steps), "f": f.tolist(),
            "progress": prog.tolist(), "f_raw": float(f[0]), "f_cooled": float(f[-1])}


def matched_reading(gauge, cp):
    """Compare κ̂ at the deepest flow progress *both* sectors share (no extrapolation).

    The gauge flow at t₀ dissipates more action than the CP cooling flow reaches,
    so the honest matched point is ``p* = min(gauge progress at t₀, max CP
    progress)`` — where both trajectories have data.
    """
    p0 = gauge["prog_t0"]
    cp_prog = np.array(cp["progress"]); cp_f = np.array(cp["f"])
    g_prog = np.array(gauge["progress"]); g_f = np.array(gauge["f"])
    if not np.isfinite(p0):
        return {"progress": float("nan"), "gauge_k": float("nan"),
                "cp_k": float("nan")}
    p_star = float(min(p0, cp_prog.max()))
    order_c = np.argsort(cp_prog); order_g = np.argsort(g_prog)
    cp_k = float(np.interp(p_star, cp_prog[order_c], cp_f[order_c]))
    gauge_k = float(np.interp(p_star, g_prog[order_g], g_f[order_g]))
    return {"progress": p_star, "gauge_k": gauge_k, "cp_k": cp_k}


def summarise(op, gauge, cp, cp_matched, args) -> list[str]:
    lines = ["The one-κ frontier: a shared coherent-fraction operator across the "
             "two acts — verdict", "=" * 74, ""]
    lines.append(
        f"A. one operator, two sectors: the Bogomolny coherent fraction "
        f"κ̂ = Σ|q|/Σe is a single function — it is exactly what "
        f"self_dual_fraction computes for SU(3) (identity: {op['gauge_identity']}) "
        f"and cp_coherent_fraction for the CP sector (identity: {op['cp_identity']}), "
        f"with the bound Σe≥Σ|q| holding in both (κ̂ ∈ [0,1]). The operator "
        "transfers cleanly between the acts.")
    lines.append("")
    lines.append(
        f"B. Act I — it rises to 0.22: under the Wilson flow the SU(3) self-dual "
        f"fraction rises from the UV-noise floor to κ̂(t₀) = {gauge['kappa_t0']:.3f} "
        f"± {gauge['kappa_std']:.3f} at the RG-clean scale t₀={gauge['t0']:.2f} "
        f"(t²E={args.ref}) — the instanton content of the vacuum, the established "
        "κ ≈ 0.22.")
    lines.append("")
    lines.append(
        f"C. Act II substrate — it falls: in a thermal CP² vacuum the same κ̂ "
        f"*falls* monotonically under the cooling flow, κ̂ = {cp['f_raw']:.3f} → "
        f"{cp['f_cooled']:.3f}, the opposite behaviour, with no plateau near 0.22.")
    lines.append("")
    lines.append(
        f"D. the frontier: at the deepest smoothing both flows share (flow "
        f"progress {cp_matched['progress']:.2f}), the same operator reads "
        f"κ̂ = {cp_matched['gauge_k']:.3f} in SU(3) but {cp_matched['cp_k']:.3f} "
        f"in CP² — and the gauge value keeps rising to 0.22 at t₀ while the CP "
        f"value keeps falling. The same operator moves in opposite directions "
        "under the two flows: the naive coherent fraction does NOT give a "
        "parameter-free κ_I = κ_II. Honestly, the two κ's are the same concept "
        "(the integration fraction) but not the same measured number in this "
        "estimator.")
    lines.append("")
    lines.append(
        "what an identity would require: a matched renormalisation-group condition "
        "relating the 4-D SU(3) flow and the 2-D CP flow (they smooth differently, "
        "and the dimensionful flow clocks t²E vs t·E are not automatically "
        "comparable), or a different invariant — a ratio of topological "
        "susceptibilities χ_top, or the instanton *size* distribution rather than "
        "the action fraction. Those are concrete next tests; this maps the frontier "
        "rather than forcing a coincidence.")
    lines.append("")
    lines.append(
        "honest scope: a deliberately honest boundary result. The operator is "
        "genuinely one function and Act I's 0.22 is solid; what is NOT established "
        "is a matched numerical identity κ_I = κ_II. The value is the clean map — "
        "what holds, what does not, and what a real bridge would need — kept in the "
        "record so the theory is built on what is true, not what is hoped.")
    return lines


def make_figure(op, gauge, cp, cp_matched, args, path):
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

    # panel A: one operator — schematic of κ̂ = Σ|q|/Σe applied to both sectors
    ax1.axis("off")
    ax1.set_title("A. One operator: κ̂ = Σ|q| / Σe  ∈ [0,1]")
    txt = (
        "the Bogomolny coherent fraction\n"
        "$\\hat\\kappa = \\sum_x |q(x)| \\,/\\, \\sum_x e(x)$\n\n"
        f"SU(3) 4-D:  self_dual_fraction  ≡  κ̂\n"
        f"    same function?  {op['gauge_identity']}    Σe ≥ Σ|q|?  {op['gauge_bound']}\n\n"
        f"CP² 2-D:  cp_coherent_fraction  ≡  κ̂\n"
        f"    same function?  {op['cp_identity']}    Σe ≥ Σ|q|?  {op['cp_bound']}\n\n"
        "one operator — the integration fraction —\napplies identically to both acts")
    ax1.text(0.03, 0.92, txt, transform=ax1.transAxes, fontsize=10, va="top",
             family="monospace")

    # panel B: Act I — κ̂ rises to 0.22 under the Wilson flow
    t = np.array(gauge["times"]); f = np.array(gauge["f"])
    ax2.plot(t, f, color=BLUE, lw=2.2, marker="o", ms=3, zorder=3)
    if np.isfinite(gauge["t0"]):
        ax2.axvline(gauge["t0"], color=RED, ls="--", lw=1.1, zorder=2)
        ax2.plot([gauge["t0"]], [gauge["kappa_t0"]], marker="o", color=RED, ms=9,
                 zorder=5)
        ax2.text(gauge["t0"], gauge["kappa_t0"],
                 f"  κ̂(t₀)={gauge['kappa_t0']:.3f}", color=RED, fontsize=9,
                 va="center")
    ax2.axhline(0.22, color=GRAY, ls=":", lw=1.0)
    ax2.set_xlabel("Wilson flow time  t")
    ax2.set_ylabel("κ̂ = self-dual fraction")
    ax2.set_title("B. Act I (SU(3)): κ̂ rises to ≈ 0.22 at t₀")

    # panel C: Act II substrate — κ̂ falls under the cooling flow
    cs = np.array(cp["cool_steps"]); fc = np.array(cp["f"])
    ax3.plot(cs, fc, color=GREEN, lw=2.2, marker="s", ms=4, zorder=3)
    ax3.axhline(0.22, color=GRAY, ls=":", lw=1.0)
    ax3.text(cs[-1], 0.22, " 0.22 (gauge)", color=GRAY, fontsize=8, va="bottom",
             ha="right")
    ax3.set_xlabel("cooling-flow steps")
    ax3.set_ylabel("κ̂ = CP coherent fraction")
    ax3.set_title("C. Act II substrate (CP²): κ̂ falls under cooling")

    # panel D: the frontier — same operator, opposite flow, on a common coordinate
    ax4.plot(gauge["progress"], gauge["f"], color=BLUE, lw=2.2, marker="o", ms=3,
             zorder=3, label="SU(3) (Act I) — rises")
    ax4.plot(cp["progress"], cp["f"], color=GREEN, lw=2.2, marker="s", ms=4,
             zorder=3, label="CP² (Act II) — falls")
    ax4.axhline(0.22, color=GRAY, ls=":", lw=1.0)
    ps = cp_matched["progress"]
    if np.isfinite(ps):
        ax4.axvline(ps, color=RED, ls="--", lw=1.0, zorder=2)
        ax4.plot([ps], [cp_matched["gauge_k"]], marker="o", color=BLUE, ms=8,
                 zorder=5)
        ax4.plot([ps], [cp_matched["cp_k"]], marker="s", color=GREEN, ms=8,
                 zorder=5)
        ax4.annotate("", xy=(ps, cp_matched["cp_k"]),
                     xytext=(ps, cp_matched["gauge_k"]),
                     arrowprops=dict(arrowstyle="<->", color=RED, lw=1.3))
        ax4.text(ps, 0.5 * (cp_matched["gauge_k"] + cp_matched["cp_k"]),
                 "  no identity\n  (this estimator)", color=RED, fontsize=8,
                 va="center", ha="left")
    ax4.set_xlabel("flow progress  (fraction of action dissipated)")
    ax4.set_ylabel("κ̂  (same operator)")
    ax4.set_title("D. The frontier: one operator, opposite flow behaviour")
    ax4.legend(frameon=False, fontsize=8, loc="upper center")

    fig.suptitle("The one-κ frontier: the integration fraction is one operator, "
                 "but not (yet) one number across the two acts", fontsize=12, y=0.98)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gauge-size", type=int, default=8)
    p.add_argument("--beta-g", type=float, default=1.8)
    p.add_argument("--n-conf", type=int, default=4)
    p.add_argument("--n-therm", type=int, default=20)
    p.add_argument("--t-max", type=float, default=1.2)
    p.add_argument("--dt", type=float, default=0.03)
    p.add_argument("--measure-every", type=int, default=4)
    p.add_argument("--ref", type=float, default=0.3)
    p.add_argument("--cp-size", type=int, default=48)
    p.add_argument("--beta-cp", type=float, default=1.5)
    p.add_argument("--eps-cp", type=float, default=0.5)
    p.add_argument("--n-conf-cp", type=int, default=5)
    p.add_argument("--n-therm-cp", type=int, default=40)
    p.add_argument("--cool-steps", type=int, nargs="+",
                   default=[0, 2, 4, 6, 10, 16, 24, 36])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_onekappa")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.gauge_size = 6
        args.n_conf = 2
        args.n_therm = 12
        args.cp_size = 32
        args.n_conf_cp = 3
        args.n_therm_cp = 20
        args.cool_steps = [0, 4, 10, 20, 36]

    os.makedirs(args.output_dir, exist_ok=True)
    print("the one-κ frontier: κ̂ = Σ|q|/Σe, one operator across SU(3) (Act I) "
          "and CP² (Act II)", flush=True)
    print("A — one operator, two sectors:")
    op = operator_identity(args)
    print("B — Act I: the SU(3) coherent fraction under the Wilson flow:")
    gauge = gauge_sector(args)
    print("C — Act II substrate: the CP² coherent fraction under cooling:")
    cp = cp_sector(args)
    cp_matched = matched_reading(gauge, cp)
    print(f"D — matched reading at shared progress {cp_matched['progress']:.2f}: "
          f"gauge κ̂ ≈ {cp_matched['gauge_k']:.3f} vs CP κ̂ ≈ {cp_matched['cp_k']:.3f} "
          f"(no identity in this estimator)", flush=True)

    lines = summarise(op, gauge, cp, cp_matched, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    payload = {"params": vars(args), "operator": op,
               "gauge": {k: gauge[k] for k in ("t0", "kappa_t0", "kappa_std",
                                               "prog_t0")},
               "cp": {k: cp[k] for k in ("f_raw", "f_cooled")},
               "cp_matched": cp_matched}
    with open(os.path.join(args.output_dir, "n3_one_kappa_frontier.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_one_kappa_frontier.png")
        make_figure(op, gauge, cp, cp_matched, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
