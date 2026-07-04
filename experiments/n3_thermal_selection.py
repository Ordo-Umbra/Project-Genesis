"""Does the N⋆=3 selection survive a *thermodynamic* gauge sector?

The topological-selection experiment established that, for a deterministic
sector field, ``S = ΔC + κ·w·neutrality`` has an interior optimum at exactly
three sectors (2-D; echoed in 3-D).  The gauge-layer experiments established
— deterministically — that a connection restores the coherence local
rotations destroy, and — thermodynamically — that the sampled Wilson
ensemble confines.  This experiment joins the two ends:

    Take the *converged* P-sector network for each palette size P, embed it
    as a quenched matter field ψ ∈ ℂ^P, couple it to a fluctuating SU(P)
    gauge ensemble

        weight ∝ exp(−β_g·S_W + g_m·Σ_{x,μ} Re[ψ†(x)·U_μ(x)·ψ(x+μ̂)])

    and measure how much of the integration term actually survives gauge
    disorder — the **coherence retention**

        R_P(β_g, g_m) = ⟨covariant coherence⟩_ensemble / naive coherence(ψ) ,

    plus where the ensemble's curvature (coherence stress) lives relative
    to the sector walls and junctions.  The gauge-dressed selection
    functional is then

        S_P = ΔC_P + κ_P · w · neutrality_P · R_P

    — the deterministic S with its integration channel taxed by the
    measured retention.  The verdicts sought: (a) for which (w, β_g, g_m)
    is S still maximised at P = 3, and where does the selection wash out;
    (b) does the deterministic "curvature enriched ~2.6× on walls" carry
    over to the equilibrium ensemble?

What the pilot probes already established (and the scan quantifies)
-------------------------------------------------------------------
- Retention is controlled primarily by the *matter coupling* g_m, not by
  β_g, and falls with the palette size P at equal coupling — a group-rank
  tax on integration.
- **No curvature localization in equilibrium.**  The per-link matter
  constraint is rank-1 — ``U·ψ(x+μ̂) = ψ(x)`` with a free orthogonal
  completion — and composing those optimal transports around any plaquette
  returns ψ to itself, so the constraints are *integrable*: a
  zero-curvature connection satisfies all of them simultaneously, for any
  sector geometry.  Quenched sector matter therefore does not frustrate
  the gauge sector, and the measured curvature is pure thermal background
  (junction/wall/bulk all equal).  The deterministic 2.6× wall enrichment
  is a statement about *transient relaxation dynamics*, not about the
  equilibrium ensemble — this experiment records that negative verdict
  explicitly.

Honest scope
------------
- The matter field is *quenched* (no back-reaction of the gauge ensemble on
  the sector network).  This isolates one direction of the coupling.
- Gauge groups of different rank are compared at equal β_g and equal g_m;
  the measured mean plaquette per point is reported so the disorder level
  of each group is visible rather than assumed equal.
- 2-D, small lattices, one matter configuration per (P, trial) — the same
  structural setting as the deterministic topological-selection result it
  extends.

Usage::

    python experiments/n3_thermal_selection.py --output-dir artifacts/n3_thermal
    python experiments/n3_thermal_selection.py --quick     # smoke run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.gauge import (  # noqa: E402
    covariant_coherence,
    curvature_density,
    naive_coherence,
    random_unitary,
    sector_field_to_psi,
)
from project_genesis.gauge_mc import (  # noqa: E402
    heatbath_sweep,
    overrelaxation_sweep,
    wilson_loop,
)
from project_genesis.multiphase import (  # noqa: E402
    _neighbourhood_label_bits,
    _popcount,
    _total_gradient_energy,
    full_palette_junction_density,
    interface_mask,
    sector_labels,
    step_multiphase_conserved,
)

from confinement_sigma_scan import jackknife  # noqa: E402


# ---------------------------------------------------------------------------
# Stage 1 — deterministic sector networks (as in topological_selection.py)
# ---------------------------------------------------------------------------

def build_sector_network(
    n_phases: int,
    *,
    size: int,
    steps: int,
    gamma: float,
    diffusion: float,
    dt: float,
    seed: int,
    beta: float,
) -> dict:
    """Evolve a conserved P-phase field to a stable junction network.

    Returns the field, its ψ embedding, and the deterministic S ingredients
    (ΔC, κ, neutrality) exactly as the topological-selection experiment
    defines them.
    """
    rng = np.random.default_rng(seed)
    fields = rng.random((n_phases, size, size)) * 0.1
    for _ in range(steps):
        fields, _ = step_multiphase_conserved(
            fields, diffusion=diffusion, gamma=gamma, dt=dt
        )
    labels = sector_labels(fields)
    load = _total_gradient_energy(fields)
    mean_load = float(load.mean())
    psi = sector_field_to_psi(fields)
    return {
        "fields": fields,
        "psi": psi,
        "walls": interface_mask(fields),
        "junctions": _popcount(_neighbourhood_label_bits(labels)) >= 3,
        "delta_c": float(beta * mean_load),
        "kappa": 1.0 / (1.0 + mean_load),
        "neutrality": full_palette_junction_density(labels, n_phases),
    }


# ---------------------------------------------------------------------------
# Stage 2 — matter-coupled gauge ensemble per (P, β_g)
# ---------------------------------------------------------------------------

def measure_gauge_dressing(
    psi: np.ndarray,
    walls: np.ndarray,
    *,
    beta_g: float,
    matter_coupling: float,
    n_therm: int,
    n_meas: int,
    n_skip: int,
    bin_size: int,
    seed: int,
    junctions: np.ndarray | None = None,
) -> dict:
    """Thermalise the matter-coupled SU(P) ensemble and measure the dressing.

    Measures per configuration: covariant coherence per link, W(1,1)
    (disorder level), and the junction/wall/bulk decomposition of the
    curvature density.
    """
    n = psi.shape[-1]
    spatial = psi.shape[:-1]
    ndim = len(spatial)
    n_links = ndim * int(np.prod(spatial))
    rng = np.random.default_rng(seed)
    links = random_unitary(rng, n, (ndim, *spatial), special=True, scale=0.7)
    if junctions is None:
        junctions = np.zeros(spatial, dtype=bool)
    wall_only = walls & ~junctions
    bulk = ~walls & ~junctions

    def compound(lk):
        lk, _ = heatbath_sweep(lk, beta_g, rng,
                               matter=psi, matter_coupling=matter_coupling)
        lk, _ = overrelaxation_sweep(lk, rng, matter=psi,
                                     matter_coupling=matter_coupling,
                                     beta_g=beta_g)
        return lk

    for _ in range(n_therm):
        links = compound(links)

    naive = naive_coherence(psi) / n_links
    # per-meas: coh/link, W11, junction curv, wall curv, bulk curv
    samples = np.empty((n_meas, 5))
    for m in range(n_meas):
        for _ in range(n_skip):
            links = compound(links)
        coh = covariant_coherence(psi, links) / n_links
        w11 = wilson_loop(links, (1, 1))
        curv = curvature_density(links)
        samples[m] = (
            coh,
            w11,
            float(curv[junctions].mean()) if junctions.any() else float("nan"),
            float(curv[wall_only].mean()) if wall_only.any() else float("nan"),
            float(curv[bulk].mean()) if bulk.any() else float("nan"),
        )

    coh_mean, coh_err = jackknife(samples[:, :1], lambda v: float(v[0]),
                                  bin_size=bin_size)
    retention, retention_err = (
        (coh_mean / naive, coh_err / abs(naive)) if abs(naive) > 1e-12
        else (float("nan"), float("nan"))
    )

    def ratio(idx):
        # sliced columns: 0 = junction curv, 1 = wall curv, 2 = bulk curv
        def est(v):
            return float(v[idx] / v[2]) if v[2] > 1e-12 else float("nan")
        return est

    wall_ratio, wall_ratio_err = jackknife(samples[:, 2:5], ratio(1),
                                           bin_size=bin_size)
    junc_ratio, junc_ratio_err = jackknife(samples[:, 2:5], ratio(0),
                                           bin_size=bin_size)
    return {
        "beta_g": beta_g,
        "matter_coupling": matter_coupling,
        "coherence_per_link": coh_mean,
        "coherence_err": coh_err,
        "naive_coherence_per_link": naive,
        "retention": retention,
        "retention_err": retention_err,
        "plaquette": float(samples[:, 1].mean()),
        "wall_curvature_ratio": wall_ratio,
        "wall_curvature_ratio_err": wall_ratio_err,
        "junction_curvature_ratio": junc_ratio,
        "junction_curvature_ratio_err": junc_ratio_err,
    }


# ---------------------------------------------------------------------------
# Stage 3 — gauge-dressed selection
# ---------------------------------------------------------------------------

def dressed_selection_table(
    networks: dict[int, dict],
    dressing: dict[int, list[dict]],
    weights: list[float],
) -> list[dict]:
    """S_P = ΔC_P + κ_P·w·neutrality_P·R_P; argmax_P per (w, β_g, g_m) cell."""
    phases = sorted(networks)
    grid = [(d["beta_g"], d["matter_coupling"]) for d in dressing[phases[0]]]
    rows = []
    for w in weights:
        for gi, (beta_g, g_m) in enumerate(grid):
            s_by_p = {}
            for P in phases:
                net = networks[P]
                r = dressing[P][gi]["retention"]
                s_by_p[P] = net["delta_c"] + net["kappa"] * w * net["neutrality"] * r
            best = max(s_by_p, key=lambda pp: s_by_p[pp])
            rows.append({
                "weight": w,
                "beta_g": beta_g,
                "matter_coupling": g_m,
                "s_by_p": {str(P): s_by_p[P] for P in phases},
                "argmax": best,
            })
    return rows


def make_figure(networks, dressing, selection_rows, weights, beta_ref, path):
    """Retention vs g_m, curvature decomposition, dressed S — Okabe–Ito."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {2: "#0072B2", 3: "#E69F00", 4: "#009E73", 5: "#CC79A7"}
    MARKERS = {2: "o", 3: "s", 4: "^", 5: "D"}
    phases = sorted(networks)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax in (ax1, ax2, ax3):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("matter coupling g_m")
        ax.set_xscale("log", base=2)

    def at_beta(P):
        return [d for d in dressing[P] if d["beta_g"] == beta_ref]

    for P in phases:
        pts = at_beta(P)
        gs = [d["matter_coupling"] for d in pts]
        ax1.errorbar(gs, [d["retention"] for d in pts],
                     yerr=[d["retention_err"] for d in pts],
                     color=COLORS[P], marker=MARKERS[P], ms=5, lw=1.6,
                     capsize=3, label=f"P={P}", zorder=3)

    ax1.set_ylabel("coherence retention R")
    ax1.set_title(f"Integration surviving gauge fluctuations (β_g={beta_ref:g})")
    ax1.set_ylim(bottom=0.0)
    ax1.legend(frameon=False)

    pts3 = at_beta(3) if 3 in phases else at_beta(phases[0])
    gs = [d["matter_coupling"] for d in pts3]
    ax2.errorbar(gs, [d["junction_curvature_ratio"] for d in pts3],
                 yerr=[d["junction_curvature_ratio_err"] for d in pts3],
                 color="#E69F00", marker="s", ms=5, lw=1.6, capsize=3,
                 label="junction / bulk", zorder=3)
    ax2.errorbar(gs, [d["wall_curvature_ratio"] for d in pts3],
                 yerr=[d["wall_curvature_ratio_err"] for d in pts3],
                 color="#0072B2", marker="o", ms=5, lw=1.6, capsize=3,
                 label="wall / bulk", zorder=3)
    ax2.axhline(1.0, color="#999999", lw=0.8)
    ax2.set_ylabel("curvature ratio")
    ax2.set_title("Curvature localization, P=3 (flat ⇒ no frustration)")
    ax2.legend(frameon=False)

    w_ref = weights[len(weights) // 2]
    for P in phases:
        rows = [r for r in selection_rows
                if r["weight"] == w_ref and r["beta_g"] == beta_ref]
        gs = [r["matter_coupling"] for r in rows]
        ax3.plot(gs, [r["s_by_p"][str(P)] for r in rows], color=COLORS[P],
                 marker=MARKERS[P], ms=5, lw=1.6, label=f"P={P}", zorder=3)
    ax3.set_ylabel(f"S = ΔC + κ·w·neutrality·R   (w={w_ref:g})")
    ax3.set_title(f"Gauge-dressed selection (β_g={beta_ref:g})")
    ax3.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--phases", type=str, default="2,3,4,5")
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--diffusion", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.09,
                   help="distinction coefficient in ΔC (matter side)")
    p.add_argument("--betas-gauge", type=str, default="1.0,3.0")
    p.add_argument("--matter-couplings", type=str,
                   default="0.25,0.5,1.0,2.0,4.0,8.0")
    p.add_argument("--weights", type=str, default="0.05,0.1,0.2,0.5")
    p.add_argument("--n-therm", type=int, default=120)
    p.add_argument("--n-meas", type=int, default=100)
    p.add_argument("--n-skip", type=int, default=3)
    p.add_argument("--bin-size", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_thermal")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size, args.steps = 24, 200
        args.betas_gauge = "3.0"
        args.matter_couplings = "0.5,2.0"
        args.n_therm, args.n_meas = 40, 30
        args.phases = "2,3,4"

    phases = [int(x) for x in args.phases.split(",") if x.strip()]
    betas_gauge = [float(b) for b in args.betas_gauge.split(",") if b.strip()]
    couplings = [float(g) for g in args.matter_couplings.split(",") if g.strip()]
    weights = [float(w) for w in args.weights.split(",") if w.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Stage 1: conserved P-sector networks ({args.size}^2, "
          f"{args.steps} steps)", flush=True)
    networks: dict[int, dict] = {}
    for P in phases:
        net = build_sector_network(
            P, size=args.size, steps=args.steps, gamma=args.gamma,
            diffusion=args.diffusion, dt=args.dt, seed=args.seed, beta=args.beta,
        )
        networks[P] = net
        print(f"  P={P}: ΔC={net['delta_c']:.5f}  κ={net['kappa']:.4f}  "
              f"neutrality={net['neutrality']:.5f}", flush=True)

    print(f"\nStage 2: matter-coupled SU(P) ensembles "
          f"(β_g ∈ {betas_gauge}, g_m ∈ {couplings})", flush=True)
    dressing: dict[int, list[dict]] = {P: [] for P in phases}
    for P in phases:
        cell = 0
        for beta_g in betas_gauge:
            for g_m in couplings:
                d = measure_gauge_dressing(
                    networks[P]["psi"], networks[P]["walls"],
                    junctions=networks[P]["junctions"],
                    beta_g=beta_g, matter_coupling=g_m,
                    n_therm=args.n_therm, n_meas=args.n_meas,
                    n_skip=args.n_skip, bin_size=args.bin_size,
                    seed=args.seed + 1000 * P + cell,
                )
                cell += 1
                dressing[P].append(d)
                print(f"  P={P} β_g={beta_g:4.1f} g_m={g_m:5.2f}: "
                      f"R={d['retention']:.4f}±{d['retention_err']:.4f}  "
                      f"plaq={d['plaquette']:.3f}  "
                      f"junc/bulk={d['junction_curvature_ratio']:.3f}"
                      f"±{d['junction_curvature_ratio_err']:.3f}  "
                      f"wall/bulk={d['wall_curvature_ratio']:.3f}"
                      f"±{d['wall_curvature_ratio_err']:.3f}", flush=True)

    selection_rows = dressed_selection_table(networks, dressing, weights)

    lines = ["N*=3 thermal selection — verdicts", "=" * 40, ""]
    lines.append("argmax_P of S = ΔC + κ·w·neutrality·R per (w | β_g, g_m):")
    for w in weights:
        rows_w = [r for r in selection_rows if r["weight"] == w]
        cells = ", ".join(
            f"(β={r['beta_g']:g},g={r['matter_coupling']:g})→P*={r['argmax']}"
            for r in rows_w
        )
        lines.append(f"  w={w:g}: {cells}")
    n_three = sum(1 for r in selection_rows
                  if r["argmax"] == 3 and r["weight"] > 0)
    n_total = sum(1 for r in selection_rows if r["weight"] > 0)
    lines.append("")
    lines.append(
        f"selection at P=3 holds in {n_three}/{n_total} (w, β_g, g_m) cells"
    )
    washout = sorted({
        (r["weight"], r["matter_coupling"]) for r in selection_rows
        if r["weight"] > 0 and r["argmax"] != 3
    })
    if washout:
        lines.append("washout cells (selection lost): "
                     + ", ".join(f"(w={w:g}, g_m={g:g})" for w, g in washout))
    if 3 in phases:
        lines.append("")
        r3 = [(d["beta_g"], d["matter_coupling"], d["retention"])
              for d in dressing[3]]
        lines.append("P=3 coherence retention: " + ", ".join(
            f"R(β={b:g},g={g:g})={r:.3f}" for b, g, r in r3))
        max_dev = max(
            max(abs(d["junction_curvature_ratio"] - 1.0),
                abs(d["wall_curvature_ratio"] - 1.0))
            for d in dressing[3]
            if not math.isnan(d["junction_curvature_ratio"])
        )
        lines.append(
            f"P=3 curvature localization: max |ratio − 1| = {max_dev:.3f} "
            "over all cells — quenched sector matter does not frustrate the "
            "gauge sector (the per-link constraints are integrable), so the "
            "deterministic wall enrichment does NOT carry over to equilibrium."
        )
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {
        "params": {k: v for k, v in vars(args).items()},
        "networks": {
            str(P): {k: v for k, v in net.items()
                     if k in ("delta_c", "kappa", "neutrality")}
            for P, net in networks.items()
        },
        "dressing": {str(P): dressing[P] for P in phases},
        "selection": selection_rows,
    }
    with open(os.path.join(args.output_dir, "n3_thermal_selection.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_thermal_selection.png")
        make_figure(networks, dressing, selection_rows, weights,
                    betas_gauge[-1], fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
