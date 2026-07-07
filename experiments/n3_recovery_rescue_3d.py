"""Can faster recovery rescue 3-D memory connectivity — and at what cost?

``n3_memory_clusters_3d.py`` found that in 3-D the memory backbone
*de-percolates* at criticality (spanning probability → 0.15) where the 2-D
one merely bends (≥ 0.98), because the denser 3-D critical structure drains
so much capacity that the fertile fraction craters below even the lower 3-D
percolation threshold.  That was measured at one recovery rate (r = 0.8).

But recovery is a dial, and steady-state capacity is ``κ = r/(r + c·load)``
— so a high enough r must lift κ back above threshold everywhere and
reconnect the soil, in any dimension.  The questions are therefore
quantitative:

1. **Does faster recovery rescue 3-D connectivity?**  Scanning r, does the
   worst-case (over the critical temperature window) spanning probability
   climb back through ½ — the backbone re-percolating?
2. **How much more does 3-D need than 2-D?**  In 2-D r = 0.8 already held
   the backbone; 3-D at r = 0.8 broke.  Running the same r-scan in both
   dimensions locates the reconnection threshold r\* where each first keeps
   the worst-case spanning ≥ ½, and the ratio r\*₃ / r\*₂ measures the extra
   capacity-healing a third dimension demands to hold memory together.

For each (dimension, r) the fertile mask is scored over a temperature
window straddling the melt, and the *worst* point (minimum spanning
probability, the hardest place to stay connected) is kept — the recovery
rate must rescue the tightest bottleneck, not the average.

Usage::

    python experiments/n3_recovery_rescue_3d.py --output-dir artifacts/n3_rescue_3d
    python experiments/n3_recovery_rescue_3d.py --quick    # smoke run
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

from project_genesis.annealed_matter import joint_sweep, sector_amplitudes
from project_genesis.soil_clusters import (
    SITE_PERCOLATION_PC,
    SITE_PERCOLATION_PC_3D,
    cluster_report,
    label_periodic,
)

from n3_memory_clusters_3d import equilibrate_dim  # noqa: E402
from n3_potts_transition import potts_magnetisation  # noqa: E402


def measure_point(*, dim, size, temperature, recovery, args, seed,
                  keep_slice=False) -> dict:
    """Snapshot-averaged fertile-mask observables at one (dim, T, r)."""
    psi, links, kappa, rng, step = equilibrate_dim(
        dim=dim, size=size, temperature=temperature, beta_g=args.beta_g,
        matter_coupling=args.matter_coupling, quartic_u=args.quartic_u,
        consumption=args.consumption, recovery=recovery,
        n_therm=args.n_therm, seed=seed)
    kp = dict(consumption=args.consumption, recovery=recovery)
    spans = np.empty(args.n_meas)
    pinf = np.empty(args.n_meas)
    fert = np.empty(args.n_meas)
    mag = np.empty(args.n_meas)
    slice_labels = None
    for i in range(args.n_meas):
        for _ in range(args.n_skip):
            psi, links, kappa, _ = joint_sweep(
                psi, links, rng, temperature=temperature, beta_g=args.beta_g,
                matter_coupling=args.matter_coupling, quartic_u=args.quartic_u,
                frac_penalty=0.0, step_scale=step, kappa=kappa, kappa_params=kp)
        rep = cluster_report(kappa >= args.threshold)
        spans[i] = 1.0 if rep["spans"] else 0.0
        pinf[i] = rep["p_inf"]
        fert[i] = rep["fertile_fraction"]
        mag[i] = potts_magnetisation(sector_amplitudes(psi))
        if keep_slice and i == args.n_meas - 1 and dim == 3:
            zc = size // 2
            slice_labels, _ = label_periodic((kappa >= args.threshold)[..., zc])
    return {"dim": dim, "temperature": temperature, "recovery": recovery,
            "spanning_prob": float(spans.mean()), "p_inf": float(pinf.mean()),
            "fertile_fraction": float(fert.mean()),
            "magnetisation": float(mag.mean()),
            "slice_labels": slice_labels.tolist() if slice_labels is not None else None}


def worst_over_window(*, dim, size, recovery, window, args, seed,
                      slice_temp=None) -> dict:
    """Scan the critical T window at fixed r; keep the worst (min-spanning) point."""
    pts = []
    for ti, t in enumerate(window):
        pt = measure_point(dim=dim, size=size, temperature=t, recovery=recovery,
                           args=args, seed=seed + 17 * ti,
                           keep_slice=(slice_temp is not None and
                                       abs(t - slice_temp) < 1e-9))
        pts.append(pt)
    worst = min(pts, key=lambda p: p["spanning_prob"])
    slice_labels = next((p["slice_labels"] for p in pts
                         if p["slice_labels"] is not None), None)
    return {"dim": dim, "recovery": recovery,
            "min_spanning": worst["spanning_prob"],
            "min_spanning_T": worst["temperature"],
            "min_p_inf": min(p["p_inf"] for p in pts),
            "min_fertile": min(p["fertile_fraction"] for p in pts),
            "slice_labels": slice_labels,
            "window": [{k: v for k, v in p.items() if k != "slice_labels"}
                       for p in pts]}


def reconnection_threshold(rows, level=0.5):
    """Interpolated recovery rate where min-spanning first rises through ``level``.

    ``rows`` is the per-r list (ascending recovery).  Returns the first
    upward crossing of ``level``, ``rows[0]['recovery']`` if already at/above
    it at the smallest r, or ``None`` if it never reaches ``level``.
    """
    if rows and rows[0]["min_spanning"] >= level:
        return float(rows[0]["recovery"])
    for a, b in zip(rows, rows[1:]):
        sa, sb = a["min_spanning"], b["min_spanning"]
        if sa < level <= sb and sb != sa:
            return float(a["recovery"] + (b["recovery"] - a["recovery"]) *
                         (level - sa) / (sb - sa))
    return None


def summarise(by_dim, r_star) -> list[str]:
    lines = ["can recovery rescue 3-D memory connectivity? — verdict",
             "=" * 54, ""]
    for dim in (2, 3):
        rows = by_dim[dim]
        lo, hi = rows[0], rows[-1]
        rs = r_star[dim]
        rs_str = "not within scan" if rs is None else f"r* ≈ {rs:.2f}"
        lines.append(
            f"{dim}-D: worst-case spanning {lo['min_spanning']:.2f} "
            f"(r={lo['recovery']:g}) → {hi['min_spanning']:.2f} (r={hi['recovery']:g}); "
            f"reconnection threshold {rs_str}")
    lines.append("")
    r2, r3 = r_star[2], r_star[3]
    if r2 and r3 and r3 > r2:
        lines.append(
            f"faster recovery DOES rescue 3-D connectivity: the worst-case "
            f"spanning probability climbs back through ½ once r ≈ {r3:.2f}. But "
            f"3-D needs markedly more healing than 2-D (r* ≈ {r3:.2f} vs "
            f"{r2:.2f}, a factor of ~{r3 / r2:.1f}×): the extra dimension's "
            "denser critical structure drains capacity faster, so it takes a "
            "correspondingly faster refill to hold the backbone together.")
    elif r3 and (r2 is None):
        lines.append(
            f"3-D reconnects at r* ≈ {r3:.2f}, while 2-D holds across the entire "
            "scanned range (already connected at the slowest rate) — 3-D needs a "
            "finite, higher recovery to achieve what 2-D has for free.")
    elif r3 is None:
        lines.append(
            "even the fastest scanned recovery does not lift the worst-case 3-D "
            "spanning back through ½ — 3-D connectivity needs more healing than "
            "this range provides (raise --recoveries to locate r*).")
    else:
        lines.append(
            f"both dimensions reconnect within the scan (r* 2-D ≈ {r2}, 3-D ≈ {r3}).")
    lines.append(
        "honest scope: worst-case over a fixed critical-T window, capacity-gated "
        "fertile mask (κ ≥ threshold); single sizes (2-D L², 3-D L³), one "
        "consumption/coupling point; r* is interpolated on a coarse r grid.")
    return lines


def make_figure(by_dim, r_star, montage, montage_rs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    C2, C3, GRAY, PURPLE = "#0072B2", "#D55E00", "#999999", "#8551A6"
    fig = plt.figure(figsize=(15, 8.4))
    gs = GridSpec(2, 6, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[0, 3:6])
    for ax in (ax1, ax2):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # panel 1: worst-case spanning probability vs r, both dims
    for dim, col in ((2, C2), (3, C3)):
        rows = by_dim[dim]
        rs = [r["recovery"] for r in rows]
        ax1.plot(rs, [r["min_spanning"] for r in rows], color=col, marker="o",
                 ms=6, lw=1.9, label=f"{dim}-D", zorder=3)
        if r_star[dim] is not None:
            ax1.axvline(r_star[dim], color=col, lw=1.0, ls="--", zorder=1)
    ax1.axhline(0.5, color=GRAY, lw=0.9, ls=":", zorder=1,
                label="percolation (spanning = ½)")
    ax1.set_xscale("log")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_xlabel("capacity recovery rate r")
    ax1.set_ylabel("worst-case spanning probability")
    ax1.set_title("Does recovery reconnect the backbone?")
    ax1.legend(frameon=False, fontsize=8, loc="center right")

    # panel 2: worst-case fertile fraction vs r, with p_c lines
    for dim, col in ((2, C2), (3, C3)):
        rows = by_dim[dim]
        rs = [r["recovery"] for r in rows]
        ax2.plot(rs, [r["min_fertile"] for r in rows], color=col, marker="s",
                 ms=6, lw=1.9, label=f"min fertile ({dim}-D)", zorder=3)
    ax2.axhline(SITE_PERCOLATION_PC, color=C2, lw=1.0, ls=":", zorder=1,
                label=f"p_c 2-D = {SITE_PERCOLATION_PC:g}")
    ax2.axhline(SITE_PERCOLATION_PC_3D, color=C3, lw=1.0, ls=":", zorder=1,
                label=f"p_c 3-D = {SITE_PERCOLATION_PC_3D:g}")
    ax2.set_xscale("log")
    ax2.set_xlabel("capacity recovery rate r")
    ax2.set_ylabel("worst-case fertile fraction")
    ax2.set_title("Recovery refills the capacity trough")
    ax2.legend(frameon=False, fontsize=8)

    # bottom: 3-D reconnection montage — z-slices at increasing r
    for j, (rval, terr) in enumerate(zip(montage_rs, montage)):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        if terr is None or terr.get("slice_labels") is None:
            ax.axis("off")
            continue
        labels = np.array(terr["slice_labels"])
        n = int(labels.max())
        rng = np.random.default_rng(0)
        palette = np.zeros((n + 1, 3))
        palette[0] = (1, 1, 1)
        if n >= 1:
            sizes = np.bincount(labels.ravel())[1:]
            biggest = int(sizes.argmax()) + 1
            for lab in range(1, n + 1):
                palette[lab] = (rng.uniform(0.55, 0.9), rng.uniform(0.55, 0.9),
                                rng.uniform(0.55, 0.9))
            palette[biggest] = (0.0, 0.45, 0.70)
        ax.imshow(palette[labels], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        span = terr["min_spanning"]
        tag = "reconnected ✓" if span >= 0.5 else "fragmented"
        ax.set_title(f"3-D z-slice, r={rval:g}  (min span {span:.2f}, {tag})",
                     fontsize=9)
    fig.suptitle("Recovery rescues 3-D memory connectivity — but 3-D needs a "
                 "faster refill than 2-D", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size2d", type=int, default=32)
    p.add_argument("--size3d", type=int, default=14)
    p.add_argument("--recoveries", type=str, default="0.3,0.5,0.7,1.0,1.4,2.4")
    p.add_argument("--window", type=str, default="0.10,0.13,0.16,0.20",
                   help="critical-T window to take the worst case over")
    p.add_argument("--consumption", type=float, default=5.0)
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--montage-recoveries", type=str, default="0.3,1.0,2.4")
    p.add_argument("--montage-temp", type=float, default=0.13)
    p.add_argument("--n-therm", type=int, default=450)
    p.add_argument("--n-meas", type=int, default=40)
    p.add_argument("--n-skip", type=int, default=3)
    p.add_argument("--seed", type=int, default=61)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_rescue_3d")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size2d, args.size3d = 20, 10
        args.recoveries = "0.4,1.6,6.4"
        args.window = "0.12,0.16"
        args.montage_recoveries = "0.4,6.4"
        args.montage_temp = 0.16
        args.n_therm, args.n_meas = 200, 30

    recoveries = [float(r) for r in args.recoveries.split(",") if r.strip()]
    window = [float(t) for t in args.window.split(",") if t.strip()]
    montage_rs = [float(r) for r in args.montage_recoveries.split(",") if r.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    by_dim = {}
    for dim in (2, 3):
        size = args.size2d if dim == 2 else args.size3d
        print(f"{dim}-D recovery-rescue scan (L={size}), worst over T∈{window}",
              flush=True)
        rows = []
        for ri, r in enumerate(recoveries):
            slice_t = args.montage_temp if (dim == 3 and r in montage_rs) else None
            row = worst_over_window(dim=dim, size=size, recovery=r, window=window,
                                    args=args, seed=args.seed + 1000 * dim + 100 * ri,
                                    slice_temp=slice_t)
            rows.append(row)
            print(f"  r={r:g}: min spanning={row['min_spanning']:.2f} "
                  f"(T={row['min_spanning_T']:g}), min P∞={row['min_p_inf']:.2f}, "
                  f"min fertile={row['min_fertile']:.2f}", flush=True)
        by_dim[dim] = rows

    r_star = {dim: reconnection_threshold(by_dim[dim]) for dim in (2, 3)}
    lines = summarise(by_dim, r_star)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": vars(args), "r_star": r_star,
               "by_dim": {str(d): [{k: v for k, v in r.items()
                                    if k != "slice_labels"} for r in by_dim[d]]
                          for d in (2, 3)}}
    with open(os.path.join(args.output_dir, "n3_recovery_rescue_3d.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        by_r3 = {r["recovery"]: r for r in by_dim[3]}
        montage = [by_r3.get(rv) for rv in montage_rs]
        fig_path = os.path.join(args.output_dir, "n3_recovery_rescue_3d.png")
        make_figure(by_dim, r_star, montage, montage_rs, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
