"""Competing memories: two seeds, one capacity budget.

Every experiment so far has followed a *single* recalled memory.  But the
κ-as-soil rule has a consequence the moment a *second* memory wants the same
ground: rooting **consumes** capacity (``κ *= 1 − cost``), so the first seed
to root in a region draws the soil down and can *lock out* a rival — unless
the capacity heals first.  Whether a later memory can be written over an
earlier one is therefore decided by the same dial that rescued recall: the
recovery rate ``r``.

This is the write mechanism, isolated.  A recalled seed roots only where
local κ ≥ threshold (the engine's own criterion); we do not model the
subsequent coherence competition (which is confounded by the ambient
domain — see the seed-rooting honest scope).  The clean, mechanistic
question is the **capacity gate**: after memory A is written at a site, is
there enough capacity left to write memory B there, and how fast must the
soil heal for the answer to flip from *no* to *yes*?

Three measurements at fixed consumption, scanning the recovery rate:

1. **The persistence↔plasticity crossover.**  Plant seed A, let the soil
   recover for a fixed delay, then attempt rival seed B at the same site:
   the overwrite probability P(B roots | A rooted) vs r.  At slow recovery
   the first mover locks the region (write-once memory); at fast recovery
   the ground heals and B overwrites (plastic memory).
2. **Rewrite capacity.**  How many memories a single site can accept in
   succession before the soil is exhausted — the number of consecutive
   roots vs r.
3. **Territory.**  A wave of A across the lattice, then a wave of B: the
   last-writer map at low vs high r — a frozen first-mover mosaic giving
   way to a plastic last-mover one.

Usage::

    python experiments/n3_memory_competition.py --output-dir artifacts/n3_compete
    python experiments/n3_memory_competition.py --quick    # smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.annealed_matter import joint_sweep
from project_genesis.sector_seeds import (
    DEFAULT_KAPPA_THRESHOLD,
    local_kappa_mean,
    make_sector_seed,
    plant_sector_seed,
)

from n3_seed_rooting import equilibrate  # noqa: E402


def recover(psi, links, kappa, rng, args, r, n_steps, step):
    """Run ``n_steps`` joint sweeps at recovery rate ``r`` (the soil heals)."""
    kp = dict(consumption=args.consumption, recovery=r)
    for _ in range(n_steps):
        psi, links, kappa, _ = joint_sweep(
            psi, links, rng, temperature=args.temperature, beta_g=args.beta_g,
            matter_coupling=args.matter_coupling, quartic_u=args.quartic_u,
            frac_penalty=0.0, step_scale=step, kappa=kappa, kappa_params=kp)
    return psi, links, kappa


def overwrite_crossover(args, r, seed) -> dict:
    """P(rival B roots | A rooted) after a fixed recovery delay, at rate r."""
    psi, links, kappa, rng, step = equilibrate(
        size=args.size, temperature=args.temperature, beta_g=args.beta_g,
        matter_coupling=args.matter_coupling, quartic_u=args.quartic_u,
        consumption=args.consumption, kappa_recovery=r,
        n_therm=args.n_therm, seed=seed)
    seed_a = make_sector_seed(3, args.seed_size, sector=1)
    seed_b = make_sector_seed(3, args.seed_size, sector=2)
    n_a = n_b = 0
    k_before_b = []
    for _ in range(args.n_trials):
        site = (int(rng.integers(0, args.size)), int(rng.integers(0, args.size)))
        psi, kappa, rooted_a, _ = plant_sector_seed(
            psi, kappa, seed_a, site, threshold=args.threshold)
        if not rooted_a:
            continue
        n_a += 1
        psi, links, kappa = recover(psi, links, kappa, rng, args, r,
                                    args.delay, step)
        k_before_b.append(local_kappa_mean(kappa, site, args.seed_size))
        psi, kappa, rooted_b, _ = plant_sector_seed(
            psi, kappa, seed_b, site, threshold=args.threshold)
        if rooted_b:
            n_b += 1
        # let the field re-thermalise a little between independent trials
        psi, links, kappa = recover(psi, links, kappa, rng, args, r,
                                    args.between, step)
    return {"recovery": r, "n_a": n_a, "n_b": n_b,
            "overwrite_prob": (n_b / n_a) if n_a else float("nan"),
            "kappa_before_b": float(np.mean(k_before_b)) if k_before_b else float("nan")}


def rewrite_capacity(args, r, seed) -> dict:
    """Mean number of consecutive memories a site accepts before barren."""
    psi, links, kappa, rng, step = equilibrate(
        size=args.size, temperature=args.temperature, beta_g=args.beta_g,
        matter_coupling=args.matter_coupling, quartic_u=args.quartic_u,
        consumption=args.consumption, kappa_recovery=r,
        n_therm=args.n_therm, seed=seed + 5)
    seeds = [make_sector_seed(3, args.seed_size, sector=s) for s in (1, 2)]
    counts = []
    for _ in range(args.n_sites):
        site = (int(rng.integers(0, args.size)), int(rng.integers(0, args.size)))
        rooted = 0
        for w in range(args.max_writes):
            psi, kappa, ok, _ = plant_sector_seed(
                psi, kappa, seeds[w % 2], site, threshold=args.threshold)
            if not ok:
                break
            rooted += 1
            psi, links, kappa = recover(psi, links, kappa, rng, args, r,
                                        args.delay, step)
        counts.append(rooted)
    return {"recovery": r, "rewrite_capacity": float(np.mean(counts)),
            "rewrite_capacity_err": float(np.std(counts) / max(1, len(counts) ** 0.5)),
            "max_writes": args.max_writes}


def territory_map(args, r, seed) -> dict:
    """Two waves (A then B) across a tiling of sites; the last-writer map.

    Returns a 2-D int map (0 barren/none, 1 = A held, 2 = B overwrote) and
    the territory split.
    """
    psi, links, kappa, rng, step = equilibrate(
        size=args.size, temperature=args.temperature, beta_g=args.beta_g,
        matter_coupling=args.matter_coupling, quartic_u=args.quartic_u,
        consumption=args.consumption, kappa_recovery=r,
        n_therm=args.n_therm, seed=seed + 9)
    seed_a = make_sector_seed(3, args.seed_size, sector=1)
    seed_b = make_sector_seed(3, args.seed_size, sector=2)
    ssz = args.seed_size
    sites = [(y, x) for y in range(0, args.size, ssz)
             for x in range(0, args.size, ssz)]
    writer = np.zeros((args.size, args.size), dtype=int)

    def stamp(site, val):
        ys = (site[0] + np.arange(ssz)) % args.size
        xs = (site[1] + np.arange(ssz)) % args.size
        writer[np.ix_(ys, xs)] = val

    # wave 1: A everywhere it can root
    for site in sites:
        psi, kappa, ok, _ = plant_sector_seed(
            psi, kappa, seed_a, site, threshold=args.threshold)
        if ok:
            stamp(site, 1)
    psi, links, kappa = recover(psi, links, kappa, rng, args, r,
                                args.wave_delay, step)
    # wave 2: B tries to overwrite
    for site in sites:
        psi, kappa, ok, _ = plant_sector_seed(
            psi, kappa, seed_b, site, threshold=args.threshold)
        if ok:
            stamp(site, 2)
    n = writer.size
    return {"recovery": r, "writer_map": writer.tolist(),
            "a_frac": float(np.mean(writer == 1)),
            "b_frac": float(np.mean(writer == 2)),
            "barren_frac": float(np.mean(writer == 0))}


def crossover_rate(cross, level: float = 0.5):
    """Interpolated recovery rate where the overwrite probability passes ``level``.

    ``cross`` is the list of overwrite-crossover dicts (ascending recovery).
    Returns the first upward crossing of ``level`` by linear interpolation,
    or ``None`` if the curve never rises through it within the scan.
    """
    for a, b in zip(cross, cross[1:]):
        pa, pb = a["overwrite_prob"], b["overwrite_prob"]
        if pa < level <= pb and pb != pa:
            return float(a["recovery"] + (b["recovery"] - a["recovery"]) *
                         (level - pa) / (pb - pa))
    return None


def make_figure(cross, rewrite, territories, recall_band, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#E69F00", "#009E73", "#999999"
    fig = plt.figure(figsize=(15, 8.4))
    gs = GridSpec(2, 6, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[0, 3:6])
    for ax in (ax1, ax2):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    rs = [c["recovery"] for c in cross]
    # panel 1: overwrite crossover
    if recall_band:
        ax1.axvspan(recall_band[0], recall_band[1], color=GREEN, alpha=0.10,
                    zorder=0, label="recall-rescue r range")
    ax1.plot(rs, [c["overwrite_prob"] for c in cross], color=BLUE, marker="o",
             ms=6, lw=1.9, zorder=3, label="P(B roots | A rooted)")
    ax1.axhline(0.5, color=GRAY, lw=0.9, ls=":", zorder=1)
    ax1.set_xscale("log")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_xlabel("capacity recovery rate r")
    ax1.set_ylabel("overwrite probability")
    ax1.set_title("Persistence → plasticity: can a rival overwrite memory?")
    ax1.legend(frameon=False, fontsize=8, loc="center left")

    # panel 2: rewrite capacity
    ax2.errorbar([w["recovery"] for w in rewrite],
                 [w["rewrite_capacity"] for w in rewrite],
                 yerr=[w["rewrite_capacity_err"] for w in rewrite],
                 color=ORANGE, marker="D", ms=6, lw=1.9, capsize=3, zorder=3)
    ax2.axhline(1.0, color=GRAY, lw=0.9, ls=":", zorder=1,
                label="write-once (first mover only)")
    ax2.set_xscale("log")
    ax2.set_xlabel("capacity recovery rate r")
    ax2.set_ylabel("consecutive memories a site accepts")
    ax2.set_title("How many memories can one site hold in turn?")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")

    # bottom: last-writer territory montages
    cmap = ListedColormap([(1, 1, 1), (0.0, 0.45, 0.70), (0.90, 0.62, 0.0)])
    for j, terr in enumerate(territories[:3]):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        ax.imshow(np.array(terr["writer_map"]), cmap=cmap, vmin=0, vmax=2,
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"r={terr['recovery']:g}:  A held {terr['a_frac']:.0%} "
                     f"(blue) · B overwrote {terr['b_frac']:.0%} (orange)",
                     fontsize=9)
    fig.suptitle("Competing memories: the first mover locks out the second — "
                 "until capacity heals fast enough to let it in",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=40)
    p.add_argument("--recoveries", type=str, default="0.02,0.05,0.1,0.2,0.4,0.8")
    p.add_argument("--temperature", type=float, default=0.05,
                   help="a fertile (ordered) seedbed temperature")
    p.add_argument("--consumption", type=float, default=5.0)
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=DEFAULT_KAPPA_THRESHOLD)
    p.add_argument("--seed-size", type=int, default=4)
    p.add_argument("--delay", type=int, default=30,
                   help="recovery sweeps between writing A and attempting B")
    p.add_argument("--between", type=int, default=8,
                   help="recovery sweeps between independent crossover trials")
    p.add_argument("--wave-delay", type=int, default=40,
                   help="recovery sweeps between the A wave and the B wave")
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--n-sites", type=int, default=14)
    p.add_argument("--max-writes", type=int, default=6)
    p.add_argument("--territory-recoveries", type=str, default="0.02,0.1,0.8")
    p.add_argument("--recall-band", type=str, default="0.1,0.8",
                   help="r range where recall is rescued (for annotation)")
    p.add_argument("--n-therm", type=int, default=600)
    p.add_argument("--seed", type=int, default=53)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_compete")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size = 20
        args.recoveries = "0.02,0.2,0.8"
        args.territory_recoveries = "0.02,0.8"
        args.n_therm, args.n_trials, args.n_sites = 200, 16, 6
        args.delay, args.between, args.wave_delay = 15, 4, 20
        args.max_writes = 4

    recoveries = [float(r) for r in args.recoveries.split(",") if r.strip()]
    terr_rs = [float(r) for r in args.territory_recoveries.split(",") if r.strip()]
    recall_band = [float(x) for x in args.recall_band.split(",")] \
        if args.recall_band else None
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Competing memories at c={args.consumption:g}, T={args.temperature:g}, "
          f"L={args.size}, delay={args.delay}", flush=True)
    print("Measurement 1: overwrite crossover P(B roots | A rooted)", flush=True)
    cross = []
    for i, r in enumerate(recoveries):
        c = overwrite_crossover(args, r, seed=args.seed + 100 * i)
        cross.append(c)
        print(f"  r={r:g}: A rooted {c['n_a']}× → B overwrote {c['n_b']}× "
              f"(P={c['overwrite_prob']:.2f}); ⟨κ⟩ before B = "
              f"{c['kappa_before_b']:.3f}", flush=True)

    print("Measurement 2: rewrite capacity (consecutive memories per site)",
          flush=True)
    rewrite = []
    for i, r in enumerate(recoveries):
        w = rewrite_capacity(args, r, seed=args.seed + 100 * i)
        rewrite.append(w)
        print(f"  r={r:g}: {w['rewrite_capacity']:.2f} ± "
              f"{w['rewrite_capacity_err']:.2f} consecutive roots "
              f"(cap {w['max_writes']})", flush=True)

    print("Measurement 3: last-writer territory maps", flush=True)
    territories = []
    for i, r in enumerate(terr_rs):
        t = territory_map(args, r, seed=args.seed + 100 * i)
        territories.append(t)
        print(f"  r={r:g}: A held {t['a_frac']:.0%}, B overwrote "
              f"{t['b_frac']:.0%}, barren {t['barren_frac']:.0%}", flush=True)

    # verdict
    lines = ["competing memories — verdict", "=" * 28, ""]
    lo, hi = cross[0], cross[-1]
    lines.append(
        f"overwrite probability climbs from {lo['overwrite_prob']:.2f} "
        f"(r={lo['recovery']:g}) to {hi['overwrite_prob']:.2f} (r={hi['recovery']:g}): "
        "the recovery rate is the dial between write-once and plastic memory")
    # crossover r where overwrite passes ½
    cross_r = crossover_rate(cross, 0.5)
    if cross_r is not None:
        lines.append(
            f"the persistence→plasticity crossover (overwrite passes ½) sits at "
            f"r ≈ {cross_r:.3f}: below it the first memory locks the ground and "
            "a rival is turned away; above it the soil heals fast enough that a "
            "later memory writes over the earlier one")
    else:
        trend = ("never reaches ½ — memory stays write-once across the scanned "
                 "range" if hi["overwrite_prob"] < 0.5 else
                 "already above ½ at the slowest rate — memory is plastic "
                 "throughout the scanned range")
        lines.append(f"the overwrite probability {trend}")
    lines.append(
        f"rewrite capacity rises from {rewrite[0]['rewrite_capacity']:.2f} "
        f"(r={rewrite[0]['recovery']:g}) to {rewrite[-1]['rewrite_capacity']:.2f} "
        f"(r={rewrite[-1]['recovery']:g}) consecutive memories per site: faster "
        "healing lets a location hold a longer succession of overwritten memories")
    if len(territories) >= 2:
        t_lo, t_hi = territories[0], territories[-1]
        lines.append(
            f"territory: at slow recovery (r={t_lo['recovery']:g}) the first mover "
            f"holds {t_lo['a_frac']:.0%} of the ground and B claims only "
            f"{t_lo['b_frac']:.0%}; at fast recovery (r={t_hi['recovery']:g}) B "
            f"overwrites {t_hi['b_frac']:.0%} — a frozen first-mover mosaic gives "
            "way to a plastic last-mover one")
    lines.append("")
    lines.append(
        "honest scope: this is the capacity-gated WRITE mechanism (a rival roots "
        "only where κ has healed above threshold), not the subsequent coherence "
        "competition between domains, which is confounded by the ambient field "
        "and deliberately not claimed. One consumption/temperature/lattice point.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": vars(args), "crossover": cross, "rewrite": rewrite,
               "territories": [{k: v for k, v in t.items() if k != "writer_map"}
                               for t in territories]}
    with open(os.path.join(args.output_dir, "n3_memory_competition.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_memory_competition.png")
        make_figure(cross, rewrite, territories, recall_band, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
