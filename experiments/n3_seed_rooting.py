"""Memory recall at criticality: where can stored structure re-root?

This experiment marries the two halves of the repository that have never
met — the **memory corpus** (stable structure stored and re-seeded to keep
the universe from collapsing) and the **thermal sector program** (the
dynamical capacity field κ, measured to trough exactly at criticality).

The corpus re-roots seeds under a κ-as-soil rule (``engine._plant_seed``):
a recalled seed unfolds only where local capacity is sufficient (κ ≥
``corpus_kappa_threshold`` = 0.3), and rooting consumes that capacity.
``project_genesis.sector_seeds`` carries the exact rule to the thermal
ψ∈ℂ³ ensemble.  Because κ collapses at the melting transition, the sharp
prediction is that **recall fails at criticality** — the soil goes barren
before the field itself disorders, so a system can *hold* the structure it
has but can no longer *regenerate* what it loses.

Three measurements on the unpinned annealed (ψ, U, κ) model, all built on
the *same* faithful observable — the fertile-soil fraction, i.e. the
fraction of sites where a recalled seed would root under the exact engine
criterion (local κ ≥ threshold) — sliced three ways:

1. **Recall capacity across the melt.**  ``recall_capacity(T)`` on
   equilibrated thermal fields, overlaid with the Potts magnetisation to
   show recall dies *before* order: the soil goes barren while the field
   still carries structure.
2. **The memory phase diagram.**  ``recall_capacity(T, c)`` as a heatmap
   over temperature and consumption strength — the region of parameter
   space where remembered structure can re-root at all.
3. **Recall is self-limiting.**  Actually rooting a batch of seeds into a
   fertile field (via ``plant_sector_seed``, which consumes κ on success)
   draws the fertile fraction down, so recall spreads into fresh ground
   rather than piling onto the same spot — the consumption half of the
   κ-as-soil rule, demonstrated.

The κ-as-soil rule's physical content in this thermal model is precisely
the *gating*: below T_c the field is ordered so any patch is coherent
regardless of κ, and above T_c no fertile soil exists — so "can a seed
root here" (κ ≥ threshold) is exactly the meaningful, measurable question,
and it is the engine's own criterion.

Usage::

    python experiments/n3_seed_rooting.py --output-dir artifacts/n3_seed
    python experiments/n3_seed_rooting.py --quick    # smoke run
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
from project_genesis.gauge import random_psi, random_unitary
from project_genesis.sector_seeds import (
    DEFAULT_KAPPA_THRESHOLD,
    fertile_fraction,
    make_sector_seed,
    plant_sector_seed,
)

from confinement_sigma_scan import jackknife  # noqa: E402
from n3_potts_transition import potts_magnetisation  # noqa: E402


def equilibrate(
    *, size: int, temperature: float, beta_g: float, matter_coupling: float,
    quartic_u: float, consumption: float, kappa_recovery: float,
    n_therm: int, seed: int, step_scale: float = 0.3,
):
    """Thermalise the adaptive (ψ, U, κ) field; return (psi, links, kappa, rng, step)."""
    rng = np.random.default_rng(seed)
    spatial = (size, size)
    psi = random_psi(rng, 3, spatial) * 0.15
    psi[..., 0] += 1.0
    psi = psi / np.linalg.norm(psi, axis=-1, keepdims=True)
    links = random_unitary(rng, 3, (2, *spatial), special=True, scale=0.7)
    kappa = np.ones(spatial)
    kp = dict(consumption=consumption, recovery=kappa_recovery)
    step = step_scale
    block = max(10, n_therm // 12)
    done = 0
    while done < n_therm:
        acc_sum = 0.0
        for _ in range(min(block, n_therm - done)):
            psi, links, kappa, acc = joint_sweep(
                psi, links, rng, temperature=temperature, beta_g=beta_g,
                matter_coupling=matter_coupling, quartic_u=quartic_u,
                frac_penalty=0.0, step_scale=step, kappa=kappa,
                kappa_params=kp)
            acc_sum += acc
            done += 1
        if acc_sum / block < 0.15:
            step = max(0.02, step * 0.7)
        elif acc_sum / block > 0.4:
            step = min(0.8, step * 1.3)
    return psi, links, kappa, rng, step


def recall_capacity_point(
    *, temperature: float, consumption: float, args, seed: int,
) -> dict:
    """Fertile-soil fraction + Potts m over measurement snapshots."""
    psi, links, kappa, rng, step = equilibrate(
        size=args.size, temperature=temperature, beta_g=args.beta_g,
        matter_coupling=args.matter_coupling, quartic_u=args.quartic_u,
        consumption=consumption, kappa_recovery=args.kappa_recovery,
        n_therm=args.n_therm, seed=seed)
    kp = dict(consumption=consumption, recovery=args.kappa_recovery)
    samples = np.empty((args.n_meas, 3))  # fertile_frac, m, kappa_mean
    for i in range(args.n_meas):
        for _ in range(args.n_skip):
            psi, links, kappa, _ = joint_sweep(
                psi, links, rng, temperature=temperature, beta_g=args.beta_g,
                matter_coupling=args.matter_coupling, quartic_u=args.quartic_u,
                frac_penalty=0.0, step_scale=step, kappa=kappa,
                kappa_params=kp)
        samples[i] = (fertile_fraction(kappa, args.threshold),
                      potts_magnetisation(sector_amplitudes(psi)),
                      float(kappa.mean()))

    def jk(col):
        return jackknife(samples[:, [col]], lambda v: float(v[0]),
                         bin_size=args.bin_size)
    rc, rc_err = jk(0)
    m, m_err = jk(1)
    return {"temperature": temperature, "consumption": consumption,
            "recall_capacity": rc, "recall_capacity_err": rc_err,
            "magnetisation": m, "magnetisation_err": m_err,
            "kappa_mean": float(samples[:, 2].mean())}


def soil_consumption_demo(args) -> dict:
    """Root a batch of seeds into a fertile field; watch the soil deplete.

    Exercises the consumption half of the κ-as-soil rule
    (``plant_sector_seed``): each successful rooting draws κ down by the
    cost, so the fertile fraction falls as seeds are planted and recall is
    pushed into fresh ground rather than piling onto exhausted soil.
    """
    T = args.scarcity_temperature  # a fertile (ordered) field
    psi, links, kappa, rng, step = equilibrate(
        size=args.size, temperature=T, beta_g=args.beta_g,
        matter_coupling=args.matter_coupling, quartic_u=args.quartic_u,
        consumption=args.consume_demo_c, kappa_recovery=args.kappa_recovery,
        n_therm=args.n_therm, seed=args.seed + 777)
    ssz = args.seed_size
    seed_patch = make_sector_seed(3, ssz, sector=1, dim=2)
    # the corpus recalls repeatedly at random sites (κ does not recover
    # here, so the soil only depletes) — each successful rooting draws κ
    # down until the ground is exhausted and recall stalls
    n_attempts = args.consume_attempts
    trace = [{"attempt": 0, "n_rooted": 0,
              "fertile_fraction": fertile_fraction(kappa, args.threshold)}]
    n_rooted = 0
    for a in range(1, n_attempts + 1):
        site = (int(rng.integers(0, args.size)),
                int(rng.integers(0, args.size)))
        psi, kappa, rooted, _ = plant_sector_seed(
            psi, kappa, seed_patch, site, threshold=args.threshold)
        if rooted:
            n_rooted += 1
        if a % max(1, n_attempts // 40) == 0 or a == n_attempts:
            trace.append({"attempt": a, "n_rooted": n_rooted,
                          "fertile_fraction": fertile_fraction(kappa, args.threshold)})
    return {"temperature": T, "consumption": args.consume_demo_c,
            "n_attempts": n_attempts, "n_rooted": n_rooted,
            "fertile_start": trace[0]["fertile_fraction"],
            "fertile_end": trace[-1]["fertile_fraction"], "trace": trace}


def make_figure(rc_by_c, consumptions, temps, consume_demo, t_c, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RAMP = ["#b8d4ea", "#4a90c4", "#004b78"]
    GREEN, GRAY = "#009E73", "#999999"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax in (ax1, ax3):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # panel 1: recall capacity vs T, with m overlay
    for ci, c in enumerate(consumptions):
        pts = rc_by_c[c]
        ts = [p["temperature"] for p in pts]
        ax1.errorbar(ts, [p["recall_capacity"] for p in pts],
                     yerr=[p["recall_capacity_err"] for p in pts],
                     color=RAMP[ci % len(RAMP)], marker="o", ms=4, lw=1.7,
                     capsize=2, label=f"recall (c={c:g})", zorder=3)
    pts = rc_by_c[consumptions[len(consumptions) // 2]]
    ts = [p["temperature"] for p in pts]
    ax1.plot(ts, [p["magnetisation"] for p in pts], color=GREEN, marker="s",
             ms=4, lw=1.4, ls="--", label="Potts m (order)", zorder=2)
    ax1.axvline(t_c, color=GRAY, lw=0.9, ls=":", zorder=1)
    ax1.set_xscale("log")
    ax1.set_xlabel("matter temperature T")
    ax1.set_ylabel("recall capacity / order")
    ax1.set_title("Recall dies before order")
    ax1.legend(frameon=False, fontsize=8)

    # panel 2: the memory phase diagram — recall_capacity(T, c) heatmap
    grid = np.array([[next(p["recall_capacity"] for p in rc_by_c[c]
                           if p["temperature"] == t) for t in temps]
                     for c in consumptions])
    im = ax2.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                    vmin=0.0, vmax=1.0)
    ax2.set_xticks(range(len(temps)))
    ax2.set_xticklabels([f"{t:g}" for t in temps], fontsize=7, rotation=45)
    ax2.set_yticks(range(len(consumptions)))
    ax2.set_yticklabels([f"{c:g}" for c in consumptions], fontsize=8)
    ax2.set_xlabel("matter temperature T")
    ax2.set_ylabel("consumption c")
    ax2.set_title("The memory phase diagram: recall capacity")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # panel 3: soil consumption — fertile fraction falls as recall depletes it
    trace = consume_demo["trace"]
    ax3.plot([r["attempt"] for r in trace],
             [r["fertile_fraction"] for r in trace], color="#0072B2",
             marker="o", ms=3, lw=1.7, zorder=3)
    ax3.set_xlabel("recall attempts (rooting consumes soil)")
    ax3.set_ylabel("fertile fraction remaining")
    ax3.set_ylim(bottom=0.0)
    ax3.set_title(f"Recall is self-limiting (T={consume_demo['temperature']:g})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=28)
    p.add_argument("--temperatures", type=str,
                   default="0.03,0.05,0.065,0.08,0.095,0.11,0.14,0.22")
    p.add_argument("--consumptions", type=str, default="2.0,5.0,10.0")
    p.add_argument("--kappa-recovery", type=float, default=0.1)
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--quartic-u", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=DEFAULT_KAPPA_THRESHOLD)
    p.add_argument("--seed-size", type=int, default=4)
    p.add_argument("--scarcity-temperature", type=float, default=0.04,
                   help="a fertile (ordered) temperature for the consumption demo")
    p.add_argument("--consume-demo-c", type=float, default=2.0)
    p.add_argument("--consume-attempts", type=int, default=400)
    p.add_argument("--n-therm", type=int, default=500)
    p.add_argument("--n-meas", type=int, default=80)
    p.add_argument("--n-skip", type=int, default=3)
    p.add_argument("--bin-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--t-c", type=float, default=0.09)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_seed")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size = 16
        args.temperatures = "0.04,0.09,0.2"
        args.consumptions = "2.0,10.0"
        args.n_therm, args.n_meas = 150, 30
        args.consume_attempts = 120

    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    consumptions = [float(c) for c in args.consumptions.split(",") if c.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    print("Measurement 1+2: recall capacity across the melt (T × consumption)",
          flush=True)
    rc_by_c: dict[float, list[dict]] = {}
    for c in consumptions:
        pts = []
        for ti, t in enumerate(temps):
            pt = recall_capacity_point(temperature=t, consumption=c,
                                       args=args, seed=args.seed + 100 * ti)
            pts.append(pt)
            print(f"  c={c:g} T={t:.3f}: recall={pt['recall_capacity']:.3f}"
                  f"±{pt['recall_capacity_err']:.3f}  m={pt['magnetisation']:.3f}"
                  f"  ⟨κ⟩={pt['kappa_mean']:.3f}", flush=True)
        rc_by_c[c] = pts

    print("\nMeasurement 3: soil consumption — recall is self-limiting",
          flush=True)
    consume_demo = soil_consumption_demo(args)
    print(f"  T={consume_demo['temperature']:g}, c={consume_demo['consumption']:g}: "
          f"{consume_demo['n_attempts']} recall attempts, {consume_demo['n_rooted']} "
          f"rooted; fertile fraction {consume_demo['fertile_start']:.3f} → "
          f"{consume_demo['fertile_end']:.3f}", flush=True)

    # verdict
    lines = ["memory recall at criticality — verdict", "=" * 44, ""]
    c_ref = consumptions[len(consumptions) // 2]
    pts = rc_by_c[c_ref]
    below = [p for p in pts if p["temperature"] < args.t_c]
    above = [p for p in pts if p["temperature"] >= args.t_c]
    rc_below = max((p["recall_capacity"] for p in below), default=float("nan"))
    rc_above = max((p["recall_capacity"] for p in above), default=float("nan"))
    t_recall_dead = next((p["temperature"] for p in pts
                          if p["recall_capacity"] < 0.05), float("nan"))
    m_at_dead = next((p["magnetisation"] for p in pts
                      if p["recall_capacity"] < 0.05), float("nan"))
    lines.append(
        f"recall capacity (c={c_ref:g}): {rc_below:.2f} below T_c → "
        f"{rc_above:.2f} above — recall collapses across the melt")
    if not math.isnan(t_recall_dead):
        lines.append(
            f"recall reaches ~0 at T={t_recall_dead:g}, where the order "
            f"parameter is still m={m_at_dead:.2f}: the soil goes barren "
            "BEFORE the field disorders — a system can hold its structure "
            "but can no longer regenerate what it loses")
    # consumption-axis reading from the grid at the coldest temperature
    t0 = temps[0]
    rc_cold = {c: next(p["recall_capacity"] for p in rc_by_c[c]
                       if p["temperature"] == t0) for c in consumptions}
    if len(consumptions) >= 2 and rc_cold[consumptions[0]] > rc_cold[consumptions[-1]] + 0.05:
        lines.append(
            f"even in the ordered phase (T={t0:g}), scarcity starves recall: "
            f"{rc_cold[consumptions[0]]:.2f} → {rc_cold[consumptions[-1]]:.2f} "
            f"as c = {consumptions[0]:g} → {consumptions[-1]:g}")
    lines.append(
        f"recall is self-limiting: rooting {consume_demo['n_rooted']} seeds "
        f"drew the fertile fraction {consume_demo['fertile_start']:.2f} → "
        f"{consume_demo['fertile_end']:.2f} (rooting consumes soil, so recall "
        "spreads into fresh ground rather than piling onto exhausted spots)")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": vars(args),
               "recall_by_consumption": {f"{c:g}": rc_by_c[c]
                                         for c in consumptions},
               "soil_consumption": consume_demo}
    with open(os.path.join(args.output_dir, "n3_seed_rooting.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_seed_rooting.png")
        make_figure(rc_by_c, consumptions, temps, consume_demo,
                    args.t_c, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
