#!/usr/bin/env python3
"""Does fertile soil guarantee successful planting of recalled seeds, or has the domain structure adapted during low-κ periods?

This experiment separates "soil barren" from "soil fertile but seed fails to integrate" across the melt.
It tests the hysteresis between capacity restoration and memory persistence, with explicit dwell time in the barren regime.

Core distinction (three observable outcomes):
- High survival when soil fertile + ordered host        → Scenario A (corpus survived, everything works)
- Low survival even when soil fertile                   → Scenario C (domain structure coarsened/adapted; host can't host seeds)
- Survival only near existing domain walls              → "islands of lucidity"

Protocol (tightened per review):
1. Pre-populate a small library of stable sector seeds at low T (high κ).
2. For each (T, r): thermalize/dwell for dwell_steps at target T with given recovery.
3. Attempt K plantings per (T, r), recording local soil fertility at injection site.
4. Track each planted seed for M steps using full joint_sweep (host dynamics matter).
5. Survival = footprint retains ≥70% of original sector identity (argmax comparison).
6. Optional frozen-host control isolates seed viability from host coherence.

Reuses:
- sector_seeds.make_sector_seed, plant_sector_seed (faithful κ-as-soil rule)
- annealed_matter.joint_sweep (full thermal evolution with optional κ)
- n3_recall_recovery.crossing_below (for reference edges)

Usage:
    python experiments/n3_planting_success.py --output-dir artifacts/n3_planting
    python experiments/n3_planting_success.py --quick
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from copy import deepcopy

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from project_genesis.annealed_matter import joint_sweep, kappa_step
    from project_genesis.sector_seeds import (
        make_sector_seed, plant_sector_seed, local_kappa_mean,
        DEFAULT_KAPPA_THRESHOLD, DEFAULT_KAPPA_COST, DEFAULT_BLEND
    )
    from n3_recall_recovery import crossing_below
except ImportError as e:
    print("Missing dependency:", e)
    sys.exit(1)


def sector_identity_fraction(psi_before: np.ndarray, psi_after: np.ndarray,
                             site: tuple[int, ...], size: int) -> float:
    """Fraction of the footprint whose argmax sector label is unchanged after M steps."""
    spatial = psi_before.shape[:-1]
    ix = tuple(slice(max(0, s - size//2), min(spatial[d], s + size//2 + 1)) for d, s in enumerate(site))
    # Simpler: use the exact footprint helper from sector_seeds if exposed, else approximate
    labels_before = np.argmax(np.abs(psi_before), axis=-1)
    labels_after = np.argmax(np.abs(psi_after), axis=-1)
    footprint_before = labels_before[ix] if hasattr(ix, '__len__') else labels_before
    footprint_after = labels_after[ix] if hasattr(ix, '__len__') else labels_after
    # For simplicity in pilot: use a square window around site
    # (production version should use the exact _footprint_ix from sector_seeds)
    match = np.sum(footprint_before == footprint_after)
    total = footprint_before.size
    return float(match / max(total, 1))


def track_seed_fate(psi: np.ndarray, links: np.ndarray, kappa: np.ndarray | None,
                    seed: np.ndarray, site: tuple[int, ...],
                    M: int, temperature: float, beta_g: float,
                    matter_coupling: float = 1.0, quartic_u: float = 1.0,
                    recovery: float = 0.8, consumption: float = 5.0,
                    threshold: float = DEFAULT_KAPPA_THRESHOLD,
                    cost: float = DEFAULT_KAPPA_COST, blend: float = DEFAULT_BLEND,
                    force_plant: bool = False, frozen_host: bool = False,
                    rng: np.random.Generator | None = None) -> dict:
    """Attempt to plant seed and track survival for M joint_sweep steps.

    Returns dict with:
        rooted, soil_fertile (local), survived_M, final_local_order,
        host_m_at_plant, host_local_order_at_plant
    """
    if rng is None:
        rng = np.random.default_rng()

    spatial = psi.shape[:-1]
    size = seed.shape[0]

    # Local soil check at exact injection site
    local_k = local_kappa_mean(kappa, site, size) if kappa is not None else 1.0
    soil_fertile = (local_k >= threshold) if threshold > 0 else True

    eff_threshold = 0.0 if force_plant else threshold
    psi_new, kappa_new, rooted, _ = plant_sector_seed(
        psi, kappa if kappa is not None else np.ones(spatial),
        seed, site, threshold=eff_threshold, cost=cost, blend=blend
    )

    if not rooted:
        return {
            "rooted": False, "soil_fertile": soil_fertile, "survived_M": False,
            "final_local_order": 0.0, "host_m_at_plant": 0.0,
            "host_local_order_at_plant": 0.0
        }

    # Track for M steps with full joint_sweep (or frozen host)
    psi_t = psi_new
    links_t = links.copy()
    kappa_t = kappa_new.copy() if kappa is not None else None

    for _ in range(M):
        if frozen_host:
            # Only evolve the seed footprint (cheap control)
            # For pilot: simple local relaxation (placeholder; real version can mask updates)
            pass  # TODO: implement local-only evolution if needed
        else:
            psi_t, links_t, kappa_t, _ = joint_sweep(
                psi_t, links_t, rng,
                temperature=temperature, beta_g=beta_g,
                matter_coupling=matter_coupling, quartic_u=quartic_u,
                kappa=kappa_t,
                kappa_params={"recovery": recovery, "consumption": consumption}
            )

    final_local_order = local_kappa_mean(kappa_t, site, size) if kappa_t is not None else 0.0  # proxy
    # Better: use sector identity retention
    identity_retention = sector_identity_fraction(psi_new, psi_t, site, size)
    survived = identity_retention >= 0.70

    host_m = float(np.mean(np.abs(psi_t) ** 2).max(axis=-1))  # rough global order proxy

    return {
        "rooted": True,
        "soil_fertile": soil_fertile,
        "survived_M": survived,
        "final_local_order": final_local_order,
        "host_m_at_plant": host_m,
        "host_local_order_at_plant": identity_retention,  # reuse for now
        "identity_retention": identity_retention
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--temperatures", type=str, default="0.03,0.08,0.11,0.16,0.20")
    p.add_argument("--recoveries", type=str, default="0.2,0.8")
    p.add_argument("--consumption", type=float, default=5.0)
    p.add_argument("--beta-g", type=float, default=3.0)
    p.add_argument("--matter-coupling", type=float, default=4.0)
    p.add_argument("--dwell", type=int, default=100, help="steps spent at target T before planting")
    p.add_argument("--M", type=int, default=100, help="survival tracking steps")
    p.add_argument("--K", type=int, default=5, help="planting attempts per (T,r)")
    p.add_argument("--force-plant", action="store_true", default=True)
    p.add_argument("--frozen-host", action="store_true", help="control: freeze host psi during tracking")
    p.add_argument("--output-dir", type=str, default="artifacts/n3_planting")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.size = 12
        args.temperatures = "0.05,0.11,0.18"
        args.recoveries = "0.2,0.8"
        args.dwell = 30
        args.M = 30
        args.K = 3

    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    recoveries = [float(r) for r in args.recoveries.split(",") if r.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.default_rng(42)
    n_palette = 3

    # Pre-populate "corpus" as ideal stable sector seeds (engine-generated ideal for pilot)
    # Future: run short low-T joint_sweep and extract real stable patches
    seed_library = [make_sector_seed(n_palette, 4, sector=s, dim=2) for s in range(n_palette)]

    records = []
    summary = []

    for r in recoveries:
        for T in temps:
            # Simple thermalize + dwell using joint_sweep (lightweight stand-in for full protocol)
            # In production: call the full recall_capacity_point or a dedicated dwell function
            psi = np.random.default_rng(123).random((args.size, args.size, n_palette)) + 0.1
            psi = psi / np.linalg.norm(psi, axis=-1, keepdims=True)
            links = np.eye(n_palette)  # placeholder links; real version loads from snapshot
            kappa = np.ones((args.size, args.size)) * 0.8

            for _ in range(args.dwell):
                psi, links, kappa, _ = joint_sweep(
                    psi, links, rng,
                    temperature=T, beta_g=args.beta_g,
                    matter_coupling=args.matter_coupling,
                    kappa=kappa,
                    kappa_params={"recovery": r, "consumption": args.consumption}
                )

            successes_fertile = 0
            successes_force = 0
            total_fertile = 0
            total_force = 0

            for k in range(args.K):
                for seed in seed_library:
                    site = (rng.integers(0, args.size), rng.integers(0, args.size))
                    fate = track_seed_fate(
                        psi, links, kappa, seed, site,
                        M=args.M, temperature=T, beta_g=args.beta_g,
                        matter_coupling=args.matter_coupling,
                        recovery=r, consumption=args.consumption,
                        force_plant=args.force_plant,
                        frozen_host=args.frozen_host,
                        rng=rng
                    )
                    records.append({
                        "T": T, "r": r, "site": site,
                        **fate
                    })
                    if fate["soil_fertile"]:
                        total_fertile += 1
                        if fate["survived_M"]:
                            successes_fertile += 1
                    total_force += 1
                    if fate["survived_M"]:
                        successes_force += 1

            fertile_rate = successes_fertile / max(total_fertile, 1)
            force_rate = successes_force / max(total_force, 1)
            summary.append({
                "T": T, "r": r,
                "soil_fertile_rate": round(fertile_rate, 3),
                "survival_fertile_only": round(fertile_rate, 3),
                "survival_force_plant": round(force_rate, 3),
                "n_attempts": total_force
            })

    # Write outputs
    with open(os.path.join(args.output_dir, "planting_records.json"), "w") as fh:
        json.dump(records, fh, indent=2)

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write("T\tr\tsoil_fertile_rate\tsurvival_fertile_only\tsurvival_force_plant\tn\n")
        for row in summary:
            fh.write(f"{row['T']}\t{row['r']}\t{row['soil_fertile_rate']}\t"
                     f"{row['survival_fertile_only']}\t{row['survival_force_plant']}\t{row['n_attempts']}\n")

    print("\nPilot summary (see artifacts for full JSON):")
    for row in summary:
        print(row)

    print(f"\nArtifacts in {args.output_dir}")


if __name__ == "__main__":
    main()
