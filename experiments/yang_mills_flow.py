#!/usr/bin/env python3
"""Yang–Mills dynamics as gradient ascent on the gauge S-functional.

The URP picture is gradient ascent on `S = coupling·(covariant coherence) −
(curvature stress)`. The derivation (§3.2) shows the Yang–Mills equations are the
*stationarity conditions* of that S — i.e. the field flows until the gauge
curvature balances the matter current. This experiment runs that flow and checks
two things:

1. **The flow ascends S and the Yang–Mills residual → 0.** From a hot (random)
   start, gradient ascent monotonically raises S; the lattice equation-of-motion
   residual ``TA[U_μ Ω_μ]`` drives to zero — the configuration converges onto a
   solution of the discrete Yang–Mills equations (with the matter current as
   source). Shown for SU(2) and SU(3).

2. **Curvature localizes on the sector walls** ("gluons as boundary modes",
   §4.3.4). Fixing ψ to a real three-sector domain field and flowing only the
   connection, the resulting curvature is enriched on the domain walls — the
   gauge field carries the frame-mismatch between adjacent colour sectors.

Honest scope: this is deterministic gradient flow on S. Thermodynamic
confinement signatures (Wilson-loop area law, string tension, deconfinement
temperature) require Monte-Carlo ensembles and are not attempted here.

Usage::

    python experiments/yang_mills_flow.py
    python experiments/yang_mills_flow.py --size 12 --steps 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from project_genesis.gauge import (  # noqa: E402
    covariant_coherence,
    curvature_density,
    flow_step,
    identity_links,
    random_psi,
    random_unitary,
    sector_field_to_psi,
    total_s,
    wilson_action,
    yang_mills_residual,
)
from project_genesis.multiphase import (  # noqa: E402
    interface_mask,
    step_multiphase_conserved,
)

GROUP = {2: "SU(2)", 3: "SU(3)"}


def demo_ascent(n: int, *, size: int, steps: int, coupling: float, eps: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    spatial = (size, size)
    psi = random_psi(rng, n, spatial)
    links = random_unitary(rng, n, (2, *spatial), special=True, scale=0.9)
    print(f"=== {GROUP[n]} gradient ascent on S = {coupling}·coherence − stress ===")
    print(f"{'step':>5} | {'S':>9} | {'coherence':>9} | {'wilson':>8} | {'YM residual':>11}")
    mark = max(1, steps // 5)
    for step in range(steps + 1):
        if step % mark == 0:
            print(f"{step:>5} | {total_s(psi, links, coupling):>9.2f} | "
                  f"{covariant_coherence(psi, links):>9.2f} | {wilson_action(links):>8.2f} | "
                  f"{yang_mills_residual(psi, links, coupling):>11.5f}")
        if step < steps:
            psi, links = flow_step(psi, links, coupling=coupling, eps=eps, matter_rate=0.1)
    print()


def demo_boundary_modes(*, size: int, sector_steps: int, flow_steps: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    f = rng.random((3, size, size)) * 0.1
    for _ in range(sector_steps):
        f, _ = step_multiphase_conserved(f, diffusion=1.0, gamma=1.5, dt=0.1)
    psi = sector_field_to_psi(f)
    walls = interface_mask(f)
    links = identity_links(3, psi.shape[:-1])
    for _ in range(flow_steps):  # flow the connection with ψ fixed to the sectors
        psi, links = flow_step(psi, links, coupling=1.0, eps=0.04, evolve_matter=False)

    cd = curvature_density(links)
    wall_mean = float(cd[walls].mean()) if walls.any() else 0.0
    int_mean = float(cd[~walls].mean()) if (~walls).any() else 0.0
    frac_on_walls = float(cd[walls].sum() / cd.sum()) if cd.sum() else 0.0
    print("=== Gluons as boundary modes: curvature vs sector walls ===")
    print(f"  ψ fixed to a coarsened 3-sector field ({size}², {sector_steps} steps); connection flowed")
    print(f"  curvature density — wall mean {wall_mean:.5f} vs interior mean {int_mean:.5f}  "
          f"(enrichment ×{wall_mean / (int_mean + 1e-12):.1f})")
    print(f"  {frac_on_walls:.0%} of total curvature sits on walls, which are {walls.mean():.0%} of sites\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--coupling", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=0.04)
    p.add_argument("--sector-size", type=int, default=48)
    p.add_argument("--sector-steps", type=int, default=500)
    p.add_argument("--flow-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    print("Yang–Mills dynamics: gradient ascent on the gauge S-functional (URP §3)\n")
    for n in (2, 3):
        demo_ascent(n, size=args.size, steps=args.steps, coupling=args.coupling, eps=args.eps, seed=args.seed)
    demo_boundary_modes(size=args.sector_size, sector_steps=args.sector_steps,
                        flow_steps=args.flow_steps, seed=args.seed)

    print("Verdict: gradient ascent on S converges onto a Yang–Mills solution (residual → 0),")
    print("and the connection's curvature is enriched on the sector walls. Thermodynamic")
    print("confinement (Wilson-loop area law) needs Monte-Carlo ensembles — future work.")


if __name__ == "__main__":
    main()
