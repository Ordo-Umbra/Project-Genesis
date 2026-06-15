#!/usr/bin/env python3
"""The gauge connection as the minimal structure that restores coherence.

Demonstrates the core argument of the URP gauge derivation (§2–3) on a lattice,
for U(1), SU(2) and SU(3) — and on the actual Ψ∈ℂ³ sector field built by
`project_genesis.multiphase`. Three claims, each checked numerically:

1. **Local rotations destroy naive coherence.** With no connection, the
   inter-site coherence ``Σ Re[ψ†(x) ψ(x+μ̂)]`` is scrambled by a local gauge
   transform ``ψ(x) → g(x)ψ(x)`` — path-dependent transport, the drop in ΔI the
   derivation describes.

2. **A connection restores it, and is gauge-invariant.** The covariant coherence
   ``Σ Re[ψ†(x) U_μ(x) ψ(x+μ̂)]`` is exactly invariant under the joint transform
   ``ψ→gψ, U_μ→g(x)U_μ g(x+μ̂)†`` — the connection is the minimal structure that
   keeps integration intact under local distinction.

3. **Curvature is coherence stress.** A flat (pure-gauge) connection has zero
   Wilson action; a connection whose parallel transport is path-dependent has
   positive action — the lattice ``Tr(F_{μν}F^{μν})``.

Usage::

    python experiments/gauge_coherence.py
    python experiments/gauge_coherence.py --size 16 --groups 1,2,3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from project_genesis.gauge import (  # noqa: E402
    apply_gauge_transform,
    covariant_coherence,
    identity_links,
    naive_coherence,
    pure_gauge_links,
    random_psi,
    random_unitary,
    sector_field_to_psi,
    wilson_action,
)
from project_genesis.multiphase import step_multiphase_conserved  # noqa: E402

GROUP_NAME = {1: "U(1)", 2: "SU(2)", 3: "SU(3)"}


def demo_group(n: int, *, size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    spatial = (size, size)
    ndim = len(spatial)
    special = n > 1

    psi = random_psi(rng, n, spatial)
    links = random_unitary(rng, n, (ndim, *spatial), special=special, scale=0.7)
    g = random_unitary(rng, n, spatial, special=special)

    # Claim 1 & 2: invariance of covariant vs naive coherence under g.
    cov0, nai0 = covariant_coherence(psi, links), naive_coherence(psi)
    psi_g, links_g = apply_gauge_transform(psi, links, g)
    cov1, nai1 = covariant_coherence(psi_g, links_g), naive_coherence(psi_g)

    # Claim "minimal structure restores ΔI": a uniform field has maximal naive
    # coherence; rotating it locally destroys that; the pure-gauge connection
    # built from the SAME g restores the covariant coherence to the maximum.
    uniform = np.zeros((*spatial, n), dtype=np.complex128)
    uniform[..., 0] = 1.0
    max_coh = naive_coherence(uniform)
    rotated = np.einsum("...ij,...j->...i", g, uniform)
    rotated_naive = naive_coherence(rotated)
    restored = covariant_coherence(rotated, pure_gauge_links(g, ndim))

    # Claim 3: curvature of flat vs generic connection.
    wa_flat = wilson_action(pure_gauge_links(g, ndim))
    wa_curved = wilson_action(links)

    print(f"=== {GROUP_NAME[n]} ===")
    print(f"  covariant coherence — gauge-invariant?   |Δ| = {abs(cov1 - cov0):.2e}  (→ 0)")
    print(f"  naive coherence    — gauge-invariant?   |Δ| = {abs(nai1 - nai0):.3f}  (≠ 0, scrambled)")
    print(f"  restore ΔI: uniform={max_coh:.1f}  rotated(naive)={rotated_naive:.1f}  "
          f"covariant w/ connection={restored:.1f}  (→ matches uniform)")
    print(f"  curvature: Wilson(flat)={wa_flat:.2e} (→ 0)   Wilson(curved)={wa_curved:.1f} (> 0)\n")


def demo_on_sectors(*, size: int, steps: int, seed: int) -> None:
    """Lay an SU(3) connection over a real three-sector domain field."""
    rng = np.random.default_rng(seed)
    f = rng.random((3, size, size)) * 0.1
    for _ in range(steps):
        f, _ = step_multiphase_conserved(f, diffusion=1.0, gamma=1.5, dt=0.1)
    psi = sector_field_to_psi(f)  # ψ ∈ ℂ³ from R/G/B membership
    ndim = psi.ndim - 1
    links = identity_links(3, psi.shape[:-1])
    g = random_unitary(rng, 3, psi.shape[:-1], special=True)

    cov0 = covariant_coherence(psi, links)
    psi_g, links_g = apply_gauge_transform(psi, links, g)
    cov1 = covariant_coherence(psi_g, links_g)
    print("=== SU(3) connection on the real Ψ∈ℂ³ sector field ===")
    print(f"  ψ built from a coarsened 3-sector domain field ({size}², {steps} steps)")
    print(f"  covariant coherence across domain walls — gauge-invariant? |Δ| = {abs(cov1 - cov0):.2e}")
    print("  (the colour sectors R/G/B are the ℂ³ basis; the connection makes")
    print("   inter-domain coherence independent of local colour-frame choice)\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--groups", type=str, default="1,2,3", help="Which N to demo (1=U(1),2,3).")
    p.add_argument("--sector-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    print("Gauge connection as coherence restoration — URP §2–3, on the lattice\n")
    for n in (int(x) for x in args.groups.split(",") if x.strip()):
        demo_group(n, size=args.size, seed=args.seed)
    demo_on_sectors(size=args.size * 3, steps=args.sector_steps, seed=args.seed)

    print("Verdict: the covariant coherence is gauge-invariant to machine precision and the")
    print("pure-gauge connection has zero curvature — the connection is the minimal structure")
    print("that preserves integration (ΔI) under local distinction. Yang–Mills *dynamics*")
    print("(evolving U to extremise S, confinement) remain future work.")


if __name__ == "__main__":
    main()
