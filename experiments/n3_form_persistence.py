"""Form persistence under noise: living junction networks and finite-size self-repair.

Deterministic multiphase Allen–Cahn coarsens by mean curvature: walls and
junctions shrink and the lighter generations (0- and 1-cells) are driven toward
zero.  Continuous noise competes with that drain, but *unconstrained*
Allen–Cahn + additive Langevin either collapses to one domain or scrambles
into texture — measured in the first persistence runs (P1 failed).

The dynamics that hold reliable junction networks elsewhere in the repo are
**volume-conserving** (``step_multiphase_conserved``) and **κ-gated**
(``step_multiphase_kappa``: capacity is consumed at walls and regenerates with
slack, pinning the network where load is high).  Conserved multiphase alone
fixed the ordered noise window (P1 ✓) but light-form *density* still fell with
size — a surface-to-volume geometry effect, not necessarily a failure of
self-repair.  With the domain width now reported alongside the density, that
reading has direct evidence: across sizes 20/28/36 the density falls while the
diameter *rises*, so the structure is not degrading with the box.

**κ-gating, run: the recovery rate decides, and the default is below the
knee.**  The hypothesis was that capacity-limited coarsening would hold the
network where curvature-limited coarsening could not.  At the default recovery
rate it does the opposite — capacity is consumed faster than it regenerates,
⟨κ⟩ sits at ≈ 0.020, the coupling is gated to near zero everywhere and the
initial random configuration never coarsens at all.  Every field in that scan
is texture, at every noise level including zero.  That is not κ-gating failing
to hold structure; it is κ-gating starving the field before structure exists.
Raise the rate and the hypothesis lands as written:

================  =========  ======================  ========
kappa_recovery    ⟨κ⟩        diameter at zero noise  score
================  =========  ======================  ========
0.1 (default)     0.020      2.23 — texture          0/3
0.8               0.41-0.55  7.95                    **3/3**
3.0               0.83       7.16                    ordered throughout
================  =========  ======================  ========

At ``r = 0.8`` this is the only arm that passes P2, and it passes by the
predicted mechanism: across sizes 20/28/36 the capacity buffer holds and grows
(⟨κ⟩ = 0.407 → 0.436 → 0.435) with the domain width rising alongside it
(4.31 → 4.67 → 4.73).  ``r = 0.8`` is not a new constant — it is the setting at
which the memory-recall work found recall outliving order, and at which 3-D
fertile soil re-percolates.

Primary default remains conserved; κ and conserved+κ are first-class options.

**The texture guard.**  P1's whole content is "ordered structure, not noise",
so it lives or dies on what separates the two.  Until this revision that job
was done by ``n_phases`` — the number of distinct sector labels present
anywhere on the lattice.  It cannot do it: salt-and-pepper texture carries
every label too, so ``n_phases`` reads ``3.0`` at *every* noise level scanned,
including at literal texture.  Measured, size 16, 3-D, conserved, 3 seeds:

====== ========== ==========
noise  n_phases   diameter
====== ========== ==========
0.00   3.0        7.76
0.02   3.0        5.76
0.08   3.0        4.67
0.20   3.0        2.56
0.40   3.0        1.00
====== ========== ==========

``n_phases`` is flat across the entire range; ``domain_diameter`` — the repo's
dimension-corrected texture guard, which reads ``1.0`` for pure noise in every
dimension — falls monotonically and reaches noise at 0.40.  The old predicate
therefore passed P1 on the noise = 0.20 point, where the domains are ~2.5 cells
wide and the light-form gain is mostly the lattice fragmenting.

The floor is ``3.0`` lattice units and it is **derived, not chosen**: the
light-form census reads ``local_dimension`` off a ``3^d`` neighbourhood, so a
domain narrower than three cells cannot be resolved as a domain by the very
instrument counting its cells.  Below that floor the count is a fact about the
lattice.

Pre-registered predictions:

P1. **Noise window for persistence.**  At fixed small lattice, intermediate
    noise raises late-time light-form density above the zero-noise baseline
    while the field is still *tessellated* — domains wide enough for a
    junction reading to mean anything (see the texture guard below).
P2. **Finite-size self-repair.**  At fixed intermediate noise — the densest
    still-tessellated point in the sweep — larger lattices retain higher
    late-time light-form density *or* (under κ dynamics) a higher capacity
    buffer, on fields that are still tessellated at every size.  Statistical
    buffering against local collapse.
P3. **Baseline without noise.**  With noise = 0 the light-form density sits at
    a low floor (conserved multi-domain residual, or coarsened state under
    unconstrained / κ-only dynamics).

Usage::

    python experiments/n3_form_persistence.py --output-dir artifacts/n3_persistence
    python experiments/n3_form_persistence.py --quick
    python experiments/n3_form_persistence.py --dynamics kappa
    python experiments/n3_form_persistence.py --dynamics conserved_kappa
    python experiments/n3_form_persistence.py --dynamics conserved_kappa \
        --kappa-recovery 0.8      # above the knee; below it the field never forms
    python experiments/n3_form_persistence.py --dynamics unconstrained
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.dimensional_forms import dimensional_census, local_dimension
from project_genesis.multiphase import (
    domain_diameter,
    sector_labels,
    step_multiphase,
    step_multiphase_conserved,
    step_multiphase_kappa,
)


#: Minimum domain width, in lattice units, for a light-form count to be a
#: statement about structure rather than about the lattice.  The census reads
#: ``local_dimension`` off a ``3^d`` neighbourhood, so a domain narrower than
#: three cells is not resolvable as a domain by the instrument counting it.
#: ``domain_diameter`` is dimension-corrected and reads ``1.0`` for pure noise
#: in 2-, 3- and 4-D alike, which is the property that makes it usable here
#: where the raw ``domain_scale`` (floor 2.5) would mean a different thing in
#: each dimension.
MIN_DOMAIN_DIAMETER = 3.0


def tessellated(n_phases: float, diameter: float, palette: int) -> bool:
    """Is this a multi-domain tessellation, or is it texture?

    Both halves are load-bearing and neither is sufficient.  ``n_phases``
    alone cannot see texture — noise carries every label.  ``diameter`` alone
    cannot see collapse — a single domain has no walls and reads ``inf``.
    """
    return (
        n_phases >= max(2.0, palette - 1.1)
        and diameter >= MIN_DOMAIN_DIAMETER
    )


def pick_size_scan_noise(mids: list, palette: int) -> float:
    """Which noise level P2's size scan runs at.

    Third site of the same blind predicate, and the one that did the most
    damage.  This used to filter on ``n_phases`` — which admits everything —
    and then take the *highest* light-form density, i.e. it deliberately ran
    the size scan on the most fragmented field in the sweep.  Since texture
    reads higher on that quantity than structure does, the selection reliably
    picked the point least able to say anything about self-repair.

    P2 asks whether a persistent network repairs itself as the box grows, so it
    has to be run where P1 found a network: the densest point that is still
    tessellated.  Falls back to the raw maximum only when nothing in the sweep
    is tessellated, in which case P2's own guard refuses it anyway.
    """
    ordered = [
        r for r in mids
        if tessellated(r["n_phases_mean"], r["diameter_mean"], palette)
    ]
    return float(max(ordered or mids, key=lambda r: r["late_mean"])["noise"])


def light_form_density(labels: np.ndarray) -> float:
    """Fraction of sites that are 0-cells or 1-cells (light generations)."""
    dim = local_dimension(labels)
    return float(np.mean(dim <= 1))


def evolve(
    size: int,
    *,
    ndim: int,
    palette: int,
    steps: int,
    noise: float,
    seed: int,
    dynamics: str = "conserved",
    dt: float = 0.1,
    record_every: int = 20,
    kappa_baseline: float = 1.0,
    kappa_recovery: float = 0.1,
    kappa_consumption: float = 5.0,
    kappa_diffusion: float = 0.5,
) -> dict:
    """Run multiphase dynamics with optional noise and optional κ field."""
    rng = np.random.default_rng(seed)
    shape = (palette,) + (size,) * ndim
    fields = 0.1 * rng.standard_normal(shape)
    norms = np.sqrt(np.sum(fields * fields, axis=0, keepdims=True) + 1e-12)
    fields = fields / (norms * 0.7)
    noise_scale = np.sqrt(2.0 * max(noise, 0.0) * dt)

    use_kappa = dynamics in ("kappa", "conserved_kappa")
    kappa = (
        np.full(shape[1:], kappa_baseline, dtype=np.float64) if use_kappa else None
    )

    history = []
    kappa_hist = []
    for t in range(steps):
        if dynamics == "conserved":
            fields, _ = step_multiphase_conserved(
                fields, None, diffusion=1.0, gamma=1.5, dt=dt
            )
        elif dynamics == "conserved_kappa":
            fields, kappa = step_multiphase_conserved(
                fields,
                kappa,
                diffusion=1.0,
                gamma=1.5,
                dt=dt,
                kappa_baseline=kappa_baseline,
                kappa_recovery=kappa_recovery,
                kappa_consumption=kappa_consumption,
                kappa_diffusion=kappa_diffusion,
            )
        elif dynamics == "kappa":
            fields, kappa = step_multiphase_kappa(
                fields,
                kappa,
                diffusion=1.0,
                gamma=1.5,
                dt=dt,
                kappa_baseline=kappa_baseline,
                kappa_recovery=kappa_recovery,
                kappa_consumption=kappa_consumption,
                kappa_diffusion=kappa_diffusion,
            )
        else:  # unconstrained
            fields = step_multiphase(
                fields, diffusion=1.0, gamma=1.5, dt=dt
            )

        if noise > 0.0:
            fields = fields + noise_scale * rng.standard_normal(fields.shape)

        if t % record_every == 0 or t == steps - 1:
            labels = sector_labels(fields)
            history.append(light_form_density(labels))
            if kappa is not None:
                kappa_hist.append(float(kappa.mean()))

    labels = sector_labels(fields)
    census = dimensional_census(labels)
    late = history[max(1, 2 * len(history) // 3) :]
    late_kappa = (
        float(np.mean(kappa_hist[max(1, 2 * len(kappa_hist) // 3) :]))
        if kappa_hist
        else None
    )
    return {
        "size": size,
        "noise": noise,
        "ndim": ndim,
        "dynamics": dynamics,
        "final_light_density": float(history[-1]),
        "late_mean_light_density": float(np.mean(late)),
        "history": [float(h) for h in history],
        "late_mean_kappa": late_kappa,
        "final_kappa": float(kappa.mean()) if kappa is not None else None,
        "cells": {str(k): int(v) for k, v in census["cells"].items()},
        "euler": int(census["euler"]),
        "n_phases": int(len(np.unique(labels))),
        "domain_diameter": float(domain_diameter(labels)),
        "valence_by_dim": {
            str(k): float(v) for k, v in census.get("valence_by_dim", {}).items()
        },
    }


def mean_over_seeds(args, *, size: int, noise: float) -> dict:
    rows = [
        evolve(
            size,
            ndim=args.ndim,
            palette=args.palette,
            steps=args.steps,
            noise=noise,
            seed=args.seed + 97 * s + 13 * size + 31 * args.ndim,
            dynamics=args.dynamics,
            record_every=args.record_every,
            kappa_baseline=args.kappa_baseline,
            kappa_recovery=args.kappa_recovery,
            kappa_consumption=args.kappa_consumption,
            kappa_diffusion=args.kappa_diffusion,
        )
        for s in range(args.n_seeds)
    ]
    kappas = [r["late_mean_kappa"] for r in rows if r["late_mean_kappa"] is not None]
    return {
        "size": size,
        "noise": noise,
        "late_mean": float(np.mean([r["late_mean_light_density"] for r in rows])),
        "late_std": float(np.std([r["late_mean_light_density"] for r in rows])),
        "final_mean": float(np.mean([r["final_light_density"] for r in rows])),
        "n_phases_mean": float(np.mean([r["n_phases"] for r in rows])),
        "diameter_mean": float(np.mean([r["domain_diameter"] for r in rows])),
        "late_kappa_mean": float(np.mean(kappas)) if kappas else None,
        "runs": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--ndim", type=int, default=3, choices=[2, 3],
        help="spatial dimension (3 is default; 2 for faster exploration)",
    )
    p.add_argument(
        "--dynamics",
        choices=["conserved", "kappa", "conserved_kappa", "unconstrained"],
        default="conserved",
        help="conserved = volume-conserving (default); "
             "kappa = capacity-gated Allen-Cahn; "
             "conserved_kappa = both; "
             "unconstrained = plain Allen-Cahn",
    )
    p.add_argument("--palette", type=int, default=3)
    p.add_argument("--sizes", type=int, nargs="+", default=None)
    p.add_argument(
        "--noises", type=float, nargs="+",
        default=[0.0, 0.01, 0.03, 0.06, 0.12, 0.25],
    )
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--record-every", type=int, default=20)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kappa-baseline", type=float, default=1.0)
    p.add_argument("--kappa-recovery", type=float, default=0.1)
    p.add_argument("--kappa-consumption", type=float, default=5.0)
    p.add_argument("--kappa-diffusion", type=float, default=0.5)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_persistence")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.sizes is None:
        args.sizes = [20, 28, 36] if args.ndim == 3 else [32, 48, 64]
    if args.steps is None:
        args.steps = 280 if args.ndim == 3 else 400

    if args.quick:
        if args.ndim == 3:
            args.sizes = [16, 24]
            args.noises = [0.0, 0.02, 0.08, 0.20]
            args.steps = 140
            args.n_seeds = 2
        else:
            args.sizes = [24, 40]
            args.noises = [0.0, 0.02, 0.08, 0.20]
            args.steps = 200
            args.n_seeds = 2

    os.makedirs(args.output_dir, exist_ok=True)
    print(
        "form persistence under noise: living networks and finite-size "
        "self-repair",
        flush=True,
    )
    print(
        f"  dynamics={args.dynamics}  ndim={args.ndim}  palette={args.palette}  "
        f"steps={args.steps}  sizes={args.sizes}  noises={args.noises}",
        flush=True,
    )

    small = min(args.sizes)
    noise_scan = [mean_over_seeds(args, size=small, noise=n) for n in args.noises]
    print(f"  noise scan at size={small}:", flush=True)
    for r in noise_scan:
        kmsg = (
            f"  kappa={r['late_kappa_mean']:.3f}"
            if r["late_kappa_mean"] is not None
            else ""
        )
        print(
            f"    noise={r['noise']:.3f}: late_light={r['late_mean']:.4f} "
            f"± {r['late_std']:.4f}  n_phases={r['n_phases_mean']:.1f}  "
            f"diam={r['diameter_mean']:.2f}"
            f"{'' if tessellated(r['n_phases_mean'], r['diameter_mean'], args.palette) else '  [texture]'}"
            f"{kmsg}",
            flush=True,
        )

    by_noise = {r["noise"]: r for r in noise_scan}
    zero = by_noise[0.0]
    mids = [r for r in noise_scan if 0.0 < r["noise"] < max(args.noises)]

    # P1 passes on an intermediate noise that raises the light-form density
    # above the zero-noise baseline *while the field is still tessellated*.
    # The second half used to be `n_phases`, which cannot see texture at all;
    # it is now the domain width.  There is no third clause: the previous
    # version carried `(below strong) or ordered` with `ordered` already
    # required in the conjunction, so it could never change an outcome.
    p1 = False
    p1_at = None
    for r in mids:
        above_zero = r["late_mean"] > zero["late_mean"] * 1.4 + 1e-4
        if above_zero and tessellated(
            r["n_phases_mean"], r["diameter_mean"], args.palette
        ):
            p1 = True
            p1_at = r
            break
    # What the guard rejected, reported rather than left implicit.
    rejected = [
        r
        for r in noise_scan
        if r["noise"] > 0.0
        and not tessellated(r["n_phases_mean"], r["diameter_mean"], args.palette)
    ]

    floor = 0.08 if args.dynamics in ("conserved", "conserved_kappa") else (
        0.06 if args.ndim == 3 else 0.08
    )
    p3 = (
        zero["late_mean"] < floor
        or zero["late_mean"] < zero["runs"][0]["history"][0] * 0.5
    )

    if mids:
        mid_noise = pick_size_scan_noise(mids, args.palette)
    else:
        mid_noise = args.noises[len(args.noises) // 2]

    size_scan = [
        mean_over_seeds(args, size=s, noise=mid_noise) for s in args.sizes
    ]
    print(f"  size scan at noise={mid_noise:.3f}:", flush=True)
    for r in size_scan:
        kmsg = (
            f"  kappa={r['late_kappa_mean']:.3f}"
            if r["late_kappa_mean"] is not None
            else ""
        )
        print(
            f"    size={r['size']}: late_light={r['late_mean']:.4f} "
            f"± {r['late_std']:.4f}  n_phases={r['n_phases_mean']:.1f}  "
            f"diam={r['diameter_mean']:.2f}"
            f"{'' if tessellated(r['n_phases_mean'], r['diameter_mean'], args.palette) else '  [texture]'}"
            f"{kmsg}",
            flush=True,
        )

    lates = [r["late_mean"] for r in size_scan]
    # Density-based self-repair OR (under kappa) non-decreasing mean kappa
    # with non-collapsing phases — capacity buffer grows with system size.
    #
    # Both arms are gated on tessellation, for the same reason P1 is: a
    # light-form density measured on a field of unresolvable domains is a fact
    # about the lattice, and it *rises* as the structure degrades, so an
    # ungated P2 is passed most readily by the runs that have least to do with
    # the claim.  The κ arm at default consumption is exactly such a run.
    size_scan_ok = all(
        tessellated(r["n_phases_mean"], r["diameter_mean"], args.palette)
        for r in size_scan
    )
    p2_density = size_scan_ok and len(lates) >= 2 and lates[-1] >= lates[0] * 0.95 and (
        lates[-1] > lates[0] * 1.05 or max(lates) == lates[-1]
    )
    kappas = [r["late_kappa_mean"] for r in size_scan if r["late_kappa_mean"] is not None]
    p2_kappa = (
        size_scan_ok
        and len(kappas) >= 2
        and kappas[-1] >= kappas[0] * 0.98
    )
    p2 = p2_density or p2_kappa

    dim_label = "3-D (vertices + triple-lines)" if args.ndim == 3 else "2-D"
    dyn_label = args.dynamics
    lines = ["Form persistence under noise — verdict", "=" * 74, ""]
    lines.append(
        f"P1 (noise window for persistence, {dim_label}, {dyn_label}): at "
        f"size={small}, late light-form densities across noises {args.noises} "
        f"are "
        + "/".join(f"{r['late_mean']:.4f}" for r in noise_scan)
        + " (domain diameter "
        + "/".join(f"{r['diameter_mean']:.2f}" for r in noise_scan)
        + ", n_phases "
        + "/".join(f"{r['n_phases_mean']:.1f}" for r in noise_scan)
        + ") — "
        + (
            f"✓ at noise={p1_at['noise']:.3f} the light-form density is "
            f"{p1_at['late_mean'] / max(zero['late_mean'], 1e-12):.2f}x the "
            f"zero-noise baseline on a field still tessellated at "
            f"{p1_at['diameter_mean']:.2f} lattice units."
            if p1
            else "✗ no intermediate noise raises the density on a field that "
            "is still tessellated."
        )
    )
    if rejected:
        lines.append(
            "    texture guard rejected noise "
            + "/".join(f"{r['noise']:.2f}" for r in rejected)
            + " (domain diameter "
            + "/".join(f"{r['diameter_mean']:.2f}" for r in rejected)
            + f" < {MIN_DOMAIN_DIAMETER:.1f}); n_phases reads "
            + "/".join(f"{r['n_phases_mean']:.1f}" for r in rejected)
            + " there, which is why it cannot do this job."
        )
    kappa_str = ""
    if kappas:
        kappa_str = (
            " kappa "
            + "/".join(f"{k:.3f}" for k in kappas)
        )
    lines.append(
        f"P2 (finite-size self-repair): at noise={mid_noise:.3f}, late "
        f"light-form density vs size {args.sizes} is "
        + "/".join(f"{r['late_mean']:.4f}" for r in size_scan)
        + kappa_str
        + " — "
        + (
            "✓ larger lattices retain higher density and/or capacity buffer "
            "— statistical self-repair."
            if p2
            else "✗ the size scan is not tessellated (domain diameter "
            + "/".join(f"{r['diameter_mean']:.2f}" for r in size_scan)
            + "), so the density trend is a fact about the lattice rather "
            "than about self-repair."
            if not size_scan_ok
            else "✗ size dependence does not show the expected self-repair trend "
            "(density often falls as surface/volume; κ path may still buffer)."
        )
    )
    baseline_note = {
        "conserved": "(conserved multi-domain floor; single-domain collapse forbidden).",
        "conserved_kappa": "(conserved + κ-pinned multi-domain floor).",
        "kappa": "(κ-pinned / coarsened baseline).",
        "unconstrained": "(mean-curvature coarsening).",
    }.get(args.dynamics, ".")
    lines.append(
        f"P3 (baseline without noise): noise=0 late density = "
        f"{zero['late_mean']:.4f} — "
        + (
            f"✓ light forms sit at a low baseline under pure dynamics {baseline_note}"
            if p3
            else "✗ zero-noise baseline did not suppress light forms on this window."
        )
    )
    lines.append("")
    lines.append(
        f"score: {int(p1) + int(p2) + int(p3)}/3 pre-registered predictions land."
    )
    lines.append("")
    lines.append(
        "honest scope: dynamics options are volume-conserving multiphase, "
        "κ-gated Allen–Cahn, both combined, or unconstrained.  Additive "
        "Langevin noise supplies fluctuations; not a true thermal Gibbs "
        "measure.  Conserved fixed the ordered noise window; κ is the capacity "
        "pin that arrests coarsening where load is high (same rule as the "
        "thermal capacity program).  Density vs size is partly geometric "
        "(surface/volume); under κ dynamics P2 can also pass via a stable or "
        "rising mean capacity on fields that are still tessellated.  The "
        f"texture guard is a domain width of {MIN_DOMAIN_DIAMETER:.1f} lattice "
        "units, derived from the 3^d census neighbourhood rather than chosen, "
        "and it is a guard against texture, not a general-purpose length "
        "scale — it over-reads on anisotropic (slab-like) morphologies by "
        "roughly a factor of d.  Finite window; primary "
        f"path 3-D (ndim={args.ndim}); no continuum limit."
    )
    text = "\n".join(lines)
    print("\n" + text)

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(
        os.path.join(args.output_dir, "n3_form_persistence.json"), "w"
    ) as fh:
        json.dump(
            {
                "params": vars(args),
                "noise_scan": [
                    {k: v for k, v in r.items() if k != "runs"}
                    for r in noise_scan
                ],
                "size_scan": [
                    {k: v for k, v in r.items() if k != "runs"}
                    for r in size_scan
                ],
                "analysis": {
                    "p1": bool(p1),
                    "p2": bool(p2),
                    "p2_density": bool(p2_density),
                    "p2_kappa": bool(p2_kappa),
                    "p3": bool(p3),
                    "mid_noise": float(mid_noise),
                    "ndim": int(args.ndim),
                    "dynamics": args.dynamics,
                },
            },
            fh,
            indent=2,
            default=str,
        )


if __name__ == "__main__":
    main()
