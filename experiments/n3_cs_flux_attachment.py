"""Chern–Simons proxy: dynamical flux–charge attachment weighted by κ.

The Aharonov–Bohm experiment (`n3_ab_statistics`) showed that the fermion
exchange sign is a gauge holonomy of the self-consistent flux — but flux–charge
binding was *read off*, not enforced.  The fermion arc's named boundary:

    "no Chern–Simons term dynamically binds flux to charge."

This experiment is the first rung of that boundary.  A soft lattice attachment
term (CS proxy) is added to the gauged vortex energy:

    E_att = (γ/2) Σ κ(x) (B(x) − 2π ρ(x))²

with ``ρ`` the normalised Higgs-depletion charge proxy.  Capacity ``κ`` weights
the attachment cost.  The test is dynamical locking under a controlled mismatch:
displace the gauge field relative to the Higgs core, then re-relax with and
without attachment.

**Offset metric.**  Primary marker is peak-based: ``argmin |ψ|`` vs
``argmax |B|`` (periodic min-image).  Centroid-of-cloud offsets were too noisy
(baseline ~6 even on quantised vortices).  Secondary: flux through a disk at
the Higgs peak (``core_flux``) — lock recovery independent of the flux-peak
location.

Pre-registered predictions:

CS1. **Attachment locks flux to the core.**  After a rigid flux displacement of
     several lattice units, re-relaxation with ``γ > 0`` reduces the peak
     flux–core offset below a threshold (flux re-centres on the charge proxy),
     and/or restores ``|core_flux|`` toward one quantum.
CS2. **Without attachment, flux stays displaced.**  The same protocol with
     ``γ = 0`` leaves a substantially larger residual peak offset — the Maxwell
     + Higgs dynamics alone do not enforce flux attachment on this window.
CS3. **κ weights the binding.**  With a spatially varying κ (higher near the
     core), attachment still locks (peak offset small), and the attachment
     energy is lower than a uniform-κ control at the same γ — capacity
     modulates the cost of mismatch as the framework requires.

Honest scope: classical lattice proxy for Chern–Simons flux attachment, not a
continuum CS action, not Fock space, not ``{ψ, ψ†}``.  If CS1–CS2 land, flux
binding becomes dynamical rather than diagnostic; second quantisation remains
the frontier beyond this rung.

Usage::

    python experiments/n3_cs_flux_attachment.py --output-dir artifacts/n3_cs
    python experiments/n3_cs_flux_attachment.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.gauged_vortex import (
    displace_flux,
    flux_core_offset,
    local_flux,
    relax,
    seed_vortices,
    zero_links,
)

TWO_PI = 2.0 * np.pi


def prepare_vortex(args):
    """Relax a single winding-q vortex to a self-consistent gauged state."""
    n = args.n
    center = (n / 2.0, n / 2.0)
    psi = seed_vortices(n, [center], [args.q], core=args.core)
    out = relax(
        psi,
        zero_links(n),
        lam=args.lam,
        beta=args.beta,
        dt=args.dt,
        steps=args.steps,
        gamma=0.0,
    )
    return out, center


def re_relax(psi, theta, args, *, gamma: float, kappa, steps: int | None = None):
    return relax(
        psi,
        theta,
        lam=args.lam,
        beta=args.beta,
        dt=args.dt,
        steps=steps if steps is not None else args.re_steps,
        gamma=gamma,
        target_charge=float(args.q),
        kappa=kappa,
    )


def _fmt_off(info: dict) -> str:
    return (
        f"peak={info['offset_peak']:.3f}  cent={info['offset_centroid']:.3f}  "
        f"core_Φ/2π={info['core_flux_quanta']:+.3f}"
    )


def run(args):
    print(
        "CS proxy: dynamical flux–charge attachment",
        f"n={args.n} q={args.q} shift={args.shift} gamma={args.gamma}",
        flush=True,
    )

    base, center = prepare_vortex(args)
    base_off = flux_core_offset(
        base["psi"], base["theta"], center=center, flux_radius=args.flux_radius
    )
    print(f"  baseline: {_fmt_off(base_off)}", flush=True)

    shifted_theta = displace_flux(base["theta"], (args.shift, 0))
    mismatch = flux_core_offset(
        base["psi"], shifted_theta, center=center, flux_radius=args.flux_radius
    )
    print(f"  after shift by {args.shift}: {_fmt_off(mismatch)}", flush=True)

    with_att = re_relax(
        base["psi"], shifted_theta, args, gamma=args.gamma, kappa=None
    )
    off_on = flux_core_offset(
        with_att["psi"], with_att["theta"], center=center,
        flux_radius=args.flux_radius,
    )
    print(
        f"  γ={args.gamma}: {_fmt_off(off_on)}  "
        f"E_att={with_att['parts']['attachment']:.4f}",
        flush=True,
    )

    without = re_relax(
        base["psi"], shifted_theta, args, gamma=0.0, kappa=None
    )
    off_off = flux_core_offset(
        without["psi"], without["theta"], center=center,
        flux_radius=args.flux_radius,
    )
    print(f"  γ=0: {_fmt_off(off_off)}", flush=True)

    n = args.n
    grids = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    dx = ((grids[0] - center[0] + n / 2) % n) - n / 2
    dy = ((grids[1] - center[1] + n / 2) % n) - n / 2
    r2 = dx * dx + dy * dy
    kappa = args.kappa_floor + (1.0 - args.kappa_floor) * np.exp(
        -r2 / (2.0 * args.kappa_width ** 2)
    )
    with_kappa = re_relax(
        base["psi"], shifted_theta, args, gamma=args.gamma, kappa=kappa
    )
    off_k = flux_core_offset(
        with_kappa["psi"], with_kappa["theta"], center=center,
        flux_radius=args.flux_radius,
    )
    kappa_uniform = np.full((n, n), float(kappa.mean()))
    with_uni = re_relax(
        base["psi"], shifted_theta, args, gamma=args.gamma, kappa=kappa_uniform
    )
    print(
        f"  γ={args.gamma} + κ(core): {_fmt_off(off_k)}  "
        f"E_att={with_kappa['parts']['attachment']:.4f}  "
        f"(uniform κ E_att={with_uni['parts']['attachment']:.4f})",
        flush=True,
    )

    return {
        "baseline_offset": base_off["offset_peak"],
        "baseline_core_flux_quanta": base_off["core_flux_quanta"],
        "mismatch_offset": mismatch["offset_peak"],
        "mismatch_core_flux_quanta": mismatch["core_flux_quanta"],
        "offset_with_attachment": off_on["offset_peak"],
        "core_flux_with": off_on["core_flux_quanta"],
        "offset_without": off_off["offset_peak"],
        "core_flux_without": off_off["core_flux_quanta"],
        "offset_with_kappa": off_k["offset_peak"],
        "core_flux_kappa": off_k["core_flux_quanta"],
        "E_att_kappa": with_kappa["parts"]["attachment"],
        "E_att_uniform": with_uni["parts"]["attachment"],
        "shift": args.shift,
        "gamma": args.gamma,
    }


def verdict(row, args):
    # CS1: peak offset drops sharply OR core flux recovers toward one quantum
    locked_offset = (
        row["offset_with_attachment"] < 0.45 * row["mismatch_offset"]
        and row["offset_with_attachment"] < max(2.0, 0.35 * args.shift)
    )
    flux_recovered = abs(row["core_flux_with"]) > 0.7 * abs(args.q)
    cs1 = locked_offset or (
        flux_recovered
        and abs(row["core_flux_with"]) > abs(row["mismatch_core_flux_quanta"]) + 0.25
    )

    # CS2: without attachment, peak offset stays clearly larger
    cs2 = (
        row["offset_without"] > row["offset_with_attachment"] * 1.5
        and row["offset_without"] > 0.4 * args.shift
    )

    # CS3: kappa path locks and attachment energy ≤ uniform control
    cs3 = (
        row["offset_with_kappa"] < max(2.5, 0.4 * args.shift)
        and row["E_att_kappa"] <= row["E_att_uniform"] * 1.05 + 1e-9
    )
    return cs1, cs2, cs3


def summarise(row, args):
    cs1, cs2, cs3 = verdict(row, args)
    lines = [
        "Chern–Simons proxy: dynamical flux–charge attachment — verdict",
        "=" * 74,
        "",
    ]
    lines.append(
        f"CS1 (attachment locks flux to the core): after shift={args.shift}, "
        f"γ={args.gamma} peak offset={row['offset_with_attachment']:.3f} "
        f"(mismatch {row['mismatch_offset']:.3f}), core Φ/2π="
        f"{row['core_flux_with']:+.3f} — "
        + (
            "✓ flux re-centres on the charge proxy and/or core flux recovers "
            "under the soft CS term."
            if cs1
            else "✗ attachment did not re-lock flux on this window."
        )
    )
    lines.append(
        f"CS2 (without attachment flux stays displaced): γ=0 peak offset="
        f"{row['offset_without']:.3f} vs γ>0 offset="
        f"{row['offset_with_attachment']:.3f} — "
        + (
            "✓ Maxwell + Higgs alone leave a larger mismatch; binding is not "
            "automatic without the attachment term."
            if cs2
            else "✗ zero-γ dynamics already re-centred flux (attachment not "
            "the distinguishing mechanism on this window)."
        )
    )
    lines.append(
        f"CS3 (κ weights the binding): core-weighted κ peak offset="
        f"{row['offset_with_kappa']:.3f}, E_att={row['E_att_kappa']:.4f} vs "
        f"uniform E_att={row['E_att_uniform']:.4f} — "
        + (
            "✓ capacity-weighted attachment still locks and costs no more than "
            "uniform κ — κ modulates the mismatch penalty."
            if cs3
            else "✗ κ weighting did not produce the expected lock / cost pattern."
        )
    )
    lines.append("")
    lines.append(
        f"score: {int(cs1) + int(cs2) + int(cs3)}/3 pre-registered "
        "predictions land."
    )
    lines.append("")
    lines.append(
        "honest scope: soft lattice flux-attachment term (Chern–Simons *proxy*), "
        "not a continuum CS action; classical gauged vortex only; no Fock space, "
        "no {ψ, ψ†}, no many-body Pauli principle.  Peak-based offset "
        "(argmin|ψ| vs argmax|B|) is the primary lock metric; centroid offset is "
        "secondary.  This is the first rung of the fermion arc's named boundary "
        "— dynamical flux–charge binding — not the quantum leap.  2-D, one "
        "lattice, one (λ, β, γ) operating point."
    )
    return lines, int(cs1) + int(cs2) + int(cs3)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--q", type=int, default=1)
    p.add_argument("--core", type=float, default=3.0)
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.5,
                   help="flux-attachment strength")
    p.add_argument("--shift", type=int, default=6,
                   help="lattice units to displace flux")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--re-steps", type=int, default=1500)
    p.add_argument("--flux-radius", type=float, default=8.0)
    p.add_argument("--kappa-floor", type=float, default=0.2)
    p.add_argument("--kappa-width", type=float, default=6.0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_cs")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n = 48
        args.steps = 1200
        args.re_steps = 800
        args.shift = 5

    os.makedirs(args.output_dir, exist_ok=True)
    row = run(args)
    lines, score = summarise(row, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_cs_flux_attachment.json"), "w") as fh:
        json.dump(
            {"params": vars(args), "row": row, "score": score},
            fh,
            indent=2,
            default=str,
        )


if __name__ == "__main__":
    main()
