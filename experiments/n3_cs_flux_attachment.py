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

Pre-registered predictions:

CS1. **Attachment locks flux to the core.**  After a rigid flux displacement of
     several lattice units, re-relaxation with ``γ > 0`` reduces the flux–core
     offset below a threshold (flux re-centres on the charge proxy).
CS2. **Without attachment, flux stays displaced.**  The same protocol with
     ``γ = 0`` leaves a substantially larger residual offset — the Maxwell +
     Higgs dynamics alone do not enforce flux attachment on this window.
CS3. **κ weights the binding.**  With a spatially varying κ (higher near the
     core), attachment still locks (offset small), and the attachment energy is
     lower than a uniform-κ control at the same γ — capacity modulates the cost
     of mismatch as the framework requires.

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


def run(args):
    print("CS proxy: dynamical flux–charge attachment",
          f"n={args.n} q={args.q} shift={args.shift} gamma={args.gamma}",
          flush=True)

    base, center = prepare_vortex(args)
    base_off = flux_core_offset(base["psi"], base["theta"], center=center)
    base_flux = local_flux(base["theta"], center, args.flux_radius)
    print(
        f"  baseline: offset={base_off['offset']:.3f}  "
        f"Φ/2π={base_flux / TWO_PI:.3f}",
        flush=True,
    )

    # Controlled mismatch: translate flux, hold Higgs fixed
    shifted_theta = displace_flux(
        base["theta"], (args.shift, 0)
    )
    mismatch = flux_core_offset(
        base["psi"], shifted_theta, center=center
    )
    print(
        f"  after shift by {args.shift}: offset={mismatch['offset']:.3f}",
        flush=True,
    )

    # CS1: re-relax WITH attachment
    with_att = re_relax(
        base["psi"], shifted_theta, args, gamma=args.gamma, kappa=None
    )
    off_on = flux_core_offset(with_att["psi"], with_att["theta"], center=center)
    flux_on = local_flux(with_att["theta"], center, args.flux_radius)
    print(
        f"  γ={args.gamma}: offset={off_on['offset']:.3f}  "
        f"Φ/2π={flux_on / TWO_PI:.3f}  E_att={with_att['parts']['attachment']:.4f}",
        flush=True,
    )

    # CS2: re-relax WITHOUT attachment
    without = re_relax(
        base["psi"], shifted_theta, args, gamma=0.0, kappa=None
    )
    off_off = flux_core_offset(without["psi"], without["theta"], center=center)
    flux_off = local_flux(without["theta"], center, args.flux_radius)
    print(
        f"  γ=0: offset={off_off['offset']:.3f}  "
        f"Φ/2π={flux_off / TWO_PI:.3f}",
        flush=True,
    )

    # CS3: spatially varying κ — higher near core
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
        with_kappa["psi"], with_kappa["theta"], center=center
    )
    # uniform-κ control at mean κ for energy comparison
    kappa_uniform = np.full((n, n), float(kappa.mean()))
    with_uni = re_relax(
        base["psi"], shifted_theta, args, gamma=args.gamma, kappa=kappa_uniform
    )
    print(
        f"  γ={args.gamma} + κ(core): offset={off_k['offset']:.3f}  "
        f"E_att={with_kappa['parts']['attachment']:.4f}  "
        f"(uniform κ E_att={with_uni['parts']['attachment']:.4f})",
        flush=True,
    )

    return {
        "baseline_offset": base_off["offset"],
        "mismatch_offset": mismatch["offset"],
        "offset_with_attachment": off_on["offset"],
        "offset_without": off_off["offset"],
        "offset_with_kappa": off_k["offset"],
        "flux_with": flux_on / TWO_PI,
        "flux_without": flux_off / TWO_PI,
        "E_att_kappa": with_kappa["parts"]["attachment"],
        "E_att_uniform": with_uni["parts"]["attachment"],
        "shift": args.shift,
        "gamma": args.gamma,
    }


def verdict(row, args):
    # CS1: attachment brings offset well below the imposed shift
    cs1 = (
        row["offset_with_attachment"] < 0.45 * row["mismatch_offset"]
        and row["offset_with_attachment"] < max(2.0, 0.35 * args.shift)
    )
    # CS2: without attachment, residual offset stays large relative to CS1
    cs2 = (
        row["offset_without"] > row["offset_with_attachment"] * 1.5
        and row["offset_without"] > 0.4 * args.shift
    )
    # CS3: kappa-weighted still locks, and attachment energy ≤ uniform control
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
        f"γ={args.gamma} residual offset={row['offset_with_attachment']:.3f} "
        f"(mismatch was {row['mismatch_offset']:.3f}) — "
        + (
            "✓ flux re-centres on the charge proxy under the soft CS term."
            if cs1
            else "✗ attachment did not re-lock flux on this window."
        )
    )
    lines.append(
        f"CS2 (without attachment flux stays displaced): γ=0 residual "
        f"offset={row['offset_without']:.3f} vs γ>0 offset="
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
        f"CS3 (κ weights the binding): core-weighted κ offset="
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
        "no {ψ, ψ†}, no many-body Pauli principle.  This is the first rung of "
        "the fermion arc's named boundary — dynamical flux–charge binding — not "
        "the quantum leap.  2-D, one lattice, one (λ, β, γ) operating point."
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
