"""Retarded κ-gravity: moving masses drag, radiate, and inspiral.

The κ light cone gave the capacity field a finite update rate; this
experiment couples that field to **moving masses** — the step the whole
gravity arc named as missing physics.  The telegrapher field is a *damped*
wave equation, so its energy law is exact (multiply by κ̇):

    d/dt [ T_mass + F[κ] + ∫(τ/2)κ̇² ] = −∫κ̇²  ≤ 0 ,

and for a well co-moving at speed v ≪ c_κ the quasi-static field has
``κ̇ ≈ −v·∂ₓκ``, so the dissipated power is **predictable from statics**:

    P(v) = ∫κ̇² ≈ v²·Σ(∂ₓκ_static)²  —  gravitational drag, linear response.

As v → c_κ the co-moving well can no longer keep up (it steepens and sheds
waves), and beyond c_κ a source **outruns its own field**: strict causality
says the region ahead of a supersonic mass is silent.  For a bound system
the consequence is astrophysical in shape: a binary whose orbital motion
must be continually re-encoded by a finite-rate field loses mechanical
energy and **inspirals** — the analogue of gravitational-wave decay — while
the adiabatic (instantaneous) field conserves orbital energy exactly.

Instruments: a kinematically towed mass (steady-state ∫κ̇² vs the static
prediction; probes fore and aft), and the retarded two-body integrator
(``capacity_waves.evolve_inertial_retarded``) against the adiabatic control
(``capacity_dynamics.evolve_inertial``), on the orbital experiment's own
calibrated regime (mass 1.2, c = 0.8, r = 0.02, d₀ = 12, v_circ = 0.8).

Pre-registered predictions:

G1. **The drag law — statics predict dissipation**: towed-mass power fits
    P ∝ v^p with p ∈ [1.8, 2.2] over the sub-c_κ scan, and the drag
    coefficient P/v² matches Σ(∂ₓκ_static)² within a factor of 2.
G2. **The causal wall**: P/v² rises monotonically as v → c_κ (the
    co-moving operator has D_eff = D(1−v²/c_κ²) — the well steepens as the
    source approaches the field's update rate; the direction is derivable,
    the magnitude is reported, not scored), and a supersonic mass
    (v = 1.5·c_κ) is **silent ahead**: the fore/aft disturbance ratio is
    < 0.01, measured outside the startup transient's cone.
G3. **The binary inspirals — and the adiabatic control does not**: the
    retarded binary's Lyapunov energy decreases monotonically with a net
    loss > 10× the adiabatic control's |drift| over the same span, the
    mean separation shrinks, and the *early* loss (t ≤ 60, before the
    merged floor) is ordered in the update rate: the slower field (τ = 1)
    dissipates more than the faster (τ = 0.25).

Usage::

    python experiments/n3_retarded_gravity.py --output-dir artifacts/n3_retarded
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_dynamics import evolve_inertial
from project_genesis.capacity_gravity import gaussian_load, relax_capacity
from project_genesis.capacity_waves import (
    evolve_inertial_retarded,
    step_capacity_wave,
    wave_speed,
)


def field_kwargs(args):
    return dict(kappa_diffusion=1.0, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0)


def tow_run(args, v: float, tau: float, *, start_frac: float = 0.25,
            tow_time: float | None = None) -> dict:
    """Tow one mass at constant v; return steady dissipation and probes.

    The supersonic caller starts at ``start_frac = 1/8`` and stops before
    the box edge, so the fore probe never wraps into the wake or into the
    startup transient's cone (both contaminated the first design — the
    probe must sit outside ``c_κ·t`` of the start point).
    """
    n = int(args.size)
    fkw = field_kwargs(args)
    x0 = np.array([n * start_frac, n / 2.0])
    kappa = relax_capacity(gaussian_load((n, n), [tuple(x0)], args.width,
                                         args.mass), **fkw)
    kdot = np.zeros_like(kappa)
    T = args.tow_time if tow_time is None else tow_time
    steps = int(T / args.dt)
    p_hist = []
    pos = x0.copy()
    for i in range(steps):
        pos = x0 + np.array([v * (i + 1) * args.dt, 0.0])
        load = gaussian_load((n, n), [tuple(pos % n)], args.width, args.mass)
        kappa, kdot = step_capacity_wave(kappa, kdot, load, tau=tau,
                                         dt=args.dt, **fkw)
        p_hist.append(float(np.sum(kdot ** 2)))
    p = np.asarray(p_hist)
    steady = p[int(0.6 * steps):].mean()
    # fore/aft probes, 15 cells from the final source position along x
    kbar_line = kappa[:, n // 2]
    xi = int(pos[0] % n)
    fore = abs(float(kbar_line[(xi + 15) % n]) - 1.0)
    aft = abs(float(kbar_line[(xi - 15) % n]) - 1.0)
    cone_clear = (pos[0] + 15.0) - (x0[0] + wave_speed(1.0, tau) * T)
    return {"v": v, "P": float(steady), "fore": fore, "aft": aft,
            "cone_clearance": float(cone_clear)}


def static_drag_coefficient(args) -> float:
    """Σ(∂ₓκ_static)² — the drag coefficient, from the relaxed well alone."""
    n = int(args.size)
    kappa = relax_capacity(gaussian_load((n, n), [(n / 4.0, n / 2.0)],
                                         args.width, args.mass),
                           **field_kwargs(args))
    gx = 0.5 * (np.roll(kappa, -1, 0) - np.roll(kappa, 1, 0))
    return float(np.sum(gx ** 2))


def binary_run(args, tau: float | None) -> dict:
    """Equal-mass binary; ``tau=None`` → adiabatic control."""
    L = args.size
    pos = [[L / 2 - args.d0 / 2, L / 2], [L / 2 + args.d0 / 2, L / 2]]
    vel = [[0.0, +args.v_circ], [0.0, -args.v_circ]]
    if tau is None:
        hist = evolve_inertial(pos, vel, [args.mass] * 2, shape=(L, L),
                               width=args.width, dt=args.dt_adiabatic,
                               steps=int(args.orbit_time / args.dt_adiabatic),
                               max_iters=4000, **field_kwargs(args))
        e = np.asarray(hist["energy"])
        sep = np.asarray(hist["separation"])
    else:
        hist = evolve_inertial_retarded(
            pos, vel, [args.mass] * 2, shape=(L, L), width=args.width,
            tau=tau, dt=args.dt, steps=int(args.orbit_time / args.dt),
            record_every=20, **field_kwargs(args))
        e = np.asarray(hist["energy"])
        sep = np.asarray(hist["separation"])
    q = len(e) // 8
    dt_rec = (args.dt_adiabatic if tau is None else 20 * args.dt)
    t_rec = dt_rec * np.arange(len(e))
    e_early = float(np.interp(args.early_time, t_rec, e))
    return {"tau": tau, "e0": float(e[:q].mean()), "e1": float(e[-q:].mean()),
            "loss": float(e[:q].mean() - e[-q:].mean()),
            "early_loss": float(e[0] - e_early),
            "monotone_frac": float(np.mean(np.diff(e) <= 1e-9)),
            "sep0": float(sep[:q].mean()), "sep1": float(sep[-q:].mean())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=float, default=96.0)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--mass", type=float, default=1.2)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--tau-tow", type=float, default=1.0)
    p.add_argument("--tow-speeds", type=float, nargs="+",
                   default=[0.1, 0.15, 0.2, 0.3, 0.6, 0.9])
    p.add_argument("--tow-time", type=float, default=60.0)
    p.add_argument("--d0", type=float, default=12.0)
    p.add_argument("--v-circ", type=float, default=0.8)
    p.add_argument("--orbit-time", type=float, default=280.0)
    p.add_argument("--taus-binary", type=float, nargs="+", default=[0.25, 1.0])
    p.add_argument("--early-time", type=float, default=60.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--dt-adiabatic", type=float, default=0.5)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_retarded")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tow_speeds = [0.1, 0.2, 0.9]
        args.tow_time, args.orbit_time = 40.0, 120.0

    os.makedirs(args.output_dir, exist_ok=True)
    print("retarded kappa-gravity: drag, causal wall, inspiral", flush=True)
    c_kappa = wave_speed(1.0, args.tau_tow)

    # --- G1/G2: the towed mass
    coeff_static = static_drag_coefficient(args)
    print(f"  static drag coefficient Σ(∂ₓκ)² = {coeff_static:.5f}",
          flush=True)
    tows = [tow_run(args, v, args.tau_tow) for v in args.tow_speeds]
    for t in tows:
        print(f"  v={t['v']:<5}: P={t['P']:.3e}  P/v²={t['P']/t['v']**2:.5f}",
              flush=True)
    sub = [t for t in tows if t["v"] <= 0.5 * c_kappa]
    lv = np.log([t["v"] for t in sub])
    lp = np.log([t["P"] for t in sub])
    p_exp = float(np.polyfit(lv, lp, 1)[0])
    coeff_meas = float(np.mean([t["P"] / t["v"] ** 2 for t in sub]))
    v_super = 1.5 * c_kappa
    n = args.size
    t_super = (0.80 * n - n / 8.0 - 20.0) / v_super
    super_tow = tow_run(args, v_super, args.tau_tow,
                        start_frac=0.125, tow_time=t_super)
    fore_aft = super_tow["fore"] / super_tow["aft"]
    print(f"  supersonic v={v_super:g}: fore/aft = {fore_aft:.2e} "
          f"(cone clearance {super_tow['cone_clearance']:.1f} cells)",
          flush=True)

    g1 = (1.8 <= p_exp <= 2.2
          and 0.5 <= coeff_meas / coeff_static <= 2.0)
    ratios = [t["P"] / t["v"] ** 2 for t in tows]
    g2 = (all(b >= a * 0.98 for a, b in zip(ratios, ratios[1:]))
          and fore_aft < 0.01)

    # --- G3: the binary
    runs = {tau: binary_run(args, tau) for tau in args.taus_binary}
    control = binary_run(args, None)
    for tau, rr in runs.items():
        print(f"  binary τ={tau}: E {rr['e0']:.4f}→{rr['e1']:.4f} "
              f"(loss {rr['loss']:.4f}, monotone {rr['monotone_frac']:.2f}), "
              f"sep {rr['sep0']:.1f}→{rr['sep1']:.1f}", flush=True)
    print(f"  adiabatic control: |ΔE| = {abs(control['loss']):.5f}, "
          f"sep {control['sep0']:.1f}→{control['sep1']:.1f}", flush=True)

    slow, fast = runs[max(runs)], runs[min(runs)]
    g3 = (all(rr["monotone_frac"] > 0.9 for rr in runs.values())
          and all(rr["loss"] > 10.0 * abs(control["loss"])
                  for rr in runs.values())
          and all(rr["sep1"] < rr["sep0"] for rr in runs.values())
          and slow["early_loss"] > fast["early_loss"])

    lines = ["Retarded κ-gravity — verdict", "=" * 74, ""]
    lines.append(
        f"G1 (the drag law): P ∝ v^{p_exp:.2f} (target 2), coefficient "
        f"{coeff_meas:.5f} vs static Σ(∂ₓκ)² = {coeff_static:.5f} "
        f"(ratio {coeff_meas/coeff_static:.2f}) — "
        + ("✓ gravitational drag, predicted from the relaxed well alone."
           if g1 else "✗ the linear-response drag law fails."))
    lines.append(
        f"G2 (the causal wall): P/v² rises monotonically "
        f"{ratios[0]:.5f} → {ratios[-1]:.5f} toward c_κ "
        f"(D_eff = D(1−v²/c_κ²) steepens the co-moving well); supersonic "
        f"fore/aft = {fore_aft:.1e} — "
        + ("✓ the field stiffens as the source approaches its update rate, "
           "and ahead of a supersonic mass it is silent." if g2 else
           "✗ no causal stiffening/silence resolved."))
    lines.append(
        f"G3 (the inspiral): retarded losses {fast['loss']:.4f} (τ = "
        f"{min(runs):g}) and {slow['loss']:.4f} (τ = {max(runs):g}) vs "
        f"adiabatic drift {abs(control['loss']):.5f}; separations "
        f"{fast['sep0']:.1f}→{fast['sep1']:.1f}, "
        f"{slow['sep0']:.1f}→{slow['sep1']:.1f}; early loss (t ≤ "
        f"{args.early_time:g}) {fast['early_loss']:.3f} vs "
        f"{slow['early_loss']:.3f} — "
        + ("✓ finite update rate makes binaries decay — the analogue's "
           "gravitational-wave inspiral, with the adiabatic control "
           "conserving and the slower field losing faster." if g3 else
           "✗ no clean retarded decay above the adiabatic drift."))
    lines.append("")
    lines.append(f"score: {int(g1)+int(g2)+int(g3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: the drag is the damped-telegrapher's ∫κ̇² — this κ "
        "theory dissipates in the field (a resistive medium), so the decay "
        "mixes drag and wave emission rather than isolating far-zone "
        "radiation (the GR-style flux split is the harder follow-up); 2-D, "
        "one mass/width; the orbital regime is inherited from "
        "n3_orbital_gravity; supersonic silence is a strict-causality "
        "check, not a Mach-cone tomography; the Lyapunov energy uses the "
        "same F[κ] as the adiabatic instrument so the control comparison "
        "is like-for-like.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_retarded_gravity.json"),
              "w") as fh:
        json.dump({"params": vars(args), "tows": tows,
                   "supersonic": super_tow,
                   "binary": {str(k): v for k, v in runs.items()},
                   "control": control,
                   "analysis": {"g1": bool(g1), "g2": bool(g2),
                                "g3": bool(g3), "p_exp": p_exp,
                                "coeff_meas": coeff_meas,
                                "coeff_static": coeff_static,
                                "fore_aft": fore_aft}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
