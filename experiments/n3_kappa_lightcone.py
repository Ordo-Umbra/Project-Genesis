"""The κ light cone: the field's update rate as the speed limit, measured.

Everywhere in the framework so far, κ-gravity acts **instantaneously** — the
capacity field is relaxed adiabatically before masses move, a named piece of
missing physics.  ``capacity_waves`` gives κ the minimal honest extension: a
finite update latency τ (the telegrapher equation), under which the field
acquires its own **causal cone** with front speed ``c_κ = √(D_κ/τ)`` — the
update rate of the field *is* its speed limit — and the linearised loaded
mode carries a **mass** ``m² = r + c·ρ``: the same Debye term that screens
static κ-gravity (screening knee, local screening, environmental growth),
now governing *propagation*.  Waves are massive inside matter; the massless
channel exists exactly where ``r + c·ρ → 0``.

Instruments:

- **Front tracking**: a small Gaussian δκ bump released at the centre of an
  empty periodic box; the front radius r_f(t) (outermost threshold crossing)
  is fitted for speed and for the ballistic-vs-diffusive exponent
  ``r_f ∝ t^p``.  The τ → 0 (parabolic) control runs the identical protocol.
- **Standing-wave dispersion**: a plane-wave δκ on a loaded background; the
  mode amplitude a(t) oscillates at ω(k) (zero-crossing estimator), giving
  the dispersion relation and its density gap.

Pre-registered predictions:

W1. **The cone exists, and τ is its condition**: with finite τ the front
    moves at constant speed (two-window speed ratio v_late/v_early within
    [0.8, 1.25] at every τ); the parabolic control under the identical
    protocol decelerates (ratio < 0.6) — the condition toggled, as the
    criticality transplant demands of a mechanism claim.
W2. **The update rate is the speed limit**: across a τ-scan spanning two
    decades, the measured front speed matches ``√(D_κ/τ)`` within a factor
    of 2 at every τ, with log-log slope −0.5 ± 0.1.
W3. **The Debye mass propagates**: on loaded backgrounds the standing-wave
    frequency gap grows with density — ``ω²(ρ)`` at fixed k is linear with
    slope ``c/τ`` within a factor of 2 — the matter term measured a fourth
    way, now in the field's *dynamics*: the same number that shortens
    gravity's range gives its waves a mass.

Usage::

    python experiments/n3_kappa_lightcone.py --output-dir artifacts/n3_cone
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_waves import (
    dispersion_omega,
    front_radius,
    steady_kappa,
    step_capacity_parabolic,
    step_capacity_wave,
    wave_speed,
)


def front_run(args, tau: float | None) -> dict:
    """Release a δκ bump in an empty box; track r_f(t).  ``tau=None`` → parabolic.

    The threshold contour starts at the bump's own tail radius
    ``r₀ = w·√(2·ln(bump/threshold))``, so speeds are fitted beyond r₀ + 2,
    and ballistic-vs-diffusive is decided by the **two-window speed ratio**
    (constant speed vs deceleration) — immune to the r₀ offset that biases
    a naive log–log exponent.
    """
    n = args.front_size
    kappa = np.full((n, n), 1.0)
    grids = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    d2 = sum((g - n / 2.0) ** 2 for g in grids)
    kappa += args.bump * np.exp(-d2 / (2.0 * args.bump_width ** 2))
    kdot = np.zeros_like(kappa)
    r0 = args.bump_width * np.sqrt(2.0 * np.log(args.bump / args.threshold))
    if tau:
        speed = wave_speed(1.0, tau)
        horizon = 0.32 * n / speed
        dt = min(args.dt, 0.4 / speed)
    else:
        horizon, dt = args.par_time, args.dt
    steps = int(horizon / dt)
    kw = dict(kappa_recovery=args.recovery, kappa_consumption=2.0, dt=dt)
    times, radii = [], []
    for i in range(steps):
        if tau:
            kappa, kdot = step_capacity_wave(kappa, kdot, 0.0, tau=tau, **kw)
        else:
            kappa = step_capacity_parabolic(kappa, 0.0, **kw)
        if i % max(1, steps // 80) == 0:
            times.append((i + 1) * dt)
            radii.append(front_radius(kappa - steady_kappa(
                args.recovery, 2.0, 0.0), (n / 2.0, n / 2.0),
                args.threshold))
    t, rf = np.asarray(times), np.asarray(radii)
    sel = (rf > r0 + 2.0) & (rf < 0.32 * n)
    v, _ = np.polyfit(t[sel], rf[sel], 1)
    t_mid = float(np.median(t[sel]))
    early, late = sel & (t <= t_mid), sel & (t > t_mid)
    v_early = float(np.polyfit(t[early], rf[early], 1)[0])
    v_late = float(np.polyfit(t[late], rf[late], 1)[0])
    return {"tau": tau, "v": float(v), "v_early": v_early, "v_late": v_late,
            "ratio": v_late / v_early, "n_fit": int(sel.sum()),
            "t": t.tolist(), "r": rf.tolist()}


def mode_frequency(args, rho: float, k: float) -> float:
    """ω of a standing δκ plane wave on a uniform load ρ (zero crossings)."""
    n = args.strip_len
    kbar = steady_kappa(args.recovery, 2.0, rho)
    x = np.arange(n)
    kappa = kbar + args.bump * np.cos(k * x)[:, None] * np.ones((1, 4))
    kdot = np.zeros_like(kappa)
    proj = np.cos(k * x)
    dt = args.dt
    steps = int(args.disp_time / dt)
    a_hist = []
    for _ in range(steps):
        kappa, kdot = step_capacity_wave(
            kappa, kdot, rho, tau=args.tau_disp,
            kappa_recovery=args.recovery, kappa_consumption=2.0, dt=dt)
        a_hist.append(2.0 * float(np.mean((kappa[:, 0] - kbar) * proj)))
    a = np.asarray(a_hist)
    t = dt * (1.0 + np.arange(len(a)))
    crossings = []
    for i in range(1, len(a)):
        if a[i - 1] * a[i] < 0:
            crossings.append(t[i - 1] + dt * a[i - 1] / (a[i - 1] - a[i]))
    if len(crossings) < 3:
        return float("nan")
    half_period = float(np.mean(np.diff(crossings)))
    return np.pi / half_period


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--front-size", type=int, default=160)
    p.add_argument("--taus", type=float, nargs="+",
                   default=[4.0, 16.0, 64.0, 256.0])
    p.add_argument("--bump", type=float, default=1e-3)
    p.add_argument("--bump-width", type=float, default=2.0)
    p.add_argument("--threshold", type=float, default=1e-9)
    p.add_argument("--par-time", type=float, default=60.0)
    p.add_argument("--recovery", type=float, default=0.05)
    p.add_argument("--tau-disp", type=float, default=64.0)
    p.add_argument("--strip-len", type=int, default=256)
    p.add_argument("--k-mode", type=float, default=2.0 * np.pi / 32.0)
    p.add_argument("--rhos", type=float, nargs="+",
                   default=[0.0, 0.05, 0.1, 0.2])
    p.add_argument("--disp-time", type=float, default=1200.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_cone")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.front_size, args.taus = 96, [4.0, 64.0]
        args.rhos, args.disp_time = [0.0, 0.2], 600.0

    os.makedirs(args.output_dir, exist_ok=True)
    print("the kappa light cone: update rate as speed limit", flush=True)

    # --- W1/W2: front tracking, hyperbolic scan + parabolic control
    fronts = [front_run(args, tau) for tau in args.taus]
    for f in fronts:
        c_th = wave_speed(1.0, f["tau"])
        print(f"  τ={f['tau']:<5}: v={f['v']:.3f} (√(D/τ) = {c_th:.3f}), "
              f"v_late/v_early={f['ratio']:.2f}", flush=True)
    control = front_run(args, None)
    print(f"  parabolic control: v_late/v_early={control['ratio']:.2f}",
          flush=True)

    # --- W3: dispersion gap vs density
    disp = []
    for rho in args.rhos:
        w = mode_frequency(args, rho, args.k_mode)
        disp.append({"rho": rho, "omega": w,
                     "omega_theory": dispersion_omega(
                         args.k_mode, tau=args.tau_disp,
                         kappa_recovery=args.recovery, rho=rho)})
        print(f"  ρ={rho:<5}: ω={w:.4f} (theory "
              f"{disp[-1]['omega_theory']:.4f})", flush=True)
    rhos = np.array([d["rho"] for d in disp])
    w2 = np.array([d["omega"] ** 2 for d in disp])
    ok = np.isfinite(w2)
    gap_slope, _ = np.polyfit(rhos[ok], w2[ok], 1)
    gap_theory = 2.0 / args.tau_disp                      # c/τ

    w1 = (all(0.8 <= f["ratio"] <= 1.25 for f in fronts)
          and control["ratio"] < 0.6)
    v_ratios = [f["v"] / wave_speed(1.0, f["tau"]) for f in fronts]
    ln_t = np.log([f["tau"] for f in fronts])
    ln_v = np.log([f["v"] for f in fronts])
    slope_v = float(np.polyfit(ln_t, ln_v, 1)[0])
    w2ok = (all(0.5 <= q <= 2.0 for q in v_ratios)
            and abs(slope_v - (-0.5)) <= 0.1)
    w3 = 0.5 <= gap_slope / gap_theory <= 2.0

    lines = ["The κ light cone — verdict", "=" * 74, ""]
    lines.append(
        "W1 (the cone exists, τ its condition): front speed constancy "
        "v_late/v_early = "
        + ", ".join(f"{f['ratio']:.2f}" for f in fronts)
        + f" (ballistic) vs parabolic control {control['ratio']:.2f} "
        "(decelerating) — "
        + ("✓ finite τ makes propagation ballistic; removing it restores "
           "diffusion — the condition toggled." if w1 else
           "✗ no clean ballistic/diffusive separation."))
    lines.append(
        "W2 (the update rate is the speed limit): v/√(D/τ) = "
        + ", ".join(f"{q:.2f}" for q in v_ratios)
        + f", log-log slope {slope_v:.2f} vs −0.5 — "
        + ("✓ the front moves at the field's own update rate across two "
           "decades of τ." if w2ok else "✗ the speed does not track √(D/τ)."))
    lines.append(
        f"W3 (the Debye mass propagates): dω²/dρ = {gap_slope:.5f} vs "
        f"c/τ = {gap_theory:.5f} — "
        + ("✓ the same matter term that shortens static gravity's range "
           "gives the propagating wave its mass — measured a fourth way."
           if w3 else "✗ the density gap does not match the loaded law."))
    lines.append("")
    lines.append(f"score: {int(w1)+int(w2ok)+int(w3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: linear-response amplitudes (unclipped integrator, "
        "δκ = 1e-3); the numerical lattice bound (1 cell/step) sits far "
        "above every c_κ tested, so the measured limit is the field's, not "
        "the integrator's; 2-D fronts carry the usual 2-D wake (Huygens "
        "fails in 2-D — the FRONT is what is fitted); one D value; the "
        "dispersion estimator needs Q = 2ωτ ≳ 2, so the gap scan runs at "
        "large τ. Gravity experiments elsewhere in the repo remain "
        "adiabatic — coupling the finite-speed field to moving masses "
        "(retarded κ-gravity) is the registered next step.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_kappa_lightcone.json"),
              "w") as fh:
        json.dump({"params": vars(args),
                   "fronts": [{k: v for k, v in f.items()
                               if k not in ("t", "r")} for f in fronts],
                   "control_ratio": control["ratio"], "dispersion": disp,
                   "analysis": {"w1": bool(w1), "w2": bool(w2ok),
                                "w3": bool(w3), "v_ratios": v_ratios,
                                "slope_v": slope_v,
                                "gap_slope": float(gap_slope),
                                "gap_theory": gap_theory}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
