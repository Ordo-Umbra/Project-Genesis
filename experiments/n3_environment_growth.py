"""Environmental growth: the static field predicts the dynamics.

Local screening established, statically, that gravity's range is set by the
capacity environment where the disturbance lives.  This experiment closes
the loop dynamically: **one N-body universe with two environments** — a
dense band and a sparse band of particles (6× mass contrast) — with the same
plane-wave perturbation planted through both, and the growth rate measured
*per band* in a single run.  Bands (not wave packets) keep the mode
spectrally exact; the perturbation is along x, the environment varies in y.

The prediction is assembled **entirely from static measurements** on the
initial relaxed field — no dynamical fitting, no free constants:

    S_band ∝ ρ_band · κ̄_band² · (kℓ_band)²/(1+(kℓ_band)²)

with ρ (mean load), κ̄ (mean capacity), and ℓ (probe-response range, the
local-screening instrument re-used verbatim) each measured per band.  The
ratio ``R(λ) = S_dense/S_sparse`` then has no normalisation freedom at all
(the footprint factor cancels at equal λ).  The loaded law makes a
counterintuitive call: κ̄² coupling suppression and a shorter local range
beat the 6× density advantage, so the dense band grows **slower** —
gravitational near-sightedness wins over extra mass.

Pre-registered predictions:

E1. **Near-sightedness wins**: in the same run, R = S_dense/S_sparse < 0.8
    at both wavelengths — six times the matter, slower growth, because the
    matter's own capacity consumption weakens (κ̄²) and shortens (ℓ) its
    gravity.
E2. **The static field predicts the dynamics**: measured R within a factor
    of 2 of the statically-assembled prediction at both wavelengths —
    cross-instrument closure between the relaxed-field instruments and the
    N-body growth.
E3. **The environment × scale interaction**: R(λ_short) > R(λ_long), with
    the double ratio within a factor of 2 of the screen-only prediction —
    in the double ratio ρ and κ̄ cancel exactly, isolating the *locality of
    the range* from any coupling-suppression modelling.

Usage::

    python experiments/n3_environment_growth.py --output-dir artifacts/n3_env
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.capacity_dynamics import evolve_cosmological
from project_genesis.capacity_gravity import gaussian_load, relax_capacity
from n3_local_screening import fit_range, probe_response


def particle_grid(args):
    """Positions, masses, and band masks (dense: y < box/2)."""
    L, n = args.box, args.grid
    q = np.linspace(0, L, n, endpoint=False) + L / (2 * n)
    qx, qy = np.meshgrid(q, q, indexing="ij")
    qx, qy = qx.ravel(), qy.ravel()
    dense = qy < L / 2.0
    masses = np.where(dense, args.mass_dense, args.mass_sparse)
    margin = L / 8.0                      # measure central half of each band
    mask_d = dense & (qy >= margin) & (qy < L / 2.0 - margin)
    mask_s = ~dense & (qy >= L / 2.0 + margin) & (qy < L - margin)
    return qx, qy, masses, mask_d, mask_s


def static_side(args, qx, qy, masses, mask_d, mask_s) -> dict:
    """ρ, κ̄, and probe-fitted local range ℓ, per band — statics only."""
    shape = (int(args.box), int(args.box))
    load = np.zeros(shape)
    for m in np.unique(masses):
        centers = [(x, y) for x, y, mi in zip(qx, qy, masses) if mi == m]
        load += gaussian_load(shape, centers, args.width, float(m))
    kappa = relax_capacity(load, kappa_recovery=args.recovery,
                           max_iters=20000, tol=1e-10)
    out = {}
    probe_args = argparse.Namespace(
        recovery=args.recovery, probe_mass=0.002, probe_width=2.0,
        fit_min=4, fit_max=12)
    for name, mask, center in (("dense", mask_d, (args.box / 2, args.box / 4)),
                               ("sparse", mask_s,
                                (args.box / 2, 3 * args.box / 4))):
        rows = (np.arange(shape[1]) >= qy[mask].min() - 2.0) \
            & (np.arange(shape[1]) <= qy[mask].max() + 2.0)
        out[name] = {
            "rho": float(load[:, rows].mean()),
            "kappa_bar": float(kappa[:, rows].mean()),
            "ell": fit_range(probe_response(load, center, probe_args),
                             center, probe_args)["xi"],
        }
    return out


def band_growth(args, wavelength: float, qx, qy, masses,
                mask_d, mask_s) -> dict:
    """One static run; per-band first-window growth rate S of the mode."""
    L = args.box
    k = 2.0 * np.pi / wavelength
    psi = (args.amplitude / k) * np.sin(k * qx)
    pos = np.stack([qx + psi, qy], axis=-1)
    hist = evolve_cosmological(
        pos.tolist(), masses.tolist(), shape=(L, L), width=args.width,
        hubble0=1e-9, omega_m=1.0, omega_lambda=0.0, p=3.0,
        dt=args.dt, steps=args.steps, center=[L / 2.0, L / 2.0],
        kappa_recovery=args.recovery)
    out = {}
    for name, mask in (("dense", mask_d), ("sparse", mask_s)):
        d_hist = []
        for snap in hist["positions"]:
            disp = np.asarray(snap)[:, 0] - qx
            delta = 2.0 * k * float(np.mean(np.sin(k * qx[mask])
                                            * disp[mask]))
            d_hist.append(delta / args.amplitude)
        d = np.asarray(d_hist)
        t = args.dt * np.arange(len(d))
        inside = (d > 1.1) & (d < 3.0)
        first = np.argmax(inside)
        end = first
        while end < len(d) and inside[end]:
            end += 1
        window = np.zeros(len(d), dtype=bool)
        window[first:end] = inside[first:end]
        if window.sum() < 6:
            out[name] = {"S": float("nan"), "D": d.tolist()}
            continue
        slope = float(np.polyfit(t[window], np.log(d[window]), 1)[0])
        out[name] = {"S": slope ** 2, "n_window": int(window.sum()),
                     "D_final": float(d[-1])}
    return out


def screen(k: float, ell: float) -> float:
    return k ** 2 * ell ** 2 / (1.0 + k ** 2 * ell ** 2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--box", type=float, default=96.0)
    p.add_argument("--grid", type=int, default=24)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--amplitude", type=float, default=0.10)
    p.add_argument("--mass-dense", type=float, default=0.06)
    p.add_argument("--mass-sparse", type=float, default=0.01)
    p.add_argument("--recovery", type=float, default=0.05)
    p.add_argument("--lambda-short", type=float, default=12.0)
    p.add_argument("--lambda-long", type=float, default=32.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=220)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_env")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.steps = 120

    os.makedirs(args.output_dir, exist_ok=True)
    print("environmental growth: the static field predicts the dynamics",
          flush=True)
    qx, qy, masses, mask_d, mask_s = particle_grid(args)

    stat = static_side(args, qx, qy, masses, mask_d, mask_s)
    print(f"  static dense : ρ={stat['dense']['rho']:.4f} "
          f"κ̄={stat['dense']['kappa_bar']:.3f} ℓ={stat['dense']['ell']:.2f}",
          flush=True)
    print(f"  static sparse: ρ={stat['sparse']['rho']:.4f} "
          f"κ̄={stat['sparse']['kappa_bar']:.3f} "
          f"ℓ={stat['sparse']['ell']:.2f}", flush=True)

    runs, r_meas, r_pred = {}, {}, {}
    for tag, lam in (("short", args.lambda_short), ("long", args.lambda_long)):
        g = band_growth(args, lam, qx, qy, masses, mask_d, mask_s)
        runs[tag] = {"wavelength": lam, **g}
        r_meas[tag] = g["dense"]["S"] / g["sparse"]["S"]
        k = 2.0 * np.pi / lam
        r_pred[tag] = ((stat["dense"]["rho"] / stat["sparse"]["rho"])
                       * (stat["dense"]["kappa_bar"]
                          / stat["sparse"]["kappa_bar"]) ** 2
                       * screen(k, stat["dense"]["ell"])
                       / screen(k, stat["sparse"]["ell"]))
        print(f"  λ={lam:g}: S_dense={g['dense']['S']:.5f} "
              f"S_sparse={g['sparse']['S']:.5f} → R={r_meas[tag]:.3f} "
              f"(static prediction {r_pred[tag]:.3f})", flush=True)

    e1 = all(np.isfinite(r_meas[t]) and r_meas[t] < 0.8
             for t in ("short", "long"))
    e2 = all(0.5 <= r_meas[t] / r_pred[t] <= 2.0 for t in ("short", "long"))
    dbl_meas = r_meas["short"] / r_meas["long"]
    k_s, k_l = (2.0 * np.pi / args.lambda_short,
                2.0 * np.pi / args.lambda_long)
    dbl_pred = ((screen(k_s, stat["dense"]["ell"])
                 / screen(k_s, stat["sparse"]["ell"]))
                / (screen(k_l, stat["dense"]["ell"])
                   / screen(k_l, stat["sparse"]["ell"])))
    e3 = dbl_meas > 1.0 and 0.5 <= dbl_meas / dbl_pred <= 2.0

    lines = ["Environmental growth — verdict", "=" * 74, ""]
    lines.append(
        f"E1 (near-sightedness wins): R = S_dense/S_sparse = "
        f"{r_meas['short']:.2f} (λ = {args.lambda_short:g}), "
        f"{r_meas['long']:.2f} (λ = {args.lambda_long:g}) — "
        + ("✓ six times the matter, slower growth: the matter's own "
           "capacity consumption weakens and shortens its gravity."
           if e1 else "✗ the dense band does not grow slower."))
    lines.append(
        f"E2 (the static field predicts the dynamics): measured/predicted "
        f"R = {r_meas['short']/r_pred['short']:.2f}, "
        f"{r_meas['long']/r_pred['long']:.2f} — "
        + ("✓ the relaxed field's numbers (ρ, κ̄, ℓ per band) land on the "
           "N-body's growth: cross-instrument closure." if e2 else
           "✗ the static assembly misses the measured growth."))
    lines.append(
        f"E3 (environment × scale): R(short)/R(long) = {dbl_meas:.2f} vs "
        f"screen-only prediction {dbl_pred:.2f} — "
        + ("✓ the range's locality isolated: ρ and κ̄ cancel in the double "
           "ratio, and what remains is screening." if e3 else
           "✗ no scale-dependence of the environmental suppression."))
    lines.append("")
    lines.append(f"score: {int(e1)+int(e2)+int(e3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: static-background growth (h₀ ≈ 0) as in the parents' "
        "calibration runs, one band geometry, one mass contrast, one "
        "recovery rate; bands are measured in their central halves (≥ 3.7 "
        "ℓ_sparse from interfaces) but interface leakage is not zero; the "
        "S ∝ ρκ̄²·screen assembly is first-order (mean-field per band, "
        "linear response); per-band S carries fit noise not propagated "
        "into error bars — the factor-2 bars absorb both.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_environment_growth.json"),
              "w") as fh:
        json.dump({"params": vars(args), "static": stat, "runs": runs,
                   "analysis": {"e1": bool(e1), "e2": bool(e2),
                                "e3": bool(e3), "r_meas": r_meas,
                                "r_pred": r_pred, "dbl_meas": dbl_meas,
                                "dbl_pred": dbl_pred}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
