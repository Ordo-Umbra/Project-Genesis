"""Local screening: is gravity's range set by the environment, or by the mean?

The screening knee measured the loaded (Debye) law ``ℓ = √(D_κ/(r + c·⟨ρ⟩))``
— matter consumes capacity, and consumed capacity screens gravity — but as a
**mean-field, global** statement: one uniform matter bath, one effective
range.  The law's own linearisation says something stronger: the screened
mode's mass is set by the capacity environment **where the disturbance
lives**, so gravity should be shorter-ranged inside matter and longer-ranged
in voids *within the same universe* — the phenomenology class of chameleon
screening, as a falsifiable statement of the analogue.

Instrument: no dynamics at all.  A box whose left half carries a uniform
matter bath (load ρ) and whose right half is void, relaxed once; a **light
test probe** (Gaussian load, amplitude ≪ bath) is added at the centre of one
region at a time, and the probe's induced capacity deficit
``δκ = κ_background − κ_probe`` is the mediator's propagator, read directly.
Its radial decay is fitted with the 2-D screened form (``δκ ∝ K₀(d/ξ)``, so
``ln(δκ·√d)`` is linear in ``d`` on the shoulder), giving the **local range
ξ** of κ-gravity in that environment.

Pre-registered predictions:

L1. **One field, two gravities**: in a single relaxed configuration
    (ρ_bath = 0.1, void opposite), the probe's range is shorter in the bath,
    with ``ξ_bath`` within a factor of 2 of ``√(D_κ/(r + c·ρ_bath))``.
L2. **Locality beats mean-field**: the void probe's range matches its
    *local* vacuum law ``√(D_κ/r)`` strictly better than the *global*
    mean-load prediction ``√(D_κ/(r + c·⟨ρ⟩_box))`` (relative error smaller,
    and within a factor of 2 of the local value) — screening is set by where
    you are, not by the box average.
L3. **The chameleon curve**: sweeping the bath density, the regression of
    ``1/ξ_bath²`` on ρ recovers both constants of the loaded law — slope
    ``c/D_κ`` and intercept ``r/D_κ``, each within a factor of 2 — the knee
    experiment's K3 transposed from field dials to *environments* at fixed
    dials.

Usage::

    python experiments/n3_local_screening.py --output-dir artifacts/n3_local
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import (
    gaussian_load,
    relax_capacity,
    screening_length,
)

# c is relax_capacity's kappa_consumption default — the constant the
# relaxation actually uses (see the knee experiment for why it is not a flag).
CONSUMPTION = 2.0


def half_bath(shape, rho: float) -> np.ndarray:
    """Uniform load ρ on the left half of the box, void on the right."""
    load = np.zeros(shape, dtype=float)
    load[: shape[0] // 2, :] = rho
    return load


def probe_response(background: np.ndarray, center, args) -> np.ndarray:
    """δκ: the relaxed field's response to a light test probe at ``center``."""
    kw = dict(kappa_recovery=args.recovery, max_iters=20000, tol=1e-10)
    kappa_bg = relax_capacity(background, **kw)
    probe = gaussian_load(background.shape, [tuple(center)],
                          args.probe_width, args.probe_mass)
    kappa = relax_capacity(background + probe, **kw)
    return kappa_bg - kappa


def fit_range(delta: np.ndarray, center, args) -> dict:
    """Local range ξ from the radial decay ``ln(δκ·√d) = a − d/ξ``."""
    shape = delta.shape
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    d2 = sum((((g - c + s / 2) % s) - s / 2) ** 2
             for g, c, s in zip(grids, center, shape))
    dist = np.sqrt(d2)
    shells = np.arange(1, args.fit_max + 1)
    prof = np.array([float(delta[(dist >= s - 0.5) & (dist < s + 0.5)].mean())
                     for s in shells])
    sel = (shells >= args.fit_min) & (prof > 1e-12)
    y = np.log(prof[sel] * np.sqrt(shells[sel]))
    slope, intercept = np.polyfit(shells[sel], y, 1)
    resid = y - (slope * shells[sel] + intercept)
    r2 = 1.0 - float(np.sum(resid ** 2)) / float(np.sum((y - y.mean()) ** 2))
    return {"xi": float(-1.0 / slope), "r2": r2, "n": int(sel.sum()),
            "profile": prof.tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--recovery", type=float, default=0.2)
    p.add_argument("--rho-main", type=float, default=0.1)
    p.add_argument("--rho-sweep", type=float, nargs="+",
                   default=[0.025, 0.05, 0.1, 0.15, 0.2])
    p.add_argument("--probe-mass", type=float, default=0.02)
    p.add_argument("--probe-width", type=float, default=2.0)
    p.add_argument("--fit-min", type=int, default=4)
    p.add_argument("--fit-max", type=int, default=18)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_local")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.size = 96
        args.rho_sweep = [0.05, 0.2]

    os.makedirs(args.output_dir, exist_ok=True)
    print("local screening: one field, measured in two environments",
          flush=True)
    shape = (args.size, args.size)
    r, c = args.recovery, CONSUMPTION

    # --- L1/L2: the half-bath configuration, probed in each region
    bath = half_bath(shape, args.rho_main)
    bath_center = (args.size // 4, args.size // 2)
    void_center = (3 * args.size // 4, args.size // 2)
    fit_bath = fit_range(probe_response(bath, bath_center, args),
                         bath_center, args)
    fit_void = fit_range(probe_response(bath, void_center, args),
                         void_center, args)
    xi_bath_pred = float(np.sqrt(1.0 / (r + c * args.rho_main)))
    xi_void_pred = screening_length(1.0, r)
    xi_global_pred = float(np.sqrt(1.0 / (r + c * float(bath.mean()))))
    print(f"  bath: ξ={fit_bath['xi']:.2f} (local law {xi_bath_pred:.2f}), "
          f"R²={fit_bath['r2']:.3f}", flush=True)
    print(f"  void: ξ={fit_void['xi']:.2f} (local law {xi_void_pred:.2f}, "
          f"global-mean law {xi_global_pred:.2f}), R²={fit_void['r2']:.3f}",
          flush=True)

    # --- L3: the chameleon curve — range vs environment density
    sweep = []
    for rho in args.rho_sweep:
        fit = fit_range(probe_response(half_bath(shape, rho),
                                       bath_center, args),
                        bath_center, args)
        sweep.append({"rho": rho, "xi": fit["xi"], "r2": fit["r2"],
                      "kappa_bar": r / (r + c * rho)})
        print(f"  ρ={rho:<6}: ξ={fit['xi']:.2f} "
              f"(law {np.sqrt(1.0/(r + c*rho)):.2f}), R²={fit['r2']:.3f}",
              flush=True)
    rhos = np.array([s["rho"] for s in sweep])
    inv_xi2 = np.array([s["xi"] ** -2 for s in sweep])
    slope, intercept = np.polyfit(rhos, inv_xi2, 1)

    q_bath = fit_bath["xi"] / xi_bath_pred
    l1 = (fit_bath["xi"] < fit_void["xi"]
          and np.isfinite(q_bath) and 0.5 <= q_bath <= 2.0)
    err_local = abs(fit_void["xi"] - xi_void_pred) / xi_void_pred
    err_global = abs(fit_void["xi"] - xi_global_pred) / xi_global_pred
    l2 = (err_local < err_global
          and 0.5 <= fit_void["xi"] / xi_void_pred <= 2.0)
    l3 = (0.5 <= slope / c <= 2.0) and (0.5 <= intercept / r <= 2.0)

    lines = ["Local screening — verdict", "=" * 74, ""]
    lines.append(
        f"L1 (one field, two gravities): ξ_bath = {fit_bath['xi']:.2f} vs "
        f"ξ_void = {fit_void['xi']:.2f} in the same relaxed field; "
        f"ξ_bath/√(D/(r+cρ)) = {q_bath:.2f} — "
        + ("✓ gravity is shorter-ranged inside matter, at the loaded law's "
           "own value." if l1 else "✗ no local range difference resolved."))
    lines.append(
        f"L2 (locality beats mean-field): void range error vs local law "
        f"{err_local:.1%}, vs global-mean law {err_global:.1%} — "
        + ("✓ the void keeps its vacuum reach: screening is set by where "
           "you are, not by the box average." if l2 else
           "✗ the global mean describes the void as well as the local law."))
    lines.append(
        f"L3 (the chameleon curve): 1/ξ² vs ρ gives slope {slope:.2f} "
        f"(c = {c:g}) and intercept {intercept:.3f} (r = {r:g}) — "
        + ("✓ the loaded law read from environments at fixed dials — the "
           "knee's regression, transposed." if l3 else
           "✗ the environment sweep does not recover the law's constants."))
    lines.append("")
    lines.append(f"score: {int(l1)+int(l2)+int(l3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: static linear-response measurement (relaxed fields, "
        "light test probe) — the dynamical counterpart (growth measured "
        "per-environment in an N-body run) is the harder follow-up; sharp "
        "bath edges and one recovery rate; the 2-D K₀ shoulder fit is used "
        "everywhere so form bias cancels in comparisons but not in absolute "
        "ξ; probe back-reaction is bounded by the probe/bath load ratio "
        "(0.2 at ρ = 0.1).")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_local_screening.json"),
              "w") as fh:
        json.dump({"params": vars(args),
                   "half_bath": {"bath": fit_bath, "void": fit_void,
                                 "xi_bath_pred": xi_bath_pred,
                                 "xi_void_pred": xi_void_pred,
                                 "xi_global_pred": xi_global_pred},
                   "sweep": sweep,
                   "analysis": {"l1": bool(l1), "l2": bool(l2),
                                "l3": bool(l3), "slope": float(slope),
                                "intercept": float(intercept),
                                "err_local": err_local,
                                "err_global": err_global}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
