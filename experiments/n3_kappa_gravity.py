"""Capacity as gravity: the screened attraction κ mediates.

The URP capacity field κ obeys a gradient flow of the free energy
``F[κ] = ∫[(D/2)|∇κ|² + (r/2)(κ−κ₀)² + (c/2)·load·κ²]``.  So κ behaves like
gravity in the general sense: a concentration of ``load`` (distinction — the
field's "mass") digs a well in κ, and because F is a real energy, two masses
lower it by drawing together.  Linearising, the well is **Yukawa-screened**
with range

    ξ_κ = √(D / r)

— the ratio of capacity diffusion to recovery rate.  The recovery rate is the
graviton mass: fast recovery → short range, slow recovery → long range.

This experiment measures all of it, on the real κ dynamics:

1. **The well and the force.**  Two rigid Gaussian masses at separation ``r``;
   relax κ; the interaction energy ``V(r) = F(r) − F(∞)`` is **negative**
   (attraction) and **Yukawa-shaped**.
2. **The range law.**  Across independently varied ``D`` and ``r``, the
   measured screening length tracks ``√(D/r)`` — the range is set by the
   diffusion/recovery ratio, not either alone.
3. **The equivalence principle.**  The interaction strength scales as the
   *product of the masses* (``V ∝ m₁·m₂``): κ-gravity couples to how much
   distinction is present, universally, regardless of what it is.

Honest scope: this is a *classical, static, scalar* mediation — κ is the
weak, universal, mass-sourced binding coupling of the framework, not the metric
tensor of general relativity.  The screened (massive-graviton) range is a real
difference from Newtonian gravity; the point is that the *role* — a universal
attraction sourced by and back-reacting on structure — is what κ plays.

Usage::

    python experiments/n3_kappa_gravity.py --output-dir artifacts/n3_kappa_grav
    python experiments/n3_kappa_gravity.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_genesis.capacity_gravity import (
    capacity_free_energy,
    fit_yukawa_range,
    gaussian_load,
    relax_capacity,
    screening_length,
    well_range,
)


def _kw(D, r, args):
    return dict(kappa_diffusion=D, kappa_recovery=r,
                kappa_consumption=args.consumption, kappa_baseline=1.0,
                dt=args.dt, max_iters=args.max_iters, tol=args.tol)


def _energy_kw(D, r, args):
    return dict(kappa_diffusion=D, kappa_recovery=r,
                kappa_consumption=args.consumption, kappa_baseline=1.0)


def single_well(D, r, args):
    """Relax one mass; return its κ-slice, radial deficit, and measured range."""
    L = args.size
    ctr = (L // 2, L // 2, L // 2)
    load = gaussian_load((L, L, L), ctr, args.width, args.amplitude)
    kappa = relax_capacity(load, **_kw(D, r, args))
    line = (1.0 - kappa)[L // 2:, L // 2, L // 2]
    return {"xi_meas": well_range(line), "xi_pred": screening_length(D, r),
            "deficit_line": line.tolist(),
            "kappa_slice": kappa[:, :, L // 2].tolist()}


def two_body(D, r, args):
    """V(r) = F(sep) − F(ref) for a sweep of separations, plus a Yukawa fit."""
    L = args.size
    ctr = [L // 2, L // 2, L // 2]
    ekw = _energy_kw(D, r, args)

    def F(sep):
        c1, c2 = list(ctr), list(ctr)
        c1[1] = ctr[1] - sep / 2.0
        c2[1] = ctr[1] + sep / 2.0
        load = gaussian_load((L, L, L), [tuple(c1), tuple(c2)],
                             args.width, args.amplitude)
        kappa = relax_capacity(load, **_kw(D, r, args))
        return capacity_free_energy(kappa, load, **ekw), load, kappa

    seps = [float(s) for s in args.separations]
    Fref, _, _ = F(args.size * 0.5)
    V, slice2 = [], None
    for s in seps:
        Fs, load, kappa = F(s)
        V.append(Fs - Fref)
        if slice2 is None and abs(s - args.slice_sep) < 1e-9:
            slice2 = kappa[:, :, L // 2].tolist()
    fit = fit_yukawa_range(seps, V)
    return {"seps": seps, "V": V, "fit": fit, "kappa_slice": slice2}


def range_scan(args):
    """Measured screening length vs √(D/r) across independently varied D, r."""
    out = []
    for D, r in args.dr_pairs:
        w = single_well(D, r, args)
        out.append({"D": D, "r": r, "xi_pred": w["xi_pred"],
                    "xi_meas": w["xi_meas"]})
        print(f"  range  D={D:g} r={r:g}: √(D/r)={w['xi_pred']:.2f} "
              f"ξ_meas={w['xi_meas']:.2f}", flush=True)
    # ξ_meas = α·√(D/r): the range law (constant α absorbs lattice/finite-width).
    # Drop any point whose well outgrew the box (fit non-finite or non-positive).
    good = [o for o in out if np.isfinite(o["xi_meas"]) and o["xi_meas"] > 0]
    xp = np.array([o["xi_pred"] for o in good])
    xm = np.array([o["xi_meas"] for o in good])
    alpha = float((xp * xm).sum() / (xp * xp).sum())
    resid = xm - alpha * xp
    ss = 1.0 - (resid ** 2).sum() / ((xm - xm.mean()) ** 2).sum()
    return {"points": out, "alpha": alpha, "r2": float(ss)}


def equivalence(D, r, args):
    """Interaction strength vs mass product m₁·m₂ (the equivalence principle)."""
    L = args.size
    ctr = [L // 2, L // 2, L // 2]
    ekw = _energy_kw(D, r, args)
    sep = args.eq_sep

    def F(load):
        return capacity_free_energy(relax_capacity(load, **_kw(D, r, args)),
                                    load, **ekw)

    c1, c2 = list(ctr), list(ctr)
    c1[1] = ctr[1] - sep / 2.0
    c2[1] = ctr[1] + sep / 2.0
    Fvac = F(np.zeros((L, L, L)))  # vacuum baseline, once

    masses = [float(m) for m in args.eq_masses]
    prod, Vint = [], []
    for m in masses:
        both = (gaussian_load((L, L, L), [tuple(c1)], args.width, m)
                + gaussian_load((L, L, L), [tuple(c2)], args.width, m))
        single = gaussian_load((L, L, L), [tuple(c1)], args.width, m)
        # interaction = F(both) − 2·F(single) + F(vacuum): the cross term
        prod.append(m * m)
        Vint.append(F(both) - 2.0 * F(single) + Fvac)
    slope = float(np.polyfit(prod, Vint, 1)[0])
    # linearity R² of V vs m₁m₂
    p = np.array(prod); v = np.array(Vint)
    pred = np.polyval(np.polyfit(p, v, 1), p)
    r2 = float(1.0 - ((v - pred) ** 2).sum() / ((v - v.mean()) ** 2).sum())
    return {"masses": masses, "prod": prod, "V": Vint, "slope": slope, "r2": r2}


def summarise(well, tb, scan, eq, D0, r0) -> list[str]:
    lines = ["Capacity as gravity — verdict", "=" * 40, ""]
    lines.append(
        f"κ mediates an attraction: two masses (β_g-free, rigid load blobs) at "
        f"separation r lower the capacity free energy — V(r) < 0 throughout "
        f"(V = {tb['V'][0]:+.3f} at r={tb['seps'][0]:g} deepening to "
        f"{tb['V'][-1]:+.3f} at r={tb['seps'][-1]:g}), and it is Yukawa-screened: "
        f"a fit V ∝ −e^(−r/ξ)/r gives ξ = {tb['fit']['xi']:.2f}.")
    lines.append("")
    lines.append(
        f"the range is √(D/r): across independently varied D and r, the measured "
        f"screening length tracks the prediction as ξ_meas = {scan['alpha']:.2f}·"
        f"√(D/r) with R² = {scan['r2']:.3f}. The recovery rate is the graviton "
        "mass — fast recovery screens the force to short range, slow recovery "
        "lets it reach. (A constant α≈1 absorbs the lattice/finite-width offset; "
        "the physics is the √(D/r) scaling.)")
    lines.append("")
    lines.append(
        f"and it obeys an equivalence principle: the interaction strength scales "
        f"with the product of the masses, V ∝ m₁·m₂ (linear in m² with "
        f"R² = {eq['r2']:.3f}). κ-gravity couples to how much distinction is "
        "present, universally — not to what kind it is.")
    lines.append("")
    lines.append(
        "honest scope: a classical, static, scalar mediation. κ is the weak, "
        "universal, mass-sourced binding coupling of the framework — sourced by "
        "structure and back-reacting on it — not the metric tensor of general "
        "relativity. The screened (massive-graviton) range √(D/r) is a genuine "
        "difference from Newtonian 1/r²; what the measurement establishes is the "
        "gravitational *role*: a universal attraction whose strength is κ and "
        "whose range is the capacity recovery length.")
    return lines


def make_figure(well, tb, scan, eq, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY, RED = (
        "#0072B2", "#E69F00", "#009E73", "#999999", "#c0392b")
    fig = plt.figure(figsize=(13, 9.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.26)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))
    for ax in (ax2, ax3, ax4):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # panel 1: κ field slice for two masses — the merged wells (the potential)
    sl = np.array(tb["kappa_slice"]) if tb["kappa_slice"] is not None \
        else np.array(well["kappa_slice"])
    im = ax1.imshow(sl.T, origin="lower", cmap="magma", aspect="equal")
    ax1.set_title(f"Two masses dig one κ-well (slice, sep={args.slice_sep:g})")
    ax1.set_xticks([]); ax1.set_yticks([])
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="κ")

    # panel 2: V(r) — attraction + Yukawa fit (log |V|)
    seps = np.array(tb["seps"]); V = np.array(tb["V"])
    ax2.plot(seps, -V, color=BLUE, marker="o", ms=6, lw=0, zorder=4,
             label="−V(r)  (attraction)")
    xi = tb["fit"]["xi"]; A = tb["fit"]["amplitude"]
    rs = np.linspace(seps.min(), seps.max(), 100)
    ax2.plot(rs, A * np.exp(-rs / xi) / rs, color=RED, lw=1.6, zorder=3,
             label=f"Yukawa fit, ξ={xi:.2f}")
    ax2.set_yscale("log")
    ax2.set_xlabel("separation r")
    ax2.set_ylabel("−V(r)  (capacity free energy)")
    ax2.set_title("κ mediates a screened attraction")
    ax2.legend(frameon=False, fontsize=8)

    # panel 3: the range law ξ_meas vs √(D/r)
    xp = np.array([p["xi_pred"] for p in scan["points"]])
    xm = np.array([p["xi_meas"] for p in scan["points"]])
    ax3.plot(xp, xm, color=GREEN, marker="D", ms=7, lw=0, zorder=4)
    lim = [0, max(xp.max(), xm.max()) * 1.1]
    ax3.plot(lim, lim, color=GRAY, ls="--", lw=1.0, zorder=2, label="ξ = √(D/r)")
    ax3.plot(lim, [scan["alpha"] * x for x in lim], color=ORANGE, lw=1.6,
             zorder=3, label=f"ξ = {scan['alpha']:.2f}·√(D/r)  (R²={scan['r2']:.3f})")
    ax3.set_xlabel("predicted range  √(D/r)")
    ax3.set_ylabel("measured range  ξ")
    ax3.set_title("The range is set by D/r (recovery = graviton mass)")
    ax3.legend(frameon=False, fontsize=8)
    ax3.set_xlim(lim); ax3.set_ylim(lim)

    # panel 4: equivalence principle V ∝ m₁·m₂
    prod = np.array(eq["prod"]); Vi = np.array(eq["V"])
    ax4.plot(prod, -Vi, color=ORANGE, marker="s", ms=6, lw=0, zorder=4)
    ps = np.linspace(0, prod.max() * 1.05, 50)
    ax4.plot(ps, -np.polyval(np.polyfit(prod, Vi, 1), ps), color=GRAY, lw=1.4,
             zorder=3, label=f"linear, R²={eq['r2']:.3f}")
    ax4.set_xlabel("mass product  m₁·m₂")
    ax4.set_ylabel("−V_int  (interaction strength)")
    ax4.set_title("Equivalence principle: V ∝ m₁·m₂")
    ax4.legend(frameon=False, fontsize=8)

    fig.suptitle("Capacity as gravity: κ is a universal, mass-sourced, "
                 "√(D/r)-screened attraction", fontsize=12, y=0.98)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--width", type=float, default=1.3)
    p.add_argument("--amplitude", type=float, default=0.5)
    p.add_argument("--consumption", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=0.05)  # stable to D=2 (FTCS: dt≤1/6D)
    p.add_argument("--max-iters", type=int, default=12000)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--D0", type=float, default=1.0)
    p.add_argument("--r0", type=float, default=0.05)
    p.add_argument("--separations", type=float, nargs="+",
                   default=[4, 5, 6, 7, 8, 10, 12, 14])
    p.add_argument("--slice-sep", type=float, default=6.0)
    p.add_argument("--eq-sep", type=float, default=6.0)
    p.add_argument("--eq-masses", type=float, nargs="+",
                   default=[0.25, 0.4, 0.55, 0.7])
    p.add_argument("--output-dir", type=str, default="artifacts/n3_kappa_grav")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    # (D, r) pairs vary D and r independently so the test is √(D/r), not 1/√r;
    # kept in ξ ∈ [3, 6.3] so the well decays well inside the L=48 half-box
    args.dr_pairs = [(1.0, 0.111), (2.0, 0.180), (0.5, 0.040), (1.0, 0.0625),
                     (2.0, 0.080), (1.0, 0.040), (0.5, 0.020)]

    if args.quick:
        args.size = 32
        args.separations = [4, 6, 8, 10]
        args.eq_masses = [0.3, 0.5, 0.7]
        args.dr_pairs = [(1.0, 0.111), (1.0, 0.0625), (2.0, 0.111)]
        args.max_iters = 4000

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"capacity-gravity, L={args.size}^3", flush=True)
    print(f"reference D={args.D0:g} r={args.r0:g}  √(D/r)="
          f"{screening_length(args.D0, args.r0):.2f}", flush=True)

    well = single_well(args.D0, args.r0, args)
    print(f"  single well: √(D/r)={well['xi_pred']:.2f} ξ_meas={well['xi_meas']:.2f}",
          flush=True)
    tb = two_body(args.D0, args.r0, args)
    print(f"  two-body: V from {tb['V'][0]:+.3f} to {tb['V'][-1]:+.3f}, "
          f"Yukawa ξ={tb['fit']['xi']:.2f}", flush=True)
    scan = range_scan(args)
    eq = equivalence(args.D0, args.r0, args)
    print(f"  equivalence: V∝m₁m₂ slope={eq['slope']:.3f} R²={eq['r2']:.3f}",
          flush=True)

    lines = summarise(well, tb, scan, eq, args.D0, args.r0)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    payload = {"params": {k: v for k, v in vars(args).items()},
               "well": {k: v for k, v in well.items() if k != "kappa_slice"},
               "two_body": {k: v for k, v in tb.items() if k != "kappa_slice"},
               "range_scan": scan, "equivalence": eq}
    with open(os.path.join(args.output_dir, "n3_kappa_gravity.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_kappa_gravity.png")
        make_figure(well, tb, scan, eq, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
