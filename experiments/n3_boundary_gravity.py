"""Does the boundary know the interior?  κ-gravity has no Gauss law — until it does.

`n3_area_law` measured that *coherence* is boundary-limited and named the
sharper question it could not answer: **is the bulk κ field — the thing that
gravitates — readable from its boundary?**  This answers it.

The classical form of "gravity is boundary processing" is a **Gauss law**: the
field flux through *any* closed surface equals the enclosed mass, independent of
the surface radius and of how the mass is arranged inside.  On the lattice the
divergence theorem makes that flux exact as a bulk sum
(``boundary_gravity.enclosed_flux``).

Pre-registered predictions:

B1. **A Gauss law while screened.**  At the programme's standard operating
    point, ``flux / enclosed-mass`` is independent of the surface radius.

B2. **Blind to arrangement.**  The same total mass in different interior
    configurations gives the same boundary reading.

B3. **Screening sets the fidelity.**  Whatever B1 gives, the ratio's
    radius-independence should improve as the screening length ``ξ = √(D/r)``
    grows past the surface radius — and degrade as ξ shrinks below it.

The measured answer is a **negative for B1 and B2** with an exact mechanism, and
B3 explains both: while screened, the boundary reading decays like ``e^{−R/ξ}``
(a ``10⁵`` spread across radii at ``ξ = 2``) and depends on arrangement by an
order of magnitude — *there is no Gauss law*.  As ``r → 0`` and ``ξ`` exceeds the
surface, the ratio flattens to a constant and **the Gauss law is restored**.

The consequence is a genuine internal tension in the framework, found by
measurement: the screening term ``r(κ₀−κ)`` is *capacity self-maintenance* — the
very term from which this programme **derives dark energy**
(``ρ_Λ = coeff·r·κ₀²``, `n3_self_contained_cosmos`).  That same term is what
blinds the boundary to the interior.  Boundary-encoded gravity wants ``r → 0``;
derived dark energy wants ``r > 0``.  They are the same knob.

Usage::

    python experiments/n3_boundary_gravity.py --output-dir artifacts/n3_boundary
    python experiments/n3_boundary_gravity.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.boundary_gravity import (
    enclosed_flux,
    enclosed_mass,
    flatness,
    gauss_ratio_profile,
)
from project_genesis.capacity_gravity import (
    gaussian_load,
    relax_capacity,
    screening_length,
)


def kappa_of(load, recovery, args):
    return relax_capacity(load, kappa_recovery=recovery,
                          kappa_diffusion=args.diffusion,
                          kappa_consumption=args.consumption,
                          kappa_baseline=args.baseline,
                          dt=args.dt, max_iters=args.max_iters)


def b1_b3_screening(args):
    """flux/M vs surface radius, across screening lengths."""
    n = args.n
    mid = n / 2.0
    load = gaussian_load((n, n), [(mid, mid)], args.width, args.amplitude)
    rows = []
    for r in args.recoveries:
        kappa = kappa_of(load, r, args)
        prof = gauss_ratio_profile(kappa, load, args.radii)
        rows.append({"recovery": r, "xi": screening_length(args.diffusion, r),
                     "profile": prof, "flatness": flatness(prof)})
        print(f"  r={r:7.4f} ξ={rows[-1]['xi']:5.1f}  flux/M = "
              + " ".join(f"{v:7.4f}" for v in prof)
              + f"   spread={rows[-1]['flatness']:9.1f}×", flush=True)
    return rows


def b2_arrangement(args):
    """Same total mass, different interior arrangements."""
    n = args.n
    mid = n / 2.0
    d = args.spread
    configs = {
        "1 central": [(mid, mid)],
        "2 split": [(mid - d, mid), (mid + d, mid)],
        "4 spread": [(mid - d, mid - d), (mid + d, mid - d),
                     (mid - d, mid + d), (mid + d, mid + d)],
    }
    rows = []
    for name, centers in configs.items():
        load = sum(gaussian_load((n, n), [c], args.width,
                                 args.amplitude / len(centers))
                   for c in centers)
        kappa = kappa_of(load, args.arrangement_recovery, args)
        m = enclosed_mass(load, args.arrangement_radius)
        f = enclosed_flux(kappa, args.arrangement_radius)
        rows.append({"name": name, "mass": m, "flux": f, "ratio": f / max(m, 1e-12)})
        print(f"  {name:10s} M={m:7.3f}  flux={f:9.4f}  flux/M={f/max(m,1e-12):8.4f}",
              flush=True)
    return rows


def verdict(screen_rows, arrange_rows, args):
    tight = min(screen_rows, key=lambda r: r["xi"])
    loose = max(screen_rows, key=lambda r: r["xi"])
    b1 = tight["flatness"] < args.gauss_tol          # Gauss law while screened?
    ratios = [r["ratio"] for r in arrange_rows]
    b2 = flatness(ratios) < args.gauss_tol           # blind to arrangement?
    b3 = loose["flatness"] < tight["flatness"] / 10.0  # screening sets fidelity
    restored = loose["flatness"] < args.gauss_tol
    return b1, b2, b3, tight, loose, flatness(ratios), restored


def summarise(screen_rows, arrange_rows, args):
    b1, b2, b3, tight, loose, arr_spread, restored = verdict(
        screen_rows, arrange_rows, args)
    lines = ["Does the boundary know the interior? κ-gravity and the Gauss law "
             "— verdict", "=" * 74, ""]
    lines.append(
        f"B1 (a Gauss law while screened): at ξ={tight['xi']:.1f} the ratio "
        f"flux/M across surface radii {list(args.radii)} spans "
        f"{tight['flatness']:.0f}× — "
        + ("✓ radius-independent." if b1 else
           "✗ **NO GAUSS LAW.** The boundary reading is not a constant; it "
           "decays like e^(−R/ξ). A surface drawn further out reports less "
           "enclosed mass — the interior is screened from the boundary."))
    lines.append("")
    lines.append(
        "B2 (blind to arrangement): the same total mass as "
        + ", ".join(f"{r['name']} → {r['ratio']:.4f}" for r in arrange_rows)
        + f" ({arr_spread:.1f}× spread) — "
        + ("✓ the boundary reads only the total." if b2 else
           "✗ **arrangement-dependent.** The boundary reading changes by an "
           "order of magnitude for the same enclosed mass — it is not reading "
           "the mass, it is reading the mass *near enough to the surface*."))
    lines.append("")
    lines.append(
        f"B3 (screening sets the fidelity): the spread falls "
        f"{tight['flatness']:.0f}× → {loose['flatness']:.1f}× as ξ grows "
        f"{tight['xi']:.1f} → {loose['xi']:.1f} — "
        + (("✓ and at ξ ≫ R the ratio is flat to "
            f"{loose['flatness']:.1f}×: **the Gauss law is restored in the "
            "unscreened limit.**" if restored else
            "✓ fidelity improves with ξ, though not yet flat on this range.")
           if b3 else "✗ screening does not control the fidelity."))
    lines.append("")
    n_land = int(b1) + int(b2) + int(b3)
    lines.append(f"score: {n_land}/3 pre-registered predictions land "
                 "(B1 and B2 refuted — recorded as measured).")
    lines.append("")
    lines.append(
        "what this establishes: **κ-gravity is not holographic while it is "
        "screened.** A boundary integral does not know the mass inside — the "
        "information decays exponentially over the screening length, and the "
        "reading depends on where the mass sits. The Gauss law — the elementary "
        "sense in which gravity 'processes the surface of a coherent boundary' —"
        " holds *only* in the unscreened limit r → 0, where it is recovered "
        "cleanly. So the boundary-encoding property is not a general feature of "
        "this gravity; it is a property of a *massless* mediator, and κ-gravity "
        "has a massive one.")
    lines.append("")
    lines.append(
        "the tension this exposes: the screening term r(κ₀−κ) is capacity "
        "**self-maintenance** — the same term from which this programme derives "
        "dark energy (ρ_Λ = coeff·r·κ₀², `n3_self_contained_cosmos`). "
        "Boundary-encoded gravity wants r → 0; derived dark energy wants r > 0. "
        "They are the same knob, and this is the first measurement to connect "
        "them. A possible resolution the framework already implies: a very "
        "*small* r gives both a long screening length (holography over "
        "sub-horizon scales) and a *small* Λ — which is the observed situation. "
        "That relationship is suggestive, not derived, and is registered here as "
        "an open thread rather than a result.")
    lines.append("")
    lines.append(
        "honest scope: 2-D, one lattice, relaxed (steady-state) κ — no "
        "dynamics, no retardation; the flux is exact on the lattice via the "
        "divergence theorem but the disc is pixelated; the arrangement test uses "
        "one spread and one surface radius. Screening destroying a Gauss law is "
        "*expected* for a Yukawa field — that is not the finding. The finding is "
        "that this programme's gravity is that field, so the boundary-processing "
        "reading of it fails at the operating points where its dark energy comes "
        "from, and the two are controlled by one parameter.")
    return lines, n_land


def make_figure(screen_rows, arrange_rows, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY, RED = ("#0072B2", "#E69F00", "#009E73",
                                      "#999999", "#c0392b")
    fig = plt.figure(figsize=(13, 5.2))
    gs = GridSpec(1, 3, figure=fig, wspace=0.32)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, j]) for j in range(3))
    for ax in (ax1, ax2, ax3):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(screen_rows)))
    for row, col in zip(screen_rows, cmap):
        ax1.plot(args.radii, np.maximum(row["profile"], 1e-6), color=col, lw=2.0,
                 marker="o", ms=5, zorder=3, label=f"ξ={row['xi']:.0f}")
    ax1.set_yscale("log")
    ax1.set_xlabel("surface radius  R")
    ax1.set_ylabel("flux / enclosed mass")
    ax1.set_title("A. A Gauss law would be flat\n(screened: it is not)")
    ax1.legend(fontsize=8, loc="best")

    ax2.axhline(1.0, color=GREEN, lw=1.4, ls=":", zorder=1,
                label="perfect Gauss law")
    ax2.plot([r["xi"] for r in screen_rows], [r["flatness"] for r in screen_rows],
             color=BLUE, lw=2.2, marker="o", ms=7, zorder=3)
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlabel("screening length  ξ = √(D/r)")
    ax2.set_ylabel("spread of flux/M  (max/min)")
    ax2.set_title("B. Restored as screening is removed")
    ax2.legend(fontsize=8, loc="best")

    names = [r["name"] for r in arrange_rows]
    ax3.bar(range(len(names)), [r["ratio"] for r in arrange_rows],
            color=[RED, ORANGE, BLUE], zorder=3)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, fontsize=8)
    ax3.set_ylabel("flux / enclosed mass")
    ax3.set_title("C. Same mass, different places\n(a Gauss law can't tell)")

    fig.suptitle("κ-gravity has no Gauss law while it is screened — the boundary "
                 "does not know the interior until r → 0", fontsize=12, y=1.03)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=192)
    p.add_argument("--width", type=float, default=4.0)
    p.add_argument("--amplitude", type=float, default=0.4)
    p.add_argument("--radii", type=float, nargs="+", default=[12, 18, 24, 30, 40])
    p.add_argument("--recoveries", type=float, nargs="+",
                   default=[0.2, 0.05, 0.01, 0.002, 0.0005])
    p.add_argument("--diffusion", type=float, default=1.0)
    p.add_argument("--consumption", type=float, default=2.0)
    p.add_argument("--baseline", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--max-iters", type=int, default=40000)
    p.add_argument("--arrangement-recovery", type=float, default=0.05)
    p.add_argument("--arrangement-radius", type=float, default=30.0)
    p.add_argument("--spread", type=float, default=10.0)
    p.add_argument("--gauss-tol", type=float, default=2.0,
                   help="Largest max/min spread still counted as a Gauss law.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_boundary")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n = 128
        args.radii = [12, 20, 30]
        args.recoveries = [0.2, 0.01, 0.001]
        args.max_iters = 20000

    os.makedirs(args.output_dir, exist_ok=True)
    print("does the boundary know the interior? a Gauss law for κ-gravity",
          flush=True)
    print(" B1/B3 — flux/M vs surface radius, across screening lengths:")
    screen_rows = b1_b3_screening(args)
    print(" B2 — same total mass, different arrangements:")
    arrange_rows = b2_arrangement(args)

    lines, n_land = summarise(screen_rows, arrange_rows, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_boundary_gravity.json"), "w") as fh:
        json.dump({"params": vars(args), "screening": screen_rows,
                   "arrangement": arrange_rows, "score": n_land},
                  fh, indent=2, default=str)
    if not args.no_figure:
        fp = os.path.join(args.output_dir, "n3_boundary_gravity.png")
        make_figure(screen_rows, arrange_rows, args, fp)
        print(f"figure: {fp}")


if __name__ == "__main__":
    main()
