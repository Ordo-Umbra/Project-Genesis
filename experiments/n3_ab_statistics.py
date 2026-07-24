"""The Aharonov–Bohm statistical phase: statistics as a gauge holonomy.

The self-consistent gauge field (`n3_gauge_field.py`) made the vortex a genuine
gauged particle carrying a quantised flux ``Φ = 2π·q``.  This experiment takes
the deep step it sets up: showing that the **exchange sign** of the fermion —
measured three earlier ways as a *topological* invariant (the braid,
`n3_exchange_statistics.py`) and a *geometric* one (the ``2π`` self-rotation) —
is *also* a **dynamical gauge phase**, the **Aharonov–Bohm** phase of the flux
the gauge field carries.  This is Wilczek's **statistical transmutation**: a
composite of charge and flux exchanges with a phase set by the flux it drags,
and one flux quantum on a unit charge turns a boson into a **fermion**.

The measurement uses the Wilson loop of the *self-consistent* gauge field
(`gauged_vortex.wilson_loop`): parallel-transporting a test charge around the
vortex accumulates ``∮A·dl = Φ`` (lattice Stokes).

Pre-registered predictions:

AB1. **The Aharonov–Bohm phase is the enclosed flux.**  The Wilson-loop
     holonomy of the relaxed gauge field about a winding-``q`` vortex is
     ``Φ ≈ 2π·q``; a **unit** test charge encircling it gets ``e^{iΦ} = +1``
     (Dirac — an integer flux quantum is invisible to an integer charge), while
     a **half** charge gets ``e^{iΦ/2} = (−1)^q`` — the flux quantum is visible
     to a fractional charge as a sign.

AB2. **Statistical transmutation — the flux–charge composite is a fermion.**
     Exchanging two identical composites (charge ``1``, flux ``Φ``) is *half a
     braid*, so each charge sees half the partner's flux: the statistical phase
     is ``θ = Φ/2 = π·q`` → exchange sign ``(−1)^q``.  One flux quantum
     (``q = 1``) makes a **fermion** (``−1``); two (``q = 2``) a **boson**
     (``+1``).  The gauge field turns the topological winding into a dynamical
     statistic.

AB3. **The three faces of ``(−1)^{2s}`` agree.**  The Aharonov–Bohm gauge
     phase ``(−1)^q`` equals the *topological* braid exchange sign
     (`spin_statistics.exchange_holonomy`) and the *geometric* ``2π``
     self-rotation holonomy (`self_rotation_sign`), for every ``q`` — spin,
     statistics, and gauge, one sign.

If AB1–AB3 land, the exchange sign is realised as a genuine dynamical gauge
(Aharonov–Bohm) phase, not only a topological invariant — the field-theoretic
mechanism of fermionic statistics, on the programme's own self-consistent gauge
field.  Honest boundary: this is the **classical** abelian statistical phase
(the Wilczek flux–charge composite at the classical level); it is *still* not
Fock-space anticommutation — the operator ``{ψ, ψ†}`` and a many-body Pauli
principle between identical quanta remain the frontier.

Usage::

    python experiments/n3_ab_statistics.py --output-dir artifacts/n3_ab
    python experiments/n3_ab_statistics.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.gauged_vortex import (
    local_flux,
    relax,
    seed_vortices,
    zero_links,
)
from project_genesis.spin_statistics import exchange_holonomy, self_rotation_sign

TWO_PI = 2.0 * np.pi


def measure(args):
    n = args.n
    c0 = (n / 2 - args.sep / 2, n / 2)
    c1 = (n / 2 + args.sep / 2, n / 2)
    rows = []
    for q in args.charges:
        psi = seed_vortices(n, [c0, c1], [q, -q], core=args.core)
        out = relax(psi, zero_links(n), lam=args.lam, beta=args.beta,
                    dt=args.dt, steps=args.steps)
        # the AB holonomy ∮A·dl = the enclosed flux (a circular Wilson loop)
        holonomy = local_flux(out["theta"], c0, args.flux_radius)
        # AB1: unit vs half charge
        ab_unit = float(np.cos(holonomy))            # e^{iΦ}, unit charge
        ab_half = float(np.cos(0.5 * holonomy))      # e^{iΦ/2}, half charge
        # AB2: the exchange statistical phase of the flux–charge composite
        theta_stat = 0.5 * holonomy                  # Φ/2 (exchange = half braid)
        stat_sign = float(np.cos(theta_stat))
        # AB3: the topological braid and the spin self-rotation, for this q
        braid = exchange_holonomy((args.braid_n, args.braid_n),
                                  (args.braid_n / 2, args.braid_n / 2),
                                  args.braid_sep, (q, q), turns=0.5,
                                  n_steps=args.braid_steps)["sign"]
        rot = self_rotation_sign((args.braid_n, args.braid_n),
                                 (args.braid_n / 2, args.braid_n / 2), q)
        rows.append({"q": q, "holonomy": holonomy, "quanta": holonomy / TWO_PI,
                     "ab_unit": ab_unit, "ab_half": ab_half,
                     "stat_sign": stat_sign, "braid_sign": braid,
                     "rot_sign": rot, "expected": float((-1) ** q)})
        print(f"  q={q}: Φ/2π={holonomy/TWO_PI:+.2f}  AB(unit)={ab_unit:+.2f}  "
              f"AB(½)/stat={stat_sign:+.2f}  braid={braid:+.2f}  rot={rot:+.2f}  "
              f"(expect {(-1)**q:+d})", flush=True)
    return rows


def verdict(rows):
    # AB1: holonomy quantised (≈q), unit charge → +1, half charge → (−1)^q
    v1 = all(abs(abs(r["quanta"]) - r["q"]) < 0.15 and r["ab_unit"] > 0.8
             and abs(r["ab_half"] - r["expected"]) < 0.2 for r in rows)
    # AB2: statistical sign = (−1)^q (fermion odd, boson even)
    v2 = all(abs(r["stat_sign"] - r["expected"]) < 0.2 for r in rows)
    # AB3: the three faces agree
    v3 = all(abs(r["stat_sign"] - r["braid_sign"]) < 0.25
             and abs(r["braid_sign"] - r["rot_sign"]) < 0.25 for r in rows)
    return v1, v2, v3


def summarise(rows, args):
    v1, v2, v3 = verdict(rows)
    n = int(v1) + int(v2) + int(v3)
    lines = ["The Aharonov–Bohm statistical phase: statistics as a gauge "
             "holonomy — verdict", "=" * 74, ""]
    lines.append(
        "AB1 (the AB phase is the enclosed flux): the Wilson-loop holonomy of "
        "the self-consistent gauge field is " + ", ".join(
            f"q={r['q']}: Φ={r['quanta']:+.2f}·2π" for r in rows)
        + "; a unit charge encircling gets "
        + ", ".join(f"{r['ab_unit']:+.2f}" for r in rows)
        + " (Dirac: +1), a half charge "
        + ", ".join(f"{r['ab_half']:+.2f}" for r in rows) + " — "
        + ("✓ the flux quantum is invisible to a unit charge (+1) but a sign to "
           "a half charge ((−1)^q): a real Aharonov–Bohm phase carried by the "
           "gauge dynamics." if v1 else "✗ the AB phases did not come out clean."))
    lines.append("")
    lines.append(
        "AB2 (statistical transmutation): the flux–charge composite (charge 1, "
        "flux Φ=2πq) exchanges with phase θ=Φ/2=πq → sign "
        + ", ".join(f"q={r['q']}:{r['stat_sign']:+.2f}" for r in rows) + " — "
        + ("✓ one flux quantum makes a FERMION (q=1 → −1), two a boson "
           "(q=2 → +1): the gauge field turns the topological winding into a "
           "dynamical exchange statistic — Wilczek flux attachment."
           if v2 else "✗ the statistical signs are not (−1)^q."))
    lines.append("")
    lines.append(
        "AB3 (the three faces of (−1)^2s agree): per q, the Aharonov–Bohm gauge "
        "phase, the topological braid exchange sign, and the 2π self-rotation "
        "holonomy read "
        + "; ".join(f"q={r['q']}: AB {r['stat_sign']:+.1f} / braid "
                    f"{r['braid_sign']:+.1f} / spin {r['rot_sign']:+.1f}"
                    for r in rows) + " — "
        + ("✓ spin, statistics (braid), and gauge (Aharonov–Bohm) give one and "
           "the same sign: the exchange sign is now a dynamical gauge phase, not "
           "only a topological invariant." if v3 else
           "✗ the three determinations do not coincide."))
    lines.append("")
    lines.append(f"score: {n}/3 pre-registered predictions land.")
    lines.append("")
    lines.append(
        "what this establishes: the fermion's exchange sign — the ``−1`` — is "
        "realised on the programme's own self-consistent gauge field as an "
        "**Aharonov–Bohm phase**: the flux the gauge field quantises (`Φ=2πq`) "
        "gives the flux–charge composite an exchange statistic ``(−1)^q``, the "
        "field-theoretic mechanism (Wilczek statistical transmutation) by which "
        "a gauge field turns a topological winding into fermionic statistics. "
        "Spin (the 2π rotation), statistics (the braid), and the gauge holonomy "
        "(Aharonov–Bohm) are three measured faces of the one ``(−1)^{2s}``.")
    lines.append("")
    lines.append(
        "honest scope: this is the **classical** abelian statistical phase — the "
        "Wilczek flux–charge composite at the level of a classical gauge field "
        "and its Aharonov–Bohm holonomy. It is **not** Fock-space "
        "anticommutation: no ``{ψ, ψ†}`` operator algebra, no many-body Pauli "
        "principle between identical *quanta*, no Chern–Simons action solved "
        "dynamically (the flux–charge binding is read off, not enforced by a CS "
        "term). Second quantisation remains the frontier. 2-D, one lattice, one "
        "(λ, β) operating point; the holonomy is the Wilson loop on the relaxed "
        "links, the statistical phase its half.")
    return lines, n


def make_figure(rows, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY, RED, PURPLE = ("#0072B2", "#E69F00", "#009E73",
                                              "#999999", "#c0392b", "#8551A6")
    qs = [r["q"] for r in rows]
    fig = plt.figure(figsize=(13, 9.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.26)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))
    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # A: holonomy = 2πq, and the AB phases
    ax1.plot(qs, [abs(r["quanta"]) for r in rows], color=BLUE, lw=2.0,
             marker="o", ms=7, zorder=3, label="Wilson holonomy Φ/2π")
    ax1.plot(qs, qs, color=GRAY, lw=1.2, ls=":", zorder=1, label="q (ideal)")
    ax1.set_xlabel("vortex winding  q")
    ax1.set_ylabel("Φ / 2π  (flux quanta)")
    ax1.set_xticks(qs)
    ax1.set_title("A. The Wilson-loop holonomy = the flux Φ = 2π·q")
    ax1.legend(fontsize=8, loc="upper left")

    # B: AB phases: unit charge (Dirac +1) vs half charge / statistical ((−1)^q)
    ax2.axhline(0, color=GRAY, lw=1.0, zorder=1)
    w = 0.35
    ax2.bar([q - w / 2 for q in qs], [r["ab_unit"] for r in rows], width=w,
            color=GREEN, zorder=3, label="unit charge (Dirac)")
    ax2.bar([q + w / 2 for q in qs], [r["stat_sign"] for r in rows], width=w,
            color=BLUE, zorder=3, label="½ charge / exchange")
    ax2.set_xticks(qs)
    ax2.set_ylim(-1.35, 1.35)
    ax2.set_xlabel("vortex winding  q")
    ax2.set_ylabel("Aharonov–Bohm phase (real part)")
    ax2.set_title("B. Statistical transmutation: 1 quantum → fermion")
    ax2.legend(fontsize=8, loc="lower right")

    # C: the three faces agree
    ax3.axhline(0, color=GRAY, lw=1.0, zorder=1)
    for key, color, label, off in (("stat_sign", BLUE, "gauge (Aharonov–Bohm)", -0.22),
                                    ("braid_sign", ORANGE, "topological braid", 0.0),
                                    ("rot_sign", PURPLE, "spin (2π rotation)", 0.22)):
        ax3.bar([q + off for q in qs], [r[key] for r in rows], width=0.2,
                color=color, zorder=3, label=label)
    ax3.set_xticks(qs)
    ax3.set_ylim(-1.35, 1.35)
    ax3.set_xlabel("vortex winding  q  (s = q/2)")
    ax3.set_ylabel("exchange sign")
    ax3.set_title("C. Three faces of (−1)$^{2s}$: one sign")
    ax3.legend(fontsize=8, loc="lower right")

    # D: verdict
    v1, v2, v3 = verdict(rows)
    ax4.axis("off")
    ax4.set_title("D. The verdict")
    txt = ("exchange sign as an Aharonov-Bohm phase\n\n"
           f"AB1 holonomy = flux Φ=2πq   -> {'YES ok' if v1 else 'no'}\n"
           "   unit charge: +1 (Dirac)\n"
           "   half charge: (-1)^q (a sign)\n\n"
           f"AB2 flux-charge = fermion    -> {'YES ok' if v2 else 'no'}\n"
           + "".join(f"   q={r['q']}: stat {r['stat_sign']:+.0f} "
                     f"({'fermion' if r['q'] % 2 else 'boson'})\n" for r in rows)
           + f"\nAB3 spin=braid=gauge         -> {'YES ok' if v3 else 'no'}\n"
           "   one (-1)^2s, three ways\n\n"
           "the gauge field turns the winding into a\n"
           "dynamical exchange statistic (Wilczek flux\n"
           "attachment). classical abelian phase;\n"
           "Fock {psi,psi+} anticommutation is next.")
    ax4.text(0.02, 0.98, txt, transform=ax4.transAxes, fontsize=8.8, va="top",
             family="monospace")

    fig.suptitle("The Aharonov–Bohm statistical phase: the fermion's −1 as a "
                 "gauge holonomy of the self-consistent flux",
                 fontsize=12, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--sep", type=float, default=32.0)
    p.add_argument("--core", type=float, default=3.0)
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--flux-radius", type=float, default=6.0)
    p.add_argument("--charges", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--braid-n", type=int, default=200,
                   help="Lattice for the topological-braid cross-check (AB3).")
    p.add_argument("--braid-sep", type=float, default=34.0)
    p.add_argument("--braid-steps", type=int, default=241)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_ab")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n = 64
        args.sep = 32.0
        args.steps = 2200
        args.charges = [1, 2]
        args.braid_n = 140
        args.braid_sep = 24.0
        args.braid_steps = 181

    os.makedirs(args.output_dir, exist_ok=True)
    print("the Aharonov–Bohm statistical phase: statistics as a gauge holonomy",
          flush=True)
    rows = measure(args)
    lines, n = summarise(rows, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_ab_statistics.json"), "w") as fh:
        json.dump({"params": vars(args), "rows": rows, "score": n},
                  fh, indent=2, default=str)
    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_ab_statistics.png")
        make_figure(rows, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
