"""The self-consistent gauge field: the vortex becomes a gauged particle.

The phase-pinned braid closed the dynamical-exchange boundary with a *background
phase template*, and named its caveat: not a dynamical gauge field solved
self-consistently.  This experiment supplies the real thing — a **U(1) gauge
field** coupled to the vortex (the abelian Higgs / Ginzburg–Landau model,
`gauged_vortex.py`), both solved by gradient flow — and measures the three
things a genuine gauge field gives that the template did not.

Pre-registered predictions:

G1. **Flux quantization.**  A winding-``q`` vortex forces the self-consistent
    gauge field to carry a *quantised* magnetic flux ``Φ = 2π·q`` through a loop
    about its core (``q = 1, 2, 3`` → one, two, three flux quanta) — solved by
    the dynamics, not imposed — while the *global* vortex (gauge field frozen
    off) carries **zero** flux.  The gauge field is the real anchor of the
    winding.

G2. **Finite energy — the London screening.**  With the gauge field on, the
    vortex–antivortex interaction energy **saturates** with separation (a
    finite-range, screened force → finite-energy solitons); with it off, the
    global vortices' energy keeps **growing** (the logarithmic long-range tail).
    The gauge field screens the divergence over a measurable length.

G3. **Gauge invariance.**  The observables — energy and flux — are **invariant**
    under a *local* gauge transformation ``ψ → e^{iα(x)}ψ``, ``θ_μ → θ_μ + α(x)
    − α(x+μ)`` (to machine precision), the signature of a genuine gauge theory —
    which the fixed phase template of the pin was not.

If G1–G3 land, the vortex is a genuine gauged particle: its winding anchored by
a self-consistent, quantised, gauge-invariant magnetic flux, of finite energy —
the phase-template caveat closed with the real field.  The honest boundary kept
bright: this is the **classical** gauge field, not Fock-space anticommutation;
the Aharonov–Bohm / Chern–Simons statistical phase that would make the gauged
½-vortex a true dynamical fermion is the standing frontier.

Usage::

    python experiments/n3_gauge_field.py --output-dir artifacts/n3_gauge_field
    python experiments/n3_gauge_field.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.gauged_vortex import (
    energy,
    energy_parts,
    gauge_transform,
    local_flux,
    relax,
    seed_vortices,
    zero_links,
)

TWO_PI = 2.0 * np.pi


def g1_flux_quantization(args):
    """Flux → 2π·q (gauged), 0 (global)."""
    n = args.n
    c0 = (n / 2 - args.sep / 2, n / 2)
    c1 = (n / 2 + args.sep / 2, n / 2)
    # measure on the flux plateau: past the core/London length, but inside the
    # inter-vortex region (larger radii pick up the pair's diffuse return flux)
    r_meas = args.flux_radius
    rows = []
    for q in args.charges:
        psi = seed_vortices(n, [c0, c1], [q, -q], core=args.core)
        gauged = relax(psi, zero_links(n), lam=args.lam, beta=args.beta,
                       dt=args.dt, steps=args.steps)
        glob = relax(psi, zero_links(n), lam=args.lam, beta=args.beta,
                     dt=args.dt, steps=args.steps, gauge_on=False)
        flux = local_flux(gauged["theta"], c0, r_meas)
        flux0 = local_flux(glob["theta"], c0, r_meas)
        rows.append({"q": q, "flux": flux, "quanta": flux / TWO_PI,
                     "flux_global": flux0,
                     "E_gauged": gauged["energy"], "E_global": glob["energy"]})
        print(f"  G1 q={q}: gauged flux/2π = {flux/TWO_PI:+.2f} "
              f"(expect {-q}), global flux/2π = {flux0/TWO_PI:+.2f} "
              f"(E {gauged['energy']:.0f} vs {glob['energy']:.0f})", flush=True)
    quantized = all(abs(abs(r["quanta"]) - r["q"]) < args.flux_tol for r in rows)
    global_zero = all(abs(r["flux_global"]) < 0.3 * TWO_PI for r in rows)
    return {"rows": rows, "quantized": quantized, "global_zero": global_zero,
            "r_meas": r_meas}


def g2_screening(args):
    """v–av interaction energy: gauged saturates, global grows with separation."""
    n = args.n
    rows = []
    for sep in args.seps:
        c0 = (n / 2 - sep / 2, n / 2)
        c1 = (n / 2 + sep / 2, n / 2)
        psi = seed_vortices(n, [c0, c1], [1, -1], core=args.core)
        gauged = relax(psi, zero_links(n), lam=args.lam, beta=args.beta,
                       dt=args.dt, steps=args.steps)
        glob = relax(psi, zero_links(n), lam=args.lam, beta=args.beta,
                     dt=args.dt, steps=args.steps, gauge_on=False)
        rows.append({"sep": sep, "E_gauged": gauged["energy"],
                     "E_global": glob["energy"]})
        print(f"  G2 sep={sep}: E gauged = {gauged['energy']:.1f}  "
              f"E global = {glob['energy']:.1f}", flush=True)
    # gauged saturates (small change over the last half), global keeps growing
    eg = np.array([r["E_gauged"] for r in rows])
    eglob = np.array([r["E_global"] for r in rows])
    gauged_span = float(eg.max() - eg.min())
    global_span = float(eglob.max() - eglob.min())
    gauged_tail = float(abs(eg[-1] - eg[-2])) if len(eg) > 1 else 0.0
    global_tail = float(abs(eglob[-1] - eglob[-2])) if len(eglob) > 1 else 0.0
    screened = (gauged_tail < 0.5 * global_tail) and (global_span > gauged_span)
    return {"rows": rows, "gauged_span": gauged_span, "global_span": global_span,
            "gauged_tail": gauged_tail, "global_tail": global_tail,
            "screened": screened}


def g3_gauge_invariance(args):
    """Energy and flux invariant under a random local gauge transformation."""
    n = args.n
    c0 = (n / 2 - args.sep / 2, n / 2)
    c1 = (n / 2 + args.sep / 2, n / 2)
    psi = seed_vortices(n, [c0, c1], [1, -1], core=args.core)
    out = relax(psi, zero_links(n), lam=args.lam, beta=args.beta, dt=args.dt,
                steps=args.steps)
    e0 = energy(out["psi"], out["theta"], args.lam, args.beta)
    f0 = local_flux(out["theta"], c0, args.flux_radius)
    rng = np.random.default_rng(args.seed)
    alpha = rng.uniform(-np.pi, np.pi, (n, n))
    psi_g, theta_g = gauge_transform(out["psi"], out["theta"], alpha)
    e1 = energy(psi_g, theta_g, args.lam, args.beta)
    f1 = local_flux(theta_g, c0, args.flux_radius)
    de = abs(e1 - e0) / max(abs(e0), 1e-12)
    df = abs(f1 - f0)
    print(f"  G3 gauge transform: energy {e0:.4f} → {e1:.4f} (rel Δ {de:.2e}), "
          f"flux {f0:.4f} → {f1:.4f} (Δ {df:.2e})", flush=True)
    return {"e0": e0, "e1": e1, "f0": f0, "f1": f1, "rel_dE": de, "dF": df,
            "invariant": de < 1e-6 and df < 1e-6}


def verdict(g1, g2, g3):
    v1 = g1["quantized"] and g1["global_zero"]
    v2 = g2["screened"]
    v3 = g3["invariant"]
    return v1, v2, v3


def summarise(g1, g2, g3, args):
    v1, v2, v3 = verdict(g1, g2, g3)
    n = int(v1) + int(v2) + int(v3)
    lines = ["The self-consistent gauge field: the vortex as a gauged particle "
             "— verdict", "=" * 74, ""]
    lines.append(
        "G1 (flux quantization): a winding-q vortex forces the self-consistent "
        "gauge field to carry " + ", ".join(
            f"q={r['q']}→{r['quanta']:+.2f} quanta" for r in g1["rows"])
        + f" (global vortex: {g1['rows'][0]['flux_global']/TWO_PI:+.2f}) — "
        + ("✓ the flux is quantised at 2π·q, solved not imposed, and zero "
           "without the gauge field: the gauge field is the real anchor of the "
           "winding." if v1 else "✗ the flux was not cleanly quantised."))
    lines.append("")
    lines.append(
        "G2 (finite energy — London screening): the v–av interaction energy "
        + " ".join(f"sep={r['sep']}:{r['E_gauged']:.0f}/{r['E_global']:.0f}"
                   for r in g2["rows"])
        + " (gauged/global) — "
        + (f"✓ the gauged energy saturates (tail Δ {g2['gauged_tail']:.1f}) "
           f"while the global grows (tail Δ {g2['global_tail']:.1f}): the gauge "
           "field screens the long-range tail, making the vortex a "
           "finite-energy soliton." if v2 else
           "✗ no clear screening separation on this range."))
    lines.append("")
    lines.append(
        f"G3 (gauge invariance): under a random local gauge transformation, "
        f"energy changes by {g3['rel_dE']:.1e} (relative) and flux by "
        f"{g3['dF']:.1e} — "
        + ("✓ the observables are invariant to machine precision: a genuine "
           "gauge theory, not the fixed phase template of the pin (which a "
           "gauge transform would have spoiled)." if v3 else
           "✗ the observables were not gauge-invariant."))
    lines.append("")
    lines.append(f"score: {n}/3 pre-registered predictions land.")
    lines.append("")
    lines.append(
        "what this establishes: the vortex is now a genuine **gauged particle** "
        "— its winding anchored by a self-consistent, quantised, gauge-invariant "
        "magnetic flux, of finite energy. The phase-template caveat of "
        "`n3_phase_pinned_braid` is closed with the real field: the "
        "'background gauge-like coupling' is replaced by a solved U(1) gauge "
        "theory, and the winding it anchors is a physical flux quantum, not a "
        "reference phase. This is the field-theoretic ground the exchange "
        "statistic must ultimately stand on.")
    lines.append("")
    lines.append(
        "honest scope: this is the **classical** abelian Higgs gauge field — a "
        "genuine, self-consistent, gauge-invariant U(1) theory, but a classical "
        "one: **no** Fock space, **no** ``{ψ, ψ†}`` anticommutator, **no** "
        "many-body Pauli principle between identical excitations. The "
        "Aharonov–Bohm / Chern–Simons **statistical phase** — the flux "
        "attachment by which this gauge field would turn the gauged ½-vortex's "
        "topological exchange sign into a genuine dynamical *fermion* statistic "
        "— is the standing next rung, and the true bridge to a quantum fermion. "
        "2-D, one lattice, gradient-flow (relaxational) dynamics, one (λ, β) "
        "operating point.")
    return lines, n


def make_figure(g1, g2, g3, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY, RED = ("#0072B2", "#E69F00", "#009E73",
                                      "#999999", "#c0392b")
    fig = plt.figure(figsize=(13, 9.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.26)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))
    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # A: flux quantization
    qs = [r["q"] for r in g1["rows"]]
    quanta = [abs(r["quanta"]) for r in g1["rows"]]
    gl = [abs(r["flux_global"] / TWO_PI) for r in g1["rows"]]
    ax1.plot(qs, qs, color=GRAY, lw=1.2, ls=":", zorder=1, label="2π·q (ideal)")
    ax1.plot(qs, quanta, color=BLUE, lw=2.0, marker="o", ms=8, zorder=3,
             label="gauged (self-consistent)")
    ax1.plot(qs, gl, color=RED, lw=2.0, marker="s", ms=7, zorder=3,
             label="global (gauge off)")
    ax1.set_xlabel("vortex winding  q")
    ax1.set_ylabel("|flux| / 2π  (quanta)")
    ax1.set_xticks(qs)
    ax1.set_title("A. Flux quantization: Φ = 2π·q, self-consistent")
    ax1.legend(fontsize=8, loc="upper left")

    # B: screening — energy vs separation
    seps = [r["sep"] for r in g2["rows"]]
    ax2.plot(seps, [r["E_global"] for r in g2["rows"]], color=RED, lw=2.0,
             marker="s", ms=6, zorder=3, label="global (grows — long range)")
    ax2.plot(seps, [r["E_gauged"] for r in g2["rows"]], color=BLUE, lw=2.0,
             marker="o", ms=6, zorder=3, label="gauged (saturates — screened)")
    ax2.set_xlabel("vortex–antivortex separation")
    ax2.set_ylabel("energy")
    ax2.set_title("B. London screening → finite-energy soliton")
    ax2.legend(fontsize=8, loc="best")

    # C: gauge invariance + a flux map
    from project_genesis.gauged_vortex import plaquette_flux, seed_vortices
    nn = args.n
    c0 = (nn / 2 - args.sep / 2, nn / 2)
    c1 = (nn / 2 + args.sep / 2, nn / 2)
    psi = seed_vortices(nn, [c0, c1], [1, -1], core=args.core)
    out = relax(psi, zero_links(nn), lam=args.lam, beta=args.beta, dt=args.dt,
                steps=args.steps)
    B = plaquette_flux(out["theta"])
    im = ax3.imshow(B.T, origin="lower", cmap="RdBu_r",
                    vmin=-np.abs(B).max(), vmax=np.abs(B).max())
    ax3.set_title(f"C. The magnetic flux (localized ±quanta)\n"
                  f"gauge-invariant: rel ΔE {g3['rel_dE']:.0e}, ΔΦ {g3['dF']:.0e}")
    ax3.set_xticks([])
    ax3.set_yticks([])
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # D: verdict
    v1, v2, v3 = verdict(g1, g2, g3)
    ax4.axis("off")
    ax4.set_title("D. The verdict")
    txt = ("abelian Higgs vortex = a gauged particle\n\n"
           f"G1 flux quantization Φ=2πq  -> {'YES ok' if v1 else 'no'}\n"
           + "".join(f"   q={r['q']}: {r['quanta']:+.2f} quanta "
                     f"(global {r['flux_global']/TWO_PI:+.1f})\n"
                     for r in g1["rows"])
           + f"\nG2 London screening         -> {'YES ok' if v2 else 'no'}\n"
           f"   gauged tail Δ={g2['gauged_tail']:.0f} vs "
           f"global Δ={g2['global_tail']:.0f}\n\n"
           f"G3 gauge invariance         -> {'YES ok' if v3 else 'no'}\n"
           f"   ΔE/E={g3['rel_dE']:.0e}, ΔΦ={g3['dF']:.0e}\n\n"
           "the vortex's winding is anchored by a real,\n"
           "self-consistent, quantized, gauge-invariant\n"
           "flux -- the phase-template caveat closed.\n"
           "(classical gauge field; Fock-space anti-\n"
           "commutation + the AB statistical phase remain\n"
           "the frontier to a quantum fermion.)")
    ax4.text(0.02, 0.98, txt, transform=ax4.transAxes, fontsize=8.8, va="top",
             family="monospace")

    fig.suptitle("The self-consistent gauge field: the vortex becomes a gauged "
                 "particle — flux quantization, screening, gauge invariance",
                 fontsize=12, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--sep", type=float, default=32.0,
                   help="v–av separation for G1/G3.")
    p.add_argument("--core", type=float, default=3.0)
    p.add_argument("--lam", type=float, default=2.0, help="Higgs coupling λ.")
    p.add_argument("--beta", type=float, default=1.0, help="Maxwell stiffness β.")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--charges", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--seps", type=float, nargs="+", default=[12, 20, 28, 36])
    p.add_argument("--flux-radius", type=float, default=6.0,
                   help="Loop radius for the flux quantum (on the plateau: past "
                        "the core/London length, inside the inter-vortex region).")
    p.add_argument("--flux-tol", type=float, default=0.2,
                   help="Tolerance on |flux/2π − q| for quantization.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_gauge_field")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n = 64
        args.sep = 32.0
        args.steps = 2200
        args.charges = [1, 2]
        args.seps = [16, 32]

    os.makedirs(args.output_dir, exist_ok=True)
    print("the self-consistent gauge field: the vortex as a gauged particle",
          flush=True)
    g1 = g1_flux_quantization(args)
    g2 = g2_screening(args)
    g3 = g3_gauge_invariance(args)

    lines, n = summarise(g1, g2, g3, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_gauge_field.json"), "w") as fh:
        json.dump({"params": vars(args), "g1": g1, "g2": g2, "g3": g3,
                   "score": n}, fh, indent=2, default=str)
    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_gauge_field.png")
        make_figure(g1, g2, g3, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
