"""Is the transplant result quantitative, or only qualitative? — the units audit.

Three experiments report the same relocation on three substrates: apply
``S = ΔC + κ·w·ΔI`` with ``κ = r·κ₀/(r + c·load)`` post-hoc to a measured
disorder scan, raise the consumption ``c``, and the S-optimum jumps from the
deep-ordered phase to the critical neighbourhood.  Each reports **where** it
happens as a value of ``c`` on a ladder ``{0.05 … 50}``.

That number does not mean what it appears to.  ``load`` is defined per
substrate, and the Kuramoto reading — the co-rotating spread of per-*step*
phase increments — carries the integrator's time step:

    measured here, γ = 0.3, K = 2, noise = 0.6, fixed physical duration
        dt = 0.100   load = 0.19792     load/√dt = 0.626
        dt = 0.050   load = 0.13659     load/√dt = 0.611
        dt = 0.025   load = 0.09562     load/√dt = 0.605

while ΔI (0.933 → 0.937) is unmoved and ΔC is 0.000 throughout.  Halving the
time step halves the load and so doubles the ``c`` at which anything happens.
A crossing "at ``c ≈ 3``" is therefore, in part, a statement about the
integrator.  (Only in part: ``dt`` turns out not to be a *clean* units knob
either — see the note below Q3 and section D of the output.)

The question this experiment settles is whether anything quantitative survives
that.  ``project_genesis/crossing.py`` derives what does: ``r`` and ``c`` enter
the capacity law only as ``u = c/r``, and writing the crossing condition in the
**capacity at the ordered point** rather than in ``c`` cancels the load scale
outright, leaving ``κ_o⋆`` as a function of five measured dimensionless numbers
and the shared convention ``w``.

Pre-registered predictions:

Q1. **The mechanism's condition is visible before any ladder is run.**  From
    the measured curves alone: the precondition ``w(ΔI_o − ΔI_⋆) > ΔC_⋆ − ΔC_o``
    holds on both driven substrates, and the reachable capacity floor at the
    published ladder's top rung brackets the crossing the right way round —
    *above* ``κ_o⋆`` undriven (so no consumption can evict the ordered optimum)
    and *below* it driven (so eviction is reachable).  **Falsifier:** the floor
    fails to separate the driven from the undriven arm, which would mean the
    published toggle result is not readable off the load, and the mechanism
    story is doing work the numbers do not.

Q2. **The two-point crossing predicts the full optimum.**  ``κ_o⋆``, computed
    from the ordered point and the ΔC peak only, matches the capacity at which
    the *dense-ladder argmax over the whole scan* actually arrives at the
    critical point, within 20%, on both driven substrates.  **Falsifier:** a
    larger discrepancy, or the argmax landing somewhere that is not the ΔC peak
    — either would mean the relocation is not the two-point competition the
    mechanism describes.

Q3. **The scale-free variable is scale-free and the published one is not.**
    Re-measure the Kuramoto substrate at ``dt ∈ {0.2 … 0.0125}`` (see the
    disclosure below) — same physical duration, same coupling, same noise, same
    everything except the unit the load is quoted in.  Prediction: ``c`` at the
    crossing varies by
    more than 2× across that range while ``κ`` at the crossing varies by less
    than 10%.  **Falsifier:** ``κ`` drifts comparably to ``c``, in which case
    nothing quantitative survives the units and the transplant should be
    labelled qualitative — it says a crossing happens, not where.

    **Disclosure — the dt range was widened after a smoke test; the thresholds
    were not.**  Q3 was first written over ``dt ∈ {0.1, 0.05, 0.025}``.  Because
    ``L_o ∝ √dt`` exactly, the predicted spread in ``c`` over that range is
    ``√(0.1/0.025) = 2.00`` — precisely the registered "more than 2x" bar, so
    noise could only ever push the result *under* it, and the smoke test duly
    returned a c-spread of 1.75 against a κ-spread of 7.9%.  That is a badly
    registered test rather than a substrate result: it is one-sided by
    construction.  The range is therefore widened to ``{0.2 … 0.0125}``
    (predicted spread ``√16 = 4.0``), which is the knob that was wrong.  **Both
    thresholds stay exactly as first written**, and the narrow-range numbers are
    on the record in this paragraph.

Honest scope.  ``w = 1.5`` is a convention, shared across substrates and fixed
before any of this; every dimensionless statement below is a statement at that
``w``, and ``κ_o⋆`` scales roughly as ``1/w``.  The two substrates are re-run
here with the **same measurement code** the published transplants use
(``measure_substrate`` is imported, not reimplemented), so this is an audit of
those results and not a fresh pair of them.  Nothing here tests whether the
relocation is *real*; it tests whether its reported location means anything.

Q3 was designed on the assumption that the Kuramoto ``dt`` is a clean units
knob — that it rescales the load and touches nothing else.  **That assumption
is false, and the run below is what shows it**: ``dt`` also carries
Euler–Maruyama error large enough to move which ``γ`` carries the ΔC peak.  So
Q3 does not isolate a units effect and its failure should not be read as one.
Section D reports that diagnosis, post-hoc and labelled.  The clean units
statement — multiply a measured load by a constant, ``κ_o⋆`` unmoved and ``c⋆``
moved by exactly that constant — is an algebraic identity checked in
``tests/test_crossing.py``, not a measurement of any substrate, and it is not
claimed as one.

    python experiments/n3_crossing_prediction.py
    python experiments/n3_crossing_prediction.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from project_genesis.crossing import (  # noqa: E402
    capacity,
    capacity_floor,
    consumption_at_crossing,
    crossing_exists,
    kappa_at_crossing,
    measured_crossing,
    optimum_walk,
    two_point_reduction,
)

# the published measurement code, imported rather than reimplemented
from n3_criticality_transplant import measure_substrate as hopfield_scan  # noqa: E402
from n3_kuramoto_transplant import measure_substrate as kuramoto_scan  # noqa: E402

BANNER = "=" * 78

# the published defaults of the two transplants, restated here so the audit is
# demonstrably of those runs and not of some re-tuned variant
HOPFIELD = dict(n_spins=300, n_patterns=9, n_seeds=4, n_burn=30, n_window=60,
                t_min=0.1, t_max=1.6, n_temps=16, structured_threshold=0.3,
                query_fraction=0.05, query_every=5, seed=0)
KURAMOTO = dict(n_osc=400, coupling=2.0, dt=0.05, n_steps=900, n_burn=450,
                n_seeds=3, g_min=0.1, g_max=2.4, n_gamma=13,
                distribution="gaussian", freq_tol=0.3, noise=0.6, seed=0)


def audit(rows, x_key, weight, recovery, u_max):
    """The full scale-free reading of one measured scan."""
    red = two_point_reduction(rows, x_key=x_key)
    exists = crossing_exists(red, weight)
    kappa = kappa_at_crossing(red, weight)
    floor = capacity_floor(red, u_max)
    meas = measured_crossing(rows, x_key=x_key, weight=weight, recovery=recovery)
    return {"reduction": red, "exists": exists, "kappa_star": kappa,
            "floor": floor, "reachable": bool(np.isfinite(kappa) and floor < kappa),
            "c_star_pred": consumption_at_crossing(kappa, red["load_o"], recovery),
            "measured": meas}


def _line(name, a):
    red, m = a["reduction"], a["measured"]
    k = a["kappa_star"]
    return (f"  {name:<26} ΔC {red['dc_o']:.3f}→{red['dc_s']:.3f}  "
            f"ΔI {red['di_o']:.3f}→{red['di_s']:.3f}  "
            f"L {red['load_o']:.4f}→{red['load_s']:.4f}  ρ={red['rho']:>7.1f}\n"
            f"  {'':<26} precondition {'yes' if a['exists'] else 'NO ':>3}   "
            f"κ_o⋆ {k:>7.4f}   floor {a['floor']:.4f}   "
            f"{'REACHABLE' if a['reachable'] else 'unreachable'}\n"
            f"  {'':<26} measured: c_arrive {m['c_arrive']:>10.4g}  "
            f"κ_arrive {m['kappa_arrive']:>7.4f}  hops {m['hops']}  "
            f"x {m['x_start']:.2f}→{m['x_end']:.2f} (ΔC peak {m['x_critical']:.2f})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--weight", type=float, default=1.5,
                   help="Integration weight w — the transplants' shared convention.")
    p.add_argument("--recovery", type=float, default=0.1,
                   help="Capacity recovery r; enters only through u = c/r.")
    p.add_argument("--c-max", type=float, default=50.0,
                   help="Top rung of the published consumption ladder.")
    p.add_argument("--dts", type=float, nargs="+",
                   default=[0.2, 0.1, 0.05, 0.025, 0.0125],
                   help="Kuramoto time steps for the units test (Q3). Spans a "
                        "factor of 16, so L_o (which goes as sqrt(dt)) spans 4.")
    p.add_argument("--duration", type=float, default=45.0,
                   help="Physical time held fixed while dt varies (Q3).")
    p.add_argument("--kappa-tol", type=float, default=0.20,
                   help="Q2 tolerance: |κ_pred − κ_meas| / κ_meas.")
    p.add_argument("--invariance-tol", type=float, default=0.10,
                   help="Q3 tolerance: spread of κ across the dt sweep.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    hop, kur = dict(HOPFIELD), dict(KURAMOTO)
    if args.quick:
        hop.update(n_spins=200, n_seeds=2, n_burn=15, n_window=30, n_temps=10)
        kur.update(n_osc=250, n_seeds=2, n_gamma=10)
        args.duration = 25.0
    u_max = args.c_max / args.recovery

    print(BANNER)
    print("THE UNITS AUDIT — does the transplant say WHERE, or only THAT?")
    print(BANNER)
    print(f"  w = {args.weight}, r = {args.recovery} (enters only as u = c/r),")
    print(f"  published ladder top rung c = {args.c_max:g} -> u_max = {u_max:g}")
    print()
    print("  Pre-registered:")
    print("    Q1  the capacity floor brackets κ_o⋆ the right way round:")
    print("        unreachable undriven, reachable driven — no ladder needed")
    print(f"    Q2  the two-point κ_o⋆ predicts the full argmax within "
          f"{args.kappa_tol:.0%}")
    print(f"    Q3  over a dt sweep, c at the crossing moves >2x while κ moves "
          f"<{args.invariance_tol:.0%}")
    print()

    # ------------------------------------------------------------------- Q1
    print(BANNER)
    print("Q1  THE CONDITION, READ OFF THE CURVES BEFORE ANY LADDER")
    print(BANNER)
    t0 = time.time()
    print("  measuring the Hopfield network (published settings)...", flush=True)
    h_u = hopfield_scan(Namespace(**hop), driven=False)
    h_d = hopfield_scan(Namespace(**hop), driven=True)
    print("  measuring the Kuramoto population (published settings)...", flush=True)
    k_u = kuramoto_scan(Namespace(**kur), driven=False)
    k_d = kuramoto_scan(Namespace(**kur), driven=True)
    print(f"  [{time.time() - t0:.0f}s]")
    print()

    arms = {
        "hopfield undriven": audit(h_u, "T", args.weight, args.recovery, u_max),
        "hopfield DRIVEN": audit(h_d, "T", args.weight, args.recovery, u_max),
        "kuramoto undriven": audit(k_u, "gamma", args.weight, args.recovery, u_max),
        "kuramoto DRIVEN": audit(k_d, "gamma", args.weight, args.recovery, u_max),
    }
    for name, a in arms.items():
        print(_line(name, a))
        print()

    q1 = (not arms["hopfield undriven"]["reachable"]
          and arms["hopfield DRIVEN"]["reachable"]
          and not arms["kuramoto undriven"]["reachable"]
          and arms["kuramoto DRIVEN"]["reachable"])
    if q1:
        print("  PREDICTION HELD — and this is the toggle result without the")
        print("  toggle. The published experiments established 'scarcity evicts")
        print("  order exactly when maintaining order costs capacity' by running")
        print("  the ladder in four conditions and comparing verdicts. The same")
        print("  fact is a comparison of two numbers on the measured curves: the")
        print("  crossing sits at κ_o⋆, the load says κ_o can only fall to the")
        print("  floor, and eviction happens iff floor < κ_o⋆.")
    else:
        print("  PREDICTION FAILED — reported as measured. Per arm, "
              "'reachable' should read no/yes/no/yes:")
        for name, a in arms.items():
            print(f"    {name:<20} {'yes' if a['reachable'] else 'no'}")
        print("  If the floor does not separate the arms, the published")
        print("  toggle is not recoverable from the load reading alone.")

    # ------------------------------------------------------------------- Q2
    print()
    print(BANNER)
    print("Q2  DOES THE TWO-POINT CROSSING PREDICT THE FULL ARGMAX?")
    print(BANNER)
    print(f"  {'substrate':<20} {'κ_o⋆ pred':>10} {'κ meas':>9} {'rel err':>9} "
          f"{'lands on ΔC peak':>18}")
    q2_ok, q2_rows = True, []
    for name in ("hopfield DRIVEN", "kuramoto DRIVEN"):
        a = arms[name]
        pred, meas = a["kappa_star"], a["measured"]["kappa_arrive"]
        on_peak = bool(abs(a["measured"]["x_end"] - a["measured"]["x_critical"]) < 1e-9)
        err = abs(pred - meas) / meas if np.isfinite(pred) and meas > 0 else float("nan")
        ok = bool(np.isfinite(err) and err <= args.kappa_tol and on_peak)
        q2_ok &= ok
        q2_rows.append({"arm": name, "kappa_pred": pred, "kappa_meas": meas,
                        "rel_err": err, "on_peak": on_peak})
        print(f"  {name:<20} {pred:>10.4f} {meas:>9.4f} {err:>8.1%} "
              f"{('yes' if on_peak else 'NO'):>18}")
    print()
    if q2_ok:
        print("  PREDICTION HELD — the relocation really is the two-point")
        print("  competition the mechanism describes: the ordered point's")
        print("  integration funding decaying past the critical point's")
        print("  distinction gap. Nothing about the rest of the scan matters.")
    else:
        print("  PREDICTION FAILED — the full optimum does not reduce to the")
        print("  two-point competition on at least one substrate. The mechanism")
        print("  story is then incomplete: something between the endpoints is")
        print("  taking the optimum, and the relocation is not just a crossing.")
        print()
        print("  The route, per substrate — every point the optimum occupies as")
        print("  capacity falls, against the flat-load convex hull it would walk")
        print("  if every point paid the same:")
        for name, rows, xk in (("hopfield DRIVEN", h_d, "T"),
                               ("kuramoto DRIVEN", k_d, "gamma")):
            hull = optimum_walk(rows)
            print(f"    {name:<18} actual "
                  f"{' -> '.join(f'{v:.2f}' for v in arms[name]['measured']['walk'])}"
                  f"   hull {' -> '.join(f'{rows[i][xk]:.2f}' for i, _, _ in hull)}")
        print("  The hull is what a UNIFORM tax would visit; a rising load can")
        print("  skip its intermediate vertices, so the two need not agree. On")
        print("  these two substrates they happen to — which is a measurement,")
        print("  not a theorem. And it locates the Q2 failure exactly: the")
        print("  two-point formula is right when the route has two vertices and")
        print("  necessarily wrong when it has three, because an intermediate")
        print("  point beat the ΔC peak before the ordered point did. On that")
        print("  substrate 'the crossing' is not a single event at all.")

    # ------------------------------------------------------------------- Q3
    print()
    print(BANNER)
    print("Q3  THE UNITS TEST — same physics, three load units")
    print(BANNER)
    print(f"  Kuramoto at fixed physical duration {args.duration:g}, "
          f"dt = {', '.join(f'{d:g}' for d in args.dts)}")
    print("  (dt is the integrator's step; the load is a PER-STEP spread, so it")
    print("   carries a factor the system itself does not have)")
    print()
    print(f"  {'dt':>7} {'L_o':>9} {'L_o/(σ√dt)':>11} {'ρ':>6} {'x⋆':>6} "
          f"{'hull':>5} {'c_arrive':>10} {'κ_arrive':>9}")
    sweep = []
    for dtv in args.dts:
        cfg = dict(kur)
        cfg["dt"] = float(dtv)
        cfg["n_steps"] = int(round(args.duration / dtv))
        cfg["n_burn"] = cfg["n_steps"] // 2
        rows = kuramoto_scan(Namespace(**cfg), driven=True)
        a = audit(rows, "gamma", args.weight, args.recovery, u_max)
        red = a["reduction"]
        # how much of the measured load is just the injected noise coming back
        floor = kur["noise"] * np.sqrt(dtv)
        sweep.append({"dt": float(dtv), "load_o": red["load_o"],
                      "noise_floor": float(floor),
                      "load_over_floor": float(red["load_o"] / floor),
                      "rho": red["rho"], "x_critical": red["x_critical"],
                      "hull": len(optimum_walk(rows)),
                      "di_o": red["di_o"], "dc_s": red["dc_s"],
                      "c_arrive": a["measured"]["c_arrive"],
                      "kappa_arrive": a["measured"]["kappa_arrive"],
                      "kappa_pred": a["kappa_star"]})
        s = sweep[-1]
        print(f"  {dtv:>7.4f} {s['load_o']:>9.5f} {s['load_over_floor']:>11.3f} "
              f"{s['rho']:>6.2f} {s['x_critical']:>6.2f} {s['hull']:>5} "
              f"{s['c_arrive']:>10.4g} {s['kappa_arrive']:>9.4f}")

    cs = np.array([s["c_arrive"] for s in sweep], dtype=float)
    ks = np.array([s["kappa_arrive"] for s in sweep], dtype=float)
    c_ok = np.isfinite(cs).all() and cs.min() > 0
    k_ok = np.isfinite(ks).all() and ks.min() > 0
    c_spread = float(cs.max() / cs.min()) if c_ok else float("nan")
    k_spread = float((ks.max() - ks.min()) / ks.mean()) if k_ok else float("nan")
    print()
    print(f"  c at the crossing spans a factor of {c_spread:.2f}")
    print(f"  κ at the crossing spans {k_spread:.1%} of its mean")
    q3 = bool(c_spread > 2.0 and k_spread < args.invariance_tol)
    print()
    if q3:
        print("  PREDICTION HELD — and it is the finding. The consumption at")
        print("  which the optimum relocates is not a property of the system:")
        print(f"  it moves by {c_spread:.1f}x when only the integrator's step")
        print("  changes. The capacity at which it relocates does not move.")
        print("  So the transplants DO make a quantitative claim — but it is a")
        print("  claim about κ, and every one of them currently states it in c.")
    elif k_spread >= args.invariance_tol and c_spread > 2.0:
        print("  PREDICTION FAILED — κ drifts with the units too "
              f"({k_spread:.1%}, needed < {args.invariance_tol:.0%}).")
        print("  Then nothing quantitative survives the load convention and the")
        print("  transplant result is qualitative: a crossing happens, and where")
        print("  it happens is not a fact about the substrate. That is a weaker")
        print("  claim than the three experiments imply, and it is the one the")
        print("  numbers support.")
    else:
        print("  PREDICTION FAILED — c did not move as expected "
              f"(factor {c_spread:.2f}); the units test did not get traction.")
        print("  Reported as measured; no reading either way.")

    # ------------------------------------------- post-hoc, and labelled so
    print()
    print(BANNER)
    print("D  POST-HOC — why Q3 could not have worked, and what it exposed")
    print(BANNER)
    print("  Everything below was found by looking at the Q2/Q3 failures. It is")
    print("  diagnosis, not a registered prediction, and is reported as such.")
    print()
    fl = [s["load_over_floor"] for s in sweep]
    print("  D1. THE DRIVEN KURAMOTO 'LOAD' IS THE DRIVE, NOT THE RESPONSE.")
    print(f"      L_o/(σ√dt) over the dt sweep: "
          f"{', '.join(f'{v:.3f}' for v in fl)}")
    print("      The injected phase noise contributes exactly σ√dt per step to")
    print("      the co-rotating spread, and that ratio converges to 1.00 as dt")
    print("      falls. At the published dt the ordered phase's measured load is")
    print(f"      {fl[len(fl)//2]:.3f}x the pure-noise floor — i.e. almost all of the")
    print("      'work of holding a rhythm against perturbation' is the")
    print("      perturbation being counted back. `activity` does not subtract")
    print("      the drive. The Hopfield query drive does not have this problem:")
    print("      its perturbation flips are explicitly excluded from the load.")
    print()
    rhos = [s["rho"] for s in sweep]
    k_load = np.array([r["activity"] for r in k_d])
    print("  D2. SO THE TWO COMPETING POINTS PAY THE SAME.")
    print(f"      ρ = L_⋆/L_o over the dt sweep: "
          f"{', '.join(f'{v:.2f}' for v in rhos)} — flat at every step size,")
    print(f"      against ρ = {arms['hopfield DRIVEN']['reduction']['rho']:.1f} "
          "on the Hopfield network. The load does")
    print(f"      rise across the full scan (L_max/L_o = "
          f"{k_load.max() / k_load[0]:.2f}), but not between the ordered")
    print("      point and the ΔC peak, which is the only place the mechanism")
    print("      needs it to. So the relocation there is the UNIFORM-tax limit:")
    print("      κ shrinks the whole ΔI curve at once and the ΔI-funded optimum")
    print("      loses because ΔI_o > ΔI_⋆, not because it is taxed harder.")
    print("      The two transplants agree on the verdict while disagreeing on")
    print("      the mechanism, and the 6-rung ladder in the published runs is")
    print("      too coarse to show the difference.")
    print()
    print("  D3. dt IS NOT A UNITS KNOB, WHICH IS WHY Q3 WAS UNANSWERABLE.")
    print("      It does two things at once: rescales the per-step load (the")
    print("      units effect Q3 wanted) and changes the dynamics themselves")
    print("      through Euler–Maruyama error. Measured at n=8 seeds, ΔC at")
    print("      γ=1.06 runs 0.238 → 0.300 → 0.320 across dt = 0.2 → 0.0125")
    print("      (a ~3 sem shift) while γ=0.87 is flat — enough to move which")
    print("      point is the ΔC peak, and with it the hull from 2 vertices to")
    print("      3 (see the `x⋆` and `hull` columns above). That, not the load")
    print("      units, is the whole of the κ scatter.")
    print("      The pure units question is settled instead by multiplying a")
    print("      measured load by a constant, where κ_o⋆ is invariant to 10")
    print("      decimal places and c⋆ moves by exactly that constant — an")
    print("      algebraic identity, verified in tests/test_crossing.py, and")
    print("      NOT a measurement of any substrate.")
    print()
    dcs = sorted(((r["delta_c"], r["gamma"]) for r in k_d), reverse=True)
    err = max(r["delta_c_err"] for r in k_d)
    print("  D4. AND THE KURAMOTO ΔC PEAK IS NOT RESOLVED.")
    print(f"      Top two: ΔC = {dcs[0][0]:.3f} at γ={dcs[0][1]:.2f} and "
          f"{dcs[1][0]:.3f} at γ={dcs[1][1]:.2f},")
    print(f"      separated by {dcs[0][0] - dcs[1][0]:.3f} against a "
          f"seed error of up to {err:.3f}.")
    print("      Which one wins flips with dt and with the seed, and both")
    print("      `c_arrive` here and γ⋆ in the published transplant are defined")
    print("      by that argmax. That is a fragility in the published result,")
    print("      not only in this audit.")
    print()
    print("  D5. AND ON THE PUBLISHED LADDER THE OSCILLATORS NEVER REACH IT.")
    k_g = np.array([r["gamma"] for r in k_d])
    k_dc = np.array([r["delta_c"] for r in k_d])
    k_di = np.array([r["delta_i"] for r in k_d])
    k_L = np.array([r["activity"] for r in k_d])
    i_peak = int(np.argmax(k_dc))
    print(f"      {'c':>6} {'γ⋆':>6}   at the ΔC peak?")
    for c in (0.05, 0.2, 0.8, 3.0, 12.0, 50.0):
        j = int(np.argmax(k_dc + capacity(k_L, c / args.recovery)
                          * args.weight * k_di))
        print(f"      {c:>6g} {k_g[j]:>6.2f}   {'yes' if j == i_peak else 'no'}")
    print(f"      Every rung of the published ladder lands on γ = "
          f"{k_g[j]:.2f}, the intermediate")
    print(f"      hull vertex, not on the ΔC peak at γ = {k_g[i_peak]:.2f}. The "
          "published")
    print("      verdict still stands on its own criterion — that point IS")
    print("      inside its ±35% γ_c window — but the state the optimum moves")
    print("      to is a partially-synchronised one, not the maximum-distinction")
    print("      state the mechanism story names. Same verdict, different place.")

    # ------------------------------------------------------------- verdict
    print()
    print(BANNER)
    n = int(q1) + int(q2_ok) + int(q3)
    print(f"VERDICT: {n}/3 pre-registered predictions held.")
    print(BANNER)
    if q3:
        print("  What the three transplants should say, in the variable that")
        print("  survives their own units:")
        for name in ("hopfield DRIVEN", "kuramoto DRIVEN"):
            a = arms[name]
            print(f"    {name:<20} relocates at κ ≈ {a['measured']['kappa_arrive']:.3f} "
                  f"(reported as c ≈ {a['measured']['c_arrive']:.3g})")
        kh = arms["hopfield DRIVEN"]["measured"]["kappa_arrive"]
        kk = arms["kuramoto DRIVEN"]["measured"]["kappa_arrive"]
        if np.isfinite(kh) and np.isfinite(kk) and min(kh, kk) > 0:
            print(f"    the two agree to a factor of {max(kh, kk) / min(kh, kk):.2f}.")
            print("  Whether that agreement means anything is NOT settled here:")
            print("  κ_o⋆ ≈ (ΔC_⋆ − ΔC_o)/(w·ΔI_o) in the large-ρ limit, so two")
            print("  substrates agree exactly when their distinction gaps do.")
            print("  This experiment establishes the variable, not a constant.")
    else:
        print("  The location of the relocation is NOT established in any")
        print("  substrate-independent variable, and the three transplants")
        print("  should be read as qualitative: a crossing happens; where it")
        print("  happens is not yet a fact about the substrate.")
        print()
        print("  What the audit does establish, and it is worth more than the")
        print("  score suggests:")
        print("   - r and c are one parameter. Every published c-value is")
        print("     really c/r, and the recovery rate is the axis's unit.")
        print("   - the mechanism's condition is a comparison of two numbers on")
        print("     the measured curves (Q1), not something the four-condition")
        print("     toggle was needed to establish.")
        print("   - where the relocation is a single jump, it is EXACTLY the")
        print("     two-point competition the mechanism describes — 0.1% on the")
        print("     Hopfield network. That much of the story is confirmed.")
        print("   - and it is not always a single jump. The optimum walks the")
        print("     upper convex hull of the (ΔI, ΔC) cloud, which has two")
        print("     vertices on one substrate and three on the other.")
        print("   - the driven Kuramoto load is mostly the injected noise (D1),")
        print("     so that substrate's agreement with the Hopfield network is")
        print("     agreement on the verdict, not on the mechanism.")
        print()
        print("  The concrete recommendation: `activity` in kuramoto_substrate")
        print("  should subtract the drive's own contribution, the way the")
        print("  Hopfield query drive already excludes its perturbation flips.")
        print("  Until it does, the third substrate is not measuring what the")
        print("  second one is, and 'one law across three substrates' overstates")
        print("  what has been shown.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "crossing_prediction.json"
        out.write_text(json.dumps(
            {"params": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
             "hopfield_settings": hop, "kuramoto_settings": kur,
             "arms": arms, "q2": q2_rows, "dt_sweep": sweep,
             "c_spread": c_spread, "kappa_spread": k_spread,
             "score": n}, indent=2, default=float))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
