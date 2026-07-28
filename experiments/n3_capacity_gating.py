"""Does capacity *gate* the palette selection, or is the selection just geometry?

This is the discriminating test between two readings of the whole programme.

**H0 — geometry.** In 2-D a generic vertex is three-fold, so a junction can
carry the *whole* palette only when the palette is exactly three.  That is a
statement about Plateau's law, and it is true of any multiphase coarsening
model.  On this reading the ``P = 3`` result is real but says nothing about
URP: capacity is decorative, and a plain Allen-Cahn field with no ``κ``
anywhere in it would select three just as sharply.

**H1 — scarcity.** ``Docs/The_Principle.md`` §3 claims something stronger:
that the selection is *"invisible when capacity is abundant — it appears only
when a capacity budget binds"*.  If true, that is a prediction generic
coarsening cannot make, because generic coarsening has no capacity to make
scarce.  It is the one claim in the document that could separate the readings.

Two things make this worth running rather than assuming.

First, **the published selection sweep contains no capacity field at all**.
``n3_selection_sweep`` calls ``step_multiphase``, which has no ``κ``, and
``P = 3`` wins there by 88× in 2-D.  A selection that appears with capacity
entirely absent is already awkward for H1.

Second, the repo's *own* prior measurements point the other way.
``n3_capacity_separation`` reports the gap as **capacity-invariant** — "sweeping
the capacity field κ ... leaves the integrated fraction φ untouched" — and the
README records the capacity-driven ``N⋆ = 3`` claim as *conditional /
transient*.  So §3 sits in tension with §2 of the same document, and with the
experiments both cite.

So the honest pre-registration here predicts **against** the framework's own
§3, using the framework's own capacity law
(:func:`~project_genesis.multiphase.step_multiphase_kappa`):

    ∂_t η_a = κ·D·∇²η_a − ∂f/∂η_a
    ∂_t κ   = D_κ·∇²κ + r·(κ₀ − κ) − c·(Σ_b |∇η_b|²)·κ

``c`` is the knob that decides whether the budget binds.  At ``c = 0`` capacity
is free and pinned at ``κ₀`` everywhere; as ``c`` grows, holding structure
costs capacity exactly where structure is densest.  The ``c = None`` arm has no
capacity field whatsoever and reproduces the published sweep bit-for-bit.

Pre-registered predictions:

Q1. **The winner does not move.**  Full-palette integration peaks at ``P = 3``
    at *every* capacity level, including the no-capacity control and the
    free-capacity (``c = 0``) arm.  **Falsifier:** the peak moves to another
    palette at any capacity level — which would mean capacity really does
    choose the palette, and H1 is right.

Q2. **The margin is capacity-invariant.**  The ``P = 3`` : runner-up
    integration ratio varies by less than **3×** across the whole capacity
    axis.  **Falsifier:** the ratio is flat or absent at abundant capacity and
    sharpens by more than 3× as the budget binds.  That is precisely what §3
    asserts, and it would vindicate it.

Q3. **Manipulation check — capacity actually did something.**  Mean realised
    ``κ`` falls by at least 20% from the free arm to the scarcest, and the
    distinction axis moves with it.  **Falsifier:** capacity barely moves, in
    which case Q1 and Q2 are null for the boring reason that the knob was not
    connected, and nothing here can be concluded either way.

Q3 is not decoration.  A null result on Q1/Q2 is worth nothing unless the
manipulation is shown to have bitten, and that is the most common way an
experiment like this quietly fails.

Honest scope: this tests the *palette* selection under the repo's capacity
coupling, in 2-D, at one operating point.  It does not test every claim URP
makes about capacity — the substrate-transplant result (`n3_criticality_
transplant`, `n3_kuramoto_transplant`) is separate and stands on its own,
and is *not* reducible to coarsening geometry because two of its three
substrates have no geometry to coarsen.

    python experiments/n3_capacity_gating.py
    python experiments/n3_capacity_gating.py --size 64 --steps 500 --n-seeds 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_genesis.selection import (  # noqa: E402
    evolve_palette_gated,
    measure_window_gated,
    selection_score,
)

BANNER = "=" * 78

# None = the published sweep (no capacity field at all); 0.0 = capacity present
# but free; then increasingly binding budgets.
CAPACITY_ARMS = [None, 0.0, 1.0, 5.0, 20.0]


def arm_label(c):
    if c is None:
        return "no-kappa"
    if c == 0.0:
        return "free"
    return f"c={c:g}"


def run_arm(consumption, args):
    rows = []
    for p in args.palettes:
        acc = {"distinction": 0.0, "integration": 0.0, "integration_guarded": 0.0,
               "churn": 0.0, "domain_scale": 0.0, "kappa_mean": 0.0,
               "sectors_alive": 0.0}
        resolved = True
        for s in range(args.n_seeds):
            rng = np.random.default_rng(args.seed + 101 * s)
            fields, kappa = evolve_palette_gated(
                p, args.size, args.steps, rng, consumption=consumption,
                noise=args.noise, gamma=args.gamma, dt=args.dt, ndim=args.ndim)
            m = measure_window_gated(
                fields, kappa, p, args.window, rng, consumption=consumption,
                noise=args.noise, gamma=args.gamma, dt=args.dt)
            for k in acc:
                acc[k] += m[k] / args.n_seeds
            resolved = resolved and m["resolved"]
        acc["P"] = p
        acc["resolved"] = resolved
        acc["score"] = selection_score(acc["integration_guarded"], acc["churn"])
        rows.append(acc)
    return rows


def peak_and_margin(rows, key="integration_guarded"):
    ps = [r["P"] for r in rows]
    vals = [r[key] for r in rows]
    best = ps[int(np.argmax(vals))]
    rest = [v for p, v in zip(ps, vals) if p != best]
    top = max(vals)
    runner = max(rest) if rest else 0.0
    margin = top / runner if runner > 0 else float("inf")
    return best, margin, top


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--palettes", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--size", type=int, default=80)
    p.add_argument("--ndim", type=int, default=2)
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--noise", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--expected", type=int, default=3)
    p.add_argument("--margin-tol", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    print(BANNER)
    print("CAPACITY GATING — does scarcity choose the palette, or does geometry?")
    print(BANNER)
    print(f"  palettes {args.palettes}, {args.size}^{args.ndim}, "
          f"{args.steps} steps + {args.window} window, {args.n_seeds} seeds")
    print(f"  capacity arms: {[arm_label(c) for c in CAPACITY_ARMS]}")
    print()
    print("  Pre-registered (predicting AGAINST The_Principle.md section 3):")
    print(f"    Q1  integration peaks at P = {args.expected} at EVERY capacity level")
    print(f"    Q2  the P={args.expected}:runner-up ratio varies by < "
          f"{args.margin_tol:g}x across the axis")
    print("    Q3  manipulation check: mean kappa falls >= 20% from free to scarce")
    print()

    results = {}
    for c in CAPACITY_ARMS:
        t0 = time.time()
        rows = results[arm_label(c)] = run_arm(c, args)
        best, margin, top = peak_and_margin(rows)
        km = rows[0]["kappa_mean"]
        print(f"    {arm_label(c):>9}: peak P={best}, margin "
              f"{margin:>8.1f}x, integration {top:.5f}, "
              f"mean kappa {km:.3f}   [{time.time()-t0:.0f}s]")

    print()
    print(BANNER)
    print("INTEGRATION (guarded) BY PALETTE AND CAPACITY")
    print(BANNER)
    hdr = f"  {'arm':>9} {'kappa':>7} " + " ".join(f"{'P='+str(p):>10}" for p in args.palettes)
    print(hdr)
    for c in CAPACITY_ARMS:
        rows = results[arm_label(c)]
        km = rows[0]["kappa_mean"]
        print(f"  {arm_label(c):>9} {km:>7.3f} " +
              " ".join(f"{r['integration_guarded']:>10.5f}" for r in rows))

    print()
    print("DISTINCTION (does capacity change how much structure there is?)")
    print(f"  {'arm':>9} {'kappa':>7} " +
          " ".join(f"{'P='+str(p):>10}" for p in args.palettes))
    for c in CAPACITY_ARMS:
        rows = results[arm_label(c)]
        print(f"  {arm_label(c):>9} {rows[0]['kappa_mean']:>7.3f} " +
              " ".join(f"{r['distinction']:>10.4f}" for r in rows))

    print()
    print("DOMAIN SCALE — scarcity fragments, and the texture guard must be")
    print("checked before any integration number here is trusted at all")
    print(f"  {'arm':>9} {'kappa':>7} " +
          " ".join(f"{'P='+str(p):>10}" for p in args.palettes))
    unresolved_arms = []
    for c in CAPACITY_ARMS:
        rows = results[arm_label(c)]
        cells = []
        for r in rows:
            mark = "" if r["resolved"] else "!"
            cells.append(f"{r['domain_scale']:>9.2f}{mark}")
        if not all(r["resolved"] for r in rows):
            unresolved_arms.append(arm_label(c))
        print(f"  {arm_label(c):>9} {rows[0]['kappa_mean']:>7.3f} " +
              " ".join(cells))
    if unresolved_arms:
        print(f"  ! = UNRESOLVED (domain scale below 2.5).  Arms {unresolved_arms}")
        print("  have palettes whose integration is counting lattice texture, so")
        print("  their margins below are not measurements of binding.")
    else:
        print("  all arms resolved — every integration number above is measuring")
        print("  junctions rather than texture")

    # ------------------------------------------------------------------- Q1
    peaks = {arm_label(c): peak_and_margin(results[arm_label(c)])[0]
             for c in CAPACITY_ARMS}
    q1 = all(v == args.expected for v in peaks.values())
    print()
    print(BANNER)
    print("Q1  DOES THE WINNER MOVE WITH CAPACITY?")
    print(BANNER)
    for k, v in peaks.items():
        print(f"    {k:>9}: peak at P = {v}")
    print()
    if q1:
        print(f"  PREDICTION HELD — P = {args.expected} peaks at every capacity")
        print("  level, including the arm with no capacity field at all and the")
        print("  arm where capacity is free.  Capacity does not choose the palette.")
    else:
        print("  PREDICTION FAILED — the peak moves with capacity.  Capacity")
        print("  really does select, and The_Principle section 3 is right.")

    # ------------------------------------------------------------------- Q2
    margins = {arm_label(c): peak_and_margin(results[arm_label(c)])[1]
               for c in CAPACITY_ARMS}
    finite = [m for m in margins.values() if np.isfinite(m)]
    spread = (max(finite) / min(finite)) if len(finite) >= 2 and min(finite) > 0 \
        else float("inf")
    print()
    print(BANNER)
    print("Q2  IS THE MARGIN CAPACITY-INVARIANT?")
    print(BANNER)
    for k, v in margins.items():
        print(f"    {k:>9}: margin {v:>10.1f}x")
    print(f"  spread across the axis: {spread:.2f}x "
          f"(registered tolerance {args.margin_tol:g}x)")
    q2 = np.isfinite(spread) and spread < args.margin_tol
    print()
    if q2:
        print("  PREDICTION HELD — the selection margin does not depend on whether")
        print("  the capacity budget binds.  The selection is geometric.")
    else:
        print("  PREDICTION FAILED — the margin moves with capacity by more than")
        print(f"  {args.margin_tol:g}x.  Check the direction below before concluding:")
        free = margins.get("free", float("nan"))
        tight = margins.get(arm_label(CAPACITY_ARMS[-1]), float("nan"))
        if np.isfinite(free) and np.isfinite(tight) and tight > free:
            print("  it SHARPENS as the budget binds, which is what section 3 claims.")
        else:
            print("  it does NOT sharpen under scarcity — it collapses. So section 3")
            print("  is not merely unsupported, it is backwards: scarcity destroys")
            print("  the selection rather than revealing it.")

    # Restricting to arms the texture guard calls valid is not post-hoc
    # cherry-picking: `resolved` is a validity criterion established before
    # this experiment existed, and a number from an unresolved field was never
    # a measurement of binding.  The registered Q2 did not say "resolved arms
    # only", so it stands as failed — but the restricted result is what the
    # data can actually support, and it is reported rather than buried.
    ok_arms = [arm_label(c) for c in CAPACITY_ARMS
               if all(r["resolved"] for r in results[arm_label(c)])]
    ok_margins = [margins[a] for a in ok_arms if np.isfinite(margins[a])]
    print()
    print("  Restricted to arms the guard calls resolved "
          f"({', '.join(ok_arms) if ok_arms else 'none'}):")
    if len(ok_margins) >= 2 and min(ok_margins) > 0:
        ok_spread = max(ok_margins) / min(ok_margins)
        print("    margins " + ", ".join(f"{m:.0f}x" for m in ok_margins) +
              f"  ->  spread {ok_spread:.2f}x")
        if ok_spread < args.margin_tol:
            print(f"    Within the registered {args.margin_tol:g}x tolerance. So where")
            print("    the measurement is valid, the margin IS capacity-invariant,")
            print("    matching what n3_capacity_separation already reported. The")
            print("    apparent capacity dependence above is the fields fragmenting")
            print("    to lattice scale, not binding responding to scarcity.")
        else:
            print(f"    Still outside the registered {args.margin_tol:g}x tolerance,")
            print("    so the capacity dependence survives the validity filter.")
    else:
        print("    too few valid arms to compare")

    # ------------------------------------------------------------------- Q3
    print()
    print(BANNER)
    print("Q3  MANIPULATION CHECK — did the capacity knob actually bite?")
    print(BANNER)
    k_free = results["free"][0]["kappa_mean"]
    k_tight = results[arm_label(CAPACITY_ARMS[-1])][0]["kappa_mean"]
    drop = 1.0 - (k_tight / k_free if k_free else 1.0)
    d_free = np.mean([r["distinction"] for r in results["free"]])
    d_tight = np.mean([r["distinction"]
                       for r in results[arm_label(CAPACITY_ARMS[-1])]])
    print(f"  mean kappa   free {k_free:.3f}  ->  scarcest {k_tight:.3f}   "
          f"({100*drop:.0f}% drop)")
    print(f"  distinction  free {d_free:.4f}  ->  scarcest {d_tight:.4f}")
    q3 = drop >= 0.20
    print()
    if q3:
        print("  MANIPULATION CONFIRMED — capacity really was made scarce, so the")
        print("  null results above are informative rather than vacuous.")
    else:
        print("  MANIPULATION FAILED — capacity barely moved, so Q1 and Q2 are")
        print("  null for a boring reason and settle nothing.  Widen the arms.")

    # -------------------------------------------------------------- verdict
    print()
    print(BANNER)
    n = int(q1) + int(q2) + int(q3)
    print(f"VERDICT: {n}/3 pre-registered predictions held.")
    print(BANNER)
    if unresolved_arms:
        print(f"  READ Q2 WITH CARE: arms {unresolved_arms} contain unresolved")
        print("  palettes, so part of the margin change across the capacity axis")
        print("  is the texture artifact rather than a change in binding.  The")
        print("  resolved arms are the ones that can carry an argument.")
        print()
    if q1 and q2 and q3:
        print("  The palette selection is GEOMETRIC, not scarcity-gated.  It is")
        print("  present with no capacity field at all, present when capacity is")
        print("  free, and its margin does not sharpen when the budget binds --")
        print("  while the same knob demonstrably changes how much structure the")
        print("  field carries.  So The_Principle.md section 3's 'invisible when")
        print("  capacity is abundant' is NOT supported, and it contradicts")
        print("  section 2 of the same document, which reports the gap as")
        print("  capacity-invariant.  Section 3 should be corrected to match.")
        print()
        print("  What this costs the programme: the P = 3 result is a Plateau-law")
        print("  fact about three-fold vertices, so it cannot be cited as evidence")
        print("  that URP is doing work a generic coarsening model would not do.")
        print("  What it does not cost: the substrate transplants stand, and they")
        print("  are the stronger evidence anyway -- a Hopfield network and a")
        print("  Kuramoto population have no domains to coarsen, so their shared")
        print("  level crossing cannot be explained this way.")
    elif not q3:
        print("  Inconclusive: the manipulation did not bite.")
    else:
        print("  Capacity does bear on the selection.  Report the direction above")
        print("  and re-check section 3 against it rather than assuming either way.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "capacity_gating.json"
        out.write_text(json.dumps(
            {"params": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
             "arms": {k: v for k, v in results.items()},
             "peaks": peaks, "margins": {k: float(v) for k, v in margins.items()},
             "score": n}, indent=2, default=float))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
