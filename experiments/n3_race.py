"""Is the early lead a cause, or only a marker? — and is it actually a race?

``n3_survival`` found that ``maxfrac``, the largest sector's share at step 60,
predicts survival with a held-out AUC of **0.801** on a fresh ensemble.  That
was confirmed out of sample, so the association is real.  It is still only an
association: the early lead and the eventual death could both be downstream of
something else in the draw, in which case ``maxfrac`` is a marker and the "race"
reading is a story told over a correlation.

The fix is the one the observer protocol already forced: stop observing the
lead and **set** it.  A fair random field, one sector's amplitude offset by a
measured amount, everything else untouched.  If survival falls with the imposed
lead, the lead is doing causal work.

**Disclosure — the lead grid was chosen after an exploratory scan.**  A
dose-response needs to be sampled where the response actually moves, and that
range was not known in advance; a coarse scan showed survival collapsing
somewhere between an imposed lead of 0 and 0.02.  So Q1 below is **confirmed on
a fresh seed**, not claimed from the scan that located it.  Q2 and Q3 were
registered before being run at all.

Pre-registered predictions:

Q1. **The lead is causal, and the dose-response is a cliff rather than a
    slope.**  On a fresh seed: survival exceeds **0.30** with no imposed lead,
    and falls below **0.05** once the realised largest-sector share reaches
    0.28.  **Falsifier:** survival flat in the imposed lead — which would mean
    ``maxfrac`` is a marker of something else and the race reading is wrong.

Q2. **It is a race, not a drift and not a diffusion.**  A race compounds: the
    leader's advantage grows in proportion to itself, so the gap between the
    top two shares grows **exponentially** and ``log(gap)`` is straight in
    time.  Prediction: over the early window the log-linear fit beats both the
    linear fit (steady drift, ``gap ∝ t``) and the square-root fit (diffusion,
    ``gap ∝ √t``) on ``R²``, with a positive growth rate.  **Falsifier:**
    either alternative fits better, in which case the mechanism is not a race
    however good the correlation is.

Q3. **More competitors make runaway harder.**  With no imposed lead, survival
    increases with palette size ``P``: a leader has to out-run more rivals, so
    the contested state is easier to hold.  **Falsifier:** survival flat or
    decreasing in ``P``.  This is the axis that connects to the selection
    sweep, where ``P = 4`` behaved differently in 3-D, so either direction is
    informative about that too.

Honest scope: one lattice size, one noise level, one death criterion, 3-D.  The
intervention tilts the initial amplitudes and then lets the field alone — it
does not hold the lead in place, so what is measured is whether a head start is
*self-amplifying*, which is exactly the race claim and nothing broader.

    python experiments/n3_race.py
    python experiments/n3_race.py --runs 96 --steps 900
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_genesis.race import growth_rate, run_race  # noqa: E402

BANNER = "=" * 78

LEADS = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=int, default=64)
    p.add_argument("--palette", type=int, default=4)
    p.add_argument("--palettes", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--ndim", type=int, default=3)
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--dt", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=4242,
                   help="fresh seed: the scan that located the range used seed 0")
    p.add_argument("--fit-lo", type=int, default=20)
    p.add_argument("--fit-hi", type=int, default=200)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    common = dict(size=args.size, ndim=args.ndim, gamma=args.gamma,
                  dt=args.dt, noise=args.noise, seed=args.seed)

    print(BANNER)
    print("THE RACE — is the early lead a cause, and does it compound?")
    print(BANNER)
    print(f"  {args.runs} runs/arm, {args.size}^{args.ndim}, palette {args.palette}, "
          f"{args.steps} steps, seed {args.seed}")
    print("  the lead is IMPOSED: one sector's amplitude is offset before")
    print("  normalising, so the field is still a fair configuration, only tilted")
    print()
    print("  Pre-registered:")
    print("    Q1  survival > 0.30 unbiased, < 0.05 once realised share hits 0.28")
    print("        (range located by an exploratory scan; confirmed here on a fresh seed)")
    print("    Q2  log(gap) fits better than gap~t or gap~sqrt(t): a race, not drift")
    print("    Q3  survival RISES with palette size at zero lead")
    print()

    # ------------------------------------------------------------------- Q1
    print(BANNER)
    print("Q1  DOSE-RESPONSE — does an imposed lead kill?")
    print(BANNER)
    t0 = time.time()
    rows = []
    print(f"  {'imposed':>8} {'realised share':>15} {'survival':>9} {'deaths':>8}")
    for lead in LEADS:
        r = run_race(args.runs, args.palette, steps=args.steps, lead=lead, **common)
        rows.append({"lead": lead, "measured": r["lead_measured"],
                     "survival": r["survival"],
                     "deaths": int((~r["censored"]).sum())})
        print(f"  {lead:>8.3f} {r['lead_measured']:>15.3f} {r['survival']:>9.2f} "
              f"{int((~r['censored']).sum()):>8}")
        if lead == 0.0:
            base = r
    print(f"  [{time.time() - t0:.0f}s]")

    unbiased = rows[0]["survival"]
    above = [r for r in rows if r["measured"] >= 0.28]
    q1 = unbiased > 0.30 and bool(above) and max(r["survival"] for r in above) < 0.05
    print()
    if q1:
        print("  PREDICTION HELD — and the shape is the point. Survival is a")
        print("  CLIFF in the imposed lead, not a slope: a shift of a couple of")
        print("  percentage points in the initial share takes it from a coin")
        print("  flip to certain death. So the lead is not a marker of something")
        print("  else -- imposing it is sufficient.")
        print()
        print("  Read that against the unbiased arm: it survives not because no")
        print("  sector leads, but because no sector leads SYSTEMATICALLY. The")
        print("  fluctuation lead keeps changing hands; a tilt does not.")
    else:
        print("  PREDICTION FAILED — reported as measured:")
        print(f"    unbiased survival {unbiased:.2f} (needed > 0.30)")
        if above:
            print(f"    best survival above 0.28 share: "
                  f"{max(r['survival'] for r in above):.2f} (needed < 0.05)")
        else:
            print("    no arm reached a realised share of 0.28 — range too narrow")

    # ------------------------------------------------------------------- Q2
    print()
    print(BANNER)
    print("Q2  IS IT A RACE?  (compounding) vs DRIFT (linear) vs DIFFUSION (sqrt)")
    print(BANNER)
    lam, r2_log, r2_lin, r2_sqrt = growth_rate(
        base["times"], base["gaps"], lo=args.fit_lo, hi=args.fit_hi)
    print(f"  fitted on the unbiased ensemble, steps {args.fit_lo}-{args.fit_hi}")
    print()
    print(f"  {'model':>24} {'form':>14} {'R^2':>8}")
    print(f"  {'race (compounding)':>24} {'log gap ~ t':>14} {r2_log:>8.4f}")
    print(f"  {'steady drift':>24} {'gap ~ t':>14} {r2_lin:>8.4f}")
    print(f"  {'diffusion':>24} {'gap ~ sqrt(t)':>14} {r2_sqrt:>8.4f}")
    print(f"  growth rate lambda = {lam:.5f} per step")
    q2 = (lam > 0) and (r2_log > r2_lin) and (r2_log > r2_sqrt)
    print()
    if q2:
        print("  PREDICTION HELD — the lead compounds. The gap grows in")
        print("  proportion to itself, which is what makes it a race rather than")
        print("  a drift that happens to correlate with the outcome.")
    else:
        best = max([("race", r2_log), ("drift", r2_lin), ("diffusion", r2_sqrt)],
                   key=lambda kv: kv[1])
        print(f"  PREDICTION FAILED — the best-fitting law is '{best[0]}' "
              f"(R^2 {best[1]:.4f}).")
        print("  The correlation with survival can be perfectly real and the")
        print("  race mechanism still wrong; that is what this question is for.")

    # ------------------------------------------------------------------- Q3
    print()
    print(BANNER)
    print("Q3  DOES A BIGGER FIELD OF COMPETITORS PROTECT?")
    print(BANNER)
    t1 = time.time()
    pal_rows = []
    print(f"  {'palette':>8} {'survival':>9} {'deaths':>8}   (no imposed lead)")
    for pp in args.palettes:
        r = run_race(args.runs, pp, steps=args.steps, lead=0.0, **common)
        pal_rows.append({"palette": pp, "survival": r["survival"]})
        print(f"  {pp:>8} {r['survival']:>9.2f} {int((~r['censored']).sum()):>8}")
    print(f"  [{time.time() - t1:.0f}s]")

    survs = [r["survival"] for r in pal_rows]
    rising = survs[-1] > survs[0] + 0.05
    monotone = all(b >= a - 0.05 for a, b in zip(survs, survs[1:]))
    q3 = rising and monotone
    print()
    if q3:
        print("  PREDICTION HELD — more competitors, more survival. A leader has")
        print("  to out-run more rivals, so the contested state is easier to")
        print("  hold and runaway is harder to complete.")
    elif survs[-1] < survs[0] - 0.05:
        print("  PREDICTION FAILED, and in the opposite direction — MORE")
        print("  competitors means LESS survival. Each sector starts with a")
        print("  smaller share, so the field is more fragile, not more robust.")
        print("  That inverts the reasoning and is the more interesting outcome:")
        print("  it would say a large palette is harder to keep, which bears")
        print("  directly on why P=4 weakened in the 3-D selection sweep.")
    else:
        print("  PREDICTION FAILED — survival is not monotone in palette size.")
        print("  Reported as measured; no clean reading either way.")

    # -------------------------------------------------------------- verdict
    print()
    print(BANNER)
    n_held = int(q1) + int(q2) + int(q3)
    print(f"VERDICT: {n_held}/3 pre-registered predictions held.")
    print(BANNER)
    if q1:
        print("  The step-60 AUC of 0.801 was a correlation. Imposing the lead")
        print("  turns it into a dose-response, so the early advantage is doing")
        print("  causal work rather than marking something upstream of it.")
        print("  What is still not settled: whether the SAME mechanism sets the")
        print("  hazard going to zero after 2 tau, or whether that is a separate")
        print("  fact about coarsening reaching the box.")
    else:
        print("  The causal claim did not reproduce on a fresh seed. Treat the")
        print("  AUC as a marker until it does.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "race.json"
        out.write_text(json.dumps(
            {"params": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
             "dose_response": rows,
             "growth": {"lambda": lam, "r2_log": r2_log,
                        "r2_linear": r2_lin, "r2_sqrt": r2_sqrt},
             "palette": pal_rows, "score": n_held}, indent=2, default=float))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
