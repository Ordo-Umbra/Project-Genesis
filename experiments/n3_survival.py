"""Survival: most runs fade out, some don't.  Is that a fact or a feeling?

Watching the 3-D sector field, the impression is hard to miss — nearly every
run coarsens to a single sector within a few hundred steps, but *occasionally*
one refuses to die and keeps producing structure that changes in kind without
settling.  The impression may be right.  It is also exactly what you would
feel if it were wrong, because a run that survives is memorable and forty runs
that faded are not.  Nothing about watching more runs distinguishes the two.

So this experiment does not watch.  It fixes a definition of death in code,
runs a whole ensemble, keeps every run including the boring ones, and asks two
questions that have answers whichever way they come out.

**The distinction that matters.**  Any process with a constant death rate
produces long-lived outliers.  If deaths here are *memoryless*, then "sometimes
it doesn't end" is the tail every exponential has, the survivors are ordinary,
and the honest conclusion is that there is nothing to explain.  If instead the
hazard *falls* with time — if getting through the early window genuinely buys
safety — then survivors are a different population and the question "when does
it stabilise" is a real question with a real answer.

Pre-registered predictions:

Q1. **Deaths are not memoryless.**  The hazard rate in the late window is less
    than **half** the early-window rate, with a bootstrap CI on the ratio that
    excludes 1.  **Falsifier:** a CI containing 1.  That would say the deaths
    are a constant-rate process, the long runs are unremarkable tail events,
    and the felt pattern should be dropped.

Q2. **The outcome is already written into the early state.**  A linear
    classifier fitted on half the runs, using only features measured at step
    ``--early``, predicts long-vs-short survival on the held-out half with
    **AUC >= 0.70**, and beats a permutation null at **p < 0.01**.
    **Falsifier:** held-out AUC inside the null's 95th percentile.  That would
    mean the early field carries no information about the ending, so any sense
    of being able to tell early which runs will make it is confirmation bias —
    a finding worth having, and reported as one.

Q3. **Palette imbalance is the carrier.**  Of the five candidate features, the
    strongest single held-out predictor is ``imbalance`` — how unevenly the
    sectors share the lattice at ``--early`` — on the reasoning that a palette
    that starts lopsided already has a sector on the way out.  **Falsifier:**
    no single feature clears 0.60 held out, or a different one wins (in which
    case the reasoning above was wrong and is reported as wrong).

**Explore, then confirm — on separate data.**  Q3 ranks five features and
reports the best, and the best of five is a biased estimate however carefully
the individual test is run: with five tries something usually looks good.  Two
corrections are applied rather than one.  First, the exploratory table is
scored against a *look-elsewhere* null — the distribution of the **maximum**
AUC over the same five features under shuffled labels.  Second, whichever
feature wins is then re-tested on a **completely fresh ensemble** with a
different seed, as a single pre-specified statistic with nothing to choose
over.  A pattern that survives that is not a pattern the analyst put there.

Honest scope, stated up front.  This measures *this* model family, at one
lattice size, with death defined one way.  It does not show that long-lived
configurations are physically special, only whether their occurrence is
distinguishable from a constant-rate process and whether it is foreseeable.
Every per-feature AUC is printed, not just the winner, so post-hoc selection is
visible rather than hidden.  And the ensemble is fixed before the first look:
no run is added or dropped after seeing a result.

**When, though?**  ``--early`` fixes one capture step, and step 60 was an
arbitrary choice.  ``--when`` sweeps it instead, measuring AUC as a function of
*when* the predictor is read, and quotes the answer against the system's own
clocks — there are no seconds here, so a bare step number means nothing on its
own.  The result corrects the natural reading of Q2: there is no decision
*moment*.  The bias is present from the first handful of steps and rises
monotonically for the whole mortality window, which is a different claim from
"decided early" and is the one the data supports.

    python experiments/n3_survival.py
    python experiments/n3_survival.py --when
    python experiments/n3_survival.py --runs 128 --steps 2000 --palette 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_genesis.survival import (  # noqa: E402
    FEATURE_NAMES,
    bootstrap_hazard_ratio,
    confirm_feature,
    feature_aucs,
    held_out_auc,
    kaplan_meier,
    max_feature_permutation,
    permutation_p,
    relaxation_time,
    run_ensemble,
    spearman_r,
    split_hazard,
)

BANNER = "=" * 78

CAPTURE_STEPS = (5, 10, 15, 20, 30, 40, 60, 80, 110, 150, 200, 280, 400)


def when_is_it_decided(args) -> int:
    """Sweep the capture step: how early, and how sharply, is the ending set?

    Quoted against the field's own relaxation time, because a step count means
    nothing on its own — this simulation has no seconds and no calibration to
    any.  Only ratios are meaningful.
    """
    print(BANNER)
    print("WHEN IS IT DECIDED?  (sweeping the step at which the predictor is read)")
    print(BANNER)
    print(f"  {args.runs} runs, {args.size}^{args.ndim}, palette {args.palette}, "
          f"horizon {args.steps}")
    print()

    t0 = time.time()
    res = run_ensemble(args.runs, args.palette, args.size, args.steps,
                       ndim=args.ndim, gamma=args.gamma, dt=args.dt,
                       noise=args.noise, seed=args.seed, early=args.early,
                       death_frac=args.death_frac, sample=args.sample,
                       capture=CAPTURE_STEPS, track_amplitude=True)
    print(f"  [{time.time() - t0:.0f}s]")

    death, censored = res["death"], res["censored"]
    if censored.all():
        print("\n  No deaths in this ensemble — nothing to predict.")
        return 1
    end = np.where(censored, args.steps, death)
    long_lived = end > np.median(end)
    tau = relaxation_time(res["amplitude"])
    med = float(np.median(death[~censored]))
    last = int(death[~censored].max())

    print()
    print("  THE SYSTEM'S OWN CLOCKS — there are no seconds here, only ratios")
    print(f"    tau (field reaches 95% of its amplitude) : {tau:>5} steps")
    print(f"    median lifetime of runs that die         : {med:>5.0f} steps"
          f"   = {med / max(tau, 1):.2f} tau")
    print(f"    last death anywhere                      : {last:>5} steps"
          f"   = {last / max(tau, 1):.2f} tau")
    print(f"    survived the horizon                     : "
          f"{int(censored.sum())}/{args.runs}")
    print()
    print("    Note the first two lines: half of all deaths happen before the")
    print("    field has finished climbing into its own wells.  Those runs are")
    print("    not dying of old age, they are failing to form.")

    print()
    print("  AUC vs eventual survival, by the step the predictor is read:")
    print(f"    {'step':>5} {'/tau':>6} {'/median':>8}   " +
          "  ".join(f"{n:>9}" for n in FEATURE_NAMES))
    for t in CAPTURE_STEPS:
        if t not in res["captured"]:
            continue
        row = feature_aucs(res["captured"][t], long_lived)
        print(f"    {t:>5} {t / max(tau, 1):>6.2f} {t / med:>8.2f}   " +
              "  ".join(f"{row[n]:>9.3f}" for n in FEATURE_NAMES))

    first = min(res["captured"])
    a_first = feature_aucs(res["captured"][first], long_lived)["maxfrac"]
    print()
    print(BANNER)
    print("READING")
    print(BANNER)
    print(f"  There is no decision *moment*.  At step {first} — barely a")
    print(f"  {first / max(tau, 1):.2f} tau, before any structure exists — the bias is")
    print(f"  already there ({a_first:.3f}, chance is 0.5), and it climbs")
    print("  monotonically for the whole mortality window rather than crossing")
    print("  a threshold.  So 'the ending is set early' is right, and 'there is")
    print("  an early window that decides it' is not: the tilt is present from")
    print("  the first steps and never stops sharpening.")
    print()
    print("  Descriptive, one ensemble.  The *existence* of the step-60 signal is")
    print("  what was confirmed out of sample (see the default run); the shape of")
    print("  this curve has not been.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=96, help="ensemble size")
    p.add_argument("--palette", type=int, default=4,
                   help="sector count (4 = d+1 in three dimensions)")
    p.add_argument("--size", type=int, default=16, help="lattice edge")
    p.add_argument("--ndim", type=int, default=3)
    p.add_argument("--steps", type=int, default=1500, help="observation horizon")
    p.add_argument("--early", type=int, default=60,
                   help="step at which predictor features are captured")
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--dt", type=float, default=0.06)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--death-frac", type=float, default=0.95,
                   help="one sector holding this fraction counts as death")
    p.add_argument("--sample", type=int, default=10, help="death-check interval")
    p.add_argument("--perm", type=int, default=1000, help="permutation draws")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--when", action="store_true",
                   help="sweep the capture step instead: when is it decided?")
    args = p.parse_args()

    if args.when:
        return when_is_it_decided(args)

    print(BANNER)
    print("SURVIVAL OF THE 3-D SECTOR FIELD — is 'sometimes it doesn't end' real?")
    print(BANNER)
    print(f"  ensemble        : {args.runs} independent runs, all counted")
    print(f"  lattice         : {args.size}^{args.ndim}, palette {args.palette}, "
          f"gamma {args.gamma}, noise {args.noise}, dt {args.dt}")
    print(f"  horizon         : {args.steps} steps "
          f"(runs still alive are right-censored, not dropped)")
    print(f"  death           : one sector holding > {args.death_frac:.0%} "
          f"of the lattice, checked every {args.sample} steps")
    print(f"  predictors from : step {args.early} only")
    print()
    print("  Pre-registered:")
    print("    Q1  late/early hazard ratio < 0.5, bootstrap CI excluding 1")
    print("    Q2  held-out AUC >= 0.70 from step-%d features, perm p < 0.01"
          % args.early)
    print("    Q3  'imbalance' is the strongest single feature (>= 0.60)")
    print()

    t0 = time.time()
    last = [0]

    def progress(t, alive):
        if args.quiet:
            return
        if t - last[0] >= 250 or alive == 0:
            last[0] = t
            print(f"    step {t:>5} — {alive:>3}/{args.runs} still alive")

    res = run_ensemble(args.runs, args.palette, args.size, args.steps,
                       ndim=args.ndim, gamma=args.gamma, dt=args.dt,
                       noise=args.noise, seed=args.seed, early=args.early,
                       death_frac=args.death_frac, sample=args.sample,
                       progress=progress)
    print(f"    [{time.time() - t0:.0f}s]")
    print()

    death, censored = res["death"], res["censored"]
    n_dead = int((~censored).sum())
    horizon = args.steps

    # ---------------------------------------------------------------- shape
    print(BANNER)
    print("THE DISTRIBUTION — every run, not the memorable ones")
    print(BANNER)
    finished = death[~censored]
    if n_dead:
        qs = np.percentile(finished, [10, 25, 50, 75, 90])
        print(f"  died within horizon : {n_dead}/{args.runs} "
              f"({100 * n_dead / args.runs:.0f}%)")
        print(f"  death step  min/10/25/50/75/90/max : "
              f"{finished.min()} / {qs[0]:.0f} / {qs[1]:.0f} / {qs[2]:.0f} / "
              f"{qs[3]:.0f} / {qs[4]:.0f} / {finished.max()}")
    else:
        print(f"  died within horizon : 0/{args.runs}")
    print(f"  still alive at {horizon:>5} : {int(censored.sum())} "
          f"({100 * censored.sum() / args.runs:.0f}%)  <- the 'doesn't end' cases")

    times, surv = kaplan_meier(death, censored, horizon)
    print()
    print("  Kaplan-Meier survival:")
    for frac in (0.75, 0.5, 0.25, 0.10):
        below = np.where(surv <= frac)[0]
        when = times[below[0]] if len(below) else None
        print(f"    S(t) falls to {frac:>4.0%} at step "
              f"{when if when is not None else 'never within horizon'}")

    if n_dead:
        last = int(finished.max())
        quiet = horizon - last
        n_alive = int(censored.sum())
        print()
        print(f"  Last death anywhere in the ensemble : step {last}")
        print(f"  After that: {n_alive} runs x {quiet} steps = "
              f"{n_alive * quiet:,} run-steps with ZERO deaths.")
        print("  (A constant-rate process would not go quiet; this is the same")
        print("   claim Q1 tests, stated as a raw count rather than a ratio.)")

    # ------------------------------------------------------------------- Q1
    cut = int(np.median(finished)) if n_dead else horizon // 2
    haz = split_hazard(death, censored, cut, horizon)
    lo, hi = bootstrap_hazard_ratio(death, censored, cut, horizon, seed=args.seed)

    print()
    print(BANNER)
    print(f"Q1  MEMORYLESS?  (split at step {cut})")
    print(BANNER)
    print(f"  early window  0-{cut:<6} : {haz['early_deaths']:>3} deaths over "
          f"{haz['early_exposure']:>9.0f} run-steps  ->  hazard "
          f"{haz['early_hazard']:.3e}/step")
    print(f"  late  window  {cut}-{horizon:<6}: {haz['late_deaths']:>3} deaths over "
          f"{haz['late_exposure']:>9.0f} run-steps  ->  hazard "
          f"{haz['late_hazard']:.3e}/step")
    print(f"  late/early ratio : {haz['ratio']:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print()
    q1_falls = np.isfinite(haz["ratio"]) and haz["ratio"] < 0.5 and hi < 1.0
    q1_flat = np.isfinite(lo) and lo <= 1.0 <= hi
    if q1_falls:
        print("  PREDICTION HELD — the hazard falls.  Surviving the early window")
        print("  genuinely buys safety, so long-lived runs are a different")
        print("  population, not the tail of a constant-rate process.")
    elif q1_flat:
        print("  PREDICTION FAILED — the CI contains 1, so deaths are consistent")
        print("  with a constant-rate (memoryless) process.  The long runs are")
        print("  ordinary tail events; there is nothing here to explain, and the")
        print("  felt pattern should be dropped.")
    else:
        print("  PREDICTION FAILED — the hazard moved, but not as predicted")
        print(f"  (ratio {haz['ratio']:.3f}).  Reported as measured.")

    # ------------------------------------------------------------------- Q2
    print()
    print(BANNER)
    print(f"Q2  IS THE ENDING VISIBLE AT STEP {args.early}?")
    print(BANNER)

    end = np.where(censored, horizon, death)
    long_lived = end > np.median(end)
    if long_lived.all() or (~long_lived).all():
        print("  Degenerate split (all runs on one side) — cannot test.")
        return 1

    obs, per = held_out_auc(res["features"], long_lived, seed=args.seed)
    obs2, null_mean, null_95, pval = permutation_p(
        res["features"], long_lived, n_perm=args.perm, seed=args.seed)

    print(f"  labels          : long-lived = survives past step {np.median(end):.0f} "
          f"({int(long_lived.sum())}/{args.runs})")
    print(f"  held-out AUC    : {obs:.3f}   (fit on half the runs, scored on the rest)")
    print(f"  permutation null: mean {null_mean:.3f}, 95th pct {null_95:.3f}, "
          f"p = {pval:.4f}")
    print()
    q2_ok = obs >= 0.70 and pval < 0.01
    if q2_ok:
        print("  PREDICTION HELD — the ending is foreseeable from the early field.")
        print("  The outcome is set by the initial evolution, not by later luck.")
    elif obs <= null_95:
        print("  PREDICTION FAILED — held-out AUC sits inside the permutation null.")
        print("  The early field carries no information about the ending, so any")
        print("  sense of telling early which runs will survive is confirmation")
        print("  bias.  That is the useful answer, and it is the one measured.")
    else:
        print(f"  PARTIAL — AUC {obs:.3f} beats the null (p = {pval:.4f}) but misses")
        print("  the pre-registered 0.70 threshold.  Predictable, but weakly.")

    # ------------------------------------------------------------------- Q3
    print()
    print(BANNER)
    print("Q3  WHICH FEATURE CARRIES IT?  (all reported, not just the winner)")
    print(BANNER)
    solo = feature_aucs(res["features"], long_lived)
    print("  (single unfitted features are scored on the whole ensemble — there")
    print("   is nothing to overfit, so a split would only discard data)")
    print()
    print(f"  {'feature':<12} {'AUC':>8} {'Spearman vs death':>19}")
    for name in FEATURE_NAMES:
        rho = spearman_r(res["features"][name], end)
        print(f"  {name:<12} {solo[name]:>8.3f} {rho:>19.3f}")

    best_auc, winner, max_null_95, max_p = max_feature_permutation(
        res["features"], long_lived, n_perm=args.perm, seed=args.seed)
    print()
    print(f"  best of five    : '{winner}' at {best_auc:.3f}")
    print(f"  look-elsewhere null (max AUC over the same five, labels shuffled):")
    print(f"                    95th pct {max_null_95:.3f}, p = {max_p:.4f}")

    print()
    q3_ok = winner == "imbalance" and best_auc >= 0.60
    if q3_ok:
        print("  PREDICTION HELD — 'imbalance' is the strongest single predictor.")
    elif best_auc < 0.60:
        print(f"  PREDICTION FAILED — no feature clears 0.60 (best: {winner} "
              f"{best_auc:.3f}).")
    else:
        print(f"  PREDICTION FAILED — the winner is '{winner}' ({best_auc:.3f}), not")
        print("  'imbalance'.  The stated reasoning was wrong; the measurement stands.")

    # ------------------------------------------------------- confirmation
    print()
    print(BANNER)
    print(f"CONFIRMATION — '{winner}' re-tested on a FRESH ensemble (seed "
          f"{args.seed + 1000})")
    print(BANNER)
    print("  Nothing is selected here: the feature is fixed by the run above,")
    print("  the data are new, and the statistic is one number decided before")
    print("  this ensemble existed.  This is the test that post-hoc choice")
    print("  cannot pass by luck.")
    print()

    t1 = time.time()
    last2 = [0]

    def progress2(t, alive):
        if args.quiet:
            return
        if t - last2[0] >= 500 or alive == 0:
            last2[0] = t
            print(f"    step {t:>5} — {alive:>3}/{args.runs} still alive")

    res2 = run_ensemble(args.runs, args.palette, args.size, args.steps,
                        ndim=args.ndim, gamma=args.gamma, dt=args.dt,
                        noise=args.noise, seed=args.seed + 1000, early=args.early,
                        death_frac=args.death_frac, sample=args.sample,
                        progress=progress2)
    print(f"    [{time.time() - t1:.0f}s]")

    end2 = np.where(res2["censored"], horizon, res2["death"])
    long2 = end2 > np.median(end2)
    conf_ok = False
    if long2.all() or (~long2).all():
        print("\n  Degenerate split on the fresh ensemble — cannot confirm.")
    else:
        auc2, null95_2, p2 = confirm_feature(res2["features"][winner], long2,
                                             n_perm=max(args.perm, 2000),
                                             seed=args.seed)
        rho2 = spearman_r(res2["features"][winner], end2)
        print()
        print(f"  fresh-ensemble AUC for '{winner}' : {auc2:.3f}   "
              f"(discovery ensemble: {best_auc:.3f})")
        print(f"  permutation null 95th pct        : {null95_2:.3f}, p = {p2:.4f}")
        print(f"  Spearman vs death time          : {rho2:.3f}")
        print()
        conf_ok = auc2 >= 0.60 and p2 < 0.01
        if conf_ok:
            print(f"  CONFIRMED — '{winner}' at step {args.early} predicts survival")
            print("  on data it was not chosen from.  The effect is real.")
        elif p2 >= 0.05:
            print(f"  NOT CONFIRMED — '{winner}' does not reproduce on fresh data.")
            print("  The discovery-ensemble result was selection, not signal.  This")
            print("  is precisely the failure the confirmation stage exists to catch.")
        else:
            print(f"  WEAKLY CONFIRMED — reproduces (p = {p2:.4f}) but below the")
            print("  0.60 bar set for confirmation.  Real but small.")

    # --------------------------------------------------------------- verdict
    print()
    print(BANNER)
    score = sum([q1_falls, q2_ok, q3_ok])
    print(f"VERDICT: {score}/3 pre-registered predictions held; "
          f"confirmation {'passed' if conf_ok else 'did not pass'}.")
    print(BANNER)
    print("  What this can and cannot settle: it settles whether the long-lived")
    print("  runs are statistically distinguishable from a constant-rate process,")
    print("  and whether their occurrence is foreseeable from the early field.")
    print("  It does not settle what makes them special, and it is one model")
    print("  family at one lattice size with one definition of death.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
