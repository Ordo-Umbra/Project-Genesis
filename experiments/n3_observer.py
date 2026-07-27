"""Can attention change the outcome?  A protocol that could actually tell.

The claim is that whether — and how — the field is watched changes whether a
run persists.  It is worth separating three things that get bundled together
when this is judged by watching:

1. **A mechanical channel.**  In the original viewer there was one, and it was
   large: the physics advanced once per rendered frame, and browsers throttle
   rendering hard for a window that is not in front.  Measured, a foreground
   tab computed **4.7x** as many steps as a background tab over the same
   twenty seconds.  Attention really did change the simulation — via the
   window manager.  That channel is now closed (physics runs on a wall clock,
   rendering on frames; re-measured at 1.0x), and any test run on the old
   build measured this and nothing else.

2. **A sampling channel.**  The interesting phase is short and early: half of
   all runs are dead by step 420 and the last death in a 96-run ensemble was
   step 720.  Watch from the start and you see the alive phase; glance over
   later and you see the end state.  "It dies when I don't attend" is then
   guaranteed without anything acting on anything.

3. **Reverse causation, which is the subtle one.**  ``n3_survival`` measured
   that the outcome is *legible at step 60* — the largest sector's share
   predicts survival with AUC 0.80 on fresh data.  So a viewer genuinely can
   tell early which runs will make it.  The perception is real and now
   quantified.  But that makes the early state the common cause of both the
   outcome **and** how engaging the run is to watch, which produces exactly
   the felt correlation with the arrow pointing the other way.

None of that settles the claim.  It means an honest test has to hold all three
fixed, and that is what the protocol below does.  The design is not exotic;
it is a randomised controlled trial with a sham arm.

    python experiments/n3_observer.py                 # protocol + power
    python experiments/n3_observer.py --log trials.json   # analyse a real run
    python experiments/n3_observer.py --demo-null      # what a null looks like

THE PROTOCOL
------------

**Fixed step budget.**  Each trial runs exactly ``--budget`` steps, decided
before the trial starts.  Attention cannot change how much computation
happens, so the only channel left is the random stream itself — which is the
actual claim, stripped of confounds.

**Randomised assignment.**  Each trial is assigned ATTEND / IGNORE by a
generator seeded independently of the field.  You do not choose; you comply.
Choosing is how condition gets correlated with how promising a run looked.

**A sham arm.**  On some trials the display is a *replay of a previously
recorded run* — the pixels look live, but the outcome was fixed before the
trial began and nothing you do can touch it.  This is the control that
matters.  A real effect appears in live trials and not in sham ones.  An
effect that appears in both is in the observing, not in the field.

**Outcome scored by code, not by eye.**  Death is "one sector holds more than
95% of the lattice", computed and written to the log.  Nothing is judged.

**A stopping rule fixed in advance.**  Run all ``N`` trials and then analyse.
Watching the running total and stopping when it looks good is the single most
reliable way to manufacture a positive result, and it is what most informal
versions of this test do.

**Analysis fixed in advance.**  Two-proportion permutation test on survival
rate, ATTEND vs IGNORE, live trials only; the sham arm tested the same way as
a negative control.

WHAT THIS CAN CONCLUDE
----------------------

A null result here does not prove the claim false in general — it bounds the
effect in this system, at this power.  That bound is the useful thing, and it
is what the power table below is for: an informal run of twenty trials cannot
distinguish a large effect from nothing, so a pattern seen in twenty trials is
not evidence either way.  Knowing the number needed is what makes the question
answerable rather than unanswerable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BANNER = "=" * 78


# --------------------------------------------------------------------------

def power_n(p0: float, delta: float, alpha: float = 0.05,
            power: float = 0.80) -> int:
    """Trials **per arm** to detect a shift from ``p0`` to ``p0 + delta``.

    Standard two-proportion sample size.  No simulation needed and no room to
    argue with the answer.
    """
    p1 = p0 + delta
    if not (0.0 < p1 < 1.0):
        return -1
    z_a = 1.959963985 if abs(alpha - 0.05) < 1e-9 else _z(1 - alpha / 2)
    z_b = _z(power)
    pbar = 0.5 * (p0 + p1)
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar))
           + z_b * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return int(math.ceil(num / (delta ** 2)))


def _z(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, plenty accurate)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def two_prop_permutation(survived, condition, *, n_perm: int = 20000, seed: int = 0):
    """Permutation test on the survival-rate difference between two arms.

    ``condition`` is boolean: True = ATTEND.  Shuffling the labels is exactly
    the null "the label had nothing to do with the outcome", which is the
    hypothesis at issue, so no distributional assumption enters.
    """
    survived = np.asarray(survived, dtype=bool)
    condition = np.asarray(condition, dtype=bool)
    if condition.all() or (~condition).all():
        return {"n_attend": int(condition.sum()), "n_ignore": int((~condition).sum()),
                "p_attend": float("nan"), "p_ignore": float("nan"),
                "diff": float("nan"), "p_value": float("nan")}

    def diff(cond):
        return survived[cond].mean() - survived[~cond].mean()

    observed = diff(condition)
    rng = np.random.default_rng(seed)
    null = np.array([diff(rng.permutation(condition)) for _ in range(int(n_perm))])
    p = float((np.abs(null) >= abs(observed)).sum() + 1) / (int(n_perm) + 1)
    return {"n_attend": int(condition.sum()),
            "n_ignore": int((~condition).sum()),
            "p_attend": float(survived[condition].mean()),
            "p_ignore": float(survived[~condition].mean()),
            "diff": float(observed),
            "p_value": p}


def load_log(path: Path):
    """Read a trial log written by ``web_toy/observer_trial.html``.

    Expected: a list of objects with ``sham`` (bool), ``attend`` (bool) and
    ``survived`` (bool).  Extra fields are ignored.
    """
    data = json.loads(Path(path).read_text())
    trials = data["trials"] if isinstance(data, dict) else data
    out = []
    for t in trials:
        out.append({"sham": bool(t["sham"]),
                    "attend": bool(t["attend"]),
                    "survived": bool(t["survived"])})
    return out


# --------------------------------------------------------------------------

def report_arm(name: str, trials, *, seed: int, n_perm: int) -> dict:
    surv = [t["survived"] for t in trials]
    cond = [t["attend"] for t in trials]
    r = two_prop_permutation(surv, cond, n_perm=n_perm, seed=seed)
    print(f"  {name}")
    if math.isnan(r["diff"]):
        print(f"    only one arm present ({r['n_attend']} attend, "
              f"{r['n_ignore']} ignore) — no comparison possible")
        return r
    print(f"    ATTEND : {r['p_attend']:.3f} survived  (n = {r['n_attend']})")
    print(f"    IGNORE : {r['p_ignore']:.3f} survived  (n = {r['n_ignore']})")
    print(f"    difference {r['diff']:+.3f},  permutation p = {r['p_value']:.4f}")
    return r


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path, default=None,
                   help="trial log JSON from web_toy/observer_trial.html")
    p.add_argument("--demo-null", action="store_true",
                   help="run the analysis on synthetic no-effect data")
    p.add_argument("--baseline", type=float, default=0.40,
                   help="survival rate with no effect (measured: ~0.40)")
    p.add_argument("--perm", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(BANNER)
    print("OBSERVER PROTOCOL — what it would take to test this properly")
    print(BANNER)
    print("  Closed already, by measurement:")
    print("    rendering-rate channel  — foreground vs background tab computed")
    print("                              4.7x different step counts on the old")
    print("                              build; now 1.0x (physics on a wall clock)")
    print("    sampling channel        — fixed step budget per trial, so every")
    print("                              trial is scored at the same age")
    print("    reverse causation       — assignment is randomised, not chosen,")
    print("                              so 'promising' runs cannot attract")
    print("                              attention and then be credited to it")
    print("    scoring channel         — outcome computed in code, never judged")
    print()

    # ------------------------------------------------------------ power
    print(BANNER)
    print("HOW MANY TRIALS?  (this is the number that makes it answerable)")
    print(BANNER)
    b = args.baseline
    print(f"  baseline survival {b:.0%}, alpha 0.05, power 80%, two-sided")
    print()
    print(f"  {'if attention shifts survival by':>34} {'trials per arm':>15} "
          f"{'total':>8} {'at ~30 s each':>15}")
    for delta in (0.30, 0.20, 0.15, 0.10, 0.05):
        n = power_n(b, delta, power=0.80)
        if n < 0:
            continue
        hours = 2 * n * 30 / 3600
        print(f"  {'+' + format(delta, '.0%'):>34} {n:>15,} {2*n:>8,} "
              f"{hours:>13.1f} h")
    print()
    n20 = power_n(b, 0.30, power=0.80)
    print(f"  Read the last column before running anything.  Twenty informal")
    print(f"  trials cannot separate a huge effect from nothing: detecting even")
    print(f"  a +30 point shift needs {n20} per arm.  A pattern seen in twenty")
    print(f"  trials is not weak evidence — it is no evidence, in either")
    print(f"  direction, and that cuts against a sceptic just as hard.")

    # ------------------------------------------------------------ analysis
    trials = None
    if args.demo_null:
        print()
        print(BANNER)
        print("DEMONSTRATION — the analysis run on data with NO effect")
        print(BANNER)
        print("  Synthetic trials where survival is independent of condition, to")
        print("  show what the pipeline reports when there is nothing there.")
        print()
        rng = np.random.default_rng(args.seed)
        n = 2 * power_n(b, 0.15, power=0.80)
        trials = [{"sham": bool(i % 4 == 0),
                   "attend": bool(rng.random() < 0.5),
                   "survived": bool(rng.random() < b)}
                  for i in range(n)]
    elif args.log is not None:
        trials = load_log(args.log)
        print()
        print(BANNER)
        print(f"ANALYSIS — {args.log}")
        print(BANNER)

    if trials is None:
        print()
        print(BANNER)
        print("  No log supplied.  Generate one with web_toy/observer_trial.html,")
        print("  then:  python experiments/n3_observer.py --log trials.json")
        print("  Or see the machinery on null data:  --demo-null")
        print(BANNER)
        return 0

    live = [t for t in trials if not t["sham"]]
    sham = [t for t in trials if t["sham"]]
    print(f"  {len(trials)} trials  ({len(live)} live, {len(sham)} sham)")
    print()
    r_live = report_arm("LIVE  (the test)", live, seed=args.seed, n_perm=args.perm)
    print()
    r_sham = report_arm("SHAM  (negative control — replay, outcome fixed in advance)",
                        sham, seed=args.seed + 1, n_perm=args.perm)

    print()
    print(BANNER)
    print("VERDICT")
    print(BANNER)
    live_sig = (not math.isnan(r_live["p_value"])) and r_live["p_value"] < 0.05
    sham_sig = (not math.isnan(r_sham["p_value"])) and r_sham["p_value"] < 0.05

    if live_sig and not sham_sig:
        print("  Effect in the live arm, none in the sham arm.  That is the")
        print("  signature a real effect would have.  Next step is replication")
        print("  by someone else, on their hardware, with the log pre-committed.")
    elif live_sig and sham_sig:
        print("  Effect in BOTH arms.  The sham arm cannot be influenced — its")
        print("  outcomes were fixed before the trial — so this is coming from")
        print("  the observing or the bookkeeping, not from the field.")
    elif sham_sig:
        print("  Effect in the sham arm only.  That is a bookkeeping or")
        print("  compliance artefact; fix it before reading the live arm.")
    else:
        print("  No detectable difference in either arm.  With the trial count")
        print("  above this bounds the effect rather than excluding it — report")
        print("  the bound, which is the informative quantity.")
    if not math.isnan(r_live["p_value"]):
        n_live = r_live["n_attend"] + r_live["n_ignore"]
        det = next((d for d in (0.05, 0.10, 0.15, 0.20, 0.30)
                    if 2 * power_n(b, d, power=0.80) <= n_live), None)
        print()
        if det is None:
            need = 2 * power_n(b, 0.30, power=0.80)
            print(f"  Live-arm power: {n_live} trials CANNOT detect even a +30 point")
            print(f"  shift at 80% — that needs {need}.  Whatever the difference above")
            print("  looks like, this run does not distinguish it from chance.")
        else:
            print(f"  Live-arm power: {n_live} trials can detect a shift of "
                  f"+{100*det:.0f} points at 80%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
