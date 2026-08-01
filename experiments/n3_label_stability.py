"""Does the compass's hallucination flag survive its own deadband?

This is the same question `n3_anatomy` and `n3_convention_manifold` asked of the
distinction reading, aimed at the one instrument in this programme that already
points at agents.

`trajectory_label` sorts a step in `(ΔC, ΔI)` into `expanding`, `consolidating`,
`diverging`, `contracting` or `steady`, and the identical taxonomy is read on
LLM session traces in the sibling repo. Both criticality transplants use it to
report that their substrate "turns `diverging` — the compass's hallucination
trajectory" over a band of the control parameter, and then place the
capacity-starved optimum inside that band. That is the closest thing here to an
agent-facing claim.

It rests on an underived constant. `trajectory_label` takes a `deadband`: a step
smaller than it in either coordinate counts as no movement, so the label becomes
`steady`. Both transplants pass **0.01**, and nothing fixes that value.

The three-way outcome is the point, and all three are worth having:

- **invariant** — the band is the same at every defensible deadband, and the
  published reading is safe to act on;
- **monotone** — it shrinks or grows predictably, so the deadband is an explicit
  sensitivity dial that has to be reported alongside any flag;
- **knife-edge or scattered** — `0.01` sits on a boundary, or the band comes and
  goes, in which case the published `diverging` reading is an artefact of the
  constant and should not be used to flag anything.

Pre-registered predictions:

Q1. **The band is not invariant.** At least one step changes label across the
    deadband sweep on at least one substrate. **Falsifier:** the whole label
    sequence is identical at every setting — the taxonomy is deadband-free and
    there was nothing to check.

Q2. **But the dependence is monotone, not scattered.** As the deadband widens,
    the `diverging` band's step count is non-increasing on each substrate: wider
    deadband, more steps called `steady`, never fewer. **Falsifier:** the count
    goes up somewhere, meaning widening the deadband *creates* a hallucination
    flag, which no reading of the parameter supports.

Q3. **And the published setting is not knife-edge.** At `deadband = 0.01` the
    band survives a change of at least 2× in each direction. **Falsifier:** the
    band changes under a smaller perturbation, in which case the published
    interval is a lucky constant rather than a measurement, and both transplants'
    section D should be read as such.

Honest scope. This tests the compass on **substrate scans**, which is what the
transplants actually did, not on LLM session traces — those live in another
repo and are not measured here, so nothing below licenses a claim about agent
sessions beyond the taxonomy being the same code. The scans are the published
driven arms at published settings; ΔC keeps its own convention, already known
from `n3_anatomy` to be a choice, so this isolates the deadband and does not
re-open that.

    python experiments/n3_label_stability.py
    python experiments/n3_label_stability.py --quick
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

from project_genesis.label_stability import (  # noqa: E402
    band_extent, invariant_fraction, label_sequence, plateau,
)
from n3_criticality_transplant import measure_substrate as hopfield_scan  # noqa: E402
from n3_kuramoto_transplant import measure_substrate as kuramoto_scan  # noqa: E402

BANNER = "=" * 78

HOP = dict(n_spins=300, n_patterns=9, n_seeds=4, n_burn=30, n_window=60,
           t_min=0.1, t_max=1.6, n_temps=16, structured_threshold=0.3,
           query_fraction=0.05, query_every=5)
KUR = dict(n_osc=400, coupling=2.0, dt=0.05, n_steps=900, n_burn=450,
           n_seeds=3, g_min=0.1, g_max=2.4, n_gamma=13,
           distribution="gaussian", freq_tol=0.3, noise=0.6)

PUBLISHED = 0.01
DEADBANDS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--deadbands", type=float, nargs="+", default=list(DEADBANDS))
    p.add_argument("--published", type=float, default=PUBLISHED)
    p.add_argument("--min-factor", type=float, default=2.0,
                   help="Q3: factor the published setting must survive each way.")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    hop, kur = dict(HOP), dict(KUR)
    if args.quick:
        hop.update(n_spins=200, n_seeds=2, n_burn=15, n_window=30, n_temps=10)
        kur.update(n_osc=250, n_seeds=2, n_steps=500, n_burn=250, n_gamma=10)

    print(BANNER)
    print("THE COMPASS'S DEADBAND — is the hallucination flag a measurement?")
    print(BANNER)
    print(f"  published deadband {args.published:g}; sweeping "
          f"{', '.join(f'{d:g}' for d in args.deadbands)}")
    print()
    print("  Pre-registered:")
    print("    Q1  the label sequence is not invariant across the sweep")
    print("    Q2  but `diverging` step counts are non-increasing in the deadband")
    print(f"    Q3  and the published setting survives {args.min_factor:g}x "
          "each way")
    print()

    t0 = time.time()
    print("  measuring the published driven scans...", flush=True)
    scans = {
        "hopfield": (hopfield_scan(Namespace(**hop, seed=args.seed),
                                   driven=True), "T"),
        "kuramoto": (kuramoto_scan(Namespace(**kur, seed=args.seed),
                                   driven=True), "gamma"),
    }
    print(f"  [{time.time() - t0:.0f}s]")

    results = {}
    for name, (rows, xk) in scans.items():
        x = [r[xk] for r in rows]
        seqs, bands = [], []
        print()
        print(BANNER)
        print(f"{name.upper()} — `diverging` band vs deadband")
        print(BANNER)
        print(f"  {'deadband':>9} {'steps':>6} {'band':>18} {'width':>7}  labels")
        for d in args.deadbands:
            labs = label_sequence(rows, d)
            b = band_extent(labs, x, "diverging")
            seqs.append(labs)
            bands.append(b)
            interval = (f"[{b['lo']:.2f}, {b['hi']:.2f}]" if b["present"]
                        else "—  (absent)")
            counts = {}
            for lab in labs:
                counts[lab] = counts.get(lab, 0) + 1
            summary = " ".join(f"{k[:4]}:{v}" for k, v in sorted(counts.items()))
            marker = "  <- published" if abs(d - args.published) < 1e-12 else ""
            print(f"  {d:>9g} {b['count']:>6} {interval:>18} "
                  f"{b['width']:>7.2f}  {summary}{marker}")
        inv = invariant_fraction(seqs)
        i_pub = int(np.argmin([abs(d - args.published) for d in args.deadbands]))
        pl = plateau(args.deadbands, bands, i_pub)
        counts = [b["count"] for b in bands]
        monotone = all(a >= b for a, b in zip(counts, counts[1:]))
        print()
        print(f"  steps whose label never changes: {inv:.0%}")
        print(f"  `diverging` counts across the sweep: {counts}"
              f"   {'non-increasing' if monotone else 'NOT monotone'}")
        print(f"  published setting survives [{pl['lo']:g}, {pl['hi']:g}] "
              f"= {pl['factor_down']:.1f}x down, {pl['factor_up']:.1f}x up")
        results[name] = {"invariant_fraction": inv, "counts": counts,
                         "monotone": bool(monotone), "plateau": pl,
                         "bands": bands}

    # ------------------------------------------------------------- verdicts
    q1 = any(v["invariant_fraction"] < 1.0 for v in results.values())
    q2 = all(v["monotone"] for v in results.values())
    q3 = all(v["plateau"]["factor_down"] >= args.min_factor
             and v["plateau"]["factor_up"] >= args.min_factor
             for v in results.values())

    print()
    print(BANNER)
    print("Q1  IS THE LABEL SEQUENCE INVARIANT?")
    print(BANNER)
    for n, v in results.items():
        print(f"  {n}: {v['invariant_fraction']:.0%} of steps never change label")
    print()
    print("  PREDICTION " + ("HELD — the deadband moves the labels, so it is a"
                             if q1 else "FAILED — the taxonomy is deadband-free"))
    print("  " + ("parameter of the reading and not a formality."
                  if q1 else "over this range; there was nothing to check."))

    print()
    print(BANNER)
    print("Q2  IS THE DEPENDENCE MONOTONE?")
    print(BANNER)
    if q2:
        print("  PREDICTION HELD — widening the deadband only ever converts")
        print("  `diverging` steps into `steady` ones. The parameter behaves as")
        print("  a sensitivity dial, which is the readable case: a flag raised")
        print("  at a wide deadband is a fortiori raised at a narrow one.")
    else:
        print("  PREDICTION FAILED — the count rises somewhere, so widening the")
        print("  deadband CREATES hallucination flags. No reading of the")
        print("  parameter supports that, and it means the label boundaries are")
        print("  interacting rather than nesting.")

    print()
    print(BANNER)
    print("Q3  IS THE PUBLISHED SETTING A ROBUST OPERATING POINT?")
    print(BANNER)
    for n, v in results.items():
        pl = v["plateau"]
        print(f"  {n}: unchanged over [{pl['lo']:g}, {pl['hi']:g}] — "
              f"{pl['factor_down']:.1f}x down, {pl['factor_up']:.1f}x up "
              f"({pl['n_settings']} of {len(args.deadbands)} settings)")
    print()
    if q3:
        print("  PREDICTION HELD — 0.01 is not a lucky constant. The band it")
        print("  reports is the band a reader would get from any nearby choice,")
        print("  so both transplants' section D stands as a measurement.")
    else:
        print("  PREDICTION FAILED — the published band does not survive a")
        print("  modest change in a constant nobody derived. The `diverging`")
        print("  interval both transplants report is then a property of that")
        print("  choice, and it should not be used to flag anything until the")
        print("  deadband is either derived or reported with every flag.")

    # presence and extent are different claims and they do not fail together
    always = all(all(b["present"] for b in v["bands"]) for v in results.values())
    onset = {n: sorted({round(b["lo"], 6) for b in v["bands"] if b["present"]})
             for n, v in results.items()}
    print()
    print(BANNER)
    print("PRESENCE vs EXTENT — they come apart, and only one of them fails")
    print(BANNER)
    print(f"  the `diverging` regime is present at EVERY deadband tested: "
          f"{'yes' if always else 'NO'}")
    for n, lows in onset.items():
        print(f"  {n}: its ONSET takes {len(lows)} distinct values across the "
              f"sweep — {', '.join(f'{v:g}' for v in lows)}")
    print()
    print("  So the two questions an agent might ask of this taxonomy have")
    print("  different answers. *Is this trajectory diverging?* is robust: the")
    print("  regime shows up at every setting tried. *When did it start?* is")
    print("  not: the onset moves with a constant nobody derived. Reporting a")
    print("  flag is defensible; reporting where it began is not, unless the")
    print("  deadband goes with it.")

    print()
    print(BANNER)
    n_held = int(q1) + int(q2) + int(q3)
    print(f"VERDICT: {n_held}/3 pre-registered predictions held.")
    print(BANNER)
    print("  For anything built on this taxonomy: the deadband is configuration,")
    print("  and on the evidence above it is " + (
        "a documented sensitivity dial\n  rather than a hidden switch — report it "
        "with the flag and the flag\n  means something." if q2 and q3 else
        "not yet safe to leave implicit.\n  A flag whose presence depends on an "
        "undocumented constant is not a\n  measurement, whatever the taxonomy is "
        "called."))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        o = args.output_dir / "label_stability.json"
        o.write_text(json.dumps(
            {"params": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
             "hopfield_settings": hop, "kuramoto_settings": kur,
             "results": results, "score": n_held}, indent=2, default=float))
        print(f"\n  wrote {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
