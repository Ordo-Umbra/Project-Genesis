"""Does the oscillator transplant survive once the load stops counting the drive?

`n3_crossing_prediction` measured that the driven Kuramoto substrate's load
reading is almost entirely the perturbation being fed back as though it were the
response to the perturbation.  The injected phase noise contributes exactly
``σ√dt`` to the co-rotating spread of per-step increments, and the ordered
phase's measured ``activity`` divided by that floor converges to **1.004** as
``dt → 0``.  Two consequences followed: the load carried the integrator's step,
and the two points the mechanism actually compares — the deep-locked phase and
the ΔC peak — paid the same to within 4% (``ρ = 1.04``), against ``ρ = 34`` on
the Hopfield network.  So the third substrate agreed with the second on the
*verdict* while not having the differential structure the *mechanism* names.

`kuramoto_substrate.simulate` now also reports ``repair_rate``: the mean over
the window of ``std_i(v_i − ⟨v⟩)`` for the **deterministic** phase velocity
``v_i = ω_i + K·r·sin(ψ − θ_i)``.  It removes the injected noise, and being a
rate rather than a per-step displacement it carries no ``dt``.  It keeps the one
property the transplant design needs: in a deterministic locked state every
oscillator shares one collective velocity, so it is exactly zero and the ordered
phase is genuinely free.  ``activity`` is unchanged and bit-identical — three
published results are stated in it.

This experiment re-decides the transplant on the corrected reading.  Both loads
are measured on the **same** scans, so the comparison is paired and no part of
the difference is sampling noise.

Pre-registered predictions:

Q1. **The corrected load is drive-free and unit-free.**  The undriven locked
    phase pays exactly 0 while the driven locked phase pays a nonzero rate; and
    across ``dt = 0.2 → 0.0125`` — a factor of 16, so ``√dt`` spans 4 — the
    corrected load at the ordered point varies by **less than 10%** while
    ``activity`` varies by **more than 3×**.  ("Pays exactly 0" is read as
    *negligible beside the driven cost* — under 1% of it.  A locked state only
    relaxes to machine tolerance, so both readings sit around ``1e-9``/``1e-10``
    undriven rather than at a hard zero; the first draft of this check used
    float equality, which tests the arithmetic rather than the substrate.  The
    thresholds are otherwise as first written.)  **Falsifier:** the corrected load
    still tracks the integrator's step, which would mean the units problem is
    intrinsic to how this substrate can be read at all.

Q2. **The load profile is no longer flat.**  Under the corrected metric
    ``ρ = L_⋆/L_o`` exceeds **1.2**, against ``1.04`` for the published reading.
    **Falsifier:** ``ρ`` stays within 20% of 1 — meaning even a drive-free load
    cannot tell the ordered point from the critical point, and this substrate
    has no differential structure to carry the mechanism at all.

Q3. **And the relocation is re-decided, all three clauses.**  With the corrected
    load: (a) the undriven arm stays scarcity-proof — capacity floor 1, no
    eviction at any consumption; (b) the driven arm still relocates — floor
    below ``κ_o⋆``; and (c) the route collapses to a **single hop landing on the
    ΔC peak**, matching the Hopfield network instead of the two-hop walk the
    published reading produces.  **Falsifier:** any clause fails.

    Clause (c) is the one genuinely at risk, and it is registered because its
    failure is the informative outcome.  The load correction cannot move ``ΔI``
    or ``ΔC``, so the flat-load convex hull — three vertices here, two on the
    Hopfield network — is untouched by it.  For the walk to collapse, the
    intermediate partially-synchronised state would have to be taxed enough to
    be skipped.  If it is not, then the disagreement between the second and
    third substrates was never about the load convention: it is in the
    substrate's ``(ΔI, ΔC)`` anatomy, where no load correction can reach.

Honest scope.  One coupling, one noise strength, one frequency distribution,
one population size; ``w = 1.5`` and the ladder range are the published
conventions, unchanged.  ``repair_rate`` is still a *convention* for what
"load" means on an oscillator population — a defensible one, and specifically
one that does not read back its own drive, but the transplant's dictionary is
a choice and this replaces one choice with another rather than removing the
need to make it.

    python experiments/n3_kuramoto_repair.py
    python experiments/n3_kuramoto_repair.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.crossing import (  # noqa: E402
    capacity_floor,
    kappa_at_crossing,
    measured_crossing,
    optimum_walk,
    two_point_reduction,
)
from project_genesis.kuramoto_substrate import (  # noqa: E402
    natural_frequencies,
    simulate,
    window_metrics,
)

BANNER = "=" * 78

PUBLISHED = dict(n_osc=400, coupling=2.0, dt=0.05, n_steps=900, n_burn=450,
                 n_seeds=3, g_min=0.1, g_max=2.4, n_gamma=13, freq_tol=0.3,
                 noise=0.6, seed=0)


def scan(cfg: dict, driven: bool) -> list:
    """One disorder scan recording BOTH load conventions on the same runs."""
    gammas = np.linspace(cfg["g_min"], cfg["g_max"], cfg["n_gamma"])
    noise = cfg["noise"] if driven else 0.0
    rows = []
    for g in gammas:
        acc = {k: [] for k in ("delta_c", "delta_i", "activity", "repair_rate")}
        for s in range(cfg["n_seeds"]):
            rng = np.random.default_rng(cfg["seed"] + 1000 * s)
            omega = natural_frequencies(cfg["n_osc"], g, rng,
                                        distribution="gaussian")
            phases = rng.uniform(-np.pi, np.pi, cfg["n_osc"])
            sim = simulate(phases, omega, cfg["coupling"], cfg["dt"],
                           cfg["n_steps"], noise, rng, burn_in=cfg["n_burn"])
            m = window_metrics(sim, freq_tol=cfg["freq_tol"])
            for k in acc:
                acc[k].append(m[k])
        rows.append({"gamma": float(g),
                     **{k: float(np.mean(v)) for k, v in acc.items()},
                     "delta_c_err": float(np.std(acc["delta_c"])
                                          / np.sqrt(cfg["n_seeds"]))})
    return rows


def as_load(rows, key):
    """Re-key a scan so `crossing` reads whichever load convention we want."""
    return [dict(r, activity=r[key]) for r in rows]


def verdict(rows, key, weight, recovery, u_max):
    r = as_load(rows, key)
    red = two_point_reduction(r, x_key="gamma")
    k_star = kappa_at_crossing(red, weight)
    floor = capacity_floor(red, u_max)
    meas = measured_crossing(r, x_key="gamma", weight=weight, recovery=recovery)
    return {"rho": red["rho"], "load_o": red["load_o"], "load_s": red["load_s"],
            "kappa_star": k_star, "floor": floor,
            "evicted": bool(np.isfinite(meas["c_evict"])),
            "hops": meas["hops"], "walk": meas["walk"],
            "x_critical": meas["x_critical"], "x_end": meas["x_end"],
            "kappa_arrive": meas["kappa_arrive"], "c_arrive": meas["c_arrive"]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--weight", type=float, default=1.5)
    p.add_argument("--recovery", type=float, default=0.1)
    p.add_argument("--c-max", type=float, default=50.0)
    p.add_argument("--dts", type=float, nargs="+",
                   default=[0.2, 0.1, 0.05, 0.025, 0.0125])
    p.add_argument("--duration", type=float, default=45.0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    cfg = dict(PUBLISHED)
    if args.quick:
        cfg.update(n_osc=250, n_seeds=2, n_steps=500, n_burn=250, n_gamma=10)
        args.duration = 25.0
    u_max = args.c_max / args.recovery

    print(BANNER)
    print("THE OSCILLATOR TRANSPLANT, WITH THE DRIVE TAKEN BACK OUT")
    print(BANNER)
    print("  Pre-registered:")
    print("    Q1  repair_rate is drive-free and dt-free (<10% vs activity's >3x)")
    print("    Q2  the corrected load profile is not flat: rho > 1.2")
    print("    Q3  undriven still proof, driven still relocates, and the route")
    print("        collapses to ONE hop onto the ΔC peak")
    print()

    t0 = time.time()
    print("  measuring both load conventions on the same scans...", flush=True)
    rows_u = scan(cfg, driven=False)
    rows_d = scan(cfg, driven=True)
    print(f"  [{time.time() - t0:.0f}s]")
    print()
    print(f"  {'γ':>6} {'ΔC':>7} {'ΔI':>7} {'activity':>10} {'repair_rate':>12}")
    for r in rows_d:
        print(f"  {r['gamma']:>6.2f} {r['delta_c']:>7.3f} {r['delta_i']:>7.3f} "
              f"{r['activity']:>10.4f} {r['repair_rate']:>12.4f}")

    # ------------------------------------------------------------------- Q1
    print()
    print(BANNER)
    print("Q1  IS THE CORRECTED LOAD FREE OF THE DRIVE AND OF THE INTEGRATOR?")
    print(BANNER)
    u_act = rows_u[0]["activity"]
    u_rep = rows_u[0]["repair_rate"]
    d_rep = rows_d[0]["repair_rate"]
    print(f"  undriven locked phase:  activity {u_act:.6f}   "
          f"repair_rate {u_rep:.6f}")
    print(f"  driven   locked phase:  activity {rows_d[0]['activity']:.6f}   "
          f"repair_rate {d_rep:.6f}")
    print("  (the toggle survives only if the undriven reading is 0 and the")
    print("   driven one is not — otherwise 'free order' stops being free)")
    print()
    print(f"  {'dt':>8} {'σ√dt':>8} {'activity':>10} {'act/σ√dt':>10} "
          f"{'repair_rate':>12}")
    sweep = []
    for dtv in args.dts:
        c2 = dict(cfg)
        c2["dt"] = float(dtv)
        c2["n_steps"] = int(round(args.duration / dtv))
        c2["n_burn"] = c2["n_steps"] // 2
        r0 = scan(c2, driven=True)[0]
        floor_noise = cfg["noise"] * np.sqrt(dtv)
        sweep.append({"dt": float(dtv), "activity": r0["activity"],
                      "repair_rate": r0["repair_rate"],
                      "act_over_floor": r0["activity"] / floor_noise})
        s = sweep[-1]
        print(f"  {dtv:>8.4f} {floor_noise:>8.4f} {s['activity']:>10.5f} "
              f"{s['act_over_floor']:>10.3f} {s['repair_rate']:>12.5f}")

    acts = np.array([s["activity"] for s in sweep])
    reps = np.array([s["repair_rate"] for s in sweep])
    act_span = float(acts.max() / acts.min())
    rep_span = float((reps.max() - reps.min()) / reps.mean())
    print()
    print(f"  activity spans a factor of {act_span:.2f} across the dt sweep")
    print(f"  repair_rate spans {rep_span:.1%} of its mean")
    # "pays exactly 0" is a physical claim, not a float-equality one: a locked
    # state relaxes to machine tolerance, so the undriven readings are ~1e-9
    # (the published `activity` is ~1e-10 on the same runs, and that experiment
    # calls it zero). Implemented as "negligible beside the driven cost".
    free_ratio = u_rep / d_rep if d_rep > 0 else float("inf")
    q1 = bool(free_ratio < 0.01 and d_rep > 0.0
              and rep_span < 0.10 and act_span > 3.0)
    print()
    if q1:
        print("  PREDICTION HELD — the corrected reading measures the system.")
        print("  The restoring work the coupling does against the perturbation")
        print("  is a rate, it does not depend on how finely the trajectory is")
        print("  integrated, and a rhythm nobody is perturbing still pays")
        print("  exactly nothing. That is what the load was supposed to be.")
    else:
        print("  PREDICTION FAILED — reported as measured:")
        print(f"    undriven repair_rate {u_rep:.3e} = {free_ratio:.2%} of the "
              f"driven cost (needed < 1%)")
        print(f"    driven repair_rate   {d_rep:.6f} (needed > 0)")
        print(f"    repair spread {rep_span:.1%} (needed < 10%), "
              f"activity spread {act_span:.2f}x (needed > 3x)")

    # ------------------------------------------------------------------- Q2
    print()
    print(BANNER)
    print("Q2  DOES THE ORDERED PHASE NOW PAY DIFFERENTLY FROM THE CRITICAL ONE?")
    print(BANNER)
    v_act = verdict(rows_d, "activity", args.weight, args.recovery, u_max)
    v_rep = verdict(rows_d, "repair_rate", args.weight, args.recovery, u_max)
    print(f"  {'load':>14} {'L_o':>9} {'L_⋆':>9} {'ρ = L_⋆/L_o':>13}")
    print(f"  {'activity':>14} {v_act['load_o']:>9.4f} {v_act['load_s']:>9.4f} "
          f"{v_act['rho']:>13.2f}   (published)")
    print(f"  {'repair_rate':>14} {v_rep['load_o']:>9.4f} {v_rep['load_s']:>9.4f} "
          f"{v_rep['rho']:>13.2f}   (corrected)")
    q2 = bool(v_rep["rho"] > 1.2)
    print()
    if q2:
        print("  PREDICTION HELD — there IS differential structure here, and")
        print("  the published reading was hiding it under a uniform noise")
        print("  floor. The critical neighbourhood costs more to hold than the")
        print("  locked phase does, which is the shape the mechanism needs.")
        print(f"  It is still far shallower than the Hopfield network's 34: the")
        print("  two substrates differ in degree, where before they differed in")
        print("  kind.")
    else:
        print(f"  PREDICTION FAILED — ρ = {v_rep['rho']:.2f}, still flat.")
        print("  Then even a drive-free load cannot separate the two points the")
        print("  mechanism compares, and this substrate carries the verdict")
        print("  without carrying the mechanism.")

    # ------------------------------------------------------------------- Q3
    print()
    print(BANNER)
    print("Q3  IS THE RELOCATION RE-DECIDED?")
    print(BANNER)
    v_u = verdict(rows_u, "repair_rate", args.weight, args.recovery, u_max)
    print(f"  (a) undriven, corrected load: floor {v_u['floor']:.4f} vs "
          f"κ_o⋆ {v_u['kappa_star']:.4f} -> "
          f"{'evicted' if v_u['evicted'] else 'scarcity-proof'}")
    print(f"  (b) driven,   corrected load: floor {v_rep['floor']:.4f} vs "
          f"κ_o⋆ {v_rep['kappa_star']:.4f} -> "
          f"{'RELOCATES' if v_rep['evicted'] else 'stays put'}")
    print(f"  (c) route: {' -> '.join(f'{v:.2f}' for v in v_rep['walk'])}  "
          f"({v_rep['hops']} hop{'s' if v_rep['hops'] != 1 else ''}), "
          f"ΔC peak at γ = {v_rep['x_critical']:.2f}")
    print(f"      published reading for comparison: "
          f"{' -> '.join(f'{v:.2f}' for v in v_act['walk'])} "
          f"({v_act['hops']} hops)")
    hull = optimum_walk(rows_d)
    print(f"      flat-load hull (untouched by the load correction): "
          f"{len(hull)} vertices")
    a = not v_u["evicted"]
    b = v_rep["evicted"]
    c = bool(v_rep["hops"] == 1
             and abs(v_rep["x_end"] - v_rep["x_critical"]) < 1e-9)
    q3 = a and b and c
    print()
    print(f"  (a) {'held' if a else 'FAILED'}   (b) {'held' if b else 'FAILED'}"
          f"   (c) {'held' if c else 'FAILED'}")
    print()
    if q3:
        print("  PREDICTION HELD in full — the third substrate is repaired. It")
        print("  now carries the mechanism, not just the verdict: order pays,")
        print("  the tax is differential, and the optimum crosses once onto the")
        print("  maximum-distinction state exactly as the Hopfield network does.")
    elif a and b and not c:
        print("  PREDICTION FAILED on clause (c), and that is the informative")
        print("  outcome. The load correction does what a load correction can:")
        print("  the reading is drive-free and unit-free. What it cannot do is")
        print("  move the optimum's ROUTE, because the route lives in the")
        print("  (ΔI, ΔC) cloud and the load only sets the timing of each hop.")
        if v_rep["hops"] > v_act["hops"]:
            print(f"  Here it made the walk LONGER, not shorter — "
                  f"{v_act['hops']} hop(s) on the published")
            print("  reading against "
                  f"{v_rep['hops']} on the corrected one: taxing the ordered")
            print("  phase harder evicts it EARLIER, onto a state that is only")
            print("  partly on the way to the ΔC peak. The opposite of the")
            print("  predicted collapse, and it sharpens the point rather than")
            print("  softening it.")
        print("  This substrate has a viable intermediate state — a partially")
        print("  synchronised population holding substantial coherence AND")
        print("  near-peak distinction — and the Hopfield network does not.")
        print("  So the disagreement between the two transplants was never")
        print("  about the load convention. It is anatomy, and it survives.")
    else:
        print("  PREDICTION FAILED — reported as measured; see the clauses above.")

    # -------------------------------------------------------------- verdict
    print()
    print(BANNER)
    n = int(q1) + int(q2) + int(q3)
    print(f"VERDICT: {n}/3 pre-registered predictions held.")
    print(BANNER)
    if q1 and q2:
        print("  What changed: the oscillator substrate's load no longer reads")
        print("  back its own drive, no longer carries the integrator's step,")
        print(f"  and separates the two competing points by {v_rep['rho']:.2f}x "
              "instead of")
        print(f"  {v_act['rho']:.2f}x. On that reading the mechanism's condition "
              "is genuinely")
        print("  satisfied here — order that must be maintained does pay, and")
        print("  scarcity does evict it.")
        print()
    if not q3:
        print("  What did not change: where the optimum goes. That is fixed by")
        print("  the substrate's (ΔI, ΔC) anatomy, which no choice of load")
        print("  metric can alter. 'One law across three substrates' is")
        print("  defensible for the CONDITION; it is not defensible for the")
        print("  DESTINATION, and the honest statement separates the two.")
    print()
    print("  Note the published transplants remain reproducible: `activity` is")
    print("  unchanged and bit-identical, and nothing here rewrites their")
    print("  results — it adds a second reading and says which one measures")
    print("  the system.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "kuramoto_repair.json"
        out.write_text(json.dumps(
            {"params": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
             "config": cfg, "rows_undriven": rows_u, "rows_driven": rows_d,
             "dt_sweep": sweep, "verdict_activity": v_act,
             "verdict_repair": v_rep, "verdict_undriven": v_u,
             "hull_vertices": len(hull), "score": n}, indent=2, default=float))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
