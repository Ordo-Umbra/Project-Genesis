"""Does a dictionary-free ΔC survive the sweep that broke the threshold one?

`n3_anatomy` established that the transplant programme's ΔC is a convention on
every substrate, and that the convention carries conclusions: sweeping the
Hopfield structured-overlap threshold across `0.15 … 0.50` slides the ΔC peak
from `T⋆ = 1.20` to `0.70` and out of the published critical window at the two
most permissive settings, while both substrates' optimum routes flip between a
single crossing and a walk. Since ``S = ΔC + κ·ΔI``, that is a gap in the
principle's own definition.

`project_genesis/causal_state.py` proposes a ΔC with no cut in it: two states
are the same state when the dynamics cannot tell their futures apart, and
"cannot tell apart" is calibrated against the spread of a *single* state's own
replicates.  The resolution comes from the system's stochasticity rather than
from a number anyone picked.

This experiment asks whether that actually helps, by running the **same sweep
design that broke the old reading** — and it is set up so that the answer can
be no.

**Stated before running, because it is a limitation and not a footnote.**  The
construction has no cut but it does have a *statistical power*: more replicates
resolve finer distinctions, so the absolute level of causal ΔC rises with
``n_replicates`` by construction, and for a continuous state space it would
approach its maximum in the limit.  The free parameter has therefore moved from
"what counts as structured" — a substrate semantic, with no principled value —
to "how many samples" — a statistical choice whose effect is monotone and
understood.  That is progress only if the *shape* of the ΔC curve is invariant
where its level is not, which is exactly what Q2 and Q3 test.  Absolute levels
are not compared anywhere below.

**Disclosure — the horizon range was wrong and is corrected; the thresholds are
not.**  ``τ`` was first swept over ``{2, 4, 8}`` *steps*.  A probe showed causal
ΔC is identically ``1.000`` everywhere in that range on both substrates: at
horizons short against the substrate's own relaxation, every microstate still
remembers itself, so every sampled state resolves as its own causal state and
the reading carries no information about anything.  The range was uninformative
by construction, exactly as the ``dt`` range in `n3_crossing_prediction` was
one-sided by construction.

``τ`` is therefore now swept in **units of each substrate's own relaxation
time** — one Glauber sweep for the network, ``1/(K·r) ≈ 10`` integrator steps
for the oscillators — which is also the only way a horizon can mean the same
thing on two substrates whose clocks are unrelated.  The registered thresholds
are unchanged.

The probe's numbers are on the record here, because they already foreshadow the
answer.  On the oscillators, ΔC across ``γ = 0.10 / 1.06 / 2.40``:

    τ =   2 steps   1.000  1.000  1.000     (nothing resolved)
    τ =  25 steps   0.408  1.000  1.000
    τ = 150 steps   0.141  0.797  1.000     (monotone INCREASING in disorder)
    τ = 400 steps   0.265  0.413  0.000     (PEAKED at intermediate disorder)

The shape does not converge as the horizon grows — it *inverts*.  That is worse
than a level that drifts, and Q2 is where it gets tested.

Pre-registered predictions:

Q1. **The causal reading finds the same anatomy.**  Causal ΔC peaks inside the
    published critical window ``|x⋆ − x_c| ≤ 0.35·x_c`` on both substrates at
    the mid settings.  **Falsifier:** it peaks elsewhere — then it is measuring
    something other than distinction and the comparison below is meaningless.

Q2. **And its peak is robust where the threshold peak was not.**  Across the
    sweep of everything still free — horizon ``τ``, replicate count, and merge
    confidence — the causal ΔC peak stays inside that window in **every** arm.
    The threshold reading failed this in 2 of 7 arms on the Hopfield network.
    **Falsifier:** any arm leaves the window, in which case the construction has
    relocated the arbitrariness rather than removed it.

Q3. **And the route becomes determinate.**  The sign of the chord excess — the
    optimum's number of stops — is constant across the whole causal sweep on
    each substrate, where under the threshold reading it flipped on both.
    **Falsifier:** it still flips, meaning the destination is not recoverable
    even from a dictionary-free distinction, and the honest statement stops at
    the neighbourhood permanently.

Honest scope.  ΔI is left exactly as the transplants define it — this changes
one half of ``S``, not both, and a fair comparison needs the other half fixed.
The oscillator load is the corrected ``repair_rate`` throughout.  Sampled states
are snapshots from a single equilibrated trajectory per seed, so they are not
independent draws from the stationary measure; the replicates that carry the
distinguishability *are* independent, which is what the calibration needs.
Nothing here re-opens whether the relocation happens.

    python experiments/n3_causal_delta_c.py
    python experiments/n3_causal_delta_c.py --quick
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

from project_genesis.anatomy import chord_excess, hull_size  # noqa: E402
from project_genesis.causal_state import causal_delta_c  # noqa: E402
from project_genesis.hopfield_substrate import (  # noqa: E402
    best_overlap, glauber_sweep, hebbian_weights,
)
from project_genesis.kuramoto_substrate import (  # noqa: E402
    kuramoto_step, natural_frequencies, order_parameter,
)

BANNER = "=" * 78

# one relaxation time, in each substrate's own native step: a Glauber sweep
# touches every spin once; the locked oscillator population relaxes on 1/(K r),
# which is ~10 integrator steps at K=2, dt=0.05.
RELAX = {"hopfield": 1.0, "kuramoto": 10.0}

HOP = dict(n_spins=300, n_patterns=9, n_seeds=3, n_burn=30, n_window=60,
           t_min=0.1, t_max=1.6, n_temps=16)
KUR = dict(n_osc=400, coupling=2.0, dt=0.05, n_steps=900, n_burn=450,
           n_seeds=3, g_min=0.1, g_max=2.4, n_gamma=13, noise=0.6)


# ------------------------------------------------------------------ substrates

def hopfield_point(T, cfg, seed, n_states, tau, n_rep, conf):
    """ΔI (retrieval overlap) and causal ΔC at one temperature."""
    dis, dcs = [], []
    for s in range(cfg["n_seeds"]):
        rng = np.random.default_rng(seed + 1000 * s)
        pats = rng.choice([-1.0, 1.0], size=(cfg["n_patterns"], cfg["n_spins"]))
        j = hebbian_weights(pats)
        state = pats[0].copy()
        for _ in range(cfg["n_burn"]):
            state = glauber_sweep(state, j, T, rng)
        # snapshots along the window, and the published DI reading alongside
        every = max(1, cfg["n_window"] // n_states)
        samples, overlaps = [], []
        for k in range(cfg["n_window"]):
            state = glauber_sweep(state, j, T, rng)
            overlaps.append(best_overlap(state, pats)[0])
            if k % every == 0 and len(samples) < n_states:
                samples.append(state.copy())
        dis.append(float(np.mean(overlaps)))

        def evolve(st, steps, r):
            out = st.copy()
            for _ in range(steps):
                out = glauber_sweep(out, j, T, r)
            return out

        dcs.append(causal_delta_c(samples, evolve, lambda v: v, rng,
                                  tau=tau, n_replicates=n_rep,
                                  confidence=conf)["delta_c"])
    return float(np.mean(dis)), float(np.mean(dcs))


def kuramoto_point(g, cfg, seed, n_states, tau, n_rep, conf):
    """ΔI (coherence), causal ΔC, and the corrected repair-rate load."""
    dis, dcs, loads = [], [], []
    for s in range(cfg["n_seeds"]):
        rng = np.random.default_rng(seed + 1000 * s)
        omega = natural_frequencies(cfg["n_osc"], g, rng,
                                    distribution="gaussian")
        ph = rng.uniform(-np.pi, np.pi, cfg["n_osc"])
        for _ in range(cfg["n_burn"]):
            ph = kuramoto_step(ph, omega, cfg["coupling"], cfg["dt"],
                               cfg["noise"], rng)
        n_meas = cfg["n_steps"] - cfg["n_burn"]
        every = max(1, n_meas // n_states)
        samples, rs, rep = [], [], []
        for k in range(n_meas):
            r_now, psi = order_parameter(ph)
            v = omega + cfg["coupling"] * r_now * np.sin(psi - ph)
            rep.append(float(np.std(v - v.mean())))
            ph = kuramoto_step(ph, omega, cfg["coupling"], cfg["dt"],
                               cfg["noise"], rng)
            rs.append(order_parameter(ph)[0])
            if k % every == 0 and len(samples) < n_states:
                samples.append(ph.copy())
        dis.append(float(np.mean(rs)))
        loads.append(float(np.mean(rep)))

        def evolve(st, steps, r):
            out = st.copy()
            for _ in range(steps):
                out = kuramoto_step(out, omega, cfg["coupling"], cfg["dt"],
                                    cfg["noise"], r)
            return out

        dcs.append(causal_delta_c(
            samples, evolve,
            lambda p: np.concatenate([np.cos(p), np.sin(p)]), rng,
            tau=tau, n_replicates=n_rep, confidence=conf)["delta_c"])
    return float(np.mean(dis)), float(np.mean(dcs)), float(np.mean(loads))


def scan(kind, cfg, seed, n_states, tau, n_rep, conf):
    rows = []
    if kind == "hopfield":
        for T in np.linspace(cfg["t_min"], cfg["t_max"], cfg["n_temps"]):
            di, dc = hopfield_point(T, cfg, seed, n_states, tau, n_rep, conf)
            rows.append({"T": float(T), "delta_i": di, "delta_c": dc,
                         "activity": 0.0})
    else:
        for g in np.linspace(cfg["g_min"], cfg["g_max"], cfg["n_gamma"]):
            di, dc, load = kuramoto_point(g, cfg, seed, n_states, tau, n_rep,
                                          conf)
            rows.append({"gamma": float(g), "delta_i": di, "delta_c": dc,
                         "activity": load})
    return rows


def critical_x(rows, x_key):
    for a, b in zip(rows, rows[1:]):
        if a["delta_i"] >= 0.5 >= b["delta_i"]:
            d = a["delta_i"] - b["delta_i"]
            f = (a["delta_i"] - 0.5) / d if d > 0 else 0.0
            return float(a[x_key] + f * (b[x_key] - a[x_key]))
    return float("nan")


def reading(rows, x_key, window):
    x = np.array([r[x_key] for r in rows])
    dc = np.array([r["delta_c"] for r in rows])
    x_peak = float(x[int(np.argmax(dc))])
    x_c = critical_x(rows, x_key)
    e = chord_excess(rows)
    return {"x_peak": x_peak, "x_c": x_c,
            "in_window": bool(np.isfinite(x_c)
                              and abs(x_peak - x_c) <= window * x_c),
            "hull": hull_size(rows), "excess": e["excess"],
            "excess_frac": e["excess_frac"], "dc_max": float(dc.max())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--critical-window", type=float, default=0.35)
    p.add_argument("--n-states", type=int, default=20)
    p.add_argument("--tau-scales", type=float, nargs="+",
                   default=[1.0, 5.0, 20.0, 40.0],
                   help="Horizon in units of the substrate's own relaxation "
                        "time, so it means the same thing on both.")
    p.add_argument("--replicates", type=int, nargs="+", default=[8, 12, 20])
    p.add_argument("--confidences", type=float, nargs="+",
                   default=[0.90, 0.95, 0.99])
    p.add_argument("--seed", type=int, default=31337)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    hop, kur = dict(HOP), dict(KUR)
    if args.quick:
        hop.update(n_spins=150, n_seeds=1, n_burn=15, n_window=30, n_temps=8)
        kur.update(n_osc=200, n_seeds=1, n_steps=500, n_burn=250, n_gamma=8)
        args.n_states = 10
        args.tau_scales, args.replicates, args.confidences = [5.0, 40.0], [8], [0.95]

    mid = (args.tau_scales[len(args.tau_scales) // 2],
           args.replicates[len(args.replicates) // 2],
           args.confidences[len(args.confidences) // 2])

    print(BANNER)
    print("A ΔC WITH NO CUT IN IT — does it survive what broke the old one?")
    print(BANNER)
    print(f"  states sampled per point {args.n_states}; mid settings "
          f"τ={mid[0]:g}×relax R={mid[1]} conf={mid[2]}")
    print("  τ is in units of each substrate's own relaxation time "
          f"({RELAX['hopfield']:g} sweep / {RELAX['kuramoto']:g} steps)")
    print("  NOTE, stated before running: causal ΔC's absolute LEVEL rises with")
    print("  replicate count by construction. Only its SHAPE is compared here.")
    print()
    print("  Pre-registered:")
    print("    Q1  the causal ΔC peak lands in the critical window (mid settings)")
    print("    Q2  and stays there across every arm of the sweep (threshold: 2/7 failed)")
    print("    Q3  and the route stops flipping (threshold: flipped on both)")
    print()

    t0 = time.time()
    arms = {}
    combos = [(t, r, c) for t in args.tau_scales for r in args.replicates
              for c in args.confidences]
    for kind, cfg, xk in (("hopfield", hop, "T"), ("kuramoto", kur, "gamma")):
        print(BANNER)
        print(f"{kind.upper()} — causal ΔC across the free statistical choices")
        print(BANNER)
        print(f"  {'tau/τ':>4} {'R':>4} {'conf':>5} {'hull':>5} {'E/gap':>8} "
              f"{'ΔC max':>7} {'x⋆':>6} {'x_c':>6} {'in win':>7}")
        for (scale, rep, conf) in combos:
            tau = max(1, int(round(scale * RELAX[kind])))
            rows = scan(kind, cfg, args.seed, args.n_states, tau, rep, conf)
            r = reading(rows, xk, args.critical_window)
            arms[f"{kind} tau={scale:g} R={rep} conf={conf}"] = {
                **r, "kind": kind, "tau_steps": tau}
            print(f"  {scale:>4g} {rep:>4} {conf:>5.2f} {r['hull']:>5} "
                  f"{r['excess_frac']:>8.3f} {r['dc_max']:>7.3f} "
                  f"{r['x_peak']:>6.2f} {r['x_c']:>6.2f} "
                  f"{'yes' if r['in_window'] else 'NO':>7}")
        print()
    print(f"  [{time.time() - t0:.0f}s]")

    h = {k: v for k, v in arms.items() if v["kind"] == "hopfield"}
    q = {k: v for k, v in arms.items() if v["kind"] == "kuramoto"}

    # ------------------------------------------------------------------- Q1
    print()
    print(BANNER)
    print("Q1  DOES THE CAUSAL READING FIND THE SAME ANATOMY?")
    print(BANNER)
    mid_keys = [f"hopfield tau={mid[0]:g} R={mid[1]} conf={mid[2]}",
                f"kuramoto tau={mid[0]:g} R={mid[1]} conf={mid[2]}"]
    q1 = all(arms[k]["in_window"] for k in mid_keys if k in arms)
    for k in mid_keys:
        if k in arms:
            v = arms[k]
            print(f"  {k:<40} x⋆={v['x_peak']:.2f} x_c={v['x_c']:.2f} "
                  f"{'inside' if v['in_window'] else 'OUTSIDE'}")
    print()
    print("  PREDICTION " + ("HELD — the construction is measuring distinction,"
                             if q1 else "FAILED — it peaks somewhere else,"))
    print("  " + ("so the robustness questions below mean something."
                  if q1 else "so it is not a replacement and Q2/Q3 are moot."))

    # ------------------------------------------------------------------- Q2
    print()
    print(BANNER)
    print("Q2  IS THE PEAK ROBUST WHERE THE THRESHOLD PEAK WAS NOT?")
    print(BANNER)
    out = [k for k, v in arms.items() if not v["in_window"]]
    print(f"  arms with the ΔC peak inside the critical window: "
          f"{len(arms) - len(out)}/{len(arms)}")
    for k in out:
        print(f"    OUTSIDE: {k}  x⋆={arms[k]['x_peak']:.2f} "
              f"x_c={arms[k]['x_c']:.2f}")
    for kind, d in (("hopfield", h), ("kuramoto", q)):
        peaks = sorted({v["x_peak"] for v in d.values()})
        print(f"  {kind} peak locations across the sweep: "
              f"{', '.join(f'{p:.2f}' for p in peaks)}")
    q2 = not out
    print()
    if q2:
        print("  PREDICTION HELD — every arm keeps the peak in the critical")
        print("  neighbourhood. The threshold reading could not do this: its")
        print("  peak slid T⋆ 1.20 → 0.70 and left the window twice in seven.")
        print("  Removing the cut removed that failure mode.")
    else:
        print("  PREDICTION FAILED — the peak leaves the window under choices")
        print("  that are only sample size and a p-value. Then the")
        print("  construction moved the arbitrariness instead of removing it,")
        print("  and a dictionary-free ΔC is harder than this.")

    # ------------------------------------------------------------------- Q3
    print()
    print(BANNER)
    print("Q3  DOES THE ROUTE STOP FLIPPING?")
    print(BANNER)
    hh, qh = sorted({v["hull"] for v in h.values()}), sorted({v["hull"] for v in q.values()})
    print(f"  hopfield hull sizes across the causal sweep: {hh}   "
          f"(threshold sweep gave [2, 3])")
    print(f"  kuramoto hull sizes across the causal sweep: {qh}   "
          f"(threshold sweep gave [2, 3])")
    q3 = len(hh) == 1 and len(qh) == 1
    print()
    if q3:
        print("  PREDICTION HELD — the destination is determinate. With")
        print("  distinction defined by what the dynamics keep apart, each")
        print("  substrate has ONE route, and the difference between them is")
        print("  a fact about the substrates rather than about the readings.")
        print("  That is the claim §5 had to retract; this is what it would")
        print("  take to make it again.")
    else:
        print("  PREDICTION FAILED — the route still flips.")
        print("  Then the destination is not recoverable from a dictionary-free")
        print("  ΔC either, and the honest statement stops at the")
        print("  neighbourhood: a relocation happens, and where it lands is")
        print("  not a determinate fact about the substrate.")

    print()
    print(BANNER)
    n = int(q1) + int(q2) + int(q3)
    print(f"VERDICT: {n}/3 pre-registered predictions held.")
    print(BANNER)
    print("  What the free parameters now are: a horizon, a replicate count,")
    print("  and a p-value. What they were: 'how much overlap makes a state")
    print("  structured'. The second kind has no principled value and carried")
    print("  the conclusions; the first kind is sample size. That much is a")
    print("  real change regardless of the score above.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        o = args.output_dir / "causal_delta_c.json"
        o.write_text(json.dumps(
            {"params": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
             "hopfield_settings": hop, "kuramoto_settings": kur,
             "arms": arms, "score": n}, indent=2, default=float))
        print(f"\n  wrote {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
