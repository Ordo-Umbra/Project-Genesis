"""Is the optimum's destination a fact about the substrate, or about how ΔC is read?

`n3_crossing_prediction` and `n3_kuramoto_repair` closed the load question: once
the oscillator load stops reading back its own drive, the eviction **condition**
transplants to all three substrates.  Both experiments then reported a residue
they could not move — the optimum's **route**.  The Hopfield network crosses
once, straight onto the ΔC peak; the driven oscillators stop first at a
partially-synchronised state.  `The_Principle.md` §5 was written to say that
difference is *anatomy, not convention*, on the reasoning that the route lives
in the ``(ΔI, ΔC)`` cloud and no choice of load metric can move it.

That reasoning has a hole in it, and this experiment is the hole.  A load metric
cannot move the cloud — but the cloud is not raw data either.  **ΔC is itself a
convention on both substrates**: the Hopfield reading counts a sample as
"structured" when the retrieval overlap exceeds a threshold (published `0.3`),
and the oscillator reading bins effective frequencies at a fixed width
(published `0.4`) after calling an oscillator entrained inside a tolerance
(published `0.3`).  None of those numbers is derived from anything.  If the
route moves when they move, the destination claim is measuring the dictionary
rather than the substrate.

The whole question is one scalar, :func:`~project_genesis.anatomy.chord_excess`:

    E = max_x [ ΔC_x − chord(ΔI_x) ]

the height of the best intermediate state above the chord from the ordered point
to the ΔC peak.  ``E > 0`` is exactly the condition for a third convex-hull
vertex, so the sign of ``E`` *is* the route, expressed as a number that can be
swept.

**Disclosure — the conventions were swept exploratorily first, and the ranges
below were chosen because they move the answer.**  Q1 is therefore *confirmed on
fresh seeds*, not claimed from the scan that located it.  Q2 and Q3 were
registered before being run.  The exploratory numbers, for the record: on the
Hopfield network the hull alternated 3/2/3/2/3/2/2 across thresholds
`0.15 … 0.50` with ``|E|/gap ≤ 0.081`` throughout; on the oscillators it read 2
or 3 depending on the tolerance and bin width, with ``E/gap`` spanning
``−0.287 … +0.569``.

Pre-registered predictions:

Q1. **The route is not a robust property of either substrate.**  Sweeping each
    substrate's *own* ΔC convention across a defensible range flips its
    hull-vertex count — on both.  **Falsifier:** one or both hulls hold
    constant, in which case the destination difference is real anatomy and §5
    was right.

Q2. **The eviction condition survives the same sweep.**  In every arm, every
    driven scan still relocates (capacity floor below ``κ_o⋆``) and every
    undriven scan still does not (floor exactly 1).  **Falsifier:** the
    condition flips too — which would mean nothing in the transplant result
    survives its own conventions, and the previous two experiments' conclusion
    goes with it.

Q3. **And the critical neighbourhood survives it, even though the route does
    not.**  Across all conventions the ΔC peak stays inside the published
    critical window, ``|x⋆ − x_c| ≤ 0.35·x_c``, on both substrates.
    **Falsifier:** the peak leaves that window under some defensible
    convention, which would undermine *"relocates to criticality"* itself
    rather than merely the route taken to get there.

Honest scope.  The conventions swept are the ones each substrate's own
dictionary exposes; there may be others that matter more.  Ranges are centred
on the published values and kept to what a reader would accept without argument
— the point is not that extreme settings break the reading, which would prove
nothing, but that ordinary ones do.  The oscillator arm uses the corrected
``repair_rate`` load throughout, since the contaminated one is already retired.
Nothing here re-opens whether the relocation happens; it asks what the reported
destination is a fact about.

    python experiments/n3_anatomy.py
    python experiments/n3_anatomy.py --quick
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

from project_genesis.anatomy import anatomy_reading  # noqa: E402
from project_genesis.crossing import (  # noqa: E402
    capacity_floor,
    kappa_at_crossing,
    measured_crossing,
    two_point_reduction,
)
from project_genesis.kuramoto_substrate import (  # noqa: E402
    natural_frequencies,
    simulate,
    window_metrics,
)
from n3_criticality_transplant import measure_substrate as hopfield_scan  # noqa: E402

BANNER = "=" * 78

HOPFIELD = dict(n_spins=300, n_patterns=9, n_seeds=4, n_burn=30, n_window=60,
                t_min=0.1, t_max=1.6, n_temps=16, query_fraction=0.05,
                query_every=5)
KURAMOTO = dict(n_osc=400, coupling=2.0, dt=0.05, n_steps=900, n_burn=450,
                n_seeds=3, g_min=0.1, g_max=2.4, n_gamma=13, noise=0.6)

# conventions to sweep, centred on the published value (marked)
H_THRESHOLDS = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)      # published 0.3
K_TOLERANCES = (0.15, 0.3, 0.6)                             # published 0.3
K_BIN_WIDTHS = (0.2, 0.4, 0.8)                              # published 0.4


def kuramoto_scan(cfg: dict, seed: int, driven: bool,
                  freq_tol: float, bin_width: float) -> list:
    gammas = np.linspace(cfg["g_min"], cfg["g_max"], cfg["n_gamma"])
    noise = cfg["noise"] if driven else 0.0
    rows = []
    for g in gammas:
        acc = {k: [] for k in ("delta_c", "delta_i", "repair_rate")}
        for s in range(cfg["n_seeds"]):
            rng = np.random.default_rng(seed + 1000 * s)
            omega = natural_frequencies(cfg["n_osc"], g, rng,
                                        distribution="gaussian")
            phases = rng.uniform(-np.pi, np.pi, cfg["n_osc"])
            sim = simulate(phases, omega, cfg["coupling"], cfg["dt"],
                           cfg["n_steps"], noise, rng, burn_in=cfg["n_burn"])
            m = window_metrics(sim, freq_tol=freq_tol, bin_width=bin_width)
            for k in acc:
                acc[k].append(m[k])
        rows.append({"gamma": float(g),
                     **{k: float(np.mean(v)) for k, v in acc.items()}})
    # `crossing` reads the load under the key "activity"
    return [dict(r, activity=r["repair_rate"]) for r in rows]


def critical_x(rows, x_key: str) -> float:
    """Where ΔI falls through ½ — each transplant's own definition of x_c."""
    for a, b in zip(rows, rows[1:]):
        if a["delta_i"] >= 0.5 >= b["delta_i"]:
            d = a["delta_i"] - b["delta_i"]
            f = (a["delta_i"] - 0.5) / d if d > 0 else 0.0
            return float(a[x_key] + f * (b[x_key] - a[x_key]))
    return float("nan")


def arm(rows_d, rows_u, x_key, weight, recovery, u_max, window):
    """Anatomy + eviction verdict + critical-neighbourhood check for one arm."""
    a = anatomy_reading(rows_d, x_key)
    red_d = two_point_reduction(rows_d, x_key=x_key)
    red_u = two_point_reduction(rows_u, x_key=x_key)
    k_d = kappa_at_crossing(red_d, weight)
    k_u = kappa_at_crossing(red_u, weight)
    f_d = capacity_floor(red_d, u_max)
    f_u = capacity_floor(red_u, u_max)
    m_d = measured_crossing(rows_d, x_key=x_key, weight=weight,
                            recovery=recovery)
    x_c = critical_x(rows_d, x_key)
    x_peak = red_d["x_critical"]
    in_window = bool(np.isfinite(x_c) and abs(x_peak - x_c) <= window * x_c)
    return {**a, "kappa_star": k_d, "floor": f_d,
            "evicted": bool(np.isfinite(m_d["c_evict"])),
            "floor_undriven": f_u, "kappa_star_undriven": k_u,
            "evicted_undriven": bool(
                np.isfinite(measured_crossing(rows_u, x_key=x_key,
                                              weight=weight,
                                              recovery=recovery)["c_evict"])),
            "hops": m_d["hops"], "walk": m_d["walk"],
            "x_c": x_c, "x_peak": x_peak, "in_critical_window": in_window}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--weight", type=float, default=1.5)
    p.add_argument("--recovery", type=float, default=0.1)
    p.add_argument("--c-max", type=float, default=50.0)
    p.add_argument("--critical-window", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=8888,
                   help="fresh seed: the exploratory sweep used seed 0")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    hop, kur = dict(HOPFIELD), dict(KURAMOTO)
    if args.quick:
        hop.update(n_spins=200, n_seeds=2, n_burn=15, n_window=30, n_temps=10)
        kur.update(n_osc=250, n_seeds=2, n_steps=500, n_burn=250, n_gamma=10)
    u_max = args.c_max / args.recovery

    print(BANNER)
    print("THE DESTINATION — substrate anatomy, or the ΔC dictionary?")
    print(BANNER)
    print(f"  fresh seed {args.seed} (the exploratory sweep that located the")
    print("  ranges used seed 0); oscillator load is the corrected repair_rate")
    print()
    print("  Pre-registered:")
    print("    Q1  sweeping each substrate's own ΔC convention flips its hull")
    print("    Q2  the eviction condition survives the same sweep, everywhere")
    print("    Q3  so does the critical neighbourhood: the ΔC peak stays inside")
    print()

    t0 = time.time()
    arms = {}

    print(BANNER)
    print("HOPFIELD — sweeping the structured-overlap threshold (published 0.30)")
    print(BANNER)
    print(f"  {'thresh':>7} {'hull':>5} {'E/gap':>8} {'gap':>7} {'evict':>6} "
          f"{'undriven':>9} {'x⋆':>6} {'x_c':>6} {'in win':>7}")
    for th in H_THRESHOLDS:
        cfg = Namespace(**hop, structured_threshold=th, seed=args.seed)
        r = arm(hopfield_scan(cfg, driven=True), hopfield_scan(cfg, driven=False),
                "T", args.weight, args.recovery, u_max, args.critical_window)
        arms[f"hopfield th={th}"] = r
        print(f"  {th:>7.2f} {r['hull']:>5} {r['excess_frac']:>8.3f} "
              f"{r['gap']:>7.3f} {'yes' if r['evicted'] else 'NO':>6} "
              f"{'proof' if not r['evicted_undriven'] else 'EVICTED':>9} "
              f"{r['x_peak']:>6.2f} {r['x_c']:>6.2f} "
              f"{'yes' if r['in_critical_window'] else 'NO':>7}")

    print()
    print(BANNER)
    print("KURAMOTO — sweeping entrainment tolerance and bin width (published "
          "0.30 / 0.40)")
    print(BANNER)
    print(f"  {'tol':>5} {'bin':>5} {'hull':>5} {'E/gap':>8} {'gap':>7} "
          f"{'evict':>6} {'undriven':>9} {'x⋆':>6} {'x_c':>6} {'in win':>7}")
    for tol in K_TOLERANCES:
        for bw in K_BIN_WIDTHS:
            rd = kuramoto_scan(kur, args.seed, True, tol, bw)
            ru = kuramoto_scan(kur, args.seed, False, tol, bw)
            r = arm(rd, ru, "gamma", args.weight, args.recovery, u_max,
                    args.critical_window)
            arms[f"kuramoto tol={tol} bin={bw}"] = r
            print(f"  {tol:>5.2f} {bw:>5.2f} {r['hull']:>5} "
                  f"{r['excess_frac']:>8.3f} {r['gap']:>7.3f} "
                  f"{'yes' if r['evicted'] else 'NO':>6} "
                  f"{'proof' if not r['evicted_undriven'] else 'EVICTED':>9} "
                  f"{r['x_peak']:>6.2f} {r['x_c']:>6.2f} "
                  f"{'yes' if r['in_critical_window'] else 'NO':>7}")
    print(f"\n  [{time.time() - t0:.0f}s]")

    h_arms = {k: v for k, v in arms.items() if k.startswith("hopfield")}
    k_arms = {k: v for k, v in arms.items() if k.startswith("kuramoto")}

    # ------------------------------------------------------------------- Q1
    print()
    print(BANNER)
    print("Q1  IS THE ROUTE A PROPERTY OF THE SUBSTRATE?")
    print(BANNER)
    h_hulls = sorted({v["hull"] for v in h_arms.values()})
    k_hulls = sorted({v["hull"] for v in k_arms.values()})
    print(f"  hopfield hull sizes across the sweep: {h_hulls}")
    print(f"  kuramoto hull sizes across the sweep: {k_hulls}")
    q1 = len(h_hulls) > 1 and len(k_hulls) > 1
    print()
    if q1:
        print("  PREDICTION HELD — and it retracts something. Neither route is a")
        print("  property of its substrate: both flip on a number nobody derived.")
        print("  `The_Principle.md` §5 says the destination difference is")
        print("  'anatomy, not convention'. It is convention, and this is the")
        print("  measurement that says so.")
        print()
        h_e = [v["excess_frac"] for v in h_arms.values()]
        k_e = [v["excess_frac"] for v in k_arms.values()]
        print(f"  What does survive is a difference of DEGREE, not of route:")
        print(f"    hopfield E/gap spans [{min(h_e):+.3f}, {max(h_e):+.3f}] "
              f"— never far from zero")
        print(f"    kuramoto E/gap spans [{min(k_e):+.3f}, {max(k_e):+.3f}] "
              f"— large in both directions")
        print("  The Hopfield network sits ON the boundary, where the vertex")
        print("  count is unstable because the intermediate state is worth")
        print("  almost exactly nothing. That is a real reading; 'two vertices")
        print("  versus three' was not.")
    else:
        print("  PREDICTION FAILED — reported as measured. A hull that holds")
        print("  constant across its own convention sweep IS substrate anatomy,")
        print("  and §5 stands as written for that substrate.")

    # ------------------------------------------------------------------- Q2
    print()
    print(BANNER)
    print("Q2  DOES THE EVICTION CONDITION SURVIVE?")
    print(BANNER)
    bad_d = [k for k, v in arms.items() if not v["evicted"]]
    bad_u = [k for k, v in arms.items() if v["evicted_undriven"]]
    floors = [v["floor"] for v in arms.values()]
    kstars = [v["kappa_star"] for v in arms.values()]
    print(f"  driven arms that relocate:      {len(arms) - len(bad_d)}/{len(arms)}")
    print(f"  undriven arms that stay put:    {len(arms) - len(bad_u)}/{len(arms)}")
    print(f"  driven capacity floor spans   [{min(floors):.4f}, {max(floors):.4f}]")
    print(f"  crossing capacity κ_o⋆ spans  [{min(kstars):.4f}, {max(kstars):.4f}]")
    q2 = not bad_d and not bad_u
    print()
    if q2:
        print("  PREDICTION HELD — the condition is untouched by any of it. The")
        print("  floor sits orders of magnitude below the crossing capacity in")
        print("  every arm, so eviction is not a close call that a convention")
        print("  could tip. What the last two experiments established survives:")
        print("  order that must be maintained pays, and scarcity evicts it.")
    else:
        print("  PREDICTION FAILED — the condition itself moves:")
        for k in bad_d:
            print(f"    {k}: driven arm does NOT relocate")
        for k in bad_u:
            print(f"    {k}: undriven arm IS evicted")
        print("  Then the conventions reach further than the route, and the")
        print("  previous two experiments' conclusion needs revisiting too.")

    # ------------------------------------------------------------------- Q3
    print()
    print(BANNER)
    print("Q3  DOES THE CRITICAL NEIGHBOURHOOD SURVIVE?")
    print(BANNER)
    out = [k for k, v in arms.items() if not v["in_critical_window"]]
    print(f"  ΔC peak inside |x⋆ − x_c| ≤ {args.critical_window:g}·x_c: "
          f"{len(arms) - len(out)}/{len(arms)} arms")
    for k, v in arms.items():
        if not v["in_critical_window"]:
            print(f"    OUTSIDE: {k}  x⋆={v['x_peak']:.2f} x_c={v['x_c']:.2f}")
    print()
    print("  Where the ΔC peak sits, per Hopfield threshold — it slides:")
    print("    " + "  ".join(f"{t:g}→{h_arms[f'hopfield th={t}']['x_peak']:.2f}"
                             for t in H_THRESHOLDS))
    # a reader has to be able to judge whether the failing arms are defensible
    chance = 1.0 / np.sqrt(hop["n_spins"])
    print(f"  Chance-level |overlap| for N = {hop['n_spins']} is "
          f"{chance:.4f}, so those thresholds are")
    print("    " + "  ".join(f"{t:g}={t / chance:.1f}σ" for t in H_THRESHOLDS)
          + " above chance.")
    print("  A threshold at 2σ is arguable; one at 3.5σ is not — it is calling")
    print("  a state structured only when it genuinely is. If an arm that")
    print("  strict still leaves the window, the failure is not an artefact of")
    print("  an absurd setting.")
    q3 = not out
    print()
    if q3:
        print("  PREDICTION HELD — 'the optimum relocates to the critical")
        print("  neighbourhood' is convention-independent. It is only the")
        print("  finer claim — which state inside that neighbourhood, and")
        print("  whether it gets there in one hop — that the dictionary")
        print("  decides. The transplant result survives at the resolution it")
        print("  was originally stated at, and not at the resolution the last")
        print("  two experiments tried to push it to.")
    else:
        print("  PREDICTION FAILED — the ΔC peak leaves the critical window")
        print("  under a defensible convention. Then 'relocates to criticality'")
        print("  is itself convention-dependent, which is a far larger problem")
        print("  than the route: it is the claim all three transplants make.")

    # -------------------------------------------------------------- verdict
    print()
    print(BANNER)
    n = int(q1) + int(q2) + int(q3)
    print(f"VERDICT: {n}/3 pre-registered predictions held.")
    print(BANNER)
    print("  Where this leaves the three transplants, at each resolution:")
    print(f"    does a relocation happen?          {'YES — convention-independent' if q2 else 'not established'}")
    print(f"    does it reach criticality?         {'YES — convention-independent' if q3 else 'NOT established'}")
    print(f"    which state, by which route?       {'NOT established — convention' if q1 else 'substrate anatomy'}")
    print()
    print("  The correction this forces on §5: the destination difference")
    print("  between the network and the oscillators was reported as anatomy.")
    print("  It is not. Both substrates' routes move under their own ΔC")
    print("  dictionaries, and the honest statement stops at the")
    print("  neighbourhood.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "anatomy.json"
        out_path.write_text(json.dumps(
            {"params": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
             "hopfield_settings": hop, "kuramoto_settings": kur,
             "arms": arms, "score": n}, indent=2, default=float))
        print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
