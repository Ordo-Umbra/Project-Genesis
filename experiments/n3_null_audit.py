"""Null audit: what does a field with no URP in it score on URP's measured claims?

`CLAUDE.md` principle 4 asks for a structureless baseline before a reading is
credited to the dynamics. It was written after most of the `[measured]` claims
in `Docs/The_Principle.md` existed, and only one of them, the d+1 junction
selection, has been checked against a baseline since (`n3_junction_null.py`: a
random Voronoi tessellation selects d+1 with no dynamics at all). This
experiment asks the same question of the other claims that can be run cheaply.

Each arm feeds a baseline with no capacity field, no S-functional and no
relaxation into the published instrument, unchanged, and asks whether it
reproduces the published reading. A reading the baseline reproduces is
consistent with URP but is not evidence for it over any other account that
produces a local field. A reading the baseline fails to reproduce is evidence.

**Disclosure, written before anything here runs.** Two arms are not fresh
predictions. A1b's numbers follow from figures `n3_junction_null.py` already
prints: its any-junction and full-palette rows give φ directly. A2a repeats a
post hoc check from 2026-10-28, recorded in `Docs/Session_Log.md`. Both are kept
so that the audit is complete, and both are labelled. A1a, A1c, A2b, A2c, A3 and
A4 are written before they are run. The commit holding this docstring precedes
the commit holding the results.

A1. **The cliff** (§2). φ = I/C, the share of 3+-sector junctions that carry the
    whole palette, is 1 at P = 3 and collapses at P ≥ 4. Read with
    `n3_capacity_separation`'s exact definitions.
    A1a. Pure noise labels, P = 3, in 2-D (72²) and 3-D (28³). Prediction:
         φ = 1 exactly. Reason: a cell that sees 3 or more of 3 sectors sees
         all of them, so φ(3) = 1 is an identity for any labelling whatever.
         Falsifier: any other value.
    A1b. (Follows from N1/N2, see disclosure.) Voronoi, 2-D, 160², 70 seeds,
         P = 3..6, 3 trials. Prediction: φ(3) = 1 and every φ(P ≥ 4) < 0.05.
    A1c. Pure noise labels, P = 4, 2-D. Prediction: φ > 0.5 (inclusion-exclusion
         on a 3×3 window gives about 0.72). Reading if it holds: the collapse at
         P ≥ 4 needs a tessellation, but any tessellation will do. Falsifier:
         φ ≤ 0.5.

A2. **The gap** (§2). Distinction is volume-law, integration area-law,
    n_C − n_I = 1, unmoved by capacity. Same instrument, defaults and bias
    calibration as `n3_area_law` (n = 160, halves 22–44, 48 realisations,
    tolerance 0.25 on each exponent). Mechanism, stated in advance: C(ℓ) sums a
    stationary density over the region, so its expectation is exactly ∝ ℓ²;
    I(ℓ) convolves a translation-invariant kernel with the region's edge, so it
    is ∝ ℓ once ℓ ≫ ξ. On this reading the gap is a property of the instrument
    on any stationary, short-range ensemble.
    A2a. (Replication, see disclosure.) Gaussian fields, ξ = 1.5, 3, 6.
         Prediction: |gap − 1| < 0.25 at every ξ.
    A2b. Random Voronoi, P = 3, one-hot components smoothed by a Gaussian of
         width σ (a wall needs a width for a gradient to exist). Cell widths
         6, 10 and 16 at σ = 1, plus σ = 2 at width 10, because σ is a constant
         chosen here and should be swept. Prediction: |gap − 1| < 0.25 in every
         arm, and the gap's spread across the four arms < 0.25.
    A2c. The published field (`n3_area_law.grow`, consumption c = 1), read by
         this same code. Prediction: its raw gap lies within 0.25 of the mean
         raw gap of the A2b arms. Falsifier for all of A2: the field's gap
         differs from the structureless ones by more than the published
         tolerance, in which case it carries something locality does not.

A3. **The radius sweep** (§3, described there as "the only claim in this
    programme to come out of a convention sweep wider than it went in").
    Published relaxed 2-D margins are ∞, 45.5 and 16.6 at radius 1, 2 and 3,
    with argmax P = 3 throughout. Voronoi, 2-D, 160², 70 seeds, m = 3,
    P = 2..6, 3 trials, the same tessellations read at every radius.
    Prediction: argmax = 3 at every radius. Falsifier: any other argmax, in
    which case surviving the wider probe is something the relaxation adds, and
    §3's reading of it stands.

A4. **The eviction condition** (§5): "evictable ⟺ the ordered point starts
    ahead, and holding it costs capacity", 16/16 driven arms evictable, 0/16
    undriven, agreeing with the ladder verdict on all 16. The transplants apply
    the capacity law post hoc to measured curves, so the verdict is a function
    of the curves alone. Feed it random curves: 9 scan points, ΔC and ΔI uniform
    on [0, 1], loads uniform on [0.01, 1]. Keep a scan only if it meets the
    transplants' design premises: the ordered point (index 0) is the S argmax
    at c = 0, and it is not the ΔC peak. "Driven": every load random. "Free":
    the ordered load set to 0. 1000 scans each, w = 1, r = 0.1, the published
    ladder (`crossing.measured_crossing` defaults).
    Predictions: `evictable` agrees with the ladder verdict on ≥ 99% of scans;
    ≥ 99% of driven scans are evicted; ≤ 1% of free scans are evicted.
    Reason: under the premises, as c grows every driven κ → 0 and S → ΔC, whose
    argmax is not the ordered point; a free ordered point keeps κ = 1 while
    every rival's S only falls. Reading if they hold: the toggle the
    transplants report is guaranteed by the functional and the premises, so its
    reproduction on three substrates is not evidence that the substrates share
    anything. What is empirical is whether each substrate meets the premises.
    Falsifier: any rate on the wrong side of its threshold.

Not run here, and recorded in `Docs/Null_Audit.md` with the reason: claims whose
null is a textbook result (nematic defect statistics, the Yukawa well of the
screened κ equation, Euler's formula on a trivalent tiling), and claims that
already carry an external baseline (continual learning against SGD and replay).

    python experiments/n3_null_audit.py           # ~4 min
    python experiments/n3_null_audit.py --quick   # smoke run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from n3_area_law import grow  # noqa: E402
from n3_junction_null import voronoi_labels  # noqa: E402
from project_genesis.area_law import (  # noqa: E402
    cross_boundary_mi, ensemble_correlation, gaussian_control_field,
    mi_kernel, region_content, scaling_exponent,
)
from project_genesis.crossing import (  # noqa: E402
    evictable, measured_crossing, two_point_reduction,
)
from project_genesis.junction_scale import (  # noqa: E402
    full_palette_density, selection_profile,
)
from project_genesis.multiphase import (  # noqa: E402
    _total_gradient_energy, count_triple_junctions,
    full_palette_junction_density,
)

BANNER = "=" * 78
N = 160                       # n3_area_law defaults
HALVES = [22, 28, 36, 44]
TOL = 0.25                    # n3_area_law's tolerance on each exponent
AREA = 1.0


# --- A1: the cliff ---------------------------------------------------------

def integrated_fraction(labels: np.ndarray, n_palette: int) -> float:
    """φ = I/C exactly as `n3_capacity_separation.distinction_integration`."""
    c = count_triple_junctions(labels) / labels.size
    i = full_palette_junction_density(labels, n_palette)
    return float(i / c) if c > 0 else 0.0


def arm_a1(trials, rng):
    noise3 = {"2-D": integrated_fraction(rng.integers(0, 3, (72, 72)), 3),
              "3-D": integrated_fraction(rng.integers(0, 3, (28, 28, 28)), 3)}
    noise4 = integrated_fraction(rng.integers(0, 4, (72, 72)), 4)
    vor = {}
    for P in (3, 4, 5, 6):
        vor[P] = float(np.mean([integrated_fraction(
            voronoi_labels(160, 2, 70, P, rng), P) for _ in range(trials)]))
    return {"noise3": noise3, "noise4": noise4, "voronoi": vor}


# --- A2: the gap -----------------------------------------------------------

def smoothed_voronoi(n, width, sigma, n_palette, rng):
    """One-hot Voronoi components, each blurred by a periodic Gaussian."""
    n_seeds = max(n_palette, int(round((n / width) ** 2)))
    lab = voronoi_labels(n, 2, n_seeds, n_palette, rng)
    y, x = np.indices((n, n))
    dx, dy = np.minimum(x, n - x), np.minimum(y, n - y)
    ker = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))
    kf = np.fft.rfft2(ker / ker.sum())
    return np.stack([np.fft.irfft2(np.fft.rfft2((lab == a).astype(float)) * kf,
                                   s=(n, n)) for a in range(n_palette)])


def calibrate_bias(n_real, seed=0):
    """`n3_area_law.a1_calibrate`: the MI estimator's bias on its control."""
    rng = np.random.default_rng(seed)
    biases = []
    for xi in (1.5, 3.0):
        g = ensemble_correlation([gaussian_control_field(N, xi, rng)
                                  for _ in range(n_real)])
        kernel, _ = mi_kernel(g)
        biases.append(scaling_exponent(
            HALVES, [cross_boundary_mi(kernel, h) for h in HALVES]) - AREA)
    return float(np.mean(biases))


def gap_reading(realisations, bias):
    """n_C, n_I and the gap, read exactly as `n3_area_law.a2_a3_sweep` does."""
    comps, c_acc = [], np.zeros(len(HALVES))
    for fields in realisations:
        comps.extend(list(fields))
        grad = _total_gradient_energy(fields)
        for i, h in enumerate(HALVES):
            c_acc[i] += region_content(grad, h) / len(realisations)
    kernel, _ = mi_kernel(ensemble_correlation(comps))
    n_c = scaling_exponent(HALVES, c_acc)
    n_i_raw = scaling_exponent(HALVES, [cross_boundary_mi(kernel, h)
                                        for h in HALVES])
    return {"n_C": n_c, "n_I": n_i_raw - bias, "gap": n_c - (n_i_raw - bias),
            "raw_gap": n_c - n_i_raw}


def arm_a2(n_real, rng, field_seed=0):
    bias = calibrate_bias(n_real)
    gauss = {xi: gap_reading([gaussian_control_field(N, xi, rng)[None]
                              for _ in range(n_real)], bias)
             for xi in (1.5, 3.0, 6.0)}
    vor = {}
    for width, sigma in ((6, 1.0), (10, 1.0), (16, 1.0), (10, 2.0)):
        vor[(width, sigma)] = gap_reading(
            [smoothed_voronoi(N, width, sigma, 3, rng) for _ in range(n_real)],
            bias)
    field = gap_reading([grow(N, 1.0, 600, field_seed + 7 * s)[0]
                         for s in range(n_real)], bias)
    return {"bias": bias, "gauss": gauss, "voronoi": vor, "field": field}


# --- A3: the radius sweep --------------------------------------------------

PUBLISHED_MARGINS = {1: float("inf"), 2: 45.5, 3: 16.6}


def arm_a3(trials, rng):
    palettes, radii = (2, 3, 4, 5, 6), (1, 2, 3)
    dens = {r: {P: 0.0 for P in palettes} for r in radii}
    for P in palettes:
        for _ in range(trials):
            lab = voronoi_labels(160, 2, 70, P, rng)
            for r in radii:
                dens[r][P] += full_palette_density(lab, P, radius=r) / trials
    return {r: {"densities": dens[r], **selection_profile(dens[r])}
            for r in radii}


# --- A4: the eviction condition --------------------------------------------

def random_scan(rng, *, free, weight, n_points=9):
    """Random curves meeting the transplants' two design premises."""
    while True:
        dc, di = rng.random(n_points), rng.random(n_points)
        load = rng.uniform(0.01, 1.0, n_points)
        if free:
            load[0] = 0.0
        if np.argmax(dc + weight * di) == 0 and np.argmax(dc) != 0:
            return [{"delta_c": float(dc[i]), "delta_i": float(di[i]),
                     "activity": float(load[i]), "x": float(i)}
                    for i in range(n_points)]


def eviction_verdicts(rows, weight, recovery=0.1):
    """(`evictable`, the ladder's verdict) for one scan."""
    red = two_point_reduction(rows, x_key="x")
    ladder = measured_crossing(rows, x_key="x", weight=weight,
                               recovery=recovery)
    return evictable(red, weight), bool(np.isfinite(ladder["c_evict"]))


def arm_a4(n_scans, rng, weight=1.0):
    out = {}
    for name, free in (("driven", False), ("free", True)):
        pairs = [eviction_verdicts(random_scan(rng, free=free, weight=weight),
                                   weight) for _ in range(n_scans)]
        out[name] = {"agree": float(np.mean([a == b for a, b in pairs])),
                     "evicted": float(np.mean([b for _, b in pairs])),
                     "n": n_scans}
    return out


# --- report ----------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    trials = 1 if args.quick else 3
    n_real = 12 if args.quick else 48
    n_scans = 100 if args.quick else 1000
    rng = np.random.default_rng(args.seed)
    verdicts = []

    print(BANNER + "\nA1  the cliff: φ = I/C on structureless labels\n" + BANNER)
    a1 = arm_a1(trials, rng)
    for d, phi in a1["noise3"].items():
        print(f"  noise, P=3, {d}: φ = {phi:.6f}")
    print(f"  noise, P=4, 2-D: φ = {a1['noise4']:.4f}")
    print("  Voronoi 2-D   : " + "  ".join(f"P={P}: {v:.4f}"
                                           for P, v in a1["voronoi"].items()))
    verdicts.append(("A1a", "φ(3) = 1 on pure noise, 2-D and 3-D",
                     all(v == 1.0 for v in a1["noise3"].values())))
    verdicts.append(("A1b", "Voronoi φ(3) = 1, φ(P≥4) < 0.05 (follows from N1/N2)",
                     a1["voronoi"][3] == 1.0
                     and max(a1["voronoi"][P] for P in (4, 5, 6)) < 0.05))
    verdicts.append(("A1c", "noise φ(4) > 0.5", a1["noise4"] > 0.5))

    print(BANNER + "\nA2  the gap: n_C − n_I on structureless fields\n" + BANNER)
    a2 = arm_a2(n_real, rng)
    print(f"  bias calibrated on the Gaussian control: {a2['bias']:+.3f}"
          f"  (published +0.12)")

    def row(label, r):
        print(f"  {label:28s} n_C={r['n_C']:.3f}  n_I={r['n_I']:.3f}  "
              f"gap={r['gap']:.3f}  raw gap={r['raw_gap']:.3f}")

    for xi, r in a2["gauss"].items():
        row(f"Gaussian ξ={xi}", r)
    for (w, s), r in a2["voronoi"].items():
        row(f"Voronoi width {w}, σ={s}", r)
    row("published field, c=1", a2["field"])
    vgaps = [r["gap"] for r in a2["voronoi"].values()]
    vraw = float(np.mean([r["raw_gap"] for r in a2["voronoi"].values()]))
    verdicts.append(("A2a", "Gaussian |gap − 1| < 0.25 (replication)",
                     all(abs(r["gap"] - 1) < TOL for r in a2["gauss"].values())))
    verdicts.append(("A2b", "Voronoi |gap − 1| < 0.25, spread < 0.25",
                     all(abs(g - 1) < TOL for g in vgaps)
                     and max(vgaps) - min(vgaps) < TOL))
    verdicts.append(("A2c", "field raw gap within 0.25 of Voronoi mean",
                     abs(a2["field"]["raw_gap"] - vraw) < TOL))

    print(BANNER + "\nA3  the radius sweep, Voronoi 2-D, m = 3\n" + BANNER)
    a3 = arm_a3(trials, rng)
    for r, prof in a3.items():
        dens = "  ".join(f"P={P}:{v:.4f}" for P, v in prof["densities"].items())
        print(f"  radius {r}: {dens}   argmax={prof['argmax']}  "
              f"margin={prof['margin']:.1f}  (relaxed, published: "
              f"{PUBLISHED_MARGINS[r]:.1f})")
    verdicts.append(("A3", "Voronoi argmax = 3 at radius 1, 2, 3",
                     all(prof["argmax"] == 3 for prof in a3.values())))

    print(BANNER + "\nA4  the eviction condition on random curves\n" + BANNER)
    a4 = arm_a4(n_scans, rng)
    for name, r in a4.items():
        print(f"  {name:6s} ({r['n']} scans): evictable agrees with the ladder "
              f"{r['agree']:.1%}, ladder evicts {r['evicted']:.1%}")
    verdicts.append(("A4", "agreement ≥ 99%, driven evicted ≥ 99%, free ≤ 1%",
                     min(a4["driven"]["agree"], a4["free"]["agree"]) >= 0.99
                     and a4["driven"]["evicted"] >= 0.99
                     and a4["free"]["evicted"] <= 0.01))

    print(BANNER + "\nverdicts (pre-registered in the docstring)\n" + BANNER)
    for name, text, ok in verdicts:
        print(f"  {name:4s} {'HOLDS' if ok else 'FAILS'}  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
