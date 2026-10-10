"""Does a wall make room for a new distinction?

Start from the simplest partition: two sectors, A and B, relaxed into domains.
Add a third sector C everywhere at a vanishing amplitude, let it be *free*
(non-conserved, so its total can grow) while A and B stay volume-conserved, and
couple it to both with ``γ_AC = γ_BC = g`` while ``γ_AB`` keeps the published
``1.5``. Then ask where — if anywhere — C switches on.

The linear argument (`project_genesis.wetting`): a dormant sector grows at rate
``1 − 2 Σ γ_cb f_b²``. In bulk A that is ``1 − 2g``, so C is held at zero in
the bulk exactly when ``g > 1/2``. In an A|B wall neither A nor B is fully on,
so the same suppression is weaker. The wall is where a distinction the bulk
forbids is least forbidden.

Seeding is **volume-filling** (each sector's conserved mean is ``1/P``). The
published seeding (noise on ``[0, 0.1]``) conserves a mean of ``0.05``, too
little to tile space at ``+1``; the field makes up the balance with sectors at
``−1``, about 40% of the box, which ``argmax`` labels from near-zero residuals.
Here that would put C next to ``f = 0`` regions that have nothing to do with
walls, so it would measure the seeding instead of the wall.

Pre-registered predictions (``g_lin`` is computed from the relaxed A|B field
before C is added: the ``g`` at which the linear growth rate at the most
favourable wall cell crosses zero):

W1. **Bulk invasion below 1/2.** For ``g < 0.5`` C takes over bulk as well as
    walls: its ``+1`` cover exceeds the initial wall fraction several times
    over. Falsifier: C stays confined at ``g < 0.5``.

W2. **A wall-only window.** For some ``0.5 < g < g_lin``, C grows **and** is
    concentrated where the A|B walls were: wall enrichment ``> 3`` and A–B
    direct contact reduced by more than half. Falsifier: no ``g`` gives growth
    confined to walls — the wall does not make room.

W3. **Dormant at the published coupling.** At ``g = 1.5`` (every wall the same
    cost, the published model) C never exceeds ``0.1`` anywhere.

W4. **Growth reaches the d+1 palette — or wetting forbids it.** Where C covers
    the A|B walls *completely*, A and B never meet, and a three-sector point
    junction cannot form; where it covers them *partially*, it can. So: in the
    W2 window, at least one ``g`` yields a non-zero three-sector junction
    density (``P = 3``, ``m = 3``), and junction density falls as A–B contact
    falls to zero. Falsifier: junctions absent throughout the window.

    python experiments/n3_wetting.py
    python experiments/n3_wetting.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_genesis.junction_scale import (  # noqa: E402
    full_palette_density, neighbourhood_label_bits,
)
from project_genesis.wetting import (  # noqa: E402
    coupling_matrix, growth_rate, step_multiphase_pairwise,
)

A, B, C = 0, 1, 2


def relaxed_pair(size, steps, seed):
    """Two sectors, volume-filling seed, published coupling."""
    rng = np.random.default_rng(seed)
    f = rng.random((2, size, size)) * 1.0  # mean 1/2 each
    g = coupling_matrix(2, 1.5)
    for _ in range(steps):
        f = step_multiphase_pairwise(f, g)
    return f


def ab_wall(labels):
    """Cells whose 3x3 neighbourhood holds both A and B."""
    bits = neighbourhood_label_bits(labels)
    return (bits & 0b011) == 0b011


def linear_threshold(pair):
    """``g`` at which C's linear growth rate at the best wall cell is zero."""
    s = (pair ** 2).sum(0)
    return float(1.0 / (2.0 * s.min()))


def grow(pair, g, steps, seed, amp=0.01, early=150):
    """Grow C for ``steps``; also return a snapshot at step ``early``."""
    rng = np.random.default_rng(seed + 7)
    c = rng.random(pair.shape[1:]) * amp
    f = np.concatenate([pair, c[None]], axis=0)
    gam = coupling_matrix(3, 1.5, {(A, C): g, (B, C): g})
    snap = None
    for t in range(1, steps + 1):
        f = step_multiphase_pairwise(f, gam, conserved=(True, True, False))
        if t == early:
            snap = f.copy()
    return f, snap


def onset_enrichment(pair, snap):
    """Where C *starts*: its share on the initial A|B walls at an early step.

    Read before C saturates, so it measures where growth begins rather than
    where the finished layer ends up. Ratio of mean ``|f_C|`` on the initial
    walls to mean ``|f_C|`` off them.
    """
    wall0 = ab_wall(np.argmax(pair, 0))
    c = np.abs(snap[C])
    return float(c[wall0].mean() / c[~wall0].mean())


def measure(pair, f, snap=None):
    lab0 = np.argmax(pair, 0)
    wall0 = ab_wall(lab0)
    lab = np.argmax(f, 0)
    c_on = f[C] > 0.5
    cpos = np.clip(f[C], 0.0, None)
    total = cpos.sum()
    enrich = (cpos[wall0].sum() / total) / wall0.mean() if total > 1e-9 else float("nan")
    contact = ab_wall(lab).mean() / wall0.mean()
    return {
        "c_max": float(f[C].max()),
        "c_cover": float(c_on.mean()),
        "wall_frac": float(wall0.mean()),
        "enrich": float(enrich),
        "ab_contact": float(contact),
        "tj": float(full_palette_density(lab, 3, min_valence=3)),
        "onset": onset_enrichment(pair, snap) if snap is not None else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    size = 64 if args.quick else 96
    trials = 1 if args.quick else 3
    pre, post = 600, (1000 if args.quick else 2000)
    gs = [0.3, 0.45, 0.55, 0.65, 0.75, 0.9, 1.1, 1.5]
    fine = [0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62]  # post hoc, located from --quick

    pairs = [relaxed_pair(size, pre, seed=s) for s in range(trials)]
    g_lin = float(np.mean([linear_threshold(p) for p in pairs]))
    print(f"A|B field {size}^2, {trials} seeds; wall fraction "
          f"{np.mean([ab_wall(np.argmax(p, 0)).mean() for p in pairs]):.3f}; "
          f"linear wall threshold g_lin = {g_lin:.3f}")
    print(f"{'g':>5} {'C max':>7} {'C cover':>8} {'enrich':>7} {'A-B contact':>12} "
          f"{'3-junction':>11} {'onset':>7}")
    rows = {}
    for g in sorted(set(gs) | set(fine)):
        ms = [measure(p, *grow(p, g, post, seed=s)) for s, p in enumerate(pairs)]
        r = {k: float(np.nanmean([m[k] for m in ms])) if any(
            np.isfinite(m[k]) for m in ms) else float("nan") for k in ms[0]}
        rows[g] = r
        tag = "" if g in gs else "  (post hoc)"
        print(f"{g:5.2f} {r['c_max']:7.3f} {r['c_cover']:8.3f} {r['enrich']:7.2f} "
              f"{r['ab_contact']:12.3f} {r['tj']:11.5f} {r['onset']:7.2f}{tag}")
    base = rows[1.5]["ab_contact"]
    print(f"\nno-C control (g = 1.5): A-B contact {base:.3f} of the initial walls; "
          f"contact below is relative to that")
    for g in sorted(rows):
        if rows[g]["c_max"] > 0.5:
            print(f"  g={g:.2f}: contact vs control {rows[g]['ab_contact'] / base:.3f}")

    wf = rows[gs[0]]["wall_frac"]
    window = [g for g in gs if 0.5 < g < g_lin]
    grew = [g for g in window if rows[g]["c_max"] > 0.5]
    wall_only = [g for g in grew if rows[g]["enrich"] > 3 and rows[g]["ab_contact"] < 0.5]
    verdicts = [
        ("W1 bulk invasion below 1/2",
         all(rows[g]["c_cover"] > 3 * wf for g in gs if g < 0.5)),
        ("W2 wall-only growth window", bool(wall_only)),
        ("W3 dormant at g = 1.5", rows[1.5]["c_max"] < 0.1),
        ("W4 three-sector junctions in window",
         any(rows[g]["tj"] > 0 for g in grew)),
    ]
    print(f"\nwindow (0.5, g_lin) tested: {window}; C grew at {grew}; "
          f"wall-confined at {wall_only}")
    print("verdicts")
    for name, ok in verdicts:
        print(f"  {name}: {'HOLDS' if ok else 'FAILS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
