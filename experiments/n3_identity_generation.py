"""Identity generation: sameness measured from structure, not assigned.

The exclusion series' top follow-up: the labelled load PRICES identity
once told what is identical — it does not GENERATE the identities.  This
experiment removes the assignment.  Structures here are **clusters**: each
mass's load is its own internal arrangement of sub-blobs (seeded, distinct
— the structure's distinction pattern, and the thing that gravitates), and
the corpus machinery is the identity registry: patches are certified by
``MemoryCorpus.add_if_stable``, clones are ``copy_subfield`` (lineage
kept), composites are the corpus's own ``compose``.

Sameness is then a **measurement**, and the measure is the arc's own
min-construction one level down:

    φ(A, B) = Σ min(p_A, p_B) ,   p = the structure's normalized pattern —

the duplicated fraction of the two internal distinction patterns.  φ = 1
iff the patterns are identical, → 0 as their supports separate.  The
measured φ routes the exclusion term through the existing shared-fraction
machinery (`shared_fraction_labels` + `exclusion_gap_labelled`): min on
patterns *generates* identity; identity routes min on loads.  One
principle, two levels.

Pre-registered predictions (statics; Part III's L2 showed the φ-routed
dynamics follow the statics bitwise, so the statics carry the physics):

Y1. **Identity is measured, and routes exclusion**: the lineage clone
    pair measures φ ≥ 0.99 and carries an exclusion floor (interior s\*,
    barrier ≥ 0.15); the independently-formed pair measures φ ≤ 0.8 and
    its barrier is ≤ 0.5× the clone pair's — sameness read off the
    structures orders the exclusion, with nothing assigned.
Y2. **Composition interpolates**: the corpus-composed patch measures
    φ to each parent strictly between the independent pair's φ and 1,
    and its barrier (paired against parent A) sits strictly between the
    independent and clone barriers (margins ≥ 0.02).
Y3. **The individuality threshold**: geometrically perturbing a clone
    (sub-blob jitter η) drives φ monotonically down, and the floor
    releases — the barrier falls below the Y1 floor bar at a measurable
    η\* within the scan.  The framework answers "how much must a clone
    change to become an individual" with a number.

Usage::

    python experiments/n3_identity_generation.py --output-dir artifacts/n3_identity
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import (
    capacity_free_energy,
    gaussian_load,
    relax_capacity,
)
from project_genesis.capacity_waves import (
    exclusion_gap_labelled,
    shared_fraction_labels,
)
from project_genesis.memory_corpus import MemoryCorpus


def field_kwargs(args):
    return dict(kappa_diffusion=1.0, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0)


def make_cluster(args, seed: int, jitter: float = 0.0,
                 jitter_seed: int = 0) -> np.ndarray:
    """A structure's internal pattern: seeded sub-blobs in a k×k patch."""
    k = args.patch
    rng = np.random.default_rng(seed)
    centers = rng.uniform(k / 2 - args.spread, k / 2 + args.spread,
                          size=(args.n_sub, 2))
    if jitter > 0.0:
        jrng = np.random.default_rng(1000 + jitter_seed)
        centers = centers + jitter * jrng.standard_normal(centers.shape)
    patch = np.zeros((k, k))
    for cx, cy in centers:
        patch += gaussian_load((k, k), [(cx, cy)], args.sub_width,
                               args.mass / args.n_sub)
    return patch


def phi_measured(pa: np.ndarray, pb: np.ndarray) -> float:
    """Sameness: the duplicated fraction of the normalized patterns."""
    a = pa / pa.sum()
    b = pb / pb.sum()
    return float(np.minimum(a, b).sum())


def place(args, patch: np.ndarray, center) -> np.ndarray:
    """Embed a k×k patch into the periodic box at ``center`` (cell-aligned)."""
    n, k = int(args.size), args.patch
    out = np.zeros((n, n))
    x0 = int(round(center[0])) - k // 2
    y0 = int(round(center[1])) - k // 2
    for i in range(k):
        for j in range(k):
            out[(x0 + i) % n, (y0 + j) % n] = patch[i, j]
    return out


def pair_statics(args, patch_a: np.ndarray, patch_b: np.ndarray,
                 phi: float) -> dict:
    """E(s) ladder for two structures with measured shared fraction φ."""
    n = int(args.size)
    fkw = field_kwargs(args)
    labels = shared_fraction_labels(phi)
    es = []
    for s in args.seps:
        l1 = place(args, patch_a, (n / 2 - s / 2, n / 2))
        l2 = place(args, patch_b, (n / 2 + s / 2, n / 2))
        tot = l1 + l2
        kappa = relax_capacity(tot, **fkw)
        grav = capacity_free_energy(kappa, tot, **fkw)
        ex = exclusion_gap_labelled(l1, l2, *labels, **fkw) if phi > 0 \
            else 0.0
        es.append(grav + ex)
    es = np.asarray(es)
    rel = es - es[-1]
    i_min = int(np.argmin(rel))
    interior = 0 < i_min < len(args.seps) - 1
    return {"curve": rel.tolist(), "s_star": float(args.seps[i_min]),
            "interior": bool(interior),
            "barrier": float(rel[0] - rel[i_min])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=float, default=96.0)
    p.add_argument("--patch", type=int, default=24)
    p.add_argument("--n-sub", type=int, default=5)
    p.add_argument("--spread", type=float, default=4.0)
    p.add_argument("--sub-width", type=float, default=1.2)
    p.add_argument("--mass", type=float, default=0.6)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--seps", type=float, nargs="+",
                   default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
                            12.0, 16.0])
    p.add_argument("--jitters", type=float, nargs="+",
                   default=[0.5, 1.0, 2.0, 3.0, 4.0])
    p.add_argument("--output-dir", type=str,
                   default="artifacts/n3_identity")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.seps = [1.0, 3.0, 6.0, 8.0, 10.0, 16.0]
        args.jitters = [1.0, 3.0]

    os.makedirs(args.output_dir, exist_ok=True)
    print("identity generation: sameness measured, not assigned", flush=True)

    # --- the identity registry: corpus-certified structures
    corpus = MemoryCorpus(min_stability=1, min_local_s=0.0)
    patch_a = make_cluster(args, seed=7)
    patch_b = make_cluster(args, seed=23)
    vox = np.zeros_like(patch_a, dtype=int)
    obj_a = corpus.add_if_stable(patch_a, vox, 1.0, 10)
    obj_b = corpus.add_if_stable(patch_b, vox, 1.0, 10)
    clone = obj_a.copy_subfield()                 # lineage clone of A
    mix = corpus.compose(obj_a, obj_b,
                         rng=np.random.default_rng(5))
    assert obj_a is not None and obj_b is not None and mix is not None
    print(f"  corpus: {len(corpus)} certified structures; composite "
          f"parents = {mix.parent_ids == [obj_a.object_id, obj_b.object_id]}",
          flush=True)

    # --- measured identities
    phi_clone = phi_measured(patch_a, clone)
    phi_indep = phi_measured(patch_a, patch_b)
    phi_mix_a = phi_measured(mix.subfield, patch_a)
    phi_mix_b = phi_measured(mix.subfield, patch_b)
    print(f"  φ(clone) = {phi_clone:.4f}  φ(independent) = {phi_indep:.4f}  "
          f"φ(mix,A) = {phi_mix_a:.4f}  φ(mix,B) = {phi_mix_b:.4f}",
          flush=True)

    # --- statics at the measured identities
    st_clone = pair_statics(args, patch_a, clone, phi_clone)
    st_indep = pair_statics(args, patch_a, patch_b, phi_indep)
    st_mix = pair_statics(args, mix.subfield, patch_a, phi_mix_a)
    for name, st in (("clone", st_clone), ("indep", st_indep),
                     ("mix", st_mix)):
        print(f"  {name:>6}: s* = {st['s_star']:g} (interior "
              f"{st['interior']}), barrier = {st['barrier']:.3f}",
              flush=True)

    y1 = (phi_clone >= 0.99 and st_clone["interior"]
          and st_clone["barrier"] >= 0.15
          and phi_indep <= 0.8
          and st_indep["barrier"] <= 0.5 * st_clone["barrier"])
    y2 = (phi_indep + 0.02 < phi_mix_a < 0.98
          and phi_indep + 0.02 < phi_mix_b < 0.98
          and st_indep["barrier"] + 0.02 < st_mix["barrier"]
          < st_clone["barrier"] - 0.02)

    # --- Y3: the individuality threshold
    scan = []
    for i, eta in enumerate(args.jitters):
        drifted = make_cluster(args, seed=7, jitter=eta, jitter_seed=1)
        phi_eta = phi_measured(patch_a, drifted)
        st = pair_statics(args, patch_a, drifted, phi_eta)
        scan.append({"eta": eta, "phi": phi_eta,
                     "barrier": st["barrier"]})
        print(f"  η = {eta:g}: φ = {phi_eta:.4f}, barrier = "
              f"{st['barrier']:.3f}", flush=True)
    phis = [s["phi"] for s in scan]
    bars = [s["barrier"] for s in scan]
    mono = all(b <= a + 1e-9 for a, b in zip(phis, phis[1:]))
    released = [s["eta"] for s in scan if s["barrier"] < 0.15]
    eta_star = min(released) if released else None
    y3 = mono and eta_star is not None

    lines = ["Identity generation — verdict", "=" * 74, ""]
    lines.append(
        f"Y1 (identity measured, and it routes exclusion): φ(clone) = "
        f"{phi_clone:.3f} → barrier {st_clone['barrier']:.3f} (interior "
        f"s* = {st_clone['s_star']:g}); φ(independent) = {phi_indep:.3f} "
        f"→ barrier {st_indep['barrier']:.3f} — "
        + ("✓ sameness read off the structures themselves orders the "
           "exclusion: the clone pair sits on a floor, the independently "
           "formed pair does not." if y1 else
           "✗ the measured identities do not order the exclusion."))
    lines.append(
        f"Y2 (composition interpolates): φ(mix, A) = {phi_mix_a:.3f}, "
        f"φ(mix, B) = {phi_mix_b:.3f}; barrier {st_mix['barrier']:.3f} "
        f"between {st_indep['barrier']:.3f} and {st_clone['barrier']:.3f} — "
        + ("✓ the corpus's own composite is measurably part-clone of both "
           "parents, and its floor sits between theirs." if y2 else
           "✗ the composite does not interpolate."))
    lines.append(
        "Y3 (the individuality threshold): φ(η) = "
        + ", ".join(f"{s['phi']:.3f}" for s in scan)
        + f" (monotone {mono}); barriers "
        + ", ".join(f"{s['barrier']:.3f}" for s in scan)
        + (f"; floor released at η* = {eta_star:g} (sub-blob width "
           f"{args.sub_width:g}) — " if eta_star is not None else "; no "
           "release in the scanned range — ")
        + ("✓ a clone becomes an individual when its internal pattern "
           "drifts by a measurable amount — the framework's own number."
           if y3 else "✗ no clean individuality threshold."))
    lines.append("")
    lines.append(f"score: {int(y1)+int(y2)+int(y3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: statics only (Part III's L2 established that the "
        "φ-routed dynamics follow the statics bitwise, and this "
        "experiment's new content is the measurement of φ, not the "
        "φ-physics); patterns are position-registered patches (no "
        "rotation/translation invariance in φ — an aligned-similarity "
        "measure is the follow-up); sameness is collapsed to a scalar "
        "shared fraction before routing (a full per-type decomposition "
        "of measured patterns is open); clusters are one structure "
        "family (5 sub-blobs, one spread/width); the corpus certifies "
        "and composes but its stability gate is set loose here — "
        "identity via the engine's own emergent stable objects is the "
        "deeper registered door.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_identity_generation.json"),
              "w") as fh:
        json.dump({"params": vars(args),
                   "phi": {"clone": phi_clone, "indep": phi_indep,
                           "mix_a": phi_mix_a, "mix_b": phi_mix_b},
                   "statics": {"clone": st_clone, "indep": st_indep,
                               "mix": st_mix},
                   "jitter_scan": scan,
                   "analysis": {"y1": bool(y1), "y2": bool(y2),
                                "y3": bool(y3), "eta_star": eta_star}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
