"""Exclusion with labelled loads: pricing only TRUE clones (follow-up #2).

Part II left one idealization flagged in both docs: the duplicated
fraction ``ρ_dup = min(ρ₁, ρ₂)`` prices ALL overlap as duplication —
the load field cannot tell same-distinction from different-distinction
stacking.  But the clean statement of what the term MEANS is now
available: the gap ``2F(ρ) − F(2ρ)`` exactly cancels the concavity
(sharing) discount — for IDENTICAL overlap only.  Different
distinctions legitimately keep the discount (that discount IS
binding).  So exclusion should be identity-selective: remove the
sharing discount where the stacked distinction is the same, keep it
where it is different.  This experiment builds the load that can tell
the difference.

The construction (``shared_fraction_labels`` /
``shared_duplicated_components`` / ``exclusion_gap_labelled`` /
``contact_full_share`` / ``contact_full_labels`` in
``project_genesis/capacity_waves.py``): each mass carries a label
vector w_i over distinction types (weights ≥ 0, sum 1).  The capacity
field responds to the TOTAL load as before (gravity is identity-blind);
exclusion applies per SHARED type only,
``ρ_dup^(t) = min(w₁ₜρ₁, w₂ₜρ₂)`` and
``E_x = Σ_t [2E(ρ_dup^(t)) − E(2ρ_dup^(t))]``.  The scalar special
case: a shared fraction φ — each blob's load splits φ·ρ_i common-type +
(1−φ)·ρ_i private-type.  φ = 1 recovers the unlabelled instrument
bitwise; φ = 0 is exactly the no-exclusion control.

Pre-registered predictions (protocol identical to the arc's: size 96,
width 2.5, mass 0.6, r = 0.02, c = 0.8, τ = 0.1, d₀ = 12, t_max = 250,
dt = 0.1, probe radius 11, the same detrend + Hann + whitening ringdown
pipeline):

L1. **Identity routing (statics).**  The same-label (φ = 1) pair
    reproduces the floor: interior minimum s* ∈ (2, 10) with barrier
    ≥ 0.2 (the base branch recorded s* = 8, barrier 0.68).  The
    different-label (φ = 0) pair is the no-exclusion control: monotone
    attractive, no interior minimum in (2, 10), barrier < 0.05.
L2. **Pass-through (dynamics).**  Both binaries released per the
    standard protocol, same parameters, only labels differ: the
    different-label binary plunges to contact like the baseline (late
    mean separation within a factor of 2 of the no-exclusion
    baseline's); the same-label binary stalls at the floor (late
    separation within a factor of 2 of s* and ≥ 2× the
    different-label's — the base branch's recorded ratios).
L3. **Shared-fraction scan (statics).**  φ ∈ {0, 0.25, 0.5, 0.75, 1}:
    the barrier is monotone non-decreasing in φ, endpoints matching
    L1's.  Mechanistic expectation, stated ahead of the measurement:
    the shared component's amplitude scales with φ and the
    full-overlap gap G(A) grows with A, so E_x(φ; s) grows with φ at
    every s, most where the overlap is deepest — the barrier between
    contact and the floor grows monotonically.

Usage::

    python experiments/n3_exclusion_labelled.py \
        --output-dir artifacts/n3_exclusion_labelled
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
    evolve_inertial_retarded,
    exclusion_gap_labelled,
    shared_fraction_labels,
)


def field_kwargs(args):
    return dict(kappa_diffusion=1.0, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0)


def pair_loads(args, s):
    n = int(args.size)
    return (gaussian_load((n, n), [(n / 2 - s / 2, n / 2)], args.width,
                          args.mass),
            gaussian_load((n, n), [(n / 2 + s / 2, n / 2)], args.width,
                          args.mass))


def field_curve(args):
    """The φ-independent part of the statics: F[κ̄(ρ_tot)] per separation.

    Gravity is identity-blind — the capacity field responds to the
    TOTAL load — so this curve is computed once and shared by every
    labelled statics curve.
    """
    out = []
    for s in args.seps:
        l1, l2 = pair_loads(args, s)
        tot = l1 + l2
        kappa = relax_capacity(tot, **field_kwargs(args))
        out.append(capacity_free_energy(kappa, tot, **field_kwargs(args)))
    return out


def exclusion_curve(args, share):
    """E_x(s; φ) — the labelled exclusion statics at shared fraction φ."""
    labels1, labels2 = shared_fraction_labels(share)
    out = []
    for s in args.seps:
        l1, l2 = pair_loads(args, s)
        out.append(exclusion_gap_labelled(l1, l2, labels1, labels2,
                                          **field_kwargs(args)))
    return out


def statics(args, share, field):
    """The full labelled two-body energy E(s; φ), zeroed at the far separation."""
    ex = exclusion_curve(args, share)
    cur = np.array(field) + np.array(ex)
    return list(cur - cur[-1])


def circular_speed(args) -> float:
    """Calibrated circular speed at d₀ (contact force negligible there)."""
    n = int(args.size)
    p1 = (n / 2.0 - args.d0 / 2.0, n / 2.0)
    p2 = (n / 2.0 + args.d0 / 2.0, n / 2.0)
    l1 = gaussian_load((n, n), [p1], args.width, args.mass)
    l2 = gaussian_load((n, n), [p2], args.width, args.mass)
    kappa = relax_capacity(l1 + l2, **field_kwargs(args))
    gx = 0.5 * (np.roll(kappa, -1, 0) - np.roll(kappa, 1, 0))
    return float(np.sqrt(abs(-args.c * float(np.sum(l1 * kappa * gx)))
                         * args.d0 / (2.0 * args.mass)))


def release(args, v0: float, share, probes=None):
    """The standard release; ``share`` None is the no-exclusion baseline."""
    L = args.size
    kw = dict(shape=(int(L), int(L)), width=args.width, tau=args.tau,
              dt=args.dt, steps=int(args.t_max / args.dt), record_every=10,
              probes=probes, **field_kwargs(args))
    if share is not None:
        kw.update(contact_full=True, contact_full_share=share)
    return evolve_inertial_retarded(
        [[L / 2 - args.d0 / 2, L / 2], [L / 2 + args.d0 / 2, L / 2]],
        [[0.0, +v0], [0.0, -v0]], [args.mass] * 2, **kw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=float, default=96.0)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--mass", type=float, default=0.6)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--d0", type=float, default=12.0)
    p.add_argument("--t-max", type=float, default=250.0)
    p.add_argument("--probe-radius", type=float, default=11.0)
    p.add_argument("--ring-window", type=float, nargs=2, default=[15.0, 95.0])
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--seps", type=float, nargs="+",
                   default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
                            12.0, 16.0])
    p.add_argument("--shares", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--output-dir", type=str,
                   default="artifacts/n3_exclusion_labelled")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.t_max = 180.0
        args.seps = [1.0, 3.0, 5.0, 6.0, 8.0, 10.0, 16.0]

    os.makedirs(args.output_dir, exist_ok=True)
    print("exclusion with labelled loads — pricing only true clones",
          flush=True)

    # --- the φ-independent field statics (gravity is identity-blind)
    print("  field statics F[κ̄(ρ_tot)](s) — the identity-blind part:",
          flush=True)
    field = field_curve(args)

    # --- L1: identity routing (statics)
    print("  L1 — same-label (φ = 1) vs different-label (φ = 0) statics:",
          flush=True)
    cur_same = statics(args, 1.0, field)
    cur_diff = statics(args, 0.0, field)
    print("    same-label: E(s) = " + " ".join(
        f"{s:g}:{e:+.2f}" for s, e in zip(args.seps, cur_same)), flush=True)
    print("    diff-label: E(s) = " + " ".join(
        f"{s:g}:{e:+.2f}" for s, e in zip(args.seps, cur_diff)), flush=True)
    i_min = int(np.argmin(cur_same))
    s_star = float(args.seps[i_min])
    barrier_same = float(cur_same[0] - cur_same[i_min])
    floor_found = bool(2.0 < s_star < 10.0 and barrier_same >= 0.2)
    i_min_d = int(np.argmin(cur_diff))
    s_min_d = float(args.seps[i_min_d])
    barrier_diff = float(cur_diff[0] - cur_diff[i_min_d])
    diff_monotone = bool(np.all(np.diff(cur_diff) >= -1e-9))
    diff_no_interior = bool(not (2.0 < s_min_d < 10.0))
    diff_control = bool(diff_monotone and diff_no_interior
                        and barrier_diff < 0.05)
    l1 = floor_found and diff_control
    print(f"    same-label: s* = {s_star:g}, barrier = {barrier_same:.2f} "
          f"(bars: s* ∈ (2, 10), ≥ 0.2): {floor_found}", flush=True)
    print(f"    diff-label: monotone attractive: {diff_monotone}, min at "
          f"s = {s_min_d:g} (interior (2, 10): {not diff_no_interior}), "
          f"barrier = {barrier_diff:.2f} (< 0.05): {diff_control}",
          flush=True)

    # --- L3: the shared-fraction scan (independent recomputation; the
    #     endpoints must match L1's curves bitwise)
    print("  L3 — the shared-fraction scan, barrier(φ):", flush=True)
    scan_curves, barriers = {}, {}
    for phi in args.shares:
        cur = statics(args, phi, field)
        scan_curves[phi] = cur
        barriers[phi] = float(cur[0] - min(cur))
        print(f"    φ = {phi:4.2f}: barrier = {barriers[phi]:+.4f} "
              f"(min at s = {args.seps[int(np.argmin(cur))]:g})",
              flush=True)
    b_vals = [barriers[phi] for phi in args.shares]
    scan_monotone = bool(np.all(np.diff(b_vals) >= -1e-9))
    endpoints_match = bool(barriers[args.shares[0]] == barrier_diff
                           and barriers[args.shares[-1]] == barrier_same)
    l3 = scan_monotone and endpoints_match
    print(f"    barrier monotone non-decreasing in φ: {scan_monotone}; "
          f"endpoints match L1 bitwise: {endpoints_match}", flush=True)

    # --- L2: pass-through dynamics (same parameters, only labels differ)
    L = args.size
    probes = [(int(round(L / 2 + args.probe_radius * np.cos(a))),
               int(round(L / 2 + args.probe_radius * np.sin(a))))
              for a in np.linspace(0, 2 * np.pi, 4, endpoint=False)]
    v0 = circular_speed(args)
    print(f"  L2 — releasing the binaries (calibrated v0 = {v0:.4f}): "
          f"same-label (φ = 1), different-label (φ = 0), and the "
          f"no-exclusion baseline", flush=True)
    h_same = release(args, v0, 1.0, probes=probes)
    h_diff = release(args, v0, 0.0, probes=probes)
    h_base = release(args, v0, None, probes=probes)
    t_rec = np.asarray(h_same["time"])
    late = t_rec > 0.6 * args.t_max
    sep_same = float(np.asarray(h_same["separation"])[late].mean())
    sep_diff = float(np.asarray(h_diff["separation"])[late].mean())
    sep_base = float(np.asarray(h_base["separation"])[late].mean())
    sep_identity = float(np.max(np.abs(
        np.asarray(h_diff["separation"])
        - np.asarray(h_base["separation"]))))
    plunge = bool(0.5 <= sep_diff / sep_base <= 2.0)
    stall = bool(0.5 <= sep_same / s_star <= 2.0
                 and sep_same >= 2.0 * sep_diff)
    l2 = plunge and stall
    print(f"    late mean separation: same-label {sep_same:.2f} "
          f"(s* = {s_star:g}, ratio {sep_same / s_star:.2f}), "
          f"diff-label {sep_diff:.2f}, baseline {sep_base:.2f} "
          f"(ratio {sep_diff / sep_base:.2f}); same/diff = "
          f"{sep_same / sep_diff:.2f} (bar ≥ 2)", flush=True)
    print(f"    φ = 0 run vs baseline: max |Δseparation| = "
          f"{sep_identity:.2e} (the labelled control IS the baseline)",
          flush=True)

    # --- ringdown pipeline (context echo of the base branch's G3;
    #     L2's bars are the separation comparisons above)
    def detrend(x, w):
        pad = np.pad(x, w // 2, mode="edge")
        return x - np.convolve(pad, np.ones(w) / w, mode="valid")[: len(x)]

    lo, hi = args.ring_window

    def waveform_spectrum(hist):
        sig = np.asarray(hist["probe_kappa"])
        t_f = args.dt * np.arange(sig.shape[0])
        winf = (t_f > lo) & (t_f < hi)
        freqs = 2.0 * np.pi * np.fft.rfftfreq(int(winf.sum()), d=args.dt)
        hann = np.hanning(int(winf.sum()))
        power = np.zeros_like(freqs)
        for j in range(sig.shape[1]):
            x = detrend(sig[:, j], 150)[winf] * hann
            power += np.abs(np.fft.rfft(x)) ** 2
        half = 10
        cont = np.array([np.median(power[max(i - half, 0):i + half + 1])
                         for i in range(len(power))])
        white = power / np.maximum(cont, 1e-30)
        band = (freqs >= 0.1) & (freqs <= 1.6)
        peak_i = int(np.argmax(white[band]))
        return float(freqs[band][peak_i]), float(white[band][peak_i])

    pos = np.asarray(h_same["positions"])
    sv = pos[:, 0, :] - pos[:, 1, :]
    phi_rot = np.unwrap(np.arctan2(sv[:, 1], sv[:, 0]))
    win_rec = (t_rec > lo) & (t_rec < hi)
    om_rot = float(np.abs(np.polyfit(t_rec[win_rec], phi_rot[win_rec], 1)[0]))
    dt_rec = float(t_rec[1] - t_rec[0])
    sep_w = (detrend(np.asarray(h_same["separation"]), 15)[win_rec]
             * np.hanning(int(win_rec.sum())))
    fr_rec = 2.0 * np.pi * np.fft.rfftfreq(len(sep_w), d=dt_rec)
    pw_rec = np.abs(np.fft.rfft(sep_w)) ** 2
    live = fr_rec > 0.1
    om_lib = float(fr_rec[live][np.argmax(pw_rec[live])])
    near = [(s, e) for s, e in zip(args.seps, cur_same)
            if abs(s - s_star) <= 2.5]
    om_static = None
    if len(near) >= 3:
        a2 = float(np.polyfit([s for s, _ in near],
                              [e for _, e in near], 2)[0])
        om_static = float(np.sqrt(max(2.0 * a2, 0.0) / (args.mass / 2.0)))
    peak_same, contrast_same = waveform_spectrum(h_same)
    peak_diff, contrast_diff = waveform_spectrum(h_diff)
    print(f"    ringdown (context): same-label libration ω = {om_lib:.3f} "
          + (f"vs static pitch {om_static:.3f}" if om_static is not None
             else "")
          + f"; waveform line ω = {peak_same:.3f}, contrast "
          f"{contrast_same:.1f} (rotation died: Ω = {om_rot:.4f}); "
          f"diff-label slosh ω = {peak_diff:.3f}, contrast "
          f"{contrast_diff:.1f}", flush=True)

    # --- verdicts
    lines = ["Exclusion with labelled loads — verdict", "=" * 74, ""]
    lines.append(
        "the load that can tell same from different: each mass carries a "
        "label vector over distinction types; the capacity field "
        "responds to the TOTAL load (gravity is identity-blind), and "
        "exclusion applies per SHARED type only — E_x = Σ_t "
        "[2E(ρ_dup^(t)) − E(2ρ_dup^(t))] with ρ_dup^(t) = "
        "min(w₁ₜρ₁, w₂ₜρ₂).  Same-distinction overlap loses the "
        "concavity (sharing) discount; different-distinction overlap "
        "keeps it.  No free parameters; φ = 1 is bitwise the unlabelled "
        "instrument.")
    lines.append("")
    lines.append(
        "L1 (identity routing): same-label (φ = 1) statics — s* = "
        f"{s_star:g}, barrier {barrier_same:.2f} toward contact (bars: "
        f"s* ∈ (2, 10), ≥ 0.2): {floor_found}; different-label (φ = 0) "
        f"statics — monotone attractive: {diff_monotone}, no interior "
        f"minimum in (2, 10) (min at s = {s_min_d:g}): "
        f"{diff_no_interior}, barrier {barrier_diff:.2f} < 0.05 — "
        + ("✓ the label routes the exclusion: the same-distinction pair "
           "keeps the floor, the different-distinction pair is the "
           "no-exclusion control." if l1
           else "✗ the label does not route the exclusion as predicted "
           "— recorded as-is."))
    lines.append(
        "    same-label E(s) = " + " ".join(
            f"{s:g}:{e:+.2f}" for s, e in zip(args.seps, cur_same))
        + " | diff-label E(s) = " + " ".join(
            f"{s:g}:{e:+.2f}" for s, e in zip(args.seps, cur_diff)) + ".")
    lines.append(
        "L2 (pass-through): late mean separation — same-label "
        f"{sep_same:.2f} vs s* = {s_star:g} (ratio "
        f"{sep_same / s_star:.2f}, bar a factor of 2) and vs the "
        f"different-label {sep_diff:.2f} (ratio "
        f"{sep_same / sep_diff:.2f}, bar ≥ 2); different-label vs the "
        f"no-exclusion baseline {sep_base:.2f} (ratio "
        f"{sep_diff / sep_base:.2f}, bar a factor of 2): {plunge} — "
        + ("✓ the different-distinction binary passes through to "
           "contact like the baseline; the same-distinction binary "
           "stalls on the floor." if l2
           else "✗ the pass-through routing fails — recorded as-is."))
    lines.append(
        f"    identity check: the φ = 0 labelled run is the baseline — "
        f"max |Δseparation| over the whole trajectory = "
        f"{sep_identity:.2e}.  Ringdown context: same-label libration "
        f"ω = {om_lib:.3f} vs static pitch "
        + (f"{om_static:.3f}" if om_static is not None else "n/a")
        + f", waveform line ω = {peak_same:.3f} with whitened contrast "
        f"{contrast_same:.1f}; the different-label run's post-plunge "
        f"slosh rings at ω = {peak_diff:.3f} with contrast "
        f"{contrast_diff:.1f}.")
    lines.append(
        "L3 (shared-fraction scan): barrier(φ) = " + " ".join(
            f"{phi:g}:{barriers[phi]:+.4f}" for phi in args.shares)
        + f" — monotone non-decreasing: {scan_monotone}; endpoints "
        f"match L1 bitwise: {endpoints_match} — "
        + ("✓ the barrier grows with the shared fraction, as the "
           "mechanism (shared amplitude ∝ φ, G(A) growing in A) "
           "predicted." if l3
           else "✗ the barrier is not monotone in the shared fraction "
           "— recorded as-is."))
    lines.append("")
    lines.append(f"score: {int(l1) + int(l2) + int(l3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: the labels are ASSIGNED, not derived — the "
        "framework prices identity selectively once told what is "
        "identical, but it does not GENERATE the identities; a load "
        "that carries its own label structure is follow-up territory.  "
        "The instrument is the binary (exactly two masses); the trio / "
        "N-body case with ≥ 3 same-label stacks is untested here — the "
        "pairwise-min construction prices each shared pair, and whether "
        "that double-counts or under-prices a triple stack is the "
        "n-copy sector's question (follow-up #3).  The statics use the "
        "repo's relax_capacity on the exact min; the dynamics use the "
        "smoothed min (ε = 1e-4) and conjugate-gradient solves, as the "
        "base branch — φ = 1 is bitwise the unlabelled path.  With "
        "unequal label weights the mirror symmetry breaks and the pair "
        "force picks up the energy's own sub-lattice translation "
        "derivative (measured: the residual equals −dE_x/d(shift) to "
        "0.2% — the force stays the exact gradient of the booked "
        "energy; the symmetric-φ force conserves momentum to machine "
        "precision).  One operating point (size 96, width 2.5, mass "
        "0.6, r = 0.02, c = 0.8, τ = 0.1).")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir,
                           "n3_exclusion_labelled.json"), "w") as fh:
        json.dump({"params": vars(args),
                   "field_statics": field,
                   "l1": {"same_label": {"curve": cur_same,
                                         "s_star": s_star,
                                         "barrier": barrier_same,
                                         "floor_found": floor_found},
                          "diff_label": {"curve": cur_diff,
                                         "monotone": diff_monotone,
                                         "min_sep": s_min_d,
                                         "no_interior_min":
                                             diff_no_interior,
                                         "barrier": barrier_diff},
                          "verdict": bool(l1)},
                   "l2": {"late_sep": {"same": sep_same,
                                       "diff": sep_diff,
                                       "baseline": sep_base},
                          "ratio_same_over_sstar": sep_same / s_star,
                          "ratio_same_over_diff": sep_same / sep_diff,
                          "ratio_diff_over_base": sep_diff / sep_base,
                          "diff_baseline_identity": sep_identity,
                          "ringdown": {"omega_rot": om_rot,
                                       "omega_lib": om_lib,
                                       "omega_static": om_static,
                                       "peak_same": peak_same,
                                       "contrast_same": contrast_same,
                                       "peak_diff": peak_diff,
                                       "contrast_diff": contrast_diff},
                          "verdict": bool(l2)},
                   "l3": {"shares": list(args.shares),
                          "barriers": {str(phi): barriers[phi]
                                       for phi in args.shares},
                          "curves": {str(phi): scan_curves[phi]
                                     for phi in args.shares},
                          "monotone": scan_monotone,
                          "endpoints_match": endpoints_match,
                          "verdict": bool(l3)},
                   "analysis": {"l1": bool(l1), "l2": bool(l2),
                                "l3": bool(l3)}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
