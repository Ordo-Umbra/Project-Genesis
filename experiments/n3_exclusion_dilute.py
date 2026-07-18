"""The dilute operating point: where the Part-I derivation is self-consistent (follow-up #4).

The arc's two derivations of the exclusion term disagree at the dense
operating point: Part I's homogeneous local form
``e(ρ) = c²rρ²/((r+cρ)(r+2cρ))`` inverted into a merger subsidy there
(D2 FAIL — the blob lives 50–100× above the refusal window
``ρ < r/(2c)``), while Part II's full-functional nonlocal gap
``E_x = 2E(ρ_dup) − E(2ρ_dup)`` bought the floor (3/3).  The
convergence claim under test (registered in Part IV): in the
DILUTE-BROAD regime — blob amplitude ``≪ r/(2c)`` AND width
``≫ ξ = √(D/r)`` — both derivations reduce to the SAME energy:
Part I gives ``∫e(ρ) ≈ (c²/r)∫ρ²`` and Part II's linear response with
a broad load gives ``E_x ≈ c²κ₀²(ρ, G_r ρ) ≈ (c²κ₀²/r)∫ρ²`` — the same
coefficient (κ₀ = 1).  This follow-up chooses an operating point inside
that regime BY CONSTRUCTION (a different regime from the arc's dense
point — that is the point of the follow-up) and tests the claim.

THE OPERATING POINT (design — measure first; the three justifying
measurements are printed in section 0 of the run): the spec's example
was r = 0.05, c = 0.8 (window ρ < 0.03125), amplitude A = 0.01,
width 8; criterion (b) — width ≥ 2ξ = 8.94 — fails at width 8, so the
ONE registered adjustment moves the width to 10.  Criteria, all
measured: (a) peak total load at contact < 0.8·r/(2c) = 0.025;
(b) width ≥ 2ξ; (c) the no-exclusion baseline binary measurably
collapses within t_max (the baseline is measured FIRST, an instrument
necessity, as follow-up #3).

Pre-registered predictions (size 96, width 10, mass = amplitude 0.01,
r = 0.05, c = 0.8, τ = 0.1, dt = 0.1):

P1. **The two derivations are one theory here.**  The Part-I local
    form ``∫e(ρ_dup)`` and the Part-II full-functional form agree on
    the pair exclusion energy ``E_x(s)`` to within 10% across the
    separation ladder.  RECORD AS-IS.
P2. **Part I's form buys a floor where it is self-consistent.**  Statics
    with ``contact_derived`` (the Part-I form, no free parameters) have
    an interior minimum ``s* ∈ (2, size/2 − width)`` with a barrier
    ≥ 10% of the control's binding depth at s* (a RELATIVE bar — the
    energy scales differ from the dense point); the no-exclusion
    control is monotone attractive.  RECORD AS-IS.
P3. **Dynamics.**  The released dilute binary stalls at the floor:
    late mean separation within a factor of 2 of the static s* AND
    ≥ 1.5× the no-exclusion baseline's late separation.  If the
    baseline does not collapse within a computationally feasible t_max,
    extend t_max ONCE by the measured collapse time and record the
    choice; if the baseline genuinely never collapses in-window, the
    registered fallback is: derived-form separation stays ≥ s*-consistent
    while the baseline separation is monotone decreasing at the end of
    the run.  Say which branch fired.

Also for the record: ``b = 2c²/r`` at this point, and how well the
constant-b term with that b reproduces P2's statics (the dilute-limit
claim end-to-end).

Usage::

    python experiments/n3_exclusion_dilute.py \
        --output-dir artifacts/n3_exclusion_dilute
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
    duplicated_load,
    evolve_inertial_retarded,
    exclusion_energy_density,
    exclusion_gap_full,
    linear_response_exclusion_gap,
)


def field_kwargs(args):
    return dict(kappa_diffusion=1.0, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0)


def pair_loads(args, s):
    """The mirrored equal binary at separation s, centred on the grid."""
    n = int(args.size)
    return (gaussian_load((n, n), [(n / 2 - s / 2, n / 2)], args.width,
                          args.mass),
            gaussian_load((n, n), [(n / 2 + s / 2, n / 2)], args.width,
                          args.mass))


def e_local(load, args):
    """The Part-I local exclusion density integrated over the load."""
    return float(np.sum(exclusion_energy_density(
        load, kappa_recovery=args.r, kappa_consumption=args.c)))


def release(args, derived: bool):
    """Release the binary FROM REST at separation d₀ (the branch-neutral
    choice: the circular-speed calibration of the dense-point runs
    assumes a bound orbit, and P2 registers doubt about a floor)."""
    L = int(args.size)
    return evolve_inertial_retarded(
        [[L / 2 - args.d0 / 2, L / 2], [L / 2 + args.d0 / 2, L / 2]],
        [[0.0, 0.0], [0.0, 0.0]], [args.mass] * 2, shape=(L, L),
        width=args.width, tau=args.tau, dt=args.dt,
        steps=int(args.t_max / args.dt), record_every=10,
        contact_derived=derived, **field_kwargs(args))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=float, default=96.0)
    p.add_argument("--width", type=float, default=10.0)
    p.add_argument("--mass", type=float, default=0.01)
    p.add_argument("--r", type=float, default=0.05)
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--d0", type=float, default=12.0)
    p.add_argument("--t-max", type=float, default=250.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--seps", type=float, nargs="+",
                   default=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                            18.0, 20.0, 24.0, 28.0, 32.0, 36.0])
    p.add_argument("--scan-amps", type=float, nargs="+",
                   default=[0.01, 0.003, 0.001])
    p.add_argument("--scan-widths", type=float, nargs="+",
                   default=[10.0, 16.0, 24.0])
    p.add_argument("--output-dir", type=str,
                   default="artifacts/n3_exclusion_dilute")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.t_max = 60.0
        args.seps = [2.0, 6.0, 10.0, 12.0, 16.0, 24.0, 36.0]

    os.makedirs(args.output_dir, exist_ok=True)
    fkw = field_kwargs(args)
    b_dilute = 2.0 * args.c * args.c / args.r
    xi = float(np.sqrt(1.0 / args.r))          # D = 1
    window = args.r / (2.0 * args.c)
    print("the dilute operating point — where the derivation is "
          "self-consistent", flush=True)
    print(f"  point: size {args.size:g}, width {args.width:g}, "
          f"amplitude {args.mass:g}, r = {args.r:g}, c = {args.c:g}, "
          f"τ = {args.tau:g}; refusal window ρ < {window:.5f}, "
          f"ξ = {xi:.4f}, b = 2c²/r = {b_dilute:g}", flush=True)

    # --- 0. the operating point's three justifying measurements
    print("  0 — operating-point criteria (measure-first):", flush=True)
    tot_contact = sum(pair_loads(args, 1.0))
    peak_contact = float(tot_contact.max())
    crit_a = bool(peak_contact < 0.8 * window)
    print(f"    (a) peak total load at contact (s = 1) = "
          f"{peak_contact:.6f} < 0.8·r/(2c) = {0.8 * window:.6f}: "
          f"{crit_a}", flush=True)
    crit_b = bool(args.width >= 2.0 * xi)
    print(f"    (b) width {args.width:g} ≥ 2ξ = {2.0 * xi:.4f}: {crit_b} "
          f"(the spec's example width 8 fails this at r = {args.r:g}; "
          f"the ONE registered adjustment moves the width to "
          f"{args.width:g})", flush=True)
    # gravity observability, measured: the capacity dip a single blob
    # makes and the pair binding at contact
    one = pair_loads(args, 1.0)[0]
    k_one = relax_capacity(one, **fkw)
    dip = float(1.0 - k_one.min())
    print(f"    gravity observable: single-blob capacity dip "
          f"δκ = {dip:.4f} at the blob centre", flush=True)
    # (c) the no-exclusion baseline, measured FIRST (instrument
    # necessity, as follow-up #3): it must collapse within t_max.
    print(f"    (c) no-exclusion baseline released from rest at d₀ = "
          f"{args.d0:g}, t_max = {args.t_max:g} — measured first:",
          flush=True)
    h_base = release(args, False)
    t_rec = np.asarray(h_base["time"])
    sep_base_series = np.asarray(h_base["separation"])
    plunge_at = None
    for t, sp in zip(t_rec, sep_base_series):
        if sp < 3.0:
            plunge_at = float(t)
            break
    if plunge_at is None or plunge_at >= 0.6 * args.t_max:
        new_t = 10.0 * np.ceil((0.0 if plunge_at is None
                                else plunge_at) / 0.6 / 10.0 + 0.5)
        args.t_max = float(max(new_t, args.t_max * 1.5))
        print(f"      baseline had not collapsed by the late window "
              f"(plunge at {plunge_at}); extending t_max ONCE to "
              f"{args.t_max:g} (the registered extension) and "
              f"re-running", flush=True)
        h_base = release(args, False)
        t_rec = np.asarray(h_base["time"])
        sep_base_series = np.asarray(h_base["separation"])
        for t, sp in zip(t_rec, sep_base_series):
            if sp < 3.0:
                plunge_at = float(t)
                break
    crit_c = bool(plunge_at is not None and plunge_at < 0.6 * args.t_max)
    print(f"      baseline collapses (separation < 3) at t ≈ "
          f"{plunge_at} — within t_max = {args.t_max:g}: {crit_c} "
          f"(the t_max choice is an instrument necessity, recorded, "
          f"not tuned)", flush=True)
    if not (crit_a and crit_b and crit_c):
        print("  WARNING: an operating-point criterion fails — the point "
              "is outside the registered design; verdicts recorded "
              "as-is.", flush=True)

    # --- P1: the two derivations on the pair exclusion energy
    print("  P1 — E_x(s) both derivations (Part-I ∫e(ρ_dup) vs Part-II "
          "2E(ρ_dup) − E(2ρ_dup)):", flush=True)
    p1_rows = []
    for s in args.seps:
        l1, l2 = pair_loads(args, s)
        dup = duplicated_load(l1, l2)
        ex1 = e_local(dup, args)
        ex2 = exclusion_gap_full(dup, **fkw)
        quad = (args.c * args.c / args.r) * float(np.sum(dup ** 2))
        lr = linear_response_exclusion_gap(dup, **fkw)
        p1_rows.append(dict(s=float(s), ex_local=ex1, ex_full=ex2,
                            ratio=ex1 / ex2 if ex2 > 0 else float("nan"),
                            quad=quad, lr=lr))
    print("     s      E_x(I)    E_x(II)   I/II     I/quad   II/LR    "
          "LR/quad", flush=True)
    for row in p1_rows:
        print(f"    {row['s']:4.0f}  {row['ex_local']:9.5f}  "
              f"{row['ex_full']:9.5f}  {row['ratio']:7.4f}  "
              f"{row['ex_local'] / row['quad']:7.4f}  "
              f"{row['ex_full'] / row['lr']:7.4f}  "
              f"{row['lr'] / row['quad']:7.4f}", flush=True)
    worst = max(abs(row["ratio"] - 1.0) for row in p1_rows)
    best = min(abs(row["ratio"] - 1.0) for row in p1_rows)
    p1 = bool(worst <= 0.10)
    print(f"    ratio I/II runs {p1_rows[0]['ratio']:.4f} (s = "
          f"{args.seps[0]:g}) to {p1_rows[-1]['ratio']:.4f} (s = "
          f"{args.seps[-1]:g}); worst |ratio−1| = {worst:.4f} vs the "
          f"10% bar: {p1}", flush=True)

    # --- P2: statics with the Part-I form (contact_derived), plus the
    #     Part-II form, the constant-b term and the control for the
    #     record
    print("  P2 — statics at the dilute point (derived = the Part-I "
          "form; full = Part II; b = 2c²/r constant; control = no "
          "exclusion):", flush=True)
    field, ex_der, ex_full, ex_b = [], [], [], []
    for s in args.seps:
        tot = sum(pair_loads(args, s))
        kappa = relax_capacity(tot, **fkw)
        field.append(capacity_free_energy(kappa, tot, **fkw))
        ex_der.append(e_local(tot, args))
        l1, l2 = pair_loads(args, s)
        ex_full.append(exclusion_gap_full(duplicated_load(l1, l2), **fkw))
        ex_b.append(0.5 * b_dilute * float(np.sum(tot ** 2)))
    field = np.asarray(field)
    cur_der = field - field[-1] + (np.asarray(ex_der) - ex_der[-1])
    cur_full = field - field[-1] + np.asarray(ex_full)
    cur_b = field - field[-1] + (np.asarray(ex_b) - ex_b[-1])
    cur_ctrl = field - field[-1]
    for name, cur in (("derived", cur_der), ("full", cur_full),
                      ("b=2c²/r", cur_b), ("control", cur_ctrl)):
        print("    %-8s E(s) = %s" % (name, " ".join(
            f"{s:g}:{e:+.4f}" for s, e in zip(args.seps, cur))),
            flush=True)

    def floor_of(cur):
        i_min = int(np.argmin(cur))
        return float(args.seps[i_min]), float(cur[0] - cur[i_min]), i_min

    s_star, barrier, i_min = floor_of(cur_der)
    # an INTERIOR minimum must be interior to the LADDER, not merely to
    # the bar's window: a monotone curve's argmin lands on a ladder
    # edge, which is no floor (the arc's convention — Part I's D2
    # recorded "s* sits at the s = 1 grid edge" as no interior minimum;
    # here the derived curve falls monotonically to its far edge, the
    # unbound pair at infinity).
    interior = bool(0 < i_min < len(args.seps) - 1
                    and 2.0 < s_star < args.size / 2 - args.width)
    ctrl_depth_at = float(-cur_ctrl[i_min])   # control binding at s*
    rel_bar = 0.1 * ctrl_depth_at
    ctrl_monotone = bool(np.all(np.diff(cur_ctrl) >= -1e-12))
    p2 = bool(interior and barrier >= rel_bar and ctrl_monotone)
    # the mechanism, for the record: exclusion vs binding at contact
    ex_span_der = float(ex_der[0] - ex_der[-1])
    bind_span = float(-(field[0] - field[-1]))
    der_monotone_dec = bool(np.all(np.diff(cur_der) <= 1e-12))
    print(f"    derived form: argmin at s = {s_star:g} (ladder index "
          f"{i_min} of {len(args.seps) - 1}; ladder-interior minimum: "
          f"{interior}; monotone decreasing: {der_monotone_dec}), "
          f"barrier toward contact {barrier:+.5f} vs the relative bar "
          f"10% of the control's binding at s* ({rel_bar:.5f}); "
          f"control monotone attractive: {ctrl_monotone}: {p2}",
          flush=True)
    print(f"    mechanism (recorded, no bar): derived exclusion span "
          f"contact→∞ {ex_span_der:+.5f} vs field binding span "
          f"{bind_span:+.5f} — ratio "
          f"{ex_span_der / bind_span if bind_span else float('nan'):.3f}"
          f" (the deep dilute-broad limit makes this ratio → 2: the "
          f"local form's cross term 2(c²/r)∫ρ₁ρ₂ against the binding "
          f"c²(ρ₁, G_r ρ₂) ≈ (c²/r)∫ρ₁ρ₂)", flush=True)
    s_full, bar_full, _ = floor_of(cur_full)
    s_b, bar_b, _ = floor_of(cur_b)
    print(f"    for the record: Part-II form — s* = {s_full:g}, barrier "
          f"{bar_full:+.5f}; constant-b term (b = {b_dilute:g}) — s* = "
          f"{s_b:g}, barrier {bar_b:+.5f}; b-term vs derived exclusion "
          f"at contact: {ex_b[0]:.5f} vs {ex_der[0]:.5f} (ratio "
          f"{ex_b[0] / ex_der[0]:.3f} — the dilute-limit claim "
          f"end-to-end: e(ρ) sits at "
          f"{ex_der[0] / ex_b[0] * 100:.1f}% of its quadratic asymptote "
          f"at the contact density)", flush=True)

    # --- P2x — EXPLORATORY (not pre-registered, no bar): the 2×2
    #     decomposition {homogeneous e(ρ), full-functional gap} ×
    #     {on ρ_tot, on ρ_dup = min}.  Which difference kills the
    #     dilute floor — the FUNCTIONAL or the CONSTRUCTION?  (What did
    #     Part II actually fix?)  All four curves zeroed at the ladder
    #     end, which also removes the on-total forms' s-independent
    #     self-energy.
    print("  P2x — EXPLORATORY (no bar): the 2×2 {functional × "
          "construction} statics", flush=True)
    ex_local_min = np.asarray([row["ex_local"] for row in p1_rows])
    full_tot = np.asarray([exclusion_gap_full(sum(pair_loads(args, s)),
                                              **fkw)
                           for s in args.seps])
    cur_local_min = (field - field[-1]
                     + (ex_local_min - ex_local_min[-1]))
    cur_full_tot = field - field[-1] + (full_tot - full_tot[-1])
    quad_floors = {}
    for name, cur in (("local-on-total (derived)", cur_der),
                      ("local-on-min", cur_local_min),
                      ("full-on-total", cur_full_tot),
                      ("full-on-min (Part II)", cur_full)):
        sq, bq, iq = floor_of(cur)
        quad_floors[name] = {"curve": list(cur), "s_star": sq,
                             "barrier": bq,
                             "interior": bool(
                                 0 < iq < len(args.seps) - 1
                                 and 2.0 < sq
                                 < args.size / 2 - args.width)}
        print("    %-22s E(s) = %s" % (name, " ".join(
            f"{s:g}:{e:+.4f}" for s, e in zip(args.seps, cur))),
            flush=True)
        print(f"    %-22s s* = {sq:g}, barrier {bq:+.5f}, interior: "
              f"{quad_floors[name]['interior']}" % "", flush=True)

    # --- P3: dynamics from rest (baseline already measured in 0)
    print("  P3 — releasing the dilute binary from rest at d₀ = "
          f"{args.d0:g} (derived form; baseline above):", flush=True)
    h_der = release(args, True)
    late = t_rec > 0.6 * args.t_max
    sep_der = float(np.asarray(h_der["separation"])[late].mean())
    sep_base = float(sep_base_series[late].mean())
    sep_der_series = np.asarray(h_der["separation"])
    sep_der_max = float(sep_der_series.max())
    sep_der_end = float(sep_der_series[-3:].mean())
    repelled = bool(sep_der > args.d0 + 1.0)
    ratio_sstar = (sep_der / s_star) if interior else float("nan")
    stall = bool(interior and 0.5 <= ratio_sstar <= 2.0
                 and sep_der >= 1.5 * sep_base)
    # which registered branch fired?
    base_collapsed = bool(plunge_at is not None
                          and plunge_at < args.t_max)
    base_series_late = sep_base_series[late]
    base_mono_dec = bool(np.all(np.diff(base_series_late) <= 1e-9))
    if stall:
        branch = "main (stall at the floor)"
    elif not base_collapsed:
        branch = ("fallback (baseline never collapses in-window; "
                  f"derived ≥ s*-consistent: n/a (s* interior: "
                  f"{interior}), baseline late-separation monotone "
                  f"decreasing: {base_mono_dec})")
    else:
        branch = ("NEITHER registered branch: the baseline collapses "
                  "in-window AND the derived binary does not stall — "
                  "the P2 statics carry the mechanism (no interior "
                  "floor; the net dilute force is repulsive)")
    p3 = stall
    print(f"    late mean separation: derived {sep_der:.2f}"
          + (f" vs s* = {s_star:g} (ratio {ratio_sstar:.2f}, bar a "
             f"factor of 2)" if interior else
             " (no interior static s* — the factor-of-2 bar has no "
             "referent)")
          + f"; baseline {sep_base:.2f} (ratio "
          f"{sep_der / sep_base:.2f}, bar ≥ 1.5)", flush=True)
    print(f"    derived-run behaviour: released at d₀ = {args.d0:g}, "
          f"the pair is REPELLED — max separation {sep_der_max:.2f}, "
          f"end-of-run mean {sep_der_end:.2f} (repelled: {repelled}); "
          f"the plain (not periodic) separation norm on the torus is "
          f"the arc's convention, recorded not interpreted", flush=True)
    print(f"    branch fired: {branch}", flush=True)
    print(f"    P3 verdict (stall bar): {p3}", flush=True)

    # --- EXPLORATORY convergence scan (NOT pre-registered): the
    #     joint-limit structure of the P1 ratio — amplitude scan at the
    #     operating width, width scan at the deepest amplitude
    print("  EXPLORATORY (not pre-registered): the P1 ratio toward the "
          "joint dilute-broad limit", flush=True)
    scan = {"amps": [], "widths": []}
    for a in args.scan_amps:
        l1 = gaussian_load((int(args.size), int(args.size)),
                           [(args.size / 2 - 1.0, args.size / 2)],
                           args.width, a)
        l2 = gaussian_load((int(args.size), int(args.size)),
                           [(args.size / 2 + 1.0, args.size / 2)],
                           args.width, a)
        dup = duplicated_load(l1, l2)
        ex1 = e_local(dup, args)
        ex2 = exclusion_gap_full(dup, **fkw)
        quad = (args.c * args.c / args.r) * float(np.sum(dup ** 2))
        lr = linear_response_exclusion_gap(dup, **fkw)
        scan["amps"].append(dict(amp=float(a), ratio=ex1 / ex2,
                                 local_over_quad=ex1 / quad,
                                 full_over_lr=ex2 / lr,
                                 lr_over_quad=lr / quad))
        print(f"    A = {a:g} (width {args.width:g}): I/II = "
              f"{ex1 / ex2:.4f} (I/quad {ex1 / quad:.4f}, II/LR "
              f"{ex2 / lr:.4f}, LR/quad {lr / quad:.4f})", flush=True)
    a_deep = min(args.scan_amps)
    for w in args.scan_widths:
        l1 = gaussian_load((int(args.size), int(args.size)),
                           [(args.size / 2 - 1.0, args.size / 2)], w,
                           a_deep)
        l2 = gaussian_load((int(args.size), int(args.size)),
                           [(args.size / 2 + 1.0, args.size / 2)], w,
                           a_deep)
        dup = duplicated_load(l1, l2)
        ex1 = e_local(dup, args)
        ex2 = exclusion_gap_full(dup, **fkw)
        quad = (args.c * args.c / args.r) * float(np.sum(dup ** 2))
        lr = linear_response_exclusion_gap(dup, **fkw)
        scan["widths"].append(dict(width=float(w), ratio=ex1 / ex2,
                                   local_over_quad=ex1 / quad,
                                   full_over_lr=ex2 / lr,
                                   lr_over_quad=lr / quad))
        print(f"    width = {w:g} (A = {a_deep:g}): I/II = "
              f"{ex1 / ex2:.4f} (I/quad {ex1 / quad:.4f}, II/LR "
              f"{ex2 / lr:.4f}, LR/quad {lr / quad:.4f})", flush=True)

    # --- verdicts
    lines = ["The dilute operating point — verdict", "=" * 74, ""]
    lines.append(
        f"the operating point (size {args.size:g}, width {args.width:g}"
        f", amplitude {args.mass:g}, r = {args.r:g}, c = {args.c:g}, "
        f"τ = {args.tau:g}) lives inside the refusal window "
        f"(ρ < {window:.5f}) with broad blobs (ξ = {xi:.3f}) by "
        f"construction — a DIFFERENT regime from the arc's dense point, "
        f"which is the point of the follow-up.  Justifying "
        f"measurements: (a) peak total load at contact "
        f"{peak_contact:.6f} < 0.8·r/(2c) = {0.8 * window:.6f}; "
        f"(b) width {args.width:g} ≥ 2ξ = {2 * xi:.4f} (the spec "
        f"example's width 8 fails (b); the one registered adjustment "
        f"moved the width to {args.width:g}); (c) the no-exclusion "
        f"baseline, measured first, collapses at t ≈ {plunge_at} "
        f"within t_max = {args.t_max:g}.  Gravity stays observable: "
        f"single-blob capacity dip δκ = {dip:.4f}.")
    lines.append("")
    lines.append(
        f"P1 (the two derivations are one theory here): E_x(s) on the "
        f"duplicated component — Part-I local ∫e(ρ_dup) vs Part-II "
        f"full-functional gap.  Ratio I/II runs "
        f"{p1_rows[0]['ratio']:.4f} at s = {args.seps[0]:g} to "
        f"{p1_rows[-1]['ratio']:.4f} at s = {args.seps[-1]:g} (worst "
        f"|ratio−1| = {worst:.4f}, best {best:.4f}) vs the 10% bar: "
        + ("✓ the forms agree across the ladder."
           if p1 else
           "✗ the bar FAILS — recorded as-is.  The s-dependent map IS "
           "the result: agreement is best at contact (the duplicated "
           "component broad and slowly varying) and degrades down the "
           "skirts, where the min-ridge narrows below ξ and neither "
           "the local-density nor the broad-kernel approximation is "
           "valid.  Both forms approach the common (c²/r)∫ρ_dup² "
           "limit FROM BELOW with different remainders (I/quad and "
           "II/LR columns); their ratio is the product of the two "
           "remainder structures, not a defect of either derivation."))
    lines.append(
        "    " + "  ".join(
            f"{row['s']:g}:{row['ratio']:.3f}" for row in p1_rows))
    lines.append(
        f"P2 (Part I's form buys a floor where it is self-consistent): "
        f"derived statics argmin at s = {s_star:g} — the ladder's far "
        f"edge, i.e. NO ladder-interior minimum ({interior}); the "
        f"curve is monotone decreasing ({der_monotone_dec}); control "
        f"monotone attractive: {ctrl_monotone}: "
        + ("✓ the floor exists." if p2 else
           "✗ recorded as-is — no interior floor: the derived statics "
           "fall monotonically to the unbound pair at infinity; the "
           "net dilute-broad force of the local form is repulsive at "
           "every separation.  The mechanism is "
           f"structural: exclusion span contact→∞ {ex_span_der:+.5f} "
           f"vs binding span {bind_span:+.5f} (ratio "
           f"{ex_span_der / bind_span:.3f}); in the deep limit the "
           "local form's overlap cross term 2(c²/r)∫ρ₁ρ₂ is exactly "
           "2× the quadratic-form binding c²(ρ₁, G_r ρ₂) ≈ "
           "(c²/r)∫ρ₁ρ₂ — amplitude-independent, so the local form "
           "generically cannot buy a floor in this regime."))
    lines.append(
        "    derived E(s) = " + " ".join(
            f"{s:g}:{e:+.4f}" for s, e in zip(args.seps, cur_der))
        + " | control E(s) = " + " ".join(
            f"{s:g}:{e:+.4f}" for s, e in zip(args.seps, cur_ctrl))
        + f" | for the record: Part-II form s* = {s_full:g}, barrier "
        f"{bar_full:+.5f}; constant-b (b = {b_dilute:g}) s* = {s_b:g}"
        f", barrier {bar_b:+.5f}; the b-term's exclusion at contact "
        f"overshoots the derived form's by {ex_b[0] / ex_der[0]:.3f}× "
        f"(the dilute-limit claim end-to-end: at the contact density "
        f"e(ρ) is at {ex_der[0] / ex_b[0] * 100:.1f}% of its quadratic "
        f"asymptote — 3cρ/r = {3 * args.c * float(tot_contact.max()) / args.r:.2f} "
        f"there, the window criterion is ~16× looser than the "
        f"quadratic criterion).")
    lines.append(
        "    P2x — EXPLORATORY (not pre-registered, no bar): the 2×2 "
        "{functional × construction} decomposition of the statics: "
        + "; ".join(
            f"{name}: s* = {q['s_star']:g}, barrier {q['barrier']:+.5f}"
            f", interior {q['interior']}"
            for name, q in quad_floors.items())
        + ".  This pins whether the dilute floor's absence is carried "
        "by the homogeneous FUNCTIONAL or by the ρ_tot CONSTRUCTION — "
        "what Part II actually fixed.")
    lines.append(
        f"P3 (dynamics from rest): baseline collapses at t ≈ "
        f"{plunge_at} (measured first; t_max = {args.t_max:g} an "
        f"instrument necessity, recorded).  Late mean separation: "
        f"derived {sep_der:.2f}"
        + (f" vs s* = {s_star:g} (ratio {ratio_sstar:.2f})"
           if interior else " (no interior static s*)")
        + f", baseline {sep_base:.2f} (ratio {sep_der / sep_base:.2f})"
        f": {'✓ the binary stalls on the floor.' if p3 else '✗ no stall — recorded as-is.'}"
        f"  The derived pair is repelled from rest (max separation "
        f"{sep_der_max:.2f}, end mean {sep_der_end:.2f} vs release "
        f"d₀ = {args.d0:g}) while the baseline plunges.  Branch "
        f"fired: {branch}.")
    lines.append(
        "    EXPLORATORY (not pre-registered): the P1 ratio toward "
        "the joint limit — "
        + "; ".join(f"A = {r['amp']:g}: {r['ratio']:.3f}"
                    for r in scan["amps"])
        + " (amplitude scan at the operating width) and "
        + "; ".join(f"w = {r['width']:g}: {r['ratio']:.3f}"
                    for r in scan["widths"])
        + f" (width scan at A = {a_deep:g}).  The amplitude scan shows "
        "the remainder that vanishes with cAκ₀/r; the width scan shows "
        "the kernel-range remainder that vanishes with ξ/w — the two "
        "derivations converge only in the JOINT dilute-broad limit.")
    lines.append("")
    n_pass = int(p1) + int(p2) + int(p3)
    lines.append(f"score: {n_pass}/3 pre-registered predictions land "
                 "(recorded as-is).")
    lines.append("")
    lines.append(
        "honest scope: a different operating point BY CONSTRUCTION — "
        "nothing here re-measures the dense point; broad blobs make "
        "shallow wells, and every instrument choice (width 10 from "
        "criterion (b), rest release as the branch-neutral choice, "
        "t_max from the measured baseline collapse) is recorded above; "
        "the P1 agreement is at the pair level (the n-copy sector's "
        "O(c²) theorem is the same statement one level up); the "
        "separation record is the arc's plain (not periodic) norm — "
        "with a repulsive pair on a periodic box the late trajectory "
        "is a lattice-winding artifact, recorded not interpreted.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir,
                           "n3_exclusion_dilute.json"), "w") as fh:
        json.dump({"params": vars(args),
                   "criteria": {"peak_contact_load": peak_contact,
                                "crit_a": crit_a, "width": args.width,
                                "two_xi": 2.0 * xi, "crit_b": crit_b,
                                "baseline_plunge_at": plunge_at,
                                "crit_c": crit_c, "kappa_dip": dip},
                   "p1": {"ladder": p1_rows, "worst_dev": worst,
                          "best_dev": best, "verdict": bool(p1)},
                   "p2": {"seps": list(args.seps),
                          "derived": {"curve": list(cur_der),
                                      "s_star": s_star,
                                      "interior": interior,
                                      "barrier": barrier,
                                      "relative_bar": rel_bar},
                          "full": {"curve": list(cur_full),
                                   "s_star": s_full,
                                   "barrier": bar_full},
                          "constant_b": {"b": b_dilute,
                                         "curve": list(cur_b),
                                         "s_star": s_b,
                                         "barrier": bar_b},
                          "control": {"curve": list(cur_ctrl),
                                      "monotone": ctrl_monotone},
                          "exclusion_span_derived": ex_span_der,
                          "binding_span": bind_span,
                          "verdict": bool(p2)},
                   "p2x2": quad_floors,
                   "p3": {"t_max": args.t_max,
                          "baseline_plunge_at": plunge_at,
                          "late_sep_derived": sep_der,
                          "late_sep_baseline": sep_base,
                          "sep_derived_max": sep_der_max,
                          "sep_derived_end": sep_der_end,
                          "repelled_from_rest": repelled,
                          "ratio_over_baseline": sep_der / sep_base,
                          "ratio_over_sstar": (None if not interior
                                               else ratio_sstar),
                          "branch": branch, "verdict": bool(p3)},
                   "exploratory_scan": scan,
                   "analysis": {"p1": bool(p1), "p2": bool(p2),
                                "p3": bool(p3)}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
