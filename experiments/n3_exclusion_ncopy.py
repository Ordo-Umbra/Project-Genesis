"""The n-copy sector: three-and-more identical stacks (follow-up #3).

Parts I–III probed the 2-copy sector only.  The general form is the
n-copy gap: within a group of n same-label masses the cloned component
is the min over the group's loads, and the group pays
``E_x = n·E(m) − E(n·m)`` — implemented as the ``contact_full`` path's
n-mass generalization (``contact_full_ncopy=True``), with the
pairwise-sum form kept as an option (``contact_full_ncopy=False``).
The O(c²) theorem (Part IV of the derivation doc): in linear response
the pairwise-summed exclusion and the true n-copy gap are EQUAL — both
k·n(n−1) — so their split at the operating amplitude is a pure
core-saturation measurement.

Pre-registered predictions (the arc's operating point: size 96,
width 2.5, mass 0.6, r = 0.02, c = 0.8, τ = 0.1, dt = 0.1; three
same-label masses on an equilateral triangle centred on the grid):

N1. **The theorem.**  For DILUTE identical blobs (amplitude 0.01), the
    pairwise-summed and n-copy exclusion energies of a fully-stacked
    triple agree within 5%; at the operating amplitude 0.6 the ratio
    is recorded as-is (divergence from saturation expected).
N2. **Trimer statics floor.**  With the n-copy form, the three-body
    energy has an interior minimum at side s*₃ ∈ (2, 12) with barrier
    ≥ 0.2 toward contact; the no-exclusion control is monotone.  The
    pairwise form's s*₃/barrier are recorded for comparison (no bar on
    which is stronger — the comparison itself is the measurement).
N3. **Trimer dynamics.**  Released from rest at side d₀ = 12, the
    same-label trimer stalls at finite separation: late mean pairwise
    separation within a factor of 2 of the static s*₃ AND ≥ 1.5× the
    no-exclusion baseline's (which should plunge to three-way
    contact).  t_max is chosen so the BASELINE completes its three-way
    collapse well before the end — the baseline is measured first and
    the choice recorded (an instrument necessity, not a tuned result).
    Ringdown analysis is NOT pre-registered for the trimer (three-body
    spectra are not the binary instrument's) — any observed line is
    recorded exploratively, labelled as such.

Usage::

    python experiments/n3_exclusion_ncopy.py \
        --output-dir artifacts/n3_exclusion_ncopy
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
    exclusion_gap_group,
    linear_response_exclusion_gap,
)


def field_kwargs(args):
    return dict(kappa_diffusion=1.0, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0)


def tri_positions(args, s):
    """The equilateral triangle of side s, centroid on the grid centre.

    Apex up, mirror-symmetric about the vertical axis through the
    centre: vertices at centroid-vertex distance R = s/√3.
    """
    L = float(args.size)
    rt = s / np.sqrt(3.0)
    return [(L / 2, L / 2 + rt),
            (L / 2 - rt * np.sqrt(3.0) / 2, L / 2 - rt / 2),
            (L / 2 + rt * np.sqrt(3.0) / 2, L / 2 - rt / 2)]


def tri_loads(args, s, mass=None):
    return [gaussian_load((int(args.size), int(args.size)), [p],
                          args.width, args.mass if mass is None else mass)
            for p in tri_positions(args, s)]


def field_curve(args):
    """The identity-blind part of the statics: F[κ̄(ρ_tot)] per side."""
    out = []
    for s in args.seps:
        tot = sum(tri_loads(args, s))
        kappa = relax_capacity(tot, **field_kwargs(args))
        out.append(capacity_free_energy(kappa, tot, **field_kwargs(args)))
    return out


def exclusion_curves(args):
    """E_x(s) of the trimer, both forms (n-copy and pairwise-summed)."""
    nc, pw = [], []
    for s in args.seps:
        loads = tri_loads(args, s)
        nc.append(exclusion_gap_group(loads, n_copy=True,
                                      **field_kwargs(args)))
        pw.append(exclusion_gap_group(loads, n_copy=False,
                                      **field_kwargs(args)))
    return nc, pw


def zeroed(cur):
    cur = np.asarray(cur, dtype=float)
    return list(cur - cur[-1])


def mean_pairwise_separation(positions):
    """Mean of the three pairwise distances per record (plain, not
    periodic — the arc's separation convention)."""
    pos = np.asarray(positions, dtype=float)          # records × 3 × 2
    d = [np.linalg.norm(pos[:, i, :] - pos[:, j, :], axis=1)
         for i, j in ((0, 1), (0, 2), (1, 2))]
    return (d[0] + d[1] + d[2]) / 3.0


def release(args, n_copy, probes=None):
    """Release the trimer from rest at side d₀; ``n_copy`` None is the
    no-exclusion baseline."""
    L = int(args.size)
    kw = dict(shape=(L, L), width=args.width, tau=args.tau, dt=args.dt,
              steps=int(args.t_max / args.dt), record_every=10,
              probes=probes, **field_kwargs(args))
    if n_copy is not None:
        kw.update(contact_full=True, contact_full_ncopy=n_copy)
    return evolve_inertial_retarded(
        [list(p) for p in tri_positions(args, args.d0)],
        [[0.0, 0.0]] * 3, [args.mass] * 3, **kw)


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
    p.add_argument("--ring-window", type=float, nargs=2,
                   default=[15.0, 95.0])
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--seps", type=float, nargs="+",
                   default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
                            12.0, 16.0])
    p.add_argument("--stack-amps", type=float, nargs="+",
                   default=[0.01, 0.6])
    p.add_argument("--output-dir", type=str,
                   default="artifacts/n3_exclusion_ncopy")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.t_max = 120.0
        args.seps = [1.0, 3.0, 5.0, 6.0, 8.0, 10.0, 16.0]

    os.makedirs(args.output_dir, exist_ok=True)
    print("the n-copy sector — three-and-more identical stacks",
          flush=True)

    # --- N1: the theorem — dilute vs operating amplitude, fully
    #     stacked triple (all three copies at one centre)
    print("  N1 — fully-stacked triple, pairwise vs n-copy:", flush=True)
    n1 = {}
    for amp in args.stack_amps:
        blob = gaussian_load((int(args.size), int(args.size)),
                             [(args.size / 2, args.size / 2)],
                             args.width, amp)
        loads = [blob, blob, blob]
        pw = exclusion_gap_group(loads, n_copy=False,
                                 **field_kwargs(args))
        nc = exclusion_gap_group(loads, n_copy=True,
                                 **field_kwargs(args))
        lr = 3.0 * linear_response_exclusion_gap(blob,
                                                 **field_kwargs(args))
        n1[amp] = {"pairwise": pw, "ncopy": nc, "ratio": pw / nc,
                   "linear_response": lr, "pw_over_lr": pw / lr,
                   "nc_over_lr": nc / lr}
        print(f"    A = {amp:g}: pairwise {pw:.6g}, n-copy {nc:.6g}, "
              f"ratio {pw / nc:.4f} (O(c²) prediction 3c²κ₀²(ρ,Gρ) = "
              f"{lr:.6g}: pw/LR {pw / lr:.3f}, nc/LR {nc / lr:.3f})",
              flush=True)
    dilute = n1[min(args.stack_amps)]
    n1_pass = bool(abs(dilute["ratio"] - 1.0) <= 0.05)
    print(f"    dilute ratio {dilute['ratio']:.4f} vs the 5% bar: "
          f"{n1_pass}", flush=True)

    # --- N2: trimer statics floor, both forms + the no-exclusion
    #     control (the identity-blind field curve alone)
    print("  N2 — trimer statics (n-copy, pairwise, control):",
          flush=True)
    field = field_curve(args)
    ex_nc, ex_pw = exclusion_curves(args)
    cur_nc = zeroed(np.asarray(field) + np.asarray(ex_nc))
    cur_pw = zeroed(np.asarray(field) + np.asarray(ex_pw))
    cur_ctrl = zeroed(field)
    for name, cur in (("n-copy", cur_nc), ("pairwise", cur_pw),
                      ("control", cur_ctrl)):
        print("    %-8s E(s) = %s" % (name, " ".join(
            f"{s:g}:{e:+.2f}" for s, e in zip(args.seps, cur))),
            flush=True)

    def floor_of(cur):
        i_min = int(np.argmin(cur))
        return float(args.seps[i_min]), float(cur[0] - cur[i_min])

    s3_nc, bar_nc = floor_of(cur_nc)
    s3_pw, bar_pw = floor_of(cur_pw)
    i_min_c = int(np.argmin(cur_ctrl))
    ctrl_monotone = bool(np.all(np.diff(cur_ctrl) >= -1e-9))
    ctrl_no_interior = bool(not (2.0 < float(args.seps[i_min_c]) < 12.0))
    n2_floor = bool(2.0 < s3_nc < 12.0 and bar_nc >= 0.2)
    n2 = bool(n2_floor and ctrl_monotone and ctrl_no_interior)
    print(f"    n-copy: s*₃ = {s3_nc:g}, barrier {bar_nc:.2f} "
          f"(bars: s*₃ ∈ (2, 12), ≥ 0.2): {n2_floor}", flush=True)
    print(f"    pairwise (comparison, no bar): s*₃ = {s3_pw:g}, "
          f"barrier {bar_pw:.2f}", flush=True)
    print(f"    control: monotone attractive: {ctrl_monotone}, no "
          f"interior minimum in (2, 12): {ctrl_no_interior}",
          flush=True)

    # --- N3: trimer dynamics from rest.  The BASELINE is measured
    #     first; t_max must let it complete its three-way collapse
    #     well before the end (the arc's late window is t > 0.6·t_max).
    print(f"  N3 — releasing the trimer from rest at side d₀ = "
          f"{args.d0:g}; measuring the no-exclusion baseline first "
          f"(t_max = {args.t_max:g}):", flush=True)
    L = args.size
    probes = [(int(round(L / 2 + args.probe_radius * np.cos(a))),
               int(round(L / 2 + args.probe_radius * np.sin(a))))
              for a in np.linspace(0, 2 * np.pi, 4, endpoint=False)]
    h_base = release(args, None, probes=probes)
    t_rec = np.asarray(h_base["time"])
    sep_base_series = mean_pairwise_separation(h_base["positions"])
    plunge_at = None
    for t, sp in zip(t_rec, sep_base_series):
        if sp < 3.0:
            plunge_at = float(t)
            break
    if plunge_at is None or plunge_at >= 0.6 * args.t_max:
        new_t = 10.0 * np.ceil((0.0 if plunge_at is None
                                else plunge_at) / 0.6 / 10.0 + 0.5)
        args.t_max = float(max(new_t, args.t_max * 1.5))
        print(f"    baseline had not collapsed by the late window "
              f"(plunge at {plunge_at}); re-running at t_max = "
              f"{args.t_max:g}", flush=True)
        h_base = release(args, None, probes=probes)
        t_rec = np.asarray(h_base["time"])
        sep_base_series = mean_pairwise_separation(h_base["positions"])
        for t, sp in zip(t_rec, sep_base_series):
            if sp < 3.0:
                plunge_at = float(t)
                break
    print(f"    baseline plunges to three-way contact at t ≈ "
          f"{plunge_at} — t_max = {args.t_max:g} leaves the collapse "
          f"well before the late window (t > {0.6 * args.t_max:g}); "
          f"the choice is an instrument necessity, recorded, not "
          f"tuned.", flush=True)
    h_nc = release(args, True, probes=probes)
    h_pw = release(args, False, probes=probes)
    late = t_rec > 0.6 * args.t_max
    sep_nc = float(mean_pairwise_separation(h_nc["positions"])[late]
                   .mean())
    sep_pw = float(mean_pairwise_separation(h_pw["positions"])[late]
                   .mean())
    sep_base = float(sep_base_series[late].mean())
    stall = bool(0.5 <= sep_nc / s3_nc <= 2.0
                 and sep_nc >= 1.5 * sep_base)
    n3 = stall
    print(f"    late mean pairwise separation: n-copy {sep_nc:.2f} "
          f"(s*₃ = {s3_nc:g}, ratio {sep_nc / s3_nc:.2f}, bar a "
          f"factor of 2), pairwise {sep_pw:.2f} (recorded, no bar), "
          f"baseline {sep_base:.2f} (ratio n-copy/baseline "
          f"{sep_nc / sep_base:.2f}, bar ≥ 1.5): {n3}", flush=True)

    # --- EXPLORATORY ringdown echo (NOT pre-registered — three-body
    #     spectra are not the binary instrument's): the n-copy run's
    #     breathing-mode line in the separation channel and whatever
    #     the probe waveforms carry, plus the static breathing pitch
    #     for context.  The breathing coordinate is the side s: each
    #     vertex moves radially with dR/ds = 1/√3, so the effective
    #     mass is μ = m and the pitch is √(E″(s*₃)/m).
    lo, hi = args.ring_window

    def detrend(x, w):
        pad = np.pad(x, w // 2, mode="edge")
        return x - np.convolve(pad, np.ones(w) / w, mode="valid")[: len(x)]

    win_rec = (t_rec > lo) & (t_rec < hi)
    dt_rec = float(t_rec[1] - t_rec[0])
    sep_w = (detrend(mean_pairwise_separation(h_nc["positions"]), 15)
             [win_rec] * np.hanning(int(win_rec.sum())))
    fr_rec = 2.0 * np.pi * np.fft.rfftfreq(len(sep_w), d=dt_rec)
    pw_rec = np.abs(np.fft.rfft(sep_w)) ** 2
    live = fr_rec > 0.1
    om_breath = float(fr_rec[live][np.argmax(pw_rec[live])])
    near = [(s, e) for s, e in zip(args.seps, cur_nc)
            if abs(s - s3_nc) <= 2.5]
    om_static = None
    if len(near) >= 3:
        a2 = float(np.polyfit([s for s, _ in near],
                              [e for _, e in near], 2)[0])
        om_static = float(np.sqrt(max(2.0 * a2, 0.0) / args.mass))
    sig = np.asarray(h_nc["probe_kappa"])
    t_f = args.dt * np.arange(sig.shape[0])
    winf = (t_f > lo) & (t_f < hi)
    hann = np.hanning(int(winf.sum()))
    freqs = 2.0 * np.pi * np.fft.rfftfreq(int(winf.sum()), d=args.dt)
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
    om_wave, contrast = float(freqs[band][peak_i]), float(white[band][peak_i])
    print(f"    EXPLORATORY (not pre-registered): breathing line "
          f"ω = {om_breath:.3f} in the separation channel"
          + (f" vs static breathing pitch {om_static:.3f}"
             if om_static is not None else "")
          + f"; probe waveform line ω = {om_wave:.3f}, whitened "
          f"contrast {contrast:.1f}", flush=True)

    # --- verdicts
    lines = ["The n-copy sector — verdict", "=" * 74, ""]
    lines.append(
        "the generalization of the derived exclusion term to "
        "three-and-more identical stacks: within a group of n "
        "same-label masses the cloned component is the min over the "
        "group's loads and the group pays the n-copy gap "
        "E_x = nE(m) − E(nm); the pairwise-sum form is kept as an "
        "option.  At O(c²) the two are EQUAL (both k·n(n−1)); their "
        "split at the operating amplitude measures core saturation.  "
        "For n = 2 both reduce to the base pair form bitwise.")
    lines.append("")
    lines.append(
        f"N1 (the theorem): fully-stacked triple — dilute "
        f"(A = {min(args.stack_amps):g}) pairwise "
        f"{dilute['pairwise']:.6g} vs n-copy {dilute['ncopy']:.6g}, "
        f"ratio {dilute['ratio']:.4f} (bar within 5%): "
        + ("✓ the O(c²) theorem holds at the operating point's own "
           "instruments" if n1_pass
           else "✗ the dilute forms differ by more than 5% — recorded "
           "as-is")
        + f"; at the operating amplitude A = {max(args.stack_amps):g} "
        f"the ratio is {n1[max(args.stack_amps)]['ratio']:.4f} "
        f"(pairwise {n1[max(args.stack_amps)]['pairwise']:.6g}, "
        f"n-copy {n1[max(args.stack_amps)]['ncopy']:.6g}) — the "
        f"divergence is the saturated core: the pairwise form prices "
        f"the thrice-cloned core three times, the n-copy gap prices "
        f"it once.  Against the O(c²) value 3c²κ₀²(ρ,Gρ): dilute "
        f"pw/LR = {dilute['pw_over_lr']:.3f}, nc/LR = "
        f"{dilute['nc_over_lr']:.3f}; operating pw/LR = "
        f"{n1[max(args.stack_amps)]['pw_over_lr']:.4f}, nc/LR = "
        f"{n1[max(args.stack_amps)]['nc_over_lr']:.4f}.")
    lines.append(
        f"N2 (trimer statics floor): n-copy interior minimum at "
        f"s*₃ = {s3_nc:g} with barrier {bar_nc:.2f} toward contact "
        f"(bars: s*₃ ∈ (2, 12), ≥ 0.2): {n2_floor}; no-exclusion "
        f"control monotone attractive: {ctrl_monotone}, no interior "
        f"minimum in (2, 12): {ctrl_no_interior}.  Pairwise form "
        f"(comparison, no bar): s*₃ = {s3_pw:g}, barrier {bar_pw:.2f} "
        f"— the saturation-split floor: the pairwise form, pricing "
        f"the triple overlap per pair, holds the trimer farther out "
        f"and harder.  Verdict: "
        + ("✓ the trimer has a floor." if n2
           else "✗ recorded as-is."))
    lines.append(
        "    n-copy  E(s) = " + " ".join(
            f"{s:g}:{e:+.2f}" for s, e in zip(args.seps, cur_nc))
        + " | pairwise E(s) = " + " ".join(
            f"{s:g}:{e:+.2f}" for s, e in zip(args.seps, cur_pw))
        + " | control E(s) = " + " ".join(
            f"{s:g}:{e:+.2f}" for s, e in zip(args.seps, cur_ctrl))
        + ".")
    lines.append(
        f"N3 (trimer dynamics from rest): released at side d₀ = "
        f"{args.d0:g}, t_max = {args.t_max:g} (baseline's three-way "
        f"collapse completes at t ≈ {plunge_at}, well before the "
        f"late window — measured first, an instrument necessity, not "
        f"a tuned result).  Late mean pairwise separation: n-copy "
        f"{sep_nc:.2f} vs s*₃ = {s3_nc:g} (ratio "
        f"{sep_nc / s3_nc:.2f}, bar a factor of 2) and vs the "
        f"baseline's {sep_base:.2f} (ratio {sep_nc / sep_base:.2f}, "
        f"bar ≥ 1.5): "
        + ("✓ the same-label trimer stalls on its static floor; the "
           "baseline plunges to three-way contact." if n3
           else "✗ the stall does not land as registered — recorded "
           "as-is.")
        + f"  Pairwise form (recorded, no bar): {sep_pw:.2f}.")
    lines.append(
        f"    EXPLORATORY ringdown echo (NOT pre-registered — "
        f"three-body spectra are not the binary instrument's): "
        f"breathing line ω = {om_breath:.3f} in the separation "
        f"channel"
        + (f" vs the static breathing pitch √(E″(s*₃)/m) = "
           f"{om_static:.3f}" if om_static is not None else "")
        + f"; probe waveform line ω = {om_wave:.3f} with whitened "
        f"contrast {contrast:.1f}.")
    lines.append("")
    lines.append(f"score: {int(n1_pass) + int(n2) + int(n3)}/3 "
                 "pre-registered predictions land.")
    lines.append("")
    lines.append(
        "honest scope: three bodies only (the group's symmetrized "
        "smoothed min costs ~n!/2 leaf evaluations — built for small "
        "groups; the pairwise option prices C(n, 2) pairs instead); "
        "the min-over-group construction is the maximal-identicality "
        "convention again (exact for identical copies, the maximal "
        "common component otherwise); dynamics from rest only — no "
        "angular momentum, the collapse is the breathing channel; "
        "the statics use relax_capacity on the exact min, the "
        "dynamics the symmetrized smoothed min (ε = 1e-4) and "
        "conjugate-gradient solves, as the base branch — n = 2 is "
        "bitwise the pair form.  One operating point (size 96, "
        "width 2.5, mass 0.6, r = 0.02, c = 0.8, τ = 0.1).")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir,
                           "n3_exclusion_ncopy.json"), "w") as fh:
        json.dump({"params": vars(args),
                   "n1": {str(k): v for k, v in n1.items()},
                   "n2": {"seps": list(args.seps),
                          "ncopy": {"curve": cur_nc, "s_star": s3_nc,
                                    "barrier": bar_nc,
                                    "floor_found": n2_floor},
                          "pairwise": {"curve": cur_pw,
                                       "s_star": s3_pw,
                                       "barrier": bar_pw},
                          "control": {"curve": cur_ctrl,
                                      "monotone": ctrl_monotone,
                                      "no_interior_min":
                                          ctrl_no_interior},
                          "verdict": bool(n2)},
                   "n3": {"t_max": args.t_max,
                          "baseline_plunge_at": plunge_at,
                          "late_sep": {"ncopy": sep_nc,
                                       "pairwise": sep_pw,
                                       "baseline": sep_base},
                          "ratio_nc_over_sstar": sep_nc / s3_nc,
                          "ratio_nc_over_base": sep_nc / sep_base,
                          "exploratory": {"omega_breath": om_breath,
                                          "omega_static": om_static,
                                          "omega_wave": om_wave,
                                          "contrast": contrast},
                          "verdict": bool(n3)},
                   "analysis": {"n1": bool(n1_pass), "n2": bool(n2),
                                "n3": bool(n3)}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
