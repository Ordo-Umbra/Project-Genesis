"""Deriving the exclusion coefficient from the framework's own counting.

The exclusion core (`experiments/n3_exclusion_core.py`) landed its three
predictions with a **hand-picked** contact coefficient ``b = 0.2`` — the
one free parameter in the no-cloning term ``E_x = (b/2)·∫ρ_tot²``.  This
experiment removes it.  The framework's homogeneous steady-state capacity
free energy density at load ρ (κ₀ = 1, gradient terms neglected) is

    F(ρ) = r·c·ρ / (2·(r + c·ρ))   — CONCAVE in ρ (why merging is cheap).

The URP exclusion principle (*adding the same distinction does not expand
the structure*) prices a stack of two identical copies at the extensive
cost ``2F(ρ)``, not the concave ``F(2ρ)``.  The exclusion energy density
is the gap, in closed form, with **no free parameters**:

    e(ρ) = 2F(ρ) − F(2ρ) = c²·r·ρ² / ((r + cρ)·(r + 2cρ)) .

Its verified properties: dilute limit ``e ≈ (b/2)ρ²`` with
``b = 2c²/r = 64`` at the operating point (the calibrated 0.2 is NOT in
this family); clone refusal ``E_x(2ρ) > 2E_x(ρ)`` exactly for
``ρ < r/(2c) = 0.0125``, inverting above; saturation ``e → r/2`` per
site; ``e'(ρ) > 0`` always (repulsive in overlap, fading as
``3r²/(4cρ²)`` deep in the saturated regime).  The physical reading is
``b(ρ) = 2c²ℓ²(ρ)/D`` with the loaded screening length
``ℓ² = D/(r + cρ)``.  The contact term of ``evolve_inertial_retarded``
is replaced by the exact gradient of ``E_x = Σ e(ρ_tot)``
(``contact_derived=True``), keeping the Lyapunov bookkeeping.

Pre-registered predictions:

D1. **The derivation's self-check (convexity window)**: on a ρ grid,
    ``E_x(2ρ) > 2E_x(ρ)`` for ``ρ < r/(2c)`` and the inequality INVERTS
    above ``r/(2c)`` — exactly one sign change, at ``r/(2c)``.  (Algebra
    guarantees this; the check guards the implementation.)
D2. **Statics, same bars as X1**: the two-body energy
    ``E(s) = F[κ relaxed] + Σ e(ρ_tot)`` has an interior minimum at
    ``s* ∈ (2, 10)`` with barrier ``≥ 0.2`` toward ``s = 1``, and the
    no-exclusion control is monotone attractive.  RECORD PASS OR FAIL
    AS-IS — the derived b_eff at the contact-overlap density is ~0.05,
    far below the calibrated 0.2, so the barrier is genuinely in doubt.
D3. **Dynamics, conditioned on D2's floor — run regardless**: the
    released binary per the X2/X3 protocol, reporting late mean
    separation vs the static s* (if any) and vs the no-exclusion
    baseline, the ringdown whitened contrast (same detrend+Hann
    pipeline), and the libration frequency vs the static pitch
    ``√(E″(s*)/μ)`` if a minimum exists.  Verdict: the X2+X3 bars when
    a floor exists; if D2 finds no floor, the registered expectation is
    PLUNGE-TO-CONTACT, BASELINE-LIKE: late separation within a factor
    of 2 of the no-exclusion baseline's, and a whitened contrast within
    a factor of 2 of the baseline's own (the ringdown channel is
    compared AGAINST THE CONTROL through the identical pipeline, not
    against an absolute silence bar — in the no-floor case the ring
    window catches the post-plunge slosh, which both runs ring).
    Stated here in advance, then recorded as it falls.

Usage::

    python experiments/n3_exclusion_derived.py \
        --output-dir artifacts/n3_exclusion_derived
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
    exclusion_energy_density,
)


def field_kwargs(args):
    return dict(kappa_diffusion=1.0, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0)


def total_load(args, s: float) -> np.ndarray:
    """Two Gaussian masses separation s apart, centred on the grid."""
    n = int(args.size)
    return (gaussian_load((n, n), [(n / 2 - s / 2, n / 2)], args.width,
                          args.mass)
            + gaussian_load((n, n), [(n / 2 + s / 2, n / 2)], args.width,
                            args.mass))


def pair_energy_derived(args, s: float) -> float:
    """Static two-body energy: relaxed F[κ] + Σ e(ρ_tot) (derived form)."""
    tot = total_load(args, s)
    kappa = relax_capacity(tot, **field_kwargs(args))
    ex = float(np.sum(exclusion_energy_density(
        tot, kappa_recovery=args.r, kappa_consumption=args.c)))
    return capacity_free_energy(kappa, tot, **field_kwargs(args)) + ex


def pair_energy_control(args, s: float) -> float:
    """The no-exclusion control: relaxed F[κ] only."""
    tot = total_load(args, s)
    kappa = relax_capacity(tot, **field_kwargs(args))
    return capacity_free_energy(kappa, tot, **field_kwargs(args))


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


def release(args, v0: float, derived: bool, probes=None):
    L = args.size
    return evolve_inertial_retarded(
        [[L / 2 - args.d0 / 2, L / 2], [L / 2 + args.d0 / 2, L / 2]],
        [[0.0, +v0], [0.0, -v0]], [args.mass] * 2, shape=(int(L), int(L)),
        width=args.width, tau=args.tau, dt=args.dt,
        steps=int(args.t_max / args.dt), record_every=10, probes=probes,
        contact_derived=derived, **field_kwargs(args))


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
    p.add_argument("--output-dir", type=str,
                   default="artifacts/n3_exclusion_derived")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.t_max = 180.0
        args.seps = [1.0, 3.0, 5.0, 6.0, 8.0, 16.0]

    os.makedirs(args.output_dir, exist_ok=True)
    print("deriving the exclusion coefficient from the framework's own "
          "counting", flush=True)

    # --- for the record: the derived form's own numbers at this point
    r, c = args.r, args.c
    b_dilute = 2.0 * c * c / r
    rho_refusal = r / (2.0 * c)
    saturation = r / 2.0

    def e_of(rho):
        return exclusion_energy_density(rho, kappa_recovery=r,
                                        kappa_consumption=c)

    def b_eff(rho):
        return float(2.0 * e_of(rho) / rho ** 2)

    rho_probe = 0.1
    rho_contact = float(np.max(total_load(args, 1.0)))
    print(f"  derived form: dilute b = 2c²/r = {b_dilute:g}; "
          f"refusal window ρ < r/(2c) = {rho_refusal:g}; "
          f"saturation r/2 = {saturation:g}", flush=True)
    print(f"  b_eff(ρ) = 2e(ρ)/ρ²: at ρ = {rho_probe:g}: "
          f"{b_eff(rho_probe):.4g}; at the contact-overlap density "
          f"ρ = {rho_contact:.3f} (peak total load at s = 1): "
          f"{b_eff(rho_contact):.4g}", flush=True)

    # --- D1: the convexity window, implementation self-check
    rhos = np.geomspace(1e-6 * r / c, 1e3 * rho_refusal, 4000)
    gap = e_of(2.0 * rhos) - 2.0 * e_of(rhos)
    below = rhos < rho_refusal
    one_crossing = bool(np.all(gap[below] > 0) and np.all(gap[~below] < 0))
    sign_changes = int(np.sum(gap[:-1] * gap[1:] < 0))
    i_cross = int(np.argmax(gap < 0)) if np.any(gap < 0) else -1
    rho_cross = float(np.sqrt(rhos[i_cross - 1] * rhos[i_cross])) \
        if 0 < i_cross else float("nan")
    d1 = (one_crossing and sign_changes == 1
          and abs(rho_cross - rho_refusal) / rho_refusal < 0.02)
    print(f"  D1: E_x(2ρ)−2E_x(ρ) > 0 below, < 0 above; single sign "
          f"change at ρ = {rho_cross:.6g} (theory r/(2c) = "
          f"{rho_refusal:.6g}) — {'PASS' if d1 else 'FAIL'}", flush=True)

    # --- D2: statics
    e_der = np.array([pair_energy_derived(args, s) for s in args.seps])
    e_ctl = np.array([pair_energy_control(args, s) for s in args.seps])
    cur_der = e_der - e_der[-1]
    cur_ctl = e_ctl - e_ctl[-1]
    print("  derived: E(s) = "
          + " ".join(f"{s:g}:{d:+.2f}" for s, d in zip(args.seps, cur_der)),
          flush=True)
    print("  control: E(s) = "
          + " ".join(f"{s:g}:{d:+.2f}" for s, d in zip(args.seps, cur_ctl)),
          flush=True)
    i_min = int(np.argmin(cur_der))
    s_star = float(args.seps[i_min])
    barrier = float(cur_der[0] - cur_der[i_min])
    ctrl_monotone = bool(np.all(np.diff(cur_ctl) >= -1e-9))
    floor_exists = bool(2.0 < s_star < 10.0 and barrier >= 0.2)
    d2 = floor_exists and ctrl_monotone
    print(f"  s* = {s_star:g}, barrier(s→1) = {barrier:.2f}, "
          f"control monotone attractive: {ctrl_monotone} — "
          f"D2 {'PASS' if d2 else 'FAIL (recorded as-is)'}", flush=True)

    # --- D3: the released binary (run regardless of D2)
    if not floor_exists:
        print("  D3 registered expectation (no static floor): "
              "plunge-to-contact, BASELINE-LIKE — late separation "
              "within a factor of 2 of the no-exclusion baseline's, "
              "and a whitened contrast within a factor of 2 of the "
              "baseline's own through the identical pipeline.",
              flush=True)
    L = args.size
    probes = [(int(round(L / 2 + args.probe_radius * np.cos(a))),
               int(round(L / 2 + args.probe_radius * np.sin(a))))
              for a in np.linspace(0, 2 * np.pi, 4, endpoint=False)]
    v0 = circular_speed(args)
    hx = release(args, v0, derived=True, probes=probes)
    h0 = release(args, v0, derived=False, probes=probes)
    t_rec = np.asarray(hx["time"])
    late = t_rec > 0.6 * args.t_max
    sep_x = float(np.asarray(hx["separation"])[late].mean())
    sep_0 = float(np.asarray(h0["separation"])[late].mean())
    print(f"  late mean separation: {sep_x:.2f} (derived) vs "
          f"{sep_0:.2f} (no exclusion); s* = {s_star:g}", flush=True)

    # --- ringdown pipeline: identical detrend + Hann + whitening,
    #     applied to BOTH runs (the control is the no-floor reference)
    def detrend(x, w):
        pad = np.pad(x, w // 2, mode="edge")
        return x - np.convolve(pad, np.ones(w) / w, mode="valid")[: len(x)]

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

    lo, hi = args.ring_window
    pos = np.asarray(hx["positions"])
    sv = pos[:, 0, :] - pos[:, 1, :]
    phi = np.unwrap(np.arctan2(sv[:, 1], sv[:, 0]))
    win_rec = (t_rec > lo) & (t_rec < hi)
    om_rot = float(np.abs(np.polyfit(t_rec[win_rec], phi[win_rec], 1)[0]))
    dt_rec = float(t_rec[1] - t_rec[0])
    sep_w = (detrend(np.asarray(hx["separation"]), 15)[win_rec]
             * np.hanning(int(win_rec.sum())))
    fr_rec = 2.0 * np.pi * np.fft.rfftfreq(len(sep_w), d=dt_rec)
    pw_rec = np.abs(np.fft.rfft(sep_w)) ** 2
    live = fr_rec > 0.1
    om_lib = float(fr_rec[live][np.argmax(pw_rec[live])])
    om_static = None
    if floor_exists:
        near = [(s, e) for s, e in zip(args.seps, cur_der)
                if abs(s - s_star) <= 2.5]
        a2 = float(np.polyfit([s for s, _ in near],
                              [e for _, e in near], 2)[0])
        om_static = float(np.sqrt(max(2.0 * a2, 0.0) / (args.mass / 2.0)))
    peak_w, contrast = waveform_spectrum(hx)
    peak_0, contrast_0 = waveform_spectrum(h0)
    print(f"  libration ω = {om_lib:.3f} (static pitch "
          + (f"√(E″/μ) = {om_static:.3f}" if om_static is not None
             else "n/a — no static floor")
          + f"; rotation died: Ω_rot = {om_rot:.4f}); waveform line at "
          f"ω = {peak_w:.3f}, whitened contrast {contrast:.1f} "
          f"(no-exclusion control: ω = {peak_0:.3f}, contrast "
          f"{contrast_0:.1f})", flush=True)

    if floor_exists:
        stall = (0.5 <= sep_x / s_star <= 2.0) and sep_x >= 2.0 * sep_0
        ring = (contrast >= 3.0
                and abs(peak_w - om_lib) / om_lib <= 0.25
                and 0.5 <= om_lib / om_static <= 2.0)
        d3 = stall and ring
    else:
        plunge = 0.5 <= sep_x / sep_0 <= 2.0
        baseline_like = 0.5 <= contrast / contrast_0 <= 2.0
        d3 = plunge and baseline_like

    lines = ["Deriving the exclusion coefficient — verdict", "=" * 74, ""]
    lines.append(
        f"the derived form: e(ρ) = c²rρ²/((r+cρ)(r+2cρ)) — no free "
        f"parameters.  dilute b = 2c²/r = {b_dilute:g} (calibrated b = "
        f"0.2 is not in this family); b_eff(ρ = {rho_probe:g}) = "
        f"{b_eff(rho_probe):.4g}; b_eff(ρ_contact = {rho_contact:.2f}) "
        f"= {b_eff(rho_contact):.4g}; refusal window ρ < r/(2c) = "
        f"{rho_refusal:g}; saturation r/2 = {saturation:g}.")
    lines.append("")
    lines.append(
        f"D1 (the derivation's self-check): E_x(2ρ) > 2E_x(ρ) below "
        f"r/(2c), inverted above; single sign change at ρ = "
        f"{rho_cross:.6g} vs theory {rho_refusal:.6g} — "
        + ("✓ the implementation carries the exact convexity window."
           if d1 else "✗ the implementation does not reproduce the "
           "algebra's sign structure."))
    lines.append(
        f"D2 (statics, X1 bars): interior minimum at s* = {s_star:g}, "
        f"barrier {barrier:.2f} toward contact; no-exclusion control "
        f"monotone attractive: {ctrl_monotone} — "
        + ("✓ the derived, parameter-free term still buys an exclusion "
           "floor in the statics." if d2
           else "✗ the derived term does not buy the X1 statics — "
           "recorded as-is."))
    if floor_exists:
        lines.append(
            f"D3 (dynamics, X2+X3 bars — a floor exists): late "
            f"separation {sep_x:.2f} vs s* = {s_star:g} (ratio "
            f"{sep_x/s_star:.2f}) and vs baseline {sep_0:.2f}; waveform "
            f"line ω = {peak_w:.3f} vs libration {om_lib:.3f} vs static "
            f"pitch {om_static:.3f}; whitened contrast {contrast:.1f} — "
            + ("✓ the collapse stalls on the derived floor and the "
               "contact binary rings at the statics' pitch." if d3
               else "✗ the dynamics do not meet the X2+X3 bars — "
               "recorded as-is."))
    else:
        lines.append(
            f"D3 (dynamics — no static floor, registered expectation "
            f"was plunge-to-contact, baseline-like): late separation "
            f"{sep_x:.2f} vs baseline {sep_0:.2f} (ratio "
            f"{sep_x/sep_0:.2f}); whitened contrast {contrast:.1f} vs "
            f"the control's own {contrast_0:.1f} through the identical "
            f"pipeline (ratio {contrast/contrast_0:.2f}) — "
            + ("✓ as predicted, no floor in the statics means no floor "
               "in the dynamics: the plunge and the post-plunge slosh "
               "are the baseline's." if d3
               else "✗ the dynamics depart from the registered "
               "baseline-like expectation — recorded as-is."))
    lines.append("")
    lines.append(f"score: {int(d1)+int(d2)+int(d3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: the derivation uses the HOMOGENEOUS F(ρ) — the "
        "gradient energy is neglected (ξ = √(D/r) = 7.07 vs blob width "
        "2.5 at the operating point — not negligible; matter-sector "
        "term only).  The load field cannot tell same-distinction from "
        "different-distinction stacking: e(ρ_tot) prices ALL overlap as "
        "duplication (maximal-identicality, conservative).  This is the "
        "2-copy sector of the general n-copy form nF(ρ) − F(nρ); only "
        "2-body overlap is probed.  One operating point (size 96, width "
        "2.5, mass 0.6, r = 0.02, c = 0.8, τ = 0.1).  The blob's "
        "working densities (peak load 0.6) sit deep above the refusal "
        "window ρ < 0.0125, so at the operating point the derived term "
        "acts in its saturated, inverted-convexity regime — b_eff = "
        "0.05 at the blob peak ρ = 0.6 and 0.014 at the contact "
        "overlap ρ ≈ 1.2 — and the calibrated b = 0.2's status is that "
        "of a phenomenological surrogate for this density-dependent "
        "form (b_eff(ρ) = 0.2 at ρ ≈ 0.30, the skirt of the blob); "
        "whether the surrogate was generous or conservative is exactly "
        "what D2/D3 measure.  The ringdown analysis pipeline is "
        "unchanged from the exclusion core (detrend + Hann, whitened "
        "contrast, band [0.1, 1.6]).  Instrument note, measured: in "
        "the no-floor case the plunge lands at t ≈ 8–9 and the "
        "[15, 95] ring window is dominated by the post-plunge slosh "
        "of the merged pair — the no-exclusion CONTROL rings in the "
        "same window with a whitened contrast of the same order — so "
        "the no-floor D3 comparison is made against the control's own "
        "contrast, not against an absolute silence bar (the "
        "silent-blob figure of the exclusion core belongs to its "
        "late, settled window, not to this one).")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir,
                           "n3_exclusion_derived.json"), "w") as fh:
        json.dump({"params": vars(args),
                   "derived_form": {"b_dilute": b_dilute,
                                    "rho_refusal": rho_refusal,
                                    "saturation": saturation,
                                    "b_eff_at_0.1": b_eff(rho_probe),
                                    "rho_contact": rho_contact,
                                    "b_eff_at_contact":
                                        b_eff(rho_contact)},
                   "d1": {"rho_cross": rho_cross,
                          "sign_changes": sign_changes,
                          "pass": bool(d1)},
                   "statics": {"derived": cur_der.tolist(),
                               "control": cur_ctl.tolist()},
                   "s_star": s_star, "barrier": barrier,
                   "floor_exists": floor_exists,
                   "late_sep": {"derived": sep_x, "baseline": sep_0},
                   "ringdown": {"omega_rot": om_rot, "omega_lib": om_lib,
                                "omega_static": om_static,
                                "peak": peak_w, "contrast": contrast,
                                "control_peak": peak_0,
                                "control_contrast": contrast_0},
                   "analysis": {"d1": bool(d1), "d2": bool(d2),
                                "d3": bool(d3)}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
