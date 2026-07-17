"""The exclusion core: no-cloning as degeneracy pressure, and the ringdown regained.

The plunge experiment ended every binary in a silent merged blob.  The
URP's own exclusion idea — *in a structure, adding the same distinction
does not expand the structure* — suggests what the model was missing.
Calibration first showed the obvious implementation fails: **saturating
the load does not repel** (the free energy's binding comes from the
concavity of ``F(ρ) = rcρ/2(r+cρ)``, so capping ρ makes merging cheaper,
measured: the E(s) curve barely moves).  The faithful implementation must
make duplication *cost* capacity: the minimal convex penalty

    E_x = (b/2)·∫ ρ_tot²   —   zero for separated structures,
                                quadratic where identical distinction stacks,

whose exact gradient is a **contact force**: overlapping loads repel below
the footprint scale.  ``E_x(2ρ) > 2·E_x(ρ)`` — the clone is refused —
degeneracy pressure, entering exactly as contact terms do in nuclear
equations of state.  Statics confirm: at b = 0.2 the two-body energy turns
from monotone-attractive into a curve with an **interior minimum**
(s* ≈ 6, a 0.45 barrier against contact).

The dynamical consequences are the registered predictions: the trench
plunge still happens (it is orbit-scale physics), but it must now end on
the exclusion floor rather than at a point — a **contact binary** — and a
pair held open at finite size has a real quadrupole moment, so the
previously *silent* ringdown should become audible, at the selection
rule's frequency.

Pre-registered predictions:

X1. **The exclusion barrier**: with the contact term (b = 0.2) the static
    two-body energy has an interior minimum at s* ∈ (2, 10) and rises by
    ≥ 0.2 from s* to s = 1, while the b = 0 control is monotone
    attractive into contact.
X2. **The collapse halts on the exclusion floor**: the released binary's
    late-time mean separation is within a factor of 2 of the static s*
    and ≥ 2× the b = 0 baseline's — the plunge ends in a contact binary,
    not a point.
X3. **The ringdown returns, at the exclusion well's own pitch**: the
    contact binary is no longer silent — the waveform carries a line
    (whitened contrast ≥ 3) within 25% of the separation channel's
    libration frequency, which itself sits within a factor of 2 of the
    *static* prediction ``ω = √(E″(s*)/(m/2))`` read off the X1 energy
    curve's curvature — the statics predict the ringdown's frequency.
    (Calibration note, kept: the rotational 2Ω line dies with the
    rotation — drag drains the angular momentum before the late window —
    so the audible mode is the **radial libration** on the exclusion
    floor; a spun-up variant is the door to the 2Ω version.)

Usage::

    python experiments/n3_exclusion_core.py --output-dir artifacts/n3_exclusion
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
from project_genesis.capacity_waves import evolve_inertial_retarded


def field_kwargs(args):
    return dict(kappa_diffusion=1.0, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0)


def pair_energy(args, s: float, b: float) -> float:
    """Static two-body energy at separation s: relaxed F[κ] + contact E_x."""
    n = int(args.size)
    tot = (gaussian_load((n, n), [(n / 2 - s / 2, n / 2)], args.width,
                         args.mass)
           + gaussian_load((n, n), [(n / 2 + s / 2, n / 2)], args.width,
                           args.mass))
    kappa = relax_capacity(tot, **field_kwargs(args))
    return (capacity_free_energy(kappa, tot, **field_kwargs(args))
            + 0.5 * b * float(np.sum(tot ** 2)))


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


def release(args, b: float, probes=None):
    L = args.size
    v0 = circular_speed(args)
    return evolve_inertial_retarded(
        [[L / 2 - args.d0 / 2, L / 2], [L / 2 + args.d0 / 2, L / 2]],
        [[0.0, +v0], [0.0, -v0]], [args.mass] * 2, shape=(int(L), int(L)),
        width=args.width, tau=args.tau, dt=args.dt,
        steps=int(args.t_max / args.dt), record_every=10, probes=probes,
        contact_b=b, **field_kwargs(args))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=float, default=96.0)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--mass", type=float, default=0.6)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--b", type=float, default=0.2)
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
                   default="artifacts/n3_exclusion")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.t_max = 180.0
        args.seps = [1.0, 3.0, 5.0, 6.0, 8.0, 16.0]

    os.makedirs(args.output_dir, exist_ok=True)
    print("the exclusion core: no-cloning as degeneracy pressure",
          flush=True)

    # --- X1: statics
    curves = {}
    for b in (args.b, 0.0):
        e = np.array([pair_energy(args, s, b) for s in args.seps])
        curves[b] = e - e[-1]
        print(f"  b={b:g}: E(s) = "
              + " ".join(f"{s:g}:{d:+.2f}"
                         for s, d in zip(args.seps, curves[b])), flush=True)
    ex = curves[args.b]
    i_min = int(np.argmin(ex))
    s_star = float(args.seps[i_min])
    barrier = float(ex[0] - ex[i_min])
    ctrl_monotone = bool(np.all(np.diff(curves[0.0]) >= -1e-9))
    x1 = (2.0 < s_star < 10.0 and barrier >= 0.2 and ctrl_monotone)
    print(f"  s* = {s_star:g}, barrier(s→1) = {barrier:.2f}, "
          f"b=0 monotone attractive: {ctrl_monotone}", flush=True)

    # --- X2/X3: the released binary
    L = args.size
    probes = [(int(round(L / 2 + args.probe_radius * np.cos(a))),
               int(round(L / 2 + args.probe_radius * np.sin(a))))
              for a in np.linspace(0, 2 * np.pi, 4, endpoint=False)]
    hx = release(args, args.b, probes=probes)
    h0 = release(args, 0.0)
    t_rec = np.asarray(hx["time"])
    late = t_rec > 0.6 * args.t_max
    sep_x = float(np.asarray(hx["separation"])[late].mean())
    sep_0 = float(np.asarray(h0["separation"])[late].mean())
    x2 = (0.5 <= sep_x / s_star <= 2.0) and sep_x >= 2.0 * sep_0
    print(f"  late mean separation: {sep_x:.2f} (exclusion) vs "
          f"{sep_0:.2f} (b = 0); s* = {s_star:g}", flush=True)

    # --- X3: libration channel, static pitch, whitened waveform spectrum.
    # The ring lives in the post-plunge / pre-settle window: the same
    # friction that forbids the inspiral damps the libration within a few
    # cycles, so the late window is already quiet (measured — a comb of
    # window-leakage lines, kept out by detrending + Hann + early window).
    def detrend(x, w):
        pad = np.pad(x, w // 2, mode="edge")
        return x - np.convolve(pad, np.ones(w) / w, mode="valid")[: len(x)]

    pos = np.asarray(hx["positions"])
    sv = pos[:, 0, :] - pos[:, 1, :]
    phi = np.unwrap(np.arctan2(sv[:, 1], sv[:, 0]))
    lo, hi = args.ring_window
    win_rec = (t_rec > lo) & (t_rec < hi)
    om_rot = float(np.abs(np.polyfit(t_rec[win_rec], phi[win_rec], 1)[0]))
    dt_rec = float(t_rec[1] - t_rec[0])
    sep_w = (detrend(np.asarray(hx["separation"]), 15)[win_rec]
             * np.hanning(int(win_rec.sum())))
    fr_rec = 2.0 * np.pi * np.fft.rfftfreq(len(sep_w), d=dt_rec)
    pw_rec = np.abs(np.fft.rfft(sep_w)) ** 2
    live = fr_rec > 0.1
    om_lib = float(fr_rec[live][np.argmax(pw_rec[live])])
    # static pitch: parabola through the E(s) points near s*
    near = [(s, e) for s, e in zip(args.seps, ex) if abs(s - s_star) <= 2.5]
    a2 = float(np.polyfit([s for s, _ in near], [e for _, e in near], 2)[0])
    om_static = float(np.sqrt(max(2.0 * a2, 0.0) / (args.mass / 2.0)))
    # waveform channel
    sig = np.asarray(hx["probe_kappa"])
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
    peak_w = float(freqs[band][peak_i])
    contrast = float(white[band][peak_i])
    x3 = (contrast >= 3.0
          and abs(peak_w - om_lib) / om_lib <= 0.25
          and 0.5 <= om_lib / om_static <= 2.0)
    print(f"  libration ω = {om_lib:.3f} (static pitch √(E″/μ) = "
          f"{om_static:.3f}; rotation died: Ω_rot = {om_rot:.4f}); "
          f"waveform line at ω = {peak_w:.3f}, whitened contrast "
          f"{contrast:.1f}", flush=True)

    lines = ["The exclusion core — verdict", "=" * 74, ""]
    lines.append(
        f"X1 (the exclusion barrier): interior minimum at s* = {s_star:g}, "
        f"barrier {barrier:.2f} toward contact; b = 0 control monotone "
        f"attractive — "
        + ("✓ adding the same distinction does not expand the structure — "
           "it costs: degeneracy pressure exists in the statics." if x1
           else "✗ no interior minimum / barrier."))
    lines.append(
        f"X2 (collapse halts on the exclusion floor): late separation "
        f"{sep_x:.2f} vs s* = {s_star:g} (ratio {sep_x/s_star:.2f}) and vs "
        f"baseline {sep_0:.2f} — "
        + ("✓ the trench plunge still happens, but it ends in a contact "
           "binary held open by exclusion — the collapse has a floor."
           if x2 else "✗ the pair does not stall at the static floor."))
    lines.append(
        f"X3 (the ringdown returns, at the well's own pitch): waveform "
        f"line ω = {peak_w:.3f} vs libration {om_lib:.3f} vs static "
        f"√(E″(s*)/μ) = {om_static:.3f}; whitened contrast "
        f"{contrast:.1f} (silent-blob baseline: 1.45) — "
        + ("✓ held open at finite size, the merged system finally rings — "
           "and the X1 energy curve's curvature predicted the pitch."
           if x3 else
           "✗ still silent, or the line is not the libration's."))
    lines.append("")
    lines.append(f"score: {int(x1)+int(x2)+int(x3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: the contact term is a minimal VARIANT — the "
        "smallest convex (no-cloning) penalty, added to the matter sector "
        "only (the field equation is untouched); b = 0.2 was chosen in "
        "static calibration for an interior minimum away from both walls "
        "(b = 0.1 is nearly flat, b = 0.5 pushes s* to 8), and the failed "
        "load-saturation implementation is kept in the record with its "
        "mechanism (binding here is concavity of F(ρ) — capping ρ helps "
        "merging). One (mass, coupling); τ = 0.1; the rotational 2Ω line "
        "dies with the drag-drained angular momentum, so the audible mode "
        "is the radial libration (a spun-up variant is the door to the "
        "2Ω version); the static pitch uses a coarse parabola through the "
        "E(s) grid near s*; the X3 band is [0.1, 1.6].")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_exclusion_core.json"),
              "w") as fh:
        json.dump({"params": vars(args),
                   "statics": {str(b): curves[b].tolist() for b in curves},
                   "s_star": s_star, "barrier": barrier,
                   "late_sep": {"exclusion": sep_x, "baseline": sep_0},
                   "ringdown": {"omega_rot": om_rot, "omega_lib": om_lib,
                                "omega_static": om_static, "peak": peak_w,
                                "contrast": contrast},
                   "analysis": {"x1": bool(x1), "x2": bool(x2),
                                "x3": bool(x3)}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
