"""The quadrupole line: what a binary broadcasts through the κ field.

Retarded κ-gravity made binaries inspiral; this experiment points a
spectrometer at one.  A rigidly-rotating equal-mass pair (kinematically
driven, so the lines are razor-sharp) stirs the finite-speed capacity field,
and probes on rings read the disturbance spectrum.

The registered centrepiece is an **exact symmetry argument** — the analogue
of general relativity's "no dipole radiation; the quadrupole leads":
rotating an equal-mass pair by π is the same as advancing it half an orbital
period, so the source (hence the linear field response at any point) is
periodic with period ``π/Ω`` — the spectrum may contain **only even
harmonics of Ω**.  The statement is regime-independent (it survives
damping, screening, lattice anisotropy — the square lattice is itself
π-symmetric) and is falsified by any resolved odd line.  A single mass
driven on the same circle has no half-period symmetry: its dipole line at
Ω must appear — the control that shows the 2Ω selection is the binary's
symmetry speaking, not the instrument's.

The medium adds its own derivable number: driven waves in the damped
telegrapher field carry the complex wavenumber

    k = √( (τω² − m² + iω) / D ) ,   m² = r + c·ρ ≈ r  (vacuum) ,

so the 2Ω line's amplitude must fall as ``e^{−Im(k)·R}/√R`` across probe
rings — the resistive κ-vacuum's absorption length, predicted with no fit
freedom.

Pre-registered predictions:

Q1. **Even harmonics only — the quadrupole selection rule**: the binary's
    dominant non-DC line sits at 2Ω (± 2 FFT bins), and the odd-harmonic
    power (Ω + 3Ω) is < 10% of the even-harmonic power (2Ω + 4Ω).
Q2. **The dipole control**: the single driven mass's dominant line sits at
    Ω (± 2 bins) — remove the symmetry and the odd line returns.
Q3. **The medium's complex wavenumber**: the 2Ω amplitude across rings
    fits ``ln(A·√R) = const − Im(k)·R`` with the decay length within a
    factor of 2 of the derived ``1/Im(k)``.

Usage::

    python experiments/n3_quadrupole_line.py --output-dir artifacts/n3_quad
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import gaussian_load, relax_capacity
from project_genesis.capacity_waves import step_capacity_wave


def field_kwargs(args):
    return dict(kappa_diffusion=1.0, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0)


def ring_points(center, radius, n_angles):
    return [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a))
            for a in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)]


def driven_run(args, binary: bool) -> dict:
    """Rigidly rotate the source at Ω; record δκ(t) on probe rings."""
    n = int(args.size)
    fkw = field_kwargs(args)
    ctr = np.array([n / 2.0, n / 2.0])

    def positions(t):
        a = args.omega * t
        u = np.array([np.cos(a), np.sin(a)])
        if binary:
            return [tuple(ctr + args.d0 / 2.0 * u),
                    tuple(ctr - args.d0 / 2.0 * u)]
        return [tuple(ctr + args.d0 / 2.0 * u)]

    def load_of(t):
        return gaussian_load((n, n), positions(t), args.width, args.mass)

    kappa = relax_capacity(load_of(0.0), **fkw)
    kdot = np.zeros_like(kappa)
    probes = {R: [(int(round(x)) % n, int(round(y)) % n)
                  for x, y in ring_points(ctr, R, args.n_angles)]
              for R in args.rings}
    steps_skip = int(args.t_skip / args.dt)
    steps_rec = int(args.t_record / args.dt)
    series = {R: [[] for _ in probes[R]] for R in args.rings}
    for i in range(steps_skip + steps_rec):
        t = (i + 1) * args.dt
        kappa, kdot = step_capacity_wave(kappa, kdot, load_of(t),
                                         tau=args.tau, dt=args.dt, **fkw)
        if i >= steps_skip:
            for R, pts in probes.items():
                for j, (px, py) in enumerate(pts):
                    series[R][j].append(float(kappa[px, py]))
    out = {}
    for R in args.rings:
        spectra = []
        for s in series[R]:
            a = np.asarray(s) - np.mean(s)
            spectra.append(np.abs(np.fft.rfft(a)) ** 2)
        out[R] = np.mean(spectra, axis=0)
    out["freqs"] = 2.0 * np.pi * np.fft.rfftfreq(steps_rec, d=args.dt)
    return out


def line_power(freqs, spec, omega, width_bins=2):
    i = int(np.argmin(np.abs(freqs - omega)))
    lo, hi = max(i - width_bins, 0), min(i + width_bins + 1, len(spec))
    return float(spec[lo:hi].sum()), i


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=float, default=96.0)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--mass", type=float, default=1.2)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--d0", type=float, default=12.0)
    p.add_argument("--omega", type=float, default=0.8 / 6.0)
    p.add_argument("--rings", type=float, nargs="+",
                   default=[12.0, 17.0, 22.0, 27.0])
    p.add_argument("--spec-ring", type=float, default=12.0)
    p.add_argument("--n-angles", type=int, default=8)
    p.add_argument("--t-skip", type=float, default=100.0)
    p.add_argument("--t-record", type=float, default=950.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_quad")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.t_skip, args.t_record = 60.0, 450.0
        args.rings = [12.0, 22.0]

    os.makedirs(args.output_dir, exist_ok=True)
    print("the quadrupole line: what the binary broadcasts", flush=True)
    W = args.omega

    binary = driven_run(args, binary=True)
    single = driven_run(args, binary=False)
    freqs = binary["freqs"]
    R0 = args.spec_ring

    def lines(spec):
        return {name: line_power(freqs, spec, mult * W)[0]
                for name, mult in (("W", 1), ("2W", 2), ("3W", 3), ("4W", 4))}

    lb, ls = lines(binary[R0]), lines(single[R0])
    # dominant non-DC line: exclude bins below W/2
    live = freqs > 0.5 * W
    peak_b = float(freqs[live][np.argmax(binary[R0][live])])
    peak_s = float(freqs[live][np.argmax(single[R0][live])])
    dbin = float(freqs[1] - freqs[0])
    print(f"  binary: peak at ω={peak_b:.4f} (2Ω = {2*W:.4f}); "
          f"lines W/2W/3W/4W = " + "/".join(f"{lb[k]:.3e}" for k in lb),
          flush=True)
    print(f"  single: peak at ω={peak_s:.4f} (Ω = {W:.4f}); "
          f"lines = " + "/".join(f"{ls[k]:.3e}" for k in ls), flush=True)

    odd_frac = (lb["W"] + lb["3W"]) / (lb["2W"] + lb["4W"])
    q1 = abs(peak_b - 2.0 * W) <= 2.0 * dbin and odd_frac < 0.10
    q2 = abs(peak_s - W) <= 2.0 * dbin

    # Q3: decay of the 2W line across rings
    amps = []
    for R in args.rings:
        pw, _ = line_power(freqs, binary[R], 2.0 * W)
        amps.append(np.sqrt(pw))
    amps = np.asarray(amps)
    rr = np.asarray(args.rings, dtype=float)
    slope = np.polyfit(rr, np.log(amps * np.sqrt(rr)), 1)[0]
    lam_fit = -1.0 / slope if slope < 0 else float("inf")
    m2 = args.r
    k_complex = np.sqrt((args.tau * (2 * W) ** 2 - m2 + 1j * 2 * W) / 1.0)
    lam_theory = 1.0 / float(np.imag(k_complex))
    q3 = np.isfinite(lam_fit) and 0.5 <= lam_fit / lam_theory <= 2.0
    print(f"  2Ω decay: λ_fit = {lam_fit:.2f} vs 1/Im(k) = {lam_theory:.2f}",
          flush=True)

    lines_out = ["The quadrupole line — verdict", "=" * 74, ""]
    lines_out.append(
        f"Q1 (even harmonics only): binary peak at ω = {peak_b:.4f} vs "
        f"2Ω = {2*W:.4f}; odd/even power = {odd_frac:.2e} — "
        + ("✓ the quadrupole selection rule: rotating the pair by π IS a "
           "half-period, so odd lines are forbidden — the analogue of GR's "
           "no-dipole statement, exact and measured." if q1 else
           "✗ odd harmonics survive; the selection rule fails."))
    lines_out.append(
        f"Q2 (the dipole control): single-mass peak at ω = {peak_s:.4f} vs "
        f"Ω = {W:.4f} — "
        + ("✓ remove the symmetry and the dipole line returns: the 2Ω "
           "selection is the binary's, not the instrument's." if q2 else
           "✗ the control does not radiate at Ω."))
    lines_out.append(
        f"Q3 (the medium's complex wavenumber): 2Ω decay length "
        f"λ = {lam_fit:.2f} vs derived 1/Im√((τω²−m²+iω)/D) = "
        f"{lam_theory:.2f} (ratio {lam_fit/lam_theory:.2f}) — "
        + ("✓ the resistive κ-vacuum absorbs the line exactly as the "
           "complex dispersion says." if q3 else
           "✗ the absorption length departs from the dispersion."))
    lines_out.append("")
    lines_out.append(f"score: {int(q1)+int(q2)+int(q3)}/3 pre-registered "
                     "predictions land.")
    lines_out.append("")
    lines_out.append(
        "honest scope: kinematically driven sources (rigid rotation) so the "
        "lines are sharp — the free inspiraling binary chirps and needs a "
        "spectrogram, the registered follow-up; τ = 1 is an overdamped "
        "medium (ωτ < 1), so these are driven damped waves, not ringing "
        "ones — Q1/Q2 are symmetry statements immune to that, Q3 absorbs "
        "it in the complex k; probe rings are lattice-rounded (radii exact "
        "to ±0.5 cell); one Ω, one geometry; amplitudes are linear-response "
        "small.")
    text = "\n".join(lines_out)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_quadrupole_line.json"),
              "w") as fh:
        json.dump({"params": vars(args),
                   "binary_lines": lb, "single_lines": ls,
                   "peak_binary": peak_b, "peak_single": peak_s,
                   "ring_amps": dict(zip(map(str, args.rings),
                                         amps.tolist())),
                   "analysis": {"q1": bool(q1), "q2": bool(q2),
                                "q3": bool(q3), "odd_frac": odd_frac,
                                "lam_fit": lam_fit,
                                "lam_theory": lam_theory}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
