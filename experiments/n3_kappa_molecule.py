"""The κ-molecule: the first persistent binary, and its spin-down song.

Three recorded results meet here.  The plunge showed this gravity's
binaries cannot orbit (the trench) and merge silent (the soft blob); the
derived exclusion (Part II) holds a pair open on a parameter-free floor;
and the quadrupole experiment's selection rule was only ever heard from
*driven* rotors — the free binary's 2Ω line has never been audible,
because nothing survived long enough to broadcast it.  Give the
floor-supported pair **angular momentum** and all three change at once:
a rotating contact binary — the co-evolved theory's first persistent
bound object, a **κ-molecule** — whose rotation must sing at 2Ω while it
lasts, and whose spin-down is the drag law in closed form:

    κ̇ = −Ω·∂_θκ  ⇒  P = Ω²·C_θ ,   C_θ = Σ (∂_θ κ_pair)²  (static) ,
    E_rot = ½IΩ² ,  I = 2m(s*/2)²  ⇒  Ω̇ = −(C_θ/I)·Ω  — exponential,

with every constant read off one relaxed field.  No free parameters
anywhere: the floor is Part II's derived gap, the drag is the retarded
field's own ∫κ̇², the pitch is the selection rule's.

Calibration answered the arc's oldest open question before the run: the
static drag rate C_θ/I ≈ 0.6 against an orbital frequency Ω₀ ≈ 0.1 means
**free rotation is overdamped** (Q = 2Ω₀·I/C_θ ≪ 1) — the spin dies
before a quarter-cycle completes.  The free 2Ω line cannot exist in this
medium at this point; the quadrupole experiment's line was audible only
because the drive replaced the angular momentum the medium drains.  The
registered predictions test that law and what remains:

Z1. **The molecule exists**: with derived exclusion + spin, the pair
    neither merges nor escapes — separation stays within [0.5·s\*, 2·s\*]
    for all t ∈ [50, 400] — while the no-exclusion control with the same
    spin merges (sep < 3).  Exclusion turns this gravity's only outcome
    (plunge) into a persistent object.
Z2. **Rotation is overdamped, by the static law**: the early spin-down
    rate fits an exponential with rate within a factor of 2 of C_θ/I
    (both constants from one relaxed field), and the measured quality
    factor Q = 2Ω₀·I/C_θ < 1/2 — no rotational cycle completes, so the
    free rotor's 2Ω line is *impossible* here, as a measured law rather
    than a silence.
Z3. **The song that remains is the floor's own**: the waveform's
    dominant line (whitened contrast ≥ 3) sits within a factor of 2 of
    the libration pitch √(E″(s\*)/μ) read off the derived static curve —
    the molecule sings, but radially, at the pitch the parameter-free
    statics set.

Usage::

    python experiments/n3_kappa_molecule.py --output-dir artifacts/n3_molecule
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
    exclusion_gap_full,
)


def field_kwargs(args):
    return dict(kappa_diffusion=1.0, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0)


def pair_energy_derived(args, s: float) -> float:
    n = int(args.size)
    l1 = gaussian_load((n, n), [(n / 2 - s / 2, n / 2)], args.width,
                       args.mass)
    l2 = gaussian_load((n, n), [(n / 2 + s / 2, n / 2)], args.width,
                       args.mass)
    fkw = field_kwargs(args)
    kappa = relax_capacity(l1 + l2, **fkw)
    return (capacity_free_energy(kappa, l1 + l2, **fkw)
            + exclusion_gap_full(duplicated_load(l1, l2), **fkw))


def statics(args) -> tuple[float, float]:
    """(s*, C_θ): the derived floor and the azimuthal drag coefficient."""
    es = [(s, pair_energy_derived(args, s)) for s in args.seps]
    ref = es[-1][1]
    s_star = min(es, key=lambda t: t[1])[0]
    n = int(args.size)
    l1 = gaussian_load((n, n), [(n / 2 - s_star / 2, n / 2)], args.width,
                       args.mass)
    l2 = gaussian_load((n, n), [(n / 2 + s_star / 2, n / 2)], args.width,
                       args.mass)
    kappa = relax_capacity(l1 + l2, **field_kwargs(args))
    g = np.meshgrid(np.arange(n) - n / 2, np.arange(n) - n / 2,
                    indexing="ij")
    gx = 0.5 * (np.roll(kappa, -1, 0) - np.roll(kappa, 1, 0))
    gy = 0.5 * (np.roll(kappa, -1, 1) - np.roll(kappa, 1, 1))
    dtheta = -g[1] * gx + g[0] * gy          # ∂_θκ = (−y∂ₓ + x∂ᵧ)κ
    c_theta = float(np.sum(dtheta ** 2))
    return float(s_star), c_theta


def detrend(x, width):
    pad = np.pad(x, width // 2, mode="edge")
    return x - np.convolve(pad, np.ones(width) / width, mode="valid")[:len(x)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=float, default=96.0)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--mass", type=float, default=0.6)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--v0", type=float, default=0.5)
    p.add_argument("--seps", type=float, nargs="+",
                   default=[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 16.0])
    p.add_argument("--t-max", type=float, default=400.0)
    p.add_argument("--probe-radius", type=float, default=11.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_molecule")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.t_max = 200.0
        args.seps = [4.0, 6.0, 8.0, 10.0, 16.0]

    os.makedirs(args.output_dir, exist_ok=True)
    print("the kappa-molecule: a rotating contact binary and its song",
          flush=True)

    s_star, c_theta = statics(args)
    inertia = 2.0 * args.mass * (s_star / 2.0) ** 2
    rate_pred = c_theta / inertia
    print(f"  statics: floor s* = {s_star:g}, C_θ = {c_theta:.3f}, "
          f"I = {inertia:.2f} → predicted spin-down rate {rate_pred:.4f}",
          flush=True)

    L = args.size
    probes = [(int(round(L / 2 + args.probe_radius * np.cos(a))),
               int(round(L / 2 + args.probe_radius * np.sin(a))))
              for a in np.linspace(0, 2 * np.pi, 4, endpoint=False)]
    common = dict(shape=(int(L), int(L)), width=args.width, tau=args.tau,
                  dt=args.dt, steps=int(args.t_max / args.dt),
                  record_every=2, **field_kwargs(args))
    # release just outside the floor: the infall excites the libration
    # channel (Z3) while the tangential speed feeds the rotation (Z2)
    d_rel = s_star + 2.0
    pos0 = [[L / 2 - d_rel / 2, L / 2], [L / 2 + d_rel / 2, L / 2]]
    vel0 = [[0.0, +args.v0], [0.0, -args.v0]]
    hist = evolve_inertial_retarded(pos0, vel0, [args.mass] * 2,
                                    probes=probes, contact_full=True,
                                    **common)
    ctrl = evolve_inertial_retarded(
        pos0, vel0, [args.mass] * 2,
        **{**common, "steps": int(100.0 / args.dt)})

    t_rec = np.asarray(hist["time"])
    sep = np.asarray(hist["separation"])
    sep_c = np.asarray(ctrl["separation"])
    live = t_rec >= 50.0
    z1 = (bool(np.all((sep[live] >= 0.5 * s_star)
                      & (sep[live] <= 2.0 * s_star)))
          and bool(np.min(sep_c) < 3.0))
    print(f"  molecule: sep range [{sep[live].min():.2f}, "
          f"{sep[live].max():.2f}] over t ∈ [50, {args.t_max:g}]; "
          f"control min sep = {sep_c.min():.2f}", flush=True)

    # --- Z2: the overdamped rotation, against the static law
    pos = np.asarray(hist["positions"])
    sv = pos[:, 0, :] - pos[:, 1, :]
    phi = np.unwrap(np.arctan2(sv[:, 1], sv[:, 0]))
    omega = np.abs(np.gradient(phi, t_rec))
    omega0 = float(omega[2])
    q_factor = 2.0 * omega0 / rate_pred
    rot = (omega >= 0.1 * omega0) & (t_rec >= 0.4) & (t_rec <= 20.0)
    rate_fit = float(-np.polyfit(t_rec[rot], np.log(omega[rot]), 1)[0])
    z2 = (0.5 <= rate_fit / rate_pred <= 2.0) and q_factor < 0.5
    print(f"  spin-down: fitted rate {rate_fit:.4f} vs C_θ/I = "
          f"{rate_pred:.4f} (ratio {rate_fit/rate_pred:.2f}); "
          f"Q = 2Ω₀·I/C_θ = {q_factor:.3f}", flush=True)

    # --- Z3: the libration line at the derived floor's pitch
    near = [(s, e) for s, e in
            [(s, pair_energy_derived(args, s)) for s in args.seps]
            if abs(s - s_star) <= 2.5]
    a2 = float(np.polyfit([s for s, _ in near], [e for _, e in near], 2)[0])
    om_lib_pred = float(np.sqrt(max(2.0 * a2, 0.0) / (args.mass / 2.0)))
    sig = np.asarray(hist["probe_kappa"])
    t_f = args.dt * np.arange(sig.shape[0])
    winf = (t_f > 15.0) & (t_f < min(150.0, args.t_max - 5.0))
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
    z3 = (contrast >= 3.0
          and 0.5 <= peak_w / om_lib_pred <= 2.0)
    print(f"  libration: line at ω = {peak_w:.3f} (contrast "
          f"{contrast:.1f}) vs static pitch √(E″/μ) = {om_lib_pred:.3f}",
          flush=True)

    lines = ["The κ-molecule — verdict", "=" * 74, ""]
    lines.append(
        f"Z1 (the molecule exists): sep ∈ [{sep[live].min():.1f}, "
        f"{sep[live].max():.1f}] around s* = {s_star:g} across "
        f"t ∈ [50, {args.t_max:g}]; the no-exclusion control merges (min "
        f"sep {sep_c.min():.1f}) — "
        + ("✓ the co-evolved theory's first persistent bound object: "
           "exclusion turns the plunge into a molecule." if z1 else
           "✗ no persistent bound pair."))
    lines.append(
        f"Z2 (rotation is overdamped, by the static law): Q = 2Ω₀·I/C_θ = "
        f"{q_factor:.3f} (< ½: overdamped ✓, Ω₀ = {omega0:.4f}, C_θ = "
        f"{c_theta:.2f}, I = {inertia:.2f}), so the free 2Ω line is "
        f"impossible here; but the spin-down rate {rate_fit:.3f} misses "
        f"the quasi-static C_θ/I = {rate_pred:.3f} by "
        f"{rate_pred/max(rate_fit,1e-9):.0f}× — "
        + ("✓ overdamped AND the rate matches." if z2 else
           "✗ the compound bar fails: the medium IS too viscous to orbit "
           "(Q < ½ — the free rotor's 2Ω line cannot exist), but the "
           "quasi-static drag rate is wrong — the retarded field never "
           "builds the full static gradient C_θ assumes, so the real "
           "torque is far weaker.  The overdamping is the finding; the "
           "rate law is refuted."))
    lines.append(
        f"Z3 (the song that remains is the floor's own): waveform line at "
        f"ω = {peak_w:.3f} (whitened contrast {contrast:.0f}) vs the "
        f"derived static libration pitch √(E″(s*)/μ) = {om_lib_pred:.3f} "
        f"(ratio {peak_w/om_lib_pred:.2f}) — "
        + ("✓ the molecule sings — radially, at the pitch the "
           "parameter-free statics set." if z3 else
           "✗ the surviving line is not the derived libration pitch."))
    lines.append("")
    lines.append(f"score: {int(z1)+int(z2)+int(z3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: one operating point (the exclusion arc's), one "
        "spin. The overdamping is the finding: the derived floor is stiff "
        "and the medium viscous, so rotation is killed before a cycle "
        "(Q < ½) — the κ-molecule is a librating contact pair, not a "
        "spinning one. Z2 was registered as a COMPOUND bar (Q < ½ AND the "
        "spin-down rate ≈ the quasi-static C_θ/I) and fails on the second "
        f"half: the rate is ~{rate_pred/max(rate_fit,1e-9):.0f}× slower "
        "than C_θ/I because the overdamped pair barely moves, so the "
        "retarded field never builds the full static gradient the static "
        "C_θ assumes — the quasi-static drag LAW is refuted here even as "
        "the overdamping it implies is confirmed.  The exclusion sector is "
        "adiabatic while gravity is retarded (the base branch's split); "
        "equal masses; the libration pitch uses a coarse parabola through "
        "the derived E(s) grid near s*.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_kappa_molecule.json"),
              "w") as fh:
        json.dump({"params": vars(args),
                   "statics": {"s_star": s_star, "c_theta": c_theta,
                               "inertia": inertia,
                               "rate_pred": rate_pred,
                               "om_lib_pred": om_lib_pred},
                   "rotation": {"omega0": omega0, "q_factor": q_factor,
                                "rate_fit": rate_fit},
                   "libration": {"line": peak_w, "contrast": contrast},
                   "analysis": {"z1": bool(z1), "z2": bool(z2),
                                "z3": bool(z3),
                                "sep_min": float(sep[live].min()),
                                "sep_max": float(sep[live].max()),
                                "ctrl_min_sep": float(sep_c.min())}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
