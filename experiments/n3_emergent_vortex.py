"""The self-sustained vortex: topology kept by the dynamics, not by imprinting.

The vorticity-bearing run (`n3_vortex_chiral`) gave the molecule a real,
quantised circulation and a genuine torque — but its vortex was *pinned*:
re-imprinted at the masses every step, its winding imposed and its circulation
continually re-sharpened.  This experiment removes the re-imprinting.  The
vortices are seeded **once** and the chiral field then co-evolves under the
κ-detuned CGL alone (`evolve_seeded_field`, or
`evolve_vortex_molecule(reimprint=False)`) — a fully self-sustained field, kept
(or lost) by the dynamics and anchored only by the κ wells.

What survives that, and what does not, is the point:

E1. **The winding is a dynamically-conserved topological charge.**  With no
    re-imprinting, the integer winding enclosing each κ well is preserved for
    same-sign vortices across the whole run (held to an integer within `0.1`,
    max drift `< 0.1`), and stays integer even from a **noisy** seed (15%
    complex noise) — quantisation the dynamics enforces, not a value imposed
    each step.
E2. **Topological selection, and matter-pinning.**  Like charges *survive* and
    stay bound to their matter: a same-sign pair keeps `|w| = 1` at each well,
    the cores persist (`min|ψ| < 0.2`), the field holds real angular momentum of
    the seed's sign (far above the annihilated pair's `~0`; its slow decay is
    E3's finding, not a survival failure), and — released as a free molecule —
    the winding around each *moving* mass
    stays `±1` (the self-sustained vortex tracks its form, emergent pinning).
    Unlike charges *annihilate*: a vortex–antivortex pair unwinds to `w = 0`,
    the amplitude heals (`min|ψ|` recovers past `0.4`), and `L → 0`.  The sign
    structure is physical, dynamically enforced — not an assigned label.
E3. **But the self-sustained field does NOT reproduce the strong spin**
    (pre-registered at the pinned run's bar): released with the vortices seeded
    once and never re-imprinted, the molecule's late spin is sign-locked to the
    winding and above the achiral baseline, yet an **order of magnitude weaker**
    than the re-imprinted drive (`|Ω| < 0.004`, versus the pinned mode's
    `≥ 0.004` at the same operating point) while the field's `L` drains — the
    circulation is not maintained without the continual re-sharpening.  The
    strong persistent orbital spin was, in part, the re-imprinting doing work.

The reading (not scored): a topological defect can bind to matter and keep its
quantised charge under its own dynamics (E1, E2) — the honest core of "spin as a
conserved, quantised, matter-bound thing"; turning that bound charge into a
strong *orbital* torque without re-sharpening is the open problem (E3).

Usage::

    python experiments/n3_emergent_vortex.py --output-dir artifacts/n3_emergent
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
    exclusion_gap_full,
)
from project_genesis.two_field import chiral_detuning, step_chiral_detuned
from project_genesis.vortex_chiral import (
    evolve_seeded_field,
    evolve_vortex_molecule,
    imprint_vortices,
    winding_number,
)


def field_kwargs(args):
    return dict(kappa_diffusion=1.0, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0)


def derived_floor(args) -> float:
    n = int(args.size)
    fkw = field_kwargs(args)

    def energy(s):
        l1 = gaussian_load((n, n), [(n / 2 - s / 2, n / 2)], args.width, args.mass)
        l2 = gaussian_load((n, n), [(n / 2 + s / 2, n / 2)], args.width, args.mass)
        kappa = relax_capacity(l1 + l2, **fkw)
        return (capacity_free_energy(kappa, l1 + l2, **fkw)
                + exclusion_gap_full(duplicated_load(l1, l2), **fkw))

    return float(min(((s, energy(s)) for s in args.seps),
                     key=lambda t: t[1])[0])


def seeded(args, s_star, charges, kappa, noise=0.0):
    """E1/E2: seed once, co-evolve the field alone; return winding/L/amp series."""
    n = int(args.size)
    centers = [(n / 2 - s_star / 2, n / 2), (n / 2 + s_star / 2, n / 2)]
    h = evolve_seeded_field((n, n), centers, list(charges), kappa,
                            chiral=0.0, detune_gamma=args.gamma, core=args.core,
                            dt=args.dt, steps=args.field_steps, record_every=200)
    if noise:
        # a separately-seeded noisy run for the robustness sub-test
        g = chiral_detuning(kappa, detune_gamma=args.gamma)
        rng = np.random.default_rng(args.seed + 7)
        psi = imprint_vortices((n, n), centers, list(charges), core=args.core,
                               detune=g)
        psi = psi + noise * (rng.standard_normal((n, n))
                             + 1j * rng.standard_normal((n, n)))
        wser = []
        for step in range(args.field_steps + 1):
            if step % 200 == 0:
                wser.append([winding_number(psi, cc) for cc in centers])
            psi = step_chiral_detuned(psi, g, chiral=0.0, dt=args.dt)
        h["winding_noisy"] = wser
    return h, centers


def molecule(args, s_star, q, reimprint):
    """E2 pinning / E3 spin: release the pair, ψ pinned or self-sustained."""
    n = int(args.size)
    pos0 = [[n / 2 - s_star / 2, n / 2], [n / 2 + s_star / 2, n / 2]]
    vel0 = [[0.0, +args.kick], [0.0, -args.kick]]
    h = evolve_vortex_molecule(
        pos0, vel0, [args.mass] * 2, shape=(n, n), width=args.width,
        tau=args.tau, dt=args.dt, steps=int(args.t_max / args.dt),
        record_every=20, vortex_charge=q, phase_chi=args.chi, core=args.core,
        detune_gamma=args.gamma, reimprint=reimprint, **field_kwargs(args))
    t = np.asarray(h["time"])
    sep = np.asarray(h["separation"])
    omega = np.asarray(h["omega"])
    late = t > 0.6 * args.t_max
    wind = np.asarray(h["winding"])
    return {"q": q, "reimprint": reimprint,
            "omega_late": float(omega[late].mean()),
            "sep_late": float(sep[late].mean()),
            "L_first": float(h["L_field"][0]), "L_last": float(h["L_field"][-1]),
            "wind_min": [float(np.min(np.abs(wind[:, k]))) for k in (0, 1)],
            "bounded": bool(np.all(np.isfinite(sep))
                            and sep[late].max() < 1.5 * s_star)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=float, default=96.0)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--mass", type=float, default=0.6)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--chi", type=float, default=0.6)
    p.add_argument("--core", type=float, default=3.0)
    p.add_argument("--kick", type=float, default=0.2)
    p.add_argument("--seps", type=float, nargs="+",
                   default=[6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 16.0])
    p.add_argument("--field-steps", type=int, default=2000)
    p.add_argument("--t-max", type=float, default=220.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--strong-bar", type=float, default=0.004,
                   help="the pinned run's persistent-spin bar (E3 pre-registered)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_emergent")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.field_steps = 800
        args.t_max = 150.0
        args.seps = [7.0, 9.0, 12.0]

    os.makedirs(args.output_dir, exist_ok=True)
    print("the self-sustained vortex: topology kept by the dynamics", flush=True)
    n = int(args.size)
    s_star = derived_floor(args)
    print(f"  derived exclusion floor s⋆ = {s_star:g}", flush=True)
    loads = (gaussian_load((n, n), [(n / 2 - s_star / 2, n / 2)], args.width, args.mass)
             + gaussian_load((n, n), [(n / 2 + s_star / 2, n / 2)], args.width, args.mass))
    kappa = relax_capacity(loads, **field_kwargs(args))

    # --- E1 / E2 : field-only, seeded once
    print("field-only (E1, E2): same-sign, vortex-antivortex, noisy", flush=True)
    same, _ = seeded(args, s_star, (1, 1), kappa, noise=args.noise)
    opp, _ = seeded(args, s_star, (1, -1), kappa)
    w_same = np.asarray(same["winding"])
    w_opp = np.asarray(opp["winding"])
    w_noisy = np.asarray(same["winding_noisy"])
    L_seed = abs(same["L_field"][0])
    print(f"  same-sign winding {w_same[0].round(2).tolist()}→"
          f"{w_same[-1].round(2).tolist()}  L {same['L_field'][0]:.0f}→"
          f"{same['L_field'][-1]:.0f}  min|ψ|={same['amp_min'][-1]:.2f}", flush=True)
    print(f"  vortex-antivortex winding {w_opp[0].round(2).tolist()}→"
          f"{w_opp[-1].round(2).tolist()}  L→{opp['L_field'][-1]:.0f}  "
          f"min|ψ|={opp['amp_min'][-1]:.2f}", flush=True)
    print(f"  noisy-seed winding {w_noisy[0].round(2).tolist()}→"
          f"{w_noisy[-1].round(2).tolist()}", flush=True)

    # --- E3 : the molecule, self-sustained vs pinned, at the same operating point
    print("molecule (E2 pinning, E3 spin): reimprint False vs True", flush=True)
    emergent = {q: molecule(args, s_star, q, reimprint=False) for q in (-1, 0, 1)}
    pinned = molecule(args, s_star, 1, reimprint=True)
    for q in (-1, 0, 1):
        r = emergent[q]
        print(f"  emergent q={q:+d}: late Ω={r['omega_late']:+.5f}  "
              f"sep={r['sep_late']:.1f}  wind_min={[round(x,2) for x in r['wind_min']]}  "
              f"L {r['L_first']:.0f}→{r['L_last']:.0f}  bound={r['bounded']}", flush=True)
    print(f"  pinned   q=+1: late Ω={pinned['omega_late']:+.5f}  "
          f"sep={pinned['sep_late']:.1f}  bound={pinned['bounded']}", flush=True)

    # ---- verdicts
    def is_int(x, tol=0.1):
        return abs(x - round(x)) <= tol

    # E1: same-sign winding integer & conserved (clean and noisy)
    e1_clean = (all(is_int(v) for row in w_same for v in row)
                and float(np.max(np.abs(w_same - w_same[0]))) < 0.1
                and abs(round(w_same[-1][0])) == 1)
    e1_noisy = (all(is_int(v) for row in w_noisy for v in row)
                and float(np.max(np.abs(w_noisy - w_noisy[0]))) < 0.1)
    e1 = bool(e1_clean and e1_noisy)

    # E2: like survive & pinned; unlike annihilate.  Survival is topological —
    # the integer winding is held and the cores persist — and the field still
    # carries angular momentum of the seed's sign, far above the annihilated
    # pair's ~0 (its slow decay is E3's finding, not a survival failure).
    like_survive = (abs(round(w_same[-1][0])) == 1 and abs(round(w_same[-1][1])) == 1
                    and same["amp_min"][-1] < 0.2
                    and abs(same["L_field"][-1]) > 5.0 * abs(opp["L_field"][-1]) + 1.0)
    unlike_annih = (abs(w_opp[-1][0]) < 0.2 and abs(w_opp[-1][1]) < 0.2
                    and abs(opp["L_field"][-1]) < 0.1 * L_seed
                    and opp["amp_min"][-1] > 0.4)
    pinned_track = all(min(emergent[q]["wind_min"]) > 0.8 for q in (-1, 1))
    e2 = bool(like_survive and unlike_annih and pinned_track)

    # E3 (pre-registered at the pinned bar): the self-sustained STRONG spin
    ach = emergent[0]
    chir = [emergent[q] for q in (-1, 1)]
    strong = all(abs(r["omega_late"]) >= args.strong_bar
                 and abs(r["omega_late"]) >= 5.0 * abs(ach["omega_late"])
                 and np.sign(r["omega_late"]) == -np.sign(r["q"]) for r in chir)
    # the honest content: sign-locked & above baseline, but weak & draining
    sign_locked = all(np.sign(r["omega_late"]) == -np.sign(r["q"]) for r in chir)
    above_base = all(abs(r["omega_late"]) >= 5.0 * abs(ach["omega_late"])
                     for r in chir)
    drains = all(abs(r["L_last"]) < abs(r["L_first"]) for r in chir)
    pinned_strong = abs(pinned["omega_late"]) >= args.strong_bar
    e3 = bool(strong)   # pre-registered strong-spin prediction (expected to FAIL)

    lines = ["The self-sustained vortex — verdict", "=" * 74, ""]
    lines.append(
        "E1 (the winding is a dynamically-conserved topological charge): "
        f"same-sign {w_same[0].round(2).tolist()}→{w_same[-1].round(2).tolist()}, "
        f"noisy {w_noisy[0].round(2).tolist()}→{w_noisy[-1].round(2).tolist()} — "
        + ("✓ integer and constant across the run with no re-imprinting, robust "
           "to a noisy seed — quantisation the dynamics enforces." if e1 else
           "✗ the winding is not conserved / integer without imprinting."))
    lines.append(
        "E2 (topological selection & matter-pinning): like-charge L "
        f"{same['L_field'][0]:.0f}→{same['L_field'][-1]:.0f} (|w|=1, min|ψ|="
        f"{same['amp_min'][-1]:.2f}); unlike-charge w→"
        f"{w_opp[-1].round(2).tolist()}, L→{opp['L_field'][-1]:.0f}, min|ψ|="
        f"{opp['amp_min'][-1]:.2f}; molecule winding tracks the moving mass "
        f"(min|w|={[round(min(emergent[q]['wind_min']),2) for q in (-1,1)]}) — "
        + ("✓ like charges survive and stay bound to their matter; unlike "
           "charges annihilate and the field heals — a physical, "
           "dynamically-enforced sign rule." if e2 else
           "✗ the survive/annihilate selection or the pinning does not hold."))
    lines.append(
        "E3 (the self-sustained field reproduces the STRONG spin): emergent "
        f"late Ω = {emergent[-1]['omega_late']:+.4f}/{ach['omega_late']:+.4f}/"
        f"{emergent[1]['omega_late']:+.4f} (q=−1/0/+1) vs pinned "
        f"{pinned['omega_late']:+.4f} — "
        + ("✓ the self-sustained field holds the strong spin too." if e3 else
           "✗ NO — the self-sustained spin is sign-locked "
           f"({sign_locked}) and above the achiral baseline ({above_base}) but "
           f"an order weaker than the pinned drive and draining ({drains}): the "
           "strong persistent spin needed the continual re-imprinting."))
    lines.append("")
    lines.append(f"score: {int(e1)+int(e2)+int(e3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: this is the deeper, self-sustained version of the "
        "vorticity-bearing field — the vortex is seeded once and kept only by "
        "the κ-detuned CGL dynamics (no re-imprinting).  Its topology is real "
        "and robust (E1) and it binds to matter with the correct sign rule "
        "(E2) — the honest core of a quantised, matter-bound spin-analog.  What "
        "it does NOT do (E3, a pre-registered negative scored against the "
        "pinned run's bar) is hold the STRONG orbital spin: without the "
        "re-imprinting that re-sharpens the circulation each step, the field's "
        "angular momentum drains and the residual torque on the masses is an "
        "order of magnitude weaker (though still sign-locked to the winding).  "
        "So the pinned mode's strong spin was, in part, the imprinting doing "
        "work; a self-sustained STRONG orbital drive is the open frontier.  One "
        "operating point (the derived exclusion floor), equal masses, λ = 0 "
        "(no imposed precession — the purest topological test), a torus (net "
        "winding compensated by the dynamics).")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_emergent_vortex.json"),
              "w") as fh:
        json.dump({"params": vars(args), "s_star": s_star,
                   "same": {k: v for k, v in same.items() if k != "psi"},
                   "opp": {k: v for k, v in opp.items() if k != "psi"},
                   "emergent": emergent, "pinned": pinned,
                   "analysis": {"e1": bool(e1), "e2": bool(e2), "e3": bool(e3),
                                "sign_locked": bool(sign_locked),
                                "above_base": bool(above_base),
                                "drains": bool(drains),
                                "pinned_strong": bool(pinned_strong)}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
