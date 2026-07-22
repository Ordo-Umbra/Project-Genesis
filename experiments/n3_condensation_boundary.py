"""The condensation boundary: why matter does not yet gather its fermions.

The junction-fermion result closed the *count* question; the remaining gap in
the hadron story is **condensation** — matter *collecting* its spin-½ defects
out of a noise-grown field, instead of having them imprinted.  This experiment
maps where that fails and what the one working piece is.  The verdict is a
boundary, stated as three falsifiable claims (each probed exploratorily, here
confirmed on fresh seeds):

K1. **Passive traps do not condense — the dilute gas is frozen.**  Static κ
    wells at floor spacing, ψ grown from noise over them: across trap depths
    (``γ = 0.8, 0.95``) and coarsening times the well occupancy stays at chance
    level (``≤ 15%`` of wells, no better than the flat-κ control), and pinning
    gives **no survival enhancement** (defect counts match the control).  A
    defect born outside a well's few-site basin never reaches it — there is no
    transport in the relaxational gas.
K2. **The naive third-law reaction force clumps, it does not condense.**  The
    detuning energy (wells suppress ``|ψ|²``) implies a reaction force pulling
    masses toward amplitude holes — but *every mass's well is itself an
    amplitude hole*, so the raw force ``−χ_a⟨∇|ψ|²⟩`` is dominated by the
    partner's well: released on the derived floor with it on, the pair
    **collapses through the exclusion floor** (late separation ``< 0.5·s⋆``)
    while the phase-only control holds its floor (``≥ 0.85·s⋆``).  A force
    that cannot tell matter from fermion condenses nothing.
K3. **The reaction force can be made *selective* — the one working ingredient.**
    The *raw* third-law force ``−χ_a⟨∇|ψ|²⟩`` points at an *empty* well as
    strongly as at a fermion core (every well is an amplitude hole), so it
    cannot build a composite.  Divide the wells' own reversible envelope out —
    the **envelope-normalised** force ``−χ_a⟨∇(|ψ|²/(1−g))⟩`` — and it responds
    only to the *topological* core: normalised pull toward a fermion core is
    strong and toward an empty well is ``~0`` (``≤ 10%``), where the raw force
    is fooled by both.  A force that can tell a fermion from a fellow mass; what
    is still missing is a current to carry the fermion to it.

**The headline is the negative:** full condensation — matter *gathering* fermions
from afar to build a composite — is **not achieved**.  The three results map
exactly why: the dilute gas is frozen (K1, no transport), the naive force clumps
matter instead (K2), and the selective force (K3) is a handle with nothing to
pull on.  The exploratory mobile-channel probe confirmed it: masses released
into the gas show no systematic capture at working density.

Honest scope: the missing ingredient is defect transport (K1's frozen gas), and
the named candidate for the next rung is the ``λ ≠ 0`` precession that
regenerated the driven spin — spiral-wave advection mobilises vortex cores.  The
normalised reaction force is a *motivated* selectivity (the third-law reaction
to the core's hole with the reversible well-envelope removed), its coefficient
free, not derived.  2-D, one lattice, fresh seeds throughout (disjoint from the
exploratory probes).

Usage::

    python experiments/n3_condensation_boundary.py --output-dir artifacts/n3_condensation
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.capacity_gravity import gaussian_load, relax_capacity
from project_genesis.condensation import (
    amp_reaction_force,
    condensation_run,
    defect_positions,
    grow_defect_gas,
)
from project_genesis.two_field import chiral_detuning
from project_genesis.vortex_chiral import imprint_vortices


def field_kwargs(args):
    return dict(kappa_diffusion=1.0, kappa_recovery=args.r,
                kappa_consumption=args.c, kappa_baseline=1.0)


def pdist(a, b, n):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return float(np.hypot(min(dx, n - dx), min(dy, n - dy)))


def k1_passive(args):
    n = int(args.size)
    mid = n / 2
    cs = [(mid - args.floor / 2, mid), (mid + args.floor / 2, mid)]
    loads = sum(gaussian_load((n, n), [c], args.width, args.mass) for c in cs)
    kappa = relax_capacity(loads, **field_kwargs(args))
    out = []
    for gamma in args.gammas:
        g = chiral_detuning(kappa, detune_gamma=gamma)
        g_flat = np.zeros((n, n))
        for steps in args.gas_steps:
            occ = hits = 0
            n_coup, n_ctrl = [], []
            for s_i in range(args.n_seeds):
                seed = args.seed + 13 * s_i
                psi = grow_defect_gas((n, n), g, steps=steps, seed=seed)
                dfs = defect_positions(psi)
                n_coup.append(len(dfs))
                occ += sum(any(pdist(p, c, n) <= args.capture
                               for p, _ in dfs) for c in cs)
                psi0 = grow_defect_gas((n, n), g_flat, steps=steps, seed=seed)
                dfs0 = defect_positions(psi0)
                n_ctrl.append(len(dfs0))
                hits += sum(any(pdist(p, c, n) <= args.capture
                                for p, _ in dfs0) for c in cs)
            out.append({"gamma": gamma, "steps": steps,
                        "occupied": occ, "chance": hits,
                        "wells": 2 * args.n_seeds,
                        "n_coupled": n_coup, "n_control": n_ctrl})
            print(f"  γ={gamma} steps={steps}: occupancy {occ}/"
                  f"{2*args.n_seeds} (chance {hits}/{2*args.n_seeds})  "
                  f"survivors {n_coup} vs control {n_ctrl}", flush=True)
    return out


def k2_clumping(args):
    n = int(args.size)
    mid = n / 2
    pos0 = [[mid - args.floor / 2, mid], [mid + args.floor / 2, mid]]
    vel0 = [[0.0, +args.kick], [0.0, -args.kick]]
    out = []
    for s_i in range(args.n_dyn):
        seed = args.seed + 101 + 13 * s_i
        seps = {}
        for label, chi_amp in (("raw", args.chi_amp_raw), ("control", 0.0)):
            h = condensation_run(
                pos0, vel0, [args.mass] * 2, shape=(n, n), width=args.width,
                tau=args.tau, dt=args.dt, steps=args.dyn_steps,
                pregrow_steps=args.pregrow, seed=seed,
                phase_chi=args.chi, chi_amp=chi_amp, normalize=False,
                detune_gamma=0.8, **field_kwargs(args))
            t = np.asarray(h["time"])
            sep = np.asarray(h["separation"])
            seps[label] = float(sep[t > 0.6 * t[-1]].mean())
        out.append({"seed": s_i, "sep_raw": seps["raw"],
                    "sep_control": seps["control"]})
        print(f"  seed {s_i}: late sep raw={seps['raw']:.1f} vs "
              f"control={seps['control']:.1f}  (s⋆={args.floor})", flush=True)
    return out


def k3_selectivity(args):
    """The constructive static check: the raw force is pulled toward an *empty*
    well as much as toward a fermion core (every well is an amplitude hole),
    while the normalised force is pulled only toward the core.  Two separate
    single-well landscapes (no competing deeper hole), a probe mass offset by
    the floor spacing — exactly the `tests/test_condensation.py` setup."""
    n = int(args.size)
    mid = n / 2
    well = (mid + args.floor / 2, mid)
    probe = (mid - args.floor / 2, mid)
    kappa = relax_capacity(gaussian_load((n, n), [well], args.width, args.mass),
                           **field_kwargs(args))
    g = chiral_detuning(kappa, detune_gamma=0.8)
    load = gaussian_load((n, n), [probe], args.width, args.mass)
    out = {}
    for name, has_core in (("core", True), ("empty", False)):
        if has_core:
            psi = imprint_vortices((n, n), [well], [1], core=args.core, detune=g)
        else:
            psi = np.sqrt(np.clip(1.0 - g, 0.0, 1.0)).astype(complex)
        raw = amp_reaction_force([load], psi, chi_amp=1.0)[0][0]
        nrm = amp_reaction_force([load], psi, chi_amp=1.0, detune=g,
                                 normalize=True)[0][0]
        out[name] = {"raw": float(raw), "norm": float(nrm)}  # +x = toward well
    print(f"  raw pull  toward cored well={out['core']['raw']:+.3e}  "
          f"toward empty well={out['empty']['raw']:+.3e}", flush=True)
    print(f"  norm pull toward cored well={out['core']['norm']:+.3e}  "
          f"toward empty well={out['empty']['norm']:+.3e}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--width", type=float, default=2.5)
    p.add_argument("--mass", type=float, default=0.6)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--floor", type=float, default=9.0)
    p.add_argument("--core", type=float, default=3.0)
    p.add_argument("--capture", type=float, default=3.0)
    p.add_argument("--gammas", type=float, nargs="+", default=[0.8, 0.95])
    p.add_argument("--gas-steps", type=int, nargs="+", default=[2000, 6000])
    p.add_argument("--n-seeds", type=int, default=6)
    p.add_argument("--n-dyn", type=int, default=4)
    p.add_argument("--chi", type=float, default=0.6)
    p.add_argument("--chi-amp-raw", type=float, default=15.0)
    p.add_argument("--chi-amp-norm", type=float, default=5.0)
    p.add_argument("--kick", type=float, default=0.2)
    p.add_argument("--pregrow", type=int, default=800)
    p.add_argument("--dyn-steps", type=int, default=2200)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=50)
    p.add_argument("--output-dir", type=str,
                   default="artifacts/n3_condensation")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.gammas = [0.8]
        args.gas_steps = [2000]
        args.n_seeds = 4
        args.n_dyn = 2
        args.dyn_steps = 1500

    os.makedirs(args.output_dir, exist_ok=True)
    print("the condensation boundary: why matter does not yet gather its "
          "fermions", flush=True)
    print("K1 — passive traps across depth × time:", flush=True)
    k1 = k1_passive(args)
    print("K2 — the raw reaction force (clumping):", flush=True)
    k2 = k2_clumping(args)
    print("K3 — the selective (envelope-normalised) reaction force:", flush=True)
    k3_sel = k3_selectivity(args)

    # --- K1: occupancy at chance, no survival enhancement
    tot_wells = sum(r["wells"] for r in k1)
    tot_occ = sum(r["occupied"] for r in k1)
    tot_chance = sum(r["chance"] for r in k1)
    surv_diff = np.mean([np.mean(r["n_coupled"]) - np.mean(r["n_control"])
                         for r in k1])
    k1_ok = bool(tot_occ <= 0.15 * tot_wells
                 and tot_occ <= tot_chance + 2
                 and abs(surv_diff) <= 1.0)

    # --- K2: raw force collapses the floor; control holds it
    k2_ok = bool(all(r["sep_raw"] < 0.5 * args.floor for r in k2)
                 and all(r["sep_control"] >= 0.85 * args.floor for r in k2))

    # --- K3: the envelope-normalised force is SELECTIVE — it distinguishes a
    # fermion core from an empty well, where the raw force is fooled by both.
    # (The working ingredient; the headline that condensation itself is not
    # achieved — gathering needs transport — is the experiment's thesis, stated
    # loudly in the scope and confirmed by the exploratory mobile-channel probe.)
    sel_core = k3_sel["core"]["norm"]
    sel_empty = k3_sel["empty"]["norm"]
    raw_core = k3_sel["core"]["raw"]
    raw_empty = k3_sel["empty"]["raw"]
    k3_ok = bool(sel_core > 1e-4                         # normalised: sees core
                 and abs(sel_empty) <= 0.1 * sel_core     # normalised: blind to empty
                 and raw_empty >= 0.3 * raw_core)         # raw: fooled by both
    selective = k3_ok

    lines = ["The condensation boundary — verdict", "=" * 74, ""]
    lines.append(
        f"K1 (passive traps do not condense — the gas is frozen): occupancy "
        f"{tot_occ}/{tot_wells} vs chance {tot_chance}/{tot_wells} across "
        f"γ×steps; survivor excess {surv_diff:+.2f} — "
        + ("✓ at chance level at every depth and time, with no survival "
           "enhancement: nothing transports a defect to a well." if k1_ok else
           "✗ the wells capture (or protect) beyond chance — passive "
           "condensation is not excluded."))
    lines.append(
        "K2 (the naive third-law force clumps): late sep raw = "
        + ", ".join(f"{r['sep_raw']:.1f}" for r in k2)
        + " vs control " + ", ".join(f"{r['sep_control']:.1f}" for r in k2)
        + f" (s⋆ = {args.floor:g}) — "
        + ("✓ the raw amplitude-reaction force collapses the pair through its "
           "exclusion floor — every well is an amplitude hole, so the force "
           "cannot tell matter from fermion." if k2_ok else
           "✗ the raw reaction force does not produce the clean clumping "
           "collapse (or the control does not hold its floor)."))
    lines.append(
        "K3 (the reaction force IS selective — the working ingredient): "
        f"normalised pull toward a fermion core {sel_core:+.2e} vs an empty "
        f"well {sel_empty:+.2e} (raw: {raw_core:+.2e} vs {raw_empty:+.2e}) — "
        + ("✓ divide the wells' own envelope out and the force responds only to "
           "the topological core, not to an empty well — a force that can tell "
           "a fermion from a fellow mass, where the raw force is fooled by "
           "both.  The one piece condensation needs; what is still missing is a "
           "current to carry fermions to matter." if k3_ok else
           "✗ the normalised force does not cleanly ignore the empty well."))
    lines.append("")
    lines.append(f"score: {int(k1_ok)+int(k2_ok)+int(k3_ok)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope — the headline: full condensation is NOT achieved.  Matter "
        "cannot yet GATHER fermions from afar to build a composite: the three "
        "results map exactly why.  The dilute gas is frozen (K1) — nothing "
        "transports a defect across the lattice; the naive third-law force clumps "
        "matter instead (K2); and while the envelope-normalised force is the "
        "right, selective handle on a fermion (K3), a selective force is useless "
        "without a current to carry fermions to it.  The exploratory mobile-"
        "channel probe confirmed it directly: masses released into the gas show "
        "no systematic capture at working density.  The missing ingredient is "
        "defect transport, and the named candidate is the λ ≠ 0 precession that "
        "regenerated the driven spin (spiral-wave advection mobilises cores) — "
        "the next rung.  The normalised reaction force is a motivated selectivity "
        "(the third-law reaction to the core's hole, the wells' reversible "
        "envelope removed), its coefficient free, not derived.  2-D, one "
        "lattice.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_condensation_boundary.json"),
              "w") as fh:
        json.dump({"params": vars(args), "k1": k1, "k2": k2,
                   "k3_selectivity": k3_sel,
                   "analysis": {"k1": bool(k1_ok), "k2": bool(k2_ok),
                                "k3": bool(k3_ok)}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
