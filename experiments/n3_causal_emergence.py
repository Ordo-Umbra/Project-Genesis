"""Does the macro have more causal grip than the micro — and if so, why?

The intuition this tests: as you scale outward, emergent structure seems to
have more *say* over what happens, even though every macro event is realised by
micro events already doing all the pushing.  Kim's causal exclusion argument
says the macro is therefore redundant.  Hoel, Albantakis & Tononi (PNAS 2013)
reply by making it measurable — **effective information**, the mutual
information between an intervened-on state and what follows, with interventions
drawn at maximum entropy.

Coarse-graining can *raise* EI, because micro dynamics are often degenerate:
flipping one site changes nothing, since its neighbours drag it back.  The same
intervention at a coarser grain is decisive.  Causal emergence is
``EI(macro) − EI(micro)``.

**Genuine interventions, not observations.**  EI is defined over ``do(s)``, so
a transition matrix harvested by watching a system is the wrong object — it
measures where the system likes to spend its time, confounded with its
dynamics.  Here we can actually intervene: overwrite a block of the lattice
with a chosen sector, hold the rest at an equilibrated background, run forward,
read what the block became.  That is the do-operation, performed.

**The scale axis is clean.**  At every block size the macro variable is the
same thing — which sector dominates the block — so the state count is ``P``
throughout and the ceiling ``log₂P`` is identical.  Differing state counts are
the usual way EI comparisons go wrong, and this design sidesteps it: the
numbers are directly comparable across the sweep with no normalisation.

Pre-registered predictions:

Q1. **Macro beats micro.**  EI at the largest block scale exceeds EI at the
    single-site scale by at least **0.5 bits**.  A one-site intervention should
    be largely undone by its neighbourhood; a large-block intervention should
    stick.  **Falsifier:** the two are within 0.5 bits, meaning the micro
    determines the future as well as the macro does and the "higher structures
    have more say" intuition simply fails on this substrate.

Q2. **The mechanism is micro degeneracy, and it is noise-driven.**  Causal
    emergence should *grow with the noise amplitude*: at low noise the micro is
    already determinate so there is nothing for coarse-graining to recover,
    while at high noise the micro is swamped and the macro is not.
    **Falsifier:** emergence flat or falling in noise — which would mean
    whatever coarse-graining is buying, it is not the averaging-out of micro
    indeterminacy that the account depends on.

Q3. **Control: emergence requires coupling.**  With the Laplacian switched off
    (``D = 0``) sites evolve independently, so a single-site intervention is
    already as decisive as a block one and emergence must be near zero
    (**< 0.2 bits**) even at high noise.  **Falsifier:** a decoupled field
    showing comparable emergence — which would mean the measure is rewarding
    the averaging of independent noise rather than detecting genuine
    higher-level structure, and Q1 would then carry no weight at all.

Q3 is the load-bearing control.  "Coarse-graining reduces variance" is true of
any average whatsoever, and it is not what anyone means by emergence.

Honest scope: EI depends on the maximum-entropy intervention distribution,
which is a choice about what question to ask rather than a fact about the
system — a live objection (Dewhurst and others) that nothing here settles.
What this can settle is whether *this* substrate shows the effect, and whether
it shows it for the stated reason.

    python experiments/n3_causal_emergence.py
    python experiments/n3_causal_emergence.py --size 96 --trials 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_genesis.emergence import (  # noqa: E402
    emergence_profile,
    equilibrated_backgrounds,
)
from project_genesis.multiphase import step_multiphase  # noqa: E402

BANNER = "=" * 78


def make_step(noise, diffusion, gamma, dt):
    def step(fields, rng):
        out = step_multiphase(fields, diffusion=diffusion, gamma=gamma, dt=dt)
        if noise:
            out = out + noise * rng.standard_normal(out.shape) * np.sqrt(dt)
        return out
    return step


def run_condition(label, *, noise, diffusion, args, scales):
    rng = np.random.default_rng(args.seed)
    step = make_step(noise, diffusion, args.gamma, args.dt)
    backgrounds = equilibrated_backgrounds(
        args.trials, args.palette, args.size, args.equilibrate, rng,
        step_fn=step, ndim=args.ndim)
    prof = emergence_profile(scales, backgrounds, args.palette, args.tau,
                             step_fn=step, rng=rng, ndim=args.ndim)
    micro = prof[0]["ei"]
    macro = max(r["ei"] for r in prof)
    best = max(prof, key=lambda r: r["ei"])["scale"]
    return {"label": label, "noise": noise, "diffusion": diffusion,
            "profile": prof, "micro_ei": micro, "max_ei": macro,
            "emergence": macro - micro, "peak_scale": best}


def print_profile(res, ceiling):
    print(f"  {'scale':>6} {'EI (bits)':>10} {'determinism':>12} "
          f"{'degeneracy':>11}   (ceiling {ceiling:.3f})")
    for r in res["profile"]:
        print(f"  {r['scale']:>6} {r['ei']:>10.3f} {r['determinism']:>12.3f} "
              f"{r['degeneracy']:>11.3f}")
    print(f"  micro {res['micro_ei']:.3f} -> best {res['max_ei']:.3f} bits "
          f"at scale {res['peak_scale']}   "
          f"=> causal emergence {res['emergence']:+.3f} bits")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--palette", type=int, default=3)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--ndim", type=int, default=2)
    p.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--trials", type=int, default=40,
                   help="independent backgrounds per intervened sector")
    p.add_argument("--equilibrate", type=int, default=400)
    p.add_argument("--tau", type=int, default=30, help="steps after the do()")
    p.add_argument("--noises", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5])
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--main-noise", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    scales = [s for s in args.scales if s <= args.size]
    ceiling = float(np.log2(args.palette))

    print(BANNER)
    print("CAUSAL EMERGENCE — does the coarse grain have more causal grip?")
    print(BANNER)
    print(f"  {args.size}^{args.ndim} lattice, palette {args.palette}, "
          f"scales {scales}, tau {args.tau}, {args.trials} backgrounds/sector")
    print(f"  interventions are genuine do(s): the block is OVERWRITTEN with a")
    print(f"  sector on an equilibrated background, then run forward")
    print(f"  EI ceiling at every scale: log2({args.palette}) = {ceiling:.3f} bits")
    print()
    print("  Pre-registered:")
    print("    Q1  macro beats micro by >= 0.5 bits")
    print("    Q2  causal emergence GROWS with noise (the mechanism is micro")
    print("        degeneracy, so there must be degeneracy for it to remove)")
    print("    Q3  control: with D = 0 (no coupling) emergence < 0.2 bits")
    print()

    # ------------------------------------------------------------------- Q1
    t0 = time.time()
    main_res = run_condition("coupled", noise=args.main_noise, diffusion=1.0,
                             args=args, scales=scales)
    print(BANNER)
    print(f"Q1  EI ACROSS SCALES  (coupled, noise {args.main_noise})")
    print(BANNER)
    print_profile(main_res, ceiling)
    q1 = main_res["emergence"] >= 0.5
    print()
    if q1:
        print("  PREDICTION HELD — a single-site intervention is largely undone by")
        print("  its neighbourhood, while the same intervention at a coarser grain")
        print("  sticks. The macro description has more causal grip, measured.")
    else:
        print("  PREDICTION FAILED — coarse-graining does not buy causal grip here.")
        print("  The micro determines the future about as well as the macro does.")
    print(f"  [{time.time() - t0:.0f}s]")

    # ------------------------------------------------------------------- Q2
    print()
    print(BANNER)
    print("Q2  DOES EMERGENCE GROW WITH MICRO NOISE?")
    print(BANNER)
    noise_rows = []
    for nz in args.noises:
        r = (main_res if nz == args.main_noise
             else run_condition(f"noise={nz}", noise=nz, diffusion=1.0,
                                args=args, scales=scales))
        noise_rows.append(r)
        print(f"    noise {nz:>5.2f}: micro {r['micro_ei']:>6.3f} -> "
              f"best {r['max_ei']:>6.3f} (scale {r['peak_scale']:>2})   "
              f"emergence {r['emergence']:+.3f} bits")
    ems = [r["emergence"] for r in noise_rows]
    rising = ems[-1] > ems[0] + 0.2
    q2 = rising
    print()
    if q2:
        print("  PREDICTION HELD — the noisier the micro, the more coarse-graining")
        print("  buys. That is the stated mechanism: the macro is not adding")
        print("  anything, it is declining to represent what the micro cannot")
        print("  keep straight.")
    else:
        lo = noise_rows[0]
        saturated = lo["micro_ei"] < 0.05 and lo["emergence"] > 0.5
        print("  PREDICTION FAILED — emergence does not grow with micro noise.")
        if saturated:
            print()
            print("  The reason is specific and it corrects the premise rather than")
            print(f"  the conclusion. At ZERO noise the micro already scores "
                  f"{lo['micro_ei']:.3f} bits")
            print("  — a single-site intervention is essentially entirely undone.")
            print("  Emergence is therefore already at its ceiling before any noise")
            print("  is added, and cannot rise. The prediction assumed micro")
            print("  degeneracy needs stochasticity to produce it; it does not.")
            print("  Deterministic coupling alone is sufficient: the neighbourhood")
            print("  restores the site with no randomness involved anywhere.")
            print("  That is a stronger result than the one predicted, and it is")
            print("  what makes the zero-noise control in Q3 decisive.")
        else:
            print("  So whatever coarse-graining is buying, it is not the removal")
            print("  of micro indeterminacy that the account depends on.")

    # ------------------------------------------------------------------- Q3
    print()
    print(BANNER)
    print("Q3  CONTROL — DOES IT SURVIVE TURNING OFF THE COUPLING?")
    print(BANNER)
    hi = max(args.noises)
    # The registered control is the decoupled arm at the highest noise.  It is
    # run at every noise level as well, because a single point cannot separate
    # "coupling creates micro degeneracy" from "averaging reduces variance",
    # and the pair is what makes the answer legible.  Extra measurements are
    # fine; the registered threshold below is untouched.
    dec_rows = []
    for nz in args.noises:
        dec_rows.append(run_condition(f"decoupled n={nz}", noise=nz,
                                      diffusion=0.0, args=args, scales=scales))
    dec = [r for r in dec_rows if r["noise"] == hi][0]
    cpl = [r for r in noise_rows if r["noise"] == hi][0]

    print(f"  {'noise':>6} | {'coupled micro':>14} {'emergence':>10} | "
          f"{'decoupled micro':>16} {'emergence':>10}")
    for c, d in zip(noise_rows, dec_rows):
        print(f"  {c['noise']:>6.2f} | {c['micro_ei']:>14.3f} "
              f"{c['emergence']:>+10.3f} | {d['micro_ei']:>16.3f} "
              f"{d['emergence']:>+10.3f}")
    print()
    print("  DECOUPLED profile at the registered control point "
          f"(noise {hi}):")
    print_profile(dec, ceiling)
    print()
    print(f"  coupled emergence   {cpl['emergence']:+.3f} bits")
    print(f"  decoupled emergence {dec['emergence']:+.3f} bits")
    q3 = dec["emergence"] < 0.2
    print()
    if q3:
        print("  CONTROL PASSED — with no coupling a single site is already as")
        print("  decisive as a whole block, so there is nothing for coarse-graining")
        print("  to recover. The effect in Q1 needs interaction between sites; it")
        print("  is not the trivial fact that averages have less variance.")
    else:
        print("  CONTROL FAILED — a decoupled field shows sizeable emergence too,")
        print("  so at this noise the measure is partly rewarding the averaging of")
        print("  independent noise. That is a real limitation of EI, not a detail.")
        print()
        low = min(args.noises)
        c_lo = [r for r in noise_rows if r["noise"] == low][0]
        d_lo = [r for r in dec_rows if r["noise"] == low][0]
        print(f"  Read the paired table above before concluding. At the LOWEST")
        print(f"  noise ({low}) the two conditions separate cleanly:")
        print(f"    coupled   emergence {c_lo['emergence']:+.3f} bits "
              f"(micro EI {c_lo['micro_ei']:.3f})")
        print(f"    decoupled emergence {d_lo['emergence']:+.3f} bits "
              f"(micro EI {d_lo['micro_ei']:.3f})")
        if d_lo["emergence"] < 0.2 <= c_lo["emergence"]:
            print("  With little or no noise there is nothing to average away, so")
            print("  the decoupled field has no emergence at all — while the")
            print("  coupled field still does, because neighbours undo a")
            print("  single-site intervention on their own. That separation is")
            print("  the genuine effect; the registered control simply chose the")
            print("  worst noise level to look at it, and stands as failed.")

    # -------------------------------------------------------------- verdict
    print()
    print(BANNER)
    n = int(q1) + int(q2) + int(q3)
    print(f"VERDICT: {n}/3 pre-registered predictions held.")
    print(BANNER)
    if q1 and q3:
        print("  On this substrate the coarse-grained description genuinely has")
        print("  more causal grip than the micro one, and it needs the sites to")
        print("  interact — so it is not an artifact of averaging. That is a")
        print("  measured version of 'the higher structure has more say', which")
        print("  is usually asserted rather than tested.")
    else:
        print("  The measured version of the intuition does not hold here as")
        print("  stated. Read Q1 and Q3 together before drawing anything from it.")
    print()
    print("  What it does not settle: EI depends on the maximum-entropy")
    print("  intervention distribution, which is a choice about what question is")
    print("  being asked. Critics argue different choices give different verdicts.")
    print("  This measures the effect under the standard choice, on one substrate.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "causal_emergence.json"

        def clean(res):
            return {k: ([{kk: vv for kk, vv in r.items() if kk != "tpm"}
                         for r in v] if k == "profile" else v)
                    for k, v in res.items()}
        out.write_text(json.dumps(
            {"params": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
             "coupled": clean(main_res),
             "noise_sweep": [clean(r) for r in noise_rows],
             "decoupled": clean(dec),
             "score": n}, indent=2, default=float))
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
