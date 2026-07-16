"""The scarcity-scaled benchmark: does protection pay when capacity genuinely binds?

The combined benchmark's G1 failure had a measured cause: at hidden = 48 the
trunk has room for everything, so an unprotected network loses nothing.  The
programme's own law says the interesting regime is *scarcity* — so this
experiment turns the one dial that creates it: the trunk width.  Same
sequence (A → B → XOR(A,B) → dense conflict), same optimizers that survived
the arc (plain SGD, functional κ, rehearsal at equal information), hidden
width swept from abundant to starved.

Pre-registered predictions:

H1. **Protection pays under scarcity**: at the narrowest trunk, functional κ
    beats plain SGD on the combined average, significantly.
H2. **Still competitive with rehearsal** across the scan (within noise).
H3. **The advantage is scarcity-graded**: the functional-vs-plain margin is
    larger at the narrowest width than at the widest.

Usage::

    python experiments/n3_scarcity_benchmark.py --output-dir artifacts/n3_scarcity
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from n3_combined_benchmark import one_run

OPTS = ("plain-sgd", "functional-kappa", "rehearsal")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dim", type=int, default=24)
    p.add_argument("--hiddens", type=int, nargs="+", default=[8, 12, 16, 24, 48])
    p.add_argument("--n-train", type=int, default=1200)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--consumption", type=float, default=4.0)
    p.add_argument("--recovery", type=float, default=0.1)
    p.add_argument("--buffer-size", type=int, default=64)
    p.add_argument("--n-seeds", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_scarcity")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_train, args.epochs, args.n_seeds = 600, 15, 3
        args.hiddens = [8, 48]

    os.makedirs(args.output_dir, exist_ok=True)
    print("the scarcity-scaled benchmark: trunk width as the capacity dial",
          flush=True)
    scan = []
    for hidden in args.hiddens:
        args.hidden = hidden
        row = {"hidden": hidden}
        for opt_name in OPTS:
            avgs = np.array([one_run(args, args.seed + 100 * s, opt_name)["avg"]
                             for s in range(args.n_seeds)])
            row[opt_name] = float(avgs.mean())
            row[opt_name + "_err"] = float(avgs.std() / np.sqrt(args.n_seeds))
        row["margin"] = row["functional-kappa"] - row["plain-sgd"]
        print(f"  hidden={hidden:>2}: plain={row['plain-sgd']:.3f}  "
              f"functional={row['functional-kappa']:.3f}  "
              f"rehearsal={row['rehearsal']:.3f}  "
              f"margin={row['margin']:+.3f}", flush=True)
        scan.append(row)

    narrow, wide = scan[0], scan[-1]
    e1 = float(np.hypot(narrow["functional-kappa_err"], narrow["plain-sgd_err"]))
    h1 = narrow["margin"] > max(0.02, 2.0 * e1)
    h2 = all(r["functional-kappa"] - r["rehearsal"]
             > -max(0.02, 2.0 * np.hypot(r["functional-kappa_err"],
                                         r["rehearsal_err"])) for r in scan)
    h3 = narrow["margin"] > wide["margin"] + 0.02

    lines = ["The scarcity-scaled benchmark — verdict", "=" * 74, ""]
    lines.append(
        f"H1 (protection pays under scarcity): at hidden = {narrow['hidden']}, "
        f"functional κ vs plain SGD margin {narrow['margin']:+.3f} ± {e1:.3f} — "
        + ("✓ the advantage the abundant regime could not show." if h1 else
           "✗ even a starved trunk does not reward protection here."))
    lines.append(
        "H2 (competitive with rehearsal across the scan): "
        + ("✓ within noise everywhere." if h2 else "✗ replay wins somewhere."))
    lines.append(
        f"H3 (scarcity-graded): margin {narrow['margin']:+.3f} at "
        f"hidden={narrow['hidden']} vs {wide['margin']:+.3f} at "
        f"hidden={wide['hidden']} — "
        + ("✓ the advantage grows as capacity binds — the law's own "
           "signature." if h3 else "✗ no scarcity grading."))
    lines.append("")
    lines.append(f"score: {int(h1)+int(h2)+int(h3)}/3 pre-registered "
                 "predictions land.")
    lines.append("")
    lines.append(
        "honest scope: width is one scarcity dial among several (sequence "
        "length, data, lr schedule); same miniature task family; rehearsal "
        "remains the naive schedule. Margins at starved widths carry higher "
        "variance — seeds are reported per point.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_scarcity_benchmark.json"),
              "w") as fh:
        json.dump({"params": vars(args), "scan": scan,
                   "analysis": {"h1": h1, "h2": h2, "h3": h3}},
                  fh, indent=2, default=str)


if __name__ == "__main__":
    main()
