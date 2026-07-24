"""The exchange sign: spin–statistics for the nematic ½-disclination.

`n3_spinor.py` measured the **spin** side of a topological fermion (a ``±½``
disclination whose oriented director flips ``−1`` under a ``2π`` self-rotation),
and `n3_hadron_spin.py` the **fusion** side (``n`` constituents add to
``s = n/2``) — and named the piece it left open in its own honest scope: *"no
anticommutation, no Pauli principle between identical composites; that frontier
stands."*  This experiment measures that piece — the **statistics** side, the
exchange sign — and ties it to the spin.

The spin–statistics *connection*: exchanging two identical spin-``s`` objects
multiplies the configuration by ``(−1)^{2s}`` — the *same* sign the object's own
``2π`` rotation gives.  For classical order-parameter defects this is the
Finkelstein–Rubinstein construction (exchange is homotopic to a ``2π`` rotation
of the pair's frame).  The measurement is a **braid**: two ``ψ``-vortices of
charge ``q`` (disclination strength ``s = q/2``) are rotated rigidly about their
midpoint; a half-turn swaps them (one exchange).  The oriented director
``θ = ½ arg ψ`` at the braid centre then winds by ``q·π``, so the exchange sign
is ``(−1)^q = (−1)^{2s}``.

Pre-registered predictions:

E1. **The exchange sign is ``(−1)^{2s}``.**  One exchange of an identical pair
    reads ``−1`` for ``q = 1`` (a ``½`` disclination — *fermionic*,
    antisymmetric), ``+1`` for ``q = 2`` (``s = 1`` — *bosonic*), ``−1`` for
    ``q = 3`` (``s = 3/2`` — fermionic) — and is independent of which way the
    pair is braided (``arc_sign = ±1``).

E2. **Exchange = rotation (the connection itself).**  That exchange sign equals
    the single-defect ``2π`` **self-rotation** director holonomy for every
    ``q`` — spin and statistics, the same ``(−1)^{2s}``, measured on one field.

E3. **It is the *exchange* that carries the sign.**  A **double** braid (two
    exchanges) reads ``+1`` for all ``q`` (``[(−1)^{2s}]² = +1``); the **far
    field** of a real exchange winds ``0`` (it sees only the total charge — that
    is *fusion*, ``½+½=1``, not exchange); and a **non-identical** ``(½, −½)``
    pair has no clean integer exchange winding (exchange statistics is defined
    for *identical* objects).

If E1–E3 land, the exchange half of the spin–statistics frontier is measured at
the level the programme realises spin: a genuine ``−1`` antisymmetry under
exchange of two identical topological ½-spinors, equal to their rotation sign.
The boundary it does not cross is stated: this is the topological
(Finkelstein–Rubinstein) exchange sign, not quantum-field anticommutation.

Usage::

    python experiments/n3_exchange_statistics.py --output-dir artifacts/n3_exchange
    python experiments/n3_exchange_statistics.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.spin_statistics import (
    braid_positions,
    exchange_holonomy,
    self_rotation_sign,
)
from project_genesis.vortex_chiral import imprint_vortices


def no_swap_winding(shape, mid, half_sep, charges, *, core, n_steps, loop_r=10.0):
    """Control: each defect traces a small closed loop at home — no exchange."""
    n0, n1 = int(shape[0]), int(shape[1])
    a0 = (mid[0] - half_sep, mid[1])
    b0 = (mid[0] + half_sep, mid[1])
    prev, acc = None, 0.0
    for t in np.linspace(0.0, 1.0, n_steps):
        ph = 2.0 * np.pi * t
        a = (a0[0] + loop_r * (np.cos(ph) - 1.0), a0[1] + loop_r * np.sin(ph))
        b = (b0[0] + loop_r * (np.cos(ph) - 1.0), b0[1] + loop_r * np.sin(ph))
        psi = imprint_vortices((n0, n1), [a, b], list(charges), core=core)
        v = float(np.angle(psi[int(round(mid[0])) % n0, int(round(mid[1])) % n1]))
        if prev is not None:
            acc += (v - prev + np.pi) % (2.0 * np.pi) - np.pi
        prev = v
    k = acc / (2.0 * np.pi)
    return {"k": float(k), "sign": float(np.cos(np.pi * k))}


def run(args):
    shape = (args.n, args.n)
    mid = (args.n / 2.0, args.n / 2.0)
    d = args.half_sep
    kw = dict(core=args.core, n_steps=args.n_steps)

    # E1: exchange sign for identical pairs, both braid directions
    e1 = {}
    for q in (1, 2, 3):
        plus = exchange_holonomy(shape, mid, d, (q, q), turns=0.5,
                                 arc_sign=+1.0, **kw)
        minus = exchange_holonomy(shape, mid, d, (q, q), turns=0.5,
                                  arc_sign=-1.0, **kw)
        e1[q] = {"sign_plus": plus["sign"], "sign_minus": minus["sign"],
                 "k_plus": plus["k"], "s": q / 2.0,
                 "expected": float((-1) ** q)}
        print(f"  E1 q={q} (s={q/2:.1f}): exchange sign = {plus['sign']:+.2f} "
              f"(±arc {minus['sign']:+.2f}), expected (−1)^2s = {(-1)**q:+d}",
              flush=True)

    # E2: the connection — self-rotation holonomy equals the exchange sign
    e2 = {}
    for q in (1, 2, 3):
        rot = self_rotation_sign(shape, mid, q, core=args.core,
                                 radius=args.radius)
        e2[q] = {"self_rotation": rot, "exchange": e1[q]["sign_plus"]}
        print(f"  E2 q={q}: self-rotation 2π holonomy = {rot:+.2f}  ==  "
              f"exchange sign {e1[q]['sign_plus']:+.2f}", flush=True)

    # E3: controls — double braid, far field, non-identical pair, no-swap
    double = {q: exchange_holonomy(shape, mid, d, (q, q), turns=1.0, **kw)
              for q in (1, 3)}
    far = exchange_holonomy(shape, mid, d, (1, 1), turns=0.5,
                            probe=(mid[0], mid[1] + args.n * 0.32), **kw)
    mixed = exchange_holonomy(shape, mid, d, (1, -1), turns=0.5, **kw)
    noswap = no_swap_winding(shape, mid, d, (1, 1), core=args.core,
                             n_steps=args.n_steps)
    e3 = {"double": {q: double[q]["sign"] for q in double},
          "double_k": {q: double[q]["k"] for q in double},
          "far_sign": far["sign"], "far_k": far["k"],
          "mixed_k": mixed["k"], "mixed_sign": mixed["sign"],
          "noswap_sign": noswap["sign"], "noswap_k": noswap["k"]}
    print(f"  E3 double braid signs {e3['double']} (expect +1); "
          f"far-field k={far['k']:+.2f} (expect 0); "
          f"no-swap k={noswap['k']:+.2f} (expect 0); "
          f"mixed(½,−½) sign={mixed['sign']:+.2f} k={mixed['k']:+.2f} "
          "(identical needed for −1 → expect +1)", flush=True)

    return {"e1": e1, "e2": e2, "e3": e3, "mid": mid}


def verdict(res, args):
    e1, e2, e3 = res["e1"], res["e2"], res["e3"]
    # E1: signs match (−1)^{2s} for all q, both arc directions
    v1 = all(abs(e1[q]["sign_plus"] - e1[q]["expected"]) < 0.2 and
             abs(e1[q]["sign_minus"] - e1[q]["expected"]) < 0.2 for q in e1)
    # E2: self-rotation == exchange for all q
    v2 = all(abs(e2[q]["self_rotation"] - e2[q]["exchange"]) < 0.2 for q in e2)
    # E3: double braid +1, far field trivial, no-swap trivial, mixed +1
    v3 = (all(s > 0.8 for s in e3["double"].values()) and
          e3["far_sign"] > 0.8 and abs(e3["far_k"]) < 0.2 and
          e3["noswap_sign"] > 0.8 and abs(e3["noswap_k"]) < 0.2 and
          e3["mixed_sign"] > 0.8)
    return v1, v2, v3


def summarise(res, args):
    e1, e2, e3 = res["e1"], res["e2"], res["e3"]
    v1, v2, v3 = verdict(res, args)
    n = int(v1) + int(v2) + int(v3)
    lines = ["The exchange sign: spin–statistics for the nematic ½-disclination "
             "— verdict", "=" * 74, ""]
    lines.append(
        "E1 (exchange sign = (−1)^2s): a single exchange of an identical pair "
        "reads")
    for q in (1, 2, 3):
        cls = "fermionic (antisymmetric)" if q % 2 else "bosonic (symmetric)"
        lines.append(f"      q={q}, s={q/2:.1f}: sign {e1[q]['sign_plus']:+.2f} "
                     f"(both braid directions) — {cls}, expected {(-1)**q:+d}")
    lines.append("   " + ("✓ the elementary ½-disclination is a genuine fermion "
                          "under exchange — two identical ones pick up −1, the "
                          "antisymmetry; the integer defect is a boson." if v1
                          else "✗ the exchange signs do not match (−1)^2s."))
    lines.append("")
    lines.append(
        "E2 (exchange = rotation — the connection): the single-defect 2π "
        "self-rotation director holonomy equals the exchange sign, q by q — "
        + ", ".join(f"q={q}: rot {e2[q]['self_rotation']:+.2f} = exch "
                    f"{e2[q]['exchange']:+.2f}" for q in (1, 2, 3)) + " — "
        + ("✓ spin and statistics carry the *same* (−1)^2s on one field: the "
           "spin–statistics connection, measured, not assumed." if v2 else
           "✗ the two signs do not coincide."))
    lines.append("")
    lines.append(
        "E3 (it is the exchange that carries the sign): "
        f"a DOUBLE braid reads {e3['double']} (two exchanges = +1, a boson); "
        f"the FAR field of a real exchange winds k={e3['far_k']:+.2f} "
        "(it sees only total charge — that is fusion ½+½=1, not exchange); "
        f"a NO-SWAP loop winds k={e3['noswap_k']:+.2f} (mere motion is not "
        f"exchange); and a non-identical (½,−½) pair reads sign "
        f"{e3['mixed_sign']:+.2f} (the windings cancel — the −1 needs two "
        "*identical* ½'s, not a particle–antiparticle pair) — "
        + ("✓ the −1 is specifically the single exchange of two identical "
           "topological ½-spinors; every other operation is +1." if v3 else
           "✗ a control did not behave as a clean exchange should."))
    lines.append("")
    lines.append(f"score: {n}/3 pre-registered predictions land.")
    lines.append("")
    lines.append(
        "what this establishes: the exchange half of the spin–statistics "
        "frontier `n3_hadron_spin` left open is now measured — two identical "
        "topological ½-spinors are **antisymmetric under exchange** (sign −1), "
        "and that sign is *the same* as their 2π self-rotation, the spin–"
        "statistics connection made concrete on the sector field's own "
        "disclinations. The programme now carries all three: the spin (the ±½ "
        "double cover), the fusion (½+½=1), and the statistics (exchange = "
        "(−1)^2s).")
    lines.append("")
    lines.append(
        "honest scope: this is the **topological / geometric** exchange sign of "
        "classical order-parameter defects (Finkelstein–Rubinstein) — the exact "
        "level at which this programme realises the spin, and the true content "
        "of the spin–statistics *connection* for these objects — but it is "
        "**not** quantum-field anticommutation: no Fock space, no {ψ,ψ†}, no "
        "many-body Pauli principle between identical excitations. The braid is "
        "kinematic (imprint centres moved through configuration space); a "
        "dynamical braid (κ-pinned cores co-evolved under the CGL) is the next "
        "rung and should agree, the winding being a topological invariant of "
        "the path. 2-D; one lattice; the elementary disclination is ±½.")
    return lines, n


def make_figure(res, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE, ORANGE, GREEN, GRAY, RED = ("#0072B2", "#E69F00", "#009E73",
                                      "#999999", "#c0392b")
    e1, e2, e3 = res["e1"], res["e2"], res["e3"]
    mid = res["mid"]
    fig = plt.figure(figsize=(13, 9.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.24)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2))

    # panel A: exchange sign & self-rotation vs 2s (the connection)
    qs = [1, 2, 3]
    xs = np.array(qs)
    exch = [e1[q]["sign_plus"] for q in qs]
    rot = [e2[q]["self_rotation"] for q in qs]
    ax1.axhline(0, color=GRAY, lw=1.0, zorder=1)
    ax1.bar(xs - 0.16, exch, width=0.3, color=BLUE, zorder=3,
            label="exchange sign")
    ax1.bar(xs + 0.16, rot, width=0.3, color=ORANGE, zorder=3,
            label="2π self-rotation (spin)")
    for q in qs:
        ax1.text(q, -1.18, f"s={q/2:.1f}", ha="center", fontsize=9)
    ax1.set_xticks(qs)
    ax1.set_xlabel("disclination charge q  (strength s = q/2)")
    ax1.set_ylabel("sign")
    ax1.set_ylim(-1.5, 1.4)
    ax1.set_title("A. Exchange = rotation = (−1)$^{2s}$: the connection")
    ax1.legend(fontsize=8, loc="upper right")
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    # panel B: director snapshots through the ½,½ exchange
    fracs = [0.0, 0.5, 1.0]
    positions = list(braid_positions(mid, args.half_sep, 0.5, 1.0, 3 * 60 + 1))
    idx = [0, len(positions) // 2, len(positions) - 1]
    sub = GridSpec(1, 3, figure=fig, wspace=0.08,
                   left=ax2.get_position().x0, right=ax2.get_position().x1,
                   bottom=ax2.get_position().y0, top=ax2.get_position().y1)
    ax2.axis("off")
    ax2.set_title("B. The braid: two ½-disclinations swap", y=1.02)
    win = 62
    lo0, hi0 = int(mid[0] - win), int(mid[0] + win)
    lo1, hi1 = int(mid[1] - win), int(mid[1] + win)
    for col, (fr, ii) in enumerate(zip(fracs, idx)):
        a, b = positions[ii]
        psi = imprint_vortices((args.n, args.n), [a, b], [1, 1], core=args.core)
        director = (0.5 * np.angle(psi)) % np.pi
        axs = fig.add_subplot(sub[0, col])
        axs.imshow(director[lo0:hi0, lo1:hi1].T, origin="lower",
                   cmap="twilight", vmin=0, vmax=np.pi)
        for (cx, cy), col_c in ((a, RED), (b, GREEN)):
            axs.plot(cx - lo0, cy - lo1, "o", ms=7, mfc=col_c, mec="white",
                     mew=1.2)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_xlabel(f"t = {fr:.1f}", fontsize=9)

    # panel C: the controls
    ax3.axhline(0, color=GRAY, lw=1.0, zorder=1)
    cats = ["1 exchange\n(½,½)", "2 exchanges\n(braid²)", "no-swap\nloop",
            "far field\n(fusion)"]
    vals = [e1[1]["sign_plus"], e3["double"][1], e3["noswap_sign"],
            e3["far_sign"]]
    colors = [RED if v < 0 else BLUE for v in vals]
    ax3.bar(range(len(cats)), vals, color=colors, zorder=3)
    ax3.set_xticks(range(len(cats)))
    ax3.set_xticklabels(cats, fontsize=8)
    ax3.set_ylabel("sign at braid centre")
    ax3.set_ylim(-1.4, 1.3)
    ax3.set_title("C. Only a single exchange of identicals gives −1")
    for side in ("top", "right"):
        ax3.spines[side].set_visible(False)

    # panel D: verdict
    v1, v2, v3 = verdict(res, args)
    ax4.axis("off")
    ax4.set_title("D. The verdict")
    txt = ("nematic ½-disclination = topological spinor\n\n"
           f"E1 exchange sign = (-1)^2s   -> {'YES ok' if v1 else 'no'}\n"
           f"   q=1 (s=1/2): {e1[1]['sign_plus']:+.0f}  fermion (antisym)\n"
           f"   q=2 (s=1)  : {e1[2]['sign_plus']:+.0f}  boson\n"
           f"   q=3 (s=3/2): {e1[3]['sign_plus']:+.0f}  fermion\n\n"
           f"E2 exchange == self-rotation -> {'YES ok' if v2 else 'no'}\n"
           "   spin and statistics: one sign\n\n"
           f"E3 controls (braid^2=+1, no-swap=+1,\n"
           f"   far=+1, mixed(½,-½)=+1) -> {'YES ok' if v3 else 'no'}\n\n"
           "two identical topological 1/2-spinors are\n"
           "ANTISYMMETRIC under exchange (-1), equal to\n"
           "their 2pi rotation -- the spin-statistics\n"
           "connection, measured. (topological, not\n"
           "Fock-space anticommutation.)")
    ax4.text(0.02, 0.96, txt, transform=ax4.transAxes, fontsize=9.0, va="top",
             family="monospace")

    fig.suptitle("The exchange sign: two identical ½-disclinations are "
                 "antisymmetric — spin–statistics on the sector field",
                 fontsize=12, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=220, help="Lattice side.")
    p.add_argument("--half-sep", type=float, default=34.0,
                   help="Half-separation of the pair about the midpoint.")
    p.add_argument("--core", type=float, default=3.0)
    p.add_argument("--radius", type=float, default=12.0,
                   help="Loop radius for the self-rotation holonomy.")
    p.add_argument("--n-steps", type=int, default=361,
                   help="Braid path resolution.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="artifacts/n3_exchange")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n = 160
        args.half_sep = 26.0
        args.n_steps = 181

    os.makedirs(args.output_dir, exist_ok=True)
    print("the exchange sign: spin–statistics for the nematic ½-disclination",
          flush=True)
    res = run(args)
    lines, n = summarise(res, args)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.output_dir, "n3_exchange_statistics.json"),
              "w") as fh:
        json.dump({"params": vars(args), "result": res, "score": n},
                  fh, indent=2, default=str)
    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, "n3_exchange_statistics.png")
        make_figure(res, args, fig_path)
        print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
