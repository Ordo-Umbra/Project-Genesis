"""Regenerate the README figures from the same code the experiments use.

Figures in a repository rot the moment they stop being reproducible, so these
are generated from `n3_junction_scale.relaxed` — the exact relaxation the
published measurements run — rather than drawn by hand or kept as loose PNGs
whose provenance nobody can reconstruct.

Light and dark variants are rendered separately, from the same categorical
palette stepped for each surface. That is a selection, not an automatic flip:
GitHub serves whichever matches the reader's theme via `<picture>`.

    python tools/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.colors import ListedColormap                 # noqa: E402

from n3_junction_scale import relaxed                        # noqa: E402
from project_genesis.junction_scale import (                 # noqa: E402
    full_palette_density, neighbourhood_label_bits,
)
from project_genesis.multiphase import _popcount             # noqa: E402

# Validated categorical palette (dataviz reference instance). Assigned to
# sectors in fixed slot order and never cycled; five slots for five sectors.
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
INK = {"light": "#0b0b0b", "dark": "#ffffff"}
MUTED = {"light": "#52514e", "dark": "#c3c2b7"}

PALETTES = (2, 3, 4, 5)
SIZE, STEPS, SEED = 72, 600, 71


def junction_mask(labels: np.ndarray, n_palette: int) -> np.ndarray:
    """Cells whose 3x3 neighbourhood carries the whole palette at a vertex."""
    bits = neighbourhood_label_bits(labels, 1)
    full = (1 << int(n_palette)) - 1
    return (bits == full) & (_popcount(bits) >= 3)


def render(mode: str, out: Path) -> None:
    fig, axes = plt.subplots(1, len(PALETTES), figsize=(11.5, 3.3))
    fig.patch.set_facecolor(SURFACE[mode])
    series = LIGHT if mode == "light" else DARK

    for ax, P in zip(axes, PALETTES):
        labels = relaxed(P, SIZE, 2, STEPS, 1.5, 0.1, SEED)
        dens = full_palette_density(labels, P, 1)
        ax.imshow(labels, cmap=ListedColormap(series[:P]),
                  interpolation="nearest", vmin=0, vmax=P - 1)

        mask = junction_mask(labels, P)
        ys, xs = np.nonzero(mask)
        if len(xs):
            # 2px surface ring on an overlapping mark, per the mark spec
            ax.scatter(xs, ys, s=340, facecolors="none",
                       edgecolors=SURFACE[mode], linewidths=3.5, zorder=3)
            ax.scatter(xs, ys, s=340, facecolors="none",
                       edgecolors=INK[mode], linewidths=1.8, zorder=4)

        # a count, not a density: this is ONE field, and the published density
        # is an ensemble average over three trials. Quoting the single-field
        # density here would sit next to a different number in the docs and
        # invite the reader to think one of them is wrong.
        n = int(mask.sum())
        note = "no full-palette junction" if n == 0 else (
            f"{n} junction cell" + ("s" if n != 1 else ""))
        ax.set_title(f"P = {P}", color=INK[mode], fontsize=12, pad=8)
        ax.set_xlabel(note, color=MUTED[mode], fontsize=9, labelpad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("One relaxed sector field per palette — circled cells carry "
                 "the whole palette at a vertex",
                 color=INK[mode], fontsize=12.5, y=1.06)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=SURFACE[mode])
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    for mode in ("light", "dark"):
        render(mode, ROOT / "Docs" / "img" / f"junctions-{mode}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
