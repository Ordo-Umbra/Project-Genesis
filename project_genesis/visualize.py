"""Matplotlib-based 3-D voxel visualization for terrain inspection.

Provides functions to render the quantized voxel field as an interactive
3-D scatter plot (one marker per solid voxel, coloured by band) and to
export static images or display the figure.

Usage::

    from project_genesis import GenesisEngine, EngineConfig
    from project_genesis.visualize import render_voxels_3d, plot_s_history

    engine = GenesisEngine(config=EngineConfig(chunk_size=24, seed=7))
    engine.evolve_field(steps=40, dt=0.01, record_every=5)

    # Interactive 3-D voxel view
    fig = render_voxels_3d(engine.quantize_to_voxels())
    fig.savefig("terrain.png", dpi=150)

    # S-functional history chart
    fig2 = plot_s_history(engine.history)
    fig2.savefig("s_history.png", dpi=150)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless environments.
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    _HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    _HAS_MATPLOTLIB = False

# Band display configuration (matches render.py).
BAND_CONFIG = {
    # band_id: (name, colour, alpha, marker_size)
    0: ("Void", "#000000", 0.0, 0),        # not rendered
    1: ("Air", "#88ccff", 0.15, 8),
    2: ("Soil", "#8b6914", 0.6, 12),
    3: ("Stone", "#888888", 0.85, 14),
    4: ("Bedrock", "#333333", 1.0, 16),
}


def _require_matplotlib() -> None:
    if not _HAS_MATPLOTLIB:
        raise RuntimeError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def render_voxels_3d(
    voxels: np.ndarray,
    *,
    skip_void: bool = True,
    skip_air: bool = False,
    title: str = "Project Genesis – Voxel Terrain",
    figsize: tuple[float, float] = (10, 8),
    elev: float = 25.0,
    azim: float = -60.0,
) -> "Figure":
    """Render the 3-D voxel field as a matplotlib scatter plot.

    Parameters
    ----------
    voxels:
        Integer array of shape ``(nx, ny, nz)`` with values 0–4.
    skip_void:
        If True (default), void voxels (band 0) are not drawn.
    skip_air:
        If True, air voxels (band 1) are also omitted for clarity.
    title:
        Figure title.
    figsize:
        Figure dimensions in inches.
    elev, azim:
        Initial viewing angles for the 3-D projection.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure (call ``.savefig()`` or ``plt.show()``).
    """
    _require_matplotlib()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    min_band = 2 if skip_air else (1 if skip_void else 0)

    for band_id in range(min_band, 5):
        name, colour, alpha, size = BAND_CONFIG[band_id]
        if alpha == 0.0:
            continue
        mask = voxels == band_id
        if not np.any(mask):
            continue
        xs, ys, zs = np.where(mask)
        ax.scatter(
            xs, ys, zs,
            c=colour,
            alpha=alpha,
            s=size,
            label=name,
            edgecolors="none",
            depthshade=True,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig


def render_field_slices(
    field: np.ndarray,
    *,
    title: str = "Field Cross-Sections",
    figsize: tuple[float, float] = (14, 4),
    cmap: str = "terrain",
) -> "Figure":
    """Render centre slices of the raw scalar field along x, y, z axes.

    Parameters
    ----------
    field:
        3-D scalar field array.
    title:
        Super-title for the figure.
    figsize:
        Figure dimensions in inches.
    cmap:
        Matplotlib colour map name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()

    mid = [s // 2 for s in field.shape]
    slices = [
        ("YZ (x={})".format(mid[0]), field[mid[0], :, :]),
        ("XZ (y={})".format(mid[1]), field[:, mid[1], :]),
        ("XY (z={})".format(mid[2]), field[:, :, mid[2]]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=13)

    for ax, (label, data) in zip(axes, slices):
        im = ax.imshow(data.T, origin="lower", cmap=cmap, aspect="equal")
        ax.set_title(label, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75)

    fig.tight_layout()
    return fig


def plot_s_history(
    history: list[dict[str, Any]],
    *,
    title: str = "S-Functional Over Time",
    figsize: tuple[float, float] = (10, 5),
) -> "Figure":
    """Plot S-functional components (ΔC, ΔI, κ, S) from engine history.

    Parameters
    ----------
    history:
        The ``engine.history`` list of metric snapshots.
    title:
        Figure title.
    figsize:
        Figure dimensions in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()

    steps = [int(h.get("step", i)) for i, h in enumerate(history)]
    delta_c = [float(h.get("delta_c", 0)) for h in history]
    delta_i = [float(h.get("delta_i", 0)) for h in history]
    kappa = [float(h.get("kappa", 0)) for h in history]
    s_inc = [float(h.get("s_increment", 0)) for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=13)

    ax1.plot(steps, s_inc, "g-", linewidth=1.5, label="S increment")
    ax1.plot(steps, delta_c, "b--", linewidth=1, label="ΔC")
    ax1.plot(steps, delta_i, "r--", linewidth=1, label="ΔI")
    ax1.set_ylabel("Value")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(steps, kappa, "m-", linewidth=1.5, label="κ (capacity)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("κ")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def save_visualization(
    output_dir: str | Path,
    voxels: np.ndarray,
    field: np.ndarray,
    history: list[dict[str, Any]],
    *,
    dpi: int = 150,
) -> list[Path]:
    """Generate and save all standard visualization artifacts.

    Writes:
    - ``voxel_3d.png`` — 3-D voxel scatter plot
    - ``field_slices.png`` — Centre-slice heat maps
    - ``s_history.png`` — S-functional time series

    Returns the list of paths written.
    """
    _require_matplotlib()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig = render_voxels_3d(voxels)
    path = out / "voxel_3d.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)

    fig = render_field_slices(field)
    path = out / "field_slices.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)

    if history:
        fig = plot_s_history(history)
        path = out / "s_history.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)

    return written
