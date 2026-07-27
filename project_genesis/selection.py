"""The selection sweep: is our corner *chosen*, or *selected*?

Reproducing known physics is under-determined — many frameworks can hit a
number, and hitting it shows only that yours is *one slice of possibility*.  A
selection argument is different and stronger: run the alternatives on the same
instrument and show that one corner is what survives.

This module supplies the instrument.  Over a possibility space (here the sector
count ``P`` — how many distinguishable kinds the field may have) it measures
three axes on one field, with one code path:

- **distinction** — the gradient/wall energy the field carries;
- **integration** — the *full-palette* junction density: a junction that carries
  every sector at once, the only structure that binds the whole palette;
- **churn** — the late-time rate at which sector membership changes: whether the
  configuration is still generating structure or has frozen.

The joint criterion is the point.  A configuration can be perfectly stable and
completely dead (the frozen dogma), or endlessly busy and bind nothing (the
fragmented mess).  What the framework's own "Good seed" criterion asks for is
**both** — persistence *and* open-endedness — so the selection score
(:func:`selection_score`) is the product of what binds and what still moves.

Nothing here privileges any ``P``: the same evolution, the same measures, the
same window.  Whether one corner wins is the measurement.
"""

from __future__ import annotations

import numpy as np

from .multiphase import (
    _total_gradient_energy,
    full_palette_junction_density,
    sector_labels,
    step_multiphase,
)

__all__ = [
    "evolve_palette",
    "measure_window",
    "selection_score",
    "normalise",
]


def evolve_palette(n_palette: int, size: int, steps: int, rng, *,
                   noise: float = 0.0, gamma: float = 1.5, dt: float = 0.1,
                   ndim: int = 2):
    """Evolve a ``P``-component sector field from noise; return the field.

    ``noise > 0`` drives the field (a bath that keeps it from freezing), so
    "still generating structure" is a meaningful question at late time.
    ``ndim`` selects the spatial dimension — the whole measurement chain is
    dimension-generic, which is what lets the selection be tested for
    *dimension-robustness* rather than assumed from one lattice.
    """
    fields = rng.uniform(0, 1, (int(n_palette),) + (size,) * int(ndim))
    for _ in range(steps):
        fields = step_multiphase(fields, gamma=gamma, dt=dt)
        if noise:
            fields = fields + noise * rng.standard_normal(fields.shape) * np.sqrt(dt)
    return fields


def measure_window(fields, n_palette: int, window: int, rng, *,
                   noise: float = 0.0, gamma: float = 1.5, dt: float = 0.1) -> dict:
    """The three axes, averaged over a late-time window.

    Returns ``distinction`` (mean gradient energy), ``integration`` (mean
    full-palette junction density), ``churn`` (mean fraction of sites changing
    sector per step), and ``sectors_alive`` (how many of the ``P`` kinds still
    occupy space — a configuration that collapses to one kind has lost the
    palette it was given).
    """
    labels = sector_labels(fields)
    churn, integ, dist = [], [], []
    for _ in range(window):
        fields = step_multiphase(fields, gamma=gamma, dt=dt)
        if noise:
            fields = fields + noise * rng.standard_normal(fields.shape) * np.sqrt(dt)
        new = sector_labels(fields)
        churn.append(float((new != labels).mean()))
        labels = new
        integ.append(float(full_palette_junction_density(labels, int(n_palette))))
        dist.append(float(_total_gradient_energy(fields).mean()))
    return {"distinction": float(np.mean(dist)),
            "integration": float(np.mean(integ)),
            "churn": float(np.mean(churn)),
            "sectors_alive": int(len(np.unique(labels)))}


def normalise(values) -> np.ndarray:
    """Scale a list of measures to ``[0, 1]`` by its maximum (0 if all zero)."""
    v = np.asarray(values, dtype=float)
    m = float(v.max())
    return v / m if m > 0 else np.zeros_like(v)


def selection_score(integration: float, churn: float) -> float:
    """The joint criterion: structure that **binds** and is still **moving**.

    A frozen configuration scores zero however well it binds; a configuration
    that binds nothing scores zero however busy it is.  Only *both* survives.
    """
    return float(max(integration, 0.0) * max(churn, 0.0))
