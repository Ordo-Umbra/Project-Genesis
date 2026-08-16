"""Coarse-graining as an operation, so "what survives" can be an output.

The programme's recurring problem is that `S = ΔC + κ·ΔI` **scores** a
description rather than **transforming** one. A score needs someone to have
already chosen the level of description, and §5's conclusion is that the choice
is not forced by anything — which is where the whole thing stalls.

The renormalization group is the worked precedent for getting out of that. It
does not score descriptions; it *flows* them. Pick a coarse-graining operation,
apply it repeatedly, and watch which features grow, shrink, or hold. The
classification that falls out — relevant, irrelevant, marginal — is not chosen.
Critically, **the cut is a choice and the classification is not**: universality
classes come out the same across different microscopic models and different
blocking schemes. That is exactly the property this programme wants and does not
have: an arbitrariness that stops mattering downstream.

This module supplies the missing operation. Three blocking schemes, because a
single one proves nothing — if a quantity's flow depends on which scheme you
picked, the flow is a fact about the scheme.

**The schemes are deliberately different in kind.** Majority is smoothing and
loses minority structure; decimation is sampling and keeps it; random-pick is
sampling with a different bias. A quantity whose flow agrees across all three is
a candidate for something real. One that disagrees is an artefact of the cut, and
saying so is the point.

Observables are all **dimensionless fractions**, so they can be compared across
levels where the lattice has a different size. A quantity carrying lattice units
would flow trivially — smaller lattice, smaller number — and that flow would
mean nothing.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "block_majority",
    "block_decimate",
    "block_random",
    "SCHEMES",
    "wall_density",
    "order_parameter",
    "largest_component_fraction",
    "observables",
    "rg_flow",
]


def _cropped(labels: np.ndarray, b: int) -> np.ndarray:
    """Trim each axis to a multiple of ``b`` so blocking is exact."""
    if b < 2:
        raise ValueError("block factor must be at least 2")
    sl = tuple(slice(0, (n // b) * b) for n in labels.shape)
    return labels[sl]


def _blocks(labels: np.ndarray, b: int) -> np.ndarray:
    """Reshape to ``(n1//b, b, n2//b, b, ...)`` then move block axes last."""
    a = _cropped(labels, b)
    d = a.ndim
    shape = []
    for n in a.shape:
        shape += [n // b, b]
    a = a.reshape(shape)
    # axes 0,2,4,... are cell indices; 1,3,5,... are within-block
    order = list(range(0, 2 * d, 2)) + list(range(1, 2 * d, 2))
    # a.shape[::2] is already the coarse cell grid after the reshape above --
    # dividing by b again would halve it twice.
    cells = tuple(a.shape[::2])
    return a.transpose(order).reshape(cells + (b ** d,))


def block_majority(labels: np.ndarray, b: int, n_palette: int, rng=None):
    """Each block takes the commonest label in it — smoothing.

    Loses minority structure, which is the point: it is the scheme most likely
    to wash out anything fine-grained, and therefore the one that will disagree
    with the sampling schemes if a quantity depends on fine detail.
    """
    flat = _blocks(labels, b)
    counts = np.stack([(flat == k).sum(axis=-1) for k in range(int(n_palette))],
                      axis=-1)
    return np.argmax(counts, axis=-1)


def block_decimate(labels: np.ndarray, b: int, n_palette: int, rng=None):
    """Keep one fixed representative per block — Kadanoff decimation.

    Sampling rather than smoothing: a minority label survives if it happens to
    sit at the sampled corner. Structurally different from majority, which is
    why the pair is informative.
    """
    a = _cropped(labels, b)
    return a[tuple(slice(0, None, b) for _ in range(a.ndim))]


def block_random(labels: np.ndarray, b: int, n_palette: int, rng=None):
    """Keep a uniformly random representative per block.

    Decimation without the fixed-corner bias. If a quantity's flow differs
    between this and :func:`block_decimate`, the difference is *where in the
    block* you sampled, which is as arbitrary as a cut can get.
    """
    rng = np.random.default_rng() if rng is None else rng
    flat = _blocks(labels, b)
    pick = rng.integers(0, flat.shape[-1], size=flat.shape[:-1])
    return np.take_along_axis(flat, pick[..., None], axis=-1)[..., 0]


SCHEMES = {"majority": block_majority,
           "decimate": block_decimate,
           "random": block_random}


# --------------------------------------------------------------------------
# observables — all dimensionless, so levels are comparable

def _neighbour_offsets(ndim: int):
    from itertools import product
    return [o for o in product((-1, 0, 1), repeat=ndim) if any(o)]


def wall_density(labels: np.ndarray) -> float:
    """Fraction of cells with at least one differing neighbour — *distinction*.

    The label-field analogue of gradient energy, as a fraction rather than a
    sum, so it does not shrink merely because the lattice did.
    """
    diff = np.zeros(labels.shape, dtype=bool)
    for off in _neighbour_offsets(labels.ndim):
        shifted = labels
        for axis, delta in enumerate(off):
            if delta:
                shifted = np.roll(shifted, -delta, axis=axis)
        diff |= (shifted != labels)
    return float(diff.mean())


def order_parameter(labels: np.ndarray, n_palette: int) -> float:
    """How far the field has committed to one sector, on ``[0, 1]``.

    ``0`` when every sector holds an equal share, ``1`` when one has taken
    everything. Normalised by the palette so different ``P`` are comparable.
    """
    p = int(n_palette)
    frac = max(float((labels == k).mean()) for k in range(p))
    return float((frac - 1.0 / p) / (1.0 - 1.0 / p)) if p > 1 else 1.0


def largest_component_fraction(labels: np.ndarray) -> float:
    """Share of the field in the single largest connected same-label region —
    *integration*.

    Distinct from :func:`order_parameter`: a sector can hold half the volume as
    a thousand islands (high order, low integration) or as one piece (both
    high). This is the "can it be held as one thing" reading.
    """
    try:
        from scipy import ndimage
    except ImportError:                                   # pragma: no cover
        return float("nan")
    best = 0
    structure = ndimage.generate_binary_structure(labels.ndim, 1)
    for k in np.unique(labels):
        comp, n = ndimage.label(labels == k, structure=structure)
        if n:
            best = max(best, int(np.bincount(comp.ravel())[1:].max()))
    return float(best) / float(labels.size)


def observables(labels: np.ndarray, n_palette: int) -> dict:
    return {"wall_density": wall_density(labels),
            "order_parameter": order_parameter(labels, n_palette),
            "largest_component": largest_component_fraction(labels),
            "cells": int(labels.size)}


def rg_flow(labels: np.ndarray, n_palette: int, scheme: str, *,
            levels: int = 4, b: int = 2, min_side: int = 8, rng=None) -> list:
    """Apply one blocking scheme repeatedly, measuring at every level.

    Returns the observables at level 0 (the original field) and after each
    blocking, stopping early if any axis would fall below ``min_side`` — below
    that the statistics are dominated by the boundary and the reading is about
    the lattice, not the field.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}")
    fn = SCHEMES[scheme]
    rng = np.random.default_rng(0) if rng is None else rng
    out, cur = [observables(labels, n_palette)], labels
    for _ in range(int(levels)):
        if min(s // b for s in cur.shape) < min_side:
            break
        cur = fn(cur, b, n_palette, rng)
        out.append(observables(cur, n_palette))
    return out
