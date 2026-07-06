"""Corpus-style structure seeding for the thermal sector field.

The scalar engine's memory corpus (:mod:`project_genesis.memory_corpus`,
``engine._plant_seed``) stores locally stable structure and re-roots it
under a **κ-as-soil** rule: a recalled seed unfolds only where the local
capacity is sufficient (``κ ≥ corpus_kappa_threshold``), and rooting
*consumes* that capacity (``κ *= 1 − corpus_kappa_cost``) so structure can
only re-grow where the field can support it.

That mechanism has only ever run on the scalar voxel field.  This module
carries the *same* rule — same threshold/cost/blend semantics — to the
thermal ψ∈ℂ³ sector ensemble, so the corpus's "seeds planted across
scales" can be asked of a *fluctuating* field: across the melting
transition, where can remembered structure actually re-root?  Because the
dynamical capacity field troughs exactly at criticality (see
``experiments/n3_kappa_criticality.py``), the sharp prediction is that
recall — the ability to re-root stored structure — fails at the
transition, where the soil is barren.

All helpers are dimension-agnostic (2-D point junctions or 3-D lines) and
use periodic footprints, matching the rest of the thermal layer.
"""

from __future__ import annotations

import numpy as np

# Faithful defaults from project_genesis.config (the scalar-engine κ-soil rule).
DEFAULT_KAPPA_THRESHOLD = 0.3
DEFAULT_KAPPA_COST = 0.4
DEFAULT_BLEND = 0.5


def make_sector_seed(
    n_palette: int,
    size: int,
    *,
    sector: int = 0,
    dim: int = 2,
    floor: float = 0.02,
) -> np.ndarray:
    """A fully-ordered ψ patch aligned to ``sector`` — the stored structure.

    Deep inside a converged domain ψ is (close to) a colour basis vector;
    this is that ideal, with a small ``floor`` on the other components so
    the patch is a normalised ℂ^P field rather than an exact basis vector.
    Shape ``(size,)*dim + (n_palette,)``.
    """
    shape = (size,) * dim
    seed = np.full((*shape, n_palette), floor, dtype=np.complex128)
    seed[..., sector] = 1.0
    return seed / np.linalg.norm(seed, axis=-1, keepdims=True)


def _footprint_ix(site: tuple[int, ...], size: int,
                  spatial: tuple[int, ...]) -> tuple:
    """``np.ix_`` index tuple for a cubic periodic footprint at ``site``."""
    ranges = [ (np.asarray(site)[d] + np.arange(size)) % spatial[d]
               for d in range(len(spatial)) ]
    return np.ix_(*ranges)


def local_kappa_mean(kappa: np.ndarray, site: tuple[int, ...],
                     size: int) -> float:
    """Mean capacity over the seed's periodic footprint at ``site``."""
    return float(kappa[_footprint_ix(site, size, kappa.shape)].mean())


def fertile_fraction(kappa: np.ndarray,
                     threshold: float = DEFAULT_KAPPA_THRESHOLD) -> float:
    """Fraction of the field whose capacity exceeds the rooting threshold.

    This is the *availability of fertile soil* — the fraction of sites a
    recalled seed could root into under the κ-as-soil rule.
    """
    return float(np.mean(kappa >= threshold))


def order_in_footprint(psi: np.ndarray, site: tuple[int, ...],
                       size: int) -> float:
    """Mean dominant-component amplitude ⟨max_a |ψ_a|²⟩ over the footprint.

    The local order parameter: ≈ 1 where one sector cleanly dominates
    (rooted / ordered), ≈ 1/P where the components are mixed (dissolved).
    """
    spatial = psi.shape[:-1]
    ix = _footprint_ix(site, size, spatial)
    patch = psi[ix]  # (size,)*dim × n_palette
    return float((np.abs(patch) ** 2).max(axis=-1).mean())


def plant_sector_seed(
    psi: np.ndarray,
    kappa: np.ndarray,
    seed: np.ndarray,
    site: tuple[int, ...],
    *,
    threshold: float = DEFAULT_KAPPA_THRESHOLD,
    cost: float = DEFAULT_KAPPA_COST,
    blend: float = DEFAULT_BLEND,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Attempt to root ``seed`` at ``site`` under the κ-as-soil rule.

    Faithful to ``engine._plant_seed``: if the mean local capacity is below
    ``threshold`` the seed finds barren soil and does **not** root (the
    field is returned unchanged, ``rooted=False``).  Otherwise the seed
    unfolds as a ``blend`` mix into ψ (re-normalised to keep ℂ^P unit
    vectors) and rooting draws down the local capacity by ``cost``.

    Returns ``(psi, kappa, rooted, local_kappa)`` with fresh arrays (the
    inputs are never mutated).
    """
    spatial = psi.shape[:-1]
    size = seed.shape[0]
    ix = _footprint_ix(site, size, spatial)
    local_kappa = float(kappa[ix].mean())

    if threshold > 0.0 and local_kappa < threshold:
        return psi, kappa, False, local_kappa

    psi_new = psi.copy()
    patch = (1.0 - blend) * psi_new[ix] + blend * seed
    norm = np.linalg.norm(patch, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    psi_new[ix] = patch / norm

    kappa_new = kappa.copy()
    kappa_new[ix] *= 1.0 - cost
    return psi_new, kappa_new, True, local_kappa
