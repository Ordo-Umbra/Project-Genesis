"""Pairwise-coupled multiphase dynamics: can a wall make room for a new mode?

`multiphase.step_multiphase_conserved` couples every pair of sectors with one
shared ``gamma`` and conserves every sector's volume. Both choices close the
question this module exists to ask. With one ``gamma`` every wall costs the
same, so a third sector can never lower the cost of an A|B wall by splitting it
into A|C|B; and with every volume conserved, a sector seeded near zero can
never grow — it can only move.

The free energy here is the published one with the coupling made pairwise:

    F = Σ_a (f_a⁴/4 − f_a²/2) + Σ_{a<b} γ_ab f_a² f_b² + ½ Σ_a |∇f_a|²

and each sector is independently either **conserved** (its bulk drift has its
spatial mean subtracted, as in the published step) or **free** (plain
Allen–Cahn, so its total can change).

The mechanism this makes visible is a linear one. A sector at ``f_c = 0`` has
growth rate

    λ_c(x) = 1 − 2 Σ_{b≠c} γ_cb f_b(x)²

ignoring gradients. Inside bulk ``a`` (``f_a = 1``) that is ``1 − 2γ_ca``: the
dormant sector is stable there exactly when ``γ_ca > 1/2``. Inside an A|B wall
both ``f_A`` and ``f_B`` are below one, so ``λ_c`` is larger there than in either
bulk — the wall is where a suppressed distinction is *least* suppressed. Whether
that is enough to switch it on, against the gradient cost of a thin layer, is
the measurement.
"""

from __future__ import annotations

import numpy as np

from .multiphase import periodic_laplacian

__all__ = ["coupling_matrix", "step_multiphase_pairwise", "growth_rate"]


def coupling_matrix(n_palette: int, gamma: float = 1.5,
                    overrides: dict | None = None) -> np.ndarray:
    """Symmetric ``γ`` matrix with zero diagonal.

    ``overrides`` maps ``(a, b)`` pairs to their coupling; both orders are set.
    With no overrides this is the published uniform coupling.
    """
    g = np.full((n_palette, n_palette), float(gamma))
    np.fill_diagonal(g, 0.0)
    for (a, b), v in (overrides or {}).items():
        if a == b:
            raise ValueError("a sector has no coupling to itself")
        g[a, b] = g[b, a] = float(v)
    return g


def step_multiphase_pairwise(fields: np.ndarray, gamma: np.ndarray, *,
                             conserved=None, diffusion: float = 1.0,
                             dt: float = 0.1) -> np.ndarray:
    """One explicit step of the pairwise-coupled multiphase flow.

    ``conserved`` is a length-``P`` sequence of booleans (default: all ``True``).
    With a uniform ``gamma`` matrix and every sector conserved this is the
    published `step_multiphase_conserved` with ``kappa=None``; the equivalence is
    pinned in the tests.
    """
    gamma = np.asarray(gamma, dtype=np.float64)
    n = fields.shape[0]
    if gamma.shape != (n, n):
        raise ValueError("gamma must be (P, P)")
    if not np.allclose(gamma, gamma.T) or np.any(np.diag(gamma) != 0.0):
        raise ValueError("gamma must be symmetric with a zero diagonal")
    cons = [True] * n if conserved is None else [bool(c) for c in conserved]
    if len(cons) != n:
        raise ValueError("conserved must have one entry per sector")

    sq = fields * fields
    new = np.empty_like(fields)
    for a in range(n):
        fa = fields[a]
        cross = np.tensordot(gamma[a], sq, axes=(0, 0))
        df = fa**3 - fa + 2.0 * fa * cross
        if cons[a]:
            df = df - df.mean()
        new[a] = fa + dt * (diffusion * periodic_laplacian(fa) - df)
    return new


def growth_rate(fields: np.ndarray, gamma: np.ndarray, sector: int) -> np.ndarray:
    """Local linear growth rate of ``sector`` about zero, gradients ignored.

    ``1 − 2 Σ_{b≠c} γ_cb f_b²``. Positive means a vanishing amount of the
    sector would grow there if it were free to.
    """
    gamma = np.asarray(gamma, dtype=np.float64)
    others = [b for b in range(fields.shape[0]) if b != sector]
    cross = sum(gamma[sector, b] * fields[b] ** 2 for b in others)
    return 1.0 - 2.0 * cross
