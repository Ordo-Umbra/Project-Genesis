"""Causal emergence: does a coarse-grained description have more causal grip?

The intuition is old and slippery — higher-level structure seems to "have a
say" over what happens, even though every macro event is realised by micro
events that are already doing all the pushing.  Kim's causal exclusion argument
says the macro is then redundant.  The reply that has teeth is Hoel, Albantakis
& Tononi (PNAS 2013): make it a measurement.

**Effective information.**  Take a system with a transition structure.  Set it,
by *intervention*, to each of its ``N`` states with equal probability, and see
how much the setting determines what comes next:

    EI = H(⟨p(s'|do(s))⟩_s) − ⟨H(p(s'|do(s)))⟩_s

in bits.  The first term is high when different interventions lead to different
places (low **degeneracy**); the second is low when each intervention leads
somewhere reliably (high **determinism**).  A system that ignores what you do
to it scores zero.

The reason this can favour the macro is that micro dynamics are frequently
*noisy* and *degenerate*: flipping one site often makes no difference, because
its neighbours drag it back.  Coarse-grain, and the same intervention becomes
decisive.  **Causal emergence** is then just ``EI(macro) − EI(micro)``.

**Why a simulation is the right place to measure it.**  EI is defined over
interventions, not observations, so an empirical transition matrix harvested
from watching a system is the *wrong* object — it gives observed mutual
information, which confounds the dynamics with however the system happens to
spend its time.  In a simulation we can actually perform the do-operation: set
a region to a chosen state, hold everything else at an equilibrated background,
run forward, and read off what happened.  That is what this module does.

The measure is contested, and the contested part is worth stating: EI depends
on the maximum-entropy intervention distribution, which is a choice rather than
a fact about the system.  Critics (Dewhurst among others) argue that different
choices give different verdicts, so causal emergence is partly a fact about the
observer's question.  Nothing here settles that.  What it can settle is whether
*this* substrate shows the effect at all, and whether it shows it for the
reason the framework claims.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "effective_information",
    "equilibrated_backgrounds",
    "intervention_tpm",
    "emergence_profile",
]


def effective_information(tpm: np.ndarray) -> dict:
    """EI in bits for a row-stochastic transition matrix, plus its decomposition.

    ``tpm[i, j] = p(state j at t+τ | do(state i at t))``.

    Returns ``ei`` along with the two quantities it trades off:

    - **determinism** — ``1 − ⟨H(row)⟩ / log₂N``.  1.0 when every intervention
      leads somewhere certain.
    - **degeneracy** — ``1 − H(mean row) / log₂N``.  1.0 when every
      intervention leads to the *same* place, which makes the intervention
      pointless however deterministic it is.

    ``effectiveness = EI / log₂N = determinism − degeneracy``.
    """
    tpm = np.asarray(tpm, dtype=float)
    if tpm.ndim != 2 or tpm.shape[0] != tpm.shape[1]:
        raise ValueError("tpm must be square")
    n = tpm.shape[0]
    if n < 2:
        return {"ei": 0.0, "determinism": 0.0, "degeneracy": 0.0,
                "effectiveness": 0.0, "n_states": int(n)}

    rows = tpm / np.maximum(tpm.sum(axis=1, keepdims=True), 1e-300)
    mean_row = rows.mean(axis=0)

    def entropy(p):
        p = np.asarray(p, dtype=float)
        nz = p > 0
        return float(-(p[nz] * np.log2(p[nz])).sum())

    h_mean = entropy(mean_row)
    h_rows = float(np.mean([entropy(r) for r in rows]))
    log_n = np.log2(n)
    ei = h_mean - h_rows
    return {"ei": float(ei),
            "determinism": float(1.0 - h_rows / log_n),
            "degeneracy": float(1.0 - h_mean / log_n),
            "effectiveness": float(ei / log_n),
            "n_states": int(n)}


def equilibrated_backgrounds(n_backgrounds: int, n_palette: int, size: int,
                             steps: int, rng, *, step_fn, ndim: int = 2):
    """Independent equilibrated fields to intervene on.

    The same backgrounds are reused across every scale and every intervened
    sector, so comparisons across the coarse-graining axis are *paired* — any
    difference is the scale, not the luck of the draw.
    """
    out = []
    for _ in range(int(n_backgrounds)):
        fields = rng.uniform(0, 1, (int(n_palette),) + (int(size),) * int(ndim))
        for _ in range(int(steps)):
            fields = step_fn(fields, rng)
        out.append(fields)
    return out


def _block_slices(size: int, block: int, ndim: int, offset: int = 0):
    """A centred block of edge ``block``, as an indexing tuple."""
    start = (size - block) // 2 + offset
    return (slice(None),) + tuple(slice(start, start + block) for _ in range(ndim))


def _majority_sector(fields, blk, n_palette: int) -> int:
    """The dominant sector inside the block — the macro readout at this scale."""
    labels = np.argmax(fields[blk], axis=0)
    return int(np.bincount(labels.ravel(), minlength=int(n_palette)).argmax())


def intervention_tpm(backgrounds, n_palette: int, block: int, tau: int, *,
                     step_fn, rng, ndim: int = 2) -> np.ndarray:
    """Interventional transition matrix for a block of edge ``block``.

    For each sector ``s``: overwrite the block with a clean realisation of
    ``s`` (that component set to 1, the others to 0) on an otherwise untouched
    equilibrated background, run ``tau`` steps, and record which sector the
    block ends up dominated by.  That is a genuine ``do(s)``, not an
    observation of the system's own habits.
    """
    n = int(n_palette)
    size = backgrounds[0].shape[-1]
    blk = _block_slices(size, int(block), int(ndim))
    counts = np.zeros((n, n), dtype=float)

    for s in range(n):
        for bg in backgrounds:
            fields = bg.copy()
            fields[blk] = 0.0
            fields[(s,) + blk[1:]] = 1.0
            for _ in range(int(tau)):
                fields = step_fn(fields, rng)
            counts[s, _majority_sector(fields, blk, n)] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def emergence_profile(scales, backgrounds, n_palette: int, tau: int, *,
                      step_fn, rng, ndim: int = 2) -> list:
    """EI at each coarse-graining scale, on one shared set of backgrounds.

    Every scale has the same number of macro states (``n_palette``), so the
    ceiling ``log₂P`` is identical across the sweep and the comparison needs no
    normalisation — a real convenience, since differing state counts are the
    usual way EI comparisons go wrong.
    """
    out = []
    for b in scales:
        tpm = intervention_tpm(backgrounds, n_palette, b, tau,
                               step_fn=step_fn, rng=rng, ndim=ndim)
        row = effective_information(tpm)
        row["scale"] = int(b)
        row["tpm"] = tpm
        out.append(row)
    return out
