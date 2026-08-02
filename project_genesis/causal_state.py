"""Distinction defined by what the dynamics keep apart, not by a chosen cut.

`n3_anatomy` found that the transplant programme's ΔC is a convention on every
substrate.  The Hopfield reading counts a sample as "structured" when the
retrieval overlap clears a threshold; the oscillator reading bins effective
frequencies at a fixed width after calling an oscillator entrained inside a
tolerance.  Sweeping those numbers over ranges a reader would accept flips the
optimum's route on both substrates and slides the ΔC peak clean out of the
critical window on one.  Since ``S = ΔC + κ·ΔI``, that is a gap in the
principle's own definition and not merely in three experiments.

This module is an attempt at a ΔC that does not have the gap.  The idea is the
causal-state construction from computational mechanics: **two states are the
same state if they lead to the same futures.**  Nothing is thresholded, because
the question is never "is this state structured enough to count" — it is "can
the system's own dynamics tell these two apart".

The construction, and where each piece of resolution comes from:

1. Sample microstates from the stationary window.  No readout, no cut — the
   substrate's native state.
2. From each sampled state, launch independent stochastic replicates forward by
   ``τ`` and embed the resulting futures.  The embedding is the substrate's
   canonical one (spins as ±1, phases as ``(cos θ, sin θ)``), carrying no free
   parameter.
3. Two states are **distinguishable** when their future ensembles are further
   apart than two halves of a *single* state's replicates would be.  This is the
   whole point: the resolution is calibrated by the system's own stochasticity
   (:func:`null_calibration`), so there is no scale to choose.  A noisy system
   resolves fewer distinct states, and that is correct rather than a defect.
4. Merge indistinguishable states (:func:`causal_partition`) and read ΔC as the
   normalised entropy of the surviving classes (:func:`causal_distinction`).

What remains free, stated plainly rather than hidden: the horizon ``τ``, the
number of replicates, and the statistical confidence at which two future
ensembles count as separated.  These are not substrate semantics — they are
sample size and a p-value — and `n3_causal_delta_c` sweeps all three exactly as
`n3_anatomy` swept the threshold conventions, because a reading that is only
principled in principle is worth nothing.  If ΔC moves under them as much as
the threshold reading moved under its cuts, this construction is no better and
the experiment says so.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "future_ensemble",
    "separation",
    "null_calibration",
    "causal_partition",
    "causal_distinction",
    "causal_delta_c",
]


def future_ensemble(state, evolve, embed, tau: int, n_replicates: int,
                    rng: np.random.Generator) -> np.ndarray:
    """``(R, D)`` embedded futures from independent forward runs of one state.

    ``evolve(state, steps, rng)`` advances the substrate; ``embed(state)``
    returns its canonical vector.  The replicates share a starting state and
    differ only in the noise they draw, so their spread **is** the system's own
    indeterminacy at horizon ``τ`` — the quantity everything downstream is
    calibrated against.
    """
    out = []
    for _ in range(int(n_replicates)):
        out.append(np.asarray(embed(evolve(state, int(tau), rng)),
                              dtype=float).ravel())
    return np.asarray(out, dtype=float)


def separation(a: np.ndarray, b: np.ndarray) -> float:
    """How far apart two future ensembles are, in units of their own scatter.

    A Welch-style statistic on the ensemble means: the distance between the
    centroids divided by the standard error of that distance.  Dimensionless by
    construction, so it is comparable across substrates whose state spaces have
    different sizes and units — which the raw ΔC readings were not.
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    na, nb = a.shape[0], b.shape[0]
    if na < 2 or nb < 2:
        return float("nan")
    diff = a.mean(axis=0) - b.mean(axis=0)
    # per-coordinate squared standard error of the difference of means
    var = a.var(axis=0, ddof=1) / na + b.var(axis=0, ddof=1) / nb
    denom = float(np.sqrt(var.sum()))
    return float(np.linalg.norm(diff) / denom) if denom > 0 else float("inf")


def null_calibration(ensembles, rng: np.random.Generator,
                     n_draws: int = 200) -> np.ndarray:
    """Separations between two halves of the *same* state's replicates.

    This is the null the whole construction rests on.  Splitting one state's
    futures in two and measuring their separation says what "indistinguishable"
    numerically looks like for this system at this horizon — so the merge
    criterion is read off the substrate instead of chosen.
    """
    ensembles = [np.atleast_2d(np.asarray(e, dtype=float)) for e in ensembles]
    usable = [e for e in ensembles if e.shape[0] >= 4]
    if not usable:
        return np.array([])
    out = []
    for _ in range(int(n_draws)):
        e = usable[int(rng.integers(len(usable)))]
        idx = rng.permutation(e.shape[0])
        half = e.shape[0] // 2
        s = separation(e[idx[:half]], e[idx[half:2 * half]])
        if np.isfinite(s):
            out.append(s)
    return np.asarray(out, dtype=float)


def causal_partition(ensembles, cutoff: float) -> np.ndarray:
    """Merge states whose futures are closer than ``cutoff``.

    Average-linkage agglomeration: repeatedly join the two closest classes while
    their separation is under the cutoff.  Average linkage rather than single,
    because single linkage chains — one borderline pair would drag two genuinely
    distinct classes together and silently collapse the count.
    """
    ensembles = [np.atleast_2d(np.asarray(e, dtype=float)) for e in ensembles]
    n = len(ensembles)
    if n == 0:
        return np.array([], dtype=int)
    classes = [[i] for i in range(n)]

    d = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = separation(ensembles[i], ensembles[j])

    active = list(range(n))
    while len(active) > 1:
        best, pair = np.inf, None
        for ai, i in enumerate(active):
            for j in active[ai + 1:]:
                # average linkage over the members of each class
                vals = [d[p, q] for p in classes[i] for q in classes[j]
                        if np.isfinite(d[p, q])]
                if not vals:
                    continue
                m = float(np.mean(vals))
                if m < best:
                    best, pair = m, (i, j)
        if pair is None or best >= cutoff:
            break
        i, j = pair
        classes[i] = classes[i] + classes[j]
        active.remove(j)

    labels = np.empty(n, dtype=int)
    for k, i in enumerate(active):
        for member in classes[i]:
            labels[member] = k
    return labels


def causal_distinction(labels) -> dict:
    """ΔC as the normalised entropy of the causal-state occupancy.

    Normalised by ``log₂`` of the number of *sampled* states, so the reading is
    on ``[0, 1]`` and comparable to the threshold ΔC it is meant to replace: 0
    when the dynamics resolve one state (deep order, or noise so heavy nothing
    is distinguishable), rising as the system holds more futures apart.
    """
    labels = np.asarray(labels, dtype=int)
    n = labels.size
    if n < 2:
        return {"delta_c": 0.0, "n_causal": int(n), "n_sampled": int(n)}
    counts = np.bincount(labels)
    p = counts[counts > 0] / counts.sum()
    h = float(-(p * np.log2(p)).sum())
    return {"delta_c": float(h / np.log2(n)), "n_causal": int((counts > 0).sum()),
            "n_sampled": int(n)}


def causal_delta_c(states, evolve, embed, rng: np.random.Generator, *,
                   tau: int = 5, n_replicates: int = 12,
                   confidence: float = 0.95, n_null: int = 200) -> dict:
    """The whole pipeline: sampled states in, a dictionary-free ΔC out.

    ``confidence`` sets the merge cutoff as that quantile of the same-state null
    — states separated by more than all-but-``1−confidence`` of what a single
    state produces are counted as different.  It is a p-value, not a substrate
    semantic, and the experiment sweeps it.
    """
    ensembles = [future_ensemble(s, evolve, embed, tau, n_replicates, rng)
                 for s in states]
    null = null_calibration(ensembles, rng, n_draws=n_null)
    if null.size == 0:
        return {"delta_c": 0.0, "n_causal": 0, "n_sampled": len(states),
                "cutoff": float("nan"), "null_median": float("nan")}
    cutoff = float(np.quantile(null, float(confidence)))
    labels = causal_partition(ensembles, cutoff)
    out = causal_distinction(labels)
    out["cutoff"] = cutoff
    out["null_median"] = float(np.median(null))
    out["labels"] = labels.tolist()
    return out
