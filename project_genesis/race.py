"""The symmetry-breaking race: is the early lead a cause, or only a marker?

``n3_survival`` measured that ``maxfrac`` — how much of the lattice the largest
sector already holds at step 60 — predicts whether a run survives, with a
held-out AUC of 0.801 on a fresh ensemble.  That is a strong correlation and it
was confirmed out of sample, but it is still a correlation: the early lead and
the eventual death could both be downstream of something else in the initial
condition, in which case ``maxfrac`` is a *marker* and the "race" story is
decoration.

The way to tell is to stop observing the lead and start **setting** it.  If a
lead imposed by hand, on a field that is otherwise a fair draw, reduces
survival, then the lead is doing causal work.  If survival is flat in the
imposed lead, it is not — and the shape of that dose-response is the whole
result.

This module supplies the intervention:

- :func:`seed_with_lead` — a fair random field with one sector handed a
  measured head start, everything else untouched;
- :func:`run_race` — evolve an ensemble, follow the gap between the top two
  sector shares over time, and record who dies when;
- :func:`growth_rate` — fit ``log(gap)`` against time, because a *race* has a
  specific signature: the lead grows in proportion to itself, so the gap grows
  **exponentially**.  Drift or diffusion would give a straight line or a square
  root instead, and those are different mechanisms wearing the same correlation.

The distinction matters beyond bookkeeping.  A race with two absorbing outcomes
has a **separatrix** — a critical lead above which runaway is near-certain — and
that is a threshold in a quantity we control, which is a far stronger claim than
a curve fitted through observed runs.

**Measured outcome, and it is not what the module is named for.**
``n3_race`` finds the lead is genuinely **causal** — imposing it collapses
survival from 0.44 to 0.00 across a 1.6-percentage-point change in the initial
share, a cliff rather than a slope.  But the growth law came out **linear**, not
exponential: over the early window ``gap ∝ t`` fits at ``R² = 0.994`` against
``0.949`` for the compounding form.  So the advantage does **not** compound, and
"race" is the wrong word for it.  What the data supports is a **first-passage**
picture — the gap opens at a roughly constant rate and a run dies when it
crosses threshold, so a head start buys a proportionally earlier crossing rather
than an accelerating runaway.  That also explains the cliff without needing
autocatalysis.

The module keeps the name because it is named for the hypothesis it was built to
test, and the answer to that hypothesis is on the record above.
"""

from __future__ import annotations

import numpy as np

from .survival import sector_labels_batch, step_batch

__all__ = [
    "seed_with_lead",
    "sector_fractions",
    "top_gap",
    "run_race",
    "growth_rate",
]


def seed_with_lead(n_runs: int, n_palette: int, size: int, rng, *,
                   lead: float = 0.0, ndim: int = 3, sector: int = 0):
    """A fair random field, with one sector's amplitude offset by ``lead``.

    ``lead = 0`` is exactly the unbiased seed the survival work used.  A
    positive ``lead`` adds a constant to one component *before* normalising, so
    the field is still a legitimate configuration on the unit sphere — nothing
    is clamped or hand-placed, only tilted.  The realised head start is then
    *measured* rather than assumed, since what matters is the sector share the
    tilt actually produced, not the knob that produced it.
    """
    shape = (int(n_runs), int(n_palette)) + (int(size),) * int(ndim)
    fields = rng.uniform(-0.5, 0.5, shape)
    if lead:
        fields[:, int(sector)] += float(lead)
    norm = np.sqrt((fields ** 2).sum(axis=1, keepdims=True)) + 1e-12
    return fields / norm


def sector_fractions(labels: np.ndarray, n_palette: int) -> np.ndarray:
    """``(R, P)`` occupancy shares."""
    R = labels.shape[0]
    flat = labels.reshape(R, -1)
    out = np.empty((R, int(n_palette)), dtype=float)
    for k in range(int(n_palette)):
        out[:, k] = (flat == k).mean(axis=1)
    return out


def top_gap(frac: np.ndarray) -> np.ndarray:
    """Lead of the front-runner over the runner-up, per replica.

    The *gap* rather than the leader's share, because the share is bounded above
    by 1 and saturates; the gap is what a race is actually about and it is what
    grows without a ceiling until one competitor is gone.
    """
    srt = np.sort(frac, axis=1)
    return srt[:, -1] - srt[:, -2]


def run_race(n_runs: int, n_palette: int, size: int, steps: int, *,
             lead: float = 0.0, ndim: int = 3, gamma: float = 1.5,
             dt: float = 0.06, noise: float = 0.02, seed: int = 0,
             death_frac: float = 0.95, sample: int = 10) -> dict:
    """Evolve a biased ensemble; follow the gap and record deaths.

    Returns the realised head start at the first sample (``lead_measured``),
    the gap trajectory, per-replica death steps (``-1`` = survived), and the
    survival fraction.
    """
    rng = np.random.default_rng(seed)
    fields = seed_with_lead(n_runs, n_palette, size, rng, lead=lead, ndim=ndim)
    sqdt = np.sqrt(dt)
    death = np.full(int(n_runs), -1, dtype=int)
    times, gaps, maxfracs = [], [], []
    lead_measured = None

    for t in range(1, int(steps) + 1):
        fields = step_batch(fields, gamma=gamma, dt=dt, ndim=ndim)
        if noise:
            fields = fields + noise * sqdt * rng.standard_normal(fields.shape)

        if t % int(sample) == 0 or t == 1:
            frac = sector_fractions(sector_labels_batch(fields), n_palette)
            g = top_gap(frac)
            mx = frac.max(axis=1)
            if lead_measured is None:
                lead_measured = float(mx.mean())
            times.append(t)
            gaps.append(g.copy())
            maxfracs.append(mx.copy())
            newly = (mx > float(death_frac)) & (death < 0)
            death[newly] = t
            if (death >= 0).all():
                break

    return {"death": death,
            "censored": death < 0,
            "survival": float((death < 0).mean()),
            "lead_measured": float(lead_measured if lead_measured else 0.0),
            "times": np.asarray(times),
            "gaps": np.asarray(gaps),          # (n_samples, n_runs)
            "maxfrac": np.asarray(maxfracs),
            "steps": int(steps)}


def growth_rate(times, gaps, *, lo: int = 20, hi: int = 200):
    """Exponential growth rate of the lead, fitted on ``log(gap)`` vs ``t``.

    A **race** — where the advantage compounds — gives ``gap ∝ e^{λt}`` with
    ``λ > 0``, so ``log gap`` is straight in ``t``.  Two alternatives with the
    same correlation but a different mechanism are distinguished here: steady
    **drift** gives ``gap ∝ t`` and **diffusion** gives ``gap ∝ √t``, and
    neither is straight on a log axis.

    Returns ``(lambda, r2_log, r2_linear, r2_sqrt)`` so the three candidate
    laws are compared rather than the preferred one being fitted alone.
    """
    times = np.asarray(times, dtype=float)
    gaps = np.asarray(gaps, dtype=float)
    m = (times >= lo) & (times <= hi)
    if m.sum() < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    t = times[m]
    g = np.nanmean(gaps[m], axis=1)
    g = np.maximum(g, 1e-9)

    def r2(x, y):
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return coef[0], (1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)

    lam, r2_log = r2(t, np.log(g))
    _, r2_lin = r2(t, g)
    _, r2_sqrt = r2(np.sqrt(t), g)
    return float(lam), float(r2_log), float(r2_lin), float(r2_sqrt)
