"""Survival analysis for "sometimes it doesn't end".

Watching a stochastic field and noticing that *occasionally* a run refuses to
die is the single easiest way to fool yourself.  Long-lived runs are the ones
worth screenshotting, so they are the ones you remember; the dozens that faded
out in three seconds leave no trace.  That asymmetry is confirmation bias in
its purest mechanical form, and no amount of watching more runs fixes it.

What fixes it is a design, and the design is standard — it is the same
machinery used for drug trials and component failure, which is why it does not
look eccentric when written down:

1. **Define death in code, before looking.**  Here: the first step at which one
   sector occupies more than ``death_frac`` of the lattice.  A number a
   function returns, not a judgement about whether a picture still looks alive.
2. **Count every run**, including the boring ones.  The ensemble is the unit of
   analysis; the memorable run is not evidence, it is a sample.
3. **Ask whether the survivors are actually special.**  If deaths follow a
   *memoryless* process — constant hazard — then "sometimes it doesn't end" is
   just the tail every exponential distribution has, and there is nothing to
   explain.  A hazard that *falls* with time is the opposite: it says a run
   that gets through the early window is qualitatively different, not lucky.
4. **Predict out of sample.**  If the outcome is written into the early state,
   a classifier fitted on half the runs should predict survival on the other
   half.  Held-out prediction cannot be talked into agreeing with you.
5. **Give the null a fair run.**  Every number above is compared against a
   permutation null — the same pipeline on shuffled labels — so "how big is
   big" is answered by the data rather than by intuition.

If the hazard is flat and the held-out AUC sits at 0.5, the honest conclusion
is that the long-lived runs are unremarkable and the felt pattern was not
there.  That outcome is a result, and this module is written to make it just
as reportable as the other one.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "step_batch",
    "seed_batch",
    "sector_labels_batch",
    "multiplicity_batch",
    "window_features",
    "run_ensemble",
    "relaxation_time",
    "kaplan_meier",
    "split_hazard",
    "spearman_r",
    "roc_auc",
    "feature_aucs",
    "held_out_auc",
    "permutation_p",
    "max_feature_permutation",
    "confirm_feature",
]


# --------------------------------------------------------------------------
# batched dynamics — the repo's vector Allen-Cahn, with a leading replica axis
# --------------------------------------------------------------------------

def _laplacian(fields: np.ndarray, ndim: int) -> np.ndarray:
    """Periodic Laplacian over the trailing ``ndim`` axes."""
    out = -2.0 * ndim * fields
    for ax in range(fields.ndim - ndim, fields.ndim):
        out = out + np.roll(fields, 1, ax) + np.roll(fields, -1, ax)
    return out


def step_batch(fields: np.ndarray, *, diffusion: float = 1.0,
               gamma: float = 1.5, dt: float = 0.06, ndim: int = 3) -> np.ndarray:
    """One Allen-Cahn step for a ``(R, P, *spatial)`` batch of replicas.

    Identical rule to :func:`project_genesis.multiphase.step_multiphase` —
    ``df/dη_a = η_a³ − η_a + 2γ·η_a·(Σ_b η_b² − η_a²)`` — just vectorised over
    a replica axis so an ensemble costs one pass instead of ``R`` passes.
    ``tests/test_survival.py`` pins the two against each other elementwise.
    """
    sum_sq = np.sum(fields * fields, axis=1, keepdims=True)
    lap = _laplacian(fields, ndim)
    df = fields ** 3 - fields + 2.0 * gamma * fields * (sum_sq - fields * fields)
    return fields + dt * (diffusion * lap - df)


def seed_batch(n_runs: int, n_palette: int, size: int, rng, *, ndim: int = 3):
    """``R`` independent random starts, each normalised onto the unit sphere."""
    shape = (int(n_runs), int(n_palette)) + (int(size),) * int(ndim)
    fields = rng.uniform(-0.5, 0.5, shape)
    norm = np.sqrt((fields ** 2).sum(axis=1, keepdims=True)) + 1e-12
    return fields / norm


def sector_labels_batch(fields: np.ndarray) -> np.ndarray:
    """Winning sector per site: ``(R, P, *spatial)`` -> ``(R, *spatial)``."""
    return np.argmax(fields, axis=1)


def multiplicity_batch(labels: np.ndarray, n_palette: int, ndim: int) -> np.ndarray:
    """How many distinct sectors appear in each site's 6-neighbour+self stencil.

    A site where ``m`` sectors meet is a codimension-``(m-1)`` object, so in 3-D
    ``m=2`` is a wall, ``m=3`` a triple line and ``m=4`` a quadruple point.
    """
    axes = tuple(range(labels.ndim - ndim, labels.ndim))
    m = np.zeros(labels.shape, dtype=np.int16)
    for k in range(int(n_palette)):
        eq = labels == k
        near = eq.copy()
        for ax in axes:
            near |= np.roll(eq, 1, ax)
            near |= np.roll(eq, -1, ax)
        m += near
    return m


# --------------------------------------------------------------------------
# per-replica observables
# --------------------------------------------------------------------------

def _sector_fractions(labels: np.ndarray, n_palette: int) -> np.ndarray:
    """``(R, P)`` occupancy fractions."""
    R = labels.shape[0]
    flat = labels.reshape(R, -1)
    out = np.empty((R, int(n_palette)), dtype=float)
    for k in range(int(n_palette)):
        out[:, k] = (flat == k).mean(axis=1)
    return out


def window_features(fields: np.ndarray, n_palette: int, ndim: int) -> dict:
    """The early-time state, reduced to five numbers per replica.

    These are the *candidate predictors*: if the eventual outcome is already
    written into the field this early, one of them should carry it.

    - ``ell``       — domain scale, volume per unit wall area;
    - ``maxfrac``   — largest sector occupancy (how lopsided the palette is);
    - ``imbalance`` — spread of sector occupancies;
    - ``junction``  — fraction of sites where three or more sectors meet;
    - ``grad``      — mean squared gradient, the distinction the field carries.
    """
    labels = sector_labels_batch(fields)
    R = labels.shape[0]
    n_sites = int(np.prod(labels.shape[1:]))

    frac = _sector_fractions(labels, n_palette)
    m = multiplicity_batch(labels, n_palette, ndim)
    wall = (m >= 2).reshape(R, -1).sum(axis=1)
    junction = (m >= 3).reshape(R, -1).mean(axis=1)

    grad = np.zeros(R, dtype=float)
    for ax in range(fields.ndim - ndim, fields.ndim):
        d = np.roll(fields, -1, ax) - fields
        grad += (d * d).sum(axis=1).reshape(R, -1).mean(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ell = np.where(wall > 0, n_sites / np.maximum(wall, 1), float(n_sites))

    return {"ell": ell,
            "maxfrac": frac.max(axis=1),
            "imbalance": frac.std(axis=1),
            "junction": junction,
            "grad": grad}


FEATURE_NAMES = ("ell", "maxfrac", "imbalance", "junction", "grad")


# --------------------------------------------------------------------------
# the ensemble run
# --------------------------------------------------------------------------

def relaxation_time(amplitude, frac: float = 0.95) -> int:
    """Steps for the field amplitude to reach ``frac`` of its plateau.

    The simulation has no seconds, so a step count is meaningless on its own.
    This is the system's fastest intrinsic clock — how long the field takes to
    climb into its own wells from a random start — and every other time here
    is worth quoting as a multiple of it.
    """
    a = np.asarray(amplitude, dtype=float)
    if a.size == 0:
        return 0
    plateau = float(a[-max(1, a.size // 8):].mean())
    if plateau <= 0:
        return 0
    reached = np.nonzero(a >= frac * plateau)[0]
    return int(reached[0]) + 1 if reached.size else int(a.size)


def run_ensemble(n_runs: int, n_palette: int, size: int, steps: int, *,
                 ndim: int = 3, gamma: float = 1.5, dt: float = 0.06,
                 noise: float = 0.02, seed: int = 0, early: int = 60,
                 death_frac: float = 0.95, sample: int = 10,
                 capture=None, track_amplitude: bool = False,
                 progress=None) -> dict:
    """Evolve ``n_runs`` replicas together; record when each one dies.

    Death is checked every ``sample`` steps and defined *before any result is
    seen*: the first step at which a single sector holds more than
    ``death_frac`` of the lattice.  Runs still alive at ``steps`` are recorded
    as right-censored (``death = -1``) rather than dropped — discarding them is
    exactly the bias this module exists to avoid.

    Features are captured once, at step ``early``, so they cannot encode
    anything about the outcome except through the dynamics.
    """
    rng = np.random.default_rng(seed)
    fields = seed_batch(n_runs, n_palette, size, rng, ndim=ndim)
    sqdt = np.sqrt(dt)

    death = np.full(int(n_runs), -1, dtype=int)
    features = None
    trace_t, trace_alive = [], []
    capture_steps = sorted({int(c) for c in capture}) if capture else []
    captured, amplitude = {}, []

    for t in range(1, int(steps) + 1):
        fields = step_batch(fields, gamma=gamma, dt=dt, ndim=ndim)
        if noise:
            fields = fields + noise * sqdt * rng.standard_normal(fields.shape)

        if track_amplitude:
            amplitude.append(float(np.sqrt((fields ** 2).sum(axis=1)).mean()))

        if t == int(early):
            features = window_features(fields, n_palette, ndim)
        if capture_steps and t in capture_steps:
            captured[t] = window_features(fields, n_palette, ndim)

        if t % int(sample) == 0 or t == int(steps):
            labels = sector_labels_batch(fields)
            frac = _sector_fractions(labels, n_palette)
            dead_now = frac.max(axis=1) > float(death_frac)
            newly = dead_now & (death < 0)
            death[newly] = t
            trace_t.append(t)
            trace_alive.append(int((death < 0).sum()))
            if progress is not None:
                progress(t, int((death < 0).sum()))
            # Only stop early when nothing else still needs collecting.
            if (death >= 0).all() and not track_amplitude \
                    and (not capture_steps or t >= capture_steps[-1]):
                break

    if features is None:                      # early > steps
        features = window_features(fields, n_palette, ndim)

    return {"death": death,
            "censored": death < 0,
            "steps": int(steps),
            "features": features,
            "captured": captured,
            "amplitude": np.asarray(amplitude),
            "trace_t": np.asarray(trace_t),
            "trace_alive": np.asarray(trace_alive)}


# --------------------------------------------------------------------------
# survival statistics
# --------------------------------------------------------------------------

def kaplan_meier(death: np.ndarray, censored: np.ndarray, horizon: int):
    """Kaplan-Meier survival curve, honouring right-censoring.

    Returns ``(times, survival)``.  Censored runs stay in the risk set until
    the horizon and then leave without an event, which is the whole point:
    a run that outlived the experiment is evidence *for* survival, not missing
    data.
    """
    death = np.asarray(death)
    censored = np.asarray(censored, dtype=bool)
    events = np.sort(np.unique(death[~censored]))
    n = len(death)
    times, surv = [0], [1.0]
    s = 1.0
    for t in events:
        at_risk = int(((death >= t) & ~censored).sum() + (censored.sum() if t <= horizon else 0))
        d = int(((death == t) & ~censored).sum())
        if at_risk > 0:
            s *= (1.0 - d / at_risk)
        times.append(int(t))
        surv.append(s)
    return np.asarray(times), np.asarray(surv)


def split_hazard(death: np.ndarray, censored: np.ndarray, cut: int, horizon: int):
    """Hazard rate before and after ``cut`` (deaths per run per step at risk).

    A **memoryless** process has equal hazard in both windows: surviving the
    early window buys you nothing, and long runs are just the tail.  A hazard
    that falls means survivors are genuinely a different population.
    """
    death = np.asarray(death)
    censored = np.asarray(censored, dtype=bool)

    def window(lo, hi):
        d = int((~censored & (death > lo) & (death <= hi)).sum())
        # exposure: steps each run actually spent at risk inside the window
        end = np.where(censored, horizon, death)
        expo = np.clip(end - lo, 0, hi - lo).sum()
        return d, float(expo), (d / expo if expo > 0 else np.nan)

    d1, e1, h1 = window(0, cut)
    d2, e2, h2 = window(cut, horizon)
    ratio = h2 / h1 if (h1 and np.isfinite(h1) and h1 > 0) else np.nan
    return {"early_deaths": d1, "early_exposure": e1, "early_hazard": h1,
            "late_deaths": d2, "late_exposure": e2, "late_hazard": h2,
            "ratio": ratio}


def bootstrap_hazard_ratio(death, censored, cut, horizon, *, n_boot=2000, seed=0):
    """Bootstrap CI for the late/early hazard ratio, resampling *runs*."""
    rng = np.random.default_rng(seed)
    death = np.asarray(death)
    censored = np.asarray(censored, dtype=bool)
    n = len(death)
    out = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        r = split_hazard(death[idx], censored[idx], cut, horizon)["ratio"]
        if np.isfinite(r):
            out.append(r)
    if not out:
        return (np.nan, np.nan)
    out = np.sort(np.asarray(out))
    return float(out[int(0.025 * len(out))]), float(out[int(0.975 * len(out)) - 1])


# --------------------------------------------------------------------------
# does the early state predict the outcome?
# --------------------------------------------------------------------------

def spearman_r(x, y) -> float:
    """Rank correlation, ties averaged.  No scipy needed and no assumptions."""
    def rank(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def roc_auc(scores, labels) -> float:
    """Area under the ROC curve — P(score of a positive > score of a negative)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    comp = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(comp / (len(pos) * len(neg)))


def _fit_linear(X, y, ridge: float = 1.0):
    """Ridge discriminant on standardised features.

    A plain least-squares fit of five correlated features on a few dozen
    training points overfits badly — it can score *worse* held out than the
    best single feature, which is a property of the estimator and not of the
    data.  The ridge term is fixed in advance rather than tuned, so it cannot
    be turned toward a preferred answer.
    """
    A = np.hstack([np.ones((len(X), 1)), X])
    n_col = A.shape[1]
    reg = float(ridge) * np.eye(n_col)
    reg[0, 0] = 0.0                       # never penalise the intercept
    coef = np.linalg.solve(A.T @ A + reg, A.T @ y.astype(float))
    return coef


def held_out_auc(features: dict, labels, *, names=FEATURE_NAMES, seed=0):
    """Fit on half the runs, score the other half.

    Returns the held-out AUC of the combined linear score and of each feature
    on its own.  Per-feature numbers are reported for *all* features, so
    picking the winner after the fact is visible rather than hidden.
    """
    labels = np.asarray(labels).astype(bool)
    X = np.column_stack([np.asarray(features[n], dtype=float) for n in names])
    mu, sd = X.mean(axis=0), X.std(axis=0)
    X = (X - mu) / np.where(sd > 0, sd, 1.0)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    half = len(labels) // 2
    tr, te = idx[:half], idx[half:]

    coef = _fit_linear(X[tr], labels[tr])
    score = coef[0] + X[te] @ coef[1:]
    combined = roc_auc(score, labels[te])

    per = {}
    for j, n in enumerate(names):
        a = roc_auc(X[te][:, j], labels[te])
        per[n] = max(a, 1.0 - a)          # direction-free: |discrimination|
    return combined, per


def permutation_p(features: dict, labels, *, names=FEATURE_NAMES,
                  n_perm: int = 1000, seed: int = 0):
    """Null distribution for the held-out AUC: the same pipeline, shuffled labels.

    This is what answers "is 0.72 impressive?" without appealing to intuition.
    """
    labels = np.asarray(labels).astype(bool)
    observed, _ = held_out_auc(features, labels, names=names, seed=seed)
    rng = np.random.default_rng(seed + 1)
    null = np.empty(int(n_perm))
    for i in range(int(n_perm)):
        shuffled = rng.permutation(labels)
        null[i], _ = held_out_auc(features, shuffled, names=names, seed=seed)
    p = float((null >= observed).sum() + 1) / (int(n_perm) + 1)
    return observed, float(null.mean()), float(np.percentile(null, 95)), p


def feature_aucs(features: dict, labels, *, names=FEATURE_NAMES) -> dict:
    """Direction-free AUC of each feature over the **whole** ensemble.

    No train/test split, deliberately: a single unfitted feature has nothing to
    overfit, so splitting would only throw away half the data.  The split in
    :func:`held_out_auc` exists to protect the *fitted* combination, and mixing
    the two conventions would report one feature at two different numbers.
    """
    labels = np.asarray(labels).astype(bool)
    out = {}
    for n in names:
        a = roc_auc(np.asarray(features[n], dtype=float), labels)
        out[n] = max(a, 1.0 - a)
    return out


def max_feature_permutation(features: dict, labels, *, names=FEATURE_NAMES,
                            n_perm: int = 1000, seed: int = 0):
    """Null for *the best of several* features — the look-elsewhere correction.

    Reporting the largest of five AUCs against a null built for one of them
    inflates significance: with five tries, some feature will look good by
    chance.  The honest null is the distribution of the **maximum** AUC over
    the same five features under shuffled labels.
    """
    labels = np.asarray(labels).astype(bool)

    def best(lab):
        d = feature_aucs(features, lab, names=names)
        vals = [d[n] for n in names]
        return float(np.max(vals)), names[int(np.argmax(vals))]

    observed, winner = best(labels)
    rng = np.random.default_rng(seed + 2)
    null = np.array([best(rng.permutation(labels))[0] for _ in range(int(n_perm))])
    p = float((null >= observed).sum() + 1) / (int(n_perm) + 1)
    return observed, winner, float(np.percentile(null, 95)), p


def confirm_feature(values, labels, *, n_perm: int = 2000, seed: int = 0):
    """Test **one pre-specified** feature on a fresh ensemble.

    With the feature fixed in advance there is nothing to select over, so the
    whole ensemble is used and the AUC needs no train/test split — the split
    exists only to protect against fitting, and nothing is fitted here.  This
    is the confirmatory half of an explore-then-confirm design: whatever the
    first ensemble suggested, this one either reproduces it on new data or
    does not.
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels).astype(bool)
    a = roc_auc(values, labels)
    observed = max(a, 1.0 - a)
    rng = np.random.default_rng(seed + 3)
    null = np.empty(int(n_perm))
    for i in range(int(n_perm)):
        b = roc_auc(values, rng.permutation(labels))
        null[i] = max(b, 1.0 - b)
    p = float((null >= observed).sum() + 1) / (int(n_perm) + 1)
    return observed, float(np.percentile(null, 95)), p
