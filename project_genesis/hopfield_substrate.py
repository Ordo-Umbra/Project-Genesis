"""A second substrate for the criticality claim: the thermal Hopfield network.

The sector-field programme measured a specific, deep result (`n3_s_landscape`,
`n3_kappa_criticality`, `n3_s_criticality`): **capacity-bound S-climbing
systems are pushed toward criticality**.  Scarcity taxes integration (the
κ-gated term) but not distinction, so as the capacity budget binds, the global
S-optimum performs a *level crossing* — from the deep-ordered phase (where
coherence funds S) to the critical point (where the ΔC peak funds it).

But every one of those measurements lives on the same substrate: a lattice
sector field.  If the claim is a law of capacity-bound recursion rather than a
property of that lattice, it must survive **substrate transplant**.  This
module provides the transplant target, chosen to be structurally as *unlike*
the sector field as a cheap, well-understood system can be:

    the thermal **Hopfield network** — no lattice geometry (all-to-all learned
    couplings), no sectors (attractors are *learned memories*), no gauge
    freedom, and an independent, famous order-disorder transition: retrieval
    breaks down at a loading- and temperature-dependent critical point.

The dictionary (each entry the honest analogue, stated once):

- **order parameter**  `m` — the best retrieval overlap
  `m = max_μ |⟨pattern_μ · state⟩|/N`; high in the retrieval phase, falling
  through the transition (the analogue of the sector order parameter).
- **integration ΔI** — `m` averaged over a dynamics window: how coherently
  the state binds to *some* stored structure.
- **distinction ΔC** — the *diversity of structured states visited*: the
  entropy of which attractor the state occupies, counted only while the state
  is structured (`m > threshold`), scaled by the structured fraction.  Frozen
  retrieval visits one attractor (ΔC ≈ 0); pure noise visits none that are
  structured (ΔC → 0); the neighbourhood of the transition — where the state
  hops between still-recognisable memories — is where distinction peaks.
- **capacity κ** — the same dynamical law as the field engine, in steady
  state: consumed by distinction load, regenerating at rate `r`:
  `κ = r / (r + c·load)` with `load = ΔC` (capacity is consumed by making
  distinctions — the engine's semantics, transplanted verbatim).
- **the functional** — `S = ΔC + κ·w·ΔI`, exactly the form the sector-field
  experiments maximise.

The claim under test transplants to: *sweeping temperature `T` across the
retrieval transition, `argmax_T S` sits in the deep-retrieval phase while
capacity is abundant (small consumption `c`) and relocates to the critical
neighbourhood as `c` rises — a level crossing, not a drift.*  Nothing in this
module makes that true by construction; the experiment
(`n3_criticality_transplant.py`) measures whether it happens.

The delta-form trajectory taxonomy of the S-compass (`s_engine.py` in the
Universal-Recursion-Principle repo) is mirrored here (`trajectory_label`), so
the same classification the compass applies to LLM sessions can be read on
this substrate's (ΔC, ΔI) steps — one taxonomy across field, network, and
agent traces.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "hebbian_weights",
    "glauber_sweep",
    "best_overlap",
    "window_metrics",
    "steady_state_kappa",
    "s_functional",
    "trajectory_label",
]


def hebbian_weights(patterns: np.ndarray) -> np.ndarray:
    """Hebbian couplings ``J = (1/N) Σ_μ ξ^μ ξ^μᵀ`` with zero diagonal.

    ``patterns`` is ``(P, N)`` of ±1.  All-to-all, symmetric — no lattice
    geometry anywhere in the substrate.
    """
    p = np.asarray(patterns, dtype=float)
    n = p.shape[1]
    j = p.T @ p / n
    np.fill_diagonal(j, 0.0)
    return j


def glauber_sweep(state: np.ndarray, weights: np.ndarray, temperature: float,
                  rng: np.random.Generator) -> np.ndarray:
    """One asynchronous Glauber sweep (N single-spin updates) at ``temperature``.

    Each update flips spin ``i`` to +1 with probability
    ``σ(2h_i/T)`` where ``h_i = Σ_j J_ij s_j`` — detailed balance w.r.t. the
    Hopfield energy.  ``temperature → 0`` reduces to deterministic descent.
    """
    s = state.copy()
    n = s.size
    order = rng.permutation(n)
    t = max(temperature, 1e-12)
    for i in order:
        h = float(weights[i] @ s)
        p_up = 1.0 / (1.0 + np.exp(-2.0 * h / t))
        s[i] = 1 if rng.random() < p_up else -1
    return s


def best_overlap(state: np.ndarray, patterns: np.ndarray) -> tuple[float, int]:
    """Best absolute retrieval overlap and which memory carries it.

    Returns ``(m, index)`` with ``m = max_μ |ξ^μ·s|/N ∈ [0, 1]``.
    """
    m = patterns @ state / state.size
    idx = int(np.argmax(np.abs(m)))
    return float(np.abs(m[idx])), idx


def window_metrics(states_overlaps: list[tuple[float, int]],
                   n_patterns: int,
                   structured_threshold: float = 0.3) -> dict:
    """ΔC, ΔI, and the structured fraction from a window of dynamics.

    ``states_overlaps`` is a list of ``(m, attractor_index)`` samples.

    - ``delta_i`` — mean best overlap (the integration reading).
    - ``delta_c`` — normalised entropy of *which* attractor the structured
      samples occupy, times the structured fraction: zero when frozen in one
      memory, zero when nothing structured is visited, maximal when the state
      hops among many still-recognisable memories.
    - ``structured_fraction`` — share of samples with ``m > threshold``.
    """
    ms = np.array([m for m, _ in states_overlaps])
    delta_i = float(ms.mean()) if ms.size else 0.0
    structured = [(m, idx) for m, idx in states_overlaps
                  if m > structured_threshold]
    frac = len(structured) / max(1, len(states_overlaps))
    if not structured or n_patterns < 2:
        return {"delta_c": 0.0, "delta_i": delta_i,
                "structured_fraction": frac}
    counts = np.bincount([idx for _, idx in structured], minlength=n_patterns)
    probs = counts[counts > 0] / counts.sum()
    entropy = float(-(probs * np.log(probs)).sum())
    max_entropy = np.log(n_patterns)
    delta_c = float(entropy / max_entropy * frac)
    return {"delta_c": delta_c, "delta_i": delta_i, "structured_fraction": frac}


def steady_state_kappa(load: float, consumption: float, recovery: float,
                       kappa0: float = 1.0) -> float:
    """The engine's capacity law in steady state: ``κ = r·κ₀ / (r + c·load)``.

    The fixed point of ``∂κ/∂t = r(κ₀ − κ) − c·load·κ`` — the same equation
    the field engine integrates (`multiphase.py` / `capacity_dynamics.py`),
    transplanted with ``load`` = distinction activity.
    """
    return float(recovery * kappa0 / (recovery + consumption * load))


def s_functional(delta_c: float, delta_i: float, kappa: float,
                 weight: float) -> float:
    """``S = ΔC + κ·w·ΔI`` — the exact form the sector-field experiments maximise."""
    return float(delta_c + kappa * weight * delta_i)


def trajectory_label(delta_c_step: float, delta_i_step: float,
                     deadband: float = 1e-3) -> str:
    """The S-compass delta-form trajectory taxonomy, mirrored for this substrate.

    Matches ``s_compass.s_engine`` in the Universal-Recursion-Principle repo:
    ``expanding`` (ΔC↑, ΔI↑), ``consolidating`` (ΔC↓, ΔI↑), ``diverging``
    (ΔC↑, ΔI↓ — the hallucination trajectory), ``contracting`` (both ↓),
    ``steady`` (inside the deadband).  One taxonomy across LLM sessions,
    lattice fields, and attractor networks.
    """
    up_c = delta_c_step > deadband
    down_c = delta_c_step < -deadband
    up_i = delta_i_step > deadband
    down_i = delta_i_step < -deadband
    if not (up_c or down_c) and not (up_i or down_i):
        return "steady"
    if up_c and down_i:
        return "diverging"
    if down_c and up_i:
        return "consolidating"
    if down_c and down_i:
        return "contracting"
    if up_c and up_i:
        return "expanding"
    # one component in the deadband: classify by the live one
    if up_c or up_i:
        return "expanding"
    return "contracting"
