"""A third substrate for the criticality claim: the Kuramoto oscillator network.

The programme measured a deep, emergent result on the sector lattice
(`n3_s_landscape`, `n3_kappa_criticality`) and then showed it **survives
transplant** to a thermal Hopfield network (`n3_criticality_transplant.py`,
`hopfield_substrate.py`): *a capacity-bound S-climbing system is pushed to
criticality exactly when maintaining order costs capacity.*  Two substrates is
not universality — the transplant experiment says so in its own honest scope:
"universality wants **more substrates**, not more claims."  This module is the
third substrate, chosen to be structurally as *unlike* the first two as a
cheap, textbook system can be:

    the **Kuramoto model** — a population of phase oscillators
    ``θ_i`` with heterogeneous natural frequencies ``ω_i`` that pull toward a
    common rhythm.  No lattice geometry (mean-field, all-to-all), no learned
    memories (the attractor is *collective synchrony*, not a stored pattern),
    no discrete states at all (the order parameter is continuous), and an
    independent, famous order–disorder transition: coherence switches on above
    a critical coupling / below a critical frequency spread.

The dictionary (each entry the honest analogue, stated once — mirroring
`hopfield_substrate.py` so the *same* functional reads on all three):

- **order parameter** ``r`` — the Kuramoto coherence
  ``r·e^{iψ} = (1/N)Σ_j e^{iθ_j}``; ``r ≈ 1`` fully synchronised, ``r ≈ 0``
  incoherent, falling through the transition (the analogue of the sector
  order parameter and the Hopfield retrieval overlap).
- **integration ΔI** — ``r`` averaged over a dynamics window: how coherently
  the population binds into a single collective rhythm.
- **distinction ΔC** — the *diversity of frequency clusters* among the
  oscillators that are participating in structure, scaled by the coherent
  fraction.  Deep synchrony is one cluster at the collective frequency
  (ΔC ≈ 0); full incoherence has many clusters but none coherent
  (coherent-fraction → 0, so ΔC → 0); the neighbourhood of the transition —
  a locked core plus a spread of drifters — is where distinction peaks.  The
  direct analogue of the Hopfield "which attractor is visited" entropy, with
  *effective frequency* standing in for *attractor identity*.
- **capacity κ** — the same dynamical law as the field engine and the Hopfield
  transplant, in steady state: consumed by distinction load, regenerating at
  rate ``r_κ``: ``κ = r_κ/(r_κ + c·load)`` with ``load`` the distinction
  activity — the engine's semantics, transplanted verbatim.
- **the functional** — ``S = ΔC + κ·w·ΔI``, exactly the form the sector-field
  and Hopfield experiments maximise.

The capacity law (:func:`steady_state_kappa`), the functional
(:func:`s_functional`), and the S-compass trajectory taxonomy
(:func:`trajectory_label`) are **imported unchanged from
`hopfield_substrate`** — not re-implemented — so "one law across substrates" is
a fact about the code, not a claim in prose.  This module adds only what is
specific to the oscillator population: the dynamics, the order parameter, the
effective frequencies, and the ΔC/ΔI/activity readings.

The claim under test transplants to: *sweeping the disorder ``γ`` (natural
frequency spread) across the synchronisation transition, ``argmax_γ S`` sits in
the deep-synchronised phase while capacity is abundant and relocates to the
critical neighbourhood as consumption rises — a level crossing — but only when
holding synchrony actually costs capacity.*  As on the Hopfield network, the
deterministic locked state is (in the co-rotating frame) essentially free, so
the mechanism predicts **no** relocation there; add phase noise the synchrony
must continuously repair and the cost — and the relocation — return.  Nothing
here makes that true by construction; `experiments/n3_kuramoto_transplant.py`
measures whether it happens.
"""

from __future__ import annotations

import numpy as np

# One law across substrates: the capacity dynamics, the functional, and the
# S-compass taxonomy are the Hopfield module's, imported verbatim.
from .hopfield_substrate import (
    steady_state_kappa,
    s_functional,
    trajectory_label,
)

__all__ = [
    "natural_frequencies",
    "order_parameter",
    "drift_field",
    "kuramoto_step",
    "simulate",
    "window_metrics",
    "steady_state_kappa",
    "s_functional",
    "trajectory_label",
]


def natural_frequencies(n: int, spread: float,
                        rng: np.random.Generator,
                        distribution: str = "lorentzian") -> np.ndarray:
    """``N`` natural frequencies centred at zero with scale ``spread``.

    Centred at zero so the collective frequency of a synchronised population is
    ~0 and "entrained" means "effective frequency near zero" — the reading the
    ΔC/coherence metrics use.  ``distribution`` is ``"lorentzian"`` (the
    Cauchy law of Kuramoto's exact solution, ``γ`` its half-width) or
    ``"gaussian"`` (``γ`` its standard deviation).
    """
    if spread <= 0.0:
        return np.zeros(n)
    if distribution == "gaussian":
        return spread * rng.standard_normal(n)
    if distribution == "lorentzian":
        # inverse-CDF sampling of a zero-centred Cauchy with half-width `spread`
        u = rng.uniform(-0.5 + 1e-9, 0.5 - 1e-9, size=n)
        return spread * np.tan(np.pi * u)
    raise ValueError(f"unknown distribution {distribution!r}")


def order_parameter(phases: np.ndarray) -> tuple[float, float]:
    """The Kuramoto coherence ``(r, ψ)`` with ``r·e^{iψ} = ⟨e^{iθ}⟩``.

    ``r ∈ [0, 1]``: 1 fully phase-locked, 0 uniformly spread.
    """
    z = np.exp(1j * np.asarray(phases)).mean()
    return float(np.abs(z)), float(np.angle(z))


def drift_field(phases: np.ndarray, omega: np.ndarray,
                coupling: float) -> np.ndarray:
    """The deterministic phase velocity ``ω_i + K·r·sin(ψ − θ_i)``.

    Split out of :func:`kuramoto_step` so the *system's own* motion can be read
    without the injected noise mixed into it — see ``repair_rate`` in
    :func:`simulate` for why that distinction turned out to matter.
    """
    r, psi = order_parameter(phases)
    return omega + coupling * r * np.sin(psi - np.asarray(phases))


def kuramoto_step(phases: np.ndarray, omega: np.ndarray, coupling: float,
                  dt: float, noise: float,
                  rng: np.random.Generator) -> np.ndarray:
    """One Euler(–Maruyama) step of the mean-field Kuramoto dynamics.

    ``dθ_i = [ω_i + K·r·sin(ψ − θ_i)]·dt + noise·√dt·ξ_i``.

    Uses the exact mean-field identity ``(1/N)Σ_j sin(θ_j − θ_i) =
    r·sin(ψ − θ_i)`` so the coupling is ``O(N)`` rather than ``O(N²)``.  With
    ``noise = 0`` this is the deterministic model; ``noise > 0`` is the driven
    condition — a rhythm the population must actively repair.
    """
    increment = drift_field(phases, omega, coupling) * dt
    if noise > 0.0:
        increment = increment + noise * np.sqrt(dt) * rng.standard_normal(
            phases.size)
    return phases + increment


def simulate(phases: np.ndarray, omega: np.ndarray, coupling: float,
             dt: float, steps: int, noise: float,
             rng: np.random.Generator,
             burn_in: int = 0) -> dict:
    """Evolve the population and read the order parameter and effective drift.

    Returns a dict with

    - ``r_samples`` — the order parameter at every post-burn-in step,
    - ``eff_freq`` — each oscillator's mean phase velocity over the measured
      window ``⟨dθ_i/dt⟩`` (the "effective frequency" that clusters read),
    - ``activity`` — the mean, over the window, of the *per-step* population
      spread of *co-rotating* phase increments ``std_i(Δθ_i − ⟨Δθ⟩)``: how much
      the population reconfigures beyond rigid collective rotation each step —
      the oscillator analogue of the Hopfield flip rate, and in the same O(0.1)
      range.  In a deterministic locked state this is ~0 (everyone rotates
      together); under noise it is the repair activity the ordered rhythm pays.

      **This reading is contaminated by the drive, and `n3_crossing_prediction`
      measured by how much.**  The injected noise contributes exactly ``σ√dt``
      to the same spread, and ``activity/(σ√dt)`` at the ordered end converges
      to ``1.004`` as ``dt → 0``: essentially all of it is the perturbation
      being counted back as though it were the population's response to the
      perturbation.  It is kept unchanged because three published results are
      stated in it.
    - ``repair_rate`` — the same idea with the drive removed and the
      integrator's step divided out: the mean over the window of
      ``std_i(v_i − ⟨v⟩)`` where ``v_i = ω_i + K·r·sin(ψ − θ_i)`` is the
      *deterministic* phase velocity.  Three properties the raw ``activity``
      does not have: it contains no injected noise, it is a **rate** rather
      than a per-step displacement so it does not carry ``dt`` at all, and it
      still vanishes in the deterministic locked state (where every oscillator
      shares one collective velocity), which is the toggle the transplant
      design needs.  What it measures is the restoring work the coupling
      actually does against the perturbation.
    - ``phases`` — the final phases.
    """
    phases = np.array(phases, dtype=float)
    burn_in = min(burn_in, steps)
    for _ in range(burn_in):
        phases = kuramoto_step(phases, omega, coupling, dt, noise, rng)

    r_samples: list[float] = []
    total_phase = np.zeros(phases.size)
    activity_acc = 0.0
    repair_acc = 0.0
    n_measure = max(1, steps - burn_in)
    for _ in range(n_measure):
        prev = phases
        # the system's own velocity, read BEFORE the step so it is the drift
        # that produced this increment and carries none of its noise
        v = drift_field(prev, omega, coupling)
        repair_acc += float(np.std(v - v.mean()))
        phases = kuramoto_step(phases, omega, coupling, dt, noise, rng)
        increment = phases - prev
        total_phase += increment
        # per-step spread of increments about the rigid-rotation mean
        # (co-rotating frame) — the analogue of the Hopfield flip rate
        activity_acc += float(np.std(increment - increment.mean()))
        r_samples.append(order_parameter(phases)[0])

    eff_freq = total_phase / (n_measure * dt)
    return {
        "r_samples": np.array(r_samples),
        "eff_freq": eff_freq,
        "activity": activity_acc / n_measure,
        "repair_rate": repair_acc / n_measure,
        "phases": phases,
    }


def window_metrics(sim: dict, freq_tol: float = 0.3,
                   bin_width: float = 0.4, freq_range: float = 3.0) -> dict:
    """ΔC, ΔI, and the coherent fraction from a :func:`simulate` result.

    - ``delta_i`` — mean order parameter over the window (the integration
      reading).
    - ``delta_c`` — normalised entropy of *which effective-frequency cluster*
      the oscillators occupy, times the coherent fraction: zero when all lock
      to one collective frequency (deep synchrony), zero when nothing is
      coherent (incoherence), maximal when a locked core coexists with a spread
      of drifters (the critical neighbourhood).
    - ``coherent_fraction`` — share of oscillators entrained to the collective
      frequency (``|Ω_i| < freq_tol``; frequencies are zero-centred).
    - ``activity`` — passed through from the simulation (the published load
      reading, drive-contaminated; see :func:`simulate`).
    - ``repair_rate`` — passed through from the simulation (the drive-free,
      dt-free load reading).

    The frequency histogram uses **fixed-width bins** on a fixed range, not
    data-driven edges: a tight locked cluster then stays in one bin whatever
    the per-oscillator noise, so a synchronised-but-noisy population reads
    ΔC ≈ 0 (one rhythm) rather than a spurious spread — the property that lets
    the deep-ordered phase be genuinely low-distinction under a repair drive.
    """
    r_samples = np.asarray(sim["r_samples"])
    eff = np.asarray(sim["eff_freq"])
    delta_i = float(r_samples.mean()) if r_samples.size else 0.0

    entrained = np.abs(eff) < freq_tol
    coherent_fraction = float(entrained.mean()) if eff.size else 0.0

    # entropy of the effective-frequency histogram on FIXED bins: how many
    # distinct absolute rhythms the population carries
    edges = np.arange(-freq_range, freq_range + bin_width, bin_width)
    n_bins = len(edges) - 1
    if eff.size < 2 or n_bins < 2:
        delta_c = 0.0
    else:
        counts, _ = np.histogram(np.clip(eff, edges[0], edges[-1]), bins=edges)
        probs = counts[counts > 0] / counts.sum()
        entropy = float(-(probs * np.log(probs)).sum())
        norm_entropy = entropy / np.log(n_bins)
        delta_c = float(norm_entropy * coherent_fraction)

    return {
        "delta_c": delta_c,
        "delta_i": delta_i,
        "coherent_fraction": coherent_fraction,
        "activity": float(sim.get("activity", 0.0)),
        "repair_rate": float(sim.get("repair_rate", 0.0)),
    }
