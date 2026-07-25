"""The scaling dimension of the generative gap: distinction, integration, area laws.

The programme's core asymmetry — *distinction is cheap, integration is
expensive* — has been measured many ways (the capacity-separation cliff, the
κ-trough at criticality, the memory arc).  Every one of those measures it as a
**magnitude**.  This module measures it as a **scaling dimension**: how the two
terms grow with the size of the region you look at.

For nested regions ``R(ℓ)`` of half-width ``ℓ`` in ``d`` dimensions:

- **Distinction content** ``C(ℓ) = Σ_{x∈R} |∇Ψ(x)|²`` — the gradient/wall energy
  the region contains.  If distinction fills space, ``C ~ ℓ^d`` (a *volume* law).
- **Integration content** ``I(ℓ)`` — the coherence shared between the region and
  its complement.  If integration is boundary-limited, ``I ~ ℓ^{d−1}`` (an
  *area* law) — the holographic scaling the URP corpus asserts as the geometric
  consequence of the gap.

``I(ℓ)`` is the classical (Gaussian) mutual-information proxy: for jointly
Gaussian variables ``MI ≈ ρ²/2``, so

    I(ℓ)  =  Σ_{x∈R, y∉R} ρ(x−y)² ,     ρ(r) = G(r)/G(0),

with ``G`` the field's **connected** correlation function.  Computed as one FFT
convolution against the region indicator (:func:`cross_boundary_mi`).

**The instrument must be calibrated before it is believed.**  Two traps, both
found the hard way and both handled here:

1. *Signed cancellation.*  A mean-subtracted ``G`` sums to zero over the torus,
   so the un-squared coupling cancels to numerical noise.  Squaring (the
   Gaussian-MI form) removes this — and is the principled estimator anyway.
2. *The correlation noise floor.*  A finite-sample ``ρ²`` has a positive floor
   at large ``r``; convolved against a region it contributes a spurious
   **volume** term that inflates the exponent.  On a synthetic field whose true
   answer is exactly 1.0, a single realisation reads **1.77**.  The fix is an
   ensemble-averaged ``G`` plus explicit floor subtraction
   (:func:`mi_kernel`) — and, crucially, quoting the residual bias measured on
   a control whose answer is known (:func:`gaussian_control_field`).

Use :func:`gaussian_control_field` at the *same* ensemble size as the real
measurement, recover its known area law, and quote the difference as the
instrument's bias.  Nothing here should be trusted without that step.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gaussian_control_field",
    "ensemble_correlation",
    "radial_profile",
    "correlation_length",
    "mi_kernel",
    "region_indicator",
    "cross_boundary_mi",
    "region_content",
    "scaling_exponent",
]


def gaussian_control_field(n: int, xi: float, rng) -> np.ndarray:
    """A Gaussian random field with correlation length ``xi`` — the control.

    Its mutual information between a region and its complement obeys an
    **exact area law** (exponent ``d−1``) once ``ℓ ≫ xi``, so it is ground
    truth for calibrating the estimator.
    """
    w = rng.standard_normal((n, n))
    y, x = np.indices((n, n))
    dx = np.minimum(x, n - x)
    dy = np.minimum(y, n - y)
    ker = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * xi ** 2))
    f = np.fft.irfft2(np.fft.rfft2(w) * np.fft.rfft2(ker), s=(n, n))
    return (f - f.mean()) / f.std()


def ensemble_correlation(components) -> np.ndarray:
    """Connected correlation ``G(r) = ⟨δΨ(x)·δΨ(x+r)⟩``, averaged over an
    ensemble.

    ``components`` is any iterable of 2-D arrays (field components and/or
    independent realisations).  Averaging is what beats down the noise floor
    that otherwise fakes a volume law.
    """
    comps = list(components)
    n = comps[0].shape[0]
    acc = np.zeros((n, n))
    for c in comps:
        d = np.asarray(c, dtype=float)
        d = d - d.mean()
        f = np.fft.rfft2(d)
        acc += np.fft.irfft2(f * np.conj(f), s=(n, n)).real
    return acc / len(comps) / (n * n)


def radial_profile(a: np.ndarray) -> np.ndarray:
    """Radial average of a periodic 2-D array (index = integer radius)."""
    n = a.shape[0]
    y, x = np.indices((n, n))
    dx = np.minimum(x, n - x)
    dy = np.minimum(y, n - y)
    r = np.hypot(dx, dy).astype(int)
    return (np.bincount(r.ravel(), a.ravel())
            / np.maximum(np.bincount(r.ravel()), 1))


def correlation_length(g: np.ndarray) -> float:
    """The ``1/e`` radius of the connected correlation — the coherence length."""
    prof = radial_profile(g)
    n = g.shape[0]
    g0 = prof[0]
    for i, v in enumerate(prof[:n // 2]):
        if v < g0 / np.e:
            return float(i)
    return float(n // 2)


def mi_kernel(g: np.ndarray, floor_lo: float = 0.25,
              floor_hi: float = 0.5) -> tuple:
    """The Gaussian-MI kernel ``ρ²`` with its noise floor removed.

    ``ρ = G/G(0)``; the floor is the mean of ``ρ²`` over radii in
    ``[floor_lo·n, floor_hi·n]`` (where physical correlation is dead).
    Returns ``(kernel, floor)``.
    """
    n = g.shape[0]
    rho2 = (g / g[0, 0]) ** 2
    prof = radial_profile(rho2)
    floor = float(prof[int(floor_lo * n):int(floor_hi * n)].mean())
    return np.maximum(rho2 - floor, 0.0), floor


def region_indicator(n: int, half: int) -> np.ndarray:
    """Indicator of the centred square region of half-width ``half``."""
    chi = np.zeros((n, n))
    c = n // 2
    chi[c - half:c + half, c - half:c + half] = 1.0
    return chi


def cross_boundary_mi(kernel: np.ndarray, half: int) -> float:
    """``I(ℓ) = Σ_{x∈R, y∉R} kernel(x−y)`` — region↔complement coherence.

    One FFT convolution: ``⟨χ_R , kernel ⊛ (1−χ_R)⟩``.
    """
    n = kernel.shape[0]
    chi = region_indicator(n, half)
    conv = np.fft.irfft2(np.fft.rfft2(kernel) * np.fft.rfft2(1.0 - chi),
                         s=(n, n)).real
    return float((chi * conv).sum())


def region_content(density: np.ndarray, half: int) -> float:
    """``C(ℓ) = Σ_{x∈R} density(x)`` — e.g. the gradient (distinction) energy."""
    return float((density * region_indicator(density.shape[0], half)).sum())


def scaling_exponent(halves, values) -> float:
    """Log-log slope ``n`` in ``value ~ ℓ^n`` (``d``=volume, ``d−1``=area)."""
    h = np.log(np.asarray(halves, dtype=float))
    v = np.log(np.maximum(np.asarray(values, dtype=float), 1e-300))
    return float(np.polyfit(h, v, 1)[0])
