"""Tests for Monte Carlo confinement behaviour aligned with URP expectations.

These tests do **not** assume precise numerical values for the string tension
or Polyakov loop. Instead they check qualitative behaviours that the
β-sectorisation / emergent SU(3) theory says should be present:

- In a confined regime (strong coupling, modest lattice), ensemble-averaged
  Wilson loops should decay with area and the fitted σ should be positive.
- The Polyakov loop should be near zero in the confined regime.

The goal is to confirm that the Monte Carlo layer is capable of expressing the
confinement signatures the URP derivation attributes to the gauge sector.
"""

from __future__ import annotations

import math

import numpy as np

from project_genesis.gauge_mc import (
    thermalize_and_measure_pure_gauge,
)


RNG = np.random.default_rng(260625)


def _strong_coupling_summary(beta_g: float = 0.8) -> dict:
    """Helper: run a small SU(2) lattice at strong coupling and return summary.

    Parameters are chosen to favour a confined-like regime with visible area
    suppression on modest lattices.
    """
    summary, _ = thermalize_and_measure_pure_gauge(
        size=8,
        n=2,
        beta_g=beta_g,
        rng=RNG,
        ndim=2,
        n_therm=200,
        n_meas=40,
        n_skip=5,
        updater="metropolis",
        loop_sizes=[(1, 1), (2, 2), (3, 3)],
    )
    return summary


class TestConfinementWilsonLoops:
    def test_area_law_sigma_positive(self):
        """At strong coupling, ensemble-averaged loops should obey an area law
        with σ > 0.
        """
        summary = _strong_coupling_summary(beta_g=0.8)
        loops = summary["loop_averages"]
        w11 = loops["W_1_1"]
        w33 = loops["W_3_3"]

        # Basic sanity: values in [-1,1]
        assert -1.0 <= w11 <= 1.0
        assert -1.0 <= w33 <= 1.0

        # Area suppression: |W(3,3)| <= |W(1,1)|
        assert abs(w33) <= abs(w11), (
            f"Expected |W(3,3)| <= |W(1,1)| at strong coupling; "
            f"got W(1,1)={w11:.4f}, W(3,3)={w33:.4f}"
        )

        sigma = summary["area_law_fit"]["sigma"]
        # We expect σ > 0 in a confined-like regime; enforce a modest threshold
        assert sigma > 0.0, f"Expected σ>0 at strong coupling, got σ={sigma:.6f}"


class TestConfinementPolyakovLoop:
    def test_polyakov_near_zero(self):
        """Polyakov loop should be small (near zero) in the confined regime.
        """
        summary = _strong_coupling_summary(beta_g=0.8)
        p_mean = summary["polyakov_mean"]

        # Polyakov in confined regime: close to zero, but allow some noise.
        assert abs(p_mean) < 0.2, (
            f"Expected |⟨P⟩| ≈ 0 at strong coupling; got {p_mean:.4f}"
        )
