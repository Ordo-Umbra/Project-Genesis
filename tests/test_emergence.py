"""Tests for effective information and the intervention machinery.

EI is easy to compute and easy to compute *wrongly*, and the failure modes are
silent — a measure that rewards degeneracy, or an "intervention" that is really
an observation, both produce plausible-looking numbers. So the tests pin the
analytic limits exactly, and pin that the do-operation actually overwrites the
system rather than watching it.
"""

from __future__ import annotations

import numpy as np
import pytest

from project_genesis.emergence import (
    _block_slices,
    _majority_sector,
    effective_information,
    emergence_profile,
    equilibrated_backgrounds,
    intervention_tpm,
)
from project_genesis.multiphase import step_multiphase


# ------------------------------------------------------- analytic limits

def test_identity_tpm_saturates_the_ceiling():
    """Perfectly determinate, perfectly non-degenerate -> EI = log2 N."""
    for n in (2, 3, 4, 8):
        out = effective_information(np.eye(n))
        assert out["ei"] == pytest.approx(np.log2(n))
        assert out["determinism"] == pytest.approx(1.0)
        assert out["degeneracy"] == pytest.approx(0.0)
        assert out["effectiveness"] == pytest.approx(1.0)


def test_uniform_tpm_has_zero_effective_information():
    """A system that ignores what you do to it scores zero."""
    out = effective_information(np.full((4, 4), 0.25))
    assert out["ei"] == pytest.approx(0.0)


def test_fully_degenerate_tpm_scores_zero_despite_being_deterministic():
    """Every intervention leading to the SAME place is causally useless.

    This is the trap: determinism alone must not earn EI.
    """
    tpm = np.zeros((4, 4))
    tpm[:, 0] = 1.0
    out = effective_information(tpm)
    assert out["determinism"] == pytest.approx(1.0)
    assert out["degeneracy"] == pytest.approx(1.0)
    assert out["ei"] == pytest.approx(0.0, abs=1e-12)


def test_effectiveness_is_determinism_minus_degeneracy():
    rng = np.random.default_rng(0)
    tpm = rng.random((5, 5))
    tpm /= tpm.sum(axis=1, keepdims=True)
    out = effective_information(tpm)
    assert out["effectiveness"] == pytest.approx(
        out["determinism"] - out["degeneracy"])


def test_ei_never_exceeds_the_ceiling():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = int(rng.integers(2, 7))
        tpm = rng.random((n, n))
        tpm /= tpm.sum(axis=1, keepdims=True)
        assert effective_information(tpm)["ei"] <= np.log2(n) + 1e-9


def test_ei_is_non_negative_for_stochastic_matrices():
    rng = np.random.default_rng(2)
    for _ in range(20):
        tpm = rng.random((4, 4))
        tpm /= tpm.sum(axis=1, keepdims=True)
        assert effective_information(tpm)["ei"] >= -1e-12


def test_partial_noise_sits_between_the_limits():
    eps = 0.2
    tpm = (1 - eps) * np.eye(3) + eps * np.full((3, 3), 1 / 3)
    ei = effective_information(tpm)["ei"]
    assert 0.0 < ei < np.log2(3)


def test_unnormalised_rows_are_normalised():
    a = effective_information(np.eye(3) * 7.0)
    assert a["ei"] == pytest.approx(np.log2(3))


def test_non_square_input_is_rejected():
    with pytest.raises(ValueError):
        effective_information(np.ones((2, 3)))


def test_single_state_is_degenerate_by_definition():
    out = effective_information(np.array([[1.0]]))
    assert out["ei"] == 0.0


# ------------------------------------------------------------- geometry

def test_block_slices_are_centred_and_the_right_size():
    blk = _block_slices(16, 4, 2)
    arr = np.zeros((3, 16, 16))
    arr[blk] = 1.0
    assert arr.sum() == 3 * 4 * 4
    assert arr[0, 6:10, 6:10].all()


def test_block_of_full_size_covers_everything():
    blk = _block_slices(8, 8, 2)
    arr = np.zeros((2, 8, 8))
    arr[blk] = 1.0
    assert arr.all()


def test_majority_sector_picks_the_dominant_component():
    fields = np.zeros((3, 4, 4))
    fields[2] = 1.0
    fields[0, 0, 0] = 5.0          # one dissenting cell must not win
    blk = _block_slices(4, 4, 2)
    assert _majority_sector(fields, blk, 3) == 2


# ---------------------------------------------------- the do() is a do()

def _step(noise, diffusion):
    def step(fields, rng):
        out = step_multiphase(fields, diffusion=diffusion, gamma=1.5, dt=0.1)
        if noise:
            out = out + noise * rng.standard_normal(out.shape) * np.sqrt(0.1)
        return out
    return step


def test_intervention_tpm_rows_are_probability_distributions():
    rng = np.random.default_rng(3)
    step = _step(0.1, 1.0)
    bgs = equilibrated_backgrounds(4, 3, 16, 30, rng, step_fn=step, ndim=2)
    tpm = intervention_tpm(bgs, 3, 4, 5, step_fn=step, rng=rng, ndim=2)
    assert tpm.shape == (3, 3)
    assert np.allclose(tpm.sum(axis=1), 1.0)
    assert (tpm >= 0).all()


def test_intervention_actually_overwrites_the_block():
    """With zero steps of evolution, do(s) must read back exactly s.

    If this fails the 'intervention' is an observation, and every number the
    module produces is measuring the wrong thing.
    """
    rng = np.random.default_rng(4)
    step = _step(0.0, 1.0)
    bgs = equilibrated_backgrounds(3, 3, 16, 30, rng, step_fn=step, ndim=2)
    tpm = intervention_tpm(bgs, 3, 4, 0, step_fn=step, rng=rng, ndim=2)
    assert np.allclose(tpm, np.eye(3))


def test_a_whole_lattice_intervention_is_maximally_effective():
    """Overwrite everything and nothing can argue: EI must hit the ceiling."""
    rng = np.random.default_rng(5)
    step = _step(0.05, 1.0)
    bgs = equilibrated_backgrounds(3, 3, 12, 30, rng, step_fn=step, ndim=2)
    tpm = intervention_tpm(bgs, 3, 12, 10, step_fn=step, rng=rng, ndim=2)
    assert effective_information(tpm)["ei"] == pytest.approx(np.log2(3))


def test_backgrounds_are_independent_of_each_other():
    rng = np.random.default_rng(6)
    step = _step(0.1, 1.0)
    bgs = equilibrated_backgrounds(3, 3, 12, 20, rng, step_fn=step, ndim=2)
    assert not np.allclose(bgs[0], bgs[1])
    assert not np.allclose(bgs[1], bgs[2])


def test_emergence_profile_covers_every_requested_scale():
    rng = np.random.default_rng(7)
    step = _step(0.1, 1.0)
    bgs = equilibrated_backgrounds(3, 3, 16, 30, rng, step_fn=step, ndim=2)
    prof = emergence_profile([1, 4, 16], bgs, 3, 5, step_fn=step, rng=rng, ndim=2)
    assert [r["scale"] for r in prof] == [1, 4, 16]
    for r in prof:
        assert r["tpm"].shape == (3, 3)
        assert np.isfinite(r["ei"])


def test_larger_blocks_are_at_least_as_effective_as_single_sites():
    """The substantive sanity check: a bigger intervention cannot have less
    grip than a one-cell one in a coupled field."""
    rng = np.random.default_rng(8)
    step = _step(0.2, 1.0)
    bgs = equilibrated_backgrounds(12, 3, 24, 120, rng, step_fn=step, ndim=2)
    prof = emergence_profile([1, 12], bgs, 3, 15, step_fn=step, rng=rng, ndim=2)
    assert prof[1]["ei"] >= prof[0]["ei"] - 1e-9
