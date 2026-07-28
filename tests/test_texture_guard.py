"""Tests for the texture guard on the integration measure.

The guard exists because ``full_palette_junction_density`` reports a very high
"integration" for a field that binds nothing, once the domain scale falls to
the lattice.  So the tests pin both halves of that: the failure mode must be
reproducible (otherwise there is nothing to guard against), and the guard must
not damage a field with genuine domains (otherwise it would quietly rewrite
the published numbers).
"""

from __future__ import annotations

import numpy as np
import pytest

from project_genesis.multiphase import (
    domain_scale,
    full_palette_junction_density,
    majority_filter,
    resolved_palette_junction_density,
    sector_labels,
)
from project_genesis.selection import evolve_palette, measure_window


# ------------------------------------------------------------- domain scale

def test_uniform_field_has_maximal_domain_scale():
    labels = np.zeros((8, 8, 8), dtype=int)
    assert domain_scale(labels) == pytest.approx(labels.size)


def test_lattice_scale_noise_has_domain_scale_one():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, (20, 20, 20))
    assert domain_scale(labels) == pytest.approx(1.0, abs=0.05)


def test_domain_scale_grows_with_block_size():
    """Bigger blocks, fewer wall sites per unit volume, larger scale."""
    scales = []
    for block in (2, 4, 8):
        idx = np.indices((32, 32))
        labels = ((idx[0] // block) + (idx[1] // block)) % 2
        scales.append(domain_scale(labels))
    assert scales == sorted(scales)


# ---------------------------------------------------------- the failure mode

def test_the_artifact_is_real_and_severe():
    """Pure lattice-scale noise must score near-maximal on the RAW measure.

    This is the bug the guard exists for. If this test ever stops failing-by-
    design, the guard is solving a problem that no longer exists.
    """
    rng = np.random.default_rng(1)
    mush = rng.integers(0, 4, (24, 24, 24))
    raw = full_palette_junction_density(mush, 4)
    assert raw > 0.9
    assert domain_scale(mush) < 1.5


def test_guard_flags_the_artifact_as_unresolved():
    rng = np.random.default_rng(2)
    mush = rng.integers(0, 4, (24, 24, 24))
    out = resolved_palette_junction_density(mush, 4)
    assert out["resolved"] is False
    assert out["scale"] < 1.5


def test_guard_lowers_but_does_not_rescue_the_artifact():
    """Honest bookkeeping: the filter helps, the resolved flag is the real guard."""
    rng = np.random.default_rng(3)
    mush = rng.integers(0, 4, (24, 24, 24))
    out = resolved_palette_junction_density(mush, 4)
    assert out["density"] < out["raw"]          # filtering helps
    assert out["density"] > 0.3                 # but nowhere near enough alone
    assert out["resolved"] is False             # so the flag must carry it


# --------------------------------------------------- must not damage signal

def test_guard_is_a_no_op_on_genuine_domains():
    labels = np.zeros((24, 24, 24), dtype=int)
    labels[12:, :12] = 1
    labels[:12, 12:] = 2
    labels[12:, 12:] = 3
    out = resolved_palette_junction_density(labels, 4)
    assert out["resolved"] is True
    assert out["density"] == pytest.approx(out["raw"], rel=1e-6)


def test_majority_filter_preserves_a_large_domain_partition():
    labels = np.zeros((24, 24), dtype=int)
    labels[12:, :12] = 1
    labels[:12, 12:] = 2
    labels[12:, 12:] = 3
    filtered = majority_filter(labels, 4)
    # only the immediate seam can move; the interiors must be untouched
    assert (filtered[2:10, 2:10] == 0).all()
    assert (filtered[14:22, 2:10] == 1).all()
    assert (filtered[2:10, 14:22] == 2).all()
    assert (filtered[14:22, 14:22] == 3).all()


def test_majority_filter_removes_isolated_cells():
    labels = np.zeros((12, 12), dtype=int)
    labels[6, 6] = 1                    # a single stray cell
    assert majority_filter(labels, 2)[6, 6] == 0


def test_majority_filter_is_idempotent_on_clean_domains():
    labels = np.zeros((20, 20), dtype=int)
    labels[10:] = 1
    once = majority_filter(labels, 2)
    twice = majority_filter(once, 2)
    assert np.array_equal(once, twice)


def test_majority_filter_never_invents_a_sector():
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 3, (16, 16, 16))
    out = majority_filter(labels, 3)
    assert set(np.unique(out)).issubset(set(range(3)))


def test_filter_passes_are_cumulative():
    rng = np.random.default_rng(5)
    mush = rng.integers(0, 4, (20, 20, 20))
    d1 = full_palette_junction_density(majority_filter(mush, 4, passes=1), 4)
    d2 = full_palette_junction_density(majority_filter(mush, 4, passes=2), 4)
    assert d2 < d1


# ---------------------------------------------------- wiring into the sweep

def test_measure_window_reports_the_guarded_quantities():
    rng = np.random.default_rng(6)
    fields = evolve_palette(3, 24, 40, rng, noise=0.25, ndim=2)
    m = measure_window(fields, 3, 5, rng, noise=0.25)
    for key in ("integration", "integration_guarded", "domain_scale", "resolved"):
        assert key in m
    assert m["integration_guarded"] <= m["integration"] + 1e-9
    assert m["domain_scale"] >= 1.0


def test_guarded_integration_is_never_above_raw_across_palettes():
    rng = np.random.default_rng(7)
    for p in (2, 3, 4):
        fields = evolve_palette(p, 24, 40, rng, noise=0.25, ndim=2)
        m = measure_window(fields, p, 4, rng, noise=0.25)
        assert m["integration_guarded"] <= m["integration"] + 1e-9


def test_sector_labels_round_trip_through_the_guard():
    rng = np.random.default_rng(8)
    fields = evolve_palette(3, 20, 30, rng, noise=0.2, ndim=2)
    labels = sector_labels(fields)
    out = resolved_palette_junction_density(labels, 3)
    assert 0.0 <= out["density"] <= 1.0
    assert out["scale"] >= 1.0
