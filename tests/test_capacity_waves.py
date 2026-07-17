"""Checks for the finite-speed (telegrapher) capacity dynamics."""

import numpy as np
import pytest

from project_genesis.capacity_waves import (
    dispersion_omega,
    front_radius,
    steady_kappa,
    step_capacity_parabolic,
    step_capacity_wave,
    wave_mass2,
    wave_speed,
)


def test_steady_state_is_fixed_point():
    rho, r = 0.1, 0.05
    kbar = steady_kappa(r, 2.0, rho)
    kappa = np.full((16, 16), kbar)
    kdot = np.zeros_like(kappa)
    for _ in range(50):
        kappa, kdot = step_capacity_wave(kappa, kdot, rho, tau=4.0,
                                         kappa_recovery=r, dt=0.1)
    assert np.allclose(kappa, kbar, atol=1e-12)
    assert np.allclose(kdot, 0.0, atol=1e-12)


def test_parabolic_step_matches_relaxation_direction():
    kappa = np.full((16, 16), 0.5)
    out = step_capacity_parabolic(kappa, 0.0, kappa_recovery=0.2, dt=0.1)
    assert np.all(out > kappa)          # recovery pulls toward baseline


def test_cfl_guard():
    kappa = np.ones((8, 8))
    with pytest.raises(ValueError):
        step_capacity_wave(kappa, np.zeros_like(kappa), 0.0,
                           tau=0.01, dt=0.1)


def test_front_respects_causality():
    n, tau, dt = 96, 4.0, 0.1
    c = wave_speed(1.0, tau)
    kappa = np.full((n, n), 1.0)
    g = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    d2 = sum((x - n / 2.0) ** 2 for x in g)
    kappa += 1e-3 * np.exp(-d2 / 8.0)
    kdot = np.zeros_like(kappa)
    r0 = 2.0 * np.sqrt(2.0 * np.log(1e-3 / 1e-12))
    for i in range(400):
        kappa, kdot = step_capacity_wave(kappa, kdot, 0.0, tau=tau,
                                         kappa_recovery=0.05, dt=dt)
    t = 400 * dt
    rf = front_radius(kappa - steady_kappa(0.05, 2.0, 0.0),
                      (n / 2.0, n / 2.0), 1e-12)
    assert rf <= c * t + r0 + 2.0       # nothing outruns the cone


def test_dispersion_matches_theory():
    n, tau, r, rho = 128, 32.0, 0.05, 0.1
    k = 2.0 * np.pi / 16.0
    kbar = steady_kappa(r, 2.0, rho)
    x = np.arange(n)
    kappa = kbar + 1e-4 * np.cos(k * x)[:, None] * np.ones((1, 4))
    kdot = np.zeros_like(kappa)
    proj = np.cos(k * x)
    a = []
    for _ in range(2500):
        kappa, kdot = step_capacity_wave(kappa, kdot, rho, tau=tau,
                                         kappa_recovery=r, dt=0.1)
        a.append(2.0 * float(np.mean((kappa[:, 0] - kbar) * proj)))
    a = np.asarray(a)
    t = 0.1 * (1 + np.arange(len(a)))
    crossings = [t[i - 1] + 0.1 * a[i - 1] / (a[i - 1] - a[i])
                 for i in range(1, len(a)) if a[i - 1] * a[i] < 0]
    omega = np.pi / float(np.mean(np.diff(crossings)))
    theory = dispersion_omega(k, tau=tau, kappa_recovery=r, rho=rho)
    assert abs(omega - theory) / theory < 0.05


def test_mass_grows_with_density_and_speed_with_update_rate():
    assert wave_mass2(0.05, 2.0, 0.2) > wave_mass2(0.05, 2.0, 0.0)
    assert wave_speed(1.0, 4.0) > wave_speed(1.0, 64.0)
    assert np.isnan(dispersion_omega(0.01, tau=0.5))   # overdamped mode


def test_retarded_lone_mass_is_quiet():
    from project_genesis.capacity_waves import evolve_inertial_retarded
    hist = evolve_inertial_retarded(
        [[24.0, 24.0]], [[0.0, 0.0]], [1.2],
        shape=(48, 48), tau=1.0, dt=0.1, steps=200, record_every=20,
        kappa_recovery=0.02, kappa_consumption=0.8)
    e = np.asarray(hist["energy"])
    # a lone mass at rest in its own relaxed well: no motion, no
    # dissipation — energy conserved to fine tolerance
    assert abs(e[-1] - e[0]) / abs(e[0]) < 1e-6
    # residual kappa-dot from the finite relax tolerance seeds ~1e-11
    assert max(hist["dissipation"]) < 1e-9


def test_retarded_energy_is_lyapunov_for_moving_binary():
    from project_genesis.capacity_waves import evolve_inertial_retarded
    hist = evolve_inertial_retarded(
        [[18.0, 24.0], [30.0, 24.0]], [[0.0, 0.8], [0.0, -0.8]], [1.2, 1.2],
        shape=(48, 48), tau=1.0, dt=0.1, steps=400, record_every=20,
        kappa_recovery=0.02, kappa_consumption=0.8)
    e = np.asarray(hist["energy"])
    assert e[-1] < e[0]                      # net dissipation
    assert np.mean(np.diff(e) <= 1e-9) > 0.9  # monotone (Lyapunov)
    assert all(d >= 0.0 for d in hist["dissipation"])
