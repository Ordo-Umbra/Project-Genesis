import numpy as np

from .numba_kernels import (
    correlation_kernel_3d,
    gradient_dot_product_3d,
    gradient_squared_3d,
    laplacian_3d,
    solve_poisson_jacobi,
)


def calculate_gradients(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Laplacian and |∇φ|² using Numba-accelerated kernels."""
    lap = np.empty_like(field, dtype=np.float64)
    gsq = np.empty_like(field, dtype=np.float64)
    laplacian_3d(np.ascontiguousarray(field, dtype=np.float64), lap)
    gradient_squared_3d(np.ascontiguousarray(field, dtype=np.float64), gsq)
    return lap, gsq


def compute_s_functional(
    field: np.ndarray,
    prev_field: np.ndarray | None,
    beta: float,
    gravity: float,
    *,
    coherence_potential: bool = False,
    poisson_iterations: int = 30,
    integration_functional: bool = False,
    integration_radius: int = 2,
    integration_decay: float = 1.0,
) -> dict[str, float]:
    """Compute S-functional components: S = ΔC + κΔI."""
    laplacian, gradient_squared = calculate_gradients(field)

    delta_c = float(np.mean(beta * gradient_squared))

    # Capacity field: high gradient energy suppresses integration capacity.
    gradient_energy_density = float(np.mean(gradient_squared))
    kappa = 1.0 / (1.0 + gradient_energy_density)

    if prev_field is None:
        delta_i = 0.0
    else:
        # ΔI ≥ 0: only count smoothing (Laplacian reduction) as integration gain.
        prev_laplacian, _ = calculate_gradients(prev_field)
        delta_i = float(max(0.0, np.mean(np.abs(prev_laplacian)) - np.mean(np.abs(laplacian))))

    s_increment = delta_c + kappa * delta_i

    result: dict[str, float] = {
        "delta_c": delta_c,
        "delta_i": delta_i,
        "kappa": kappa,
        "s_increment": s_increment,
    }

    f64 = np.ascontiguousarray(field, dtype=np.float64)

    if coherence_potential:
        potential = np.zeros_like(f64)
        solve_poisson_jacobi(f64, potential, poisson_iterations)
        grad_dot = np.empty_like(f64)
        gradient_dot_product_3d(potential, f64, grad_dot)
        result["coherence_advection_mean"] = float(np.mean(gravity * grad_dot))

    if integration_functional:
        integ = np.empty_like(f64)
        correlation_kernel_3d(f64, integ, integration_radius, integration_decay)
        result["integration_mean"] = float(np.mean(integ))

    return result


def compute_local_s(patch: np.ndarray, beta: float) -> float:
    """Compute a scalar S-functional proxy for a small 3-D patch.

    Uses the same formulation as the global S-functional but restricted to
    the patch volume.  Suitable for evaluating candidate stable objects.
    """
    lap = np.empty_like(patch, dtype=np.float64)
    gsq = np.empty_like(patch, dtype=np.float64)
    laplacian_3d(np.ascontiguousarray(patch, dtype=np.float64), lap)
    gradient_squared_3d(np.ascontiguousarray(patch, dtype=np.float64), gsq)
    delta_c = float(np.mean(beta * gsq))
    gradient_energy_density = float(np.mean(gsq))
    kappa = 1.0 / (1.0 + gradient_energy_density)
    # Without a previous state for the patch we approximate ΔI as 0.
    return delta_c + kappa * 0.0


def summarize_field(
    field: np.ndarray,
    voxel_chunk: np.ndarray,
    beta: float,
    gravity: float,
    *,
    step: int | None = None,
    prev_field: np.ndarray | None = None,
    coherence_potential: bool = False,
    poisson_iterations: int = 30,
    integration_functional: bool = False,
    integration_radius: int = 2,
    integration_decay: float = 1.0,
) -> dict[str, float | int]:
    laplacian, gradient_squared = calculate_gradients(field)
    complexity_term = beta * gradient_squared
    gravity_term = gravity * field

    metrics: dict[str, float | int] = {
        "field_min": float(np.min(field)),
        "field_max": float(np.max(field)),
        "field_mean": float(np.mean(field)),
        "field_std": float(np.std(field)),
        "complexity_mean": float(np.mean(complexity_term)),
        "laplacian_mean": float(np.mean(laplacian)),
        "laplacian_abs_mean": float(np.mean(np.abs(laplacian))),
        "gravity_mean": float(np.mean(gravity_term)),
        "mean_delta": float(np.mean(laplacian + complexity_term - gravity_term)),
        "void_ratio": float(np.mean(voxel_chunk == 0)),
        "air_ratio": float(np.mean(voxel_chunk == 1)),
        "soil_ratio": float(np.mean(voxel_chunk == 2)),
        "stone_ratio": float(np.mean(voxel_chunk == 3)),
        "bedrock_ratio": float(np.mean(voxel_chunk == 4)),
    }
    if step is not None:
        metrics["step"] = step

    s_metrics = compute_s_functional(
        field,
        prev_field,
        beta,
        gravity,
        coherence_potential=coherence_potential,
        poisson_iterations=poisson_iterations,
        integration_functional=integration_functional,
        integration_radius=integration_radius,
        integration_decay=integration_decay,
    )
    metrics.update(s_metrics)

    return metrics
