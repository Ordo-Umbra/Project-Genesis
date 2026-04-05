import numpy as np
import scipy.ndimage as nd


def calculate_gradients(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    laplacian = nd.laplace(field)
    grad_x, grad_y, grad_z = np.gradient(field)
    gradient_squared = grad_x**2 + grad_y**2 + grad_z**2
    return laplacian, gradient_squared


def compute_s_functional(
    field: np.ndarray,
    prev_field: np.ndarray | None,
    beta: float,
    gravity: float,
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

    return {
        "delta_c": delta_c,
        "delta_i": delta_i,
        "kappa": kappa,
        "s_increment": s_increment,
    }


def summarize_field(
    field: np.ndarray,
    voxel_chunk: np.ndarray,
    beta: float,
    gravity: float,
    *,
    step: int | None = None,
    prev_field: np.ndarray | None = None,
) -> dict[str, float | int]:
    laplacian, gradient_squared = calculate_gradients(field)
    complexity_term = beta * gradient_squared
    gravity_term = gravity * field

    metrics = {
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

    s_metrics = compute_s_functional(field, prev_field, beta, gravity)
    metrics.update(s_metrics)

    return metrics
