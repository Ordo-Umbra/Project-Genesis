import numpy as np
import scipy.ndimage as nd


def calculate_gradients(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    laplacian = nd.laplace(field)
    grad_x, grad_y, grad_z = np.gradient(field)
    gradient_squared = grad_x**2 + grad_y**2 + grad_z**2
    return laplacian, gradient_squared


def summarize_field(
    field: np.ndarray,
    voxel_chunk: np.ndarray,
    beta: float,
    gravity: float,
    *,
    step: int | None = None,
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
        "air_ratio": float(np.mean(voxel_chunk == 0)),
        "soil_ratio": float(np.mean(voxel_chunk == 1)),
        "stone_ratio": float(np.mean(voxel_chunk == 2)),
    }
    if step is not None:
        metrics["step"] = step
    return metrics
