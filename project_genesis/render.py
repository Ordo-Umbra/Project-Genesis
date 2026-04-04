from pathlib import Path

import numpy as np

VOXEL_SYMBOLS = {
    0: " ",   # void
    1: ".",   # air
    2: "+",   # soil
    3: "#",   # stone
    4: "@",   # bedrock
}


def render_voxel_slice(
    voxels: np.ndarray,
    *,
    axis: str = "z",
    index: int | None = None,
) -> str:
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_map:
        raise ValueError("axis must be one of: x, y, z")

    dimension = axis_map[axis]
    slice_index = voxels.shape[dimension] // 2 if index is None else index
    if not 0 <= slice_index < voxels.shape[dimension]:
        raise IndexError("slice index is out of bounds")

    if dimension == 0:
        plane = voxels[slice_index, :, :]
    elif dimension == 1:
        plane = voxels[:, slice_index, :]
    else:
        plane = voxels[:, :, slice_index]

    rows = ["".join(VOXEL_SYMBOLS[int(cell)] for cell in row) for row in plane]
    return "\n".join(rows)


def write_voxel_slice(
    path: str | Path,
    voxels: np.ndarray,
    *,
    axis: str = "z",
    index: int | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_voxel_slice(voxels, axis=axis, index=index) + "\n", encoding="utf-8")
    return target
