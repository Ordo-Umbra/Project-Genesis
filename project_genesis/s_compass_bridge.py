"""S-compass connector bridge.

Takes the perception data produced by :meth:`Agent.get_perception` and
computes a recommended action vector using the S-compass algorithm
(directional derivatives of the local S-functional).

The returned action is an axis-aligned unit move (one of the six
cardinal directions) toward the direction that maximises the local
S-functional gradient.  If all neighbours are equal the bridge returns
``(0, 0, 0)`` (stay).
"""

from __future__ import annotations

import numpy as np


def compute_s_gradient(
    s_field: np.ndarray,
    *,
    beta: float = 0.09,
) -> tuple[int, int, int]:
    """Compute the recommended move from a local S-functional sub-grid.

    Parameters
    ----------
    s_field:
        A small 3-D array of S-functional values centred on the agent.
        The agent sits at the centre voxel.
    beta:
        Complexity coupling (unused for the directional derivative but
        kept for future extensions).

    Returns
    -------
    A ``(dx, dy, dz)`` tuple representing the best axis-aligned move.
    """
    if s_field.ndim != 3:
        raise ValueError("s_field must be 3-dimensional")

    centre = tuple(s // 2 for s in s_field.shape)
    centre_val = float(s_field[centre])

    best_delta = 0.0
    best_direction: tuple[int, int, int] = (0, 0, 0)

    for axis in range(3):
        for sign in (-1, 1):
            neighbour = list(centre)
            neighbour[axis] += sign
            # If the sub-grid is large enough, look at the direct neighbour.
            if 0 <= neighbour[axis] < s_field.shape[axis]:
                val = float(s_field[tuple(neighbour)])
                delta = val - centre_val
                if delta > best_delta:
                    best_delta = delta
                    direction = [0, 0, 0]
                    direction[axis] = sign
                    best_direction = (direction[0], direction[1], direction[2])

    return best_direction


def perception_to_action(
    perception: dict,
    *,
    beta: float = 0.09,
) -> dict:
    """Convert a full perception dict into a recommended action.

    Parameters
    ----------
    perception:
        The dictionary returned by :meth:`Agent.get_perception`.

    Returns
    -------
    A dictionary with ``"type": "move"`` and a ``"direction"`` key.
    """
    s_subgrid = np.asarray(perception["s_field"])
    direction = compute_s_gradient(s_subgrid, beta=beta)

    return {
        "type": "move",
        "direction": list(direction),
    }
