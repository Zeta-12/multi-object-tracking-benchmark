"""Assignment between tracks and detections."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian(cost: np.ndarray, max_cost: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum total cost assignment, then the pairs above max_cost are dropped.

    Returns the (row, col) matches and the unmatched rows and columns, all sorted.
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.empty((0, 2), dtype=int), np.arange(n), np.arange(m)
    rows, cols = linear_sum_assignment(cost)
    keep = cost[rows, cols] <= max_cost
    matches = np.stack([rows[keep], cols[keep]], axis=1)
    return matches, np.setdiff1d(np.arange(n), matches[:, 0]), np.setdiff1d(np.arange(m), matches[:, 1])
