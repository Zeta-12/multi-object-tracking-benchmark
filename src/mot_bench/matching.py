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
    return _result(rows[keep], cols[keep], n, m)


def capped(cost: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum total cost assignment where a pair never costs more than `limit`, then those pairs are dropped.

    A pair at the limit costs as much as leaving its row and column unmatched, so this is the best assignment
    that is allowed to leave rows and columns out. It's what lap.lapjv(cost, extend_cost=True,
    cost_limit=limit) returns, the solver of ByteTrack. `hungarian` with max_cost=limit can differ: it sees
    the full cost of the pairs above the limit and can give up a very good pair to avoid one of them.
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.empty((0, 2), dtype=int), np.arange(n), np.arange(m)
    rows, cols = linear_sum_assignment(np.minimum(cost, limit))
    keep = cost[rows, cols] < limit
    return _result(rows[keep], cols[keep], n, m)


def _result(rows: np.ndarray, cols: np.ndarray, n: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matches = np.stack([rows, cols], axis=1)
    return matches, np.setdiff1d(np.arange(n), rows), np.setdiff1d(np.arange(m), cols)
