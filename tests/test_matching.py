import itertools

import numpy as np
import pytest

from mot_bench import matching


def brute_force(cost):
    n, m = cost.shape
    if n > m:
        return brute_force(cost.T)
    return min(sum(cost[i, c] for i, c in enumerate(cols)) for cols in itertools.permutations(range(m), n))


@pytest.mark.parametrize("shape", [(3, 3), (2, 4), (4, 2), (5, 5)])
def test_hungarian_is_optimal(shape):
    cost = np.random.default_rng(sum(shape)).uniform(size=shape)
    matches, _, _ = matching.hungarian(cost, max_cost=np.inf)
    assert len(matches) == min(shape)
    assert cost[matches[:, 0], matches[:, 1]].sum() == pytest.approx(brute_force(cost))


def test_pairs_above_max_cost_are_dropped():
    cost = np.array([[0.1, 0.9], [0.8, 0.95]])
    matches, rows, cols = matching.hungarian(cost, max_cost=0.5)
    assert matches.tolist() == [[0, 0]]
    assert rows.tolist() == [1]
    assert cols.tolist() == [1]


def test_empty():
    matches, rows, cols = matching.hungarian(np.zeros((0, 3)), 1.0)
    assert matches.shape == (0, 2)
    assert rows.tolist() == [] and cols.tolist() == [0, 1, 2]


def brute_force_capped(cost, limit):
    """Best total over every partial assignment, a pair costing limit / 2 per row and column left out."""
    n, m = cost.shape
    best = (n + m) * limit / 2
    for k in range(1, min(n, m) + 1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.permutations(range(m), k):
                total = sum(cost[r, c] for r, c in zip(rows, cols, strict=True)) + (n + m - 2 * k) * limit / 2
                best = min(best, total)
    return best


@pytest.mark.parametrize("seed", range(5))
def test_capped_is_optimal(seed):
    rng = np.random.default_rng(seed)
    cost = np.where(rng.random((4, 5)) < 0.5, 1.0, rng.random((4, 5)))
    matches, rows, cols = matching.capped(cost, 0.7)
    picked = cost[matches[:, 0], matches[:, 1]]
    assert (picked < 0.7).all()
    total = picked.sum() + (9 - 2 * len(matches)) * 0.35
    assert total == pytest.approx(brute_force_capped(cost, 0.7))
    assert len(rows) == 4 - len(matches) and len(cols) == 5 - len(matches)


def test_capped_and_hungarian_differ():
    # the Hungarian step pays 1.0 for the bad pair and prefers two medium pairs,
    # with the cap the bad pair costs the limit and the very good pair wins
    cost = np.array([[0.05, 0.5], [0.5, 1.0]])
    assert matching.capped(cost, 0.9)[0].tolist() == [[0, 0]]
    assert matching.hungarian(cost, 0.9)[0].tolist() == [[0, 1], [1, 0]]
