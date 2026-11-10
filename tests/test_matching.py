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
