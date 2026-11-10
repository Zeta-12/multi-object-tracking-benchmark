import numpy as np
import pytest

from mot_bench import boxes


def test_conversions_round_trip():
    tlwh = np.array([[10.0, 20.0, 30.0, 60.0], [0.5, 1.5, 2.0, 4.0]])
    tlbr = boxes.tlwh_to_tlbr(tlwh)
    assert tlbr[0].tolist() == [10, 20, 40, 80]
    np.testing.assert_allclose(boxes.tlbr_to_tlwh(tlbr), tlwh)
    x = boxes.tlbr_to_xysr(tlbr[0])
    assert x.tolist() == [25, 50, 1800, 0.5]
    np.testing.assert_allclose(boxes.xysr_to_tlbr(x), tlbr[0])


def test_negative_area_gives_nan():
    assert np.isnan(boxes.xysr_to_tlbr(np.array([0.0, 0.0, -5.0, 0.5]))).all()


def test_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30], [0, 0, 5, 5]])
    np.testing.assert_allclose(boxes.iou(a, b), [[1, 1 / 3, 0, 0.25]])
    assert boxes.iou(a, np.zeros((0, 4))).shape == (1, 0)
    assert boxes.iou(np.zeros((0, 4)), b).shape == (0, 4)


@pytest.mark.parametrize("seed", range(3))
def test_iou_is_symmetric(seed):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, (6, 2))
    b = np.concatenate([xy, xy + rng.uniform(5, 40, (6, 2))], axis=1)
    m = boxes.iou(b, b)
    np.testing.assert_allclose(m, m.T)
    np.testing.assert_allclose(np.diag(m), 1)
