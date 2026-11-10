"""Box formats and IoU.

tlwh  left, top, width, height (MOTChallenge files)
tlbr  left, top, right, bottom (what the trackers work with)
xysr  centre x, centre y, area, aspect ratio w/h (SORT state)
xyah  centre x, centre y, aspect ratio w/h, height (DeepSORT and ByteTrack state)
"""

from __future__ import annotations

import numpy as np


def tlwh_to_tlbr(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    return np.concatenate([b[..., :2], b[..., :2] + b[..., 2:4]], axis=-1)


def tlbr_to_tlwh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    return np.concatenate([b[..., :2], b[..., 2:4] - b[..., :2]], axis=-1)


def tlbr_to_xysr(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / h])


def xysr_to_tlbr(x: np.ndarray) -> np.ndarray:
    # nan when the area or the ratio went negative, the tracker drops those tracks
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


def tlbr_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / h, h])


def xyah_to_tlbr(x: np.ndarray) -> np.ndarray:
    w, h = x[2] * x[3], x[3]
    left, top = x[0] - w / 2, x[1] - h / 2
    return np.array([left, top, left + w, top + h])


def iou(a: np.ndarray, b: np.ndarray, plus_one: bool = False) -> np.ndarray:
    """Pairwise IoU of two sets of tlbr boxes, shape (len(a), len(b)).

    plus_one counts a box from x1 to x2 as x2 - x1 + 1 wide, the pixel convention of the Faster R-CNN code.
    ByteTrack computes its IoU that way (cython_bbox).
    """
    a = np.asarray(a, dtype=float).reshape(-1, 4)[:, None]
    b = np.asarray(b, dtype=float).reshape(-1, 4)[None]
    one = 1.0 if plus_one else 0.0
    w = np.maximum(0.0, np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]) + one)
    h = np.maximum(0.0, np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]) + one)
    inter = w * h
    area_a = (a[..., 2] - a[..., 0] + one) * (a[..., 3] - a[..., 1] + one)
    area_b = (b[..., 2] - b[..., 0] + one) * (b[..., 3] - b[..., 1] + one)
    return inter / (area_a + area_b - inter)
