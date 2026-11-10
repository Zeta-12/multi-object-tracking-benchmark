"""Box formats and IoU.

tlwh  left, top, width, height (MOTChallenge files)
tlbr  left, top, right, bottom (what the trackers work with)
xysr  centre x, centre y, area, aspect ratio w/h (SORT state)
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


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of two sets of tlbr boxes, shape (len(a), len(b))."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)[:, None]
    b = np.asarray(b, dtype=float).reshape(-1, 4)[None]
    w = np.maximum(0.0, np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]))
    h = np.maximum(0.0, np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]))
    inter = w * h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter)
