"""SORT (Bewley et al., 2016): one Kalman filter per track, IoU between the predicted boxes and the
detections, Hungarian assignment. No appearance model.

Same behaviour as the reference code (github.com/abewley/sort, sort.py):
- a track is created with hit_streak 0, so with min_hits=3 it is reported from its 4th consecutive
  detection on, except during the first min_hits frames of the video where every matched track is;
- a track is only reported on the frames where it got a detection, with its updated Kalman box;
- a track is deleted once it has gone more than max_age frames without a detection.
"""

from __future__ import annotations

import numpy as np

from mot_bench import matching
from mot_bench.boxes import iou
from mot_bench.kalman import SortBox


class Track:
    def __init__(self, track_id: int, det: np.ndarray):
        self.id = track_id
        self.kf = SortBox(det[:4])
        self.score = float(det[4])
        self.time_since_update = 0
        self.hit_streak = 0

    def predict(self) -> None:
        self.kf.predict()
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

    def update(self, det: np.ndarray) -> None:
        self.kf.update(det[:4])
        self.score = float(det[4])
        self.time_since_update = 0
        self.hit_streak += 1


def associate(ious: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """(det, track) matches and unmatched detections."""
    above = ious > threshold
    if above.size and above.sum(1).max() == 1 and above.sum(0).max() == 1:
        # one candidate per row and column: the reference takes these pairs and skips the assignment
        matches = np.argwhere(above)
        return matches, np.setdiff1d(np.arange(len(ious)), matches[:, 0])
    matches, unmatched, _ = matching.hungarian(-ious, -threshold)
    return matches, unmatched


class Sort:
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[Track] = []
        self.frame = 0
        self.next_id = 1

    def update(self, dets: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
        """dets: (n, 5) left, top, right, bottom, score. Returns (m, 6): the boxes, track id and score."""
        self.frame += 1
        for t in self.tracks:
            t.predict()
        self.tracks = [t for t in self.tracks if np.isfinite(t.kf.box).all()]

        boxes = np.array([t.kf.box for t in self.tracks]).reshape(-1, 4)
        matches, unmatched = associate(iou(dets[:, :4], boxes), self.iou_threshold)
        for d, k in matches:
            self.tracks[k].update(dets[d])
        for d in unmatched:
            self.tracks.append(Track(self.next_id, dets[d]))
            self.next_id += 1

        out = [
            [*t.kf.box, t.id, t.score]
            for t in self.tracks
            if t.time_since_update == 0 and (t.hit_streak >= self.min_hits or self.frame <= self.min_hits)
        ]
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return np.array(out).reshape(-1, 6)
