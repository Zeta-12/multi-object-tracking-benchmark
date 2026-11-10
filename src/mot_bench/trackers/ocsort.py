"""OC-SORT (Cao et al., 2023): SORT made to trust the detections more than the Kalman filter.

On top of SORT:
- ORU, observation-centric re-update: when a lost track gets a detection again, its Kalman filter runs the
  gap again on a straight line between the two detections (kalman.OruBox);
- OCM, observation-centric momentum: the cost adds inertia x score x (pi/2 - angle) / pi, where angle is
  between the track direction (from its detections delta_t frames apart) and the direction from its
  detection of delta_t frames ago to each new detection;
- OCR, observation-centric recovery: the tracks and detections left after the assignment are matched again
  on the IoU with the last detection of the track instead of its prediction;
- the reported box is the detection of the frame, not the Kalman state.
Lost tracks are kept max_age=30 frames, against 1 in SORT. use_byte adds ByteTrack's low score association,
off in the paper's MOT17 settings.

Same behaviour as the reference code (github.com/noahcao/OC_SORT, trackers/ocsort_tracker/ocsort.py, with
the output filter of evaluate_ocsort in yolox/evaluators/mot_evaluator.py):
- only detections over det_thresh (0.6) are used, there is no other score threshold;
- a new track starts with hit_streak 0 and isn't put in the ORU history: as in SORT it's reported from its
  4th consecutive detection, except during the first min_hits frames;
- the first assignment takes the unambiguous pairs directly, as in SORT, else the Hungarian step on
  -(IoU + OCM), and drops the pairs with IoU < iou_threshold. OCR runs the Hungarian step only when one IoU
  is over the threshold;
- the reference marks "no detection yet" with a box of -1 and tests sum() < 0, and a direction is only used
  when the old detection has a score >= 0. Both are kept;
- the output skips boxes under 100 px2 and boxes more than 1.6 times wider than tall.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from mot_bench.boxes import iou
from mot_bench.kalman import OruBox

NO_BOX = np.full(5, -1.0)


def direction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit vector (dy, dx) from the centre of box a to the centre of box b."""
    cx1, cy1 = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    cx2, cy2 = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    return speed / (np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6)


class Track:
    def __init__(self, track_id: int, det: np.ndarray, delta_t: int, oru: bool = True):
        self.id = track_id
        self.kf = OruBox(det[:4], replay=oru)
        self.score = float(det[4])
        self.delta_t = delta_t
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0
        self.last_obs = NO_BOX
        self.observations: dict[int, np.ndarray] = {}
        self.velocity = None

    def old_obs(self) -> np.ndarray:
        """The detection of delta_t frames ago, else the closest one after it, else the last one."""
        if not self.observations:
            return NO_BOX
        for dt in range(self.delta_t, 0, -1):
            if self.age - dt in self.observations:
                return self.observations[self.age - dt]
        return self.observations[max(self.observations)]

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.box

    def update(self, det: np.ndarray) -> None:
        if self.last_obs.sum() >= 0:
            previous = None
            for dt in range(self.delta_t, 0, -1):
                if self.age - dt in self.observations:
                    previous = self.observations[self.age - dt]
                    break
            self.velocity = direction(self.last_obs if previous is None else previous, det)
        self.last_obs = det
        self.observations[self.age] = det
        self.score = float(det[4])
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(det[:4])

    def miss(self) -> None:
        self.kf.miss()


def assign(gain: np.ndarray) -> np.ndarray:
    """Maximum total gain assignment, (row, col) pairs."""
    rows, cols = linear_sum_assignment(-gain)
    return np.stack([rows, cols], axis=1)


def momentum(dets: np.ndarray, old: np.ndarray, velocities: np.ndarray, inertia: float) -> np.ndarray:
    """OCM term, (n_dets, n_tracks)."""
    cx1, cy1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    cx2, cy2 = (old[:, 0] + old[:, 2])[:, None] / 2.0, (old[:, 1] + old[:, 3])[:, None] / 2.0
    dx, dy = cx1 - cx2, cy1 - cy2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    cos = velocities[:, 1:2] * (dx / norm) + velocities[:, 0:1] * (dy / norm)
    angle = np.arccos(np.clip(cos, -1, 1))
    diff = (np.pi / 2.0 - np.abs(angle)) / np.pi
    valid = (old[:, 4] >= 0).astype(float)[:, None]
    return ((valid * diff) * inertia).T * dets[:, 4:5]


def first_association(dets, boxes, threshold, old, velocities, inertia):
    """(det, track) matches, then the unmatched detections and tracks in the order of the reference."""
    n, m = len(dets), len(boxes)
    if m == 0:
        return np.empty((0, 2), dtype=int), np.arange(n), np.empty(0, dtype=int)
    ious = iou(dets[:, :4], boxes)
    if n == 0:
        return np.empty((0, 2), dtype=int), np.arange(n), np.arange(m)
    above = ious > threshold
    if above.sum(1).max() == 1 and above.sum(0).max() == 1:
        pairs = np.argwhere(above)
    else:
        pairs = assign(ious + momentum(dets, old, velocities, inertia))
    low = ious[pairs[:, 0], pairs[:, 1]] < threshold
    # the pairs under the threshold go at the end, after the sorted never matched ones
    free_dets = np.concatenate([np.setdiff1d(np.arange(n), pairs[:, 0]), pairs[low, 0]])
    free_trks = np.concatenate([np.setdiff1d(np.arange(m), pairs[:, 1]), pairs[low, 1]])
    return pairs[~low], free_dets.astype(int), free_trks.astype(int)


def second_chance(ious: np.ndarray, threshold: float) -> np.ndarray:
    """Pairs of a second assignment (BYTE or OCR) on the IoU alone."""
    pairs = assign(ious)
    return pairs[ious[pairs[:, 0], pairs[:, 1]] >= threshold]


class OCSort:
    def __init__(
        self,
        det_thresh: float = 0.6,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        inertia: float = 0.2,
        use_byte: bool = False,
        oru: bool = True,
        ocr: bool = True,
        kalman_box: bool = False,
        min_box_area: float = 100.0,
        max_aspect: float = 1.6,
    ):
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.inertia = inertia
        self.use_byte = use_byte
        # oru, ocr and inertia=0 turn the parts of OC-SORT off, for ablations
        self.oru = oru
        self.ocr = ocr
        # report the Kalman box instead of the detection, as SORT and ByteTrack do
        self.kalman_box = kalman_box
        self.min_box_area = min_box_area
        self.max_aspect = max_aspect
        self.tracks: list[Track] = []
        self.frame = 0
        self.next_id = 1

    def update(self, dets: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
        """dets: (n, 5) left, top, right, bottom, score. Returns (m, 6): the boxes, track id and score."""
        self.frame += 1
        scores = dets[:, 4]
        low = dets[(scores > 0.1) & (scores < self.det_thresh)]
        dets = dets[scores > self.det_thresh]

        boxes = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        keep = ~np.isnan(boxes).any(1)
        self.tracks = [t for t, k in zip(self.tracks, keep, strict=True) if k]
        boxes = boxes[keep]
        velocities = np.array([(0, 0) if t.velocity is None else t.velocity for t in self.tracks]).reshape(-1, 2)
        last = np.array([t.last_obs for t in self.tracks]).reshape(-1, 5)
        old = np.array([t.old_obs() for t in self.tracks]).reshape(-1, 5)

        th = self.iou_threshold
        matches, free_dets, free_trks = first_association(dets, boxes, th, old, velocities, self.inertia)
        for d, k in matches:
            self.tracks[k].update(dets[d])

        # BYTE: the low detections and the predictions of the free tracks.
        # The second assignments only run if one IoU is over the threshold, and then sort the free lists.
        if self.use_byte and len(low) and len(free_trks):
            ious = iou(low[:, :4], boxes[free_trks])
            if ious.max() > th:
                pairs = second_chance(ious, th)
                for d, k in pairs:
                    self.tracks[free_trks[k]].update(low[d])
                free_trks = np.setdiff1d(free_trks, free_trks[pairs[:, 1]])

        # OCR: the high detections and the last detection of the free tracks
        if self.ocr and len(free_dets) and len(free_trks):
            ious = iou(dets[free_dets, :4], last[free_trks, :4])
            if ious.max() > th:
                pairs = second_chance(ious, th)
                for d, k in pairs:
                    self.tracks[free_trks[k]].update(dets[free_dets[d]])
                free_dets = np.setdiff1d(free_dets, free_dets[pairs[:, 0]])
                free_trks = np.setdiff1d(free_trks, free_trks[pairs[:, 1]])

        for k in free_trks:
            self.tracks[k].miss()
        for d in free_dets:
            self.tracks.append(Track(self.next_id, dets[d], self.delta_t, self.oru))
            self.next_id += 1

        out = []
        for t in self.tracks:
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits or self.frame <= self.min_hits):
                box = t.kf.box if self.kalman_box or t.last_obs.sum() < 0 else t.last_obs[:4]
                w, h = box[2] - box[0], box[3] - box[1]
                if w * h > self.min_box_area and not w / h > self.max_aspect:
                    out.append([*box, t.id, t.score])
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return np.array(out).reshape(-1, 6)
