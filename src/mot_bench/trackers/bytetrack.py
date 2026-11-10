"""ByteTrack (Zhang et al., 2022): SORT-like tracking that also uses the low score detections.

The detections of a frame are split by score. The high ones (> track_thresh) are matched first, to every
track including the lost ones. The low ones (low_thresh to track_thresh) are then matched by IoU to the tracks
that are still free: they're mostly occluded people, and they keep a track going without starting new ones.
A track without detection stays lost for track_buffer frames (at 30 fps) and can come back with its id.

Same behaviour as the reference code (github.com/ifzhang/ByteTrack, yolox/tracker/byte_tracker.py, plus the
output filter of yolox/evaluators/mot_evaluator.py):
- the first association costs 1 - IoU x score (fuse_score, off for MOT20), and the IoU counts boxes one
  pixel wider and taller (cython_bbox);
- the assignment is lap.lapjv with a cost limit, see matching.capped: 0.9, then 0.5 for the low detections;
- a new track needs a score of at least track_thresh + 0.1. It's reported from its second detection on,
  except on the first frame of the video, and it isn't predicted before that second detection;
- a detection with a score of exactly track_thresh is in neither group;
- a lost track doesn't keep its height velocity;
- a lost track is removed one frame late. It stays in the lost list on the frame it's removed and can
  still come back on the next one. If it does, its id stays in the removed list, and the next time it's
  lost it's dropped at once;
- a tracked and a lost track with IoU > 0.85 are the same person, the one tracked for less time is dropped;
- the output skips boxes under 100 px2 and boxes more than 1.6 times wider than tall.
"""

from __future__ import annotations

import numpy as np

from mot_bench import matching
from mot_bench.boxes import iou
from mot_bench.kalman import XyahBox

TRACKED, LOST, REMOVED = "tracked", "lost", "removed"
LOW_MATCH = 0.5
UNCONFIRMED_MATCH = 0.7
DUPLICATE = 0.15  # on 1 - IoU


class Track:
    def __init__(self, track_id: int, det: np.ndarray, frame: int, activated: bool):
        self.id = track_id
        self.kf = XyahBox(det[:4])
        self.score = float(det[4])
        self.state = TRACKED
        self.activated = activated
        self.start = self.frame = frame

    def predict(self) -> None:
        if self.state != TRACKED:
            self.kf.x[7] = 0.0
        self.kf.predict()

    def update(self, det: np.ndarray, frame: int) -> None:
        self.kf.update(det[:4])
        self.score = float(det[4])
        self.state = TRACKED
        self.activated = True
        self.frame = frame


def boxes(tracks: list[Track]) -> np.ndarray:
    return np.array([t.kf.box for t in tracks]).reshape(-1, 4)


def iou_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1 - iou(a, b, plus_one=True)


def joint(a: list[Track], b: list[Track]) -> list[Track]:
    ids = {t.id for t in a}
    return a + [t for t in b if t.id not in ids]


def without(a: list[Track], b: list[Track]) -> list[Track]:
    ids = {t.id for t in b}
    return [t for t in a if t.id not in ids]


def drop_duplicates(tracked: list[Track], lost: list[Track]) -> tuple[list[Track], list[Track]]:
    drop_tracked, drop_lost = set(), set()
    for p, q in np.argwhere(iou_cost(boxes(tracked), boxes(lost)) < DUPLICATE):
        if tracked[p].frame - tracked[p].start > lost[q].frame - lost[q].start:
            drop_lost.add(q)
        else:
            drop_tracked.add(p)
    tracked = [t for i, t in enumerate(tracked) if i not in drop_tracked]
    return tracked, [t for i, t in enumerate(lost) if i not in drop_lost]


class ByteTrack:
    def __init__(
        self,
        track_thresh: float = 0.6,
        low_thresh: float = 0.1,
        track_buffer: int = 30,
        match_thresh: float = 0.9,
        fuse_score: bool = True,
        min_box_area: float = 100.0,
        max_aspect: float = 1.6,
        frame_rate: float = 30.0,
    ):
        self.track_thresh = track_thresh
        # low_thresh >= track_thresh turns the second association off
        self.low_thresh = low_thresh
        self.new_thresh = track_thresh + 0.1
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.min_box_area = min_box_area
        self.max_aspect = max_aspect
        self.max_lost = int(frame_rate / 30.0 * track_buffer)
        self.tracked: list[Track] = []
        self.lost: list[Track] = []
        self.removed_ids: set[int] = set()
        self.frame = 0
        self.next_id = 1

    def fuse(self, cost: np.ndarray, dets: np.ndarray) -> np.ndarray:
        if not self.fuse_score:
            return cost
        return 1 - (1 - cost) * dets[:, 4]

    def update(self, dets: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
        """dets: (n, 5) left, top, right, bottom, score. Returns (m, 6): the boxes, track id and score."""
        self.frame += 1
        scores = dets[:, 4]
        high = dets[scores > self.track_thresh]
        low = dets[(scores > self.low_thresh) & (scores < self.track_thresh)]

        unconfirmed = [t for t in self.tracked if not t.activated]
        pool = joint([t for t in self.tracked if t.activated], self.lost)
        for t in pool:
            t.predict()

        # high score detections, every confirmed track
        cost = self.fuse(iou_cost(boxes(pool), high[:, :4]), high)
        matches, free, unused = matching.capped(cost, self.match_thresh)
        updated, refound = [], []
        for k, d in matches:
            (updated if pool[k].state == TRACKED else refound).append(pool[k])
            pool[k].update(high[d], self.frame)

        # low score detections, the free tracks that weren't lost already
        free = [pool[k] for k in free if pool[k].state == TRACKED]
        matches, missed, _ = matching.capped(iou_cost(boxes(free), low[:, :4]), LOW_MATCH)
        for k, d in matches:
            free[k].update(low[d], self.frame)
            updated.append(free[k])
        lost = [free[k] for k in missed]
        for t in lost:
            t.state = LOST

        # tracks born on the previous frame, with the high detections left
        high = high[unused]
        cost = self.fuse(iou_cost(boxes(unconfirmed), high[:, :4]), high)
        matches, missed, unused = matching.capped(cost, UNCONFIRMED_MATCH)
        for k, d in matches:
            unconfirmed[k].update(high[d], self.frame)
            updated.append(unconfirmed[k])
        removed = [unconfirmed[k] for k in missed]

        for d in unused:
            if high[d, 4] >= self.new_thresh:
                updated.append(Track(self.next_id, high[d], self.frame, activated=self.frame == 1))
                self.next_id += 1

        removed += [t for t in self.lost if self.frame - t.frame > self.max_lost]
        for t in removed:
            t.state = REMOVED

        self.tracked = joint(joint([t for t in self.tracked if t.state == TRACKED], updated), refound)
        # the ids removed on this frame only count from the next one, as in the reference
        lost = without(self.lost, self.tracked) + lost
        self.lost = [t for t in lost if t.id not in self.removed_ids]
        self.removed_ids.update(t.id for t in removed)
        self.tracked, self.lost = drop_duplicates(self.tracked, self.lost)
        return self.output()

    def output(self) -> np.ndarray:
        out = []
        for t in self.tracked:
            if not t.activated:
                continue
            w, h = t.kf.x[2] * t.kf.x[3], t.kf.x[3]
            if w * h > self.min_box_area and not w / h > self.max_aspect:
                out.append([*t.kf.box, t.id, t.score])
        return np.array(out).reshape(-1, 6)
