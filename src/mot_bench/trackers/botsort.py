"""BoT-SORT (Aharon et al., 2022): ByteTrack with camera motion compensation, a width in the Kalman state and
an appearance cost.

The association steps are the ones of ByteTrack. Three things change:
- the Kalman state is (x, y, w, h) instead of (x, y, w/h, h), see kalman.XywhBox;
- after the prediction, the tracks are moved by the camera motion between the two frames (CMC), a 2 x 3
  affine map given with the frame;
- with re-ID features, the first association and the one of the new tracks take the smaller of two costs
  for each pair: the IoU cost of ByteTrack (1 - IoU x score) and half the cosine distance between the
  appearance of the track and the one of the detection. The appearance only counts when that distance is
  under appearance_thresh (0.25) and the boxes overlap (1 - IoU <= proximity_thresh, 0.5).

The appearance of a track is a moving average of the features of its detections (0.9 old, 0.1 new),
normalised after each step. Only the high score detections have features: a match with a low one moves the
box, not the appearance.

Same behaviour as the reference code (github.com/NirAharon/BoT-SORT, tracker/bot_sort.py, with the settings
and output filter of tools/track.py for MOT17), besides what's shared with ByteTrack:
- a lost track doesn't keep its width and height velocities;
- the tracks born on the previous frame aren't predicted but are moved by the camera motion;
- the first association matches up to a cost of 0.8;
- every track of the tracked list is reported, also the ones born on this frame that aren't confirmed yet;
- the output skips boxes under 10 px2 and boxes more than 1.6 times wider than tall.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from mot_bench import matching
from mot_bench.kalman import XywhBox
from mot_bench.trackers.bytetrack import (
    LOST,
    LOW_MATCH,
    REMOVED,
    TRACKED,
    UNCONFIRMED_MATCH,
    boxes,
    drop_duplicates,
    iou_cost,
    joint,
    without,
)

ALPHA = 0.9


class Track:
    def __init__(self, track_id: int, det: np.ndarray, feat: np.ndarray | None, frame: int, activated: bool):
        self.id = track_id
        self.kf = XywhBox(det[:4])
        self.score = float(det[4])
        # the feature of the detection, already normalised once
        self.feat = feat
        self.state = TRACKED
        self.activated = activated
        self.start = self.frame = frame

    def predict(self) -> None:
        if self.state != TRACKED:
            self.kf.x[6:8] = 0.0
        self.kf.predict()

    def update(self, det: np.ndarray, feat: np.ndarray | None, frame: int) -> None:
        self.kf.update(det[:4])
        if feat is not None:
            # normalised again, as in the reference
            feat = feat / np.linalg.norm(feat)
            smooth = ALPHA * self.feat + (1 - ALPHA) * feat
            self.feat = smooth / np.linalg.norm(smooth)
        self.score = float(det[4])
        self.state = TRACKED
        self.activated = True
        self.frame = frame


def appearance_cost(tracks: list[Track], feats: np.ndarray) -> np.ndarray:
    """Half the cosine distance, in [0, 1] for normalised features."""
    if not tracks or not len(feats):
        return np.zeros((len(tracks), len(feats)))
    return np.maximum(0.0, cdist(np.array([t.feat for t in tracks]), feats, "cosine")) / 2.0


class BoTSORT:
    def __init__(
        self,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.7,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        reid: bool = True,
        cmc: bool = True,
        fuse_score: bool = True,
        min_box_area: float = 10.0,
        max_aspect: float = 1.6,
        frame_rate: float = 30.0,
    ):
        self.high_thresh = track_high_thresh
        self.low_thresh = track_low_thresh
        self.new_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.reid = reid
        self.cmc = cmc
        self.fuse_score = fuse_score
        self.min_box_area = min_box_area
        self.max_aspect = max_aspect
        self.max_lost = int(frame_rate / 30.0 * track_buffer)
        self.tracked: list[Track] = []
        self.lost: list[Track] = []
        self.removed_ids: set[int] = set()
        self.frame = 0
        self.next_id = 1

    def cost(self, tracks: list[Track], dets: np.ndarray, feats: np.ndarray | None) -> np.ndarray:
        """IoU cost fused with the score, and the appearance cost where it's allowed."""
        overlap = iou_cost(boxes(tracks), dets[:, :4])
        cost = 1 - (1 - overlap) * dets[:, 4] if self.fuse_score else overlap
        if feats is None:
            return cost
        appearance = appearance_cost(tracks, feats)
        appearance[appearance > self.appearance_thresh] = 1.0
        appearance[overlap > self.proximity_thresh] = 1.0
        return np.minimum(cost, appearance)

    def update(
        self, dets: np.ndarray, features: np.ndarray | None = None, warp: np.ndarray | None = None
    ) -> np.ndarray:
        """dets: (n, 5) left, top, right, bottom, score. features: (n, d) re-ID features of these detections,
        needed with reid=True. warp: 2 x 3 camera motion from the previous frame, used with cmc=True.
        Returns (m, 6): the boxes, track id and score."""
        if self.reid and features is None:
            raise ValueError("reid=True needs the features of the detections (mot add-features), or set reid=false")
        if self.cmc and warp is None:
            raise ValueError("cmc=True needs the camera motion (mot add-cmc, then --cmc), or set cmc=false")
        self.frame += 1
        keep = dets[:, 4] > self.low_thresh
        dets = dets[keep]
        scores = dets[:, 4]
        is_high = scores > self.high_thresh
        high, low = dets[is_high], dets[~is_high & (scores < self.high_thresh)]
        feats = None
        if self.reid:
            # one vector at a time in float64, the norm of a 2-D array can differ in the last bits
            f = np.asarray(features, dtype=float)[keep][is_high]
            feats = np.array([v / np.linalg.norm(v) for v in f]).reshape(len(f), -1)

        unconfirmed = [t for t in self.tracked if not t.activated]
        pool = joint([t for t in self.tracked if t.activated], self.lost)
        for t in pool:
            t.predict()
        if self.cmc:
            for t in pool + unconfirmed:
                t.kf.warp(warp)

        # high score detections, every confirmed track
        matches, free, unused = matching.capped(self.cost(pool, high, feats), self.match_thresh)
        updated, refound = [], []
        for k, d in matches:
            (updated if pool[k].state == TRACKED else refound).append(pool[k])
            pool[k].update(high[d], None if feats is None else feats[d], self.frame)

        # low score detections, the free tracks that weren't lost already, IoU only
        free = [pool[k] for k in free if pool[k].state == TRACKED]
        matches, missed, _ = matching.capped(iou_cost(boxes(free), low[:, :4]), LOW_MATCH)
        for k, d in matches:
            free[k].update(low[d], None, self.frame)
            updated.append(free[k])
        lost = [free[k] for k in missed]
        for t in lost:
            t.state = LOST

        # tracks born on the previous frame, with the high detections left
        high = high[unused]
        feats = None if feats is None else feats[unused]
        matches, missed, unused = matching.capped(self.cost(unconfirmed, high, feats), UNCONFIRMED_MATCH)
        for k, d in matches:
            unconfirmed[k].update(high[d], None if feats is None else feats[d], self.frame)
            updated.append(unconfirmed[k])
        removed = [unconfirmed[k] for k in missed]

        for d in unused:
            if high[d, 4] >= self.new_thresh:
                feat = None if feats is None else feats[d]
                updated.append(Track(self.next_id, high[d], feat, self.frame, activated=self.frame == 1))
                self.next_id += 1

        removed += [t for t in self.lost if self.frame - t.frame > self.max_lost]
        for t in removed:
            t.state = REMOVED

        self.tracked = joint(joint([t for t in self.tracked if t.state == TRACKED], updated), refound)
        lost = without(self.lost, self.tracked) + lost
        self.lost = [t for t in lost if t.id not in self.removed_ids]
        self.removed_ids.update(t.id for t in removed)
        self.tracked, self.lost = drop_duplicates(self.tracked, self.lost)
        return self.output()

    def output(self) -> np.ndarray:
        out = []
        for t in self.tracked:
            w, h = t.kf.x[2], t.kf.x[3]
            if w * h > self.min_box_area and not w / h > self.max_aspect:
                out.append([*t.kf.box, t.id, t.score])
        return np.array(out).reshape(-1, 6)
