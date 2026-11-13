import numpy as np
import pytest

from mot_bench import trackers
from mot_bench.kalman import XywhBox
from mot_bench.trackers.botsort import BoTSORT, Track


def walker(x0, frames, score=0.9, dx=5.0, w=40.0, h=100.0, y=100.0):
    return {f: np.array([[x0 + dx * f, y, x0 + dx * f + w, y + h, score]]) for f in frames}


def run(tracker, dets_by_frame, n_frames, feats_by_frame=None, warp=None):
    out = {}
    for f in range(1, n_frames + 1):
        dets = [d[f] for d in dets_by_frame if f in d]
        dets = np.concatenate(dets) if dets else np.zeros((0, 5))
        feats = None
        if feats_by_frame is not None:
            feats = np.concatenate([fe[f] for fe in feats_by_frame if f in fe]).reshape(len(dets), -1)
        out[f] = tracker.update(dets, features=feats, warp=np.eye(2, 3) if warp is None else warp)
    return out


def ids(out, f):
    return out[f][:, 4].astype(int).tolist()


def unit(*v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


def test_registered():
    assert isinstance(trackers.make("botsort", {"reid": False}, 25.0), BoTSORT)


def test_one_walker_keeps_its_id():
    out = run(BoTSORT(reid=False), [walker(0, range(1, 21))], 20)
    assert all(ids(out, f) == [1] for f in out)


def test_new_track_is_reported_on_its_first_frame():
    # unlike ByteTrack, the tracks born on this frame are in the output
    out = run(BoTSORT(reid=False), [walker(0, range(1, 21)), walker(500, range(5, 21))], 20)
    assert min(f for f in out if 2 in ids(out, f)) == 5


def test_needs_features_and_camera_motion():
    dets = walker(0, [1])[1]
    with pytest.raises(ValueError):
        BoTSORT().update(dets, features=None, warp=np.eye(2, 3))
    with pytest.raises(ValueError):
        BoTSORT(reid=False).update(dets, warp=None)
    # both off: plain detections are enough
    assert len(BoTSORT(reid=False, cmc=False).update(dets)) == 1


def test_warp_moves_centre_and_scales_size():
    kf = XywhBox(np.array([100.0, 100.0, 140.0, 200.0]))
    kf.x[4:] = [2.0, 1.0, 0.5, 0.5]
    before = kf.x.copy()
    kf.warp(np.eye(2, 3))
    np.testing.assert_array_equal(kf.x, before)

    A = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, -5.0]])
    P = kf.P.copy()
    kf.warp(A)
    # centre: A applied to (x, y); size and velocities: only the linear part
    np.testing.assert_allclose(kf.x[:2], 2 * before[:2] + [10.0, -5.0])
    np.testing.assert_allclose(kf.x[2:], 2 * before[2:])
    np.testing.assert_allclose(kf.P, 4 * P)


def test_camera_pan_keeps_a_standing_person():
    # the person doesn't move, the camera pans: the box jumps 60 px left per frame, more than its width
    dets = {f: np.array([[1000.0 - 60 * f, 100.0, 1040.0 - 60 * f, 200.0, 0.9]]) for f in range(1, 11)}
    pan = np.array([[1.0, 0.0, -60.0], [0.0, 1.0, 0.0]])
    with_cmc = run(BoTSORT(reid=False), [dets], 10, warp=pan)
    assert all(ids(with_cmc, f) == [1] for f in with_cmc)
    without = run(BoTSORT(reid=False, cmc=False), [dets], 10)
    assert len({i for f in without for i in ids(without, f)}) > 1


def test_appearance_cost_only_for_close_and_similar_pairs():
    tracker = BoTSORT()
    t = Track(1, np.array([100.0, 100.0, 140.0, 200.0, 0.9]), unit(1, 0, 0), 1, True)
    dets = np.array(
        [
            [100.0, 100.0, 140.0, 200.0, 0.9],  # same box, same appearance
            [100.0, 100.0, 140.0, 200.0, 0.9],  # same box, other appearance
            [500.0, 100.0, 540.0, 200.0, 0.9],  # far away, same appearance
        ]
    )
    feats = np.array([unit(1, 0, 0), unit(0, 1, 0), unit(1, 0, 0)])
    cost = tracker.cost([t], dets, feats)[0]
    iou_only = tracker.cost([t], dets, None)[0]
    # identical features: cosine distance 0
    assert cost[0] == 0.0
    # distance 1 / 2 = 0.5 > 0.25: the IoU cost stays
    assert cost[1] == iou_only[1]
    # no overlap: the appearance doesn't count even if identical
    assert cost[2] == iou_only[2] == 1.0


def test_appearance_is_a_moving_average():
    t = Track(1, np.array([0.0, 0.0, 10.0, 20.0, 0.9]), unit(1, 0), 1, True)
    t.update(np.array([0.0, 0.0, 10.0, 20.0, 0.9]), unit(0, 1), 2)
    np.testing.assert_allclose(t.feat, unit(0.9, 0.1))
    # a low score detection has no feature, the appearance stays
    t.update(np.array([0.0, 0.0, 10.0, 20.0, 0.3]), None, 3)
    np.testing.assert_allclose(t.feat, unit(0.9, 0.1))


def test_appearance_picks_between_overlapping_detections():
    # a standing person, then two detections on top of it: the closer box looks like someone else
    person = {f: np.array([[100.0, 100.0, 140.0, 200.0, 0.9]]) for f in range(1, 6)}
    looks = {f: unit(1, 0, 0)[None] for f in range(1, 6)}
    closer, further = np.array([[104.0, 100.0, 144.0, 200.0, 0.9]]), np.array([[110.0, 100.0, 150.0, 200.0, 0.9]])
    person[6] = np.concatenate([closer, further])
    looks[6] = np.array([unit(0, 1, 0), unit(1, 0, 0)])

    def left_of_track_1(tracker):
        out = run(tracker, [person], 6, feats_by_frame=[looks] if tracker.reid else None)
        return out[6][out[6][:, 4] == 1][0, 0]

    # IoU alone takes the closer box, the appearance the one that looks the same. The output is the filtered
    # box, between the prediction (100) and the detection: about 103 for the closer one, 107 for the other
    assert left_of_track_1(BoTSORT(reid=False, cmc=False)) < 105.0
    assert left_of_track_1(BoTSORT(cmc=False)) > 105.0


def test_appearance_never_makes_a_pair_worse():
    tracker = BoTSORT()
    tracks = [Track(1, np.array([100.0, 100.0, 140.0, 200.0, 0.9]), unit(1, 0), 1, True)]
    rng = np.random.default_rng(0)
    dets = np.column_stack([100 + rng.uniform(-30, 30, 50), np.full(50, 100.0), np.zeros(50), np.full(50, 200.0)])
    dets[:, 2] = dets[:, 0] + 40
    dets = np.column_stack([dets, rng.uniform(0.6, 1, 50)])
    feats = np.array([unit(*v) for v in rng.normal(size=(50, 2))])
    assert np.all(tracker.cost(tracks, dets, feats) <= tracker.cost(tracks, dets, None))
