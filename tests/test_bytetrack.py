import numpy as np

from mot_bench import trackers
from mot_bench.trackers.bytetrack import ByteTrack, Track, drop_duplicates


def walker(x0, frames, score=0.9, dx=5.0, w=40.0, h=100.0):
    """One detection per frame of a box moving right."""
    return {f: np.array([[x0 + dx * f, 100.0, x0 + dx * f + w, 100.0 + h, score]]) for f in frames}


def run(tracker, dets_by_frame, n_frames):
    out = {}
    for f in range(1, n_frames + 1):
        dets = [d[f] for d in dets_by_frame if f in d]
        out[f] = tracker.update(np.concatenate(dets) if dets else np.zeros((0, 5)))
    return out


def ids(out, f):
    return out[f][:, 4].astype(int).tolist()


def test_one_walker_keeps_its_id():
    out = run(ByteTrack(), [walker(0, range(1, 21))], 20)
    assert all(ids(out, f) == [1] for f in out)
    np.testing.assert_allclose(out[20][0, :4], walker(0, [20])[20][0, :4], atol=1.0)


def test_new_track_is_reported_from_its_second_detection():
    out = run(ByteTrack(), [walker(0, range(1, 21)), walker(500, range(5, 21))], 20)
    late = [f for f in out if 2 in ids(out, f)]
    assert min(late) == 6


def test_scores_between_the_thresholds():
    # 0.65 is a high detection but under track_thresh + 0.1: it can't start a track
    out = run(ByteTrack(), [walker(0, range(2, 10), score=0.65)], 9)
    assert all(len(out[f]) == 0 for f in out)

    # a low detection (0.1 to 0.6) keeps a track going, a score of exactly 0.6 is ignored
    # and low_thresh >= track_thresh turns the second association off
    for score, params, kept in ((0.3, {}, True), (0.6, {}, False), (0.3, {"low_thresh": 0.6}, False)):
        dets = walker(0, range(1, 21))
        dets.update(walker(0, range(8, 13), score=score))
        out = run(ByteTrack(**params), [dets], 20)
        assert all(len(out[f]) == int(kept) for f in range(8, 13))
        assert ids(out, 20) == [1]


def test_lost_track_comes_back_within_the_buffer():
    # track_buffer=5 at 30 fps: lost for up to 5 frames
    frames = [f for f in range(1, 31) if not 11 <= f <= 15]
    out = run(ByteTrack(track_buffer=5), [walker(0, frames)], 30)
    assert ids(out, 16) == [1]


def test_buffer_follows_the_frame_rate():
    frames = [f for f in range(1, 31) if not 11 <= f <= 15]
    # 15 fps halves the buffer, the track is gone when the walker comes back
    out = run(ByteTrack(track_buffer=5, frame_rate=15), [walker(0, frames)], 30)
    assert ids(out, 30) == [2]


def test_track_is_removed_one_frame_late():
    # last seen on frame 10, removed on frame 16, still found on frame 17
    frames = [f for f in range(1, 31) if not 11 <= f <= 16]
    out = run(ByteTrack(track_buffer=5), [walker(0, frames)], 30)
    assert ids(out, 17) == [1]

    # a frame later it's gone
    frames = [f for f in range(1, 31) if not 11 <= f <= 17]
    out = run(ByteTrack(track_buffer=5), [walker(0, frames)], 30)
    assert len(out[18]) == 0 and ids(out, 19) == [2]

    # a track that came back that way is dropped the next time it's lost
    frames = [f for f in range(1, 31) if not 11 <= f <= 16 and f != 20]
    out = run(ByteTrack(track_buffer=5), [walker(0, frames)], 30)
    assert ids(out, 19) == [1]
    assert len(out[21]) == 0 and ids(out, 22) == [2]


def test_output_filter():
    # too wide (w/h > 1.6) or too small (area <= 100): tracked but not reported
    out = run(ByteTrack(), [walker(0, range(1, 6), w=170.0)], 5)
    assert all(len(out[f]) == 0 for f in out)
    out = run(ByteTrack(), [walker(0, range(1, 6), w=8.0, h=12.0)], 5)
    assert all(len(out[f]) == 0 for f in out)
    out = run(ByteTrack(min_box_area=0, max_aspect=10), [walker(0, range(1, 6), w=170.0)], 5)
    assert all(len(out[f]) == 1 for f in out)


def test_duplicates_keep_the_older_track():
    det = np.array([0.0, 0.0, 40.0, 100.0, 0.9])
    old, young = Track(1, det, frame=1, activated=True), Track(2, det + [1, 0, 1, 0, 0], frame=5, activated=True)
    old.frame = young.frame = 9
    tracked, lost = drop_duplicates([young], [old])
    assert tracked == [] and lost == [old]
    tracked, lost = drop_duplicates([old], [young])
    assert tracked == [old] and lost == []


def test_make_passes_the_frame_rate():
    assert trackers.make("bytetrack", {"track_buffer": 30}, frame_rate=14).max_lost == 14
    assert trackers.make("bytetrack", {"frame_rate": 30}, frame_rate=14).max_lost == 30
    # SORT doesn't take one
    assert trackers.make("sort", {}, frame_rate=14).max_age == 1
