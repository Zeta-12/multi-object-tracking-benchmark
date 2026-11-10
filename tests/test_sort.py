import numpy as np

from mot_bench.trackers.sort import Sort, associate


def walker(x0, frames, dx=5.0):
    """One detection per frame of a box moving right."""
    return {f: np.array([[x0 + dx * f, 100.0, x0 + dx * f + 40, 200.0, 0.9]]) for f in frames}


def run(tracker, dets_by_frame, n_frames):
    out = {}
    for f in range(1, n_frames + 1):
        dets = [d[f] for d in dets_by_frame if f in d]
        res = tracker.update(np.concatenate(dets) if dets else np.zeros((0, 5)))
        out[f] = res
    return out


def test_one_walker_keeps_its_id():
    out = run(Sort(), [walker(0, range(1, 21))], 20)
    assert all(len(out[f]) == 1 for f in out)
    assert {r[0, 4] for r in out.values()} == {1}
    # reported boxes are the updated Kalman boxes, close to the detections
    np.testing.assert_allclose(out[20][0, :4], walker(0, [20])[20][0, :4], atol=1.0)


def test_late_track_is_reported_from_its_4th_detection():
    out = run(Sort(min_hits=3), [walker(0, range(1, 21)), walker(500, range(8, 21))], 20)
    late = {f for f, r in out.items() if (r[:, 4] == 2).any()}
    assert min(late) == 11


def test_first_frames_report_everything():
    out = run(Sort(min_hits=3), [walker(0, range(1, 10))], 9)
    assert len(out[1]) == len(out[2]) == len(out[3]) == 1


def test_one_missed_frame_keeps_the_track_but_hides_it():
    frames = [f for f in range(1, 21) if f != 10]
    out = run(Sort(max_age=1), [walker(0, frames)], 20)
    assert len(out[10]) == 0
    # back on frame 11, hit_streak restarts from 1 so it's hidden until frame 13
    assert len(out[11]) == len(out[12]) == 0
    assert out[13][0, 4] == 1


def test_two_missed_frames_start_a_new_track():
    frames = [f for f in range(1, 21) if f not in (10, 11)]
    out = run(Sort(max_age=1), [walker(0, frames)], 20)
    assert {r[0, 4] for r in out.values() if len(r)} == {1, 2}


def test_unambiguous_pairs_skip_the_assignment():
    # the Hungarian step would pick d0-t1 + d1-t0 (sum 0.58), both under the threshold
    ious = np.array([[0.35, 0.29], [0.29, 0.0]])
    matches, unmatched = associate(ious, 0.3)
    assert matches.tolist() == [[0, 0]]
    assert unmatched.tolist() == [1]
