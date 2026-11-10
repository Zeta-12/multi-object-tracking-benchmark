import numpy as np
import pytest

from mot_bench.kalman import OruBox
from mot_bench.trackers.ocsort import NO_BOX, OCSort, momentum


def walker(x0, frames, score=0.9, dx=5.0):
    """One detection per frame of a box moving right."""
    return {f: np.array([[x0 + dx * f, 100.0, x0 + dx * f + 40, 200.0, score]]) for f in frames}


def run(tracker, dets_by_frame, n_frames):
    out = {}
    for f in range(1, n_frames + 1):
        dets = [d[f] for d in dets_by_frame if f in d]
        out[f] = tracker.update(np.concatenate(dets) if dets else np.zeros((0, 5)))
    return out


def ids(out, f):
    return out[f][:, 4].astype(int).tolist()


def test_reports_the_detection_itself():
    dets = walker(0, range(1, 21))
    out = run(OCSort(), [dets], 20)
    assert all(ids(out, f) == [1] for f in out)
    assert out[20][0, :4].tolist() == dets[20][0, :4].tolist()
    # or the Kalman box, close to it
    out = run(OCSort(kalman_box=True), [dets], 20)
    assert out[20][0, :4].tolist() != dets[20][0, :4].tolist()
    np.testing.assert_allclose(out[20][0, :4], dets[20][0, :4], atol=1.0)


def test_new_track_is_reported_from_its_4th_detection():
    out = run(OCSort(), [walker(0, range(1, 21)), walker(500, range(8, 21))], 20)
    late = [f for f in out if 2 in ids(out, f)]
    assert min(late) == 11


def test_score_must_be_over_det_thresh():
    out = run(OCSort(), [walker(0, range(1, 10), score=0.6)], 9)
    assert all(len(out[f]) == 0 for f in out)


def test_lost_track_keeps_its_id_for_max_age_frames():
    frames = [f for f in range(1, 41) if not 11 <= f <= 20]
    out = run(OCSort(), [walker(0, frames)], 40)
    assert ids(out, 40) == [1]
    out = run(OCSort(max_age=5), [walker(0, frames)], 40)
    assert ids(out, 40) == [2]


@pytest.mark.parametrize("ocr, same_id", [(True, True), (False, False)])
def test_ocr_matches_on_the_last_detection(ocr, same_id):
    # fast walker that stops while hidden: the prediction is far ahead, the last detection is still right
    dets = walker(0, range(1, 11), dx=20.0)
    stop = dets[10]
    dets.update({f: stop for f in range(16, 31)})
    out = run(OCSort(ocr=ocr), [dets], 30)
    assert (ids(out, 30) == [1]) == same_id


def walk(kf, box, step, frames):
    for _ in range(frames):
        kf.predict()
        box = box + step
        kf.update(box)
    return box


def test_oru_is_the_filter_fed_with_the_virtual_boxes():
    right, down = np.array([4.0, 0.0, 4.0, 0.0]), np.array([0.0, 4.0, 0.0, 4.0])
    start = np.array([0.0, 0.0, 40.0, 100.0])

    # turns down while hidden for 10 frames, seen again on the 11th
    kf = OruBox(start)
    last = walk(kf, start, right, 20)
    for _ in range(10):
        kf.predict()
        kf.miss()
    kf.predict()
    new = last + 11 * down
    kf.update(new)

    # same filter with a box on every frame of the straight line, then the new box once more
    ref = OruBox(start, replay=False)
    walk(ref, start, right, 20)
    walk(ref, last, (new - last) / 11, 11)
    ref.update(new)
    np.testing.assert_allclose(kf.x, ref.x, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(kf.P, ref.P, rtol=1e-5, atol=1e-9)

    # and it ends closer to the real track than the plain filter
    plain = OruBox(start, replay=False)
    walk(plain, start, right, 20)
    for _ in range(10):
        plain.predict()
        plain.miss()
    plain.predict()
    plain.update(new)
    truth = new + 5 * down
    for f in (kf, plain):
        for _ in range(5):
            f.predict()
    assert np.abs(kf.box - truth).max() < np.abs(plain.box - truth).max()


def test_measure_adds_1e6_to_the_height():
    z = OruBox.measure(np.array([0.0, 0.0, 30.0, 60.0]))
    assert z[3] == 30.0 / (60.0 + 1e-6)


def test_momentum_favours_the_direction_of_motion():
    old = np.array([[0.0, 0.0, 10.0, 10.0, 0.9]])
    velocity = np.array([[0.0, 1.0]])  # (dy, dx): moving right
    dets = np.array([[20.0, 0.0, 30.0, 10.0, 1.0], [-20.0, 0.0, -10.0, 10.0, 1.0], [0.0, 20.0, 10.0, 30.0, 1.0]])
    np.testing.assert_allclose(momentum(dets, old, velocity, 0.2)[:, 0], [0.1, -0.1, 0.0], atol=1e-3)
    # no old detection, no direction term
    assert (momentum(dets, NO_BOX[None], velocity, 0.2) == 0).all()
