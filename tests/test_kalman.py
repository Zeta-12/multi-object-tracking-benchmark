import numpy as np

from mot_bench import kalman
from mot_bench.kalman import SortBox, XyahBox


def test_update_matches_the_textbook_form():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    P = A @ A.T + np.eye(6)
    x = rng.normal(size=6)
    H = np.eye(3, 6)
    R = np.diag([1.0, 2.0, 3.0])
    z = rng.normal(size=3)

    x1, P1 = kalman.update(x, P, z, H, R)
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    np.testing.assert_allclose(x1, x + K @ (z - H @ x))
    np.testing.assert_allclose(P1, (np.eye(6) - K @ H) @ P, atol=1e-12)
    np.testing.assert_allclose(P1, P1.T)


def test_constant_velocity_matrix():
    F = kalman.constant_velocity(4, 3)
    x = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0])
    assert (F @ x).tolist() == [11, 22, 33, 4, 10, 20, 30]


def test_sort_box_learns_the_velocity():
    box = np.array([100.0, 50.0, 140.0, 150.0])
    kf = SortBox(box)
    np.testing.assert_allclose(kf.box, box)
    for _ in range(20):
        kf.predict()
        box = box + [4.0, -1.0, 4.0, -1.0]
        kf.update(box)
    np.testing.assert_allclose(kf.x[4:6], [4.0, -1.0], atol=0.05)
    kf.predict()
    np.testing.assert_allclose(kf.box, box + [4.0, -1.0, 4.0, -1.0], atol=0.5)


def test_area_velocity_is_reset_before_the_area_goes_negative():
    kf = SortBox(np.array([0.0, 0.0, 10.0, 10.0]))
    kf.x[6] = -500.0
    kf.predict()
    assert kf.x[2] == 100.0
    assert kf.x[6] == 0.0


def test_short_form_matches_joseph():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(8, 8))
    P = A @ A.T + np.eye(8)
    x, z, H, R = rng.normal(size=8), rng.normal(size=4), np.eye(4, 8), np.diag([1.0, 2.0, 3.0, 4.0])
    x1, P1 = kalman.update(x, P, z, H, R)
    x2, P2 = kalman.update(x, P, z, H, R, joseph=False)
    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(P1, P2, atol=1e-12)


def test_xyah_noise_follows_the_height():
    small = XyahBox(np.array([0.0, 0.0, 20.0, 50.0]))
    tall = XyahBox(np.array([0.0, 0.0, 40.0, 100.0]))
    np.testing.assert_allclose(
        tall.P[[0, 1, 3, 4, 5, 7], [0, 1, 3, 4, 5, 7]], 4 * small.P[[0, 1, 3, 4, 5, 7], [0, 1, 3, 4, 5, 7]]
    )
    # the ratio noise doesn't depend on the size
    assert tall.P[2, 2] == small.P[2, 2]


def test_xyah_box_learns_the_velocity():
    box = np.array([100.0, 50.0, 140.0, 150.0])
    kf = XyahBox(box)
    np.testing.assert_allclose(kf.box, box)
    # slower than SORT: the velocity noise is small next to the measurement noise
    for _ in range(50):
        kf.predict()
        box = box + [4.0, -1.0, 4.0, -1.0]
        kf.update(box)
    np.testing.assert_allclose(kf.x[4:6], [4.0, -1.0], atol=0.05)
