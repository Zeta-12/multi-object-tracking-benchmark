"""Linear Kalman filter, and the box models the trackers put in it.

    predict   x = F x
              P = F P F' + Q
    update    S = H P H' + R
              K = P H' S^-1
              x = x + K (z - H x)
              P = (I - K H) P (I - K H)' + K R K'

The covariance update is the Joseph form: same result as P - K H P in exact arithmetic, but it stays
symmetric and positive definite with rounding errors. filterpy, used by the SORT reference code, does the same.
DeepSORT and ByteTrack use the short form P - K S K' (joseph=False).
"""

from __future__ import annotations

import numpy as np

from mot_bench.boxes import tlbr_to_xyah, tlbr_to_xysr, xyah_to_tlbr, xysr_to_tlbr


def predict(x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return F @ x, F @ P @ F.T + Q


def update(
    x: np.ndarray, P: np.ndarray, z: np.ndarray, H: np.ndarray, R: np.ndarray, joseph: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    S = H @ P @ H.T + R
    # K = P H' S^-1, P and S are symmetric
    K = np.linalg.solve(S, H @ P).T
    x = x + K @ (z - H @ x)
    if joseph:
        I_KH = np.eye(len(x)) - K @ H
        return x, I_KH @ P @ I_KH.T + K @ R @ K.T
    return x, P - K @ S @ K.T


def constant_velocity(n: int, moving: int) -> np.ndarray:
    """Transition matrix for n measured values followed by the velocities of the first `moving` ones."""
    F = np.eye(n + moving)
    F[np.arange(moving), n + np.arange(moving)] = 1.0
    return F


class SortBox:
    """State (u, v, s, r, du, dv, ds): centre, area and w/h ratio, the ratio has no velocity.

    The noise values are the ones of the SORT code. The measurement noise is larger on area and ratio than
    on the centre, and the initial velocities get a large variance since one box says nothing about them.
    """

    F = constant_velocity(4, 3)
    H = np.eye(4, 7)
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 1e-2, 1e-2, 1e-4])
    R = np.diag([1.0, 1.0, 10.0, 10.0])
    P0 = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])

    def __init__(self, box: np.ndarray):
        self.x = np.r_[tlbr_to_xysr(box), 0.0, 0.0, 0.0]
        self.P = self.P0.copy()

    def predict(self) -> None:
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x, self.P = predict(self.x, self.P, self.F, self.Q)

    def update(self, box: np.ndarray) -> None:
        self.x, self.P = update(self.x, self.P, tlbr_to_xysr(box), self.H, self.R)

    @property
    def box(self) -> np.ndarray:
        return xysr_to_tlbr(self.x)


class XyahBox:
    """State (x, y, a, h, vx, vy, va, vh): centre, w/h ratio and height, with their velocities.

    The model of DeepSORT, reused by ByteTrack. The noise follows the box height: std of h/20 on the
    position and h/160 on the velocity, so a small far away pedestrian is expected to move fewer pixels.
    The ratio gets small fixed values. The noise values are the ones of the DeepSORT code.
    """

    F = constant_velocity(4, 4)
    H = np.eye(4, 8)
    POS = 1 / 20
    VEL = 1 / 160

    def __init__(self, box: np.ndarray):
        z = tlbr_to_xyah(box)
        h = z[3]
        self.x = np.r_[z, 0.0, 0.0, 0.0, 0.0]
        std = [2 * self.POS * h, 2 * self.POS * h, 1e-2, 2 * self.POS * h]
        std += [10 * self.VEL * h, 10 * self.VEL * h, 1e-5, 10 * self.VEL * h]
        self.P = np.diag(np.square(std))

    def predict(self) -> None:
        h = self.x[3]
        std = [self.POS * h, self.POS * h, 1e-2, self.POS * h, self.VEL * h, self.VEL * h, 1e-5, self.VEL * h]
        self.x, self.P = predict(self.x, self.P, self.F, np.diag(np.square(std)))

    def update(self, box: np.ndarray) -> None:
        h = self.x[3]
        R = np.diag(np.square([self.POS * h, self.POS * h, 1e-1, self.POS * h]))
        self.x, self.P = update(self.x, self.P, tlbr_to_xyah(box), self.H, R, joseph=False)

    @property
    def box(self) -> np.ndarray:
        return xyah_to_tlbr(self.x)
