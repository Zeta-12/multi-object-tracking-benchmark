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
        self.x = np.r_[self.measure(box), 0.0, 0.0, 0.0]
        self.P = self.P0.copy()

    @staticmethod
    def measure(box: np.ndarray) -> np.ndarray:
        return tlbr_to_xysr(box)

    def predict(self) -> None:
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x, self.P = predict(self.x, self.P, self.F, self.Q)

    def update(self, box: np.ndarray) -> None:
        self.x, self.P = update(self.x, self.P, self.measure(box), self.H, self.R)

    @property
    def box(self) -> np.ndarray:
        return xysr_to_tlbr(self.x)


class OruBox(SortBox):
    """SortBox with the observation-centric re-update (ORU) of OC-SORT.

    On the first frame without a detection the state is kept aside. When a detection comes back, the filter
    goes back to that state and runs the gap again on a virtual box moving in a straight line, at constant
    speed, from the last detection to the new one: one update and one predict per missed frame. The new
    detection is then applied as usual, so it counts twice, as in the reference. Without this, the velocity
    learnt before an occlusion keeps pushing the box for the whole gap.
    """

    def __init__(self, box: np.ndarray, replay: bool = True):
        super().__init__(box)
        self.replay_gaps = replay
        self.observed = False
        self.saved = None
        self.steps = 0
        self.last_z = None
        self.last_step = -1

    @staticmethod
    def measure(box: np.ndarray) -> np.ndarray:
        # the reference adds 1e-6 to the height in the ratio
        w, h = box[2] - box[0], box[3] - box[1]
        return np.array([box[0] + w / 2.0, box[1] + h / 2.0, w * h, w / (h + 1e-6)])

    def miss(self) -> None:
        if self.observed:
            self.saved = (self.x.copy(), self.P.copy())
        self.observed = False
        self.steps += 1

    def update(self, box: np.ndarray) -> None:
        z = last = self.measure(box)
        if self.replay_gaps and not self.observed and self.saved is not None:
            last = self.replay(z)
        self.observed = True
        self.x, self.P = update(self.x, self.P, z, self.H, self.R)
        self.last_z, self.last_step = last, self.steps
        self.steps += 1

    def replay(self, z: np.ndarray) -> np.ndarray:
        """Re-runs the gap since the last detection, returns the last virtual measurement."""
        self.x, self.P = self.saved
        self.saved = None
        gap = self.steps - self.last_step
        x1, y1, s1, r1 = self.last_z
        x2, y2, s2, r2 = z
        w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
        w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)
        dx, dy, dw, dh = (x2 - x1) / gap, (y2 - y1) / gap, (w2 - w1) / gap, (h2 - h1) / gap
        for i in range(1, gap + 1):
            w, h = w1 + i * dw, h1 + i * dh
            v = np.array([x1 + i * dx, y1 + i * dy, w * h, w / h])
            self.x, self.P = update(self.x, self.P, v, self.H, self.R)
            if i < gap:
                self.x, self.P = predict(self.x, self.P, self.F, self.Q)
        return v


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
