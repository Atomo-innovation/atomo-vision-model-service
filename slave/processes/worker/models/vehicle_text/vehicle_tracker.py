"""ByteTrack in plain NumPy (same idea and defaults as Ultralytics' bytetrack.yaml).

Each detection is matched to existing tracks by IoU in two rounds: confident detections
first, then low-score ones (often vehicles that are partly hidden), so IDs survive
brief occlusions. A Kalman filter predicts where each track moves between frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # boards may lack SciPy; greedy matching is close enough for IoU costs
    linear_sum_assignment = None


@dataclass
class Track:
    track_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2

    @property
    def area(self) -> int:
        return max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1])


class KalmanXYAH:
    """Constant-velocity Kalman filter on (center x, center y, aspect ratio, height)."""

    std_pos, std_vel = 1 / 20, 1 / 160

    def __init__(self) -> None:
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.std_pos * h, 2 * self.std_pos * h, 1e-2, 2 * self.std_pos * h,
               10 * self.std_vel * h, 10 * self.std_vel * h, 1e-5, 10 * self.std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self.std_pos * h, self.std_pos * h, 1e-2, self.std_pos * h,
               self.std_vel * h, self.std_vel * h, 1e-5, self.std_vel * h]
        return self.F @ mean, self.F @ cov @ self.F.T + np.diag(np.square(std))

    def update(self, mean, cov, z):
        h = mean[3]
        R = np.diag(np.square([self.std_pos * h, self.std_pos * h, 1e-1, self.std_pos * h]))
        S = self.H @ cov @ self.H.T + R
        K = np.linalg.solve(S, (cov @ self.H.T).T).T
        return mean + K @ (z - self.H @ mean), cov - K @ S @ K.T


def to_xyah(box: np.ndarray) -> np.ndarray:
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w / max(h, 1e-6), h])


def to_xyxy(xyah: np.ndarray) -> np.ndarray:
    cx, cy, a, h = xyah[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def assign(cost: np.ndarray, threshold: float):
    """Match rows to columns with cost <= threshold; returns (pairs, rows left, cols left)."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        pairs = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= threshold]
    else:
        pairs, used_r, used_c = [], set(), set()
        for flat in np.argsort(cost, axis=None):
            r, c = divmod(int(flat), cost.shape[1])
            if cost[r, c] > threshold:
                break
            if r not in used_r and c not in used_c:
                pairs.append((r, c))
                used_r.add(r)
                used_c.add(c)
    matched_r = {r for r, _ in pairs}
    matched_c = {c for _, c in pairs}
    return (pairs, [r for r in range(cost.shape[0]) if r not in matched_r],
            [c for c in range(cost.shape[1]) if c not in matched_c])


class _STrack:
    def __init__(self, box, score, label, kf: KalmanXYAH):
        self.mean, self.cov = kf.initiate(to_xyah(box))
        self.box, self.score, self.label = box, score, label
        self.track_id = 0
        self.activated = False
        self.lost = False
        self.last_frame = 0

    @property
    def predicted_box(self) -> np.ndarray:
        return to_xyxy(self.mean)


class ByteTracker:
    def __init__(self, high_thresh: float = 0.25, low_thresh: float = 0.1,
                 new_track_thresh: float = 0.25, match_thresh: float = 0.8,
                 track_buffer: int = 30, fps: float = 30.0) -> None:
        self.high, self.low, self.new = high_thresh, low_thresh, new_track_thresh
        self.match_thresh = match_thresh
        self.max_lost = int(fps / 30.0 * track_buffer)
        self.kf = KalmanXYAH()
        self.tracked: list[_STrack] = []
        self.lost: list[_STrack] = []
        self.frame = 0
        self.next_id = 1

    def update(self, boxes: np.ndarray, scores: np.ndarray, labels: list[str]) -> list[Track]:
        self.frame += 1
        high = scores >= self.high
        low = (scores >= self.low) & ~high
        dets_high = [_STrack(b, s, l, self.kf) for b, s, l, h in zip(boxes, scores, labels, high) if h]
        dets_low = [_STrack(b, s, l, self.kf) for b, s, l, lo in zip(boxes, scores, labels, low) if lo]

        confirmed = [t for t in self.tracked if t.activated]
        unconfirmed = [t for t in self.tracked if not t.activated]
        pool = confirmed + self.lost
        for t in pool:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)

        # Round 1: confident detections vs confirmed + lost tracks (IoU fused with score).
        cost = 1 - iou_matrix(self._boxes(pool), self._det_boxes(dets_high)) * \
            np.array([d.score for d in dets_high])[None, :] if pool and dets_high else \
            np.zeros((len(pool), len(dets_high)))
        pairs, left_tracks, left_dets = assign(cost, self.match_thresh)
        for ti, di in pairs:
            self._apply(pool[ti], dets_high[di])
        remaining = [pool[i] for i in left_tracks if not pool[i].lost]

        # Round 2: low-score detections vs tracks that were visible last frame.
        cost = 1 - iou_matrix(self._boxes(remaining), self._det_boxes(dets_low))
        pairs, left2, _ = assign(cost, 0.5)
        for ti, di in pairs:
            self._apply(remaining[ti], dets_low[di])
        for i in left2:
            remaining[i].lost = True

        # New detections may confirm last frame's tentative tracks.
        dets_new = [dets_high[i] for i in left_dets]
        cost = 1 - iou_matrix(self._boxes(unconfirmed), self._det_boxes(dets_new)) * \
            np.array([d.score for d in dets_new])[None, :] if unconfirmed and dets_new else \
            np.zeros((len(unconfirmed), len(dets_new)))
        pairs, left_unconf, left_new = assign(cost, 0.7)
        for ti, di in pairs:
            self._apply(unconfirmed[ti], dets_new[di])
            self._activate(unconfirmed[ti])
        dropped = {id(unconfirmed[i]) for i in left_unconf}

        for di in left_new:
            det = dets_new[di]
            if det.score < self.new:
                continue
            det.last_frame = self.frame
            if self.frame == 1:  # later tracks must be seen twice before they get an ID
                self._activate(det)
            self.tracked.append(det)

        # Lost tracks re-found in round 1 had ``lost`` cleared by _apply.
        everything = self.tracked + self.lost
        self.tracked = [t for t in everything if not t.lost and id(t) not in dropped]
        self.lost = [t for t in everything if t.lost and self.frame - t.last_frame <= self.max_lost]

        return [Track(t.track_id, t.label, float(t.score), tuple(int(v) for v in t.box))
                for t in self.tracked if t.activated and t.last_frame == self.frame]

    def _activate(self, track: _STrack) -> None:
        track.activated = True
        track.track_id = self.next_id
        self.next_id += 1

    def _apply(self, track: _STrack, det: _STrack) -> None:
        track.mean, track.cov = self.kf.update(track.mean, track.cov, to_xyah(det.box))
        track.box, track.score, track.label = det.box, det.score, det.label
        track.last_frame = self.frame
        track.lost = False

    @staticmethod
    def _boxes(tracks: list[_STrack]) -> np.ndarray:
        return np.array([t.predicted_box for t in tracks]).reshape(-1, 4)

    @staticmethod
    def _det_boxes(dets: list[_STrack]) -> np.ndarray:
        return np.array([d.box for d in dets]).reshape(-1, 4)
