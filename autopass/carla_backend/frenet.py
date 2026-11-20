"""Lane (Frenet) frame along the staged corridor.

Built once from the travel lane's centre-line waypoints. It converts world x/y to
  s  distance along the lane (0 at the ego's spawn point)
  d  lateral offset, positive toward the passing side
and back. The corridor is near-straight by construction, so a point's s is found by
seeding near the staged origin and refining on the local tangent.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


class Frenet:
    def __init__(self, xy: np.ndarray, yaw_deg: np.ndarray, s0_index: int, passing_left: bool, step: float):
        self.P = np.asarray(xy, dtype=np.float64)                   # (M, 2) centre-line points
        yaw = np.radians(np.asarray(yaw_deg, dtype=np.float64))
        self.T = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)       # tangents (direction of travel)
        right = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)       # CARLA: +y is to the right of +x
        self.N = -right if passing_left else right                  # unit vectors toward the passing side
        self.step = float(step)
        self.S = (np.arange(len(self.P)) - s0_index) * self.step    # s of each vertex
        self.yaw = yaw
        self.passing_left = passing_left
        self._origin_index = s0_index

    @property
    def s_min(self) -> float:
        return float(self.S[0])

    @property
    def s_max(self) -> float:
        return float(self.S[-1])

    def _index(self, s: np.ndarray) -> np.ndarray:
        return np.clip(np.rint((s - self.S[0]) / self.step).astype(np.int64), 0, len(self.P) - 1)

    def to_sd(self, xy: np.ndarray, *, near_s: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Project world points. A recent ego s anchors localization across bends and branches."""
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        if near_s is None:
            origin = self._origin_index
            s = (xy - self.P[origin]) @ self.T[origin]
        else:
            if len(xy) != 1:
                raise ValueError("ego projection expects one point")
            center = int(self._index(np.array([near_s]))[0])
            radius = max(1, int(12.0 / self.step))
            candidates = np.arange(max(0, center - radius), min(len(self.P), center + radius + 1))
            dist = np.sum((xy[0] - self.P[candidates]) ** 2, axis=1)
            origin = candidates[int(np.argmin(dist))]
            s = self.S[origin] + (xy - self.P[origin]) @ self.T[origin]
        for _ in range(2):
            i = self._index(s)
            rel = xy - self.P[i]
            s = self.S[i] + np.einsum("ij,ij->i", rel, self.T[i])
        i = self._index(s)
        d = np.einsum("ij,ij->i", xy - self.P[i], self.N[i])
        return s, d

    def to_xy(self, s: float, d: float) -> Tuple[float, float]:
        i = int(self._index(np.array([s]))[0])
        p = self.P[i] + self.T[i] * (s - self.S[i]) + self.N[i] * d
        return float(p[0]), float(p[1])

    def heading(self, s: float, yaw_deg: float) -> float:
        """Heading error of a world yaw against the lane direction at s, positive toward the passing side."""
        i = int(self._index(np.array([s]))[0])
        err = math.atan2(math.sin(math.radians(yaw_deg) - self.yaw[i]), math.cos(math.radians(yaw_deg) - self.yaw[i]))
        return -err if self.passing_left else err          # CARLA yaw grows clockwise (to the right)


def build(carla, travel_wp, passing_left: bool, back_m: float = 160.0, ahead_m: float = 520.0,
          step: float = 2.0) -> Frenet:
    """Walk the travel lane backward and forward from `travel_wp` and collect its centre line."""
    back, cur = [], travel_wp
    for _ in range(int(back_m / step)):
        prev = cur.previous(step)
        if not prev:
            break
        cur = min(prev, key=lambda w: abs(((w.transform.rotation.yaw - cur.transform.rotation.yaw + 180) % 360) - 180))
        back.append(cur)
    fwd, cur = [], travel_wp
    for _ in range(int(ahead_m / step)):
        nxt = cur.next(step)
        if not nxt:
            break
        cur = min(nxt, key=lambda w: abs(((w.transform.rotation.yaw - cur.transform.rotation.yaw + 180) % 360) - 180))
        fwd.append(cur)
    wps = back[::-1] + [travel_wp] + fwd
    xy = np.array([[w.transform.location.x, w.transform.location.y] for w in wps])
    yaw = np.array([w.transform.rotation.yaw for w in wps])
    return Frenet(xy, yaw, len(back), passing_left, step)
