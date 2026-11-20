"""Geometric camera model for the kinematic simulator.

A vehicle is detected by a camera when enough of its angular extent lies inside the field of
view, is not hidden behind a nearer vehicle, and it is within the camera's reach. Detection
noise, misses and reach are set by a PerceptionProfile, which can be fitted to a real
perception stack (see docs/PERCEPTION.md).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.control import CAMERAS

Interval = Tuple[float, float]


@dataclass
class PerceptionProfile:
    name: str = "oracle"
    gap_sigma_abs: float = 0.0     # metres
    gap_sigma_rel: float = 0.0     # fraction of range
    gap_bias_rel: float = 0.0      # systematic scale error
    lat_sigma: float = 0.0         # metres
    miss_base: float = 0.0         # probability of missing a visible vehicle
    miss_per_100m: float = 0.0
    range_scale: float = 1.0       # multiplies every camera's reach
    min_visible: float = 0.25      # fraction of a vehicle that must be visible

    @staticmethod
    def preset(name: str) -> "PerceptionProfile":
        return {
            "oracle": PerceptionProfile("oracle"),
            # simulator segmentation + depth: near-perfect geometry
            "segdepth": PerceptionProfile("segdepth", 0.05, 0.005, 0.0, 0.05),
            # deliberately pessimistic synthetic sensitivity profile (not fitted to NanoDet)
            "degraded": PerceptionProfile("degraded", 0.30, 0.06, 0.0, 0.30, 0.02, 0.10, 0.70),
        }[name]


@dataclass
class Box:
    """Axis-aligned vehicle footprint in the lane frame."""
    id: int
    s: float
    d: float
    length: float
    width: float

    def corners(self) -> List[Tuple[float, float]]:
        hl, hw = 0.5 * self.length, 0.5 * self.width
        return [(self.s - hl, self.d - hw), (self.s - hl, self.d + hw),
                (self.s + hl, self.d - hw), (self.s + hl, self.d + hw)]


@dataclass
class CamPose:
    name: str
    s: float
    d: float
    heading: float      # radians in the lane frame
    half_fov: float     # radians
    reach: float        # metres


def cam_pose(name: str, ego_s: float, ego_d: float, ego_yaw: float, range_scale: float = 1.0) -> CamPose:
    c = CAMERAS[name]
    cy, sy = math.cos(ego_yaw), math.sin(ego_yaw)
    return CamPose(name, ego_s + c["x"] * cy - c["y"] * sy, ego_d + c["x"] * sy + c["y"] * cy,
                   ego_yaw + math.radians(c["yaw"]), math.radians(0.5 * c["fov"]), c["range"] * range_scale)


def _bearing_interval(cam: CamPose, pts: Iterable[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
    """(bearing_min, bearing_max, nearest_distance) of a point set as seen from the camera."""
    ch, sh = math.cos(cam.heading), math.sin(cam.heading)
    lo, hi, near, any_front = math.inf, -math.inf, math.inf, False
    for ps, pd in pts:
        rs, rd = ps - cam.s, pd - cam.d
        x, y = rs * ch + rd * sh, -rs * sh + rd * ch
        near = min(near, math.hypot(x, y))
        if x > 0.05:
            any_front = True
            b = math.atan2(y, x)
        else:
            b = math.copysign(0.5 * math.pi, y)
        lo, hi = min(lo, b), max(hi, b)
    return (lo, hi, near) if any_front else None


def _visible_length(iv: Interval, half_fov: float, covers: Sequence[Interval]) -> float:
    """Length of iv inside the FOV that no cover interval hides."""
    lo, hi = max(iv[0], -half_fov), min(iv[1], half_fov)
    if hi <= lo:
        return 0.0
    hidden, cur = 0.0, lo
    for c0, c1 in sorted(covers):
        c0, c1 = max(c0, cur), min(c1, hi)
        if c1 > c0:
            hidden += c1 - c0
            cur = c1
    return (hi - lo) - hidden


def visible_fraction(cam: CamPose, target: Box, occluders: Sequence[Box]) -> Tuple[float, float]:
    """(fraction of the target's angular extent that the camera sees, nearest distance)."""
    t = _bearing_interval(cam, target.corners())
    if t is None:
        return 0.0, math.inf
    lo, hi, near = t
    covers = []
    for o in occluders:
        if o.id == target.id:
            continue
        ob = _bearing_interval(cam, o.corners())
        if ob is not None and ob[2] < near:
            covers.append((ob[0], ob[1]))
    return _visible_length((lo, hi), cam.half_fov, covers) / max(1e-6, hi - lo), near


def lane_visible_extent(cams: Sequence[CamPose], ego_s: float, lane_d: float, direction: int,
                        occluders: Sequence[Box], step: float = 4.0) -> float:
    """How far along a lane (ahead: +1, behind: -1) a car-sized patch of road stays visible."""
    reach = max(c.reach for c in cams)
    extent, seen, dist = 0.0, False, 8.0
    while dist <= reach:
        patch = Box(-1, ego_s + direction * dist, lane_d, 1.0, 1.8)
        ok = False
        for cam in cams:
            frac, near = visible_fraction(cam, patch, occluders)
            if frac >= 0.5 and near <= cam.reach:
                ok = True
                break
        if ok:
            seen, extent = True, dist
        elif seen:
            break
        dist += step
    return extent


class Sensing:
    """Turns ground-truth boxes into noisy, occlusion-aware detections for one ego pose."""

    def __init__(self, profile: PerceptionProfile, rng: random.Random):
        self.p, self.rng = profile, rng

    def poses(self, names: Sequence[str], ego_s: float, ego_d: float, ego_yaw: float) -> List[CamPose]:
        return [cam_pose(n, ego_s, ego_d, ego_yaw, self.p.range_scale) for n in names]

    def sees(self, cams: Sequence[CamPose], target: Box, others: Sequence[Box], dist_for_miss: float) -> bool:
        for cam in cams:
            frac, near = visible_fraction(cam, target, others)
            if frac >= self.p.min_visible and near <= cam.reach:
                p_miss = min(0.9, self.p.miss_base + self.p.miss_per_100m * dist_for_miss / 100.0)
                return self.rng.random() >= p_miss
        return False

    def noisy_gap(self, gap: float) -> float:
        p = self.p
        if p.gap_sigma_abs == 0.0 and p.gap_sigma_rel == 0.0 and p.gap_bias_rel == 0.0:
            return gap
        return max(0.0, gap * (1.0 + p.gap_bias_rel) + self.rng.gauss(0.0, p.gap_sigma_abs + p.gap_sigma_rel * gap))

    def noisy_lat(self, lat: float) -> float:
        return lat if self.p.lat_sigma == 0.0 else lat + self.rng.gauss(0.0, self.p.lat_sigma)


CHANNEL_CAMERAS: Dict[str, Tuple[str, ...]] = {
    "front": ("front_wide", "front_tele"),
    "passing_ahead": ("front_wide", "front_tele", "side_pass"),
    "oncoming": ("front_wide", "front_tele", "side_pass"),
    "rear": ("rq_pass", "side_pass"),
    "side": ("side_pass", "front_wide", "rq_pass"),
    "travel_behind": ("rq_travel", "side_travel"),
    "side_travel": ("side_travel", "front_wide", "rq_travel"),
}
