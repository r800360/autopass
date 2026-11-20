"""Lateral and longitudinal control shared by every backend.

The backend supplies a target in lane coordinates. Waypoint tracking constrains the target;
tracking error and departures are measured independently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# Camera suite, shared by both simulators. x: metres ahead of the ego centre; y: metres toward
# the passing side; yaw: degrees, positive toward the passing side; range: metres at which a
# car is still reliably resolved. The rear-quarter cameras sit on the mirrors so that a
# vehicle directly behind cannot hide the adjacent lane.
CAMERAS = {
    "front_wide":  {"x": 1.6, "y": 0.0,   "yaw": 0.0,    "fov": 90.0,  "range": 85.0},
    "front_tele":  {"x": 1.6, "y": 0.0,   "yaw": 0.0,    "fov": 28.0,  "range": 250.0},
    "rq_pass":     {"x": 0.6, "y": 0.95,  "yaw": 180.0,  "fov": 50.0,  "range": 220.0},
    "rq_travel":   {"x": 0.6, "y": -0.95, "yaw": 180.0,  "fov": 50.0,  "range": 220.0},
    "side_pass":   {"x": 0.0, "y": 0.0,   "yaw": 100.0,  "fov": 140.0, "range": 30.0},
    "side_travel": {"x": 0.0, "y": 0.0,   "yaw": -100.0, "fov": 140.0, "range": 30.0},
}


@dataclass
class ControlConfig:
    lookahead_min_m: float = 6.0
    lookahead_per_mps: float = 0.9
    max_steer: float = 0.75
    heading_gain_deg: float = 34.0     # degrees of heading error that saturate steering at 1.0
    lateral_gain: float = 1.0
    smooth: float = 0.45               # low-pass on the steering command
    follow_gap_min_m: float = 18.0     # cruise-control gap behind the lead
    follow_gap_s: float = 1.5
    emergency_gap_m: float = 6.0
    peek_offset_m: float = 0.8         # how far to shift inside the lane when peeking
    peek_max_s: float = 10.0


def lookahead(speed: float, c: ControlConfig) -> float:
    return max(c.lookahead_min_m, c.lookahead_per_mps * speed)


def steer_toward(dx: float, dy: float, look: float, prev: float, c: ControlConfig) -> float:
    """Heading-plus-lateral law on a target point (dx ahead, dy toward the passing side)."""
    heading_deg = math.degrees(math.atan2(dy, max(0.1, dx)))
    raw = heading_deg / c.heading_gain_deg + c.lateral_gain * math.atan2(dy, max(1.0, look)) / math.pi
    raw = max(-c.max_steer, min(c.max_steer, raw))
    out = (1.0 - c.smooth) * prev + c.smooth * raw
    return max(-c.max_steer, min(c.max_steer, out))


def follow_gap(speed: float, c: ControlConfig) -> float:
    return max(c.follow_gap_min_m, c.follow_gap_s * speed)


def cruise_speed(gap: Optional[float], lead_speed: Optional[float], ego_speed: float, limit: float,
                 c: ControlConfig) -> float:
    """Adaptive cruise: hold `follow_gap` behind whatever is ahead in our lane, never above `limit`."""
    if gap is None:
        return limit
    if gap < c.emergency_gap_m:
        return 0.0
    v_lead = lead_speed if lead_speed is not None else ego_speed
    return max(0.0, min(limit, v_lead + 0.5 * (gap - follow_gap(ego_speed, c))))
