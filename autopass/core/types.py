"""Shared value types and the World protocol every simulator backend implements."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

# Maneuver phases (see core/fsm.py).
FOLLOW, LANE_CHANGE, OVERTAKE, MERGE_BACK, ABORT, DONE = (
    "follow", "lane_change", "overtake", "merge_back", "abort", "done")

# Perception channels. Each one is a region of interest served by one camera.
FRONT = "front"                  # travel lane ahead (the lead)          - wide front camera
PASSING_AHEAD = "passing_ahead"  # same-direction passing lane ahead     - wide front camera
ONCOMING = "oncoming"            # opposing lane ahead (two-lane roads)  - telephoto front camera
REAR = "rear"                    # passing lane behind                   - rear camera
TRAVEL_BEHIND = "travel_behind"  # travel lane behind (merge-back cue)   - rear camera
SIDE = "side"                    # alongside, passing side (blind spot)  - passing-side camera
SIDE_TRAVEL = "side_travel"      # alongside, travel side (merge-back)   - travel-side camera
CAMERA_OF = {FRONT: "front_wide", PASSING_AHEAD: "front_wide", ONCOMING: "front_tele",
             REAR: "rq_pass", TRAVEL_BEHIND: "rq_travel", SIDE: "side_pass", SIDE_TRAVEL: "side_travel"}


@dataclass
class Detection:
    """One perceived road user, already associated to a lane.

    gap_m is bumper-to-nearest-surface distance along the road (always >= 0);
    lateral_m is the offset from the travel-lane centre, positive toward the passing side.
    """
    gap_m: float
    lateral_m: float
    kind: str = "car"
    bbox: Optional[Tuple[int, int, int, int]] = None   # pixel box, for overlays only
    track_id: Optional[int] = None                     # sensor-provided instance/track identity


@dataclass
class SenseResult:
    """What one region-of-interest query returns."""
    dets: List[Detection]
    range_m: float                 # reach: with no detection, the region is clear at least this far
    semantic: Dict[str, Any] = field(default_factory=dict)  # e.g. {"blinker": "left"} when available

    @property
    def nearest(self) -> Optional[Detection]:
        return min(self.dets, key=lambda d: d.gap_m) if self.dets else None


@dataclass
class EgoState:
    """Ego state from odometry + localisation against the lane map."""
    speed: float            # m/s
    d: float                # lateral offset from travel-lane centre, + toward passing side (m)
    heading: float          # heading error vs lane direction, + toward passing side (rad)
    length: float = 4.7


@dataclass
class Corridor:
    """Road-graph lookahead on the passing lane."""
    clear_m: float          # distance the passing lane stays a valid, junction-free driving lane
    legal: bool = True      # lane marking permits crossing here


@dataclass
class Control:
    steer: float            # normalised [-1, 1], + toward the passing side
    target_speed: float     # m/s; the backend maps this to throttle/brake


@dataclass
class Actor:
    """Ground truth for one vehicle in the travel-lane frame. Evaluation and rendering only."""
    id: int
    role: str               # ego | lead | rear | oncoming | blocker | side | ambient ...
    s: float                # longitudinal position (m)
    d: float                # lateral position, + toward passing side (m)
    v: float                # signed speed along +s (oncoming traffic is negative)
    yaw: float              # heading relative to lane direction (rad)
    length: float
    width: float
    kind: str = "car"


@dataclass
class Truth:
    """Evaluation-only snapshot. The agent never sees this."""
    t: float
    actors: List[Actor]
    collision: bool = False
    offroad: bool = False

    def get(self, role: str) -> Optional[Actor]:
        for a in self.actors:
            if a.role == role:
                return a
        return None


class World(Protocol):
    """A simulator backend. Implemented by sim.kinematic.KinematicWorld and carla_backend.CarlaWorld."""
    dt: float
    time: float
    lane_width: float
    speed_limit: float
    two_lane: bool

    def ego(self) -> EgoState: ...
    def sense(self, channel: str) -> SenseResult: ...
    def corridor(self) -> Corridor: ...
    def target_point(self, d_target: float, lookahead: float) -> Tuple[float, float]:
        """Lane-centre target `lookahead` metres ahead at lateral offset d_target, in the ego
        frame as (forward, toward-passing-side)."""
        ...
    def apply(self, control: Control) -> None: ...
    def step(self) -> None: ...
    def truth(self) -> Truth: ...
