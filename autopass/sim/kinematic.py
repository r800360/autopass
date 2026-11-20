"""Kinematic world: a straight two-lane strip, a bicycle-model ego, scripted traffic.

Lane frame: s runs along the road in the ego's direction of travel, d is lateral with the
travel-lane centre at 0 and the passing-lane centre at +lane_width. On a two-lane road the
passing lane carries oncoming traffic. Left/right passing is the same problem mirrored, so
the simulator does not distinguish them.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..core.metrics import ALONGSIDE_M
from ..core.scenario import NPC, Scenario
from ..core.types import (FRONT, ONCOMING, PASSING_AHEAD, REAR, SIDE, SIDE_TRAVEL, TRAVEL_BEHIND, Actor,
                          Control, Corridor, Detection, EgoState, SenseResult, Truth)
from .sensors import CHANNEL_CAMERAS, Box, PerceptionProfile, Sensing, lane_visible_extent

DIMS = {"car": (4.7, 1.9), "truck": (10.0, 2.5), "bus": (12.0, 2.5)}
WHEELBASE = 2.8
MAX_WHEEL_RAD = 0.60          # wheel angle at steer = 1.0
WHEEL_RATE = 1.2              # rad/s
MAX_LAT_ACC = 4.0             # m/s^2, tyre/comfort limit on lateral acceleration
ACC_MAX, BRAKE_MAX = 3.0, 7.0


@dataclass
class Veh:
    id: int
    role: str
    s: float
    d: float
    v: float                  # speed along its own heading
    direction: int            # +1 with the ego, -1 oncoming
    length: float
    width: float
    kind: str = "car"
    behavior: str = "cruise"
    v0: float = 0.0           # desired speed
    appear_s: float = 0.0
    active: bool = True
    s_rel0: float = 0.0
    semantic: Optional[str] = None

    def box(self) -> Box:
        return Box(self.id, self.s, self.d, self.length, self.width)


class KinematicWorld:
    def __init__(self, scn: Scenario, *, perception: Optional[PerceptionProfile] = None, seed: int = 0,
                 dt: float = 0.05, corridor_m: float = 1.0e4, crossing_legal: bool = True):
        self.scn, self.dt, self.time = scn, dt, 0.0
        self.lane_width, self.speed_limit, self.two_lane = scn.lane_width, scn.speed_limit, scn.two_lane
        self.rng = random.Random(seed)
        self.sensing = Sensing(perception or PerceptionProfile(), self.rng)
        self.corridor_m, self.crossing_legal = corridor_m, crossing_legal
        lead = scn.lead()
        v0 = min(scn.cruise_mps, lead.v) if lead else scn.cruise_mps
        self.es, self.ed, self.eyaw, self.ev = 0.0, 0.0, 0.0, v0
        self.elen, self.ewid = DIMS["car"]
        self.wheel = 0.0
        self._cmd = Control(0.0, v0)
        self.collision = False
        self.induced_decel = 0.0
        self.vehs: List[Veh] = []
        for i, n in enumerate(scn.npcs):
            length, width = DIMS.get(n.kind, DIMS["car"])
            self.vehs.append(Veh(id=i + 1, role=n.role, s=n.s, d=(0.0 if n.lane == "travel" else self.lane_width),
                                 v=n.v, direction=n.direction, length=length, width=width, kind=n.kind,
                                 behavior=n.behavior, v0=n.v, appear_s=n.appear_s, active=n.appear_s <= 0.0,
                                 s_rel0=n.s, semantic=n.semantic))

    # -- World protocol -----------------------------------------------------------------
    def ego(self) -> EgoState:
        return EgoState(speed=self.ev, d=self.ed, heading=self.eyaw, length=self.elen)

    def corridor(self) -> Corridor:
        return Corridor(clear_m=max(0.0, self.corridor_m - self.es), legal=self.crossing_legal)

    def target_point(self, d_target: float, lookahead: float) -> Tuple[float, float]:
        c, s = math.cos(self.eyaw), math.sin(self.eyaw)
        rd = d_target - self.ed
        return lookahead * c + rd * s, -lookahead * s + rd * c

    def apply(self, control: Control) -> None:
        self._cmd = control

    def step(self) -> None:
        dt = self.dt
        # ego: bicycle model with a wheel-rate limit and a lateral-acceleration cap
        want = max(-1.0, min(1.0, self._cmd.steer)) * MAX_WHEEL_RAD
        self.wheel += max(-WHEEL_RATE * dt, min(WHEEL_RATE * dt, want - self.wheel))
        cap = math.atan(MAX_LAT_ACC * WHEELBASE / max(1.0, self.ev * self.ev))
        wheel = max(-cap, min(cap, self.wheel))
        acc = max(-BRAKE_MAX, min(ACC_MAX, 1.2 * (self._cmd.target_speed - self.ev)))
        self.ev = max(0.0, self.ev + acc * dt)
        self.es += self.ev * math.cos(self.eyaw) * dt
        self.ed += self.ev * math.sin(self.eyaw) * dt
        self.eyaw += self.ev / WHEELBASE * math.tan(wheel) * dt
        # traffic
        self.induced_decel = 0.0
        for v in self.vehs:
            if not v.active:
                if self.time + dt >= v.appear_s:
                    v.active, v.s = True, self.es + v.s_rel0
                continue
            if v.behavior == "idm":
                a = self._idm(v)
                v.v = max(0.0, v.v + a * dt)
            v.s += v.direction * v.v * dt
        self.time += dt
        if not self.collision:
            self.collision = any(v.active and self._overlap(v) for v in self.vehs)

    def truth(self) -> Truth:
        actors = [Actor(0, "ego", self.es, self.ed, self.ev, self.eyaw, self.elen, self.ewid)]
        for v in self.vehs:
            if v.active:
                actors.append(Actor(v.id, v.role, v.s, v.d, v.direction * v.v,
                                    0.0 if v.direction > 0 else math.pi, v.length, v.width, v.kind))
        offroad = self.ed < -0.5 * self.lane_width - 0.9 or self.ed > 1.5 * self.lane_width + 0.9
        return Truth(self.time, actors, self.collision, offroad)

    # -- perception ---------------------------------------------------------------------
    def sense(self, channel: str) -> SenseResult:
        w = self.lane_width
        cams = self.sensing.poses(CHANNEL_CAMERAS[channel], self.es, self.ed, self.eyaw)
        boxes = [v.box() for v in self.vehs if v.active]
        dets: List[Detection] = []
        for v in self.vehs:
            if not v.active:
                continue
            ahead = (v.s - v.length / 2) - (self.es + self.elen / 2)
            behind = (self.es - self.elen / 2) - (v.s + v.length / 2)
            lat = self.sensing.noisy_lat(v.d)
            lane = "travel" if abs(lat) < 0.5 * w else ("passing" if 0.5 * w <= lat < 1.5 * w else "other")
            if channel == FRONT:
                ok, gap = lane == "travel" and ahead >= 0.0, ahead
            elif channel in (PASSING_AHEAD, ONCOMING):
                ok, gap = lane == "passing" and ahead >= ALONGSIDE_M, ahead
            elif channel == REAR:
                ok, gap = lane == "passing" and behind >= ALONGSIDE_M, behind
            elif channel == SIDE:
                ok, gap = lane == "passing" and ahead < ALONGSIDE_M and behind < ALONGSIDE_M, 0.0
            elif channel == TRAVEL_BEHIND:
                ok, gap = lane == "travel" and behind >= ALONGSIDE_M, behind
            elif channel == SIDE_TRAVEL:
                ok, gap = lane == "travel" and ahead < ALONGSIDE_M and behind < ALONGSIDE_M, 0.0
            else:
                raise ValueError(channel)
            if ok and self.sensing.sees(cams, v.box(), boxes, max(ahead, behind, 0.0)):
                dets.append(Detection(gap_m=self.sensing.noisy_gap(gap), lateral_m=lat, kind=v.kind, track_id=v.id))
        sem = {}
        if channel == FRONT and dets:
            lead = min(dets, key=lambda d: d.gap_m)
            sem_src = next((v.semantic for v in self.vehs if v.id == lead.track_id), None)
            if sem_src:
                sem["lead"] = sem_src
        return SenseResult(dets, self._reach(channel, cams, boxes), sem)

    def _reach(self, channel: str, cams, boxes) -> float:
        w = self.lane_width
        if channel == FRONT:
            return cams[0].reach
        if channel in (PASSING_AHEAD, ONCOMING):
            occ = [b for b in boxes if abs(b.d - w) > 0.5 * w]         # everything outside the passing lane
            return lane_visible_extent(cams[:2], self.es, w, +1, occ)
        if channel == REAR:
            occ = [b for b in boxes if abs(b.d - w) > 0.5 * w]
            return lane_visible_extent(cams[:1], self.es, w, -1, occ)
        if channel == TRAVEL_BEHIND:
            return cams[0].reach
        return 6.0

    # -- helpers ------------------------------------------------------------------------
    def _idm(self, me: Veh) -> float:
        """Intelligent Driver Model: this vehicle reacts to whatever is ahead in its lane."""
        gap, dv, lead_is_ego = math.inf, 0.0, False
        cands = [(v.s, v.direction * v.v, v.length, v.d, v.width, False) for v in self.vehs if v.active and v is not me]
        cands.append((self.es, self.ev, self.elen, self.ed, self.ewid, True))
        for s, vel, length, d, width, is_ego in cands:
            if abs(d - me.d) > 0.5 * (width + me.width):
                continue
            g = (s - me.s) * me.direction - 0.5 * (length + me.length)
            if 0.0 <= g < gap:
                gap, dv, lead_is_ego = g, me.v - vel * me.direction, is_ego
        a_max, b, T, s0 = 1.5, 2.0, 1.4, 2.0
        free = 1.0 - (me.v / max(0.5, me.v0)) ** 4
        if math.isinf(gap):
            return a_max * free
        s_star = s0 + max(0.0, me.v * T + me.v * dv / (2.0 * math.sqrt(a_max * b)))
        a = max(-9.0, a_max * (free - (s_star / max(0.3, gap)) ** 2))
        if lead_is_ego and a < 0.0:
            self.induced_decel = max(self.induced_decel, -a)
        return a

    def _overlap(self, v: Veh) -> bool:
        """Separating-axis test between the (rotated) ego box and an axis-aligned vehicle box."""
        c, s = math.cos(self.eyaw), math.sin(self.eyaw)
        hl, hw = 0.5 * self.elen, 0.5 * self.ewid
        ego = [(self.es + c * a - s * b, self.ed + s * a + c * b) for a, b in ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw))]
        oth = v.box().corners()
        for ax, ay in ((1.0, 0.0), (0.0, 1.0), (c, s), (-s, c)):
            pe = [x * ax + y * ay for x, y in ego]
            po = [x * ax + y * ay for x, y in oth]
            if max(pe) < min(po) or max(po) < min(pe):
                return False
        return True


def run(scn: Scenario, policy=None, *, planner=None, direct=None, seed: int = 0,
        perception: Optional[PerceptionProfile] = None, record: bool = False, **world_kw):
    """Convenience: run one scenario in the kinematic world. Returns (result, recorder_or_None)."""
    from ..core.episode import Episode, Policy
    from ..core.telemetry import Recorder
    world = KinematicWorld(scn, perception=perception, seed=seed, **world_kw)
    policy = policy or Policy()
    rec = Recorder({"scenario": scn.to_dict(), "backend": "kinematic", "seed": seed,
                    "lane_width": scn.lane_width, "policy": policy.name}) if record else None
    result = Episode(world, scn, policy, planner=planner, direct=direct, recorder=rec).run()
    result.update(backend="kinematic", seed=seed, perception=world.sensing.p.name)
    return result, rec
