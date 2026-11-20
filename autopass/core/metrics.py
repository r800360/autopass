"""Honest metrics. Everything here is computed from ground truth the agent never sees."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from .gates import GateConfig, GateInput, GateResult, evaluate
from .types import Actor, Corridor, Truth

ALONGSIDE_M = 3.0      # bumper clearance below which a vehicle counts as "alongside"
UNSEEN = 1.0e4         # ground truth has no sensing limit


def lane_of(a: Actor, lane_width: float) -> str:
    if abs(a.d) < 0.5 * lane_width:
        return "travel"
    if 0.5 * lane_width <= a.d < 1.5 * lane_width:
        return "passing"
    return "other"


def truth_gate_input(truth: Truth, corridor: Corridor, *, two_lane: bool, urgency: str, cruise: float,
                     pass_speed: float, speed_limit: float, lane_width: float) -> GateInput:
    """What the gates would read with perfect perception: used to score a commit, not to make it."""
    ego = truth.get("ego")
    x = GateInput(two_lane=two_lane, urgency=urgency, v_ego=ego.v, v_cruise=cruise, v_pass=pass_speed,
                  speed_limit=speed_limit, pass_range=UNSEEN, rear_range=UNSEEN, side_occupied=False,
                  corridor_clear=corridor.clear_m, corridor_legal=corridor.legal)
    for a in truth.actors:
        if a.role == "ego":
            continue
        ahead = (a.s - a.length / 2) - (ego.s + ego.length / 2)
        behind = (ego.s - ego.length / 2) - (a.s + a.length / 2)
        lane = lane_of(a, lane_width)
        if lane == "travel" and ahead >= 0 and (x.front_gap is None or ahead < x.front_gap):
            x.front_gap, x.lead_speed, x.lead_kind = ahead, a.v, a.kind
        elif lane == "passing":
            if ahead >= ALONGSIDE_M:
                if x.pass_gap is None or ahead < x.pass_gap:
                    x.pass_gap, x.pass_speed = ahead, a.v
            elif behind >= ALONGSIDE_M:
                if x.rear_gap is None or behind < x.rear_gap:
                    x.rear_gap, x.rear_closing = behind, a.v - ego.v
            else:
                x.side_occupied = True
    return x


def truth_gates(truth: Truth, corridor: Corridor, cfg: Optional[GateConfig] = None, **ctx) -> GateResult:
    return evaluate(truth_gate_input(truth, corridor, **ctx), cfg or GateConfig())


@dataclass
class Metrics:
    lane_width: float = 3.5
    speeds: List[float] = field(default_factory=list)
    collision: bool = False
    offroad: bool = False
    max_lane_dev: float = 0.0
    cleared_lead: bool = False
    time_to_clear: Optional[float] = None
    min_gap: float = math.inf          # closest bumper approach to a vehicle in the ego's path
    min_ttc: float = math.inf
    max_induced_decel: float = 0.0     # hardest braking forced on a reacting follower (m/s^2)
    distance: float = 0.0
    time_to_80m: Optional[float] = None
    max_overshoot: float = 0.0         # farthest excursion beyond either lane centre during the manoeuvre
    _s0: Optional[float] = None
    _settled: bool = False             # ego has reached the current lateral target
    _last_key: Optional[tuple] = None

    def update(self, truth: Truth, phase: str, d_target: float, induced_decel: float = 0.0) -> None:
        ego = truth.get("ego")
        if self._s0 is None:
            self._s0 = ego.s
        self.distance = ego.s - self._s0
        if self.distance >= 80.0 and self.time_to_80m is None:
            self.time_to_80m = round(truth.t, 2)
        self.speeds.append(ego.v)
        self.collision = self.collision or truth.collision
        self.offroad = self.offroad or truth.offroad
        # Lane keeping is a steady-state property: after the phase or the lateral target changes,
        # score it only once the ego has settled on the new target (error first below 0.15 m), so
        # transitions do not masquerade as tracking error.
        key, err = (phase, round(d_target, 2)), abs(ego.d - d_target)
        if key != self._last_key:
            self._last_key, self._settled = key, False
        self._settled = self._settled or err < 0.15
        if phase in ("follow", "done") and self._settled:
            self.max_lane_dev = max(self.max_lane_dev, err)
        self.max_overshoot = max(self.max_overshoot, ego.d - self.lane_width, -ego.d)
        self.max_induced_decel = max(self.max_induced_decel, induced_decel)
        lead = truth.get("lead")
        if lead is not None and not self.cleared_lead and ego.s - lead.s > 0.5 * (ego.length + lead.length) + 3.0:
            self.cleared_lead, self.time_to_clear = True, round(truth.t, 2)
        for a in truth.actors:
            if a.role == "ego" or abs(a.d - ego.d) > 0.5 * (a.width + ego.width) + 0.3:
                continue
            gap = abs(a.s - ego.s) - 0.5 * (a.length + ego.length)
            self.min_gap = min(self.min_gap, gap)
            closing = (ego.v - a.v) if a.s > ego.s else (a.v - ego.v)
            if closing > 0.1 and gap > 0:
                self.min_ttc = min(self.min_ttc, gap / closing)

    @property
    def mean_speed(self) -> float:
        return sum(self.speeds) / max(1, len(self.speeds))
