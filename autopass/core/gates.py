"""Hard safety gates. Deterministic, and they can only block a pass, never force one.

Two gate sets share one input type:

  physics  (default)  every threshold is derived from overtake kinematics and an explicit
                      assumption about other road users (reaction time, comfortable braking,
                      speed limit). "Nothing detected" means "clear up to the sensing reach",
                      never "clear".
  legacy              the fixed thresholds of the first release (18 / 12 / 45 / 24 m, lead
                      4 m/s slower than cruise). Kept so the two can be compared on the same
                      scenarios; see docs/EVALUATION.md.

Gates come in two kinds. A failed *warrant* gate means the pass is not worth it; a failed
*hazard* gate means it is unsafe. The benchmark counts them separately.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .kinematics import PassPlan, plan_pass, required_follower_decel

WARRANT_GATES = ("front_room", "warrant")
HAZARD_GATES = ("rear", "blind_spot", "passing_lane", "corridor", "semantic")


@dataclass
class GateConfig:
    mode: str = "physics"                      # physics | legacy
    # --- physics -----------------------------------------------------------------
    min_front_gap_m: float = 12.0              # room to pull out without clipping the lead
    warrant_slowdown: Dict[str, float] = field(   # lead must be this fraction slower than cruise
        default_factory=lambda: {"low": 0.60, "medium": 0.35, "high": 0.20})
    a_acc: float = 2.5                         # ego acceleration while passing (m/s^2)
    merge_margin_m: float = 8.0                # clearance ahead of the lead before merging back
    ego_len_m: float = 4.7
    lead_len_m: Dict[str, float] = field(default_factory=lambda: {"car": 5.0, "truck": 10.0, "bus": 12.0})
    t_enter_s: float = 1.0                     # commit -> ego starts occupying the passing lane
    t_return_s: float = 1.5                    # merge trigger -> ego back across the lane line
    t_react_s: float = 1.0                     # reaction time granted to a following driver
    standstill_m: float = 5.0                  # bumper gap that must always remain
    max_follower_decel: float = 2.0            # braking we may impose on a follower (m/s^2)
    min_rear_gap_m: float = 12.0             # floor even if a short-range track currently opens
    rear_headway_s: float = 1.0                # time gap a follower must keep once we are in its lane
    overspeed: float = 1.1                     # unseen traffic may travel at overspeed x limit
    oncoming_margin_s: float = 1.0             # time buffer at closing speed when back in lane
    corridor_margin_m: float = 10.0
    pull_in_gap_m: float = 8.0                 # room behind a faster vehicle ahead in the passing lane
    # --- legacy (first release) --------------------------------------------------
    legacy_front_m: float = 18.0
    legacy_delta_mps: float = 4.0
    legacy_rear_m: float = 12.0
    legacy_oncoming_m: float = 45.0
    legacy_pass_ahead_m: float = 24.0
    legacy_corridor_m: float = 30.0

    @staticmethod
    def legacy() -> "GateConfig":
        return GateConfig(mode="legacy")


@dataclass
class GateInput:
    """Everything the gates read. Built from the belief (decisions) or from ground truth (metrics)."""
    two_lane: bool
    urgency: str
    v_ego: float
    v_cruise: float
    v_pass: float
    speed_limit: float
    front_gap: Optional[float] = None
    lead_speed: Optional[float] = None
    lead_kind: Optional[str] = None
    pass_gap: Optional[float] = None       # nearest object ahead in the passing/oncoming lane
    pass_speed: Optional[float] = None     # its signed speed along our direction (oncoming < 0)
    pass_range: float = 0.0
    rear_gap: Optional[float] = None       # nearest object behind in the passing lane
    rear_closing: Optional[float] = None   # m/s, > 0 = catching up
    rear_range: float = 0.0
    side_occupied: Optional[bool] = None
    corridor_clear: float = 0.0
    corridor_legal: Optional[bool] = None
    semantic_veto: Optional[str] = None


@dataclass
class GateResult:
    checks: Dict[str, bool]
    blockers: List[str]
    plan: Optional[PassPlan] = None
    detail: Dict[str, float] = field(default_factory=dict)

    @property
    def warrant_ok(self) -> bool:
        return all(self.checks.get(g, True) for g in WARRANT_GATES)

    @property
    def hazard_ok(self) -> bool:
        return all(self.checks.get(g, True) for g in HAZARD_GATES)

    @property
    def can_pass(self) -> bool:
        return all(self.checks.values())

    def as_dict(self) -> dict:
        return {"checks": dict(self.checks), "blockers": list(self.blockers), "can_pass": self.can_pass,
                "warrant_ok": self.warrant_ok, "hazard_ok": self.hazard_ok,
                "detail": {k: (None if not math.isfinite(v) else round(v, 2)) for k, v in self.detail.items()}}


def evaluate(x: GateInput, cfg: GateConfig) -> GateResult:
    return _legacy(x, cfg) if cfg.mode == "legacy" else _physics(x, cfg)


# ---------------------------------------------------------------------------------------
def _physics(x: GateInput, c: GateConfig) -> GateResult:
    checks: Dict[str, bool] = {}
    why: List[str] = []
    detail: Dict[str, float] = {}

    # -- warrant: is there a lead, is it slow enough to be worth it, is there room to pull out
    if x.front_gap is None:
        checks["front_room"] = False
        why.append("no lead in view: nothing to overtake")
    else:
        checks["front_room"] = x.front_gap >= c.min_front_gap_m
        if not checks["front_room"]:
            why.append(f"front gap {x.front_gap:.1f} m < {c.min_front_gap_m:.0f} m: no room to pull out")

    need = c.warrant_slowdown.get(x.urgency, c.warrant_slowdown["high"])
    if x.lead_speed is None:
        checks["warrant"] = False
        why.append("lead speed not yet observed")
    else:
        slowdown = (x.v_cruise - x.lead_speed) / max(0.1, x.v_cruise)
        detail["slowdown"] = slowdown
        checks["warrant"] = slowdown >= need
        if not checks["warrant"]:
            why.append(f"lead only {slowdown:.0%} below cruise; {x.urgency} urgency needs {need:.0%}")

    # -- the manoeuvre itself (needs a lead and its speed)
    plan: Optional[PassPlan] = None
    if x.front_gap is not None and x.lead_speed is not None:
        plan = plan_pass(x.front_gap, x.v_ego, x.lead_speed, x.v_pass, a_acc=c.a_acc,
                         lead_len=c.lead_len_m.get(x.lead_kind or "car", 5.0), ego_len=c.ego_len_m,
                         merge_margin=c.merge_margin_m, t_return=c.t_return_s)
        detail.update(t_end=plan.t_end, d_ego=plan.d_ego)
    v_other = c.overspeed * x.speed_limit           # worst case for traffic we cannot see

    # -- rear: would a follower in the passing lane have to brake harder than is reasonable?
    if x.rear_gap is None:
        gap, closing = x.rear_range, max(0.0, v_other - x.v_ego)
        label = f"unseen traffic beyond {gap:.0f} m"
    else:
        gap = x.rear_gap
        closing = x.rear_closing if x.rear_closing is not None else max(0.0, v_other - x.v_ego)
        label = f"vehicle {gap:.1f} m behind closing at {closing:.1f} m/s" + (
            "" if x.rear_closing is not None else " (rate unknown, worst case)")
    decel = required_follower_decel(gap, closing, a_ego=c.a_acc, t_enter=c.t_enter_s,
                                    t_react=c.t_react_s, standstill=c.standstill_m)
    # ... and once we are in its lane, does it still have a time gap to us?
    gap_entry = gap - closing * c.t_enter_s + 0.5 * c.a_acc * c.t_enter_s ** 2
    headway_need = c.standstill_m + c.rear_headway_s * max(0.0, x.v_ego + closing)
    detail.update(follower_decel=decel, rear_gap_at_entry=gap_entry, rear_gap_needed=headway_need)
    checks["rear"] = gap >= c.min_rear_gap_m and decel <= c.max_follower_decel and gap_entry >= headway_need
    if not checks["rear"]:
        if gap < c.min_rear_gap_m:
            why.append(f"rear: {label} within {c.min_rear_gap_m:.0f} m minimum")
        elif decel > c.max_follower_decel:
            need_txt = "cannot avoid contact" if not math.isfinite(decel) else f"would need {decel:.1f} m/s^2 braking"
            why.append(f"rear: {label} {need_txt} (limit {c.max_follower_decel:.1f})")
        else:
            why.append(f"rear: {label} would be left {gap_entry:.0f} m, needs {headway_need:.0f} m")

    # -- blind spot: must be positively observed empty
    checks["blind_spot"] = x.side_occupied is False
    if not checks["blind_spot"]:
        why.append("blind spot occupied" if x.side_occupied else "blind spot not observed")

    # -- passing lane ahead
    if plan is None or not plan.feasible:
        checks["passing_lane"] = False
        if plan is not None:
            why.append("pass speed does not exceed lead speed: overtake cannot complete")
    elif x.two_lane:
        # Oncoming: by the time we are back in lane, the oncoming vehicle (seen, or assumed
        # just beyond what we can see) must still be a time margin away.
        clear = x.pass_gap if x.pass_gap is not None else x.pass_range
        if x.pass_gap is not None and x.pass_speed is not None:
            v_onc = max(0.0, -x.pass_speed)
        else:
            v_onc = v_other
        need_m = plan.d_ego + v_onc * plan.t_end + c.oncoming_margin_s * (x.v_pass + v_onc)
        detail.update(sight_need_m=need_m, sight_have_m=clear)
        checks["passing_lane"] = clear >= need_m
        if not checks["passing_lane"]:
            what = f"oncoming at {clear:.0f} m" if x.pass_gap is not None else f"sight distance {clear:.0f} m"
            why.append(f"{what} < {need_m:.0f} m needed to complete the pass")
    else:
        ok = True
        if x.pass_gap is None:
            need = max(c.pull_in_gap_m, 1.5 * x.v_ego + 6.0)
            ok = x.pass_range >= need
            if not ok:
                why.append(f"passing lane sight distance {x.pass_range:.0f} m < {need:.0f} m needed to pull in")
        else:
            u = x.pass_speed if x.pass_speed is not None else 0.0     # unknown rate: assume stopped
            slower_than_us = u < (x.lead_speed or 0.0) + 1.0
            reach = x.pass_gap + u * plan.t_end - plan.d_ego           # room left when we are back in lane
            if slower_than_us and reach < c.merge_margin_m:
                ok = False
                why.append(f"passing lane blocked {x.pass_gap:.0f} m ahead")
            elif x.pass_gap < max(c.pull_in_gap_m, 1.5 * (x.v_ego - u) + 6.0):
                ok = False
                why.append(f"vehicle {x.pass_gap:.0f} m ahead in passing lane: no room to pull in")
        checks["passing_lane"] = ok

    # -- corridor: the passing lane must stay valid for the whole manoeuvre, and crossing must be legal
    need_c = (plan.d_ego if plan and plan.feasible else 60.0) + c.corridor_margin_m
    detail["corridor_need_m"] = need_c
    checks["corridor"] = x.corridor_legal is True and x.corridor_clear >= need_c
    if not checks["corridor"]:
        if x.corridor_legal is None:
            why.append("corridor not checked")
        elif not x.corridor_legal:
            why.append("lane marking does not permit crossing here")
        else:
            why.append(f"passing lane valid for {x.corridor_clear:.0f} m < {need_c:.0f} m needed")

    # -- semantic veto (e.g. lead signalling a turn, emergency vehicle): can only block
    checks["semantic"] = not x.semantic_veto
    if x.semantic_veto:
        why.append(f"semantic veto: {x.semantic_veto}")

    return GateResult(checks, why, plan, detail)


# ---------------------------------------------------------------------------------------
def _legacy(x: GateInput, c: GateConfig) -> GateResult:
    """The first release's gates: fixed distances, 'not detected' treated as clear, no blind spot."""
    checks: Dict[str, bool] = {}
    why: List[str] = []
    checks["front_room"] = x.front_gap is not None and x.front_gap >= c.legacy_front_m
    if not checks["front_room"]:
        why.append("front gap unmeasured" if x.front_gap is None
                   else f"front gap {x.front_gap:.1f}m < {c.legacy_front_m:.0f}m")
    delta = None if x.lead_speed is None else x.v_cruise - x.lead_speed
    checks["warrant"] = delta is not None and delta >= c.legacy_delta_mps and x.urgency != "low"
    if not checks["warrant"]:
        why.append("lead speed unknown" if delta is None else
                   ("low urgency" if x.urgency == "low" else
                    f"lead not slow enough: delta={delta:.1f} m/s < {c.legacy_delta_mps:.1f} m/s"))
    checks["rear"] = x.rear_gap is None or x.rear_gap >= c.legacy_rear_m
    if not checks["rear"]:
        why.append(f"passing-lane rear gap {x.rear_gap:.1f}m < {c.legacy_rear_m:.0f}m")
    if x.two_lane:
        checks["passing_lane"] = x.pass_gap is None or x.pass_gap >= c.legacy_oncoming_m
        if not checks["passing_lane"]:
            why.append(f"oncoming {x.pass_gap:.1f}m < {c.legacy_oncoming_m:.0f}m")
    else:
        checks["passing_lane"] = x.pass_gap is None or x.pass_gap >= c.legacy_pass_ahead_m
        if not checks["passing_lane"]:
            why.append(f"passing lane blocked ahead ({x.pass_gap:.1f}m)")
    checks["corridor"] = x.corridor_clear >= c.legacy_corridor_m
    if not checks["corridor"]:
        why.append("corridor geometry not verified")
    return GateResult(checks, why)
