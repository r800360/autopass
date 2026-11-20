"""Closed-form overtake kinematics. The gates and the abort logic both use these,
so every threshold in the critic traces back to a physical quantity.

Model: the lead holds v_lead; the ego accelerates at a_acc from v_ego to v_pass and holds it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PassPlan:
    d_rel: float       # distance the ego must gain on the lead (m)
    t_overtake: float  # commit -> ego clear of the lead by the merge margin (s)
    t_end: float       # commit -> ego back across the lane line (s)
    d_ego: float       # distance the ego covers by t_end (m)

    @property
    def feasible(self) -> bool:
        return math.isfinite(self.t_end)


def ego_distance(t: float, v_ego: float, v_pass: float, a_acc: float) -> float:
    """Distance covered t seconds after commit."""
    t_a = max(0.0, (v_pass - v_ego) / a_acc)
    ta = min(t, t_a)
    return v_ego * ta + 0.5 * a_acc * ta * ta + v_pass * max(0.0, t - t_a)


def time_to_gain(d: float, v_ego: float, v_lead: float, v_pass: float, a_acc: float = 2.5) -> float:
    """Seconds for the ego to gain `d` metres on a lead holding v_lead."""
    if d <= 0.0:
        return 0.0
    b = v_ego - v_lead
    t_a = max(0.0, (v_pass - v_ego) / a_acc)
    gain_a = b * t_a + 0.5 * a_acc * t_a * t_a
    if gain_a >= d:
        return (-b + math.sqrt(max(0.0, b * b + 2.0 * a_acc * d))) / a_acc
    if v_pass - v_lead > 0.3:
        return t_a + (d - gain_a) / (v_pass - v_lead)
    return math.inf


def plan_pass(front_gap: float, v_ego: float, v_lead: float, v_pass: float, *,
              a_acc: float = 2.5, lead_len: float = 5.0, ego_len: float = 4.7,
              merge_margin: float = 8.0, t_return: float = 1.5) -> PassPlan:
    """Time and distance to overtake a lead `front_gap` metres ahead.

    The ego must gain d_rel = front_gap + lead_len + merge_margin + ego_len on the lead:
    from front bumper behind the lead's rear bumper to rear bumper `merge_margin` ahead of
    its front bumper. `t_return` is the extra time to get back across the lane line.
    """
    d_rel = front_gap + lead_len + merge_margin + ego_len
    t = time_to_gain(d_rel, v_ego, v_lead, v_pass, a_acc)
    if not math.isfinite(t):
        return PassPlan(d_rel, math.inf, math.inf, math.inf)
    t_end = t + t_return
    return PassPlan(d_rel, t, t_end, ego_distance(t_end, v_ego, v_pass, a_acc))


def required_follower_decel(gap: float, closing: float, *, a_ego: float = 2.5, t_enter: float = 1.0,
                            t_react: float = 1.0, standstill: float = 5.0) -> float:
    """Deceleration a vehicle behind in the passing lane would need if the ego pulls out now.

    gap: its distance behind the ego; closing: how fast it is catching up (m/s, >0 = closing).
    The ego enters its lane after t_enter while accelerating at a_ego. Returns 0 if the
    follower never has to brake and inf if no braking can avoid contact.
    (The safety criterion of the MOBIL lane-change model.)
    """
    gap_m = gap - closing * t_enter + 0.5 * a_ego * t_enter * t_enter
    c_m = closing - a_ego * t_enter
    if gap_m <= standstill:
        return math.inf
    if c_m <= 0.0:
        return 0.0
    room = gap_m - c_m * t_react - standstill
    return math.inf if room <= 0.0 else c_m * c_m / (2.0 * room)
