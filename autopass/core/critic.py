"""The deterministic critic: the only component that can approve a pass.

A proposal is approved only if (1) every piece of gating evidence was sensed recently,
and (2) the hard gates pass on that evidence. The planner (LLM or rule) cannot approve
its own proposal, and nothing an LLM writes into the belief can remove a constraint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .belief import Belief
from .gates import GateConfig, GateInput, GateResult, evaluate  # noqa: F401 (evaluate re-exported)
from .types import FRONT, REAR, Corridor, EgoState

TAU_GAP_S = 0.7        # front-gap evidence older than this is stale (and all evidence in legacy mode)
TAU_HAZARD_S = 0.3     # rear / passing lane / blind spot must be this recent: traffic moves metres per 0.1 s
TAU_CORRIDOR_S = 2.0   # road-graph lookahead changes slowly


@dataclass
class Verdict:
    approved: bool
    reason: str
    stale: List[str]
    gates: Optional[GateResult] = None

    def as_dict(self) -> dict:
        return {"approved": self.approved, "reason": self.reason, "stale": self.stale,
                "gates": self.gates.as_dict() if self.gates else None}


def gate_input(b: Belief, ego: EgoState, speed_limit: float, now: Optional[float] = None) -> GateInput:
    """Project the belief onto exactly the fields the gates consume.

    With `now`, each gap is extrapolated from when it was sensed to the present using its own
    measured rate, and only in the unsafe direction: evidence decays physically.
    """
    front, rear, lane = b.ch(FRONT), b.ch(REAR), b.ch(b.passing_channel)

    def aged(c):
        if c.gap_m is None or c.rate_mps is None or now is None:
            return c.gap_m
        return max(0.0, c.gap_m + min(0.0, c.rate_mps) * max(0.0, now - c.t))

    # Account for quantized depth and delayed rate estimates on a detected rear track.
    rear_gap = aged(rear)
    rear_closing = None if rear.rate_mps is None else -rear.rate_mps
    if now is not None and rear_gap is not None:
        rear_gap = max(0.0, rear_gap - 0.8)
        if rear_closing is not None:
            rear_closing += 0.75

    return GateInput(
        two_lane=b.two_lane, urgency=b.urgency, v_ego=ego.speed, v_cruise=b.cruise_mps,
        v_pass=b.pass_mps, speed_limit=speed_limit,
        front_gap=aged(front), lead_speed=b.lead_speed_mps, lead_kind=front.kind,
        pass_gap=aged(lane),
        pass_speed=None if lane.rate_mps is None else ego.speed + lane.rate_mps,
        pass_range=lane.range_m,
        rear_gap=rear_gap,
        rear_closing=rear_closing,
        rear_range=rear.range_m,
        side_occupied=b.side_occupied,
        corridor_clear=b.corridor_clear_m, corridor_legal=b.corridor_legal,
        semantic_veto=b.semantic_veto,
    )


def stale_channels(b: Belief, now: float, cfg: GateConfig) -> List[str]:
    """Evidence the critic requires that is missing, too old, or sensed in the wrong order."""
    legacy = cfg.mode == "legacy"
    out = []
    if not b.ch(FRONT).fresh(now, TAU_GAP_S):
        out.append(FRONT)
    tau = TAU_GAP_S if legacy else TAU_HAZARD_S
    for name in (b.passing_channel, REAR):
        if not b.ch(name).fresh(now, tau):
            out.append(name)
    if legacy:
        if b.corridor_legal is None:           # first release: corridor checked once, ever
            out.append("corridor")
        return out
    # The blind spot is the shoulder check: it must be the last hazard evidence taken.
    newest = max(b.ch(b.passing_channel).t, b.ch(REAR).t)
    newest_order = max(b.ch(b.passing_channel).order, b.ch(REAR).order)
    if now - b.side_t > tau + 1e-9 or b.side_t + 1e-9 < newest or b.side_order < newest_order:
        out.append("side")
    if now - b.corridor_t > TAU_CORRIDOR_S + 1e-9:
        out.append("corridor")
    return out


def verify(b: Belief, ego: EgoState, now: float, speed_limit: float, cfg: GateConfig,
           corridor_now: Optional[Corridor] = None) -> Verdict:
    stale = stale_channels(b, now, cfg)
    if stale:
        return Verdict(False, "not freshly sensed: " + ", ".join(stale), stale)
    candidate = gate_input(b, ego, speed_limit, now)
    if corridor_now is not None and cfg.mode != "legacy":
        candidate.corridor_clear = min(candidate.corridor_clear, corridor_now.clear_m)
        candidate.corridor_legal = bool(candidate.corridor_legal and corridor_now.legal)
    gates = evaluate(candidate, cfg)
    if not gates.can_pass:
        return Verdict(False, gates.blockers[0], [], gates)
    return Verdict(True, "all gates pass on fresh evidence", [], gates)
