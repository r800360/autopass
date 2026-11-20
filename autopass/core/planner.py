"""Planners: who chooses the next tool.

  RulePlanner   deterministic. Refreshes whatever the critic would call stale, then proposes.
  LLMPlanner    an LLM chooses. Two styles:
                  react  one tool per call, sees each observation before choosing again
                  plan   one call per cycle returns the whole tool sequence
A planner only chooses tools. It can never approve a pass; the critic does that.
The LLM call is injected (`complete`), so this module stays free of any SDK.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .tools import SENSING, TOOLS

# channel name (as the critic reports it) -> the tool that refreshes it
REFRESH = {"front": "sense_front", "passing_ahead": "sense_passing_lane", "oncoming": "sense_passing_lane",
           "rear": "sense_rear", "side": "sense_blind_spot", "corridor": "check_corridor"}


# words in a critic rejection -> the tool that re-examines that evidence (first match wins)
REJECTION_HINTS = (("lead speed", "sense_front"), ("front gap", "sense_front"), ("no lead", "sense_front"),
                   ("rear:", "sense_rear"), ("rear gap", "sense_rear"), ("blind spot", "sense_blind_spot"),
                   ("sight distance", "sense_passing_lane"), ("oncoming", "sense_passing_lane"),
                   ("passing lane", "sense_passing_lane"), ("corridor", "check_corridor"))


@dataclass
class View:
    """Everything a planner is allowed to see."""
    snapshot: Dict[str, Any]          # belief snapshot
    stale: List[str]                  # channels the critic would reject as missing or old, right now
    two_lane: bool
    urgency: str
    allowed: Tuple[str, ...]          # tool palette
    called: List[str]                 # tools already run this cycle
    observations: List[str]           # their observations, same order
    last_rejection: Optional[str]
    budget_left: int


@dataclass
class Step:
    tools: List[str]
    rationale: str = ""
    latency_s: float = 0.0            # wall-clock time the planner took (0 for rules)
    info: Dict[str, Any] = field(default_factory=dict)


class RulePlanner:
    """Refresh what is stale, peek if sight distance was the blocker, then propose."""
    kind = "rule"

    def next(self, v: View) -> Step:
        if "propose_pass" in v.called:
            return Step(["hold"], "critic rejected; gather fresh evidence next cycle")
        todo: List[str] = []
        rej = (v.last_rejection or "").lower()
        if "peek" in v.allowed and v.two_lane and "sight distance" in rej and not v.snapshot.get("peek"):
            todo.append("peek")
        # Look again at whatever blocked the last proposal: that is the evidence that has to change.
        for key, tool in REJECTION_HINTS:
            if key in rej and tool in v.allowed and tool not in v.called and tool not in todo:
                todo.append(tool)
                break
        for ch in v.stale:
            tool = REFRESH[ch]
            if tool in v.allowed and tool not in v.called and tool not in todo:
                todo.append(tool)
        if "sense_blind_spot" in v.allowed and "sense_blind_spot" not in v.called and any(
                t in todo for t in ("sense_rear", "sense_passing_lane", "sense_blind_spot")):
            todo = [t for t in todo if t != "sense_blind_spot"] + ["sense_blind_spot"]   # shoulder check last
        todo.append("propose_pass")
        return Step(todo[:max(1, v.budget_left)], "refresh stale evidence, then let the critic rule")


SYSTEM_PROMPT = """You are the planner of an autonomous car's overtaking agent. The car is following a slower lead vehicle.
Your only job is to choose which tools to run next. You cannot approve or execute a pass: a separate deterministic safety critic does that, and it only approves when
  1. front was sensed within 0.7 s; passing lane, rear and blind spot within 0.3 s, with the blind spot
     checked last (a shoulder check); and the corridor within 2 s, and
  2. hard physical gates pass (room to pull out, lead slow enough to be worth passing, no vehicle closing from behind that would have to brake hard, blind spot empty, passing lane clear for the whole manoeuvre, legal to cross).
Guidelines:
  - Sense what is listed as stale or missing; do not re-sense what is already fresh.
  - A closing speed needs two looks at the same vehicle about half a second apart, so a first look can be rejected as "rate unknown". Hold and look again next cycle.
  - On a two-lane road the lead blocks your view of the oncoming lane. If the critic says sight distance is too short, call peek, then sense the passing lane again once settled.
  - When alone or visibility is incomplete, request the needed evidence and propose for the critic to decide.
  - A proposal is a request to evaluate evidence, never a direct instruction to change lanes.
  - In step mode, choose an available sensing tool if evidence is missing.
  - Choose hold when the evidence already shows a hazard (blind spot occupied, vehicle closing fast behind, oncoming vehicle near)."""


def _menu(allowed: Tuple[str, ...]) -> str:
    return "\n".join(f"- {t}: {TOOLS[t]}" for t in allowed)


def build_user_prompt(v: View, style: str) -> str:
    ask = ("Return ordered probes and a required terminal decision. In a clear scene use any necessary probes and choose propose_pass."
           if style == "plan" else "Return the single next tool.")
    return (f"Road: {'two-lane, passing uses the ONCOMING lane' if v.two_lane else 'multi-lane, same-direction passing lane'}\n"
            f"Trip urgency: {v.urgency}\n"
            f"Tools:\n{_menu(v.allowed)}\n\n"
            f"Belief (gaps in m, rates in m/s with negative = closing, ages in s):\n{json.dumps(v.snapshot)}\n"
            f"Stale or missing right now: {v.stale or 'nothing'}\n"
            f"Already run this cycle: {v.called or 'nothing'}\n"
            f"Observations this cycle: {v.observations or 'none'}\n"
            f"Last critic rejection: {v.last_rejection or 'none'}\n"
            f"Tool calls left this cycle: {v.budget_left}\n\n{ask} Give a one-sentence rationale.")


def schema(style: str, allowed: Tuple[str, ...]) -> Dict[str, Any]:
    tool = {"type": "string", "enum": list(allowed)}
    if style == "plan":
        probes = [name for name in allowed if name not in ("propose_pass", "hold")]
        props = {"probes": {"type": "array", "items": {"type": "string", "enum": probes}},
                 "decision": {"type": "string", "enum": ["propose_pass", "hold"]}}
    else:
        props = {"tool": tool}
    props["rationale"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


Complete = Callable[[str, str, Dict[str, Any]], Tuple[Dict[str, Any], Dict[str, Any]]]


class LLMPlanner:
    """LLM tool selection. Transport errors produce a traceable hold."""

    def __init__(self, complete: Complete, style: str = "plan"):
        assert style in ("plan", "react")
        self.complete, self.style, self.kind = complete, style, f"llm_{style}"

    def next(self, v: View) -> Step:
        t0 = time.perf_counter()
        try:
            allowed = tuple(t for t in v.allowed if t not in v.called and (v.two_lane or t != "peek"))
            obj, info = self.complete(SYSTEM_PROMPT, build_user_prompt(v, self.style), schema(self.style, allowed))
            if self.style == "plan":
                probes = obj.get("probes") or []
                probes = list(dict.fromkeys(t for t in probes if t in allowed and t not in ("hold", "propose_pass")))
                decision = obj.get("decision")
                if decision not in ("hold", "propose_pass"):
                    raise ValueError("missing terminal decision")
                tools = probes[:max(0, v.budget_left-1)] + [decision]
            else:
                tools = [obj["tool"]] if obj.get("tool") in allowed else []
            if not tools:
                raise ValueError("no valid tool in reply")
            latency = float(info.get("latency_s", time.perf_counter() - t0))
            return Step(tools[:max(1, v.budget_left)], str(obj.get("rationale", ""))[:240], latency, dict(info))
        except Exception as e:  # noqa: BLE001 - any failure degrades to the deterministic planner
            return Step(["hold"], f"planner unavailable ({type(e).__name__})",
                        time.perf_counter() - t0, {"failed": True})


DIRECT_SYSTEM = """You are the overtaking decision layer of an autonomous car following a slower lead vehicle.
Decide whether to start an overtake right now. There is no other safety check after you: if you say pass, the car pulls out.
Use the measurements given. Say pass only if it is clearly safe and worthwhile; otherwise wait."""


class DirectLLM:
    """Ablation: the LLM decides pass/wait itself, with no critic and no gates."""
    kind = "llm_direct"

    def __init__(self, complete: Complete):
        self.complete = complete

    def decide(self, snapshot: Dict[str, Any], two_lane: bool, urgency: str) -> Tuple[str, str, float, Dict[str, Any]]:
        user = (f"Road: {'two-lane (passing uses the oncoming lane)' if two_lane else 'multi-lane'}\n"
                f"Trip urgency: {urgency}\nMeasurements (gaps m, rates m/s negative = closing):\n{json.dumps(snapshot)}\n"
                f"Answer pass or wait with a one-sentence reason.")
        sch = {"type": "object", "additionalProperties": False, "required": ["decision", "reason"],
               "properties": {"decision": {"type": "string", "enum": ["pass", "wait"]}, "reason": {"type": "string"}}}
        t0 = time.perf_counter()
        try:
            obj, info = self.complete(DIRECT_SYSTEM, user, sch)
            return obj["decision"], str(obj.get("reason", ""))[:240], float(info.get("latency_s", time.perf_counter() - t0)), info
        except Exception as e:  # noqa: BLE001
            return "wait", f"llm error: {type(e).__name__}", time.perf_counter() - t0, {"fallback": True}


__all__ = ["View", "Step", "RulePlanner", "LLMPlanner", "DirectLLM", "SENSING", "SYSTEM_PROMPT",
           "build_user_prompt", "schema"]
