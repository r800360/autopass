"""The planner's action space. Sensing tools run perception on one region of interest and
write one belief field; `peek` is a physical information-gathering action; `propose_pass`
and `hold` end a deliberation cycle.
"""
from __future__ import annotations

from typing import Dict

from .belief import Belief
from .types import CAMERA_OF, FRONT, REAR, SIDE, World

TOOLS: Dict[str, str] = {
    "sense_front": "Front camera, own lane: gap to the lead and its speed.",
    "sense_passing_lane": "Front camera, passing lane (the ONCOMING lane on a two-lane road): "
                          "nearest vehicle ahead and how far the lane is visibly clear.",
    "sense_rear": "Rear-quarter camera, passing lane: nearest vehicle behind. Two looks give its closing speed.",
    "sense_blind_spot": "Side camera: is a vehicle alongside in the passing lane? Do this last, like a shoulder check.",
    "check_corridor": "Road-graph lookup: how far the passing lane stays a valid, junction-free lane, "
                      "and whether crossing the lane marking is legal.",
    "peek": "Shift toward the passing side inside our own lane to see past the lead. Takes about a "
            "second to pay off; use when the passing lane's visible range is too short.",
    "propose_pass": "Submit the overtake to the safety critic. It approves only on fresh evidence.",
    "hold": "Keep following this cycle.",
}
SENSING = ("sense_front", "sense_passing_lane", "sense_rear", "sense_blind_spot", "check_corridor")
TERMINAL = ("propose_pass", "hold")
LEGACY_TOOLS = ("sense_front", "sense_passing_lane", "sense_rear", "check_corridor", "propose_pass", "hold")


def camera_for(tool: str, two_lane: bool) -> str:
    """Which camera a sensing tool reads (used for perception-cost accounting)."""
    if tool == "sense_front":
        return CAMERA_OF[FRONT]
    if tool == "sense_passing_lane":
        return "front_tele" if two_lane else "front_wide"
    if tool == "sense_rear":
        return CAMERA_OF[REAR]
    if tool == "sense_blind_spot":
        return CAMERA_OF[SIDE]
    return ""


def run_tool(tool: str, world: World, b: Belief) -> str:
    """Execute one non-terminal tool: mutate the belief, return a short observation."""
    now = world.time
    if tool == "sense_front":
        b.front_track.update(world.sense(FRONT), now)
        c = b.channels[FRONT] = b.front_track.copy()
        b.lead_speed_mps = None if c.rate_mps is None else max(0.0, world.ego().speed + c.rate_mps)
        obs = f"front gap={_f(c.gap_m)} m, lead speed={_f(b.lead_speed_mps)} m/s"
    elif tool == "sense_passing_lane":
        name = b.passing_channel
        c = b.ch(name)
        c.update(world.sense(name), now)
        c.order = b.revision + 1
        what = "oncoming" if b.two_lane else "passing-lane vehicle ahead"
        obs = f"{what}: gap={_f(c.gap_m)} m, rate={_f(c.rate_mps)} m/s, lane visible to {c.range_m:.0f} m"
    elif tool == "sense_rear":
        c = b.ch(REAR)
        c.update(world.sense(REAR), now)
        c.order = b.revision + 1
        obs = f"rear gap={_f(c.gap_m)} m, rate={_f(c.rate_mps)} m/s (reach {c.range_m:.0f} m)"
    elif tool == "sense_blind_spot":
        b.side_occupied = bool(world.sense(SIDE).dets)
        b.side_t = now
        b.side_order = b.revision + 1
        obs = "blind spot OCCUPIED" if b.side_occupied else "blind spot clear"
    elif tool == "check_corridor":
        cor = world.corridor()
        b.corridor_clear_m, b.corridor_legal, b.corridor_t = cor.clear_m, cor.legal, now
        obs = f"passing lane valid for {cor.clear_m:.0f} m, crossing {'legal' if cor.legal else 'NOT legal'}"
    elif tool == "peek":
        b.peek = True
        obs = "peeking toward the passing side; re-sense the passing lane once settled"
    else:
        return "unknown tool"
    b.tool_history.append(tool)
    b.touch()
    return obs


def _f(x) -> str:
    return "none" if x is None else f"{x:.1f}"
