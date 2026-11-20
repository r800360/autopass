"""Scenario specification, the hand-authored catalog, and a naturalistic sampler.

A Scenario is static: it fixes the trial. The belief (core/belief.py) is what evolves.
Positions are longitudinal offsets from the ego centre at t = 0 (+ ahead). Everything starts
at speed (a rolling start), so no scenario depends on a settling period.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class NPC:
    role: str                    # lead | rear | oncoming | blocker | side | queue | ambient
    lane: str                    # travel | passing
    s: float                     # metres ahead (+) or behind (-) of the ego centre at spawn
    v: float                     # speed along its own direction of travel (m/s)
    direction: int = 1           # +1 same direction as the ego, -1 oncoming
    kind: str = "car"            # car | truck | bus
    behavior: str = "cruise"     # cruise = holds speed (non-cooperative) | idm = reacts to traffic
    appear_s: float = 0.0        # spawns this many seconds into the run
    semantic: Optional[str] = None   # e.g. "left_blinker", "emergency_lights"


@dataclass
class Scenario:
    id: str
    narrative: str
    npcs: List[NPC] = field(default_factory=list)
    two_lane: bool = False       # passing lane is the oncoming lane
    passing_side: str = "left"
    urgency: str = "high"        # low | medium | high
    expected: str = "pass"       # pass | wait | yield_then_pass
    cruise_mps: float = 14.0
    pass_mps: float = 20.0
    speed_limit: float = 25.0    # used to bound traffic the ego cannot see
    budget_s: float = 32.0
    lane_width: float = 3.5
    group: str = "highway"       # catalog section
    # CARLA-only placement hints
    town: str = "Town04"
    weather: str = "clear_noon"
    corridor_rank: int = 0
    min_straight_m: float = 170.0
    ambient: int = 0
    lead_bp: str = "vehicle.audi.tt"

    def lead(self) -> Optional[NPC]:
        return next((n for n in self.npcs if n.role == "lead"), None)

    def to_dict(self) -> dict:
        return asdict(self)


def _lead(v, gap, kind="car"):
    return NPC("lead", "travel", gap, v, kind=kind)


def _hwy(i, name, narrative, lead_v, **kw):
    npcs = [_lead(lead_v, kw.pop("gap", 26.0), kw.pop("lead_kind", "car"))] + kw.pop("extra", [])
    return Scenario(id=f"s{i:02d}_t04_{name}", narrative=narrative, npcs=npcs, town="Town04",
                    speed_limit=25.0, group=kw.pop("group", "highway"), **kw)


def _art(i, town, name, narrative, lead_v, cruise, pas, **kw):
    limit = {"Town05": 16.7, "Town03": 13.9}[town]
    npcs = [_lead(lead_v, kw.pop("gap", 24.0))] + kw.pop("extra", [])
    return Scenario(id=f"s{i:02d}_t{town[-2:]}_{name}", narrative=narrative, npcs=npcs, town=town,
                    cruise_mps=cruise, pass_mps=pas, speed_limit=limit, group=kw.pop("group", "arterial"), **kw)


def _rural(i, name, narrative, lead_v, **kw):
    npcs = [_lead(lead_v, kw.pop("gap", 22.0), kw.pop("lead_kind", "car"))] + kw.pop("extra", [])
    return Scenario(id=f"s{i:02d}_t01_{name}", narrative=narrative, npcs=npcs, town="Town01", two_lane=True,
                    cruise_mps=10.0, pass_mps=13.0, speed_limit=11.1, min_straight_m=260.0,
                    budget_s=kw.pop("budget_s", 36.0), group=kw.pop("group", "two_lane"), **kw)


def catalog() -> Dict[str, Scenario]:
    """Index 23 adapted scenarios by their historical IDs and add 11 stress cases."""
    S = [
        # ---- highway passes -------------------------------------------------------------
        _hwy(1, "highway_safe_pass", "Motorway, clear lanes, urgent: the canonical overtake.", 6.0),
        _hwy(2, "highway_stalled_lead", "Motorway: a nearly stopped vehicle ahead.", 0.5, gap=28.0,
             cruise_mps=13.0, pass_mps=19.0, corridor_rank=3, lead_bp="vehicle.carlamotors.carlacola"),
        _hwy(3, "highway_wet_pass", "Motorway in the wet.", 6.5, gap=27.0, cruise_mps=13.5, pass_mps=19.0,
             corridor_rank=6, weather="wet_noon"),
        _hwy(4, "highway_right_pass", "Motorway: the open lane is on the right.", 6.0, passing_side="right",
             corridor_rank=10),
        _hwy(5, "highway_dusk_pass", "Motorway at sunset, low sun.", 6.0, corridor_rank=14, weather="clear_sunset"),
        # ---- arterial / urban passes ----------------------------------------------------
        _art(6, "Town05", "multilane_pass", "Multi-lane arterial, urgent.", 5.5, 12.5, 17.5),
        _art(7, "Town05", "cloudy_pass", "Arterial under cloud, medium urgency.", 5.0, 12.0, 17.0,
             urgency="medium", corridor_rank=8, weather="cloudy_noon"),
        _art(8, "Town05", "wet_pass", "Arterial in the wet.", 5.5, 12.5, 17.5, corridor_rank=16,
             weather="wet_cloudy_noon"),
        _art(9, "Town03", "urban_pass", "Urban road, medium urgency.", 4.5, 11.0, 15.5, gap=22.0, urgency="medium"),
        _art(10, "Town03", "clear_pass", "Urban multi-lane stretch, urgent.", 5.0, 11.5, 16.0, gap=22.0,
             corridor_rank=5),
        _hwy(11, "coast_overcast_pass", "Coastal motorway under cloud.", 6.0, corridor_rank=20,
             weather="cloudy_noon"),
        _art(12, "Town05", "arterial_clear_pass", "Arterial, another corridor, medium urgency.", 5.0, 12.0, 17.0,
             urgency="medium", corridor_rank=4),
        # ---- two-lane rural passes (the passing lane is the oncoming lane) --------------
        _rural(13, "rural_two_lane_pass", "Two-lane road, oncoming lane clear.", 4.0),
        _rural(14, "rural_hardrain_pass", "Two-lane road in heavy rain.", 4.0, corridor_rank=8, weather="hard_rain"),
        _rural(15, "rural_slow_tractor_pass", "Country lane behind a very slow vehicle.", 3.0, corridor_rank=14,
               weather="cloudy_noon"),
        # ---- hazards and declines -------------------------------------------------------
        _hwy(16, "reject_fast_lead", "Lead is nearly at our cruise speed: passing is not worth it.", 12.5,
             gap=28.0, expected="wait", corridor_rank=2, budget_s=16.0, group="hazard"),
        _hwy(17, "reject_rear_traffic", "A fast vehicle closes in the passing lane. Wait until a safe opening.", 6.0,
             cruise_mps=13.0, pass_mps=19.0, expected="yield_then_pass", corridor_rank=4, group="hazard",
             extra=[NPC("rear", "passing", -45.0, 18.0)]),
        _art(18, "Town05", "reject_fast_lead", "Lead keeps pace with traffic: keep lane.", 11.0, 12.5, 17.0,
             gap=26.0, expected="wait", corridor_rank=12, budget_s=16.0, group="hazard"),
        _rural(19, "rural_oncoming_reject", "Oncoming car in the only passing lane: wait for it, then overtake.",
               4.0, expected="yield_then_pass", corridor_rank=3, group="hazard",
               extra=[NPC("oncoming", "passing", 150.0, 8.0, direction=-1)]),
        _art(20, "Town03", "reject_fast_lead", "Urban: lead keeps pace, decline.", 10.0, 11.5, 16.0,
             expected="wait", corridor_rank=2, budget_s=16.0, group="hazard"),
        # ---- traffic and large vehicles -------------------------------------------------
        _hwy(21, "highway_ambient_pass", "Motorway with ambient traffic.", 5.0, ambient=16, budget_s=34.0,
             group="traffic"),
        _hwy(22, "highway_heavy_traffic_pass", "Motorway, heavy ambient traffic.", 5.0, gap=24.0, cruise_mps=13.0,
             pass_mps=18.0, corridor_rank=3, ambient=22, weather="cloudy_noon", budget_s=34.0, group="traffic"),
        _hwy(23, "highway_truck_pass", "Motorway behind a slow heavy truck: a longer manoeuvre.", 5.0, gap=28.0,
             lead_kind="truck", cruise_mps=15.0, pass_mps=21.0, corridor_rank=2, ambient=8, budget_s=36.0,
             lead_bp="vehicle.carlamotors.firetruck", group="traffic"),
        # ---- new in this release: cases the first release never exercised ---------------
        _hwy(24, "blind_spot_pacer", "A car sits alongside in the passing lane at our speed: never pull out.",
             6.0, expected="wait", corridor_rank=5, budget_s=18.0, group="stress",
             extra=[NPC("side", "passing", -1.0, 6.0)]),
        _hwy(25, "slow_overtaker", "A car creeps past in the passing lane: wait for it to clear, then go.", 6.0,
             expected="yield_then_pass", corridor_rank=7, group="stress",
             extra=[NPC("rear", "passing", -14.0, 8.5)]),
        _rural(26, "rural_far_fast_oncoming", "Oncoming car far away but fast: the gap looks big and is not.",
               4.0, expected="yield_then_pass", corridor_rank=5, group="stress",
               extra=[NPC("oncoming", "passing", 190.0, 12.0, direction=-1)]),
        _rural(27, "rural_late_oncoming", "An oncoming car appears just after the pass starts: abort, then retry.",
               4.0, expected="yield_then_pass", corridor_rank=6, budget_s=44.0, group="stress",
               extra=[NPC("oncoming", "passing", 115.0, 12.0, direction=-1, appear_s=4.5)]),
        _hwy(28, "passing_lane_blocked", "A car ahead in the passing lane is no faster than our lead: no gain.",
             6.0, expected="wait", corridor_rank=8, budget_s=18.0, group="stress",
             extra=[NPC("blocker", "passing", 14.0, 6.0)]),
        _hwy(29, "tailgater_behind", "A car tailgates in our own lane; the passing lane is clear.", 6.0,
             corridor_rank=9, group="stress", extra=[NPC("queue", "travel", -11.0, 6.0, behavior="idm")]),
        _rural(30, "rural_two_oncoming", "Two oncoming cars with a gap between them that is too small.", 4.0,
               expected="yield_then_pass", corridor_rank=7, budget_s=40.0, group="stress",
               extra=[NPC("oncoming", "passing", 120.0, 9.0, direction=-1),
                      NPC("oncoming", "passing", 215.0, 9.0, direction=-1)]),
        _rural(31, "rural_truck_lead", "Two-lane road behind a long truck: cannot see far enough past it, so decline.",
               4.0, lead_kind="truck", expected="wait", corridor_rank=9, budget_s=24.0, group="stress",
               lead_bp="vehicle.carlamotors.firetruck"),
        _hwy(32, "low_urgency_slow_lead", "No hurry and the lead is only moderately slow: stay put.", 8.0,
             urgency="low", expected="wait", corridor_rank=11, budget_s=16.0, group="stress"),
        _hwy(33, "low_urgency_stalled_lead", "No hurry, but the lead is nearly stopped: pass anyway.", 0.5,
             gap=28.0, urgency="low", corridor_rank=12, group="stress"),
        _hwy(34, "stopped_car_in_passing_lane", "A stopped car sits in the passing lane ahead: go past it, then overtake.",
             6.0, expected="yield_then_pass", corridor_rank=13, group="stress",
             extra=[NPC("blocker", "passing", 34.0, 0.0)]),
    ]
    return {s.id: s for s in S}


V1_IDS = [f"s{i:02d}" for i in range(1, 24)]


def first_release_ids() -> List[str]:
    return [k for k in catalog() if k[:3] in V1_IDS]


# ---------------------------------------------------------------------------------------
def sample(rng: random.Random, i: int = 0) -> Scenario:
    """Draw one scenario from a naturalistic-style distribution.

    Speeds and headways follow the shapes reported for free-flow traffic: speeds roughly
    normal around the limit, time headways log-normal. Each draw decides independently
    whether a rear vehicle, a blind-spot vehicle, a passing-lane vehicle ahead or oncoming
    traffic is present, so hazards occur in combination as well as alone.
    """
    two_lane = rng.random() < 0.35
    limit = rng.choice([11.1, 13.9]) if two_lane else rng.choice([16.7, 25.0, 30.0])
    cruise = min(limit, max(6.0, rng.gauss(0.85 * limit, 0.08 * limit)))
    lead_v = max(0.0, rng.choice([rng.uniform(0.0, 0.5 * cruise), rng.uniform(0.5 * cruise, 1.0 * cruise)]))
    pas = min(1.15 * limit, cruise + rng.uniform(2.0, 6.0))
    kind = "truck" if rng.random() < 0.15 else "car"
    npcs = [NPC("lead", "travel", rng.uniform(16.0, 40.0), lead_v, kind=kind)]
    headway = lambda: math.exp(rng.gauss(0.6, 0.7))          # seconds, median ~1.8 s
    if two_lane:
        n = rng.choice([0, 0, 1, 1, 2])
        d = rng.uniform(40.0, 320.0)
        for _ in range(n):
            v = max(4.0, rng.gauss(0.9 * limit, 0.12 * limit))
            npcs.append(NPC("oncoming", "passing", d, v, direction=-1,
                            appear_s=(rng.uniform(2.0, 8.0) if rng.random() < 0.2 else 0.0)))
            d += v * headway() + rng.uniform(20.0, 160.0)
    else:
        if rng.random() < 0.45:      # traffic approaching from behind in the passing lane
            v = max(lead_v, rng.gauss(0.95 * limit, 0.12 * limit))
            npcs.append(NPC("rear", "passing", -rng.uniform(8.0, 140.0), v,
                            behavior="idm" if rng.random() < 0.5 else "cruise"))
        if rng.random() < 0.12:      # someone pacing us in the blind spot
            npcs.append(NPC("side", "passing", rng.uniform(-4.0, 4.0), lead_v + rng.uniform(-0.5, 0.8)))
        if rng.random() < 0.25:      # vehicle ahead in the passing lane, possibly slow or stopped
            v = rng.choice([0.0, rng.uniform(lead_v, limit)])
            npcs.append(NPC("blocker", "passing", rng.uniform(15.0, 120.0), v))
        if rng.random() < 0.2:
            npcs.append(NPC("queue", "travel", -rng.uniform(9.0, 30.0), lead_v, behavior="idm"))
    return Scenario(id=f"mc{i:05d}", narrative="sampled", npcs=npcs, two_lane=two_lane,
                    urgency=rng.choice(["low", "medium", "high", "high"]), expected="?",
                    cruise_mps=round(cruise, 2), pass_mps=round(pas, 2), speed_limit=limit,
                    budget_s=40.0, group="sampled", town="-")
