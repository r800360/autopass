"""Regression checks for task ordering, optional planners and trip control."""
from dataclasses import replace
import json
import math

import pytest

from autopass.cli import browser_run, case, experiment
from autopass.core.belief import Channel
from autopass.core.critic import stale_channels
from autopass.core.gates import GateConfig
from autopass.core.imitation import ImitationSteering
from autopass.core.planner import LLMPlanner, View
from autopass.core.tools import TOOLS, run_tool
from autopass.core.types import Detection, SenseResult
from autopass.sim.kinematic import KinematicWorld
from autopass.trip import navigate, parse_request


def test_same_tick_never_adds_rate_evidence_and_object_switch_restarts_track():
    c = Channel()
    c.update(SenseResult([Detection(20, 0, track_id=3)], 80), 0)
    c.update(SenseResult([Detection(19, 0, track_id=3)], 80), 0)
    assert c.n == 1 and c.gap_m == 20
    c.update(SenseResult([Detection(18, 0, track_id=3)], 80), 0.5)
    assert c.rate_mps == -4
    c.update(SenseResult([Detection(17, 0, track_id=7)], 80), 1)
    assert c.rate_mps is None and c.n == 1


def test_shoulder_check_tool_order_matters_inside_one_timestamp():
    from autopass.core.belief import Belief
    b = Belief()
    world = KinematicWorld(case("s01"))
    run_tool("sense_blind_spot", world, b)
    run_tool("sense_rear", world, b)
    assert "side" in stale_channels(b, world.time, GateConfig())
    run_tool("sense_blind_spot", world, b)
    assert "side" not in stale_channels(b, world.time, GateConfig())


def test_failed_llm_call_holds_without_fabricating_a_tool_decision():
    def unavailable(*_):
        raise TimeoutError
    view = View({}, [], False, "high", tuple(TOOLS), [], [], None, 8)
    step = LLMPlanner(unavailable).next(view)
    assert step.tools == ["hold"] and step.info["failed"]


def test_trip_replans_in_response_to_repeated_denials():
    req = parse_request("From Depot to Clinic, urgent, within 33 seconds")
    adaptive = navigate(req)
    fixed = navigate(req, replan=False)
    assert adaptive["arrived"] and fixed["arrived"]
    assert adaptive["deadline_met"] and not fixed["deadline_met"]
    assert any(event["kind"] == "reroute" for event in adaptive["trace"])


def test_trip_declines_unknown_destination():
    with pytest.raises(ValueError):
        parse_request("Depot to the airport ASAP")


def test_imitator_stays_bounded_while_moving_scene():
    model = ImitationSteering.load()
    assert model.predict(0, 6, 6, 0, 0.75) is None
    assert model.predict(math.nan, 0, 6, 0, 0.75) is None
    assert abs(model.predict(7, 7, 7, 0, 0.75)) <= 0.75
    result, _ = experiment("s01", steering="imitation")
    assert result["success"] and result["steering"] == "imitation"


def test_browser_adapter_matches_cli():
    payload = {"scenario": "s17", "policy": "autopass", "seed": 9}
    adapted = json.loads(browser_run(json.dumps(payload)))
    raw, _ = experiment("s17", seed=9)
    assert adapted["result"] == raw and adapted["schema"] == 2


def test_startup_delay_does_not_consume_an_old_proposal():
    from autopass.core.episode import Policy
    from autopass.sim.kinematic import run

    result, _ = run(case("s01"), Policy(min_follow_s=0.52))
    assert result["t_commit_s"] == 1.0
