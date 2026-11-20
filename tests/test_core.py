"""Unit tests for the decision core: kinematics, gates, belief tracking, critic."""
import math

import pytest

from autopass.core import critic
from autopass.core.belief import Belief, Channel
from autopass.core.gates import GateConfig, GateInput, evaluate
from autopass.core.kinematics import ego_distance, plan_pass, required_follower_decel, time_to_gain
from autopass.core.types import FRONT, REAR, Detection, EgoState, SenseResult


def reading(gap, rng=100.0):
    return SenseResult([Detection(gap, 0.0)] if gap is not None else [], rng)


# ---- kinematics -------------------------------------------------------------------------
def test_pass_plan_matches_hand_calculation():
    # ego and lead at 6 m/s, pass speed 20, a = 2.5: accelerate 5.6 s gaining 39.2 m, need 41.7 m
    p = plan_pass(24.0, 6.0, 6.0, 20.0)
    assert p.d_rel == pytest.approx(24 + 5 + 8 + 4.7)
    assert p.t_overtake == pytest.approx(5.6 + (41.7 - 39.2) / 14.0, abs=1e-6)
    assert p.d_ego == pytest.approx(ego_distance(p.t_end, 6.0, 20.0, 2.5))


def test_pass_infeasible_when_not_faster_than_lead():
    assert not plan_pass(20.0, 5.0, 16.0, 15.0).feasible


def test_time_to_gain_is_monotonic():
    ts = [time_to_gain(d, 4.0, 4.0, 13.0) for d in (5, 10, 20, 40, 80)]
    assert ts == sorted(ts) and ts[0] > 0


def test_follower_decel():
    assert required_follower_decel(100.0, -2.0) == 0.0            # falling behind: never brakes
    assert required_follower_decel(3.0, 0.0) == math.inf          # ends up inside the standstill gap
    near, far = required_follower_decel(40.0, 12.0), required_follower_decel(120.0, 12.0)
    assert near > far > 0


# ---- gates ------------------------------------------------------------------------------
def clear_highway(**kw):
    x = GateInput(two_lane=False, urgency="high", v_ego=6.0, v_cruise=14.0, v_pass=20.0, speed_limit=25.0,
                  front_gap=20.0, lead_speed=6.0, pass_range=80.0, rear_range=220.0, side_occupied=False,
                  corridor_clear=1000.0, corridor_legal=True)
    for k, v in kw.items():
        setattr(x, k, v)
    return x


def test_clear_highway_passes():
    r = evaluate(clear_highway(), GateConfig())
    assert r.can_pass and r.warrant_ok and r.hazard_ok


@pytest.mark.parametrize("change,gate", [
    (dict(front_gap=8.0), "front_room"),
    (dict(lead_speed=13.0), "warrant"),
    (dict(lead_speed=None), "warrant"),
    (dict(rear_gap=30.0, rear_closing=12.0), "rear"),
    (dict(rear_gap=30.0, rear_closing=None), "rear"),            # rate unknown: worst case
    (dict(rear_range=40.0), "rear"),                             # cannot see far enough behind
    (dict(side_occupied=True), "blind_spot"),
    (dict(side_occupied=None), "blind_spot"),                    # never observed is not "clear"
    (dict(pass_gap=14.0, pass_speed=6.0), "passing_lane"),       # no faster than our lead
    (dict(corridor_clear=60.0), "corridor"),
    (dict(corridor_legal=False), "corridor"),
    (dict(semantic_veto="lead is signalling left"), "semantic"),
])
def test_each_gate_blocks(change, gate):
    r = evaluate(clear_highway(**change), GateConfig())
    assert not r.can_pass and r.checks[gate] is False


def test_warrant_depends_on_urgency():
    x = clear_highway(lead_speed=10.0)            # 29% below cruise
    assert evaluate(x, GateConfig()).checks["warrant"]
    x.urgency = "medium"
    assert not evaluate(x, GateConfig()).checks["warrant"]
    x.urgency, x.lead_speed = "low", 0.5          # even with no hurry, do not sit behind a stopped car
    assert evaluate(x, GateConfig()).checks["warrant"]


def two_lane(**kw):
    x = GateInput(two_lane=True, urgency="high", v_ego=4.0, v_cruise=10.0, v_pass=13.0, speed_limit=11.1,
                  front_gap=20.0, lead_speed=4.0, pass_range=250.0, rear_range=220.0, side_occupied=False,
                  corridor_clear=1000.0, corridor_legal=True)
    for k, v in kw.items():
        setattr(x, k, v)
    return x


def test_two_lane_needs_sight_distance():
    assert evaluate(two_lane(), GateConfig()).can_pass
    r = evaluate(two_lane(pass_range=70.0), GateConfig())           # view blocked by the lead
    assert not r.checks["passing_lane"] and "sight distance" in r.blockers[0]
    r = evaluate(two_lane(pass_gap=150.0, pass_speed=-8.0), GateConfig())
    assert not r.checks["passing_lane"] and "oncoming" in r.blockers[0]
    assert r.detail["sight_need_m"] > 150


def test_legacy_gates_treat_unseen_as_clear():
    x = two_lane(pass_range=70.0, side_occupied=True, lead_speed=4.0, front_gap=20.0)
    assert evaluate(x, GateConfig.legacy()).can_pass                # the first release would go
    assert not evaluate(x, GateConfig()).can_pass


# ---- belief tracking ----------------------------------------------------------------------
def test_rate_needs_two_looks():
    c = Channel()
    c.update(reading(40.0), 0.0)
    assert c.rate_mps is None and c.n == 1
    c.update(reading(34.0), 0.5)
    assert c.rate_mps == pytest.approx(-12.0)
    c.update(reading(None), 1.0)
    assert c.gap_m is None and c.rate_mps is None and c.clear_m() == 100.0


def test_rate_resets_on_a_different_object():
    c = Channel()
    for i, g in enumerate((40.0, 38.0, 36.0)):
        c.update(reading(g), 0.5 * i)
    assert c.rate_mps == pytest.approx(-4.0)
    c.update(reading(90.0), 1.5)                                    # nearest object changed
    assert c.rate_mps is None and c.n == 1


def test_rate_from_a_20hz_tracker_is_smooth():
    c = Channel()
    for i in range(40):
        c.update(reading(30.0 - 2.0 * 0.05 * i), 0.05 * i)
    assert c.rate_mps == pytest.approx(-2.0, abs=0.05)


# ---- critic -------------------------------------------------------------------------------
def fresh_belief(now=10.0):
    b = Belief(two_lane=False, urgency="high", cruise_mps=14.0, pass_mps=20.0)
    b.channels[FRONT].update(reading(20.0, 85.0), now - 0.5)
    b.channels[FRONT].update(reading(20.0, 85.0), now)
    b.lead_speed_mps = 6.0
    b.ch("passing_ahead").update(reading(None, 80.0), now)
    b.ch(REAR).update(reading(None, 220.0), now)
    b.side_occupied, b.side_t = False, now
    b.corridor_clear_m, b.corridor_legal, b.corridor_t = 1000.0, True, now
    return b


EGO = EgoState(speed=6.0, d=0.0, heading=0.0)


def test_critic_approves_fresh_clear_evidence():
    v = critic.verify(fresh_belief(), EGO, 10.0, 25.0, GateConfig())
    assert v.approved and not v.stale


def test_critic_rejects_stale_hazard_evidence():
    v = critic.verify(fresh_belief(), EGO, 10.4, 25.0, GateConfig())
    assert not v.approved and set(v.stale) >= {"passing_ahead", "rear", "side"}


def test_shoulder_check_must_be_last():
    b = fresh_belief()
    b.ch(REAR).update(reading(None, 220.0), 10.1)                   # rear re-sensed after the blind spot
    assert "side" in critic.stale_channels(b, 10.1, GateConfig())
    b.side_occupied, b.side_t = False, 10.15
    assert critic.verify(b, EGO, 10.15, 25.0, GateConfig()).approved


def test_critic_extrapolates_a_closing_vehicle():
    b = fresh_belief()
    b.channels[REAR] = Channel()
    b.ch(REAR).update(reading(60.0, 220.0), 9.5)
    b.ch(REAR).update(reading(54.0, 220.0), 10.0)                   # closing at 12 m/s
    b.side_t = 10.0
    x_now = critic.gate_input(b, EGO, 25.0, 10.25)
    assert x_now.rear_gap == pytest.approx(54.0 - 12.0 * 0.25 - 0.8)
    assert x_now.rear_closing == pytest.approx(12.75)


def test_legacy_critic_accepts_half_second_old_evidence():
    b = fresh_belief()
    b.corridor_legal = True
    assert not critic.stale_channels(b, 10.5, GateConfig.legacy())
    assert critic.stale_channels(b, 10.5, GateConfig())


def test_current_corridor_caps_a_recent_observation():
    from autopass.core.types import Corridor

    cfg = GateConfig()
    b = fresh_belief()
    assert critic.verify(b, EGO, 10, 25, cfg).approved
    now = Corridor(clear_m=50.0, legal=True)
    v = critic.verify(b, EGO, 10, 25, cfg, corridor_now=now)
    assert not v.approved and not v.gates.checks["corridor"]
    assert b.corridor_clear_m == 1000.0
