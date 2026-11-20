"""Closed-loop tests in the kinematic simulator: sensing geometry, catalog outcomes, regressions."""
import pytest

from autopass.core.episode import Policy
from autopass.core.scenario import NPC, Scenario, catalog, first_release_ids, sample
from autopass.core.telemetry import load
from autopass.core.types import ONCOMING, PASSING_AHEAD, REAR, SIDE
from autopass.sim.kinematic import KinematicWorld, run
from autopass.sim.sensors import PerceptionProfile

CAT = catalog()


def world(npcs, two_lane=False):
    return KinematicWorld(Scenario(id="t", narrative="", npcs=npcs, two_lane=two_lane, speed_limit=11.1 if two_lane else 25.0))


# ---- sensing geometry -------------------------------------------------------------------
def test_vehicle_alongside_is_in_the_blind_spot_not_ahead_or_behind():
    w = world([NPC("lead", "travel", 26.0, 6.0), NPC("side", "passing", -1.0, 6.0)])
    assert w.sense(SIDE).dets
    assert not w.sense(REAR).dets and not w.sense(PASSING_AHEAD).dets


def test_rear_vehicle_gap_is_bumper_to_bumper():
    w = world([NPC("lead", "travel", 26.0, 6.0), NPC("rear", "passing", -45.0, 18.0)])
    assert w.sense(REAR).nearest.gap_m == pytest.approx(45.0 - 4.7)


def test_tailgater_does_not_hide_the_passing_lane_behind():
    w = world([NPC("lead", "travel", 26.0, 6.0), NPC("queue", "travel", -9.0, 6.0)])
    assert w.sense(REAR).range_m >= 200.0


def test_lead_hides_the_oncoming_lane_until_the_ego_peeks():
    w = world([NPC("lead", "travel", 22.0, 4.0)], two_lane=True)
    blocked = w.sense(ONCOMING).range_m
    w.ed = 0.8                                           # shifted toward the centre line
    assert blocked < 100.0 < w.sense(ONCOMING).range_m


def test_truck_hides_more_than_a_car():
    car = world([NPC("lead", "travel", 22.0, 4.0)], two_lane=True)
    truck = world([NPC("lead", "travel", 24.65, 4.0, kind="truck")], two_lane=True)      # same bumper gap
    car.ed = truck.ed = 0.8
    assert truck.sense(ONCOMING).range_m < car.sense(ONCOMING).range_m


def test_oncoming_car_is_seen_at_range_when_peeking():
    w = world([NPC("lead", "travel", 22.0, 4.0), NPC("oncoming", "passing", 180.0, 9.0, direction=-1)], two_lane=True)
    w.ed = 0.8
    assert w.sense(ONCOMING).nearest.gap_m == pytest.approx(180.0 - 4.7)


# ---- catalog --------------------------------------------------------------------------------
@pytest.mark.parametrize("sid", list(CAT))
def test_autopass_behaves_as_expected(sid):
    r, _ = run(CAT[sid], Policy())
    assert r["success"], r
    assert not r["collision"] and not r["offroad"] and not r["unsafe_commit"] and not r["unwarranted_commit"]
    assert r["max_lane_dev_m"] < 0.3


@pytest.mark.parametrize("sid", list(CAT))
def test_never_pass_never_collides_and_never_overtakes(sid):
    r, _ = run(CAT[sid], Policy(name="no_pass"))
    assert not r["collision"] and not r["pass_attempted"]


def test_always_pass_is_unsafe_somewhere():
    res = [run(CAT[s], Policy(name="always_pass"))[0] for s in CAT]
    assert sum(r["unsafe_commit"] for r in res) >= 5
    assert sum(r["collision"] for r in res) >= 1


def test_abort_then_retry_when_oncoming_appears_mid_pass():
    r, rec = run(CAT["s27_t01_rural_late_oncoming"], Policy(), record=True)
    kinds = [e["kind"] for e in rec.events]
    assert r["aborts"] == 1 and kinds.count("commit") == 2 and r["overtake_completed"]
    assert kinds.index("abort") < len(kinds) - 1


def test_two_lane_pass_uses_peek():
    r, _ = run(CAT["s13_t01_rural_two_lane_pass"], Policy())
    assert r["agency"]["tool_histogram"].get("peek", 0) >= 1


def test_merge_back_is_triggered_by_vision():
    r, _ = run(CAT["s01_t04_highway_safe_pass"], Policy())
    assert r["merge_trigger"] == "vision"


# ---- regressions: what the first release's decision logic does on the same scenarios -----------
@pytest.mark.parametrize("sid", [
    "s17_t04_reject_rear_traffic", "s25_t04_slow_overtaker",
])
def test_legacy_thresholds_permit_unsafe_commits(sid):
    old, _ = run(CAT[sid], Policy.first_release())
    new, _ = run(CAT[sid], Policy())
    assert old["unsafe_commit"]
    assert not new["collision"] and not new["unsafe_commit"]


def test_first_release_pulls_out_beside_a_blind_spot_vehicle():
    old, _ = run(CAT["s24_t04_blind_spot_pacer"], Policy.first_release())
    assert old["unsafe_commit"] and "blind spot" in old["commit_blockers"][0]


def test_first_release_ids_are_the_original_23():
    assert len(first_release_ids()) == 23


# ---- sampled scenarios, noise, telemetry ---------------------------------------------------------
def test_sampled_scenarios_never_collide_with_oracle_perception():
    import random
    rng = random.Random(7)
    for i in range(60):
        r, _ = run(sample(rng, i), Policy(), seed=i)
        assert not r["collision"], r


def test_runs_are_deterministic_for_a_seed():
    scn = CAT["s17_t04_reject_rear_traffic"]
    prof = PerceptionProfile.preset("degraded")
    a, _ = run(scn, Policy(), seed=3, perception=prof)
    b, _ = run(scn, Policy(), seed=3, perception=prof)
    assert a == b


def test_telemetry_round_trip(tmp_path):
    r, rec = run(CAT["s01_t04_highway_safe_pass"], Policy(), record=True)
    path = rec.save(tmp_path / "run.json.gz")
    d = load(path)
    assert d["result"]["scenario_id"] == r["scenario_id"] and d["schema"] == 2
    assert len(d["frames"]) > 50 and d["cycles"] and d["events"][0]["kind"] == "commit"
    assert path.stat().st_size < 60_000                  # a whole run in tens of kilobytes
