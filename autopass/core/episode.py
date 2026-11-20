"""One closed-loop run: reflex layer + agent + manoeuvre state machine + controller.

Per 50 ms tick:
  1. reflex   the front camera is always on; it drives cruise control and emergency braking
  2. monitor  during a manoeuvre, watch the passing lane (abort) and the lead (merge-back)
  3. phase    follow -> lane_change -> overtake -> merge_back -> done   (abort -> follow)
  4. control  steer to a lane-centre point, track a target speed
Every 0.5 s while following, the agent deliberates (core/agent.py).

Nothing here reads ground truth to decide anything. Ground truth is used for metrics only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from . import control as ctl
from .agent import AgentConfig, PassingAgent
from .belief import Belief, Channel
from .gates import GateConfig
from .kinematics import ego_distance, plan_pass, time_to_gain
from .metrics import Metrics, truth_gates
from .planner import DirectLLM
from .scenario import Scenario
from .telemetry import Recorder
from .types import (ABORT, DONE, FOLLOW, FRONT, LANE_CHANGE, MERGE_BACK, OVERTAKE, SIDE_TRAVEL,
                    TRAVEL_BEHIND, Control, World)


@dataclass
class Policy:
    name: str = "autopass"             # autopass | no_pass | always_pass | llm_direct
    agent: AgentConfig = field(default_factory=AgentConfig)
    control: ctl.ControlConfig = field(default_factory=ctl.ControlConfig)
    cycle_s: float = 0.5               # deliberation period while following
    min_follow_s: float = 0.0          # first release used 2.5 s; a rolling start needs none
    abort: bool = True                 # may abandon a pass that turns unsafe
    monitor_s: float = 0.2             # how often the manoeuvre monitor looks
    abort_margin_s: float = 0.5        # in-flight time buffer (smaller than the commit buffer: hysteresis)
    rush_factor: float = 1.15          # speed-up when past the point of no return
    steering: str = "waypoint"
    distance_goal_m: Optional[float] = None

    @staticmethod
    def first_release() -> "Policy":
        """The configuration of the first release, for like-for-like comparison."""
        return Policy(agent=AgentConfig(gates=GateConfig.legacy(), auto_propose=True, max_tools=5,
                                        legacy_tools=True), abort=False, min_follow_s=2.5)


class EpisodeOver(Exception):
    pass


class Episode:
    def __init__(self, world: World, scn: Scenario, policy: Optional[Policy] = None, *,
                 planner: Any = None, direct: Optional[DirectLLM] = None,
                 recorder: Optional[Recorder] = None):
        self.w, self.scn, self.p = world, scn, policy or Policy()
        self.belief = Belief(two_lane=scn.two_lane, urgency=scn.urgency, cruise_mps=scn.cruise_mps,
                             pass_mps=scn.pass_mps)
        self.agent = PassingAgent(planner, self.p.agent)
        self.direct = direct
        if self.p.steering == "imitation":
            from .imitation import ImitationSteering
            self.steering_model = ImitationSteering.load()
        else:
            self.steering_model = None
        self.steering_fallbacks = 0
        self.yielded = False
        self.odometry_m = 0.0
        self.rec = recorder
        self.m = Metrics(lane_width=world.lane_width)
        self.phase = FOLLOW
        self.decision = "wait"
        self.proposal_time: Optional[float] = None
        self.steer = 0.0
        self.t_done: Optional[float] = None
        self.t_next_cycle = 0.0
        self.t_next_monitor = 0.0
        self.peek_t0: Optional[float] = None
        # manoeuvre bookkeeping (vision + odometry only)
        self.t_commit: Optional[float] = None
        self.front_gap0 = 30.0
        self.lead_v_ref = 0.0
        self.lead_track_id = None
        self.plan = None
        self.progress = 0.0
        self.mon = Channel()            # passing lane ahead, tracked during the manoeuvre
        self.clear_known = 0.0          # distance last known clear ahead in the passing lane ...
        self.clear_t = 0.0              # ... and when; it shrinks at worst-case closing speed if unobserved
        self.tb = Channel()             # travel lane behind, for merge-back
        self.side_travel_occupied = True
        self.rush = False
        self.hazard_ticks = 0
        self.aborts = 0
        self.merge_trigger: Optional[str] = None
        self.pass_attempted = False
        self.unsafe_commit = False
        self.unwarranted_commit = False
        self.commit_blockers: list = []
        self.reflex_calls = 0
        self.monitor_calls = 0
        self.ticks = 0
        self.on_tick = None             # optional callback(episode) after every step, e.g. to record video
        self.d_target = 0.0
        self.v_target = 0.0

    # -- main loop ----------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        try:
            while True:
                if self.phase == FOLLOW and self.w.time + 1e-9 >= self.t_next_cycle:
                    self._deliberate()
                self.tick()
        except EpisodeOver:
            pass
        return self.result()

    def _deliberate(self) -> None:
        w, name = self.w, self.p.name
        self.t_next_cycle = w.time + self.p.cycle_s
        if name == "no_pass":
            return
        self.belief.front_track.update(w.sense(FRONT), w.time)        # make sure the reflex view is current
        if name == "always_pass":
            self.decision = "pass"
            rec = {"t": round(w.time, 2), "cycle": 0, "tools": [], "decision": "pass",
                   "note": "always-pass baseline: overtake regardless of evidence", "planner": "none"}
        elif name == "llm_direct":
            rec = self.agent.decide_direct(self.direct, w, self.belief, self._wait)
            self.decision = rec["decision"]
        else:
            rec = self.agent.deliberate(w, self.belief, self._wait)
            self.decision = rec["decision"]
        if self.belief.peek and self.peek_t0 is None:
            self.peek_t0 = w.time
        self.proposal_time = w.time if self.decision == "pass" else None
        checks = rec.get("gates", {}).get("checks", {})
        b = self.belief
        measured_hazard = b.side_occupied or (
            (b.ch(b.passing_channel).gap_m is not None and checks.get("passing_lane") is False)
            or (b.ch("rear").gap_m is not None and checks.get("rear") is False)
        )
        self.yielded |= bool(self.decision == "wait" and measured_hazard)
        if self.rec is not None:
            self.rec.cycles.append(rec)
        if self.p.agent.realtime:
            self.t_next_cycle = max(self.t_next_cycle, w.time)

    def _wait(self, seconds: float) -> None:
        """Realtime mode: the world keeps moving while the agent thinks."""
        for _ in range(int(round(seconds / self.w.dt))):
            self.tick()

    # -- one simulation step ------------------------------------------------------------
    def tick(self) -> None:
        w, b, c = self.w, self.belief, self.p.control
        ego = w.ego()
        ft = b.front_track
        ft.update(w.sense(FRONT), w.time)
        self.reflex_calls += 1
        lead_v = None if ft.rate_mps is None else max(0.0, ego.speed + ft.rate_mps)

        if self.phase in (LANE_CHANGE, OVERTAKE):
            if ft.gap_m is not None and lead_v is not None and ft.track_id == self.lead_track_id and self.progress < self.front_gap0:
                self.lead_v_ref = lead_v                      # still behind the lead: keep its speed current
            self.progress += (ego.speed - self.lead_v_ref) * w.dt
            if w.time + 1e-9 >= self.t_next_monitor:
                self.t_next_monitor = w.time + self.p.monitor_s
                self._monitor(ego)
        self._transitions(ego, lead_v)

        d_target, v_target = self._targets(ego, lead_v)
        look = ctl.lookahead(ego.speed, c)
        dx, dy = w.target_point(d_target, look)
        if self.steering_model is not None:
            prediction = self.steering_model.predict(dx, dy, look, self.steer, c.max_steer)
            if prediction is None:
                self.steering_fallbacks += 1
            else:
                self.steer = prediction
        else:
            prediction = None
        if prediction is None:
            self.steer = ctl.steer_toward(dx, dy, look, self.steer, c)
        w.apply(Control(self.steer, v_target))
        w.step()
        self.odometry_m += 0.5 * (ego.speed + w.ego().speed) * w.dt
        self.ticks += 1

        truth = w.truth()
        self.d_target, self.v_target = d_target, v_target
        self.m.update(truth, self.phase, d_target, getattr(w, "induced_decel", 0.0))
        if self.rec is not None:
            self.rec.frame(truth, self.phase, d_target, v_target)
        if self.on_tick is not None:
            self.on_tick(self)
        if truth.collision or truth.offroad:
            if self.rec is not None:
                self.rec.event(w.time, "collision" if truth.collision else "offroad")
            raise EpisodeOver
        if self.p.distance_goal_m is not None:
            if self.odometry_m >= self.p.distance_goal_m and self.phase in (FOLLOW, DONE):
                self._event("segment_complete")
                raise EpisodeOver
        elif self.phase == DONE and self.t_done is not None and w.time - self.t_done > 2.0:
            raise EpisodeOver
        if w.time >= self.scn.budget_s:
            raise EpisodeOver

    # -- targets by phase ---------------------------------------------------------------
    def _targets(self, ego, lead_v):
        b, c, scn = self.belief, self.p.control, self.scn
        ft = b.front_track
        if self.phase in (LANE_CHANGE, OVERTAKE):
            v = scn.pass_mps * (self.p.rush_factor if self.rush else 1.0)
            if not scn.two_lane and self.mon.gap_m is not None:    # do not run into traffic in the passing lane
                u = None if self.mon.rate_mps is None else ego.speed + self.mon.rate_mps
                v = min(v, ctl.cruise_speed(self.mon.gap_m, u, ego.speed, v, c))
            return self.w.lane_width, v
        cruise = ctl.cruise_speed(ft.gap_m, lead_v, ego.speed, scn.cruise_mps, c)
        if self.phase == FOLLOW and b.peek:
            if self.peek_t0 is not None and self.w.time - self.peek_t0 > c.peek_max_s:
                b.peek, self.peek_t0 = False, None
            else:
                return c.peek_offset_m, cruise
        return 0.0, cruise

    # -- phase transitions --------------------------------------------------------------
    def _transitions(self, ego, lead_v) -> None:
        w = self.w
        if self.phase == FOLLOW:
            proposal_now = self.proposal_time is not None and abs(w.time - self.proposal_time) <= 1e-9
            if proposal_now and w.time + 1e-9 >= self.p.min_follow_s:
                self._commit(ego, lead_v)
        elif self.phase == LANE_CHANGE:
            if ego.d >= w.lane_width - 0.6:
                self.phase = OVERTAKE
        elif self.phase == OVERTAKE:
            self._maybe_merge(ego)
        elif self.phase == MERGE_BACK:
            if abs(ego.d) <= 0.6:
                self.phase, self.t_done = DONE, w.time
                self._event("done")
        elif self.phase == ABORT:
            if abs(ego.d) <= 0.6:
                self.phase, self.decision = FOLLOW, "wait"
                self.belief.intent = "deliberate"
                self.t_next_cycle = w.time + 1.0

    def _commit(self, ego, lead_v) -> None:
        w, b, g = self.w, self.belief, self.p.agent.gates
        ft = b.front_track
        self.front_gap0 = ft.gap_m if ft.gap_m is not None else 30.0
        self.lead_v_ref = lead_v if lead_v is not None else (b.lead_speed_mps or 0.0)
        self.lead_track_id = ft.track_id
        self.plan = plan_pass(self.front_gap0, ego.speed, self.lead_v_ref, self.scn.pass_mps, a_acc=g.a_acc,
                              lead_len=g.lead_len_m.get(ft.kind or "car", 5.0), ego_len=g.ego_len_m,
                              merge_margin=g.merge_margin_m, t_return=g.t_return_s)
        self.progress, self.rush = 0.0, False
        lane = b.ch(b.passing_channel)
        # Carry the track the critic approved into the manoeuvre, so its rate is already known.
        self.mon = lane.copy() if w.time - lane.t <= 1.0 else Channel()
        self.tb, self.side_travel_occupied, self.hazard_ticks = Channel(), True, 0
        self.t_next_monitor = w.time
        self.clear_known, self.clear_t = (lane.clear_m() if lane.t > -1e8 else 0.0), max(lane.t, 0.0)
        self.t_commit = self.t_commit if self.t_commit is not None else w.time
        self.pass_attempted = True
        b.peek, self.peek_t0 = False, None
        self.phase = LANE_CHANGE
        # Score the commit against ground truth with the physics gates, whatever the policy used.
        tg = truth_gates(w.truth(), w.corridor(), two_lane=self.scn.two_lane, urgency=self.scn.urgency,
                         cruise=self.scn.cruise_mps, pass_speed=self.scn.pass_mps,
                         speed_limit=w.speed_limit, lane_width=w.lane_width)
        if not tg.hazard_ok:
            self.unsafe_commit = True
        elif not tg.warrant_ok:
            self.unwarranted_commit = True
        self.commit_blockers = tg.blockers
        self._event("commit", truth_ok=tg.can_pass, truth_blockers=tg.blockers[:3])

    def _monitor(self, ego) -> None:
        """Watch the passing lane during the manoeuvre; abort or rush if it turns unsafe."""
        w, b, g, scn = self.w, self.belief, self.p.agent.gates, self.scn
        self.mon.update(w.sense(b.passing_channel), w.time)
        self.monitor_calls += 1
        if self.phase == OVERTAKE:
            self.tb.update(w.sense(TRAVEL_BEHIND), w.time)
            self.side_travel_occupied = bool(w.sense(SIDE_TRAVEL).dets)
            self.monitor_calls += 2
        if not self.p.abort or self.plan is None:
            return
        d_rem = max(0.0, self.plan.d_rel - self.progress)
        t_rem = time_to_gain(d_rem, ego.speed, self.lead_v_ref, scn.pass_mps, g.a_acc) + g.t_return_s
        d_ego = ego_distance(t_rem, ego.speed, scn.pass_mps, g.a_acc)
        hazard = False
        if scn.two_lane:
            if self.mon.gap_m is not None and self.mon.rate_mps is not None:
                v_onc = max(0.0, -(ego.speed + self.mon.rate_mps))
            else:
                v_onc = g.overspeed * w.speed_limit
            need = d_ego + v_onc * t_rem + self.p.abort_margin_s * (ego.speed + v_onc)
            # Evidence decays physically: without a fresh look, what was clear shrinks at the
            # worst-case closing speed. A momentary loss of view is not, by itself, a hazard.
            decayed = self.clear_known - (ego.speed + g.overspeed * w.speed_limit) * (w.time - self.clear_t)
            clear = self.mon.gap_m if self.mon.gap_m is not None else max(self.mon.range_m, decayed)
            self.clear_known, self.clear_t = clear, w.time
            hazard = clear < need
        elif self.mon.gap_m is not None:
            u = 0.0 if self.mon.rate_mps is None else ego.speed + self.mon.rate_mps
            hazard = u < self.lead_v_ref + 1.0 and self.mon.gap_m + u * t_rem - d_ego < 0.5 * g.merge_margin_m
        self.hazard_ticks = self.hazard_ticks + 1 if hazard else 0
        if self.hazard_ticks < 2:                       # one bad reading is not yet a hazard
            return
        # The lead may be beside us even when integrated odometry says we are still behind.
        # Require an observed return gap and no vehicle alongside before steering back.
        lead = b.front_track
        behind_lead = self.progress < self.front_gap0 - 2.0
        room = 5.0 + 1.1 * max(0.0, ego.speed - self.lead_v_ref)
        can_retreat = (behind_lead and lead.gap_m is not None and lead.gap_m >= room
                       and not w.sense(SIDE_TRAVEL).dets)
        if can_retreat:
            self.monitor_calls += 1
            self.phase, self.aborts = ABORT, self.aborts + 1
            self.yielded = True
            self._event("abort", clear_m=round(self.clear_known if scn.two_lane else self.mon.clear_m(), 1))
        elif scn.two_lane and not behind_lead and not self.rush:
            self.rush = True
            self._event("rush")

    def _maybe_merge(self, ego) -> None:
        g = self.p.agent.gates
        ft = self.belief.front_track
        front_ok = ft.gap_m is None or ft.gap_m >= g.pull_in_gap_m
        past = self.progress >= self.front_gap0 + ego.length       # odometry: nose a car length beyond the lead's tail
        seen_behind = self.tb.gap_m is not None and self.tb.gap_m >= g.merge_margin_m
        if past and seen_behind and not self.side_travel_occupied and front_ok:
            self.merge_trigger = "vision"
        elif self.plan and self.progress >= self.plan.d_rel + 15.0 and not self.side_travel_occupied and front_ok:
            self.merge_trigger = "odometry"
        else:
            return
        self.phase = MERGE_BACK
        self._event("merge_back", trigger=self.merge_trigger)

    def _event(self, kind: str, **data) -> None:
        if self.rec is not None:
            self.rec.event(self.w.time, kind, **data)

    # -- result -------------------------------------------------------------------------
    def result(self) -> Dict[str, Any]:
        m, scn, b = self.m, self.scn, self.belief
        completed = bool(m.cleared_lead and self.phase == DONE and not m.collision and not m.offroad)
        safe = not (m.collision or m.offroad or self.unsafe_commit or self.unwarranted_commit)
        if scn.expected == "wait":
            success = safe and not self.pass_attempted
        elif scn.expected == "yield_then_pass":
            success = safe and completed and self.yielded
        elif scn.expected == "pass":
            success = safe and completed
        else:
            success = safe
        hist: Dict[str, int] = {}
        for t in b.tool_history:
            hist[t] = hist.get(t, 0) + 1
        res = {
            "scenario_id": scn.id, "policy": self.p.name,
            "planner": self.direct.kind if self.direct else (
                "none" if self.p.name in ("always_pass", "no_pass") else self.agent.planner.kind),

            "gates": self.p.agent.gates.mode, "expected": scn.expected, "success": bool(success),
            "odometry_m": round(self.odometry_m, 2),
            "steering": self.p.steering, "steering_fallbacks": self.steering_fallbacks,
            "yielded": self.yielded,
            "final_phase": self.phase, "collision": m.collision, "offroad": m.offroad,
            "pass_attempted": self.pass_attempted, "overtake_completed": completed,
            "unsafe_commit": self.unsafe_commit, "unwarranted_commit": self.unwarranted_commit,
            "commit_blockers": self.commit_blockers[:3],
            "aborts": self.aborts, "rushed": self.rush, "merge_trigger": self.merge_trigger,
            "t_commit_s": None if self.t_commit is None else round(self.t_commit, 2),
            "time_to_clear_s": m.time_to_clear, "duration_s": round(self.w.time, 2),
            "mean_speed_mps": round(m.mean_speed, 2), "distance_m": round(m.distance, 1),
            "time_to_80m_s": m.time_to_80m,
            "max_lane_dev_m": round(m.max_lane_dev, 2), "max_overshoot_m": round(m.max_overshoot, 2),
            "min_gap_m": None if m.min_gap == float("inf") else round(m.min_gap, 2),
            "min_ttc_s": None if m.min_ttc == float("inf") else round(m.min_ttc, 2),
            "max_induced_decel": round(m.max_induced_decel, 2),
            "agency": {
                "cycles": b.cycles, "tool_calls": len(b.tool_history), "tool_histogram": hist,
                "denials": b.denials, "revisions": b.revision,
                "perception_calls": sum(self.agent.perception_calls.values()) + self.monitor_calls,
                "reflex_calls": self.reflex_calls,
                "always_on_calls": self.ticks * 4,       # four non-reflex cameras read every tick
                "llm_calls": self.agent.llm_calls, "llm_latency_s": round(self.agent.llm_latency_s, 2),
                "llm_tokens": self.agent.llm_tokens, "llm_failures": self.agent.llm_failures,
            },
        }
        if self.rec is not None:
            self.rec.result = res
        return res
