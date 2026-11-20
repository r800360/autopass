"""The deliberation loop: plan -> tool -> observe -> propose -> critic -> remember.

One call to `deliberate` is one cycle. The planner chooses tools, tools mutate the belief,
and only an approval from the deterministic critic turns into a "pass" decision.

Time model:
  sync      the world is frozen while the agent thinks (how the first release ran)
  realtime  thinking costs time: the caller's `wait` advances the world by the planner's
            measured latency and by a fixed cost per perception call, so evidence really
            does go stale while the agent deliberates
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import critic
from .belief import Belief
from .gates import GateConfig
from .planner import DirectLLM, RulePlanner, View
from .tools import LEGACY_TOOLS, SENSING, TOOLS, camera_for, run_tool
from .types import World

Wait = Callable[[float], None]


@dataclass
class AgentConfig:
    gates: GateConfig = field(default_factory=GateConfig)
    use_critic: bool = True            # False = ablation: a proposal is taken at face value
    auto_propose: bool = False         # first-release "greedy backstop": code proposes if the LLM does not
    max_tools: int = 8                 # tool calls per cycle
    realtime: bool = False
    perception_latency_s: float = 0.05  # cost of one sensing tool in realtime mode
    max_model_calls: int = 24          # per episode, bounds wall time and spend
    legacy_tools: bool = False         # first-release palette (no blind-spot tool, no peek)

    @property
    def palette(self) -> Tuple[str, ...]:
        return LEGACY_TOOLS if self.legacy_tools else tuple(TOOLS)


class PassingAgent:
    def __init__(self, planner: Any = None, cfg: Optional[AgentConfig] = None):
        self.planner = planner or RulePlanner()
        self.cfg = cfg or AgentConfig()
        self.perception_calls: Dict[str, int] = {}
        self.llm_calls = 0
        self.llm_latency_s = 0.0
        self.llm_tokens = 0
        self.llm_failures = 0

    # -- one deliberation cycle ---------------------------------------------------------
    def deliberate(self, world: World, b: Belief, wait: Optional[Wait] = None) -> Dict[str, Any]:
        cfg = self.cfg
        wait = wait if (wait and cfg.realtime) else (lambda s: None)
        t_start = world.time
        b.cycles += 1
        called: List[str] = []
        log: List[Dict[str, Any]] = []
        rationales: List[str] = []
        verdicts: List[Dict[str, Any]] = []
        decision, note = "wait", ""
        proposed = done = False
        batch = getattr(self.planner, "kind", "") == "llm_plan"

        while not done and len(called) < cfg.max_tools:
            view = View(snapshot=b.snapshot(world.time),
                        stale=critic.stale_channels(b, world.time, cfg.gates),
                        two_lane=b.two_lane, urgency=b.urgency, allowed=cfg.palette,
                        called=list(called), observations=[e["obs"] for e in log],
                        last_rejection=b.rejection_reasons[-1] if b.rejection_reasons else None,
                        budget_left=cfg.max_tools - len(called))
            model = getattr(self.planner, "kind", "").startswith("llm_")
            if model and self.llm_calls >= cfg.max_model_calls:
                note = "model request budget exhausted"
                break
            step = self.planner.next(view)
            valid = [tool for tool in step.tools if tool in cfg.palette]
            if not valid:
                step.tools = ["hold"]
                step.rationale = "no valid action"
            else:
                step.tools = valid
            if step.latency_s or step.info:
                self.llm_calls += 1
                self.llm_latency_s += step.latency_s
                self.llm_tokens += int(step.info.get("tokens", 0))
                self.llm_failures += bool(step.info.get("failed"))
            rationales.append(f"{'+'.join(step.tools)}: {step.rationale}")
            wait(step.latency_s)

            for tool in step.tools:
                if len(called) >= cfg.max_tools:
                    break
                called.append(tool)
                if tool == "hold":
                    log.append({"tool": "hold", "obs": step.rationale, "t": round(world.time, 2)})
                    note, done = step.rationale, True
                    break
                if tool == "propose_pass":
                    if proposed:                       # one proposal per cycle
                        done = True
                        break
                    proposed = True
                    ok, reason, vd = self._judge(world, b)
                    verdicts.append(vd)
                    log.append({"tool": "propose_pass", "obs": ("APPROVE: " if ok else "REJECT: ") + reason,
                                "t": round(world.time, 2)})
                    note = reason
                    if ok:
                        decision, b.intent, done = "pass", "commit_pass", True
                    else:
                        b.remember_denial(reason)
                    break                              # re-plan with the verdict in view
                obs = run_tool(tool, world, b)
                cam = camera_for(tool, b.two_lane)
                if cam:
                    self.perception_calls[cam] = self.perception_calls.get(cam, 0) + 1
                    wait(cfg.perception_latency_s)
                log.append({"tool": tool, "obs": obs, "t": round(world.time, 2)})
            if batch:
                note = note or "tool plan ended without a pass proposal"
                break

        if decision != "pass" and cfg.auto_propose and b.urgency in ("high", "medium"):
            ok, reason, vd = self._judge(world, b)       # the first release's backstop
            if ok:
                decision, b.intent = "pass", "commit_pass"
                note = "auto-propose: " + reason
                verdicts.append(vd)
                log.append({"tool": "propose_pass", "obs": "APPROVE (auto): " + reason, "t": round(world.time, 2)})

        gates = critic.evaluate(critic.gate_input(b, world.ego(), world.speed_limit), cfg.gates)
        return {"t": round(t_start, 2), "t_end": round(world.time, 2), "cycle": b.cycles,
                "tools": log, "rationales": rationales, "verdicts": verdicts,
                "decision": decision, "note": note, "planner": getattr(self.planner, "kind", "?"),
                "gates": gates.as_dict(), "belief": b.snapshot(world.time)}

    def _judge(self, world: World, b: Belief) -> Tuple[bool, str, Dict[str, Any]]:
        if not self.cfg.use_critic:
            return True, "no critic (ablation): proposal accepted as is", {"approved": True, "reason": "no critic"}
        v = critic.verify(b, world.ego(), world.time, world.speed_limit, self.cfg.gates)
        return v.approved, v.reason, v.as_dict()

    # -- ablation: an LLM decides directly, no critic, no gates ---------------------------
    def decide_direct(self, llm: DirectLLM, world: World, b: Belief, wait: Optional[Wait] = None) -> Dict[str, Any]:
        wait = wait if (wait and self.cfg.realtime) else (lambda s: None)
        t_start = world.time
        b.cycles += 1
        log = []
        if self.llm_calls >= self.cfg.max_model_calls:
            return {"t": round(t_start, 2), "cycle": b.cycles, "tools": [], "decision": "wait",
                    "note": "model request budget exhausted", "planner": llm.kind, "belief": b.snapshot(t_start)}
        for tool in SENSING:
            obs = run_tool(tool, world, b)
            cam = camera_for(tool, b.two_lane)
            if cam:
                self.perception_calls[cam] = self.perception_calls.get(cam, 0) + 1
                wait(self.cfg.perception_latency_s)
            log.append({"tool": tool, "obs": obs, "t": round(world.time, 2)})
        decision, reason, latency, info = llm.decide(b.snapshot(world.time), b.two_lane, b.urgency)
        self.llm_calls += 1
        self.llm_latency_s += latency
        self.llm_tokens += int(info.get("tokens", 0))
        self.llm_failures += bool(info.get("fallback"))
        wait(latency)
        if decision == "pass":
            b.intent = "commit_pass"
        gates = critic.evaluate(critic.gate_input(b, world.ego(), world.speed_limit), self.cfg.gates)
        return {"t": round(t_start, 2), "t_end": round(world.time, 2), "cycle": b.cycles, "tools": log,
                "rationales": [reason], "verdicts": [], "decision": decision, "note": reason,
                "planner": llm.kind, "gates": gates.as_dict(), "belief": b.snapshot(world.time)}
