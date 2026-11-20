"""The mutable belief DSL: what the agent currently believes, how old each belief is,
and what it remembers. Every role reads and writes this one record.

  belief     per-channel gap, gap rate, reach        written by sensing tools
  freshness  sim time each channel was last sensed   read by the critic
  memory     denials, rejection reasons, tool log    written by the agent loop
  plan       intent, peek request                    written by planner/critic
  revision   bumped on every mutation                stamps each trace record
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from .types import FRONT, ONCOMING, PASSING_AHEAD, REAR, SenseResult

NEVER = -1e9
GAP_CHANNELS = (FRONT, PASSING_AHEAD, ONCOMING, REAR)


@dataclass
class Channel:
    """Nearest object in one region of interest, tracked across readings."""
    gap_m: Optional[float] = None      # None = nothing detected within range_m
    rate_mps: Optional[float] = None   # d(gap)/dt, negative = closing; None = not yet observable
    range_m: float = 0.0               # sensing reach (occlusion-limited where applicable)
    kind: Optional[str] = None
    t: float = NEVER                   # sim time of the last reading
    n: int = 0                         # consecutive readings of the same object
    _t0: float = NEVER                 # when this object was first seen
    _gap0: float = 0.0
    _rate: Optional[float] = None
    track_id: Optional[int] = None
    order: int = 0

    def age(self, now: float) -> float:
        return now - self.t

    def fresh(self, now: float, tau: float) -> bool:
        return -1e-9 <= self.age(now) <= tau + 1e-9

    def update(self, res: SenseResult, now: float) -> None:
        """Fold in a new reading. A rate needs two looks at the same object >= 0.25 s apart."""
        if now <= self.t + 1e-9:
            return  # rereading a scene at the same time adds no independent evidence
        det = res.nearest
        new = det.gap_m if det else None
        dt = now - self.t
        track_id = det.track_id if det else None
        same = (track_id == self.track_id and new is not None and self.gap_m is not None and 0.0 < dt <= 1.6
                and not (self._rate is not None and abs(new - (self.gap_m + self._rate * dt)) > 5.0))
        if same:
            span = now - self._t0
            if span < 0.5 or self._rate is None:   # short history: two-point estimate from first sight
                self._rate = (new - self._gap0) / span
            else:                                  # then an EMA with a 0.5 s time constant
                g = min(1.0, dt / 0.5)
                self._rate += g * ((new - self.gap_m) / dt - self._rate)
            self.n += 1
        else:                                      # first sight, lost, or a different object
            self._rate, self._t0, self._gap0 = None, now, (new or 0.0)
            self.n = 1 if new is not None else 0
        self.gap_m, self.range_m, self.t = new, res.range_m, now
        self.kind = det.kind if det else None
        self.track_id = track_id
        self.rate_mps = self._rate if (self._rate is not None and now - self._t0 >= 0.25 - 1e-9) else None

    def clear_m(self) -> float:
        """Distance known to be free: the gap if something is there, else the sensing reach."""
        return self.gap_m if self.gap_m is not None else self.range_m

    def copy(self) -> "Channel":
        return replace(self)


@dataclass
class Belief:
    two_lane: bool = False
    urgency: str = "high"              # low | medium | high
    cruise_mps: float = 14.0           # speed the ego wants to travel
    pass_mps: float = 20.0             # speed it uses while overtaking

    channels: Dict[str, Channel] = field(default_factory=lambda: {c: Channel() for c in GAP_CHANNELS})
    # Always-on reflex tracker for the lead (drives cruise control and emergency braking).
    # The agent must still call sense_front to commit it as decision evidence.
    front_track: Channel = field(default_factory=Channel)
    lead_speed_mps: Optional[float] = None
    side_occupied: Optional[bool] = None
    side_t: float = NEVER
    side_order: int = 0
    corridor_clear_m: float = 0.0
    corridor_legal: Optional[bool] = None
    corridor_t: float = NEVER
    semantic_veto: Optional[str] = None     # set by a semantic check; can only block, never allow
    semantic_t: float = NEVER

    intent: str = "deliberate"         # deliberate | commit_pass | follow
    peek: bool = False                 # request: bias toward the passing side to see past the lead

    denials: int = 0
    rejection_reasons: List[str] = field(default_factory=list)
    cycles: int = 0
    tool_history: List[str] = field(default_factory=list)
    revision: int = 0

    def ch(self, name: str) -> Channel:
        return self.channels[name]

    @property
    def passing_channel(self) -> str:
        """The region a pass pulls into: the oncoming lane on a two-lane road."""
        return ONCOMING if self.two_lane else PASSING_AHEAD

    def touch(self) -> None:
        self.revision += 1

    def remember_denial(self, reason: str) -> None:
        self.denials += 1
        self.rejection_reasons.append(reason)
        self.touch()

    def snapshot(self, now: float) -> Dict[str, Any]:
        """Compact, JSON-safe view for the planner prompt and the trace."""
        def r(x, nd=1):
            return None if x is None else round(x, nd)

        out: Dict[str, Any] = {"revision": self.revision}
        for name in (FRONT, self.passing_channel, REAR):
            c = self.channels[name]
            out[name] = {"gap_m": r(c.gap_m), "rate_mps": r(c.rate_mps), "range_m": r(c.range_m, 0),
                         "age_s": r(min(c.age(now), 99.0), 2)}
        out["lead_speed_mps"] = r(self.lead_speed_mps)
        out["side"] = {"occupied": self.side_occupied, "age_s": r(min(now - self.side_t, 99.0), 2)}
        out["corridor"] = {"clear_m": r(self.corridor_clear_m, 0), "legal": self.corridor_legal,
                           "age_s": r(min(now - self.corridor_t, 99.0), 2)}
        if self.semantic_veto:
            out["semantic_veto"] = self.semantic_veto
        out.update(intent=self.intent, peek=self.peek, denials=self.denials, cycles=self.cycles,
                   last_rejection=(self.rejection_reasons[-1] if self.rejection_reasons else None))
        return out
