"""Compact per-run telemetry: everything needed to replay, re-render and audit a run.

One run is one JSON document of a few hundred kilobytes (tens when gzipped):
  meta     scenario, policy, backend, lane geometry, code version
  actors   id -> role/kind/size                (static)
  frames   [t, phase, ego s/d/yaw/v, d_target, v_target] + per-actor [id, s, d, yaw, v]
  cycles   the agent's deliberation trace      (tools, rationales, verdicts, belief)
  events   commit / abort / merge / collision  (timestamped)
  result   the metrics

Video is optional and derived; this file is the record. A 60 GB folder of PNG frames
carries less information than these files do.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import Truth

PHASES = ["follow", "lane_change", "overtake", "merge_back", "abort", "done"]
SCHEMA = 2


def _r(x: float, n: int = 2) -> float:
    return round(float(x), n)


class Recorder:
    def __init__(self, meta: Optional[Dict[str, Any]] = None, every: int = 2):
        self.meta: Dict[str, Any] = dict(meta or {})
        self.every = max(1, every)
        self.actors: Dict[str, Dict[str, Any]] = {}
        self.frames: List[list] = []
        self.cycles: List[dict] = []
        self.events: List[dict] = []
        self.result: Dict[str, Any] = {}
        self._n = 0

    def frame(self, truth: Truth, phase: str, d_target: float, v_target: float, extra: Optional[list] = None) -> None:
        self._n += 1
        if (self._n - 1) % self.every:
            return
        ego = truth.get("ego")
        others = []
        for a in truth.actors:
            key = str(a.id)
            if key not in self.actors:
                self.actors[key] = {"role": a.role, "kind": a.kind, "len": _r(a.length), "wid": _r(a.width)}
            if a.role != "ego":
                others.append([a.id, _r(a.s), _r(a.d), _r(a.yaw, 3), _r(a.v)])
        row = [_r(truth.t), PHASES.index(phase), _r(ego.s), _r(ego.d), _r(ego.yaw, 3), _r(ego.v),
               _r(d_target), _r(v_target), others]
        if extra:
            row.append(extra)
        self.frames.append(row)

    def event(self, t: float, kind: str, **data: Any) -> None:
        self.events.append({"t": _r(t), "kind": kind, **data})

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": SCHEMA, "meta": self.meta, "actors": self.actors, "phases": PHASES,
                "frame_fields": ["t", "phase", "s", "d", "yaw", "v", "d_target", "v_target", "others[id,s,d,yaw,v]"],
                "frames": self.frames, "cycles": self.cycles, "events": self.events, "result": self.result}

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.to_dict(), separators=(",", ":"))
        if path.suffix == ".gz":
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(data)
        else:
            path.write_text(data, encoding="utf-8")
        return path


def load(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))
