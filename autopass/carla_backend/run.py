"""Run catalog scenarios in CARLA for one town (one simulator process per town).

  python -m autopass.carla_backend.run --town Town04 --scenarios s01,s17 --policy autopass --out runs/demo

Each run writes  <out>/<policy>/<scenario>/  result.json  telemetry.json.gz  video.mp4  sprite.jpg
Exit code 3 means the simulator stopped answering; the batch driver restarts it and resumes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.episode import Episode
from ..core.policies import POLICIES, named_policy
from ..core.scenario import Scenario, catalog
from ..core.telemetry import Recorder
from ..core.types import FRONT
from .client import connect, load_town
from .recorder import Hud, VideoWriter, compose


def _hud(ep: Episode, world, scn: Scenario, label: str) -> Hud:
    e = world.ego()
    cyc = ep.rec.cycles[-1] if ep.rec and ep.rec.cycles else {}
    gates = dict((cyc.get("gates") or {}).get("checks", {}))
    tools = [t["tool"] for t in cyc.get("tools", [])]
    verdict = ""
    for t in cyc.get("tools", []):
        if t["tool"] in ("propose_pass", "hold"):
            verdict = t["obs"]
    if ep.p.name in ("no_pass", "always_pass"):
        verdict = cyc.get("note", "never-pass baseline: no deliberation") if cyc else "never-pass baseline: no deliberation"
    boxes = []
    w = world.lane_width
    for o in world._objects("front_wide"):
        if o.bbox is None:
            continue
        gap = o.s_min - 0.5 * world.ego_len
        if abs(o.d) < 0.5 * w and gap >= 0:
            boxes.append((o.bbox, "lead", f"lead {gap:.0f} m"))
        elif 0.5 * w <= o.d < 1.5 * w:
            boxes.append((o.bbox, "oncoming" if scn.two_lane else "passing",
                          f"{'oncoming lane' if scn.two_lane else 'passing lane'} {max(0.0, gap):.0f} m"))
        else:
            boxes.append((o.bbox, "other", f"{max(0.0, gap):.0f} m"))
    return Hud(title=f"{scn.id}   {scn.town}   {label}", t=world.time, phase=ep.phase, speed=e.speed,
               target_speed=ep.v_target, gates=gates, tools=tools, verdict=verdict,
               planner=getattr(ep.agent.planner, "kind", "rule"), boxes=boxes)


def run_one(client, scn: Scenario, policy_name: str, out: Path, *, planner: Any = None, direct: Any = None,
            video: bool = True, seed: int = 0, label: Optional[str] = None, perception: str = "segdepth", steering: str = "waypoint", min_follow_s: float = 2.5) -> Dict[str, Any]:
    from .world import CarlaWorld
    out.mkdir(parents=True, exist_ok=True)
    policy = named_policy(policy_name, steering=steering)
    # Let the CARLA drivetrain settle before any candidate can actuate. Baselines share the window.
    policy.min_follow_s = max(policy.min_follow_s, min_follow_s)
    world = CarlaWorld(client, scn, video=video, seed=seed, perception=perception)
    writer = VideoWriter(out / "video.mp4") if video else None
    try:
        rec = Recorder({"scenario": scn.to_dict(), "backend": "carla", "seed": seed, "policy": policy_name,
                        "lane_width": round(world.lane_width, 2), "passing_side": world.passing_side,
                        "corridor_m": world.corridor_len, "perception": world.perception,
                        "map": world.map.name.split("/")[-1]})
        ep = Episode(world, scn, policy, planner=planner, direct=direct, recorder=rec)
        if writer is not None:
            tag = label or policy_name.replace("_", "-")

            def capture(e: Episode) -> None:
                front, over = world.images()
                if front is not None:
                    writer.write(compose(front, over, _hud(e, world, scn, tag)))
            ep.on_tick = capture
        t0 = time.time()
        result = ep.run()
        result.update(backend="carla", seed=seed, perception=world.perception, policy_label=policy_name,
                      wall_s=round(time.time() - t0, 1), passing_side=world.passing_side,
                      collision_with=world.collision_with, min_follow_s=policy.min_follow_s)
        if world.detector:
            result["learned_sensor"] = {
                "model": "NanoDet-m-plus-1.5x-416",
                "inferences": world.detector.inferences,
                "inference_wall_s": round(world.detector.inference_s, 3),
            }
        if writer is not None:
            result["video"] = writer.close(out / "sprite.jpg")
            writer = None
        rec.result = result
        rec.save(out / "telemetry.json.gz")
        (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        return result
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass
        world.close()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run AutoPass scenarios in CARLA (one town per process).")
    ap.add_argument("--town", required=True)
    ap.add_argument("--scenarios", default="all", help="all | comma-separated id prefixes (e.g. s01,s17)")
    ap.add_argument("--policy", default="autopass", choices=POLICIES)
    ap.add_argument("--planner", default="rule", choices=("rule", "llm_plan", "llm_react"))
    ap.add_argument("--out", type=Path, default=Path("runs/carla"))
    ap.add_argument("--video", default="full", choices=["full", "none"])
    ap.add_argument("--steering", choices=("waypoint", "imitation"), default="waypoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-town", action="store_true", help="load the map on a dedicated simulator")
    ap.add_argument("--perception", choices=("segdepth", "rgbd"), default="segdepth")
    ap.add_argument("--redo", action="store_true", help="re-run scenarios that already have a result")
    ap.add_argument("--label", default=None)
    ap.add_argument("--min-follow", type=float, default=2.5, help="minimum scored rolling-start observation time")
    args = ap.parse_args(argv)
    if not 0 <= args.min_follow <= 10:
        ap.error("--min-follow must be in [0, 10]")

    cat = catalog()
    want = None if args.scenarios == "all" else args.scenarios.split(",")
    todo = [s for k, s in cat.items() if s.town == args.town and (want is None or any(k.startswith(w) for w in want))]
    if not todo:
        ap.error(f"no scenarios match for {args.town}")
    planner = None
    if args.planner != "rule":
        from ..llm.client import make_planner
        planner = make_planner(args.planner)
    label = args.label or "_".join([args.policy] +
        ([args.planner] if args.planner != "rule" else []) +
        ([args.perception] if args.perception != "segdepth" else []) +
        ([args.steering] if args.steering != "waypoint" else []))
    client = connect()
    load_town(client, args.town, reload=args.load_town)
    summary = []
    for scn in todo:
        out = args.out / label / scn.id
        if (out / "result.json").is_file() and not args.redo:
            summary.append(json.loads((out / "result.json").read_text(encoding="utf-8")))
            continue
        t0 = time.time()
        try:
            r = run_one(client, scn, args.policy, out, planner=planner, video=args.video == "full",
                        seed=args.seed, label=label, perception=args.perception,
                        steering=args.steering, min_follow_s=args.min_follow)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            msg = str(e)
            if "time-out" in msg or "timeout" in msg.lower() or "connection" in msg.lower():
                print(f"[{label}] {scn.id}: simulator not responding ({msg[:60]})", flush=True)
                return 3
            r = {"scenario_id": scn.id, "policy_label": label, "error": msg[:200], "success": False}
            out.mkdir(parents=True, exist_ok=True)
            (out / "error.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        summary.append(r)
        print(f"[{label}] {scn.id:36} ok={r.get('success')} phase={r.get('final_phase')} coll={r.get('collision')} "
              f"unsafe={r.get('unsafe_commit')} commit={r.get('t_commit_s')} clear={r.get('time_to_clear_s')} "
              f"v={r.get('mean_speed_mps')} dev={r.get('max_lane_dev_m')} ({time.time() - t0:.0f}s)", flush=True)
    return 2 if any("error" in r for r in summary) else 0


if __name__ == "__main__":
    sys.exit(main())
