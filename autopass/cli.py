"""Command-line entry points for replay, benchmarking and trip simulation."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .core.policies import SIM_POLICIES, named_policy
from .core.scenario import catalog
from .sim.kinematic import run
from .sim.sensors import PerceptionProfile

SEED = 20261119
PROFILES = ("oracle", "segdepth", "degraded")
PLANNERS = ("rule", "llm_plan", "llm_react", "llm_direct")


def case(prefix: str):
    candidates = [s for sid, s in catalog().items() if sid.startswith(prefix)]
    if len(candidates) != 1:
        raise ValueError(f"Use an unambiguous scenario ID or prefix: {prefix!r}")
    return candidates[0]


def experiment(scenario_id: str, policy_name: str = "autopass", urgency: str | None = None,
               profile: str = "oracle", seed: int = 0, steering: str = "waypoint",
               planner_kind: str = "rule", realtime: bool = False, model_calls: int = 24):
    scenario = case(scenario_id)
    if urgency is not None:
        scenario = replace(scenario, urgency=urgency, expected="?")
    planner = direct = None
    if planner_kind != "rule":
        from .llm.client import make_planner
        if planner_kind == "llm_direct":
            direct = make_planner(planner_kind)
            policy_name = "llm_direct"
        else:
            planner = make_planner(planner_kind)
    policy = named_policy(policy_name, steering=steering, realtime=realtime)
    if model_calls < 1:
        raise ValueError("Model call limit must be positive.")
    policy.agent.max_model_calls = model_calls
    return run(scenario, policy, planner=planner, direct=direct, record=True,
               seed=seed, perception=PerceptionProfile.preset(profile))


def browser_run(payload: str) -> str:
    args = json.loads(payload)
    _, record = experiment(args["scenario"], args["policy"], args.get("urgency"),
                           args.get("profile", "oracle"), int(args.get("seed", 0)),
                           args.get("steering", "waypoint"))
    return json.dumps(record.to_dict(), allow_nan=False, separators=(",", ":"))


def write_json(path: Path | None, data: dict) -> None:
    value = json.dumps(data, indent=2, allow_nan=False) + "\n"
    if path is None:
        print(value, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")


def parser():
    cli = argparse.ArgumentParser(description="AutoPass-Gen simulator")
    sub = cli.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="list catalog scenarios")
    perform = sub.add_parser("run", help="run one kinematic scenario")
    perform.add_argument("scenario")
    perform.add_argument("--policy", choices=SIM_POLICIES, default="autopass")
    perform.add_argument("--planner", choices=PLANNERS, default="rule")
    perform.add_argument("--realtime", action="store_true", help="advance simulation during measured model latency")
    perform.add_argument("--model-calls", type=int, default=24)
    perform.add_argument("--urgency", choices=("low", "medium", "high"))
    perform.add_argument("--profile", choices=PROFILES, default="oracle")
    perform.add_argument("--steering", choices=("waypoint", "imitation"), default="waypoint")
    perform.add_argument("--seed", type=int, default=0)
    perform.add_argument("--trace", type=Path, help="write JSON or compressed .json.gz telemetry")
    bench = sub.add_parser("bench", help="run paired deterministic or sampled kinematic campaigns")
    bench.add_argument("--policies", nargs="+", choices=SIM_POLICIES, default=list(SIM_POLICIES[:4]))
    bench.add_argument("--count", type=int, default=0, help="number of sampled trials per policy (0 for catalog)")
    bench.add_argument("--seed", type=int, default=SEED)
    bench.add_argument("--workers", type=int, default=1)
    bench.add_argument("--profile", choices=PROFILES, default="oracle")
    bench.add_argument("--steering", choices=("waypoint", "imitation"), default="waypoint")
    bench.add_argument("--out", type=Path)
    trip = sub.add_parser("trip", help="simulate routing over a five-node road graph")
    trip.add_argument("request")
    trip.add_argument("--llm", action="store_true", help="use structured inference to extract a map-grounded request")
    trip.add_argument("--fixed-route", action="store_true")
    trip.add_argument("--out", type=Path)
    return cli


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.action == "list":
            for scenario in catalog().values():
                print(f"{scenario.id:38s} {scenario.narrative}")
        elif args.action == "run":
            result, record = experiment(args.scenario, args.policy, args.urgency, args.profile,
                                        args.seed, args.steering, args.planner, args.realtime, args.model_calls)
            if args.trace:
                record.save(args.trace)
            write_json(None, result)
        elif args.action == "bench":
            from .evaluation import evaluate, sample_cases
            if args.count < 0 or args.workers < 1 or len(set(args.policies)) != len(args.policies):
                raise ValueError("Use a nonnegative count, positive workers, and distinct policies.")
            cases = sample_cases(args.seed, args.count) if args.count else list(catalog().values())
            result = evaluate(cases, args.policies, args.seed, args.profile, args.steering, args.workers)
            write_json(args.out, result)
        elif args.action == "trip":
            from .trip import navigate, parse_request
            complete = None
            if args.llm:
                from .llm.client import complete_json
                complete = complete_json
            write_json(args.out, navigate(parse_request(args.request, complete), replan=not args.fixed_route))
    except ValueError as exc:
        cli.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
