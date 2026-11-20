"""Paired, reproducible simulation campaigns and denominator-aware summaries."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import random
from statistics import median
from typing import Iterable

from .core.policies import named_policy
from .core.scenario import Scenario, sample
from .sim.kinematic import run
from .sim.sensors import PerceptionProfile


def wilson(successes: int, n: int) -> list[float] | None:
    """95% Wilson interval, useful for simulated draw rates within a fixed distribution."""
    if n == 0:
        return None
    z = 1.959964
    p = successes / n
    divisor = 1 + z*z/n
    centre = (p + z*z/(2*n)) / divisor
    radius = (z/divisor) * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return [round(max(0.0, centre - radius), 5), round(min(1.0, centre + radius), 5)]


def summarize(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    done = [r for r in rows if "error" not in r]
    n = len(done)
    coll = sum(r["collision"] for r in done)
    safe = [r for r in done if not r["collision"] and not r["offroad"]]
    reaching = [r["time_to_80m_s"] for r in safe if r.get("time_to_80m_s") is not None]
    return {
        "attempted": len(rows), "scored": n, "run_errors": len(rows) - n,
        "expected_behavior": sum(r["success"] for r in done if r["expected"] != "?"),
        "labeled_cases": sum(r["expected"] != "?" for r in done),
        "overtakes": sum(r["overtake_completed"] for r in done),
        "collisions": coll, "collision_interval_wilson": wilson(coll, n),
        "offroad": sum(r["offroad"] for r in done),
        "unsafe_commits": sum(r["unsafe_commit"] for r in done),
        "unwarranted_commits": sum(r["unwarranted_commit"] for r in done),
        "aborts": sum(r["aborts"] for r in done),
        "clear_80m": len(reaching),
        "median_time_to_80m_s": round(median(reaching), 2) if reaching else None,
    }


def paired_time_80m(left: list[dict], right: list[dict]) -> dict:
    def times(rows):
        return {(r["scenario_id"], r["seed"]): r["time_to_80m_s"] for r in rows
                if r.get("time_to_80m_s") is not None and not (r["collision"] or r["offroad"])}
    a, b = times(left), times(right)
    paired = [b[k] - a[k] for k in a.keys() & b.keys()]
    return {"both_clear_80m": len(paired), "median_seconds_saved": round(median(paired), 2) if paired else None}


def _evaluate(job):
    scenario, label, seed, profile, steering = job
    policy = named_policy(label, steering=steering)
    r, _ = run(scenario, policy=policy, seed=seed, perception=PerceptionProfile.preset(profile))
    return r


def evaluate(cases: list[Scenario], policies: list[str], seed: int, profile: str,
             steering: str = "waypoint", workers: int = 1) -> dict:
    jobs = [(scenario, policy, seed+i, profile, steering)
            for policy in policies for i, scenario in enumerate(cases)]
    if workers == 1:
        rows = list(map(_evaluate, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_evaluate, jobs, chunksize=16))
    groups = {p: [r for r in rows if r["policy"] == p] for p in policies}
    return {"schema": 1, "backend": "kinematic", "seed": seed, "profile": profile, "steering": steering,
            "scenario_count": len(cases), "summary": {p: summarize(rs) for p, rs in groups.items()},
            "paired_vs_never_pass": {p: paired_time_80m(rs, groups["no_pass"]) for p, rs in groups.items()
                                     if p != "no_pass"} if "no_pass" in groups else {},
            "rows": rows}


def sample_cases(seed: int, n: int) -> list[Scenario]:
    rng = random.Random(seed)
    return [sample(rng, i) for i in range(n)]
