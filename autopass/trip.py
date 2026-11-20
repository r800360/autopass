"""Trip-level simulator over a small, explicit road graph."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import heapq
import math
import re

from .core.episode import Policy
from .core.scenario import catalog
from .sim.kinematic import run

CADENCE = {"high": 0.5, "medium": 0.75, "low": 1.25}
NODES = ("Depot", "Market", "Main", "Bypass", "Clinic")


@dataclass(frozen=True)
class TripRequest:
    origin: str
    destination: str
    urgency: str = "medium"
    deadline_s: int | None = None


@dataclass(frozen=True)
class Road:
    source: str
    target: str
    length_m: float
    limit_mps: float
    corridor: str
    scenario: str

    @property
    def free_s(self) -> float:
        return self.length_m / self.limit_mps


ROADS = (
    Road("Depot", "Market", 70, 12, "Main", "s24"),
    Road("Market", "Main", 80, 13, "Main", "s24"),
    Road("Main", "Clinic", 85, 13, "Main", "s32"),
    Road("Market", "Bypass", 92, 11, "Bypass", "s06"),
    Road("Bypass", "Clinic", 100, 11, "Bypass", "s08"),
)


def parse_request(text: str, complete=None) -> TripRequest:
    """Ground place names in the simulated map and reject unknown requests."""
    if complete:
        s = {"type": "string", "enum": list(NODES)}
        fields = {"origin": s, "destination": s, "urgency": {"type": "string", "enum": list(CADENCE)},
                  "deadline_s": {"type": ["integer", "null"]}}
        obj, _ = complete("Extract a request for this simulated road graph. Use seconds for deadline.",
                          text[:500], {"type": "object", "properties": fields,
                                       "required": list(fields), "additionalProperties": False})
        request = TripRequest(**obj)
    else:
        names = "|".join(NODES)
        match = re.search(rf"\b(?:from\s+)?({names})\s+to\s+({names})\b", text, re.I)
        if not match:
            raise ValueError("Specify a route such as 'Depot to Clinic, urgent, within 40 seconds'.")
        places = {n.lower(): n for n in NODES}
        deadline = re.search(r"\b(\d{1,5})\s*(?:seconds?|secs?|s)\b", text, re.I)
        urgency = "medium"
        if re.search(r"\b(?:urgent|asap|hurry)\b", text, re.I):
            urgency = "high"
        if re.search(r"\b(?:no rush|low urgency)\b", text, re.I):
            urgency = "low"
        request = TripRequest(places[match[1].lower()], places[match[2].lower()], urgency,
                              int(deadline[1]) if deadline else None)
    if request.origin not in NODES or request.destination not in NODES or request.urgency not in CADENCE:
        raise ValueError("Request refers to a location or urgency outside the simulator.")
    if request.deadline_s is not None and (type(request.deadline_s) is not int or request.deadline_s < 1):
        raise ValueError("Deadline must be a positive integer in seconds.")
    return request


def eta(route: tuple[Road, ...], costs: dict[str, float]) -> float:
    return sum(edge.free_s * costs.get(edge.corridor, 1.0) for edge in route)


def shortest_path(start: str, goal: str, costs: dict[str, float]) -> tuple[Road, ...]:
    heap = [(0.0, start, ())]
    best = {start: 0.0}
    adj = {node: [e for e in ROADS if e.source == node] for node in NODES}
    while heap:
        cost, node, path = heapq.heappop(heap)
        if node == goal:
            return path
        if cost > best[node]:
            continue
        for edge in adj[node]:
            new_cost = cost + eta((edge,), costs)
            if new_cost < best.get(edge.target, math.inf):
                best[edge.target] = new_cost
                heapq.heappush(heap, (new_cost, edge.target, path + (edge,)))
    raise ValueError(f"No path from {start} to {goal}.")


def navigate(request: TripRequest, *, replan: bool = True, seed: int = 0) -> dict:
    """Execute each edge and replan at graph nodes using measured travel times and critic denials."""
    cases = catalog()
    costs: dict[str, float] = {}
    route = shortest_path(request.origin, request.destination, costs)
    route_nodes = lambda edges: [edge.target for edge in edges]
    trace = [{"kind": "plan", "remaining": route_nodes(route), "eta_s": round(eta(route, costs), 2)}]
    elapsed = 0.0
    node = request.origin
    segments = []
    for leg in range(len(ROADS)):
        if not route:
            break
        edge, route = route[0], route[1:]
        base = next(s for name, s in cases.items() if name.startswith(edge.scenario))
        adjusted = replace(base, expected="?", urgency=request.urgency, budget_s=70.0,
                           cruise_mps=min(base.cruise_mps, edge.limit_mps),
                           speed_limit=edge.limit_mps, pass_mps=min(base.pass_mps, edge.limit_mps))
        policy = Policy(cycle_s=CADENCE[request.urgency], distance_goal_m=edge.length_m)
        result, rec = run(adjusted, policy, seed=seed + leg, record=True, corridor_m=edge.length_m + 120)
        elapsed += result["duration_s"]
        arrived = any(e["kind"] == "segment_complete" for e in rec.events)
        trace.append({"kind": "segment", "from": node, "to": edge.target, "t_s": round(elapsed, 2),
                      "denials": result["agency"]["denials"], "passed": result["overtake_completed"], "arrived": arrived})
        segments.append({"road": asdict(edge), "result": result, "events": rec.events})
        if not arrived or not result["success"]:
            break
        node = edge.target
        observed = result["duration_s"] / edge.free_s
        costs[edge.corridor] = max(1.0, min(4.0, observed))
        repeated_denials = result["agency"]["denials"] >= 3 and not result["overtake_completed"]
        late = request.deadline_s is not None and elapsed + eta(route, costs) > request.deadline_s
        if replan and route and (repeated_denials or late):
            alternative = shortest_path(node, request.destination, costs)
            old_eta, new_eta = eta(route, costs), eta(alternative, costs)
            if new_eta + 1.0 < old_eta:
                trace.append({"kind": "reroute", "at": node, "remaining": route_nodes(alternative),
                              "prior_eta_s": round(old_eta, 2), "new_eta_s": round(new_eta, 2),
                              "reason": "repeated pass denials" if repeated_denials else "deadline risk"})
                route = alternative
    destination_reached = node == request.destination
    return {"request": asdict(request), "replan": replan, "arrived": destination_reached, "elapsed_s": round(elapsed, 2),
            "deadline_met": None if request.deadline_s is None else destination_reached and elapsed <= request.deadline_s,
            "trace": trace, "segments": segments}
