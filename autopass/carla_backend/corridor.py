"""Road-graph queries: find a straight stretch to stage an overtake, walk lanes, measure offsets.

Everything here reads CARLA's lane graph (the HD map). Nothing reads other actors.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple


def yaw_diff(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def same_dir_sides(carla, wp) -> List[str]:
    out = []
    for side, nb in (("left", wp.get_left_lane()), ("right", wp.get_right_lane())):
        if nb is not None and nb.lane_type == carla.LaneType.Driving and nb.lane_id * wp.lane_id > 0:
            out.append(side)
    return out


def opposing_side(carla, wp) -> Optional[str]:
    for side, nb in (("left", wp.get_left_lane()), ("right", wp.get_right_lane())):
        if nb is not None and nb.lane_type == carla.LaneType.Driving and nb.lane_id * wp.lane_id < 0:
            return side
    return None


def neighbour(wp, side: str):
    return wp.get_left_lane() if side == "left" else wp.get_right_lane()


def corridor_extent(carla, wp, side: str, *, opposing: bool, limit: float, step: float = 5.0,
                    max_turn_deg: float = 12.0) -> float:
    """Metres for which travel and passing lanes stay parallel, driving, straight and junction-free."""
    cur, yaw0, d = wp, wp.transform.rotation.yaw, 0.0
    while d < limit:
        nb = neighbour(cur, side)
        if nb is None or nb.lane_type != carla.LaneType.Driving:
            break
        if (nb.lane_id * cur.lane_id > 0) == opposing:
            break
        a, b = cur.transform.location, nb.transform.location
        if not 2.4 <= math.hypot(a.x - b.x, a.y - b.y) <= 6.0:
            break
        if abs(yaw_diff(cur.transform.rotation.yaw, yaw0)) > max_turn_deg:
            break
        nxt = cur.next(step)
        if not nxt or nxt[0].is_junction:
            break
        cur, d = nxt[0], d + step
    return d


def passing_lane_interior(carla, wp, side: str, opposing: bool) -> bool:
    """True if there is drivable surface beyond the passing lane (so an overshoot stays on road)."""
    if opposing:
        return True
    pl = neighbour(wp, side)
    outer = neighbour(pl, side) if pl is not None else None
    return outer is not None and outer.lane_type in (carla.LaneType.Driving, carla.LaneType.Shoulder,
                                                     carla.LaneType.Parking, carla.LaneType.Bidirectional)


def find_corridors(carla, world_map, *, min_straight_m: float, opposing: bool, preferred_side: str,
                   back_m: float = 0.0) -> List[Tuple[float, float, str, Any]]:
    """All usable corridors as (score, length, side, travel_waypoint), best first.

    `back_m` asks for that much clean road behind the start as well, for traffic spawned behind.
    """
    cands, seen = [], set()
    for wp in world_map.generate_waypoints(5.0):
        if wp.lane_type != carla.LaneType.Driving or wp.is_junction:
            continue
        key = (wp.road_id, wp.lane_id, round(wp.s / 40))
        if key in seen:
            continue
        seen.add(key)
        sides = [s for s in (opposing_side(carla, wp),) if s] if opposing else same_dir_sides(carla, wp)
        for side in sorted(sides, key=lambda s: s != preferred_side):
            ext = corridor_extent(carla, wp, side, opposing=opposing, limit=600.0)
            if ext < min_straight_m:
                continue
            if back_m > 0.0:
                prev = wp.previous(back_m)
                if not prev or prev[0].is_junction or abs(yaw_diff(prev[0].transform.rotation.yaw,
                                                                 wp.transform.rotation.yaw)) > 12.0:
                    continue
            score = (min(ext, 400.0) - abs(wp.transform.location.z) * 3.0
                     + (60.0 if passing_lane_interior(carla, wp, side, opposing) else 0.0)
                     + (25.0 if side == preferred_side else 0.0))
            cands.append((score, ext, side, wp))
            break
    cands.sort(key=lambda t: -t[0])
    return cands


def lane_to(carla, wp, lane_id: int):
    """Walk sideways from wp to the lane with this id (at most three hops)."""
    cur = wp
    for _ in range(4):
        if cur is None:
            return wp
        if cur.lane_id == lane_id:
            return cur
        best, cand = abs(cur.lane_id - lane_id), None
        for nb in (cur.get_left_lane(), cur.get_right_lane()):
            if nb is not None and nb.lane_type == carla.LaneType.Driving and abs(nb.lane_id - lane_id) < best:
                best, cand = abs(nb.lane_id - lane_id), nb
        if cand is None:
            return cur
        cur = cand
    return cur


def advance(wp, dist: float, reverse: bool = False):
    """Waypoint `dist` metres ahead in the ego's direction of travel (reverse: lane runs toward us)."""
    if dist == 0.0:
        return wp
    forward = (dist > 0) != reverse
    nxt = wp.next(abs(dist)) if forward else wp.previous(abs(dist))
    return nxt[0] if nxt else None


def lateral_offset(loc, wp) -> float:
    """Signed lateral distance of a location from a waypoint's lane centre, positive to the RIGHT."""
    t = wp.transform
    r = t.get_right_vector()
    return (loc.x - t.location.x) * r.x + (loc.y - t.location.y) * r.y


def marking_allows_crossing(carla, wp, side: str) -> bool:
    """Does the lane marking on this side permit a lane change?"""
    m = wp.left_lane_marking if side == "left" else wp.right_lane_marking
    T = carla.LaneMarkingType
    if m is None:
        return True
    return m.type in (T.Broken, T.BrokenBroken, T.BrokenSolid, T.NONE, T.Other, T.BottsDots)
