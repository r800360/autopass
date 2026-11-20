"""Perception from simulator sensors: instance segmentation + metric depth -> lane-frame objects.

This is the *privileged* perception profile: the simulator's own panoptic segmentation and
depth stand in for a perfect perception stack. The learned RGB-only profile lives in
perception_learned.py and produces the same Object records, so everything downstream is shared.

Pipeline per camera frame:
  pixels of each vehicle instance --depth--> 3-D points in the camera frame
  --camera mount--> ego frame --ego pose--> world --Frenet--> (s, d) in the lane frame
An object is then summarised by the extent of its visible surface along the lane (s_min, s_max)
and its median lateral offset d, which is all the gap logic needs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.control import CAMERAS

VEHICLE_TAGS = {14: "car", 15: "truck", 16: "bus", 18: "car", 19: "car"}   # CARLA >= 0.9.14 tag ids
ROAD_TAGS = (1, 24)                                                         # road surface, lane markings
MIN_PIXELS = 10


@dataclass
class Object:
    id: int
    kind: str
    s_min: float          # nearest/farthest visible surface along the lane, relative to the ego centre
    s_max: float
    d: float              # median lateral offset from the travel-lane centre, + toward the passing side
    pixels: int
    bbox: Optional[Tuple[int, int, int, int]] = None      # in the front-wide image, for overlays
    cam: str = ""


@dataclass
class CamGeom:
    name: str
    w: int
    h: int
    f: float              # focal length in pixels
    x: float              # mount: metres ahead of the ego centre
    y_right: float        # mount: metres to the ego's right
    z: float
    yaw_right: float      # radians, positive = toward the ego's right (CARLA convention)
    reach: float


def cam_geometry(name: str, w: int, h: int, passing_left: bool, z: float = 1.5) -> CamGeom:
    c = CAMERAS[name]
    sign = -1.0 if passing_left else 1.0            # "toward the passing side" -> CARLA's right-positive axes
    return CamGeom(name, w, h, 0.5 * w / math.tan(math.radians(0.5 * c["fov"])), c["x"], sign * c["y"], z,
                   math.radians(sign * c["yaw"]), c["range"])


def decode_bgra(img) -> np.ndarray:
    return np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)


def decode_rgb(img) -> np.ndarray:
    return decode_bgra(img)[:, :, 2::-1].copy()     # sensor delivers BGRA


def decode_depth(img) -> np.ndarray:
    a = decode_bgra(img).astype(np.float32)
    return (a[:, :, 2] + a[:, :, 1] * 256.0 + a[:, :, 0] * 65536.0) * (1000.0 / 16777215.0)


def decode_instances(img) -> Tuple[np.ndarray, np.ndarray]:
    """(semantic tag, instance id) per pixel. The id is the actor id in CARLA 0.9.16."""
    a = decode_bgra(img)
    return a[:, :, 2], a[:, :, 1].astype(np.int32) + 256 * a[:, :, 0].astype(np.int32)


def pixels_to_world(g: CamGeom, u: np.ndarray, v: np.ndarray, z: np.ndarray, ego_xy: Tuple[float, float],
                    ego_yaw_deg: float) -> np.ndarray:
    """Back-project pixels with planar depth z to world x/y (K, 2)."""
    fwd = z
    right = (u - 0.5 * g.w) * z / g.f
    cy, sy = math.cos(g.yaw_right), math.sin(g.yaw_right)
    ex = g.x + fwd * cy - right * sy                 # ego frame: x forward, y right
    ey = g.y_right + fwd * sy + right * cy
    yaw = math.radians(ego_yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([ego_xy[0] + ex * c - ey * s, ego_xy[1] + ex * s + ey * c], axis=1)


def objects_in_frame(g: CamGeom, tag: np.ndarray, inst: np.ndarray, depth: np.ndarray, frenet,
                     ego_xy: Tuple[float, float], ego_yaw_deg: float, ego_s: float, ego_d: float,
                     ego_len: float = 4.7, ego_wid: float = 1.9) -> List[Object]:
    """Vehicle instances visible in one camera frame, in lane coordinates relative to the ego."""
    mask = np.isin(tag, tuple(VEHICLE_TAGS))
    if not mask.any():
        return []
    vs, us = np.nonzero(mask)
    ids, zs, tags = inst[vs, us], depth[vs, us], tag[vs, us]
    out: List[Object] = []
    for oid in np.unique(ids):
        sel = ids == oid
        n = int(sel.sum())
        if n < MIN_PIXELS:
            continue
        u, v, z = us[sel], vs[sel], zs[sel]
        if n > 600:                                   # subsample large masks
            step = n // 600 + 1
            u, v, z = u[::step], v[::step], z[::step]
        ok = (z > 0.3) & (z < g.reach * 1.5)
        if ok.sum() < MIN_PIXELS // 2:
            continue
        s, d = frenet.to_sd(pixels_to_world(g, u[ok].astype(np.float64), v[ok].astype(np.float64),
                                            z[ok].astype(np.float64), ego_xy, ego_yaw_deg))
        s = s - ego_s
        own = (np.abs(s) < 0.5 * ego_len + 0.4) & (np.abs(d - ego_d) < 0.5 * ego_wid + 0.3)
        if own.mean() > 0.5:                          # the ego's own bodywork seen by a mirror camera
            continue
        s, d = s[~own], d[~own]
        if s.size < MIN_PIXELS // 2:
            continue
        lo, hi = np.percentile(s, [3, 97])
        if min(abs(lo), abs(hi)) > g.reach:
            continue
        kind = VEHICLE_TAGS.get(int(np.bincount(tags[sel]).argmax()), "car")
        us_all, vs_all = us[sel], vs[sel]
        out.append(Object(int(oid), kind, float(lo), float(hi), float(np.median(d)), n,
                          (int(us_all.min()), int(vs_all.min()), int(us_all.max()), int(vs_all.max())), g.name))
    return out


def merge(views: List[List[Object]]) -> Dict[int, Object]:
    """Union of per-camera detections of the same instance."""
    out: Dict[int, Object] = {}
    for objs in views:
        for o in objs:
            if o.id not in out:
                out[o.id] = Object(o.id, o.kind, o.s_min, o.s_max, o.d, o.pixels, o.bbox, o.cam)
            else:
                m = out[o.id]
                tot = m.pixels + o.pixels
                m.d = (m.d * m.pixels + o.d * o.pixels) / tot
                m.s_min, m.s_max, m.pixels = min(m.s_min, o.s_min), max(m.s_max, o.s_max), tot
                if m.bbox is None or (o.cam == "front_wide" and o.bbox is not None):
                    m.bbox, m.cam = o.bbox, o.cam
    return out


def lane_visible_extent(g: CamGeom, tag: np.ndarray, depth: np.ndarray, frenet, ego_xy, ego_yaw_deg: float,
                        ego_s: float, lane_d: float, direction: int, half_band: float = 1.0) -> float:
    """How far along a lane the road surface is visibly continuous in this camera.

    Works row by row from near to far: a row counts as visible when road pixels fall inside the
    lane's lateral band. The first sizeable stretch with none ends the visible extent, whether
    the cause is a vehicle in the way, a bend, or the end of the road.
    """
    road = np.isin(tag, ROAD_TAGS)
    rows = np.nonzero(road.any(axis=1))[0]
    if rows.size == 0:
        return 0.0
    vs, us = np.nonzero(road[:, ::2])                 # every second column is plenty
    us = us * 2
    z = depth[vs, us]
    ok = (z > 1.0) & (z < g.reach * 1.2)
    if not ok.any():
        return 0.0
    s, d = frenet.to_sd(pixels_to_world(g, us[ok].astype(np.float64), vs[ok].astype(np.float64),
                                        z[ok].astype(np.float64), ego_xy, ego_yaw_deg))
    s = (s - ego_s) * direction
    in_band = (np.abs(d - lane_d) < half_band) & (s > 2.0)
    if not in_band.any():
        return 0.0
    dist = np.sort(s[in_band])
    # Ground distance between adjacent image rows grows as d^2 / (f * camera height): allow gaps of
    # a few rows, never less than 6 m.
    gaps = np.diff(dist)
    allowed = np.maximum(6.0, 3.0 * dist[:-1] ** 2 / (g.f * g.z))
    breaks = np.nonzero(gaps > allowed)[0]
    end = dist[breaks[0]] if breaks.size else dist[-1]
    return float(min(end, g.reach))
