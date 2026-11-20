"""CARLA implementation of the World protocol.

The decision core never touches CARLA: it sees this adapter. The adapter reads only what a real
vehicle would have: its cameras, its odometry, and its position on the lane map. Ground truth
(other actors' poses) is used in exactly two places, both outside the decision path:
`truth()` for metrics/telemetry and the scripted traffic's own cruise control.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.control import CAMERAS
from ..core.metrics import ALONGSIDE_M
from ..core.scenario import NPC, Scenario
from ..core.types import (FRONT, ONCOMING, PASSING_AHEAD, REAR, SIDE, SIDE_TRAVEL, TRAVEL_BEHIND, Actor,
                          Control, Corridor, Detection, EgoState, SenseResult, Truth)
from ..sim.sensors import CHANNEL_CAMERAS
from . import corridor as C
from . import frenet as F
from . import perception as P

WEATHER = {"clear_noon": "ClearNoon", "cloudy_noon": "CloudyNoon", "wet_noon": "WetNoon",
           "wet_cloudy_noon": "WetCloudyNoon", "soft_rain": "SoftRainNoon", "hard_rain": "HardRainNoon",
           "clear_sunset": "ClearSunset"}
CAR_BPS = ["vehicle.audi.etron", "vehicle.nissan.patrol_2021", "vehicle.lincoln.mkz_2020",
           "vehicle.mercedes.coupe_2020", "vehicle.dodge.charger_2020", "vehicle.ford.mustang"]
KIND_BP = {"truck": "vehicle.carlamotors.firetruck", "bus": "vehicle.mitsubishi.fusorosa"}
ROLE_COLOR = {"lead": "200,40,40", "rear": "30,30,30", "oncoming": "240,240,240", "blocker": "230,200,30",
              "side": "120,120,130", "queue": "90,60,160"}
SIDE_RES = (480, 270)


def _throttle_brake(cur: float, target: float) -> Tuple[float, float]:
    err = target - cur
    if err >= 0:
        return min(0.85, 0.30 + 0.55 * err), 0.0
    return 0.0, min(0.8, -0.25 * err)


class CarlaWorld:
    def __init__(self, client, scn: Scenario, *, video: bool = True, res: Tuple[int, int] = (800, 450),
                 seed: int = 0, dt: float = 0.05, perception: str = "segdepth"):
        import carla
        self.carla, self.client, self.scn, self.dt, self.time = carla, client, scn, dt, 0.0
        self.video, self.res = video, res
        self.seed = seed
        self.rng = random.Random(seed)
        self.two_lane, self.speed_limit = scn.two_lane, scn.speed_limit
        self.world = client.get_world()
        self.map = self.world.get_map()
        self.actors: Dict[str, Any] = {}
        self.npcs: List[Dict[str, Any]] = []
        self.sensors: Dict[str, Any] = {}
        self.buf: Dict[str, Any] = {}
        self.geom: Dict[str, P.CamGeom] = {}
        self.collision = False
        self.collision_with: Optional[str] = None
        self.frame = 0
        self._last_ego_s = 0.0
        self._cache: Dict[Any, Any] = {}
        self._tm = None
        self.perception = perception
        self.detector = None
        if perception == "rgbd":
            from .rgbd import RGBDDetector
            self.detector = RGBDDetector()
        self._initial_settings = self.world.get_settings()
        try:
            self._setup()
        except BaseException:
            self.close()
            raise

    # ---------------------------------------------------------------------------------------
    def _setup(self) -> None:
        carla, scn, world = self.carla, self.scn, self.world
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = True, self.dt
        world.apply_settings(settings)
        try:
            world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        except Exception:  # noqa: BLE001
            pass
        world.set_weather(getattr(carla.WeatherParameters, WEATHER.get(scn.weather, "ClearNoon")))

        back = max([60.0] + [-n.s + 25.0 for n in scn.npcs if n.s < 0])
        cands = C.find_corridors(carla, self.map, min_straight_m=scn.min_straight_m, opposing=scn.two_lane,
                                 preferred_side=scn.passing_side, back_m=back)
        if not cands:
            cands = C.find_corridors(carla, self.map, min_straight_m=0.6 * scn.min_straight_m,
                                     opposing=scn.two_lane, preferred_side=scn.passing_side, back_m=back)
        if not cands:
            raise RuntimeError(f"no corridor on {self.map.name} for {scn.id}")
        _, ext, side, wp = cands[scn.corridor_rank % len(cands)]
        self.passing_side, self.passing_left = side, side == "left"
        self.corridor_len = ext
        self.travel_wp = wp
        nb = C.neighbour(wp, side)
        a, b = wp.transform.location, nb.transform.location
        self.lane_width = math.hypot(a.x - b.x, a.y - b.y)
        self.frenet = F.build(carla, wp, self.passing_left)
        self.z0 = wp.transform.location.z

        lead = scn.lead()
        v0 = min(scn.cruise_mps, lead.v) if lead else scn.cruise_mps
        ego = self._spawn_vehicle("vehicle.tesla.model3", 0.0, 0.0, "20,90,200")
        if ego is None:
            raise RuntimeError("ego spawn failed")
        self.actors["ego"] = ego
        self.ego_len = 2.0 * ego.bounding_box.extent.x
        self.ego_wid = 2.0 * ego.bounding_box.extent.y
        pending = []
        for i, n in enumerate(scn.npcs):
            if n.appear_s > 0:
                pending.append((i, n))
            else:
                self._add_npc(i, n, 0.0)
        self._pending = pending
        if scn.ambient > 0:
            self._spawn_ambient(scn.ambient)
        self._attach_sensors(ego)

        for a in [ego] + [n["actor"] for n in self.npcs]:        # let suspensions settle
            a.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(15):
            world.tick()
        self._set_speed(ego, v0, +1)
        for n in self.npcs:
            self._set_speed(n["actor"], n["spec"].v, 1)
        self._tick()
        self.time = 0.0
        self._steer_prev: Dict[int, float] = {}

    def _lane_wp(self, s: float, lane: str):
        x, y = self.frenet.to_xy(s, 0.0 if lane == "travel" else self.lane_width)
        return self.map.get_waypoint(self.carla.Location(x=x, y=y, z=self.z0), project_to_road=True,
                                     lane_type=self.carla.LaneType.Driving)

    def _spawn_vehicle(self, bp_name: str, s: float, d_lane: float, color: Optional[str]):
        carla = self.carla
        bl = self.world.get_blueprint_library()
        try:
            bp = bl.find(bp_name)
        except Exception:  # noqa: BLE001
            bp = bl.find("vehicle.audi.tt")
        if color and bp.has_attribute("color"):
            bp.set_attribute("color", color)
        wp = self._lane_wp(s, "travel" if d_lane == 0.0 else "passing")
        tf = carla.Transform(carla.Location(x=wp.transform.location.x, y=wp.transform.location.y,
                                            z=wp.transform.location.z + 0.35), wp.transform.rotation)
        return self.world.try_spawn_actor(bp, tf)

    def _add_npc(self, idx: int, n: NPC, ego_s: float) -> None:
        bp = self.scn.lead_bp if n.role == "lead" else KIND_BP.get(n.kind, CAR_BPS[idx % len(CAR_BPS)])
        actor = None
        for ds in (0.0, 3.0, -3.0, 6.0):
            actor = self._spawn_vehicle(bp, ego_s + n.s + ds, 0.0 if n.lane == "travel" else self.lane_width,
                                        ROLE_COLOR.get(n.role))
            if actor is not None:
                break
        if actor is None:
            raise RuntimeError(f"could not spawn required {n.role} actor")
        self.npcs.append({"id": idx + 1, "spec": n, "actor": actor, "v0": n.v})
        self.actors[f"{n.role}{idx}"] = actor

    def _set_speed(self, actor, v: float, _direction: int) -> None:
        # Clear the staging brake before the scored rolling start.
        actor.apply_control(self.carla.VehicleControl(throttle=0.25 if v > 0.05 else 0,
                                                       brake=0 if v > 0.05 else 1))
        f = actor.get_transform().get_forward_vector()
        actor.set_target_velocity(self.carla.Vector3D(f.x * v, f.y * v, 0.0))

    def _spawn_ambient(self, n: int) -> None:
        """Background Traffic-Manager vehicles, kept out of a box around the staged corridor."""
        carla, world = self.carla, self.world
        try:
            tm = self.client.get_trafficmanager()
            tm.set_synchronous_mode(True)
            tm.set_random_device_seed(self.seed)
            tm.set_global_distance_to_leading_vehicle(3.0)
        except Exception:  # noqa: BLE001
            return
        self._tm = tm
        spawns = self.map.get_spawn_points()
        self.rng.shuffle(spawns)
        cars = [b for b in world.get_blueprint_library().filter("vehicle.*")
                if int(b.get_attribute("number_of_wheels")) == 4]
        k = 0
        for sp in spawns:
            if k >= n:
                break
            s, d = self.frenet.to_sd(np.array([[sp.location.x, sp.location.y]]))
            if -80.0 < s[0] < 260.0 and abs(d[0]) < 12.0:
                continue
            a = world.try_spawn_actor(self.rng.choice(cars), sp)
            if a is None:
                continue
            a.set_autopilot(True, tm.get_port())
            try:
                tm.auto_lane_change(a, False)
                tm.vehicle_percentage_speed_difference(a, self.rng.uniform(-5, 25))
            except Exception:  # noqa: BLE001
                pass
            self.actors[f"ambient{k}"] = a
            k += 1

    def _attach_sensors(self, ego) -> None:
        carla, bl = self.carla, self.world.get_blueprint_library()

        def cam(kind: str, key: str, w: int, h: int, fov: float, tf) -> None:
            bp = bl.find(kind)
            bp.set_attribute("image_size_x", str(w))
            bp.set_attribute("image_size_y", str(h))
            bp.set_attribute("fov", str(fov))
            s = self.world.spawn_actor(bp, tf, attach_to=ego)
            s.listen(lambda img, k=key: self.buf.__setitem__(k, img))
            self.sensors[key] = s

        for name, c in CAMERAS.items():
            w, h = SIDE_RES if name.startswith("side") else self.res
            g = P.cam_geometry(name, w, h, self.passing_left, z=1.5)
            if name.startswith("rq"):                 # mirror cameras sit just outside the bodywork
                g.y_right = math.copysign(abs(g.y_right) + 0.12, g.y_right)
                g.z = 1.25
            self.geom[name] = g
            tf = carla.Transform(carla.Location(x=g.x, y=g.y_right, z=g.z),
                                 carla.Rotation(yaw=math.degrees(g.yaw_right)))
            if self.detector:
                from .rgbd import RANGE_M
                g.reach = RANGE_M[name]
                cam("sensor.camera.rgb", name + ":rgb", w, h, c["fov"], tf)
            else:
                cam("sensor.camera.instance_segmentation", name + ":seg", w, h, c["fov"], tf)
            cam("sensor.camera.depth", name + ":depth", w, h, c["fov"], tf)
        if self.video:
            w, h = self.res
            if not self.detector:
                cam("sensor.camera.rgb", "rgb", w, h, CAMERAS["front_wide"]["fov"],
                    carla.Transform(carla.Location(x=CAMERAS["front_wide"]["x"], z=1.5)))
            cam("sensor.camera.rgb", "overhead", w, h, 90.0,
                carla.Transform(carla.Location(x=10.0, z=46.0), carla.Rotation(pitch=-90.0)))
        col = self.world.spawn_actor(bl.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        col.listen(self._on_collision)
        self.sensors["collision"] = col

    def _on_collision(self, ev) -> None:
        self.collision = True
        try:
            self.collision_with = ev.other_actor.type_id
        except Exception:  # noqa: BLE001
            pass

    # -- World protocol ------------------------------------------------------------------------
    def _ego_pose(self) -> Dict[str, float]:
        if "ego" not in self._cache:
            a = self.actors["ego"]
            tf, vel = a.get_transform(), a.get_velocity()
            s, d = self.frenet.to_sd(np.array([[tf.location.x, tf.location.y]]), near_s=self._last_ego_s)
            self._last_ego_s = float(s[0])
            self._cache["ego"] = {"x": tf.location.x, "y": tf.location.y, "yaw": tf.rotation.yaw,
                                  "s": float(s[0]), "d": float(d[0]),
                                  "v": math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2),
                                  "h": self.frenet.heading(float(s[0]), tf.rotation.yaw)}
        return self._cache["ego"]

    def ego(self) -> EgoState:
        e = self._ego_pose()
        return EgoState(speed=e["v"], d=e["d"], heading=e["h"], length=self.ego_len)

    def corridor(self) -> Corridor:
        e = self._ego_pose()
        x, y = self.frenet.to_xy(e["s"], 0.0)
        wp = self.map.get_waypoint(self.carla.Location(x=x, y=y, z=self.z0), project_to_road=True,
                                   lane_type=self.carla.LaneType.Driving)
        legal = C.marking_allows_crossing(self.carla, wp, self.passing_side)
        return Corridor(clear_m=max(0.0, self.corridor_len - e["s"]), legal=legal)

    def target_point(self, d_target: float, lookahead: float) -> Tuple[float, float]:
        e = self._ego_pose()
        tx, ty = self.frenet.to_xy(e["s"] + lookahead, d_target)
        rx, ry = tx - e["x"], ty - e["y"]
        yaw = math.radians(e["yaw"])
        fwd = rx * math.cos(yaw) + ry * math.sin(yaw)
        right = -rx * math.sin(yaw) + ry * math.cos(yaw)
        return fwd, (-right if self.passing_left else right)

    def apply(self, control: Control) -> None:
        e = self._ego_pose()
        thr, brk = _throttle_brake(e["v"], control.target_speed)
        steer = -control.steer if self.passing_left else control.steer
        self.actors["ego"].apply_control(self.carla.VehicleControl(throttle=thr, brake=brk, steer=steer))

    def step(self) -> None:
        self._drive_npcs()
        self._tick()
        self.time += self.dt
        for i, n in list(self._pending):
            if self.time >= n.appear_s:
                self._add_npc(i, n, self._ego_pose()["s"])
                self._pending.remove((i, n))
                if self.npcs and self.npcs[-1]["id"] == i + 1:
                    self._set_speed(self.npcs[-1]["actor"], n.v, 1)

    def _tick(self) -> None:
        self.frame = self.world.tick()
        self._cache = {}
        want = [k for k in self.sensors if k != "collision"]
        t0 = time.time()
        while time.time() - t0 < 3.0:
            if all(k in self.buf and self.buf[k].frame == self.frame for k in want):
                return
            time.sleep(0.001)
        missing = [k for k in want if k not in self.buf or self.buf[k].frame != self.frame]
        raise RuntimeError(f"sensor frame synchronization timed out at {self.frame}: {', '.join(missing)}")

    # -- scripted traffic (uses ground truth; it is the environment, not the agent) ---------------
    def _drive_npcs(self) -> None:
        carla = self.carla
        e = self._ego_pose()
        for n in self.npcs:
            a, spec = n["actor"], n["spec"]
            tf, vel = a.get_transform(), a.get_velocity()
            v = math.sqrt(vel.x ** 2 + vel.y ** 2)
            target = n["v0"]
            if spec.v <= 0.05:
                a.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
                continue
            if spec.behavior == "idm":                 # keep a time gap behind the ego when it is ahead in lane
                s, d = self.frenet.to_sd(np.array([[tf.location.x, tf.location.y]]))
                gap = (e["s"] - float(s[0])) - 0.5 * (self.ego_len + 2.0 * a.bounding_box.extent.x)
                if 0.0 < gap < 60.0 and abs(float(d[0]) - e["d"]) < 1.9:
                    target = max(0.0, min(target, e["v"] + 0.6 * (gap - max(7.0, 1.4 * v))))
            wp = self.map.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            nxt = wp.next(max(4.0, 0.8 * v))
            if not nxt:
                continue
            tl = nxt[0].transform.location
            yaw = math.radians(tf.rotation.yaw)
            rx, ry = tl.x - tf.location.x, tl.y - tf.location.y
            fwd = rx * math.cos(yaw) + ry * math.sin(yaw)
            right = -rx * math.sin(yaw) + ry * math.cos(yaw)
            raw = max(-0.5, min(0.5, math.degrees(math.atan2(right, max(0.1, fwd))) / 40.0))
            prev = self._steer_prev.get(a.id, 0.0)
            steer = 0.5 * prev + 0.5 * raw
            self._steer_prev[a.id] = steer
            thr, brk = _throttle_brake(v, target)
            a.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=steer))

    # -- perception -------------------------------------------------------------------------------
    def _frame(self, cam: str):
        key = ("frame", cam)
        if key not in self._cache:
            sample = self.buf.get(cam + (":rgb" if self.detector else ":seg"))
            dep = self.buf.get(cam + ":depth")
            if sample is None or dep is None or sample.frame != dep.frame or dep.frame != self.frame:
                self._cache[key] = None
            elif self.detector:
                from .rgbd import ground_candidates
                depth = P.decode_depth(dep)
                self._cache[key] = (ground_candidates(self.geom[cam], depth), None, depth)
            else:
                tag, inst = P.decode_instances(sample)
                self._cache[key] = (tag, inst, P.decode_depth(dep))
        return self._cache[key]

    def _objects(self, cam: str) -> List[P.Object]:
        key = ("objs", cam)
        if key not in self._cache:
            fr = self._frame(cam)
            if fr is None:
                self._cache[key] = []
            elif self.detector:
                self._cache[key] = self.detector.objects(
                    self.geom[cam], P.decode_rgb(self.buf[cam + ":rgb"]), fr[2], self.frenet,
                    self._ego_pose(), self.time, self.ego_len, self.ego_wid)
            else:
                e = self._ego_pose()
                self._cache[key] = P.objects_in_frame(self.geom[cam], fr[0], fr[1], fr[2], self.frenet,
                                                      (e["x"], e["y"]), e["yaw"], e["s"], e["d"],
                                                      self.ego_len, self.ego_wid)
        return self._cache[key]

    def _extent(self, cam: str, direction: int) -> float:
        key = ("ext", cam, direction)
        if key not in self._cache:
            fr = self._frame(cam)
            e = self._ego_pose()
            self._cache[key] = 0.0 if fr is None else P.lane_visible_extent(
                self.geom[cam], fr[0], fr[2], self.frenet, (e["x"], e["y"]), e["yaw"], e["s"],
                self.lane_width, direction)
        return self._cache[key]

    def objects(self, channel: str) -> Dict[int, P.Object]:
        views = [self._objects(c) for c in CHANNEL_CAMERAS[channel]]
        if self.detector:
            # Place the main channel camera first. Dedupe secondary views by lane-frame overlap.
            objects = {}
            for view in views:
                for o in view:
                    duplicate = any(abs(o.d - a.d) < 1.3 and o.s_min <= a.s_max + 2.0
                                    and a.s_min <= o.s_max + 2.0 for a in objects.values())
                    if not duplicate:
                        objects[o.id] = o
            return objects
        return P.merge(views)

    def sense(self, channel: str) -> SenseResult:
        w, half = self.lane_width, 0.5 * self.ego_len
        dets: List[Detection] = []
        for o in self.objects(channel).values():
            ahead, behind = o.s_min - half, -o.s_max - half
            lane = "travel" if abs(o.d) < 0.5 * w else ("passing" if 0.5 * w <= o.d < 1.5 * w else "other")
            if channel == FRONT:
                ok, gap = lane == "travel" and ahead >= 0.0, ahead
            elif channel in (PASSING_AHEAD, ONCOMING):
                ok, gap = lane == "passing" and ahead >= ALONGSIDE_M, ahead
            elif channel == REAR:
                ok, gap = lane == "passing" and behind >= ALONGSIDE_M, behind
            elif channel == SIDE:
                ok, gap = lane == "passing" and ahead < ALONGSIDE_M and behind < ALONGSIDE_M, 0.0
            elif channel == TRAVEL_BEHIND:
                ok, gap = lane == "travel" and behind >= ALONGSIDE_M, behind
            elif channel == SIDE_TRAVEL:
                ok, gap = lane == "travel" and ahead < ALONGSIDE_M and behind < ALONGSIDE_M, 0.0
            else:
                raise ValueError(channel)
            if ok:
                dets.append(Detection(gap_m=max(0.0, gap), lateral_m=o.d, kind=o.kind, bbox=o.bbox, track_id=o.id))
        return SenseResult(dets, self._reach(channel))

    def _reach(self, channel: str) -> float:
        if channel == FRONT:
            return self.geom["front_wide"].reach
        if channel in (PASSING_AHEAD, ONCOMING):
            near = self._extent("front_wide", +1)
            return max(near, self._extent("front_tele", +1)) if near >= 25.0 else near
        if channel == REAR:
            return self._extent("rq_pass", -1)
        if channel == TRAVEL_BEHIND:
            return self.geom["rq_travel"].reach
        return 6.0

    # -- ground truth: metrics and telemetry only ---------------------------------------------------
    def truth(self) -> Truth:
        e = self._ego_pose()
        actors = [Actor(0, "ego", e["s"], e["d"], e["v"], e["h"], self.ego_len, self.ego_wid)]
        extra = [(f"ambient{i}", a) for i, a in enumerate(a for k, a in self.actors.items() if k.startswith("ambient"))]
        items = [(n["spec"].role, n["actor"], n["id"], n["spec"].kind) for n in self.npcs] + \
                [("ambient", a, 100 + i, "car") for i, (_, a) in enumerate(extra)]
        for role, a, aid, kind in items:
            tf, vel = a.get_transform(), a.get_velocity()
            s, d = self.frenet.to_sd(np.array([[tf.location.x, tf.location.y]]))
            s, d = float(s[0]), float(d[0])
            if role == "ambient" and (abs(s - e["s"]) > 140.0 or abs(d) > 14.0):
                continue
            i = int(self.frenet._index(np.array([s]))[0])
            v = vel.x * self.frenet.T[i, 0] + vel.y * self.frenet.T[i, 1]
            actors.append(Actor(aid, role, s, d, v, self.frenet.heading(s, tf.rotation.yaw),
                                2.0 * a.bounding_box.extent.x, 2.0 * a.bounding_box.extent.y, kind))
        offroad = e["d"] < -0.5 * self.lane_width - 0.9 or e["d"] > 1.5 * self.lane_width + 0.9
        return Truth(self.time, actors, self.collision, offroad)

    def images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """(front RGB, overhead RGB) for the video, if video sensors are attached."""
        rgb = self.buf.get("front_wide:rgb" if self.detector else "rgb")
        over = self.buf.get("overhead")
        return (P.decode_rgb(rgb) if rgb is not None else None,
                P.decode_rgb(over) if over is not None else None)

    def close(self) -> None:
        try:
            if self._tm is not None:
                self._tm.set_synchronous_mode(False)
        except Exception:  # noqa: BLE001
            pass
        for s in self.sensors.values():
            try:
                s.stop()
                s.destroy()
            except Exception:  # noqa: BLE001
                pass
        for a in self.actors.values():
            try:
                a.destroy()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.world.apply_settings(self._initial_settings)
        except Exception:  # noqa: BLE001
            pass
