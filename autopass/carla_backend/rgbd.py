"""NanoDet vehicle boxes fused with metric camera depth and lane geometry.

The detector sees RGB only. The gap adapter uses depth and the CARLA lane map.
This profile deliberately uses more conservative visibility caps than instance masks.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
import time

import cv2
import numpy as np

from .perception import Object, pixels_to_world

MODEL_SHA256 = "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186"
MODEL_NAME = "object_detection_nanodet_2022nov.onnx"
MODEL_REF = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODEL_URL = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{MODEL_REF}/models/object_detection_nanodet/{MODEL_NAME}"
COCO_KINDS = {1: "car", 2: "car", 3: "car", 5: "bus", 7: "truck"}
RANGE_M = {"front_wide": 70.0, "front_tele": 140.0, "rq_pass": 115.0, "rq_travel": 115.0,
           "side_pass": 24.0, "side_travel": 24.0}


class RGBDDetector:
    def __init__(self, path: Path | None = None, confidence: float = 0.27):
        path = Path(path or os.environ.get("AUTOPASS_NANODET_MODEL") or Path("models") / MODEL_NAME)
        if not path.is_file():
            raise FileNotFoundError("NanoDet model missing. Run python scripts/fetch_perception_model.py")
        with path.open("rb") as model:
            if sha256(model.read()).hexdigest() != MODEL_SHA256:
                raise ValueError("NanoDet weights do not match the pinned SHA-256")
        cv2.setNumThreads(2)
        self.net = cv2.dnn.readNet(str(path))
        self.confidence = confidence
        self.inferences = 0
        self.inference_s = 0.0
        self._next_id = 0
        self._tracks = {}

    def boxes(self, rgb: np.ndarray):
        """Return [(x0, y0, x1, y1, score, kind)] in input pixel coordinates."""
        start = time.perf_counter()
        square = cv2.resize(rgb[:, :, ::-1], (416, 416), interpolation=cv2.INTER_LINEAR)
        normalized = (square.astype(np.float32) - np.float32([103.53, 116.28, 123.675])) / np.float32(
            [57.375, 57.12, 58.395])
        self.net.setInput(cv2.dnn.blobFromImage(normalized))
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        raw_boxes, scores, labels = [], [], []
        classes = {out.shape[-2]: out for out in outs if out.shape[-1] == 80}
        offsets = {out.shape[-2]: out for out in outs if out.shape[-1] == 32}
        if not classes or classes.keys() != offsets.keys():
            raise ValueError("Unexpected NanoDet output tensors")
        for stride in (8, 16, 32, 64):
            anchors = (416 // stride) ** 2
            if anchors not in classes:
                continue
            cls, loc = classes[anchors], offsets[anchors]
            cls = cls.reshape(-1, cls.shape[-1])
            kind = cls.argmax(axis=1)
            score = cls.max(axis=1)
            idx = np.flatnonzero((score >= self.confidence) & np.isin(kind, list(COCO_KINDS)))
            if not len(idx):
                continue
            bins = loc.reshape(-1, 4, 8)[idx]
            mass = np.exp(bins - bins.max(axis=2, keepdims=True))
            dist = (mass @ np.arange(8) / mass.sum(axis=2)) * stride
            x = stride * (idx % (416 // stride)) + 0.5 * (stride - 1)
            y = stride * (idx // (416 // stride)) + 0.5 * (stride - 1)
            xyxy = np.clip(np.column_stack((x-dist[:, 0], y-dist[:, 1], x+dist[:, 2], y+dist[:, 3])), 0, 416)
            raw_boxes.extend((float(a), float(b), float(c-a), float(d-b)) for a, b, c, d in xyxy)
            scores.extend(float(s) for s in score[idx])
            labels.extend(COCO_KINDS[int(k)] for k in kind[idx])
        self.inferences += 1
        self.inference_s += time.perf_counter() - start
        h, w = rgb.shape[:2]
        boxes = []
        for i in np.asarray(cv2.dnn.NMSBoxes(raw_boxes, scores, self.confidence, 0.55)).reshape(-1):
            x, y, dx, dy = raw_boxes[i]
            boxes.append((int(x*w/416), int(y*h/416), int((x+dx)*w/416), int((y+dy)*h/416), scores[i], labels[i]))
        return boxes

    def objects(self, g, rgb, depth, frenet, ego, t, ego_len, ego_width):
        out = []
        for x0, y0, x1, y1, score, kind in self.boxes(rgb):
            if x1 - x0 < 4 or y1 - y0 < 3:
                continue
            v, u = np.mgrid[y0 + (y1-y0)//5:y1 - (y1-y0)//5:2,
                            x0 + (x1-x0)//5:x1 - (x1-x0)//5:2]
            z = depth[v, u].flatten()
            mask = (z >= 0.3) & (z <= g.reach)
            if mask.sum() < 4:
                continue
            near = np.percentile(z[mask], 30)
            keep = mask & (abs(z-near) <= max(2.0, near * 0.08))
            if keep.sum() < 4:
                continue
            points = pixels_to_world(g, u.flatten()[keep], v.flatten()[keep], z[keep],
                                     (ego["x"], ego["y"]), ego["yaw"])
            s, d = frenet.to_sd(points)
            s -= ego["s"]
            own = (abs(s) < ego_len * 0.5 + 0.4) & (abs(d-ego["d"]) < ego_width * 0.5 + 0.3)
            if own.mean() > 0.5:
                continue
            lo, hi = np.percentile(s, [5, 95])
            out.append(Object(-1, kind, float(lo), float(hi), float(np.median(d)), int(keep.sum()),
                              (x0,y0,x1,y1), g.name))
        return self._associate(g.name, t, ego["s"], out)

    def _associate(self, camera, t, ego_s, detections):
        old = self._tracks.get(camera, {})
        old = {key: track for key, track in old.items() if t-track[0] <= 0.7}
        available = set(old)
        results = []
        for obj in detections:
            absolute_s = ego_s + 0.5 * (obj.s_min + obj.s_max)
            options = []
            for identity in available:
                stamp, prior_s, prior_d = old[identity]
                ds, dd = abs(absolute_s-prior_s), abs(obj.d-prior_d)
                if ds < 2.5 + 38.0*(t-stamp) and dd < 1.7:
                    options.append((ds + 3.0*dd, identity))
            if options:
                _, identity = min(options)
                available.remove(identity)
            else:
                self._next_id += 1
                identity = self._next_id
            old[identity] = (t, absolute_s, obj.d)
            results.append(replace(obj, id=identity))
        self._tracks[camera] = old
        return results


def ground_candidates(g, depth):
    """Depth-only near-ground mask for a level camera. Grade can invalidate the approximation."""
    vertical = np.arange(g.h, dtype=np.float32)[:, None] - (g.h * 0.5)
    down = depth * vertical / g.f
    return (np.abs(down - g.z) < 0.6).astype(np.uint8)
