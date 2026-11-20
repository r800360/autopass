"""Video for humans: front camera with perception/decision overlay beside a bird's-eye view.

Frames are piped straight into an H.264 encoder. Nothing is written per frame: a run leaves
one small MP4, one sprite sheet of thumbnails and the telemetry JSON, instead of a folder of
PNGs a hundred times their size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PHASE_COLOR = {"follow": (90, 150, 235), "lane_change": (245, 170, 60), "overtake": (70, 200, 110),
               "merge_back": (70, 200, 110), "abort": (235, 80, 80), "done": (150, 150, 160)}
OK, BAD, UNKNOWN = (70, 200, 110), (235, 80, 80), (130, 135, 145)
GATE_LABELS = [("front_room", "ROOM"), ("warrant", "WORTH IT"), ("rear", "REAR"), ("blind_spot", "BLIND SPOT"),
               ("passing_lane", "LANE AHEAD"), ("corridor", "CORRIDOR")]
ROLE_COLOR = {"lead": (70, 200, 110), "passing": (245, 170, 60), "oncoming": (235, 80, 80), "other": (170, 175, 185)}
_FONTS: Dict[Tuple[int, bool], Any] = {}


def font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _FONTS:
        f = None
        for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
            try:
                f = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        if f is None:
            try:
                import matplotlib
                p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
                f = ImageFont.truetype(str(p), size)
            except Exception:  # noqa: BLE001
                f = ImageFont.load_default()
        _FONTS[key] = f
    return _FONTS[key]


@dataclass
class Hud:
    title: str = ""
    t: float = 0.0
    phase: str = "follow"
    speed: float = 0.0
    target_speed: float = 0.0
    gates: Dict[str, Optional[bool]] = field(default_factory=dict)
    tools: List[str] = field(default_factory=list)
    verdict: str = ""
    planner: str = ""
    boxes: List[Tuple[Tuple[int, int, int, int], str, str]] = field(default_factory=list)   # (bbox, role, label)


def _chip(draw, x, y, text, fill, f, pad=6, fg=(255, 255, 255)):
    w = int(draw.textlength(text, font=f))
    h = f.size + 2 * 3
    draw.rounded_rectangle([x, y, x + w + 2 * pad, y + h], radius=5, fill=fill)
    draw.text((x + pad, y + 2), text, fill=fg, font=f)
    return x + w + 2 * pad + 6


def draw_front(rgb: np.ndarray, hud: Hud) -> Image.Image:
    img = Image.fromarray(rgb).convert("RGBA")
    W, H = img.size
    k = H / 450.0
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    f_s, f_m, f_b = font(int(13 * k)), font(int(15 * k)), font(int(15 * k), True)
    for bbox, role, label in hud.boxes:
        col = ROLE_COLOR.get(role, ROLE_COLOR["other"])
        x0, y0, x1, y1 = bbox
        d.rectangle([x0, y0, x1, y1], outline=col + (255,), width=max(2, int(2 * k)))
        tw = int(d.textlength(label, font=f_s))
        ty = max(0, y0 - f_s.size - 6)
        d.rectangle([x0, ty, x0 + tw + 8, ty + f_s.size + 5], fill=col + (235,))
        d.text((x0 + 4, ty + 1), label, fill=(15, 18, 24, 255), font=f_s)
    # top panel
    ph = int(118 * k)
    d.rectangle([0, 0, W, ph], fill=(12, 16, 24, 205))
    x, y = int(12 * k), int(8 * k)
    d.text((x, y), hud.title, fill=(235, 238, 245, 255), font=f_b)
    right = f"t = {hud.t:5.1f} s    {hud.speed * 3.6:3.0f} km/h"
    d.text((W - int(d.textlength(right, font=f_m)) - x, y), right, fill=(200, 205, 215, 255), font=f_m)
    y += int(24 * k)
    cx = _chip(d, x, y, hud.phase.replace("_", " ").upper(), PHASE_COLOR.get(hud.phase, UNKNOWN) + (255,), f_b)
    for key, label in GATE_LABELS:
        v = hud.gates.get(key)
        cx = _chip(d, cx, y, label, (OK if v else BAD if v is False else UNKNOWN) + (255,), f_s)
    y += int(30 * k)
    tools = "  >  ".join(t.replace("sense_", "").replace("_", " ") for t in hud.tools) or "-"
    d.text((x, y), f"{hud.planner} planner:  {tools}"[: int(118 * W / 800)], fill=(170, 200, 255, 255), font=f_s)
    y += int(20 * k)
    d.text((x, y), ("critic:  " + hud.verdict)[: int(118 * W / 800)], fill=(235, 220, 170, 255), font=f_s)
    return Image.alpha_composite(img, over).convert("RGB")


def compose(front: Optional[np.ndarray], overhead: Optional[np.ndarray], hud: Hud) -> np.ndarray:
    panels = []
    if front is not None:
        panels.append(draw_front(front, hud))
    if overhead is not None:
        o = Image.fromarray(overhead)
        d = ImageDraw.Draw(o)
        d.text((10, o.height - 24), "bird's-eye view (for the viewer; the agent never sees it)",
               fill=(255, 255, 255), font=font(13), stroke_width=2, stroke_fill=(0, 0, 0))
        panels.append(o)
    if not panels:
        return np.zeros((2, 2, 3), np.uint8)
    h = panels[0].height
    out = Image.new("RGB", (sum(p.width for p in panels) + 4 * (len(panels) - 1), h), (12, 16, 24))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + 4
    w = out.width - (out.width % 2)
    return np.asarray(out)[: h - (h % 2), :w]


class VideoWriter:
    """Streams frames into ffmpeg (H.264, yuv420p, faststart) and collects thumbnails for a sprite sheet."""

    def __init__(self, path: Path, fps: int = 20, crf: int = 25, thumb_every_s: float = 1.0, thumb_w: int = 400):
        self.path, self.fps, self.crf = Path(path), fps, crf
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._gen = None
        self._n = 0
        self._thumb_every = max(1, int(round(thumb_every_s * fps)))
        self._thumb_w = thumb_w
        self.thumbs: List[Image.Image] = []

    def write(self, frame: np.ndarray) -> None:
        if self._gen is None:
            import imageio_ffmpeg
            h, w = frame.shape[:2]
            self._gen = imageio_ffmpeg.write_frames(
                str(self.path), (w, h), fps=self.fps, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                macro_block_size=2, ffmpeg_log_level="error",
                output_params=["-crf", str(self.crf), "-preset", "medium", "-movflags", "+faststart"])
            self._gen.send(None)
        if self._n % self._thumb_every == 0:
            im = Image.fromarray(frame)
            self.thumbs.append(im.resize((self._thumb_w, int(im.height * self._thumb_w / im.width))))
        self._gen.send(np.ascontiguousarray(frame).tobytes())
        self._n += 1

    def close(self, sprite_path: Optional[Path] = None, cols: int = 6) -> Dict[str, Any]:
        if self._gen is not None:
            self._gen.close()
        info: Dict[str, Any] = {"frames": self._n, "fps": self.fps}
        if sprite_path is not None and self.thumbs:
            tw, th = self.thumbs[0].size
            rows = (len(self.thumbs) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * tw, rows * th), (12, 16, 24))
            for i, t in enumerate(self.thumbs):
                sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
            sheet.save(sprite_path, quality=72)
            info["sprite"] = {"file": Path(sprite_path).name, "cols": cols, "tile": [tw, th],
                              "count": len(self.thumbs), "every_s": self._thumb_every / self.fps}
        return info
