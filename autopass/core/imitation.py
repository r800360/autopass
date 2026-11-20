"""Small supervised steering policy with explicit bounds and waypoint fallback."""
import json
import math
from dataclasses import dataclass
from importlib.resources import files

MODEL = "steering_model.json"


def features(dx: float, dy: float, look: float, prev: float) -> tuple[float, ...]:
    a = math.atan2(dy, dx)
    b = math.atan2(dy, look)
    return (1.0, a, b, prev, a**3, b**3, a*a*b, a*b*b, a**5, b**5, math.tanh(6*a),
            math.tanh(8*a), math.tanh(12*a), math.tanh(5*(a+b)), math.tanh(8*(a+b)))


@dataclass(frozen=True)
class ImitationSteering:
    weights: tuple[float, ...]

    @classmethod
    def load(cls):
        model = json.loads(files("autopass.core").joinpath(MODEL).read_text(encoding="utf-8"))
        return cls(tuple(model["weights"]))

    def predict(self, dx: float, dy: float, look: float, prev: float, limit: float) -> float | None:
        if not all(math.isfinite(x) for x in (dx, dy, look, prev, limit)):
            return None
        if not (3 <= dx <= 32 and abs(dy) <= 7.5 and 6 <= look <= 30 and abs(prev) <= limit):
            return None
        steer = sum(v*w for v, w in zip(features(dx, dy, look, prev), self.weights, strict=True))
        return max(-limit, min(limit, steer))
