"""Fit a bounded steering imitator to waypoint tracking action labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopass.core.control import ControlConfig, steer_toward
from autopass.core.imitation import features, MODEL

ROOT = Path(__file__).resolve().parents[1]


def dataset(seed: int, count: int):
    rng = random.Random(seed)
    config = ControlConfig()
    rows, commands = [], []
    for _ in range(count):
        look = rng.uniform(6, 30)
        dx = max(3., min(32., look + rng.uniform(-3., 3.)))
        dy = rng.uniform(-7.5, 7.5)
        previous = rng.uniform(-config.max_steer, config.max_steer)
        rows.append(features(dx, dy, look, previous))
        commands.append(steer_toward(dx, dy, look, previous, config))
    return np.array(rows), np.array(commands)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "autopass" / "core" / MODEL)
    args = parser.parse_args()
    train, target = dataset(61904, 25000)
    test, actual = dataset(41821, 5000)
    penalty = 0.001 * np.eye(train.shape[1])
    weights = np.linalg.solve(train.T @ train + penalty, train.T @ target)
    error = np.abs(np.clip(test @ weights, -0.75, 0.75) - actual)
    report = {
        "schema": 1, "algorithm": "ridge imitation on geometric basis features",
        "train_seed": 61904, "test_seed": 41821, "train_count": len(train), "test_count": len(test),
        "heldout_mae": round(float(np.mean(error)), 6),
        "heldout_p95_absolute_error": round(float(np.quantile(error, 0.95)), 6),
        "weights": [round(float(w), 12) for w in weights],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Held-out command MAE {report['heldout_mae']:.4f}, p95 {report['heldout_p95_absolute_error']:.4f}")


if __name__ == "__main__":
    main()
