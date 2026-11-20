"""Export a size-bounded, portable showcase bundle from audited run records."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from autopass.cli import experiment
from autopass.core.scenario import catalog
from autopass.core.telemetry import load
from autopass.evaluation import summarize

CLIPS = [
    ("clear", "s01", "Clear motorway pass", "Observe an approved pass and vision-assisted merge."),
    ("rear", "s17", "Closing rear traffic", "Inspect the rear-traffic rejection and the later commit."),
    ("blind", "s24", "Occupied blind spot", "The adjacent vehicle stays alongside. The ego holds the lane."),
    ("oncoming", "s19", "Oncoming traffic", "Inspect the sight-distance estimate and the final outcome."),
    ("abort", "s27", "Late oncoming traffic", "A staged late arrival tests the maneuver monitor."),
    ("rural", "s13", "Clear opposing lane", "Watch how the available sight and road geometry affect the pass."),
]


def write_json(p: Path, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    if p.suffix == ".gz":
        p.write_bytes(gzip.compress(text.encode(), compresslevel=9, mtime=0))
    else:
        p.write_text(text, encoding="utf-8")


def recorded(path: Path):
    rows = []
    for name in sorted(path.glob("*/*/result.json")):
        value = json.loads(name.read_text(encoding="utf-8"))
        value["policy_label"] = name.parent.parent.name
        rows.append(value)
    finished = {(r["scenario_id"], r.get("policy_label", r["policy"])) for r in rows}
    for name in sorted(path.glob("*/*/error.json")):
        key = (name.parent.name, name.parent.parent.name)
        if key not in finished:
            rows.append({"scenario_id": key[0], "policy_label": key[1], "error": "trial did not produce a result"})
    return rows


def cohort(prefix, title, note, report: dict, output: Path):
    target = f"data/{prefix}.json.gz"
    write_json(output / target, report)
    return {"id": prefix, "title": title, "note": note, "raw": target, "metrics": report["summary"]}


def clip(prefix, sid, title, note, sources, output, compressed=False, policy="autopass"):
    source = next((entry for base in sources for entry in sorted(base.glob(f"{policy}/{sid}*/video.mp4"))
                   if entry.with_name("telemetry.json.gz").is_file()), None)
    if source:
        from imageio_ffmpeg import get_ffmpeg_exe
        ff = get_ffmpeg_exe()
        telemetry = load(source.with_name("telemetry.json.gz"))
        video_out = output / "media" / f"{prefix}.mp4"
        poster = output / "media" / f"{prefix}.webp"
        if compressed or not video_out.exists() or not poster.exists():
            subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                            "-map_metadata", "-1", "-an", "-vf", "scale='min(1440,iw)':-2",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "fast",
                            "-movflags", "+faststart", str(video_out)], check=True)
            result = telemetry["result"]
            at = min(result["duration_s"]-0.2, (result.get("t_commit_s") or 1.6) + 2.1)
            subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-ss", str(at),
                            "-i", str(video_out), "-frames:v", "1", "-c:v", "libwebp", "-quality", "88",
                            str(poster)], check=True)
        capture = "CARLA 0.9.16 · " + telemetry["meta"]["scenario"]["town"]
        item = {"id": prefix, "title": title + " · CARLA", "source": capture,
                "video": f"media/{video_out.name}", "poster": f"media/{poster.name}",
                "caption": capture + " · " + title + " · Ego camera and overhead view"}
    else:
        _, record = experiment(sid)
        telemetry = record.to_dict()
        item = {"id": prefix, "title": title + " · Kinematic", "source": "Kinematic replay · Projected geometry"}
    name = f"data/trace-{prefix}.json"
    write_json(output / name, telemetry)
    return {**item, "trace": name, "note": note}


def runtime(output):
    names = []
    for path in sorted((ROOT/"autopass").rglob("*")):
        if path.suffix in (".py", ".json") and path.is_file() and "carla_backend" not in path.parts:
            names.append(path)
    with ZipFile(output/"data/runtime.zip", "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in names:
            entry = ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 11, 19, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT/"runs")
    parser.add_argument("--site", type=Path, default=ROOT/"site")
    parser.add_argument("--edition", type=int, default=2026111901)
    parser.add_argument("--refresh-video", action="store_true")
    args = parser.parse_args()
    source, out = args.runs, args.site
    (out/"media").mkdir(parents=True, exist_ok=True)
    (out/"data").mkdir(parents=True, exist_ok=True)
    specs = [
        ("catalog", "kinematic-catalog", "Kinematic catalog · 34 cases", "Paired geometric-sensing tests with 23 adapted and 11 stress scenarios."),
        ("heldout", "kinematic-sampled", "Kinematic sampled traffic · seeded", "Synthetic held-out combinations. Rates apply to these draws. The sensor model has finite range and occlusion."),
        ("degraded", "kinematic-degraded", "Kinematic sensing stress", "Same seeded traffic with a synthetic noise and miss model. These parameters are a sensitivity test."),
        ("imitation", "kinematic-imitation", "Kinematic steering ablation", "Same 34 scenarios and critic with the bounded learned steering controller."),
    ]
    cohorts = []
    for filename, slug, title, note in specs:
        report = json.loads((source/f"{filename}.json").read_text(encoding="utf-8"))
        if filename == "imitation":
            report["summary"] = {"imitation": report["summary"]["autopass"]}
        cohorts.append(cohort(slug, title, note, report, out))
    towns = set()
    for folder, title, note in (
        ("carla-final", "CARLA verified cases", "Selected corrected-backend trials driven by CARLA 0.9.16 sensor and control APIs."),
        ("carla-catalog", "CARLA engineering sweep", "Pre-release integration sweep. Startup and visibility issues discovered during this sweep guided the final verification."),
        ("carla-ablation", "CARLA ablations", "Selected simulator runs using RGB-depth perception, imitation steering or baseline policies."),
        ("carla-rolling", "CARLA rolling-start diagnostic", "The first 1.5 s window still allowed an early fast-lead pass."),
        ("carla-stable", "CARLA stabilized start", "A 2.5 s observation window and fresh commit decisions."),
        ("carla-commit", "CARLA commit guard", "Final check with map clearance revalidated at the proposal tick."),
        ("carla-localization", "CARLA localization regression", "Local route projection avoids large coordinate jumps on curved maps."),
    ):
        rows = recorded(source / folder)
        if not rows:
            continue
        keys = list(dict.fromkeys(r["policy_label"] for r in rows))
        report = {"schema": 1, "rows": rows,
                  "summary": {key: summarize(r for r in rows if r["policy_label"] == key) for key in keys}}
        towns.update(r["scenario_id"].split("_")[1] for r in rows if "error" not in r)
        cohorts.append(cohort(folder, title, note, report, out))
    lang = []
    for path in sorted((source/"llm").glob("*.json.gz")):
        record = load(path)
        if not record["result"]["agency"]["llm_calls"]:
            continue
        label = record["result"]["planner"]
        if "direct" in path.name:
            label = "llm_direct"
        name = "data/model-" + path.name.removesuffix(".gz")
        write_json(out/name, record)
        lang.append({"file": name, "planner": label, "result": record["result"]})
    if lang:
        write_json(out/"data/llm-trials.json", lang)
    for variant in ("reroute", "fixed", "llm"):
        path = source/f"trip-{variant}.json"
        if path.exists():
            shutil.copyfile(path, out/"data"/path.name)
    clips = []
    for spec in CLIPS:
        captures = (source/"carla-final", source/"verify") if spec[0] == "rear" else (
            source/"carla-stable", source/"carla-final", source/"verify")
        clips.append(clip(*spec, captures, out, compressed=args.refresh_video))
    if any((source/"carla-stable").glob("autopass/s17*/video.mp4")):
        clips.append(clip("rear-diagnostic", "s17", "Rear traffic · corridor regression",
                          "This captured run completed a late pass. At commit, the available corridor was shorter than the model required.",
                          (source/"carla-stable",), out, compressed=args.refresh_video))
    if any((source/"carla-ablation").glob("autopass_rgbd/s01*/video.mp4")):
        clips.append(clip("rgbd", "s01", "Learned RGB-depth perception",
                          "NanoDet identifies objects from RGB, with metric depth used for range.",
                          (source/"carla-ablation",), out, compressed=args.refresh_video,
                          policy="autopass_rgbd"))
    model = next((r for r in lang if r["file"] == "data/model-s17-plan.json"), None)
    if model:
        clips.append({"id": "model-rear", "title": "Qwen tool planner · Closing rear traffic",
                      "source": "Kinematic replay · Qwen 2.5 tool plan", "trace": model["file"],
                      "note": "Watch model-selected probes, critic checks, and the executed decision."})
    runtime(out)
    write_json(out/"data/manifest.json", {
        "schema": 1, "edition": args.edition,
        "stats": {"catalog": len(catalog()), "draws": cohorts[1]["metrics"]["autopass"]["scored"], "towns": len(towns)},
        "catalog": [{"id": s.id, "urgency": s.urgency, "title": s.id[8:].replace("_", " ").capitalize()}
                    for s in catalog().values()],
        "clips": clips, "cohorts": cohorts,
    })
    print(f"Wrote {len(clips)} replay records and {len(cohorts)} cohorts")


if __name__ == "__main__":
    main()
