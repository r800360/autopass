"""Generate the printable release brief from the exported evidence."""
from __future__ import annotations

import gzip
import html
import json
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"
DATA = SITE / "data"
E = html.escape

STYLE = """
@page {size: letter; margin: .45in .62in;}
:root {font-family: "Segoe UI", Arial, sans-serif; color: #182620;}
body{margin:0;background:#eef2ef;font-size:9.8pt;line-height:1.35;}
article{max-width:8.5in;margin:0 auto;background:white;padding:.42in .53in;}
section{min-height:9.55in;position:relative;break-after:page;}
section:last-child{break-after:auto;}
h1,h2,h3,p{margin:0;}h1,h2,h3{line-height:1.1;}
h1{font-size:39pt;letter-spacing:-.04em;margin:8pt 0 10pt;}
h2{font-size:25pt;letter-spacing:-.03em;margin:3pt 0 12pt;}
h3{font-size:12pt;margin:10pt 0 4pt;}
p{margin:6pt 0;}
a{color:#146148;text-decoration:underline;}
.kicker{color:#247854;font-size:8.5pt;letter-spacing:.13em;text-transform:uppercase;font-weight:700;}
.intro{font-size:13pt;max-width:93%;}
.meta{font-size:9pt;color:#4c645a;margin:11pt 0 14pt;}
img{width:100%;display:block;border-radius:6pt;max-height:2.45in;object-fit:contain;background:#18312e;}
.caption{font-size:8.6pt;color:#4c645a;margin-top:5pt;}
.facts{display:grid;grid-template-columns:repeat(3,1fr);gap:12pt;border-block:1pt solid #a8c2b5;padding:12pt 0;margin:15pt 0;}
.facts strong{display:block;color:#166348;font-size:22pt;line-height:1.1;}
.facts span{font-size:8.8pt;color:#4c645a;}
ol,ul{padding-left:16pt;}li{margin:5pt 0;}
.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:8pt;margin:13pt 0;}
.flow div{padding:8pt;background:#eef6f1;border-left:2pt solid #2b926a;border-radius:3pt;}
.flow b{display:block;margin-bottom:3pt;}
.flow span{font-size:8.5pt;color:#435d52;}
.summary{margin-top:9pt;}
table{width:100%;border-collapse:collapse;font-size:9.3pt;table-layout:fixed;}
th,td{padding:4pt 4pt;text-align:center;border-bottom:1px solid #dce8e0;}
th{color:#36584a;background:#edf5f0;font-size:8.4pt;line-height:1.2;}
td:first-child,th:first-child{text-align:left;width:28%;}
tr.primary td{color:#155d44;font-weight:700;}
.callout{background:#edf5f0;border-left:3pt solid #338d69;padding:9pt 12pt;margin:10pt 0;}
.small{font-size:9pt;color:#456055;}
pre{white-space:pre-wrap;font-size:8.9pt;background:#edf5f0;padding:9pt;border-radius:4pt;}
footer{display:flex;justify-content:space-between;margin-top:12pt;padding-top:7pt;border-top:1pt solid #d3dfd8;color:#52675d;font-size:8.3pt;}
@media print {body{background:white;}article{padding:0;}section{min-height:auto;}h3,table,.callout,.flow,.facts{break-inside:avoid;}}
"""

def read(name):
    file = DATA / name
    if file.suffix == ".gz":
        with gzip.open(file, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(file.read_text(encoding="utf-8"))

def labels(key):
    return {"autopass": "AutoPass-Gen", "no_pass": "Never pass", "always_pass": "Always pass",
            "first_release": "Fixed thresholds", "autopass_rgbd": "NanoDet RGB-depth",
            "autopass_imitation": "Imitation steering", "imitation": "Imitation steering"}.get(key, key.replace("_", " "))

def table(metrics: dict, labeled=False):
    body = []
    for key, row in metrics.items():
        n = row["scored"]
        expected = f'<td>{row["expected_behavior"]}/{row["labeled_cases"]}</td>' if labeled else ""
        body.append(f"""<tr class="{'primary' if key == 'autopass' else ''}"><td>{E(labels(key))}</td>
            <td>{n}/{row['attempted']}</td><td>{row['overtakes']}</td><td>{row['collisions']}</td>
            <td>{row['unsafe_commits']}</td>{expected}</tr>""")
    head = '<th>Expected behavior</th>' if labeled else ''
    return """<table><thead><tr><th>Policy</th><th>Scored / targets</th><th>Overtakes</th><th>Collisions</th>
        <th>Unsafe commits</th>""" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"

def model_summary():
    path = DATA/"llm-trials.json"
    if not path.exists():
        return "<p>Rule and model planners share the tool interface and critic. Model runs record calls, timing, and explicit outcomes.</p>"
    output = []
    for entry in read(path.name):
        r = entry["result"]
        name = {"llm_plan": "Batch tool plan", "llm_react": "Tool at a time", "llm_direct": "Direct decision"}.get(entry["planner"], entry["planner"])
        kind = "overtake completed" if r["overtake_completed"] else "no completed overtake"
        output.append(f"<li><b>{E(name)}, {r['scenario_id'][:3]}.</b> {r['agency']['llm_calls']} calls, "
                      f"{r['agency']['llm_latency_s']:.1f} s aggregate model time, {kind}.</li>")
    return "<ul>" + "".join(output) + "</ul>"

def carla_summary(manifest):
    output = []
    for cohort in manifest["cohorts"]:
        if cohort["id"].startswith("carla"):
            output.append(f"<h3>{E(cohort['title'])}</h3>{table(cohort['metrics'], labeled=True)}")
    return "".join(output)

def main():
    index = read("manifest.json")
    catalog = read("kinematic-catalog.json.gz")
    sample = read("kinematic-sampled.json.gz")
    stress = read("kinematic-degraded.json.gz")["summary"]["autopass"]
    imitation = read("kinematic-imitation.json.gz")["summary"]["imitation"]
    paired = catalog["paired_vs_never_pass"]["autopass"]
    own, drawn = catalog["summary"]["autopass"], sample["summary"]["autopass"]
    clip = index["clips"][0]
    trip = read("trip-reroute.json")
    fixed = read("trip-fixed.json")
    distribution = drawn["collision_interval_wilson"]
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
    <title>AutoPass-Gen release report</title><style>{STYLE}</style><body><article>
    <section>
    <p class="kicker">Release report · November 19, 2026</p>
    <h1>AutoPass-Gen</h1>
    <p class="intro">Overtaking under uncertainty in a closed-loop simulator</p>
    <p class="meta">Rohan Sachdeva · Xinwei Mai · Pranav Prabu · Rathang Pandit</p>
    <img src="{E(clip['poster'])}" alt="CARLA ego view and overhead image of an approved motorway pass">
    <p class="caption">CARLA 0.9.16, Town04. Camera and overhead views captured during an executed run.</p>
    <div class="facts"><div><strong>{own['expected_behavior']}/{own['labeled_cases']}</strong><span>Expected behavior in the geometric catalog</span></div>
    <div><strong>{drawn['overtakes']}/{drawn['scored']}</strong><span>Completed overtakes in sampled traffic</span></div>
    <div><strong>{stress['collisions']}/{stress['scored']}</strong><span>Collisions with degraded sensing</span></div></div>
    <h3>Research question</h3>
    <p>Can active evidence gathering improve progress around slow leads while explicit at-commit constraints limit maneuver risk? The same loop supports a rule planner and an optional language model.</p>
    <div class="flow"><div><b>Sensory evidence</b><span>Instance masks and depth, lane projection, relative motion</span></div>
    <div><b>Tool planner</b><span>Mutable belief, freshness, memory, tool scheduling</span></div>
    <div><b>Deterministic critic</b><span>Rear headway, visible clearance, blind spot, corridor</span></div>
    <div><b>Closed-loop control</b><span>Follow, change lanes, monitor, merge or abort</span></div></div>
    <p>Rohan built the original live CARLA integration, vision grounding, waypoint controller and evaluation harness. Xinwei led the initial hypothesis and passing architecture. Pranav led communication and contributed tool-priority debugging. Rathang audited motion evidence and contextualized failures.</p>
    <footer><a href="https://r800360.github.io/autopass/">Interactive lab</a><span>1 · System</span></footer>
    </section>

    <section>
    <p class="kicker">Evaluation</p><h2>Progress and risk</h2>
    <h3>34 authored kinematic cases</h3>
    {table(catalog["summary"], labeled=True)}
    <p class="small">Across {paired['both_clear_80m']} matched cases in which both policies reached 80 m collision-free, the median AutoPass time saving versus never-pass was {paired['median_seconds_saved']:.2f} s.</p>
    <h3>500 sampled kinematic cases per policy</h3>
    {table(sample["summary"])}
    <p class="small">Seed {sample['seed']}. AutoPass had {drawn['unsafe_commits']} unsafe and {drawn['unwarranted_commits']} unwarranted commits. The 95% Wilson interval for its collision rate was {100*distribution[0]:.2f}% to {100*distribution[1]:.2f}% under the configured draws.</p>
    <div class="callout"><b>Sensing sensitivity</b><p>With synthetic noise and missed detections, AutoPass had {stress['collisions']} collisions and {stress['unsafe_commits']} unsafe commits. The result motivates explicit sensing coverage and uncertainty checks.</p></div>
    <h3>Scoring</h3>
    <p>Completed passes clear the lead and finish the return to lane. Collision events come from the simulator. An unsafe commit is a violation of the modeled hazard gates evaluated against simulator state at commit. Catalog labels additionally include the expected hold or yield behavior.</p>
    <h3>Audit findings</h3>
    <p>Older traces contained greedy-backstop proposals and early oncoming approvals with incomplete evidence. The rebuild separates instance identities, rejects expired or out-of-order checks, budgets requests, and checks for travel-lane clearance before aborting. The release preserves failures in the benchmark rather than hiding them in a composite score.</p>
    <footer><a href="https://github.com/r800360/autopass/blob/main/docs/EVALUATION.md">Metrics and experiment commands</a><span>2 · Evaluation</span></footer>
    </section>

    <section>
    <p class="kicker">Full-simulator trials</p><h2>What CARLA reproduced</h2>
    <p>CARLA camera callbacks and vehicle controls were exercised across four staged maps. The early sweep records integration limitations. Spot checks exercise perception and controller changes. Targeted regressions audit the rolling-start observation window and lane localization.</p>
    {carla_summary(index)}
    <p class="small">Conservative holds can fail scenario expectations when the camera lacks adequate rear or opposing-lane reach. Weather, occlusion, map staging, and actor dynamics influence sensor observations.</p>
    <footer><a href="https://r800360.github.io/autopass/#lab">Recorded clips and telemetry</a><span>3 · CARLA</span></footer>
    </section>

    <section>
    <p class="kicker">Extensions</p><h2>Models, routes, and reproduction</h2>
    <h3>Live local inference</h3>
    <p>Qwen 2.5 1.5B Instruct Q4_K_M with llama.cpp. Tool sequences and terminal choices are logged. Direct model control is an experimental kinematic baseline. The runtime used the deterministic planner for the main campaigns.</p>
    {model_summary()}
    <h3>Learned steering and perception</h3>
    <p>The bounded steering imitator matched {imitation['expected_behavior']} of {imitation['labeled_cases']} catalog labels. Training used 25,000 synthetic teacher actions and validation used 5,000 independently sampled actions. Held-out mean absolute steering error was 0.0045 on normalized controls. NanoDet uses RGB to identify cars and CARLA depth to estimate metric gaps. CARLA runs characterize this hybrid sensor profile.</p>
    <h3>Trip navigation</h3>
    <p>A five-node graph invokes the passing loop on each segment. Geometry and arrival clocks define path cost. After repeated denials at Market, the adaptive route reached Clinic in {trip['elapsed_s']:.2f} s compared with {fixed['elapsed_s']:.2f} s following the fixed route in this staged example.</p>
    <h3>Reproduce locally</h3>
    <pre>python -m pip install -e '.[dev]'
python -m pytest
autopass run s17 --trace runs/rear.json.gz
autopass bench --count 500 --seed 20261119 --out runs/heldout.json</pre>
    <p class="small">The browser runs the kinematic loop with Pyodide. Video, contact sheets and compressed telemetry replace raw-frame exports for routine review. Evidence updates independently of the static page build. The experimental scope is simulated overtaking and staged routing.</p>
    <footer><a href="https://github.com/r800360/autopass">Source and reproducibility guide</a><span>4 · Extensions</span></footer>
    </section>
    </article></body></html>"""
    (SITE/"report.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
