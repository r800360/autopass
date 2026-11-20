const byId = (id) => document.getElementById(id);
const evidenceRoot = "https://raw.githubusercontent.com/r800360/autopass/evidence/";
const rawRecords = "https://github.com/r800360/autopass/tree/main/evidence";
const canvas = byId("road");
const ctx = canvas.getContext("2d");
const labels = {
  autopass: "AutoPass-Gen", no_pass: "Never pass", always_pass: "Always pass",
  first_release: "Fixed thresholds", no_critic: "Critic disabled", imitation: "Imitation steering",
  llm_plan: "LLM tool plan", llm_react: "LLM step-by-step", llm_direct: "LLM direct",
};
const phaseLabel = { follow: "Follow", lane_change: "Change lane", overtake: "Overtake",
  merge_back: "Merge back", abort: "Abort", done: "Pass complete" };
const toolLabel = { sense_front: "Front", sense_passing_lane: "Passing lane", sense_rear: "Rear",
  sense_blind_spot: "Blind spot", check_corridor: "Corridor", peek: "Peek", propose_pass: "Propose", hold: "Hold" };
let data, trace, mode = "replay", laterId = 0, position = 0, playing = false, playStart, simStart, worker;
let objectURL, activeClip, traceBase = document.baseURI;
let pending = 0;
let activeCycle;

function element(tag, text, klass) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (klass) node.className = klass;
  return node;
}

function json(url, options) {
  return fetch(url, options).then((response) => {
    if (!response.ok) throw new Error("Could not load the evidence.");
    return response.json();
  });
}

function asset(path, base = traceBase) {
  const url = new URL(path, base);
  if (url.origin !== location.origin && !url.href.startsWith(evidenceRoot)) throw new Error("Unknown evidence origin.");
  return url.href;
}

function selectOptions(select, entries) {
  select.replaceChildren(...entries.map(([value, label]) => {
    const option = element("option", label);
    option.value = value;
    return option;
  }));
}

async function evidence() {
  const snapshot = await json("data/manifest.json");
  try {
    const remote = await json(evidenceRoot + "data/manifest.json", {
      cache: "no-cache", signal: AbortSignal.timeout(2800),
    });
    if (remote.schema === 1 && remote.edition > snapshot.edition && remote.clips?.length && remote.cohorts?.length) {
      traceBase = evidenceRoot;
      return { ...remote, catalog: snapshot.catalog };
    }
  } catch {
    // The tagged local snapshot also works in previews and private networks.
  }
  return snapshot;
}

async function initialize() {
  data = await evidence();
  byId("stat-catalog").textContent = data.stats.catalog;
  byId("stat-draws").textContent = data.stats.draws;
  byId("stat-towns").textContent = data.stats.towns;
  selectOptions(byId("replay-select"), data.clips.map((c) => [c.id, c.title]));
  selectOptions(byId("scenario"), data.catalog.map((s) => [s.id, s.title]));
  byId("scenario").value = data.catalog.find((s) => s.id.startsWith("s17")).id;
  selectOptions(byId("cohort"), data.cohorts.map((c) => [c.id, c.title]));
  showCohort();
  await openClip(data.clips[0].id);
}

function setActivity(text) {
  byId("activity").textContent = text;
}

function setMode(next) {
  const prior = mode;
  mode = next;
  stop();
  if (prior !== next) {
    laterId++;
    pending = 0;
    setActivity("");
    for (const id of ["run-scenario", "run-trip"]) byId(id).disabled = false;
  }
  for (const name of ["replay", "sim", "trip"]) {
    byId("tab-" + name).setAttribute("aria-selected", String(name === mode));
    byId("tab-" + name).tabIndex = name === mode ? 0 : -1;
    byId("panel-" + name).hidden = name !== mode;
  }
  byId("trip-result").hidden = next !== "trip" || !byId("trip-result").childElementCount;
  byId("maneuver-view").hidden = next === "trip";
  if (next === "replay" && prior !== next && data) openClip(byId("replay-select").value);
  if (next === "sim" && prior !== next) setActivity("Choose a scenario and run a fresh simulation.");
  if (trace && next !== "trip") paint();
}

async function openClip(id) {
  if (!data) return;
  const request = ++laterId;
  activeClip = data.clips.find((clip) => clip.id === id);
  if (!activeClip) return;
  stop();
  setActivity("Loading recorded evidence…");
  const video = byId("hero-video");
  if (activeClip.video) {
    const url = asset(activeClip.video);
    if (video.src !== url) {
      video.poster = asset(activeClip.poster);
      video.src = url;
    }
    byId("hero-caption").textContent = activeClip.caption;
  }
  byId("replay-note").textContent = activeClip.note;
  try {
    const record = await json(asset(activeClip.trace));
    if (request !== laterId) return;
    openTrace(record, activeClip.source, asset(activeClip.trace), activeClip.video ? activeClip : null);
    setActivity("");
  } catch (error) {
    if (request === laterId) setActivity(error.message);
  }
}

function openTrace(record, source, url, clip = null) {
  if (!record.frames?.length || record.schema !== 2) throw new Error("Unsupported telemetry format.");
  trace = record;
  activeCycle = undefined;
  trace.videoClip = clip;
  byId("trace-source").textContent = source;
  const seek = byId("seek");
  seek.max = record.frames.length - 1;
  seek.value = 0;
  const result = record.result;
  const states = [];
  states.push(result.overtake_completed ? "Pass completed" : result.pass_attempted ? "Pass attempted" : "Held lane");
  states.push(result.collision ? "collision observed" : "zero collisions in this run");
  if (result.unsafe_commit) states.push("commit failed the ground-truth gate check");
  if (result.aborts) states.push("aborts " + result.aborts);
  byId("trace-summary").textContent = states.join(" · ");
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = url ? null : URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], { type: "application/json" }));
  byId("trace-link").href = url || objectURL;
  byId("trace-link").download = record.meta.scenario.id + ".json";
  position = 0;
  paint();
}

function latestCycle(t) {
  const cycles = trace.cycles;
  let low = 0, high = cycles.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if ((cycles[middle].t_end ?? cycles[middle].t) <= t + .001) low = middle + 1;
    else high = middle;
  }
  return cycles[low - 1];
}

function reading(label, value) {
  const div = element("div");
  div.append(element("dt", label), element("dd", value));
  return div;
}

function formatGap(reading) {
  if (!reading) return "Unmeasured";
  if (reading.gap_m !== null) return reading.gap_m.toFixed(1) + " m";
  if (reading.range_m > 0) return "Clear to " + reading.range_m + " m";
  return "Unmeasured";
}

function paint() {
  if (!trace || byId("maneuver-view").hidden) return;
  const f = trace.frames[position];
  if (!f) return;
  const [t, phase, s, d, yaw, v, , , traffic] = f;
  byId("clock").textContent = t.toFixed(2) + " s";
  byId("phase").textContent = phaseLabel[trace.phases[phase]] || trace.phases[phase];
  const cycle = latestCycle(t);
  if (cycle !== activeCycle || position === 0) {
    const verdict = cycle?.verdicts?.at(-1);
    const prefix = verdict ? (verdict.approved ? "Approved · " : "Held · ") : "";
    byId("verdict").textContent = cycle ? prefix + (verdict?.reason || cycle.note || "Following.") : "Following. No deliberation at this time.";
    byId("tools").replaceChildren(...(cycle?.tools || []).map((x) => element("li", toolLabel[x.tool] || x.tool)));
    activeCycle = cycle;
  }
  const belief = cycle?.belief;
  byId("readings").replaceChildren(reading("Ego speed", v.toFixed(1) + " m/s"),
    reading("Lead gap", formatGap(belief?.front)), reading("Passing lane", formatGap(belief?.passing_ahead || belief?.oncoming)));

  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = Math.min(2, devicePixelRatio || 1);
  const width = rect.width, height = rect.height;
  const pw = Math.round(width * ratio), ph = Math.round(height * ratio);
  if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph; }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.fillStyle = "#1c2f2a";
  ctx.fillRect(0, 0, width, height);
  const roadTop = Math.round(height * .22), laneH = Math.round(height * .27);
  const side = (trace.meta.passing_side || trace.meta.scenario.passing_side) === "right" ? 1 : -1;
  const baseY = roadTop + (side === -1 ? 1.5 : .5) * laneH;
  ctx.fillStyle = "#35403e";
  ctx.fillRect(0, roadTop, width, laneH * 2);
  ctx.strokeStyle = "#9ba39d";
  ctx.lineWidth = 1.2;
  for (const y of [roadTop, roadTop + laneH*2]) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
  const scale = Math.max(2.3, width / 142), x0 = width * .31;
  ctx.strokeStyle = trace.meta.scenario.two_lane ? "#d9ad5d" : "#9eaca6";
  ctx.setLineDash([12, 12]);
  ctx.lineDashOffset = (s * scale) % 24;
  ctx.beginPath(); ctx.moveTo(0, roadTop + laneH); ctx.lineTo(width, roadTop + laneH); ctx.stroke();
  ctx.setLineDash([]);
  const laneW = trace.meta.lane_width || trace.meta.scenario.lane_width || 3.5;
  const place = (a, o, ang, sv, desc, ego) => {
    const x = x0 + (a-s) * scale, y = baseY + side * o / laneW * laneH;
    if (x < -25 || x > width + 25) return;
    const length = Math.max(12, scale * (desc?.len || 4.5));
    const carW = Math.max(13, (desc?.wid || 1.9) * laneH / laneW * .58);
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate((sv < -.1 && Math.abs(ang) < 1.5 ? Math.PI : 0) + side * ang);
    ctx.fillStyle = ego ? "#abf2ca" : desc?.role === "lead" ? "#f39b5c" : "#b5c4cf";
    ctx.beginPath();
    ctx.roundRect(-length / 2, -carW / 2, length, carW, 3);
    ctx.fill();
    ctx.fillStyle = "#2a403d";
    ctx.fillRect(length*.10, -carW*.33, Math.max(2, length*.16), carW*.66);
    ctx.restore();
  };
  for (const [id, as, ad, ay, av] of traffic) place(as, ad, ay, av, trace.actors[id], false);
  place(s, d, yaw, v, trace.actors["0"], true);
  ctx.fillStyle = "#c7d1c9";
  ctx.font = "12px Segoe UI, system-ui";
  ctx.fillText(trace.meta.scenario.two_lane ? "Opposing traffic" : "Passing lane", 11, 23);
  ctx.fillText("Ego progress  " + s.toFixed(0) + " m", 11, height-15);
  if (width > 290) {
    ctx.fillText("20 m", width-23-scale*20, height-16);
    ctx.strokeStyle = "#b2c3be";
    ctx.beginPath(); ctx.moveTo(width-15-scale*20,height-9); ctx.lineTo(width-15,height-9); ctx.stroke();
  }
}

function setPosition(value) {
  if (!trace) return;
  position = Math.min(trace.frames.length - 1, Math.max(0, Number(value)));
  byId("seek").value = position;
  paint();
}

function frameAt(t) {
  let lo = 0, hi = trace.frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >>> 1;
    if (trace.frames[mid][0] <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

function tick(now) {
  if (!playing) return;
  const t = simStart + (now - playStart) / 1000;
  setPosition(frameAt(t));
  if (t >= trace.frames.at(-1)[0]) stop();
  else requestAnimationFrame(tick);
}

function stop() {
  playing = false;
  byId("play").textContent = "Play";
}

function requestModel(kind, payload) {
  const id = ++laterId;
  pending = id;
  if (!worker) {
    worker = new Worker("worker.js");
    worker.onerror = () => {
      worker.terminate();
      worker = null;
      setActivity("The Python runtime could not load. Recorded evidence is still available.");
      for (const button of ["run-scenario","run-trip"]) byId(button).disabled = false;
    };
    worker.onmessage = ({ data: msg }) => {
      if (msg.id !== pending) return;
      for (const button of ["run-scenario","run-trip"]) byId(button).disabled = false;
      if (msg.error) {
        setActivity(msg.error);
        return;
      }
      setActivity("");
      if (mode === "trip") showTrip(msg.result);
      else if (mode === "sim") openTrace(msg.result, "Kinematic simulation · Python in this browser");
    };
  }
  setActivity("Running Python simulation. The runtime download may take a moment on first use.");
  worker.postMessage({ id, kind, payload });
}

function showTrip(results) {
  const panel = byId("trip-result");
  panel.replaceChildren();
  for (const result of results) {
    const row = element("div", undefined, "trip-line");
    const status = result.arrived ? result.elapsed_s.toFixed(2) + " s" : "Trip incomplete";
    row.append(element("h3", (result.replan ? "Adaptive route" : "Fixed route") + " · " + status));
    const points = [result.request.origin, ...result.trace.filter((e) => e.kind === "segment" && e.arrived).map((e) => e.to)];
    const nodes = element("div", undefined, "trip-nodes");
    nodes.append(...points.map((point) => element("span", point)));
    row.append(nodes);
    const event = result.trace.find((e) => e.kind === "reroute");
    if (event) row.append(element("p", "Rerouted at " + event.at + " after " + event.reason + ". Predicted remaining time fell from "
      + event.prior_eta_s + " s to " + event.new_eta_s + " s.", "muted"));
    if (result.deadline_met !== null) row.append(element("p", result.deadline_met ? "Deadline met" : "Deadline missed", "muted"));
    panel.append(row);
  }
  panel.hidden = false;
}

function showCohort() {
  const target = data.cohorts.find((c) => c.id === byId("cohort").value) || data.cohorts[0];
  byId("cohort-note").textContent = target.note;
  byId("bench-evidence").href = target.raw ? asset(target.raw) : rawRecords;
  const rows = Object.entries(target.metrics).map(([policy, m]) => {
    const row = element("div", undefined, "bench-row");
    const label = element("strong", labels[policy] || policy);
    const pass = element("div", m.overtakes + " / " + m.scored);
    pass.append(element("small", "Completed overtakes"));
    const meter = element("div", undefined, "meter");
    const fill = element("span");
    fill.style.width = (m.scored ? (100 * m.overtakes / m.scored) : 0) + "%";
    meter.setAttribute("aria-hidden", "true");
    meter.append(fill); pass.append(meter);
    const col = element("div", m.collisions + " / " + m.scored, m.collisions ? "warning" : "");
    col.append(element("small", "Collisions"));
    const unsafe = element("div", String(m.unsafe_commits), m.unsafe_commits ? "warning" : "");
    unsafe.append(element("small", "Unsafe commits"));
    const coverage = element("div", m.scored + " / " + m.attempted);
    coverage.append(element("small", "Scored trials"));
    row.append(label, pass, col, unsafe, coverage);
    return row;
  });
  byId("benchmark").replaceChildren(...rows);
}

const tabs = [...document.querySelectorAll('[role="tab"]')];
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => setMode(tab.id.replace("tab-", "")));
  tab.addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
    e.preventDefault();
    const next = e.key === "Home" ? 0 : e.key === "End" ? tabs.length-1 :
      (index + (e.key === "ArrowRight" ? 1 : tabs.length-1)) % tabs.length;
    tabs[next].click();
    tabs[next].focus();
  });
});
byId("replay-select").addEventListener("change", (e) => openClip(e.target.value));
byId("cohort").addEventListener("change", showCohort);
byId("seek").addEventListener("input", (e) => {
  stop(); setPosition(e.target.value);
  const video = byId("hero-video");
  if (mode === "replay" && trace?.videoClip && Number.isFinite(video.duration)) {
    video.currentTime = Math.min(trace.frames[position][0], video.duration);
  }
});
byId("hero-video").addEventListener("timeupdate", (e) => {
  if (mode === "replay" && trace?.videoClip && !playing) setPosition(frameAt(e.target.currentTime));
});
byId("next-event").addEventListener("click", () => {
  if (!trace) return;
  stop();
  const event = trace.events.find((e) => e.t > trace.frames[position][0] + .11) || trace.events[0];
  setPosition(event ? frameAt(event.t + .11) : trace.frames.length - 1);
});
byId("play").addEventListener("click", () => {
  if (!trace) return;
  if (playing) { stop(); return; }
  byId("hero-video").pause();
  if (position === trace.frames.length - 1) setPosition(0);
  playing = true;
  playStart = performance.now();
  simStart = trace.frames[position][0];
  byId("play").textContent = "Pause";
  requestAnimationFrame(tick);
});
byId("panel-sim").addEventListener("submit", (e) => {
  e.preventDefault();
  byId("run-scenario").disabled = true;
  requestModel("run", { scenario: byId("scenario").value, policy: byId("policy").value,
    urgency: byId("urgency").value || null, profile: byId("profile").value });
});
byId("panel-trip").addEventListener("submit", (e) => {
  e.preventDefault();
  byId("run-trip").disabled = true;
  requestModel("trip", { text: byId("trip-input").value });
});
new ResizeObserver(paint).observe(canvas);
setMode("replay");
initialize().catch((error) => setActivity(error.message));
