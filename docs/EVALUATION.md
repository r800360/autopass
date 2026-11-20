# Evaluation and reproduction

Identify the backend, policy, sensor profile, and denominator when quoting a result.

- **Kinematic catalog.** 23 scenarios adapted from the original release and 11 authored stress cases. Compare the rule-based AutoPass policy, never-pass, always-pass, and a fixed-threshold approximation of the earlier logic.
- **Sampled traffic.** 500 synthetic scenarios from seed 20261119 per policy. Randomized lead speeds, overtakers, blockers, oncoming traffic, and delays test combinations absent from the authored catalog. A degraded-sensing profile uses a deliberately pessimistic noise and miss distribution. Rates describe these simulated inputs.
- **CARLA.** Version 0.9.16 with actual camera callbacks and simulation steering. The engineering sweep captures initial integration outcomes. Corrected-backend spot checks and experimental profiles have their own cohorts. The launcher starts a fresh simulator process per town.

## Run the campaigns

~~~bash
autopass bench --workers 4 --out runs/catalog.json
autopass bench --count 500 --seed 20261119 --workers 4 --out runs/heldout.json
autopass bench --count 500 --seed 20261119 --workers 4 --profile degraded --policies autopass --out runs/degraded.json
autopass bench --steering imitation --policies autopass --out runs/imitation.json
python scripts/prepare_evidence.py
~~~

The generator exports raw per-episode metrics as compressed JSON, telemetry for selected runs, and browser assets. A per-episode trace contains frames, tool calls, verdicts, observed belief, and event times. Download raw data from the website's evaluation panel or browse [the versioned bundle](../site/data).

## Read the metrics

- **Completed overtake** means clear the lead and finish returning to the travel lane without collision or corridor departure.
- **Expected behavior** uses scenario labels. A yield case requires a measured hazard hold or abort before a successful completion. A wait case requires staying in lane.
- **Unsafe and unwarranted commits** are measured by evaluating the physics gates with simulator state at commit time. They are model-based labels. Collision events are independent outcomes.
- **Progress** uses elapsed simulation time to travel the first 80 m. Paired median savings uses only trials where both policies reached 80 m without collision or departure. The paired count matters because some routes time out.
- **Lane keeping** scores settled follow/done phases and separately tracks overshoot during a maneuver. Corridor departure measures the staged lateral corridor. Ground-truth actor states support scoring and replay.
- **Collision intervals** in data are Wilson 95% intervals. They describe uncertainty over repeated draws under an assumed distribution. Generalization to real traffic requires a separate design.

## Ablations

Rebuild the learned steering weights with `python scripts/train_steering.py` using `.[fit]`. The labels come from the waypoint controller, with separate random seeds for training and validation. Test the closed-loop effect with the same scenarios. The OpenCV NanoDet RGB-depth profile is pretrained on COCO and uses CARLA depth to recover metric position. The degraded kinematic profile is a distinct synthetic experiment.

For the optional model planner, run a local OpenAI-compatible server and set its alias through `AUTOPASS_LLM_MODEL`. Trials use Qwen 2.5 1.5B Instruct Q4_K_M served by llama.cpp.

~~~bash
autopass run s01 --planner llm_plan --model-calls 8 --trace runs/llm/s01-plan.json.gz
autopass run s01 --planner llm_react --model-calls 12 --trace runs/llm/s01-react.json.gz
autopass run s01 --planner llm_direct --model-calls 6 --trace runs/llm/s01-direct.json.gz
~~~

The direct model is a kinematic-only ablation with approval disabled. To test timing sensitivity, add `--realtime`. That mode advances simulation by measured model and estimated sensing latency.

## Evidence limits

An earlier summary attributed all pass decisions to the live model. Archived traces for two hazard scenes show passes attributed to a greedy backstop. One oncoming case also contains early approvals before the oncoming actor appeared in the belief. The current traces record proposals, critic decisions, and observed execution separately.

Sensor detections and lane coverage remain imperfect. Static geometry and staged traffic simplify the task. A collision-free cohort can coexist with failures under randomized scenes or degraded sensing. Keep these cohorts distinct.
