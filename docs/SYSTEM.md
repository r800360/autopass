# System design

## Research question

Can an evidence-gathering policy improve progress around slow leaders while a deterministic critic limits maneuver risk? How much of the result comes from the planner, the sensor model, or the physical checks?

~~~mermaid
flowchart LR
    Request["Trip request and urgency"] --> Navigator["Navigator and route costs"]
    Navigator --> Planner["Rule or model planner"]
    Sensors["Instance + depth, or RGB + depth"] --> Tools["Gap, blind-spot and corridor tools"]
    Planner --> Tools
    Tools --> Belief["Belief with age, rate and history"]
    Belief --> Planner
    Belief --> Critic["Deterministic critic"]
    Critic -->|denial| Belief
    Critic -->|approval| FSM["Maneuver state machine"]
    FSM --> Control["Waypoint or learned steering"]
    Control --> World["Simulator"]
    World --> Sensors
    World -. "evaluation only" .-> Trace["Metrics and replay"]
    FSM --> Navigator
~~~

## Perception

1. CARLA supplies synchronized image pairs. The main profile separates vehicle pixels by instance ID, decodes metric depth, and undoes BGRA channel order for display. A missing frame pair stops the trial with a recorded error.
2. For pixel column `u`, focal length `f`, and depth `Z`, camera lateral position is `(u - cx) Z / f`. Camera mounting transforms and ego localization project pixels into longitudinal `s` and lateral `d` coordinates on the road.
3. Grouping visible surfaces yields one object and gap per instance per view. Region queries look ahead, behind, and alongside, with direction-aware lane association. Two readings of a continuing track support a gap-rate estimate. A track change clears that estimate.
4. Camera geometry and visible road pixels bound sensing reach. The learned profile runs a pretrained NanoDet ONNX detector on each requested RGB view and estimates object position from depth inside its boxes. It approximates visible ground from depth and camera height. The kinematic backend projects boxes into geometric camera views with occlusion and configurable noise.

## Planner and critic

The mutable belief stores the latest observations, their simulation timestamps, probe order, and rejection history. A planner chooses from the tool palette, proposes a pass, or holds. The rule planner uses freshness and denial history. The model planner emits a schema-validated sequence or one tool at a time. Model timeouts hold, and requests have a per-run budget.

The critic checks front evidence within 0.7 s and passing-lane, rear, and blind-spot evidence within 0.3 s. It requires the blind-spot check after rear and passing-lane queries even when they share a tick. It ages closing gaps conservatively and adds a margin to measured rear distance and speed. The corridor is rechecked within 2 s.

- **Warrant.** The allowed lead slowdown depends on urgency. Sufficient pull-out room is a separate gate.
- **Rear.** At lane entry, leave a minimum gap and time headway. Bound the follower braking required after a modeled reaction delay.
- **Oncoming.** Available sight must cover ego travel, approaching-vehicle travel, and a closing-speed buffer. The pass model accelerates to a target, gains enough longitudinal distance to clear the lead, then returns to lane.
- **Road and shoulder.** Require a driving-lane corridor and an observed clear blind spot. The corridor check uses CARLA's map markings and junctions.

## Controller and trip scope

~~~mermaid
stateDiagram-v2
    [*] --> Follow
    Follow --> LaneChange: critic approval
    LaneChange --> Overtake: reaches adjacent lane
    LaneChange --> Abort: hazard and visible return gap
    Overtake --> Abort: hazard and visible return gap
    Overtake --> MergeBack: clearance and travel-lane gap
    MergeBack --> Done: reaches travel lane
    Abort --> Follow: returns behind lead
~~~

Waypoint steering and adaptive cruise run every 50 ms. The monitor can abandon an overtake while a visible return gap remains. A learned steering variant predicts a bounded command from geometric features and falls back outside its training envelope. Both use the same critic, speed controller, and phase logic.

The trip demo extracts an intent over five simulated waypoints and runs the actual passing policy on each executed road segment. After observing delay and repeated denials, it recomputes shortest paths with updated travel times. Its environment stages one segment at a time. Turns, intersections, and continuous navigation between CARLA towns are outside this experiment.

[Evaluation protocol](EVALUATION.md) · [CARLA setup](CARLA.md)
