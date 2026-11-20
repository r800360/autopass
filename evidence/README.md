# Evidence index

The [release report](../site/report.pdf) summarizes the rebuild and experimental limits. The [raw bundle](../site/data) holds individual kinematic and CARLA run rows, selected telemetry, uncertainty intervals, and the browser runtime.

- `kinematic-catalog.json.gz` — four policies on 34 authored cases
- `kinematic-sampled.json.gz` — four policies on 500 seeded traffic draws each
- `kinematic-degraded.json.gz` — same traffic with synthetic sensing noise and misses
- `kinematic-imitation.json.gz` — bounded learned steering on the catalog
- `carla-*.json.gz` — engineering sweep, final verification cases, and experimental profiles
- `trace-*.json` — replay files with tool evidence and maneuver events
- `model-*.json` — local inference traces
- `trip-*.json` — five-node route runs and route decisions

Raw evidence may be inspected with Python's standard gzip and json modules. [Evaluation and commands](../docs/EVALUATION.md).
