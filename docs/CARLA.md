# Run with CARLA 0.9.16

1. Download the [official 0.9.16 release](https://github.com/carla-simulator/carla/releases/tag/0.9.16). Keep the Python client at the same version as the server.
2. Install with Python 3.12 and start a dedicated CARLA server in the same trusted environment. The RGB and depth cameras need rendering. Use `-RenderOffScreen` for headless runs.
3. Run one town per server process. Set `CARLA_HOST` and `CARLA_PORT` for an explicit endpoint. The default is `127.0.0.1:2000`.

~~~bash
python -m pip install -e '.[carla]'
python -m autopass.carla_backend.run --load-town --town Town04 --scenarios s01,s17,s24 --video full --out runs/carla
~~~

On Windows, the DirectX June 2010 runtime may be needed by the UE4 distribution. The validation used a Windows CARLA server and a Python 3.12 WSL client using standard CARLA RPC and camera streaming. For the bundled launcher, create a WSL venv at `~/.cache/autopass-linux` and install the project with `.[carla]` there.

~~~powershell
.\scripts\wsl-campaign.ps1 -Towns Town04 -Scenarios 's01,s17,s24' -Video full -Out runs/carla
~~~

Use `-CarlaRoot` and `-Distro` for your setup. The launcher starts and cleans up its own process. A dedicated simulator is required because a trial resets world settings and can load a map.

For the learned sensor profile, fetch the pinned detector weights, set `AUTOPASS_NANODET_MODEL` to the resulting path in your client environment, then pass `--perception rgbd`. With the WSL launcher pass `-Model` with the Windows path. Set `--steering imitation` to use learned lateral commands.

~~~bash
python scripts/fetch_perception_model.py
python -m autopass.carla_backend.run --town Town04 --scenarios s01 --perception rgbd --video full --out runs/carla-ablation
~~~

The default 2.5-second observation window filters out rolling-start drivetrain transients. The critic evaluates a fresh proposal at the commit tick. Pass --min-follow 0 to reproduce earlier immediate-start engineering runs.

## Replay files

Each run writes `result.json` and `telemetry.json.gz`. With `--video full` it also streams frames directly into a browser-compatible H.264 MP4 and samples a small contact sheet. Use `--video none` for a low-storage batch. Version per-run configuration and results together. Keep raw captures when needed to investigate perception.

For useful failure evidence, preserve the exact simulator launch, map, seed, policy, sensor profile, completed score, and any synchronization or spawn failure.
