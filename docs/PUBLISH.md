# Release and update the showcase

The browser lab is a static GitHub Pages deployment. It loads its Python core on demand in a worker on the visitor's machine. Replays use small H.264 clips and JSON. The showcase requires no simulator server, API key, or paid hosting.

## Publish code and layout

Push changes to main, verify CI, then dispatch the Pages workflow. It publishes the contents of `site/`. A layout or browser-runtime release requires the workflow once.

## Add evidence

1. Produce new run artifacts and regenerate the bundle with a later `--edition`.
2. Verify metrics, scrub unintended metadata, and upload the regenerated `data/` and `media/` directories to the evidence branch at its root.
3. Reload the showcase and check the new evidence. The site accepts a newer evidence edition and retains its tagged snapshot as a fallback.

The Pages workflow runs on demand. Experiment evidence and maintenance commits leave the deployed shell in place. Use dated releases to keep downloadable milestones stable. Raw private captures belong in a local archive.

Before publishing a new bundle, use `python -m http.server 8313 --bind 127.0.0.1 --directory site` and verify the video, timeline, a live browser simulation, a trip, and a mobile viewport.
