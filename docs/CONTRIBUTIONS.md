# Contributions and provenance

The research team was **Rohan Sachdeva, Xinwei Mai, Pranav Prabu, and Rathang Pandit**. Attribution follows the original report and source history.

- **Rohan Sachdeva** built the live CARLA integration, planner–critic–tool loop, belief state, waypoint control, benchmark and metrics. He implemented segmentation and metric-depth gap extraction, lane association, and evidence overlays.
- **Xinwei Mai** framed the research question and initial hypothesis, built the original passing-agent framework, and co-designed the revised architecture. The original implementation appears in [commit b54a909](https://github.com/r800360/autopass-gen/commit/b54a909).
- **Pranav Prabu** led presentation and communication, debugged LLM tool priority, and contributed to the planner-loop design and explanation.
- **Rathang Pandit** reviewed videos frame by frame, surfaced spawning and lane-keeping problems, and developed real-world motivation and context.

This portfolio release builds on [the original public source](https://github.com/r800360/autopass-gen/tree/8b424e2bbbe64391be2a9c74351f6a505e0f943d). Its consolidation, tests, expanded experiments, and documentation were developed with AI assistance. Project-wide credit remains with the original research team. The new repository includes the original MIT notice.

## External resources

- Simulator, scenes, and rendered sensor images. [CARLA 0.9.16](https://github.com/carla-simulator/carla/tree/0.9.16)
- Optional detector weights, Apache 2.0. [OpenCV Zoo NanoDet](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_nanodet)
- Optional language model and runtime. [Qwen 2.5 1.5B Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) · [llama.cpp](https://github.com/ggml-org/llama.cpp)
- Browser runtime. [Pyodide](https://github.com/pyodide/pyodide/tree/0.27.7)

Model assets are fetched separately under their own terms.
