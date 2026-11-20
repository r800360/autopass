/* A new simulation runs inside this worker using the same Python package as the CLI. */
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js");

let vm;
async function runtime() {
  if (vm) return vm;
  vm = (async () => {
    const py = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/" });
    const response = await fetch("data/runtime.zip");
    if (!response.ok) throw new Error("Runtime bundle could not be loaded.");
    py.unpackArchive(await response.arrayBuffer(), "zip", { extractDir: "/project" });
    py.runPython("import sys\nsys.path.insert(0, '/project')\nfrom autopass.cli import browser_run\nfrom autopass.trip import parse_request, navigate\nimport json");
    return py;
  })();
  try {
    return await vm;
  } catch (error) {
    vm = undefined;
    throw error;
  }
}

self.onmessage = async ({ data }) => {
  try {
    const py = await runtime();
    py.globals.set("request", JSON.stringify(data.payload));
    let output;
    if (data.kind === "trip") {
      output = py.runPython(
        "trip_request = parse_request(json.loads(request)['text'])\n" +
        "json.dumps([navigate(trip_request), navigate(trip_request, replan=False)], allow_nan=False)"
      );
    } else {
      output = py.runPython("browser_run(request)");
    }
    self.postMessage({ id: data.id, result: JSON.parse(output) });
  } catch (error) {
    const description = String(error.message || error).trim().split("\n").at(-1);
    self.postMessage({ id: data.id, error: description });
  }
};
