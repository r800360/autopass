"""Small OpenAI-compatible structured output client. Local endpoints are the default."""
from __future__ import annotations

import json
import os
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from ..core.planner import DirectLLM, LLMPlanner


def complete_json(system: str, user: str, schema: dict) -> tuple[dict, dict]:
    base = os.environ.get("AUTOPASS_LLM_URL", "http://127.0.0.1:8080/v1").rstrip("/")
    model = os.environ.get("AUTOPASS_LLM_MODEL", "autopass-local")
    parsed = urlsplit(base)
    local = parsed.hostname in ("127.0.0.1", "localhost", "::1")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
        raise ValueError("Use HTTPS or a loopback inference endpoint")
    headers = {"Content-Type": "application/json"}
    if not local:
        key = os.environ.get("AUTOPASS_LLM_KEY")
        if not key:
            raise ValueError("AUTOPASS_LLM_KEY is required for a remote endpoint")
        headers["Authorization"] = "Bearer " + key
    payload = {
        "model": model, "temperature": 0, "max_tokens": 220,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "autopass_action", "strict": True, "schema": schema,
        }},
    }
    start = time.perf_counter()
    req = Request(base + "/chat/completions", data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urlopen(req, timeout=25) as response:
        reply = json.loads(response.read(1024 * 1024))
    latency = time.perf_counter() - start
    output = json.loads(reply["choices"][0]["message"]["content"])
    tokens = int(reply.get("usage", {}).get("total_tokens", 0))
    if not isinstance(output, dict):
        raise ValueError("structured output must be an object")
    return output, {"latency_s": latency, "tokens": tokens, "model": model}


def make_planner(kind: str) -> LLMPlanner | DirectLLM:
    if kind == "llm_direct":
        return DirectLLM(complete_json)
    return LLMPlanner(complete_json, {"llm_plan": "plan", "llm_react": "react"}[kind])
