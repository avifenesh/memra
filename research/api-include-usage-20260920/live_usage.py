#!/usr/bin/env python3
"""Live OpenAI SSE usage gate; run only against an isolated, spec-off test server.

Usage: live_usage.py BASE_URL MODEL OUT_DIR
Exact full-object parity is CPU-tested with identical terminal events. Live runs
compare accounting fields separately from elapsed_s (a per-request measurement).
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

base, model, out = sys.argv[1:]
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
summary = []


def post(route, body, name, stream=False):
    raw = subprocess.check_output([
        "curl", "--fail-with-body", "--silent", "--show-error", "--max-time", "180",
        "-N", "-H", "Content-Type: application/json", "--data-binary", json.dumps(body),
        base.rstrip("/") + route,
    ])
    (out / name).write_bytes(raw)
    if not stream:
        return json.loads(raw)
    lines = [line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]", lines
    assert lines.count("[DONE]") == 1
    return [json.loads(line) for line in lines[:-1]]


def accounting(usage):
    return {key: value for key, value in usage.items() if key != "elapsed_s"}


for chat, regime in [(chat, regime) for chat in [True, False] for regime in ["short", "cached"]]:
    dialect = "chat" if chat else "completions"
    label = f"{dialect}-{regime}"
    route = "/v1/chat/completions" if chat else "/v1/completions"
    body = {"model": model, "max_tokens": 8, "temperature": 0, "seed": 42,
            "cache_salt": "include-usage-" + uuid.uuid4().hex}
    prompt = "Count upwards starting from one, using words separated by commas."
    if regime == "cached":
        prompt = ("This is shared context for a deterministic streaming usage check. " * 80) + prompt
    body.update({"messages": [{"role": "user", "content": prompt}]} if chat else {"prompt": prompt})
    # Warm the identical prompt before every scored shape. No concurrent requests.
    for i in range(2):
        post(route, body, f"{label}-warm-{i}.json")
    twin = post(route, body, f"{label}-nonstream.json")
    cached = twin["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cached > 0 if regime == "cached" else cached == 0, (label, twin["usage"])
    usages = {"nonstream": twin["usage"]}
    for mode in ["true", "absent", "false"]:
        request = dict(body, stream=True)
        if mode != "absent":
            request["stream_options"] = {"include_usage": mode == "true"}
        chunks = post(route, request, f"{label}-{mode}.sse", stream=True)
        assert all("error" not in chunk for chunk in chunks), chunks
        if mode == "true":
            assert chunks[-1]["choices"] == [], chunks[-1]
            assert all(chunk.get("choices") for chunk in chunks[:-1])
            assert all("usage" in chunk and chunk["usage"] is None for chunk in chunks[:-1])
            assert chunks[-2]["choices"][0]["finish_reason"] is not None
        else:
            assert all(chunk.get("choices") for chunk in chunks)
            assert all("usage" not in chunk for chunk in chunks[:-1])
            assert chunks[-1]["choices"][0]["finish_reason"] is not None
        usage = chunks[-1]["usage"]
        assert accounting(usage) == accounting(twin["usage"]), (dialect, mode, usage, twin["usage"])
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        usages[mode] = usage
    summary.append({"dialect": dialect, "cache_regime": regime, "result": "PASS", "usage": usages,
                    "comparison": "all usage fields except per-request elapsed_s"})

(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name != "sha256.json"}
(out / "sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print("PASS: both OpenAI dialects; include_usage true/absent/false; DONE last; cold/cached accounting parity")
