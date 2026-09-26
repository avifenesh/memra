#!/usr/bin/env python3
"""CPU stand-in for memra-server / kv-handoff-gate in m1-handoff-driver tests. Never a measurement.

Serves /v1/models, /v1/completions and /metrics (JSON) on MEMRA_ADDR and prints the exact log
formats of the real handoff path. Kind from M1_STUB_KIND (gate or server), set by the wrapper.
Host bytes grow by M1_STUB_ENTRY_BYTES per new prompt when MEMRA_KV_HOST_MB > 0. Red controls:
M1_STUB_REFUSE_EXPORT=1, M1_STUB_IMPORT_SKIPS=<n>, M1_STUB_MISS_PROBE=1, M1_STUB_TEXT_DRIFT=1.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

ENV = os.environ
KIND = ENV.get("M1_STUB_KIND", "server")
HOST_MB = int(ENV.get("MEMRA_KV_HOST_MB", "0"))
HANDOFF = ENV.get("MEMRA_KV_HOST_HANDOFF")
ENTRY = int(ENV.get("M1_STUB_ENTRY_BYTES", str(128 << 20)))
STATE = {"host_bytes": 0, "prompts": [], "restored": set()}


def log(line):
    print(line, flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            return self.reply({"data": [{"id": "gate"}]})
        if self.path == "/metrics":
            return self.reply({"prefix_host_bytes": STATE["host_bytes"]})
        self.send_error(404)

    def stream(self, body):
        n = int(body.get("max_tokens", 16))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(n):
            chunk = {"object": "text_completion", "choices": [{"index": 0, "text": f" t{i}", "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.002)
        usage = {"object": "text_completion", "choices": [], "usage": {"prompt_tokens": 70, "completion_tokens": n}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("stream"):
            return self.stream(body)
        prompt = body["prompt"]
        restored = any(prompt.startswith(p) for p in STATE["restored"])
        text = "text-" + hashlib.sha256(prompt.encode()).hexdigest()[:12]
        if restored and ENV.get("M1_STUB_TEXT_DRIFT") == "1":
            text += "-drift"
        cached = 1000 if restored and ENV.get("M1_STUB_MISS_PROBE") != "1" else 0
        if HOST_MB > 0 and prompt not in STATE["prompts"]:
            STATE["prompts"].append(prompt)
            STATE["host_bytes"] += ENTRY
        self.reply({"choices": [{"text": text}],
                    "usage": {"prompt_tokens": 7000, "prompt_tokens_details": {"cached_tokens": cached}}})


def export(*_):
    if ENV.get("M1_STUB_REFUSE_EXPORT") == "1":
        log("[handoff-gate] export refused: stub refusal")
        return
    started = time.monotonic()
    payload = json.dumps({"prompts": STATE["prompts"], "bytes": STATE["host_bytes"]}).encode()
    tmp = HANDOFF + ".tmp"
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, HANDOFF)
    ms = (time.monotonic() - started) * 1e3
    n, mb = len(STATE["prompts"]), STATE["host_bytes"] / 1e6
    log(f"[prefix-host] handoff export: {n} entries / {mb:.1f}MB to {HANDOFF} in {ms:.0f}ms "
        f"write_ms={ms * 0.7:.1f} fsync_ms={ms * 0.3:.1f} (drain-demoted 2 device entries first; "
        f"0 skipped over the MEMRA_KV_HOST_HANDOFF_MB cap)")
    log("[handoff-gate] export ok " + json.dumps({"entries": n, "bytes": STATE["host_bytes"]}))


def main():
    host, port = ENV["MEMRA_ADDR"].rsplit(":", 1)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    time.sleep(0.3)  # "model load"
    server = ThreadingHTTPServer((host, int(port)), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"[server] worker ready; serving models: ['gate'] ({KIND})")
    if HANDOFF and HOST_MB > 0 and Path(HANDOFF).exists():
        data = json.loads(Path(HANDOFF).read_text())
        n = len(data["prompts"])
        log(f"[prefix-host] handoff import armed at boot: {n} entries / {data['bytes'] / 1e6:.1f}MB from "
            f"{HANDOFF} (file age 1s); re-materializing one per tick")
        time.sleep(0.6)
        skips = int(ENV.get("M1_STUB_IMPORT_SKIPS", "0"))
        STATE["restored"] = set(data["prompts"][skips:])
        STATE["prompts"] = list(data["prompts"])
        STATE["host_bytes"] = data["bytes"]
        log(f"[prefix-host] handoff import DONE: {n - skips} entries / {data['bytes'] / 1e6:.1f}MB "
            f"re-materialized, {skips} skipped, in 0.6s from {HANDOFF}")
        os.unlink(HANDOFF)
    if KIND == "gate":
        signal.signal(signal.SIGUSR1, export)
        log("[handoff-gate] armed: SIGUSR1 exports the host tier")
    stop.wait()
    server.shutdown()
    if KIND == "gate":
        log("[handoff-gate] handles dropped on drain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
