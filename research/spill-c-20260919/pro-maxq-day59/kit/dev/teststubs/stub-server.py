#!/usr/bin/env python3
"""Laptop plumbing-test stub for memra-server-c60: /v1/models and /v1/completions on MEMRA_ADDR, a deterministic
greedy-looking text per prompt (identical in both arms). Records the card env it saw. Never touches a GPU."""
import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

env = {k: os.environ.get(k) for k in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "MEMRA_MOE_PREFETCH",
                                      "MEMRA_MOE_SLOTS", "MEMRA_CTX", "MEMRA_MAX_SESSIONS", "MEMRA_MODELS")}
with open(os.environ["MAXQ_STUB_LOG"], "a") as f:
    f.write(f"memra-server-c60 env={env}\n")


class H(BaseHTTPRequestHandler):
    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"data": [{"id": "gate"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send({"choices": [{"text": "stub " + hashlib.sha256(req["prompt"].encode()).hexdigest()[:16]}]})

    def log_message(self, *args):
        pass


host, port = os.environ["MEMRA_ADDR"].rsplit(":", 1)
print(f"stub memra-server listening on {host}:{port}", flush=True)
ThreadingHTTPServer((host, int(port)), H).serve_forever()
