#!/usr/bin/env python3
"""A CPU stand-in for memra-server, only to exercise the kit's bash and reader end to end (no model, no GPU).

It emulates what the DAY38D reader reads: per-namespace parked entries, cached_tokens on resumes, the park-compact
lines on the on/red arms, plain-affinity rewinds, MEMRA_STEP_OOM_FAULT's first-decode failure and the red arm's
extra park. Arm = the name of the bins/<arm>/ directory the stub was started from.
"""
import ctypes, hashlib, json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ctypes.CDLL(None).prctl(15, b"memra-server", 0, 0, 0)  # comm = memra-server, so pgrep -x finds it
arm = os.path.basename(os.path.dirname(os.path.abspath(sys.argv[1])))
fault = os.environ.get("MEMRA_STEP_OOM_FAULT") == "1"
host, port = os.environ["MEMRA_ADDR"].rsplit(":", 1)
lock = threading.Lock()
parked = {}  # namespace -> committed ids
hits = [0]
faults_left = [1 if fault else 0]


def log(s):
    sys.stderr.write(s + "\n")
    sys.stderr.flush()


log(f"[worker] Engine ready (device=0, MEMRA_FAST=true) stub arm={arm} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")


def text_for(key, n):
    h = hashlib.sha256(json.dumps(key).encode()).hexdigest()
    return ("w" + h[:6] + " ") * max(1, n // 4)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/readyz":
            return self.send(200, {"ok": True})
        if self.path == "/metrics":
            with lock:
                return self.send(200, {"cuda_driver_free_bytes": 6 << 30, "cuda_pool_cached_bytes": 8 << 30,
                                       "cuda_pool_reserved_bytes": 90 << 30, "continuation_pool_entries": len(parked),
                                       "continuation_pool_hits": hits[0], "active_sessions": 0})
        self.send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/v1/tokenize":
            return self.send(200, {"tokens": [ord(c) % 4096 for c in body.get("prompt", "")]})
        if self.path == "/v1/detokenize":
            return self.send(200, {"text": "".join(chr(32 + t % 90) for t in body.get("tokens", []))})
        salt = body.get("cache_salt", "")
        mt = int(body.get("max_tokens", 1))
        if self.path == "/v1/completions":
            ids = body["prompt_ids"]
            key = ids
        else:
            ids = [ord(c) % 4096 for c in json.dumps(body["messages"])]
            key = body["messages"]
        with lock:
            prev = parked.get(salt)
            cached = 0
            if prev and len(prev) < len(ids) and ids[:len(prev)] == prev:
                cached = len(prev)
            sid = self.headers.get("x-session-id")
            if self.path != "/v1/completions" and sid and prev:
                cached = len(prev) - 16
                log(f"[worker] plain-affinity: rewound to {cached} of {len(ids)} prompt tokens (stub)")
            if cached:
                hits[0] += 1
                if arm != "off":
                    log(f"[kv-reuse] park-compact grow: {len(prev)} -> {len(ids) + 512} rows (stub)")
            if faults_left[0] > 0:
                faults_left[0] -= 1
                log("[admit-oom] MEMRA_STEP_OOM_FAULT fired: this batched decode chunk reports a synthetic CUDA OOM (1 session(s))")
                if arm == "red":
                    parked[salt] = ids
                    log(f"[kv-reuse] park-compact: {len(ids)} of 262144 rows retained in 1.2ms (model q38)")
                return self.send(500, {"error": {"message": "CUDA_ERROR_OUT_OF_MEMORY (synthetic: MEMRA_STEP_OOM_FAULT diagnostic door)"}})
            text = text_for(key, mt)
            parked[salt] = ids + [ord(c) % 4096 for c in text]
            if arm != "off":
                log(f"[kv-reuse] park-compact: {len(parked[salt])} of 262144 rows retained in 1.2ms (model q38)")
        usage = {"prompt_tokens": len(ids), "completion_tokens": mt, "prompt_tokens_details": {"cached_tokens": cached}}
        ch = {"text": text, "finish_reason": "length"} if self.path == "/v1/completions" else \
            {"message": {"role": "assistant", "content": text}, "finish_reason": "length"}
        self.send(200, {"choices": [ch], "usage": usage})


ThreadingHTTPServer((host, int(port)), H).serve_forever()
