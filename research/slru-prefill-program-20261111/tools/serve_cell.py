#!/usr/bin/env python3
"""One served cell, run INSIDE a gpulock lease: boot memra-server, run the requests, stop it.

usage: serve_cell.py <server binary> <out dir> <mode> [ENV=V ...]
  mode = pair    one /v1/completions board-2048 request, temperature 0, 128 tokens
         itl     session A streams 384 tokens from a short prompt; when A has 48 tokens, session B
                 sends the board-2048 request; A's inter-token latency is reported for the
                 window B is in flight, and for the same-length window before B was sent

Writes server.log, response*.json, memory.csv (nvidia-smi memory.used every 250 ms), result.json.
"""
import http.client
import json
import os
import pathlib
import signal
import statistics
import subprocess
import sys
import threading
import time

server, out, mode = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
extra = dict(kv.split("=", 1) for kv in sys.argv[4:])
out.mkdir(parents=True, exist_ok=True)
REPO = pathlib.Path.home() / ".cache/memra-1217/wt"
MODEL = "/data/ai-ml/hf-models/ornith15-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf"
PORT = 18217
prompt = (REPO / "research/e2e/prompts/board-2048.txt").read_text()

env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
env.update({"MEMRA_MODELS": f"ornith={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}"})
env.update(extra)
# MEMRA_GPU_LOCK is inherited from the lease on purpose (the server must not flock it itself).
if "MEMRA_GPU_LOCK" in os.environ:
    env["MEMRA_GPU_LOCK"] = os.environ["MEMRA_GPU_LOCK"]

mem_stop = threading.Event()


def sample_memory():
    with open(out / "memory.csv", "w") as f:
        f.write("t_s,used_mib\n")
        t0 = time.time()
        while not mem_stop.is_set():
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True)
            f.write(f"{time.time() - t0:.3f},{r.stdout.strip()}\n")
            f.flush()
            mem_stop.wait(0.25)


def post(body, stream=False, timeout=600):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    c.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
    return c, c.getresponse()


log = open(out / "server.log", "w")
boot0 = time.time()
proc = subprocess.Popen([server], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
                        start_new_session=True)
mem = threading.Thread(target=sample_memory, daemon=True)
mem.start()
result = {"mode": mode, "env": extra, "server": server}
try:
    deadline = time.time() + 600
    while True:
        if proc.poll() is not None:
            raise SystemExit(f"server exited rc={proc.returncode} during boot")
        try:
            c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=5)
            c.request("GET", "/v1/models")
            if c.getresponse().status == 200:
                break
        except OSError:
            pass
        if time.time() > deadline:
            raise SystemExit("boot timeout")
        time.sleep(1)
    result["boot_s"] = round(time.time() - boot0, 1)
    result["mem_ready_s"] = round(time.time() - boot0, 3)

    if mode == "pair":
        t0 = time.time()
        _, r = post({"model": "ornith", "prompt": prompt, "max_tokens": 128, "temperature": 0})
        body = json.loads(r.read())
        result["client_s"] = round(time.time() - t0, 3)
        (out / "response.json").write_text(json.dumps(body, indent=1))
        import hashlib
        text = body["text"] if "text" in body else body["choices"][0]["text"]
        result["server_elapsed_s"] = body.get("elapsed_s")
        result["tokens_sha16"] = hashlib.sha256(json.dumps(body.get("tokens")).encode()).hexdigest()[:16]
        result["text_sha16"] = hashlib.sha256(text.encode()).hexdigest()[:16]
        result["usage"] = body.get("usage") or {k: body.get(k) for k in ("n_tokens", "prompt_tokens", "cached_tokens", "stop_reason")}
    elif mode == "itl":
        arrivals = []
        b_times = {}

        def session_b():
            b_times["sent"] = time.time()
            _, r = post({"model": "ornith", "prompt": prompt, "max_tokens": 16, "temperature": 0})
            body = json.loads(r.read())
            b_times["done"] = time.time()
            (out / "response_b.json").write_text(json.dumps(body, indent=1))

        tb = None
        c, r = post({"model": "ornith", "prompt": "Write a long story about a lighthouse keeper.",
                     "max_tokens": 384, "temperature": 0, "stream": True})
        buf = b""
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(1)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                if ev.startswith(b"data: ") and ev[6:].strip() != b"[DONE]":
                    arrivals.append(time.time())
                    if len(arrivals) == 48 and tb is None:
                        tb = threading.Thread(target=session_b)
                        tb.start()
        if tb:
            tb.join()
        gaps = [(arrivals[i], arrivals[i] - arrivals[i - 1]) for i in range(1, len(arrivals))]
        s, d = b_times["sent"], b_times["done"]
        during = [g * 1000 for t, g in gaps if s < t <= d + 0.0]
        span = d - s
        before = [g * 1000 for t, g in gaps if s - span < t <= s]

        def pct(v, q):
            if not v:
                return None
            v = sorted(v)
            return round(v[min(len(v) - 1, int(round(q * (len(v) - 1))))], 2)

        result.update({
            "a_events": len(arrivals),
            "b_s": round(span, 3),
            "itl_during_b": {"n": len(during), "p50_ms": pct(during, 0.5),
                             "p99_ms": pct(during, 0.99), "max_ms": round(max(during), 2) if during else None},
            "itl_before_b": {"n": len(before), "p50_ms": pct(before, 0.5),
                             "p99_ms": pct(before, 0.99)},
        })
        (out / "arrivals.json").write_text(json.dumps({"a": arrivals, "b": b_times}))
finally:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except Exception:
        os.killpg(proc.pid, signal.SIGKILL)
    mem_stop.set()
    mem.join(timeout=5)
    rows = [l.split(",") for l in (out / "memory.csv").read_text().splitlines()[1:] if "," in l]
    used = [int(u) for _, u in rows if u.strip().isdigit()]
    result["peak_mem_mib"] = max(used) if used else None
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
