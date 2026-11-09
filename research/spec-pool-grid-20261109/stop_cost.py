#!/usr/bin/env python3
"""Does a checkpoint stop cost TTFT, at the same prompt length? (memra#1093 item 1 cost note, item 5)

The pool-grid cost cells showed primes of 53+ rows at about 0.22 s and primes of 37 rows or fewer
at about 0.12 s, and every 53+ row prime there also stopped at a checkpoint. This separates the
two: cold chat prompts of 70 to 79 tokens with a `user` id (too short for a prefix-cache seed, so
the seed adds no stop) at the device default, where the prime stops at the turn checkpoint, against
MEMRA_AFFINITY=0, where no checkpoint stop is armed. Each prompt is distinct (no reuse). Boots
alternate the arms. MTP route, greedy, thinking off, max_tokens 8, streamed (TTFT = first chunk).

usage: stop_cost.py OUT BINARY [BOOTS]
Writes OUT/server-<n>-<arm>.log and OUT/runs.jsonl; prints medians per arm.
"""
import json
import os
import pathlib
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
BIN = sys.argv[2]
BOOTS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
PORT = int(os.environ.get("CELL_PORT", "18193"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
COMMON = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
          "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2"}
ARMS = {"stop": {}, "nostop": {"MEMRA_AFFINITY": "0"}}
TOPICS = ["rivers", "bridges", "volcanoes", "glaciers", "deserts", "forests", "comets", "tides",
          "harbors", "canyons", "islands", "meadows"]


def boot(log, env):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} busy")
    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    e.update(env)
    p = subprocess.Popen([BIN], env=e, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            raise SystemExit(f"server exited {p.returncode}; see {log}")
        try:
            urllib.request.urlopen(URL + "/v1/models", timeout=3).read()
            return p
        except Exception:
            time.sleep(1)
    p.kill()
    raise SystemExit("server not ready in 900 s")


def stop(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def call(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())


def messages(topic, k):
    # The topic leads, so no two prompts share a prefix-cache entry (a cold spec session publishes
    # its checkpoint boundary, 32 tokens here).
    return [{"role": "system", "content": f"On {topic}: you answer briefly." + " Thank you." * k},
            {"role": "user", "content": f"Tell me one fact about {topic}."}]


def sized(topic):
    """The padding that renders 70..79 tokens (no seed stop: the aligned cut 64 leaves under 16)."""
    for k in range(60):
        n = call("/v1/tokenize", {"model": "q", "messages": messages(topic, k), "add_generation_prompt": True,
                                  "chat_template_kwargs": {"enable_thinking": False}})["count"]
        if 70 <= n <= 79:
            return k, n
    raise SystemExit(f"no padding renders 70..79 tokens for {topic}")


def ttft(msgs, user):
    body = {"model": "q", "messages": msgs, "temperature": 0, "max_tokens": 8, "stream": True,
            "user": user, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if line.startswith("data:") and line != "data: [DONE]":
                for ch in json.loads(line[5:]).get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        return time.perf_counter() - t0
    return None


order = [("stop", "nostop", "nostop", "stop")[i % 4] for i in range(BOOTS * 2)]
rows = []
for n, arm in enumerate(order, 1):
    log = OUT / f"server-{n}-{arm}.log"
    p = boot(log, dict(COMMON, **ARMS[arm]))
    try:
        for i, topic in enumerate(TOPICS):
            k, toks = sized(topic)
            t = ttft(messages(topic, k), f"stop-cost-{n}-{i}")
            r = {"boot": n, "arm": arm, "topic": topic, "prompt_tokens": toks, "ttft_s": round(t or -1, 4)}
            rows.append(r)
            with open(OUT / "runs.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
    finally:
        stop(p)
for arm in ARMS:
    v = [r["ttft_s"] for r in rows if r["arm"] == arm]
    print(arm, "n", len(v), "median", round(statistics.median(v), 4), "min", min(v), "max", max(v))
