#!/usr/bin/env python3
"""What a per-session recurrent checkpoint buys and costs on the RTX 5090 (memra#1093 item 5).

Sessions with no checkpoint resume nothing from the pools under the grid law: a markerless prompt
with no session id is the default case. The same conversations sent with a `user` id arm the turn
checkpoint (the engine's stable boundary falls back to the raw guard window), so that arm prices
"a checkpoint for every recurrent session" with no code change. Arms, one binary, boots ABBA:

  anon   markerless /v1/completions, no `user`: no checkpoint; a recurrent spec session is freed at
         retire (memra#1093 item 1), and the next turn restores from the prefix cache's seed.
  named  the same requests with a `user` id per conversation: the checkpoint is armed, the session
         parks, and the next turn rewinds to the checkpoint.

Per boot, CONVS conversations of three turns (turn N+1 = turn N's prompt + its completion + a
follow-on), greedy, max_tokens 128, streamed (TTFT = first chunk). After each request /metrics
gives spec_pool_entries, cuda_pool_used_bytes, cuda_driver_free_bytes and prefix_cache_bytes; the
server log segment gives the path.

usage: price_cell.py OUT BINARY [ORDER]   (ORDER default anon,named,named,anon,anon,named,named,anon)
Writes OUT/server-<n>-<arm>.log, OUT/runs.jsonl and OUT/summary.json.
"""
import json
import os
import pathlib
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
BIN = sys.argv[2]
ORDER = (sys.argv[3] if len(sys.argv) > 3 else "anon,named,named,anon,anon,named,named,anon").split(",")
PORT = int(os.environ.get("CELL_PORT", "18193"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
COMMON = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
          "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2"}
# PRICE_ENV: space-separated VAR=VALUE added to every boot (the prefix-cache-off bound).
for kv in os.environ.get("PRICE_ENV", "").split():
    k, v = kv.split("=", 1)
    COMMON[k] = v
CONVS = int(os.environ.get("CONVS", "3"))
TOPICS = ["rivers", "bridges", "volcanoes", "glaciers", "harbors", "canyons"]
FOLLOW = ["\n\nA second point about the same notes:", "\n\nOne last remark:"]
LINES = {
    "park_skip": r"spec-park SKIPPED \(grid law\)",
    "grid_rewind": r"grid-rewind: spec extension of (\d+) committed rows rewound to (\d+)",
    "affinity": r"spec-affinity: rewound to (\d+) of (\d+)",
    "spec_restore": r"spec restore: (\d+) of (\d+) prompt tokens",
    "grid_decline": r"grid-rewind: spec declined|spec-affinity: declined \(grid law",
}
METRICS = ("spec_pool_entries", "cuda_pool_used_bytes", "cuda_driver_free_bytes", "prefix_cache_bytes")


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


def metrics():
    with urllib.request.urlopen(URL + "/metrics", timeout=30) as r:
        m = json.loads(r.read().decode())
    return {k: m.get(k) for k in METRICS}


def stream(body):
    req = urllib.request.Request(URL + "/v1/completions", data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, text = None, []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            for ch in json.loads(line[5:]).get("choices") or []:
                if ch.get("text"):
                    ttft = ttft if ttft is not None else time.perf_counter() - t0
                    text.append(ch["text"])
    return ttft, time.perf_counter() - t0, "".join(text)


def seed_text(topic):
    return (f"Field notes on {topic}. " + " ".join(
        f"Note {i}: the {topic} survey records one measurement per site, and a later visit "
        f"extends the same record with a new reading." for i in range(1, 31)) + "\n\nSummary:")


rows = []
for n, arm in enumerate(ORDER, 1):
    log = OUT / f"server-{n}-{arm}.log"
    p = boot(log, COMMON)
    try:
        for c in range(CONVS):
            topic = TOPICS[(n * CONVS + c) % len(TOPICS)]
            prompt = seed_text(topic) + f" (boot {n})"
            for t in range(1, 4):
                body = {"model": "q", "prompt": prompt, "temperature": 0, "max_tokens": 128}
                if arm == "named":
                    body["user"] = f"price-{n}-{c}"
                before = log.stat().st_size
                ttft, total, text = stream(body)
                time.sleep(0.5)
                with open(log, errors="replace") as f:
                    f.seek(before)
                    seg = f.read()
                r = {"boot": n, "arm": arm, "conv": c, "turn": t, "ttft_s": round(ttft or -1, 4),
                     "total_s": round(total, 4), "metrics": metrics(),
                     "lines": {k: [m.groups() for m in re.finditer(rx, seg)] for k, rx in LINES.items()}}
                rows.append(r)
                with open(OUT / "runs.jsonl", "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(arm, n, c, t, r["ttft_s"], r["metrics"]["spec_pool_entries"],
                      {k: v for k, v in r["lines"].items() if v}, flush=True)
                if t < 3:
                    prompt += text + FOLLOW[t - 1]
    finally:
        stop(p)
summary = {}
for arm in ("anon", "named"):
    for t in (1, 2, 3):
        v = [r["ttft_s"] for r in rows if r["arm"] == arm and r["turn"] == t]
        if v:
            summary[f"{arm}/t{t}"] = {"n": len(v), "ttft_median_s": round(statistics.median(v), 4),
                                      "min": min(v), "max": max(v)}
    used = [r["metrics"]["cuda_pool_used_bytes"] for r in rows if r["arm"] == arm and r["turn"] == 3]
    ents = [r["metrics"]["spec_pool_entries"] for r in rows if r["arm"] == arm and r["turn"] == 3]
    summary[f"{arm}/after-t3"] = {"cuda_pool_used_bytes": used, "spec_pool_entries": ents}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
