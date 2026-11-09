#!/usr/bin/env python3
"""The affinity tiers' 1..15-row suffix (memra#1093 item 3): reachable, and warm == cold?

Shape: an "edit the last message and resend" client. Turn 1 sends a system line and a user
question with a `user` id; the session parks with its rewind checkpoint at the stable boundary
b = the last turn marker rounded down to the 32-token grid, at least PRIME_MIN_T (16) rows under
the prompt end (`plain_checkpoint_boundary`). Turn 2 resends the same conversation with the
question's tail cut so the new prompt keeps turn 1's tokens through b and ends 1..15 rows past it.
The affinity tier then rewinds to b and primes that short suffix token by token. One warm boot
(turn 1, then turn 2) against one cold boot (turn 2 alone); the answers must be byte-equal.

b is computed from /v1/tokenize: the prompt without the generation header ends at the header's
turn marker, so its length is the marker's index.

usage: floor_pair_cells.py OUT CASE...   CASE = name=BINARY|ROUTE|ENV
  ROUTE: plain (greedy with presence_penalty 0.5, which the spec route refuses: greedy-penalized
  requests serve plain) or spec (greedy, no penalty: the MTP spec route).
  ENV: space-separated VAR=VALUE for every boot (empty = none).
Writes OUT/<name>/ and OUT/verdicts.jsonl; exit 1 if any case's bytes differ.
"""
import hashlib
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.request

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("CELL_PORT", "18193"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
COMMON = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
          "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2"}
SYSTEM = ("You are a careful reviewer of an inference engine's serving code. "
          + " ".join(f"Rule {i}: answer in plain words and keep every number exact." for i in range(1, 9)))
QUESTION = ("Explain in two short paragraphs why a recurrent state cache cannot be rewound for free, "
            "and name one test that catches a wrong rewind in a serving engine written in Rust")
LINES = {
    "plain_affinity": r"plain-affinity: rewound to (\d+) of (\d+)",
    "plain_affinity_declined": r"plain-affinity: declined \(([^;]*)",
    "spec_affinity": r"spec-affinity: rewound to (\d+) of (\d+)",
    "spec_affinity_declined": r"spec-affinity: declined \(([^;]*)",
    "prefix_hit": r"\[prefix-cache\] hit: (\d+) of (\d+)",
    "spec_restore": r"spec restore: (\d+) of (\d+)",
}


def boot(log, env):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} busy")
    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    e.update(env)
    p = subprocess.Popen([env["_BIN"]], env=e, stdout=open(log, "w"), stderr=subprocess.STDOUT)
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


def msgs(q):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]


def tokens(q, gen):
    return call("/v1/tokenize", {"model": "q", "messages": msgs(q), "add_generation_prompt": gen,
                                 "chat_template_kwargs": {"enable_thinking": False}})["tokens"]


def plan():
    """Turn 2's question: turn 1's tokens through b, 1..15 rows past b. Needs a live server."""
    p1 = tokens(QUESTION, True)
    marker = len(tokens(QUESTION, False))
    b = marker // 32 * 32
    while b >= 32 and len(p1) - b < 16:
        b -= 32
    words = QUESTION.split(" ")
    for k in range(len(words) - 1, 0, -1):
        q2 = " ".join(words[:k])
        p2 = tokens(q2, True)
        if b < len(p2) < b + 16 and p2[:b] == p1[:b]:
            return {"b": b, "p1": len(p1), "p2": len(p2), "q2": q2}
    raise SystemExit(f"no cut of the question keeps the tokens through {b} and ends 1..15 past it")


def ask(q, req, out):
    """Streamed: returns (reasoning + NUL + content, TTFT in s). The raw chunks go to `out`."""
    body = dict(req, model="q", messages=msgs(q), user="floor-cell", stream=True,
                chat_template_kwargs={"enable_thinking": False})
    r = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, reasoning, content, chunks = None, [], [], []
    with urllib.request.urlopen(r, timeout=900) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            obj = json.loads(line[5:])
            chunks.append(obj)
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                for key, acc in (("reasoning_content", reasoning), ("content", content)):
                    if d.get(key):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        acc.append(d[key])
    out.write_text(json.dumps({"request": body, "ttft_s": ttft, "total_s": time.perf_counter() - t0,
                               "chunks": chunks}))
    return "".join(reasoning) + "\x00" + "".join(content), ttft


def case(name, binary, route, extra):
    ev = OUT / name
    ev.mkdir(parents=True, exist_ok=True)
    env = dict(COMMON, _BIN=binary)
    for kv in extra.split():
        k, v = kv.split("=", 1)
        env[k] = v
    req = {"temperature": 0, "max_tokens": 160}
    if route == "plain":
        req["presence_penalty"] = 0.5
    (ev / "binary.sha").write_text(hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest()[:16] + "\n")
    log = ev / "warm-server.log"
    p = boot(log, env)
    try:
        shape = plan()
        ask(QUESTION, req, ev / "warm-t1.json")
        before = log.stat().st_size
        warm, warm_ttft = ask(shape["q2"], req, ev / "warm-t2.json")
        time.sleep(0.3)
        with open(log, errors="replace") as f:
            f.seek(before)
            seg = f.read()
    finally:
        stop(p)
    p = boot(ev / "cold-t2-server.log", env)
    try:
        cold, cold_ttft = ask(shape["q2"], req, ev / "cold-t2.json")
    finally:
        stop(p)
    lines = {k: [m.groups() for m in re.finditer(rx, seg)] for k, rx in LINES.items()}
    return {"case": name, "binary": binary, "route": route, "env": extra or "(device default)",
            "shape": shape, "equal": warm == cold, "warm_len": len(warm) - 1, "cold_len": len(cold) - 1,
            "warm_ttft_s": round(warm_ttft or -1, 4), "cold_ttft_s": round(cold_ttft or -1, 4),
            "turn2": {k: v for k, v in lines.items() if v}}


failed = False
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1)
    binary, route, extra = rest.split("|", 2)
    v = case(name, binary, route, extra)
    failed |= not v["equal"]
    print(json.dumps(v), flush=True)
    with open(OUT / "verdicts.jsonl", "a") as f:
        f.write(json.dumps(v) + "\n")
sys.exit(1 if failed else 0)
