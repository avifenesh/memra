#!/usr/bin/env python3
"""Does the spec depth K change the output at the RTX 5090 default? (memra#1093 K-rung follow-up)

The same cold long prompts, served at K=3 (the shipped cold-long rung), at K=2 (the cached-long
rung, forced on cold prompts through the `MEMRA_SPEC_K_COLD_LONG=2` measurement door), and at K=3
again (a determinism control: a repeat boot must reproduce the first byte for byte). Four distinct
prompts of about 1,400 tokens (no shared prefix, so no cache entry serves another), each asked for a
long answer (max_tokens 400), thinking off, greedy and T=1 (seed 1234). Each request's `[spec-k]`
line is recorded, so every row states the K it ran.

usage: k_invariance.py OUT BINARY
Writes OUT/server-<arm>.log, OUT/<arm>.jsonl and OUT/verdict.json.
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
BIN = sys.argv[2]
PORT = int(os.environ.get("CELL_PORT", "18193"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
COMMON = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
          "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2"}
ARMS = [("k3", {}), ("k2", {"MEMRA_SPEC_K_COLD_LONG": "2"}), ("k3-repeat", {})]
TOPICS = ["rivers", "bridges", "volcanoes", "glaciers"]
SAMPLINGS = [("greedy", {"temperature": 0}), ("t1", {"temperature": 1.0, "seed": 1234})]


def prompt(topic, sampling):
    # The sampling leads the prompt, so the T=1 prompts share no prefix-cache entry with the greedy
    # ones and every request is a cold prime at the arm's cold-long K.
    notes = " ".join(f"Note {i}: the {topic} survey records one measurement per site, and a later visit "
                     f"extends the same record with a new reading." for i in range(1, 46))
    return (f"[{sampling}] Field notes on {topic}. {notes}\n\nWrite a detailed report of about 300 words on what "
            f"these notes say about {topic}, in four paragraphs.")


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


def ask(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())


results = {}
for arm, env in ARMS:
    log = OUT / f"server-{arm}.log"
    p = boot(log, dict(COMMON, **env))
    rows = []
    try:
        for name, samp in SAMPLINGS:
            for topic in TOPICS:
                body = {"model": "q", "messages": [{"role": "user", "content": prompt(topic, name)}],
                        "max_tokens": 400, "chat_template_kwargs": {"enable_thinking": False}, **samp}
                before = log.stat().st_size
                obj = ask(body)
                time.sleep(0.3)
                with open(log, errors="replace") as f:
                    f.seek(before)
                    seg = f.read()
                k = re.findall(r"\[spec-k\] model=\"q\" tenant=\"[^\"]*\" K=(\d+) source=(\S+) prompt=(\d+) cached=(\d+)", seg)
                text = obj["choices"][0]["message"].get("content") or ""
                rows.append({"sampling": name, "topic": topic, "k": k, "chars": len(text),
                             "completion_tokens": (obj.get("usage") or {}).get("completion_tokens"),
                             "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text})
    finally:
        stop(p)
    results[arm] = rows
    with open(OUT / f"{arm}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
verdict = []
for i, r in enumerate(results["k3"]):
    verdict.append({"sampling": r["sampling"], "topic": r["topic"],
                    "k": {arm: results[arm][i]["k"] for arm, _ in ARMS},
                    "k3_eq_repeat": r["sha"] == results["k3-repeat"][i]["sha"],
                    "k3_eq_k2": r["sha"] == results["k2"][i]["sha"],
                    "chars": {arm: results[arm][i]["chars"] for arm, _ in ARMS}})
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=1))
for v in verdict:
    print(v["sampling"], v["topic"], "K", [x[0][0] if x else None for x in v["k"].values()],
          "repeat_eq", v["k3_eq_repeat"], "k2_eq", v["k3_eq_k2"], v["chars"])
