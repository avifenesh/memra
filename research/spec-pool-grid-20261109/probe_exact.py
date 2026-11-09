#!/usr/bin/env python3
"""Find first turns whose spec-pool exact tier fires from an on-grid checkpoint (memra#1093 item 1).

The exact tier resumes only when the parked session's committed tokens (prompt, answer, the EOS
and any verify overshoot past it) prefix the next prompt, and its rewind needs the turn checkpoint
on the grid. A prompt too short for a stable boundary keeps its checkpoint at the prompt end, so a
first turn of exactly 32 or 64 tokens has it on the grid. For each candidate question this pads a
system line through /v1/tokenize to 32 and to 64 rendered tokens, sends turn 1 and turn 2 with
max_tokens omitted, greedy, thinking off, and records which tier served turn 2.

usage: probe_exact.py BINARY OUT
Writes OUT/server.log and OUT/candidates.jsonl.
"""
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.request

BIN, OUT = sys.argv[1], pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CELL_PORT", "18193")
PORT = int(os.environ["CELL_PORT"])
URL = f"http://127.0.0.1:{PORT}"
MODEL = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
ENV = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
       "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2"}
QUESTIONS = ["List three prime numbers.", "Name two colors.", "What is 7 times 8?", "Name a planet.",
             "Say hello in French.", "What is the capital of Italy?", "Give one even number.",
             "Name three fruits.", "What day comes after Monday?", "Spell the word cat."]
FOLLOW = "Now three more, larger ones."


def boot(log):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} busy")
    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    e.update(ENV)
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


def padded(q, target):
    for k in range(60):
        system = "Answer." + " Please." * k
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
        n = call("/v1/tokenize", {"model": "q", "messages": msgs, "add_generation_prompt": True,
                                  "chat_template_kwargs": {"enable_thinking": False}})["count"]
        if n == target:
            return system
        if n > target:
            return None
    return None


def main():
    log = OUT / "server.log"
    p = boot(log)
    try:
        for q in QUESTIONS:
            for target in (32, 64):
                system = padded(q, target)
                if system is None:
                    continue
                msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
                body = {"model": "q", "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
                a1 = call("/v1/chat/completions", dict(body, messages=msgs))["choices"][0]["message"]["content"]
                msgs += [{"role": "assistant", "content": a1}, {"role": "user", "content": FOLLOW}]
                before = log.stat().st_size
                call("/v1/chat/completions", dict(body, messages=msgs))
                time.sleep(0.3)
                with open(log, errors="replace") as f:
                    f.seek(before)
                    seg = f.read()
                row = {"question": q, "system": system, "turn1_tokens": target,
                       "exact": re.findall(r"spec-reuse: (\d+) committed tokens resumed", seg),
                       "rewind": re.findall(r"grid-rewind: spec extension of (\d+) committed rows rewound to (\d+)", seg),
                       "affinity": re.findall(r"spec-affinity: rewound to (\d+)", seg)}
                print(json.dumps(row), flush=True)
                with open(OUT / "candidates.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
    finally:
        stop(p)


main()
