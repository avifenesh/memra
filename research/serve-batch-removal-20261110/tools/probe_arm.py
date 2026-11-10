#!/usr/bin/env python3
"""memra #1158: one served arm of the spec-vs-plain comparison on a safetensors dir.

Boots memra-server once with the given extra environment, sends each prompt twice (chat for
text and usage, native completions with chat:true for the token ids), saves every response and
the server log, and stops the server. Greedy, max_tokens from the command line.

usage: probe_arm.py SERVER_BIN ST_DIR OUT_DIR MAX_TOKENS [NAME=VALUE ...]
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

BIN, ST, OUT, MAXTOK = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
EXTRA = dict(a.split("=", 1) for a in sys.argv[5:])
PORT = 8180
BASE = f"http://127.0.0.1:{PORT}"
PROMPTS = {
    "gate": "What is the capital of France? Answer in one short sentence.",
    "sched": (
        "Explain, step by step and in full detail, how a speculative decoding scheduler "
        "should decide between running a draft-and-verify burst and taking a batched decode "
        "step when many sessions share one GPU. Cover the throughput arithmetic, the latency "
        "consequences, and how hysteresis prevents mode thrash."
    ),
    "essay": (
        "Write a long, careful essay about the history of memory hierarchies in computer "
        "architecture, from core memory through to HBM stacks."
    ),
}


def post(path, body, timeout=600):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


os.makedirs(OUT, exist_ok=True)
busy = subprocess.run(["ss", "-tln"], capture_output=True, text=True).stdout
if f":{PORT} " in busy:
    sys.exit(f"port {PORT} is already listening; refusing to measure against it")
env = dict(os.environ)
env.update(EXTRA)
env.update({"MEMRA_MODELS": f"st={ST}", "MEMRA_ADDR": f"127.0.0.1:{PORT}"})
log = open(os.path.join(OUT, "server.log"), "wb")
proc = subprocess.Popen([BIN], stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
try:
    for _ in range(600):
        if proc.poll() is not None:
            sys.exit("server exited during boot")
        try:
            urllib.request.urlopen(BASE + "/health", timeout=2).read()
            break
        except Exception:
            time.sleep(1)
    else:
        sys.exit("server never came up")
    for name, prompt in PROMPTS.items():
        chat = post(
            "/v1/chat/completions",
            {"model": "st", "messages": [{"role": "user", "content": prompt}],
             "max_tokens": MAXTOK, "temperature": 0},
        )
        json.dump(chat, open(os.path.join(OUT, f"{name}-chat.json"), "w"))
        comp = post(
            "/v1/completions",
            {"model": "st", "prompt": prompt, "chat": True, "max_tokens": MAXTOK, "temperature": 0},
        )
        json.dump(comp, open(os.path.join(OUT, f"{name}-completions.json"), "w"))
        m = chat["choices"][0]["message"]
        print(
            f"{name}: chat tokens={chat['usage']['completion_tokens']} "
            f"finish={chat['choices'][0].get('finish_reason')} "
            f"chars={len((m.get('content') or '') + (m.get('reasoning') or ''))} "
            f"spec={chat['usage'].get('spec')} | completions ids={len(comp.get('tokens') or [])}"
        )
finally:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=60)
    except Exception:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
    log.close()
