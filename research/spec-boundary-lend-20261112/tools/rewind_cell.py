#!/usr/bin/env python3
"""memra#1221 part 2, review round 1 (should-fix 5): a turn whose speculative boundary capture sits
above its turn checkpoint. Turn 1's chat prompt is padded to a multiple of the 32-token grid, so the
cold prime publishes its seed at the prompt end (`AtPromptEnd`) while the turn checkpoint sits at
the stable boundary below it (the last turn marker, on the grid). The entry then reads the
session's rows past the checkpoint. Turn 2 extends the conversation under the same `user`
(explicit affinity), so the parked session (every answer ends on EOS within 512 tokens, so the
session parks) is rewound to its checkpoint: below the lend.

One warm boot serves turns 1 to 3 in order; one cold boot per later turn primes that turn from
scratch, and every warm answer must equal its cold twin. The warm log is counted: rewinds refused
below a lend (the session is dropped and the request primes cold), rewinds through a fresh cache
(the session survives), affinity rewinds, prefix hits and the spec boundary inserts.

usage: rewind_cell.py OUT NAME BINARY [VAR=VALUE ...]   (PAIR_MODEL, CELL_PORT as pairs_1221b.py)
Writes OUT/NAME/ (per-turn JSON, server logs, summary.json); exit 1 if a warm answer differs."""
import hashlib
import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import time
import types
import urllib.request

signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
PORT = int(os.environ.get("CELL_PORT", "18193"))
MODEL = os.path.expanduser(os.environ.get(
    "PAIR_MODEL", "~/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"))
WORDS = "alpha river copper lantern orbit meadow cipher harbor violet granite ember falcon".split()


def deep_users(n):
    doc = "\n".join(f"Record {i}: " + " ".join(WORDS[(i * 7 + j * 3) % 12] for j in range(14)) + f" {i % 97}."
                    for i in range(n))
    return [doc + "\n\nName the first three records whose line ends in 7.", "Now name three more.",
            "Which of all six has the largest record number?"]


def boot(log, env):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} busy")
    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    e.update(env)
    proc = subprocess.Popen([env["_BIN"]], env=e, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 900:
        if proc.poll() is not None:
            raise SystemExit(f"server exited {proc.returncode}; see {log}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3).read()
            return proc
        except Exception:
            time.sleep(1)
    proc.kill()
    raise SystemExit("server not ready in 900 s")


def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# The pair harness's server shape (pairs_1221b.py): the same config, port and model.
P = types.SimpleNamespace(
    URL=f"http://127.0.0.1:{PORT}", boot=boot, stop=stop, deep_users=deep_users,
    COMMON={"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
            "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2", "MEMRA_TICK_TRACE": "1"})

OUT, NAME, BINARY = pathlib.Path(sys.argv[1]), sys.argv[2], os.path.expanduser(sys.argv[3])
EXTRA = dict(kv.split("=", 1) for kv in sys.argv[4:])
ev = OUT / NAME
ev.mkdir(parents=True, exist_ok=True)
# The served default (MEMRA_PREFIX_STABLE_BOUNDARY off): the cold prime's seed capture sits at the
# prompt end on the grid, the turn checkpoint at the stable boundary under it.
env = dict(P.COMMON, _BIN=BINARY)
env.update(EXTRA)
GRID = 32


def post(path, body):
    r = urllib.request.Request(P.URL + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=3600) as resp:
        return json.loads(resp.read().decode())


def count(msgs):
    obj = post("/v1/tokenize", {"model": "q", "messages": msgs, "add_generation_prompt": True,
                                "chat_template_kwargs": {"enable_thinking": False}})
    return obj.get("count") or len(obj.get("tokens") or [])


def chat(msgs, out, max_tokens=512):
    body = {"model": "q", "temperature": 0, "max_tokens": max_tokens, "user": "rewind-cell",
            "messages": msgs, "chat_template_kwargs": {"enable_thinking": False}}
    obj = post("/v1/chat/completions", body)
    out.write_text(json.dumps(obj))
    m = obj["choices"][0]["message"]
    return m.get("content") or ""


users = P.deep_users(38)
p = P.boot(ev / "warm-server.log", env)
try:
    # Pad turn 1 onto the grid: the prompt end is a multiple of 32 tokens.
    pad = 0
    while count([{"role": "user", "content": users[0] + " ok" * pad}]) % GRID:
        pad += 1
        if pad > 4 * GRID:
            raise SystemExit("no padding puts turn 1 on the grid")
    users[0] = users[0] + " ok" * pad
    n1 = count([{"role": "user", "content": users[0]}])
    answers, warm = [], []
    for t in range(1, 4):
        msgs = []
        for i in range(t):
            msgs.append({"role": "user", "content": users[i]})
            if i < t - 1:
                msgs.append({"role": "assistant", "content": answers[i]})
        a = chat(msgs, ev / f"warm-t{t}.json")
        answers.append(a)
        warm.append((msgs, a))
finally:
    P.stop(p)
cold = {}
for t in (2, 3):
    p = P.boot(ev / f"cold-t{t}-server.log", env)
    try:
        cold[t] = chat(warm[t - 1][0], ev / f"cold-t{t}.json")
    finally:
        P.stop(p)
log = (ev / "warm-server.log").read_text(errors="replace")
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]  # noqa: E731
summary = {
    "name": NAME, "binary": BINARY, "env": EXTRA, "turn1_prompt_tokens": n1, "pad_words": pad,
    "texts": {t: {"warm": sha(warm[t - 1][1]), "cold": sha(cold[t]), "equal": warm[t - 1][1] == cold[t]}
              for t in (2, 3)},
    "turn1": sha(warm[0][1]),
    "refused_below_lend": len(re.findall(r"\[spec\] rewind refused below a lent prefix row", log)),
    "fresh_cache_rewinds": len(re.findall(r"\[spec\] rewind below a lent prefix row", log)),
    "affinity_rewinds": len(re.findall(r"spec-affinity: rewound to", log)),
    "affinity_failed": len(re.findall(r"affinity rewind failed", log)),
    "grid_rewinds": len(re.findall(r"grid-rewind: spec extension", log)),
    "inserts": re.findall(r"\[prefix-cache\] insert \(([a-z0-9-]+)\): (\d+) tokens", log),
    "hits": re.findall(r"\[prefix-cache\] hit: (\d+) of (\d+) prompt tokens", log),
    "usage": {f"warm-t{t}": json.loads((ev / f"warm-t{t}.json").read_text()).get("usage") for t in (1, 2, 3)},
}
(ev / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary), flush=True)
sys.exit(0 if all(v["equal"] for v in summary["texts"].values()) else 1)
