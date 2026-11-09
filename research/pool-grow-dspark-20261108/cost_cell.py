#!/usr/bin/env python3
"""What the pool grid law (memra#1091) costs where it sends a resume cold (memra#1093 item 2), RTX 5090.

Copied from research/pool-resume-cost-20261107/cost_cell.py for the DSpark-pool grow (memra#1093
follow-up) with one added shape, `chat-dspark-agentic`: the chat with max_tokens 512 on every turn,
so each turn's cap outgrows the session the turn before it parked and the pool resume needs a grow.

Since #1091 a recurrent plain cache resumes an exact extension only from its grid checkpoint; a
session with no checkpoint primes the turn cold (the prefix cache can still serve part of it).
Sessions with no checkpoint by default include boots with MEMRA_AFFINITY=0 and prompts with no
turn markers and no session id. Both shapes, base (before #1091) against fix, interleaved boots:

  shape `chat-affinity0`: MEMRA_AFFINITY=0, the three-turn chat, turn 2 with room for 256 tokens
                          (its parked cap covers turn 3, so the continuation pool takes turn 3).
  shape `raw`:            /v1/completions, no chat template, no session id: turn N+1 = turn N's
                          prompt + its completion + a short follow-on; turn 2 with room for 256.
  shape `chat-default`:   the same chat at the default (the session keeps its checkpoint, so the
                          fix rewinds to it where the base resumed at the parked end).
  shape `chat-affinity0-nopc`: MEMRA_AFFINITY=0 with the prefix cache off, the bound when no
                          on-grid prefix entry can serve the declined turn.
  shape `chat-dspark-long`: the chat at the default with max_tokens 96 / 1024 / 512, so turns 2
                          and 3 run DFlash and turn 3 is the DSpark pool's exact extension.
  shape `chat-dspark-long-switch`: the same with MEMRA_DSPARK_PREFIX_RESTORE=1 (the restore switch
                          the RTX 5090 keeps off), so a declined turn can restore DFlash from an
                          on-grid prefix entry that carries the draft tail.

Qwen3.8-27B NVFP4 + Q5_K GGUF + DFlash2, context 8,192, two sessions, switches at the device
default. Each boot serves CONVS distinct conversations (distinct first line, no shared prefix), one
turn at a time, greedy, thinking off, streamed: TTFT = first content chunk, total = end of stream.
Per request the server log segment gives the pool resume or rewind, a grid-law decline and the
prefix-cache hit.

usage: cost_cell.py OUT BIN_BASE BIN_FIX [ORDER]
  ORDER: comma list of shape:arm boots, default ABBA per shape.
Writes OUT/server-<n>-<shape>-<arm>.log, OUT/runs.jsonl, OUT/summary.json.
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

OUT, BIN_BASE, BIN_FIX = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
DEFAULT_ORDER = ",".join(f"{s}:{a}" for s in ("chat-affinity0", "raw") for a in ("base", "fix", "fix", "base"))
ORDER = [x.split(":") for x in (sys.argv[4] if len(sys.argv) > 4 else DEFAULT_ORDER).split(",")]
OUT.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("CELL_PORT", "18177"))
CONVS = int(os.environ.get("CELL_CONVS", "3"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
DRAFT = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-DFlash2-memra"
ENV = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
       "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2", "MEMRA_DSPARK_SPEC": "1", "MEMRA_DSPARK_DRAFT": DRAFT}
SHAPE_ENV = {"chat-affinity0": {"MEMRA_AFFINITY": "0"}, "raw": {}, "chat-default": {},
             "chat-affinity0-nopc": {"MEMRA_AFFINITY": "0", "MEMRA_PREFIX_CACHE_MB": "0"},
             "chat-dspark-long": {}, "chat-dspark-long-switch": {"MEMRA_DSPARK_PREFIX_RESTORE": "1"},
             "chat-dspark-agentic": {}}
SHAPE_MAXT = {"chat-dspark-long": [96, 1024, 512], "chat-dspark-long-switch": [96, 1024, 512],
              "chat-dspark-agentic": [512, 512, 512]}
MAXT = [96, 256, 96]
USERS = [
    "Summarize the notes in three sentences.",
    "Which note numbers mention the device budget? Answer with a short list.",
    "Now write one test idea that would catch a restore that differs from a cold prime.",
]
LOGS = {
    "pool_resume": r"kv-reuse: (\d+) of (\d+) prompt tokens resumed",
    "pool_rewind": r"grid-rewind: plain extension of (\d+) committed rows rewound to (\d+)",
    "pool_decline": r"grid-rewind: plain declined \(([^)]*)\)",
    "prefix_hit": r"\[prefix-cache\] hit: (\d+) of (\d+) prompt tokens",
    "dspark_restore": r"DSPARK restore: (\d+) of (\d+) prompt tokens",
    "dspark_pool": r"dspark-reuse: (\d+) committed tokens resumed",
    "dspark_decline": r"dspark-reuse DECLINED \(grid law\)",
    "dspark_grow": r"dspark-reuse: grew the rewound session from (\d+) to (\d+) rows",
    "dspark_seat_decline": r"dspark-reuse DECLINED: the rewound session cannot be seated",
}


def notes(k):
    return (f"Reviewer {k}: you are checking batch {k * 11 + 5} of an inference engine's serving notes.\n\n"
            + "\n".join(
                f"Note {i}: the scheduler admits a request only when its KV booking fits the device budget; "
                f"a parked session keeps its prefix entry, and a later turn that extends the same prompt "
                f"restores that entry and primes only the new suffix. Item {i} of checklist {k} asks whether "
                f"the restored state is byte-identical to a cold prime."
                for i in range(1, 41)))


class Server:
    def __init__(self, tag, binary, env):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                raise SystemExit(f"port {PORT} busy")
        self.log = OUT / f"server-{tag}.log"
        e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
        e.update(env)
        self.p = subprocess.Popen([binary], env=e, stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise SystemExit(f"server {tag} exited {self.p.returncode}")
            try:
                urllib.request.urlopen(URL + "/v1/models", timeout=3).read()
                return
            except Exception:
                time.sleep(2)
        raise SystemExit("server not ready")

    def text(self):
        return self.log.read_text(errors="replace")

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()


def stream(path, body):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    ttft, parts, usage = None, [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                piece = ch.get("text") or (d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(piece)
    return "".join(parts), ttft, time.perf_counter() - t0, usage


def conversation(shape, k):
    maxt = SHAPE_MAXT.get(shape, MAXT)
    shape = "chat" if shape.startswith("chat") else shape
    """Yields (turn, path, body) with the history built from the answers sent back."""
    answers = []
    if shape == "raw":
        prompt = notes(k) + "\n\nSummary of the notes:"
        follow = ["\n\nNote numbers that mention the device budget:", "\n\nOne test idea:"]
        for t in range(3):
            text = yield t + 1, "/v1/completions", {"model": "q", "prompt": prompt, "max_tokens": MAXT[t], "temperature": 0}
            if t < 2:
                prompt = prompt + text + follow[t]
        return
    for t in range(3):
        msgs = [{"role": "system", "content": notes(k)}]
        for i in range(t):
            msgs += [{"role": "user", "content": USERS[i]}, {"role": "assistant", "content": answers[i]}]
        msgs.append({"role": "user", "content": USERS[t]})
        text = yield t + 1, "/v1/chat/completions", {"model": "q", "messages": msgs, "max_tokens": maxt[t],
                                                     "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        answers.append(text)


def run(n, shape, arm):
    s = Server(f"{n}-{shape}-{arm}", BIN_BASE if arm == "base" else BIN_FIX, dict(ENV, **SHAPE_ENV[shape]))
    recs = []
    try:
        stream("/v1/chat/completions", {"model": "q", "messages": [{"role": "user", "content": "Say hello."}],
                                        "max_tokens": 8, "temperature": 0})
        for k in range(1, CONVS + 1):
            g = conversation(shape, k)
            item = next(g)
            while True:
                t, path, body = item
                mark = len(s.text())
                text, ttft, total, usage = stream(path, body)
                time.sleep(0.5)
                seg = s.text()[mark:]
                rec = {"boot": n, "shape": shape, "arm": arm, "conv": k, "turn": t,
                       "prompt_tokens": (usage or {}).get("prompt_tokens"),
                       "completion_tokens": (usage or {}).get("completion_tokens"),
                       "ttft_s": round(ttft or -1, 4), "total_s": round(total, 4),
                       **{key: [m.groups() for m in re.finditer(rx, seg)] for key, rx in LOGS.items()}}
                recs.append(rec)
                with open(OUT / "runs.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps(rec), flush=True)
                try:
                    item = g.send(text)
                except StopIteration:
                    break
    finally:
        s.stop()
    return recs


allr = []
for n, (shape, arm) in enumerate(ORDER, 1):
    allr += run(n, shape, arm)
summary = {}
for shape in sorted({r["shape"] for r in allr}):
    for arm in ("base", "fix"):
        for t in (2, 3):
            xs = [r for r in allr if r["shape"] == shape and r["arm"] == arm and r["turn"] == t]
            if xs:
                summary[f"{shape}/{arm}/turn{t}"] = {
                    "n": len(xs),
                    "ttft_median_s": round(statistics.median(r["ttft_s"] for r in xs), 4),
                    "total_median_s": round(statistics.median(r["total_s"] for r in xs), 4),
                    "pool_resumes": sum(bool(r["pool_resume"]) for r in xs),
                    "pool_rewinds": sum(bool(r["pool_rewind"]) for r in xs),
                    "pool_declines": sum(bool(r["pool_decline"]) for r in xs),
                    "prefix_hits": [r["prefix_hit"] for r in xs][:3],
                }
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
