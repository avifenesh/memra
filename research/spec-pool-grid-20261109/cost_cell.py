#!/usr/bin/env python3
"""What the spec pool's grid law (memra#1093 item 1) costs, RTX 5090, base against fix.

The fix sends a recurrent spec-pool hit with no on-grid checkpoint cold, and rewinds one with an
on-grid checkpoint to it. Shapes (MTP route, every switch at the device default unless named):

  chat-short-open   three short chat turns, max_tokens omitted: the base resumes turn 2 at the
                    parked end (exact tier) and turn 3 at an off-grid checkpoint; the fix primes both
                    cold (no prefix entry exists for prompts this short).
  chat-mid-open     two-paragraph answers: the base rewinds to the off-grid checkpoints 30 and 286;
                    the fix primes turn 2 cold and turn 3 restores the prefix entry turn 2 published.
  chat-story-open   the story turns: on-grid checkpoints (32, 256) on both arms, a control.
  raw-long-nopc     /v1/completions with a long first turn, the prefix cache off, a `user` id: both
                    arms rewind on the grid through the affinity tier (a control).
  chat-exact-open   the pair cells' chat-exact shape (a 32-token first turn behind a fixed system
                    line, one conversation per boot): the base resumes turn 2 at the parked end (64),
                    the fix rewinds the exact tier to 32, and arm `nopool` (the fix binary with
                    MEMRA_REUSE_POOL=0) primes turn 2 cold, the cost of a decline.

Each boot serves CONVS distinct conversations (a distinct first line, no shared prefix), one turn at
a time, greedy, thinking off, streamed: TTFT = first content chunk, total = end of stream. The server
log segment per request gives the pool resume, rewind or decline and the prefix-cache hit.

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

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
BINS = {"base": sys.argv[2], "fix": sys.argv[3], "nopool": sys.argv[3]}
ARM_ENV = {"nopool": {"MEMRA_REUSE_POOL": "0"}}
PORT = int(os.environ.get("CELL_PORT", "18193"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
COMMON = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
          "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2"}
CONVS = int(os.environ.get("CONVS", "3"))
OPENERS = ["Hello.", "Good evening.", "Hi there, quick question.", "Hey.", "Greetings."]
LONG = " ".join(f"Note {i}: a parked session keeps its prefix entry, and a later turn that extends the "
                f"same prompt restores that entry and primes only the new suffix." for i in range(1, 31))
SHAPES = {
    "chat-short-open": ("chat", ["List three prime numbers.", "Now three more, larger ones.",
                                 "Which of all six is the largest?"], {}),
    "chat-mid-open": ("chat", ["Explain in two short paragraphs why a recurrent state cache cannot be rewound for free.",
                               "Give one concrete example of a test that would catch a wrong rewind.",
                               "Summarize both answers in one sentence."], {}),
    "chat-story-open": ("chat", ["Write a story of about 150 words about a lighthouse keeper who finds a bottle.",
                                 "Continue the story for another 150 words.", "Now write a 100-word epilogue."], {}),
    # A `user` id per conversation: the engine arms the turn checkpoint at the stable boundary
    # for a named (or nominatable) prompt, so turn 1's checkpoint sits on the grid.
    "chat-exact-open": ("chat", ["List three prime numbers.", "Now three more, larger ones.",
                                 "Which of all six is the largest?"],
                        {"_SYSTEM": "Answer. Please. Please. Please. Please.", "_CONVS": "1"}),
    "raw-long-nopc": ("raw", [LONG + "\n\nList three prime numbers.", "Now three more, larger ones.",
                              "Which of all six is the largest?"], {"MEMRA_PREFIX_CACHE_MB": "0", "_USER": "1"}),
}
GEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
LINES = {
    "parked_end": r"spec-reuse: (\d+) committed tokens resumed",
    "affinity": r"spec-affinity: rewound to (\d+) of (\d+)",
    "grid_rewind": r"grid-rewind: spec extension of (\d+) committed rows rewound to (\d+)",
    "grid_decline": r"grid-rewind: spec declined|spec-affinity: declined \(grid law",
    "prefix_hit": r"\[prefix-cache\] hit: (\d+) of (\d+)",
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


def stream(path, body):
    body = dict(body, stream=True)
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, text, usage = None, [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            obj = json.loads(line[5:])
            usage = obj.get("usage") or usage
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") if "delta" in ch else ch.get("text")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(piece)
    return ttft, time.perf_counter() - t0, "".join(text), usage


def converse(kind, users, opener, log, user=None, system=None):
    rows, answers = [], []
    for n in range(1, 4):
        body = {"model": "q", "temperature": 0}
        if user:
            body["user"] = user
        if kind == "raw":
            prompt = ""
            for i in range(n):
                u = (opener + " " + users[i]) if i == 0 else users[i]
                prompt += f"<|im_start|>user\n{u}<|im_end|>\n" + GEN
                if i < n - 1:
                    prompt += answers[i] + "<|im_end|>\n"
            body["prompt"] = prompt
            path = "/v1/completions"
        else:
            msgs = [{"role": "system", "content": system}] if system else []
            for i in range(n):
                first = (opener + " " + users[i]) if opener else users[i]
                msgs.append({"role": "user", "content": first if i == 0 else users[i]})
                if i < n - 1:
                    msgs.append({"role": "assistant", "content": answers[i]})
            body["messages"] = msgs
            body["chat_template_kwargs"] = {"enable_thinking": False}
            path = "/v1/chat/completions"
        before = log.stat().st_size
        ttft, total, text, usage = stream(path, body)
        time.sleep(0.3)
        with open(log, errors="replace") as f:
            f.seek(before)
            seg = f.read()
        answers.append(text)
        rows.append({"turn": n, "ttft_s": round(ttft or -1, 4), "total_s": round(total, 4),
                     "chars": len(text), "usage": usage,
                     "lines": {k: [m.groups() for m in re.finditer(rx, seg)] for k, rx in LINES.items()}})
    return rows


def main():
    order = sys.argv[4].split(",") if len(sys.argv) > 4 else [
        f"{s}:{a}" for s in SHAPES for a in ("base", "fix", "fix", "base")]
    runs = OUT / "runs.jsonl"
    for n, item in enumerate(order, 1):
        if (OUT / "YIELD").exists():
            print("yield before", item, flush=True)
            break
        shape, arm = item.split(":")
        kind, users, extra = SHAPES[shape]
        extra = dict(extra)
        named = extra.pop("_USER", None)
        system = extra.pop("_SYSTEM", None)
        convs = int(extra.pop("_CONVS", CONVS))
        log = OUT / f"server-{n}-{shape}-{arm}.log"
        p = boot(log, dict(COMMON, **extra, **ARM_ENV.get(arm, {}), _BIN=BINS[arm]))
        try:
            for c in range(convs):
                for r in converse(kind, users, None if system else OPENERS[c % len(OPENERS)], log,
                                  f"cost-conv-{c}" if named else None, system):
                    r.update(boot=n, shape=shape, arm=arm, conv=c)
                    with open(runs, "a") as f:
                        f.write(json.dumps(r) + "\n")
                    print(shape, arm, c, r["turn"], r["ttft_s"], {k: v for k, v in r["lines"].items() if v}, flush=True)
        finally:
            stop(p)
    rows = [json.loads(l) for l in open(runs)]
    summary = {}
    for r in rows:
        key = f"{r['shape']}/{r['arm']}/t{r['turn']}"
        summary.setdefault(key, []).append(r["ttft_s"])
    summary = {k: {"n": len(v), "ttft_median_s": round(statistics.median(v), 4),
                   "ttft_min_s": min(v), "ttft_max_s": max(v)} for k, v in sorted(summary.items())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


main()
