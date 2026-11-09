#!/usr/bin/env python3
"""What a sampled resume at its cold twin's K costs on the RTX 5090 (memra#1093 K-rung follow-up).

Base: a sampled turn that restores 1,024 tokens or more runs the cached-long K=2. Fix: it runs the
cold-long K=3 its cold twin runs; greedy resumes keep K=2 on both (the control). Boots alternate
base fix fix base, twice. Each boot serves three conversations of three chat turns: a first turn of
about 1,400 tokens (45 notes and a request), then two follow-ups that each ask for a paragraph, so
turns 2 and 3 restore the prefix cache's seed (1,024 tokens or more). Conversations 0 and 1 sample
at T=1 (seeds 11 and 12), conversation 2 is greedy. max_tokens 256, thinking off, streamed: TTFT is
the first content chunk, decode rate = completion tokens / (end of stream - TTFT).

usage: k_cost.py OUT BIN_BASE BIN_FIX [ORDER]
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
BINS = {"base": sys.argv[2], "fix": sys.argv[3]}
ORDER = (sys.argv[4] if len(sys.argv) > 4 else "base,fix,fix,base,base,fix,fix,base").split(",")
PORT = int(os.environ.get("CELL_PORT", "18193"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
COMMON = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
          "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2"}
TOPICS = ["rivers", "bridges", "volcanoes", "glaciers", "harbors", "canyons", "islands", "meadows"]
CONVS = [("t1", {"temperature": 1.0, "seed": 11}), ("t1", {"temperature": 1.0, "seed": 12}),
         ("greedy", {"temperature": 0})]
FOLLOW = ["Write one more paragraph about what the notes leave out.",
          "Now write a closing paragraph that summarizes both."]
MAX_TOKENS = 256
# K_COST_LONG=1: the decode-focused variant (run 2): three T=1 conversations and the greedy control
# per boot, 400-token answers to requests for about 300 words, so more turns run to the cap.
if os.environ.get("K_COST_LONG") == "1":
    CONVS = [("t1", {"temperature": 1.0, "seed": 21}), ("t1", {"temperature": 1.0, "seed": 22}),
             ("t1", {"temperature": 1.0, "seed": 23}), ("greedy", {"temperature": 0})]
    FOLLOW = ["Write about 300 more words on what the notes leave out.",
              "Now write about 300 words that summarize both answers."]
    MAX_TOKENS = 400
KLINE = r"\[spec-k\] model=\"q\" tenant=\"[^\"]*\" K=(\d+) source=(\S+) prompt=(\d+) cached=(\d+)"


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


def stream(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(dict(body, stream=True)).encode(),
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
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    ttft = ttft if ttft is not None else time.perf_counter() - t0
                    text.append(piece)
    return ttft, time.perf_counter() - t0, "".join(text), usage


def first_turn(topic, n):
    notes = " ".join(f"Note {i}: the {topic} survey records one measurement per site, and a later visit "
                     f"extends the same record with a new reading." for i in range(1, 46))
    ask = "Write about 300 words on what these notes say." if MAX_TOKENS > 256 else "Write a paragraph on what these notes say."
    return f"Boot {n}. Field notes on {topic}. {notes}\n\n{ask}"


rows = []
for n, arm in enumerate(ORDER, 1):
    log = OUT / f"server-{n}-{arm}.log"
    p = boot(log, dict(COMMON, _BIN=BINS[arm]))
    try:
        for c, (samp_name, samp) in enumerate(CONVS):
            topic = TOPICS[(n * len(CONVS) + c) % len(TOPICS)]
            msgs = [{"role": "user", "content": first_turn(topic, n)}]
            for t in range(1, 4):
                body = {"model": "q", "messages": msgs, "max_tokens": MAX_TOKENS,
                        "chat_template_kwargs": {"enable_thinking": False}, **samp}
                before = log.stat().st_size
                ttft, total, text, usage = stream(body)
                time.sleep(0.3)
                with open(log, errors="replace") as f:
                    f.seek(before)
                    seg = f.read()
                k = re.findall(KLINE, seg)
                toks = (usage or {}).get("completion_tokens") or 0
                rate = toks / (total - ttft) if ttft is not None and total > ttft and toks else None
                r = {"boot": n, "arm": arm, "conv": c, "sampling": samp_name, "turn": t,
                     "ttft_s": round(ttft or -1, 4), "total_s": round(total, 4), "completion_tokens": toks,
                     "decode_tok_s": round(rate, 2) if rate else None,
                     "acceptance": ((usage or {}).get("spec") or {}).get("acceptance_rate"),
                     "k": k[0] if k else None}
                rows.append(r)
                with open(OUT / "runs.jsonl", "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(arm, n, c, samp_name, t, r["ttft_s"], r["decode_tok_s"], r["k"], flush=True)
                msgs = msgs + [{"role": "assistant", "content": text}]
                if t < 3:
                    msgs.append({"role": "user", "content": FOLLOW[t - 1]})
    finally:
        stop(p)
summary = {}
for arm in BINS:
    for samp in ("t1", "greedy"):
        for t in (1, 2, 3):
            sel = [r for r in rows if r["arm"] == arm and r["sampling"] == samp and r["turn"] == t]
            if not sel:
                continue
            rates = [r["decode_tok_s"] for r in sel if r["decode_tok_s"]]
            summary[f"{arm}/{samp}/t{t}"] = {
                "n": len(sel), "ttft_median_s": round(statistics.median(r["ttft_s"] for r in sel), 4),
                "decode_median_tok_s": round(statistics.median(rates), 2) if rates else None,
                "decode_range": [min(rates), max(rates)] if rates else None,
                "acceptance_median": round(statistics.median(r["acceptance"] for r in sel if r["acceptance"] is not None), 4),
                "k": sorted({r["k"][0] for r in sel if r["k"]})}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
