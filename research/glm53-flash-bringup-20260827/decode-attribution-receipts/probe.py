#!/usr/bin/env python3
"""One decode cell against the local memra server. Real prompts only, reasoning_effort PINNED.

usage: probe.py <label> <greedy|sampled> <max_tokens> <prompt_idx> [effort]
Emits one JSON line on stdout.
"""
import json, sys, time, urllib.request

LABEL, ARM, MT, PIDX = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
EFFORT = sys.argv[5] if len(sys.argv) > 5 else "low"
POOL = json.load(open("/home/ubuntu/prompts.json"))["decode"]
p = POOL[PIDX % len(POOL)]

body = {
    "model": "zai/glm-5.3-flash",
    "messages": [{"role": "user", "content": p["text"]}],
    "max_tokens": MT,
    "stream": True,
    "stream_options": {"include_usage": True},
    "reasoning_effort": EFFORT,          # TRAP:reasoning-effort-unpinned-decode-cell
}
if ARM == "greedy":
    body["temperature"] = 0.0            # the INSTRUMENT (byte-deterministic)
else:
    body["temperature"] = 1.0            # vendor default = the PRODUCT
    body["top_p"] = 0.95
    body["seed"] = 20260828

def disk():
    for l in open("/proc/diskstats"):
        f = l.split()
        if f[2] == "nvme0n1":
            return int(f[5]) * 512
    return 0

d0 = disk(); t0 = time.time(); tf = None
n = 0; think = 0; out = 0; usage = None; txt = []; think_txt = []; err = None
req = urllib.request.Request(
    "http://127.0.0.1:18400/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"content-type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            pay = line[5:].strip()
            if pay == "[DONE]":
                break
            try:
                o = json.loads(pay)
            except Exception:
                continue
            if "error" in o:
                err = o["error"].get("message", "")[:300]; break
            if o.get("usage"):
                usage = o["usage"]
            d = (o.get("choices") or [{}])[0].get("delta", {})
            piece = d.get("content") or d.get("reasoning")
            if piece is not None:
                if tf is None:
                    tf = time.time()
                n += 1
                if d.get("content"):
                    out += 1; txt.append(d["content"])
                else:
                    think += 1; think_txt.append(d["reasoning"])
except Exception as e:
    err = f"{type(e).__name__}: {e}"[:300]
t1 = time.time(); d1 = disk()

full = "".join(think_txt) + "".join(txt)
rec = {
    "label": LABEL, "arm": ARM, "effort": EFFORT, "max_tokens": MT,
    "prompt_idx": PIDX, "prompt_sha": p["sha256_16"], "prompt_chars": p["chars"],
    "ttft_s": round(tf - t0, 4) if tf else None,
    "stream_tok": n, "reasoning_tok": think, "content_tok": out,
    "gen_toks": round((n - 1) / (t1 - tf), 4) if (tf and n > 1) else None,
    "wall_s": round(t1 - t0, 4),
    "completion_tokens": (usage or {}).get("completion_tokens"),
    "prompt_tokens": (usage or {}).get("prompt_tokens"),
    "server_elapsed_s": (usage or {}).get("elapsed_s"),
    "server_toks": (round(usage["completion_tokens"] / usage["elapsed_s"], 4)
                    if usage and usage.get("elapsed_s") else None),
    "disk_mib": round((d1 - d0) / 2**20, 1),
    "out_sha": __import__("hashlib").sha256(full.encode()).hexdigest()[:16],
    "out_len": len(full),
    "head": full[:120],
    "error": err,
}
print(json.dumps(rec))
