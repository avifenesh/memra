#!/usr/bin/env python3
"""VENDOR-DEFAULT sampled probe: the PRODUCT shape.

NO temperature, NO top_p, NO top_k, NO seed in the body — exactly what an omitting client
sends. reasoning_effort is PINNED (TRAP:reasoning-effort-unpinned-decode-cell).
Banks the FULL reasoning + content text so quality is judged from bytes, not a summary.

usage: vprobe.py <label> <prompt_idx> <max_tokens> <effort> <outdir> [rep]
"""
import hashlib, json, os, sys, time, urllib.request

LABEL, PIDX, MT, EFFORT, OUTDIR = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
REP = sys.argv[6] if len(sys.argv) > 6 else "0"
POOL = json.load(open("/home/ubuntu/prompts.json"))["decode"]
p = POOL[PIDX % len(POOL)]
os.makedirs(OUTDIR, exist_ok=True)

body = {
    "model": "zai/glm-5.3-flash",
    "messages": [{"role": "user", "content": p["text"]}],
    "max_tokens": MT,
    "stream": True,
    "stream_options": {"include_usage": True},
}
if EFFORT != "none":
    body["reasoning_effort"] = EFFORT

t0 = time.time(); tf = None
n = think = out = 0; usage = None; txt = []; think_txt = []; err = None; fr = None
req = urllib.request.Request("http://127.0.0.1:18400/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"content-type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=3600) as r:
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
                err = json.dumps(o["error"])[:400]; break
            if o.get("usage"):
                usage = o["usage"]
            ch = (o.get("choices") or [{}])[0]
            if ch.get("finish_reason"):
                fr = ch["finish_reason"]
            d = ch.get("delta", {})
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
    err = f"{type(e).__name__}: {e}"[:400]
t1 = time.time()

reasoning = "".join(think_txt); content = "".join(txt)
stem = f"{OUTDIR}/{LABEL}-p{PIDX}-rep{REP}"
open(stem + ".reasoning.txt", "w").write(reasoning)
open(stem + ".content.txt", "w").write(content)
open(stem + ".request.json", "w").write(json.dumps(body, ensure_ascii=False))

def loopiness(s, w=48):
    """crude degeneration signal: fraction of the tail covered by the most repeated window"""
    if len(s) < 4 * w:
        return 0.0
    tail = s[-2000:]
    best = 0
    seen = {}
    for i in range(0, len(tail) - w):
        k = tail[i:i+w]
        seen[k] = seen.get(k, 0) + 1
        best = max(best, seen[k])
    return round(best * w / len(tail), 3)

u = usage or {}
print(json.dumps({
    "label": LABEL, "rep": REP, "prompt_idx": PIDX, "prompt_sha": p["sha256_16"],
    "effort_sent": (EFFORT if EFFORT != "none" else None),
    "sampling_params_sent": [k for k in ("temperature","top_p","top_k","seed") if k in body],
    "max_tokens": MT, "finish_reason": fr,
    "ttft_s": round(tf - t0, 4) if tf else None, "wall_s": round(t1 - t0, 4),
    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
    "server_elapsed_s": u.get("elapsed_s"),
    "server_toks": (round(u["completion_tokens"]/u["elapsed_s"], 3) if u.get("elapsed_s") else None),
    "reasoning_tok": think, "content_tok": out,
    "reasoning_chars": len(reasoning), "content_chars": len(content),
    "reasoning_sha16": hashlib.sha256(reasoning.encode()).hexdigest()[:16],
    "content_sha16": hashlib.sha256(content.encode()).hexdigest()[:16],
    "loop_score_reasoning": loopiness(reasoning), "loop_score_content": loopiness(content),
    "content_head": content[:200], "reasoning_head": reasoning[:160],
    "error": err, "files": stem,
}))
