#!/usr/bin/env python3
"""One rung of the 262k 2-card ladder (lane/glm5-262k-2card-receipt): stream a REAL-corpus
prompt of a target token count through the serving surface, record the prime wall-clock
(time to first decoded token = the monolithic prefill on this eager model), then per-token
arrival times for the decode.

VERBATIM copy of the 1m-demo lane's primeprobe.py (lane/glm53-1m-demo,
research/glm53-flash-bringup-20260827/1m-demo-20260829/primeprobe.py) with two named
deviations: EP and MODEL are env-parametrized (this cell serves on 127.0.0.1:18600, the
coordinator's port slot for this lane), and the summary rows carry a finish/loop note.

Sampling arms (fleet law: reasoning_effort pinned on every arm):
  greedy  temperature 0.0                       byte-receipt instrument
  vendor  NO sampling params at all             the real traffic shape (models.toml defaults)

Token counts come from the server's own usage.prompt_tokens, never estimated.
usage: primeprobe262.py <label> <corpus.txt> <chars> <max_tokens> <greedy|vendor> <out.json>
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

LABEL, CORPUS, CHARS, MAXTOK, MODE, OUTFILE = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])
EP = os.environ.get("EP", "http://127.0.0.1:18600")
MODEL = os.environ.get("MODEL", "zai/glm-5.3-flash")

text = open(CORPUS, encoding="utf-8", errors="strict").read()[:CHARS]
ask = ("\n\n---\nThe text above is a corpus of classic literature. In one short paragraph, "
       "name two of the works it contains and one theme they share.")
body = {"model": MODEL,
        "messages": [{"role": "user", "content": text + ask}],
        "max_tokens": MAXTOK,
        "reasoning_effort": "low",
        "stream": True,
        "stream_options": {"include_usage": True}}
if MODE == "greedy":
    body["temperature"] = 0.0
elif MODE != "vendor":
    sys.exit(f"mode must be greedy|vendor, got {MODE}")

req = urllib.request.Request(EP + "/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"content-type": "application/json"})
t0 = time.time()
events = []          # (t_rel, n_chunk_chars)
pieces = []
reasoning_pieces = []
usage = None
err = ""
status = -1
t_first_content = None
t_last = None
try:
    with urllib.request.urlopen(req, timeout=64800) as r:
        status = r.status
        for line in r:
            now = time.time()
            if not line.startswith(b"data:"):
                continue
            frag = line[5:].strip()
            if not frag or frag == b"[DONE]":
                continue
            try:
                d = json.loads(frag)
            except Exception:
                continue
            u = d.get("usage")
            if u:
                usage = u
            e2 = (d.get("error") or {}).get("message")
            if e2:
                err = e2
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                # glm5 thinks: the first decoded tokens usually arrive on the REASONING
                # channel, so first-token time and per-token cadence must count both.
                c = delta.get("content")
                rc = delta.get("reasoning_content") or delta.get("reasoning")
                if c or rc:
                    if t_first_content is None:
                        t_first_content = now
                    t_last = now
                    events.append((round(now - t0, 4), len(c or rc or "")))
                if rc:
                    reasoning_pieces.append(rc)
                if c:
                    pieces.append(c)
except urllib.error.HTTPError as e:
    status = e.code
    err = e.read().decode(errors="replace")[:800]
except Exception as e:
    err = f"{type(e).__name__}: {e}"

wall = time.time() - t0
out_text = "".join(pieces)
reasoning_text = "".join(reasoning_pieces)
pt = (usage or {}).get("prompt_tokens")
ct = (usage or {}).get("completion_tokens")
prefill_s = round(t_first_content - t0, 3) if t_first_content else None
prefill_tps = round(pt / prefill_s, 2) if (pt and prefill_s) else None
decode_tps = None
if ct and ct > 1 and t_first_content and t_last and t_last > t_first_content:
    decode_tps = round((ct - 1) / (t_last - t_first_content), 2)

summary = {"label": LABEL, "mode": MODE, "corpus": CORPUS, "chars": CHARS,
           "max_tokens": MAXTOK, "status": status, "usage": usage,
           "prefill_s": prefill_s, "prefill_tok_s": prefill_tps,
           "decode_tok_s": decode_tps, "wall_s": round(wall, 1),
           "n_content_chunks": len(events), "error": err,
           "output": out_text, "reasoning": reasoning_text, "chunk_times": events}
with open(OUTFILE, "w") as f:
    json.dump(summary, f, indent=1)
print(f"[{LABEL}] mode={MODE} status={status} prompt_tokens={pt} completion_tokens={ct}\n"
      f"  prefill={prefill_s}s ({prefill_tps} tok/s)  decode={decode_tps} tok/s  "
      f"wall={round(wall,1)}s  chunks={len(events)}")
if err:
    print(f"  ERROR: {err[:300]}")
print(f"  reasoning[:200]: {reasoning_text[:200]!r}")
print(f"  output[:400]: {out_text[:400]!r}")
sys.exit(0 if (status == 200 and not err and t_first_content) else 2)
