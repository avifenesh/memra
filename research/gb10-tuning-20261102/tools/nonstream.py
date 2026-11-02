#!/usr/bin/env python3
"""Non-streaming long-prompt cell: one chat request with a ~TOKENS-token prompt, stream=false,
no timeout_ms (the derived unified-memory default applies). Prints status, wall time and usage.
Usage: nonstream.py BASE TOKENS [MAX_TOKENS]"""
import json
import sys
import time
import urllib.error
import urllib.request

base, n = sys.argv[1].rstrip('/'), int(sys.argv[2])
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 64
words = "the quick brown fox jumps over the lazy dog while seven engineers measure prefill".split()
lines, k = [], 0
while len(lines) * 14.4 < n:
    lines.append(f"{k}: " + " ".join(words[(k + i) % len(words)] for i in range(9)) + ".")
    k += 1
text = "\n".join(lines) + "\n\nHow many numbered lines are above? Answer with the number only."
body = json.dumps({"model": "q38", "stream": False, "max_tokens": max_tokens, "temperature": 0,
                   "messages": [{"role": "user", "content": text}],
                   "chat_template_kwargs": {"enable_thinking": False}}).encode()
req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=4000) as r:
        out = json.load(r)
        status = r.status
except urllib.error.HTTPError as e:
    status, out = e.code, json.loads(e.read() or b"{}")
dt = time.time() - t0
print(json.dumps({"tokens_target": n, "body_bytes": len(body), "status": status, "wall_s": round(dt, 1),
                  "usage": out.get("usage"), "finish": (out.get("choices") or [{}])[0].get("finish_reason"),
                  "content": ((out.get("choices") or [{}])[0].get("message") or {}).get("content", "")[:80],
                  "error": out.get("error")}))
