#!/usr/bin/env python3
"""Served 16-wide byte gate for the deepseek-v4.1 route: each prompt greedy alone, then all of them at once
(one decode call carries them when MEMRA_DSV41_BATCH_ROWS allows), and every answer must be the same text and
token count as its solo run. Prompts are token-id files (int32), sent to /v1/completions with ignore_eos.

usage: wide_gate.py ADDR MAX_TOKENS OUT.json PROMPT.i32...   (exit 0 on PASS, 1 on FAIL)
"""
import concurrent.futures
import http.client
import json
import struct
import sys
import threading
import time

addr, max_tokens, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
files = sys.argv[4:]
host, port = addr.split(":")


def ids_of(path):
    b = open(path, "rb").read()
    return list(struct.unpack(f"<{len(b) // 4}i", b))


def complete(ids):
    body = {"model": "v41", "prompt_ids": ids, "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
            "stream": False, "ignore_eos": True}
    c = http.client.HTTPConnection(host, int(port), timeout=600)
    t = time.time()
    c.request("POST", "/v1/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    raw = r.read()
    c.close()
    if r.status != 200:
        return {"error": f"HTTP {r.status}: {raw[:300]!r}"}
    d = json.loads(raw)
    return {"text": d["choices"][0]["text"], "tokens": d["usage"]["completion_tokens"], "s": round(time.time() - t, 3)}


prompts = [ids_of(f) for f in files]
solo = [complete(p) for p in prompts]
bar = threading.Barrier(len(prompts))


def at_once(p):
    bar.wait()
    return complete(p)


with concurrent.futures.ThreadPoolExecutor(len(prompts)) as ex:
    wide = list(ex.map(at_once, prompts))
rows = []
ok = True
for f, s, w in zip(files, solo, wide):
    same = "error" not in s and "error" not in w and s["text"] == w["text"] and s["tokens"] == w["tokens"]
    ok &= same and s.get("tokens") == max_tokens
    rows.append({"prompt": f, "equal": same, "solo": s, "wide": w})
json.dump({"pass": ok, "n": len(prompts), "max_tokens": max_tokens, "rows": rows}, open(out, "w"), ensure_ascii=False,
          indent=1)
print(f"wide-gate: {'PASS' if ok else 'FAIL'} {sum(r['equal'] for r in rows)} of {len(rows)} answers equal solo "
      f"({max_tokens} tokens each, all at once)")
sys.exit(0 if ok else 1)
