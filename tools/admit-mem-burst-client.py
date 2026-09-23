#!/usr/bin/env python3
"""Burst client for tools/admit-mem-burst-gate.sh (memra#680). Sends B concurrent open-output chat requests
(no max_tokens, temperature 0, non-streaming), released on one barrier, each a one-sentence summary of a 5000-char
docs/SERVING.md slice with its own salt (the lane B day-31 burst shape). Writes one JSON line per request with
status, Retry-After, usage and finish_reason, then exits 0. It judges nothing; the gate does.

usage: admit-mem-burst-client.py <base-url> <model> <burst> <out.jsonl>
"""
import json, os, sys, threading, time, urllib.error, urllib.request

base, model, burst, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
text = open(os.path.join(os.path.dirname(__file__), "..", "docs", "SERVING.md"), encoding="utf-8").read()
rows, lock = [], threading.Lock()
barrier = threading.Barrier(burst)


def one(b):
    salt = 900 + b
    content = f"[{salt}] Summarize the following in one sentence.\n\n" + text[salt * 13:(salt * 13) + 5000]
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    barrier.wait()
    t0 = time.time(); status = None; retry_after = None; usage = None; finish = None; err = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            status = r.status; j = json.load(r); usage = j.get("usage")
            finish = (j.get("choices") or [{}])[0].get("finish_reason")
    except urllib.error.HTTPError as e:
        status = e.code; retry_after = e.headers.get("Retry-After"); err = e.read().decode(errors="replace")[:300]
    except Exception as e:  # a dropped connection is recorded, never retried
        err = repr(e)[:300]
    with lock:
        rows.append({"tag": f"b{b}", "status": status, "retry_after": retry_after, "usage": usage,
                     "finish_reason": finish, "err": err, "ms": int((time.time() - t0) * 1000)})


threads = [threading.Thread(target=one, args=(b,)) for b in range(burst)]
for t in threads:
    t.start()
for t in threads:
    t.join()
with open(out, "w") as f:
    for r in sorted(rows, key=lambda r: int(r["tag"][1:])):
        f.write(json.dumps(r) + "\n")
