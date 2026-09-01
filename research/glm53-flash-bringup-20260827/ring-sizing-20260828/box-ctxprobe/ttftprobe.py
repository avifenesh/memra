#!/usr/bin/env python3
"""Streamed TTFT + total wall for the ctxprobe rows around the ring-OFF 408 boundary.
The 90s server deadline cannot be raised, so the stream shows how far prefill got:
time-to-first-delta is the prefill wall on this eager model.
usage: ttftprobe.py <label> [targets...]"""
import json, sys, time, urllib.request, urllib.error

LABEL = sys.argv[1]
TARGETS = [int(a) for a in sys.argv[2:]] or [5000, 6000, 7000]
UNIT = "The quick brown fox jumps over the lazy dog. "
for target in TARGETS:
    reps = max(1, int(target * 4.15 / len(UNIT)))
    body = {"model": "zai/glm-5.3-flash",
            "messages": [{"role": "user", "content": UNIT * reps + "\n\nReply with the single word: ok"}],
            "max_tokens": 8, "temperature": 0.0, "reasoning_effort": "low",
            "stream": True}
    req = urllib.request.Request("http://127.0.0.1:18400/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    first = None
    status = None
    err = ""
    try:
        with urllib.request.urlopen(req, timeout=200) as r:
            status = r.status
            for line in r:
                if line.startswith(b"data:") and first is None:
                    first = time.time() - t0
    except urllib.error.HTTPError as e:
        status = e.code
        err = e.read().decode()[:100]
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    total = time.time() - t0
    print("%s target ~%d: status=%s ttfd=%s total=%.1fs %s"
          % (LABEL, target, status,
             ("%.1fs" % first) if first is not None else "none", total, err), flush=True)
