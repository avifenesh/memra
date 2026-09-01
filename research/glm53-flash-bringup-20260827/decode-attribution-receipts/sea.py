#!/usr/bin/env python3
"""Continuity cell: the bring-up receipts' own short prompt, warm, repeated — the row that
is comparable to BRINGUP.md's 19.70 greedy / 12.70 sampled.
usage: sea.py <tag> <greedy|sampled> <max_tokens> <reps>
"""
import json, statistics, subprocess, sys, time, urllib.request
TAG, ARM, MT, REPS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

def one():
    body = {"model": "zai/glm-5.3-flash",
            "messages": [{"role": "user", "content": "Write one short sentence about the sea."}],
            "max_tokens": MT, "stream": True, "stream_options": {"include_usage": True}}
    if ARM == "greedy":
        body["temperature"] = 0.0
    else:
        body["temperature"] = 1.0; body["top_p"] = 0.95
    t0 = time.time(); tf = None; usage = None; txt = []
    req = urllib.request.Request("http://127.0.0.1:18400/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            s = raw.decode().strip()
            if not s.startswith("data:"): continue
            p = s[5:].strip()
            if p == "[DONE]": break
            try: o = json.loads(p)
            except Exception: continue
            if o.get("usage"): usage = o["usage"]
            d = (o.get("choices") or [{}])[0].get("delta", {})
            pc = d.get("content") or d.get("reasoning")
            if pc is not None:
                if tf is None: tf = time.time()
                txt.append(pc)
    return {"ttft": round(tf - t0, 3) if tf else None,
            "ctok": (usage or {}).get("completion_tokens"),
            "srv": round(usage["completion_tokens"] / usage["elapsed_s"], 3) if usage and usage.get("elapsed_s") else None,
            "sha": __import__("hashlib").sha256("".join(txt).encode()).hexdigest()[:16],
            "head": "".join(txt)[:100]}

rows = [one() for _ in range(REPS)]
srv = [r["srv"] for r in rows if r["srv"]]
print(json.dumps({"tag": TAG, "SEA": True, "arm": ARM, "max_tokens": MT,
                  "srv_median": round(statistics.median(srv), 3) if srv else None,
                  "srv_all": srv, "ttft_all": [r["ttft"] for r in rows],
                  "ctoks": [r["ctok"] for r in rows],
                  "shas": sorted({r["sha"] for r in rows}), "head": rows[-1]["head"]}), flush=True)
