#!/usr/bin/env python3
"""Decode-step times of a served run (memra-server's [hy4f-step] lines with no prefill rows):
p50/p95/p99 by running count, ms, and tokens a step from the run's turns+bench is not read here."""
import re, sys, json
out = {}
for tag in sys.argv[2:]:
    xs = {}
    for line in open(f"{sys.argv[1]}/{tag}/server.log", errors="replace"):
        if "[hy4f-step]" not in line:
            continue
        v = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
        if v.get("prefill_rows") != "0":
            continue
        xs.setdefault(v["running"], []).append(float(v["step_s"]) * 1e3)
    row = {}
    for r, a in sorted(xs.items()):
        a.sort()
        q = lambda p: a[min(len(a) - 1, int(p * len(a)))]
        row[r] = {"n": len(a), "p50": round(q(.5), 1), "p95": round(q(.95), 1), "p99": round(q(.99), 1)}
    out[tag] = row
    print(tag, row.get("1"), "c4:", row.get("4"))
json.dump(out, open(sys.argv[1] + "/step-times.json", "w"), indent=1)
