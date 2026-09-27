#!/usr/bin/env python3
"""WP-B day 39 readings (DAY39.md 1.4, "Readings, no bound"), beside day33-compare.py's verdicts (unchanged).

usage: day39-read.py <card> <boot dir> ...
Per boot: the burst's HTTP status counts, and over the `[admit-mem] id=` lines inside the burst window (marks.json,
plus 60 s as day33-compare.py reads it) the peak of pending_prime= and of pending_prime_v1= (green only; the red binary
prints day 33's value as pending_prime=), each with the other field on the same line.
"""
import json, os, re, sys

card = sys.argv[1]
KV = re.compile(r"(\w+)=(\S+)")
for d in sys.argv[2:]:
    name = os.path.basename(d.rstrip("/"))
    marks = {}
    try:
        marks = json.load(open(os.path.join(d, "marks.json")))
    except (OSError, ValueError):
        pass
    b0, b1 = marks.get("burst_start_ms"), marks.get("burst_end_ms")
    status = {}
    for l in open(os.path.join(d, "rows.jsonl")):
        r = json.loads(l)
        if r.get("arm") == "burst" or str(r.get("tag", "")).startswith("burst"):
            status[r.get("status")] = status.get(r.get("status"), 0) + 1
    peak_new = peak_v1 = None
    lines = 0
    for line in open(os.path.join(d, "server.log"), errors="replace"):
        if "[admit-mem] id=" not in line:
            continue
        m = re.match(r"(\d+) ", line)
        t = int(m.group(1)) if m else None
        if not (b0 and b1 and t is not None and b0 <= t <= b1 + 60000):
            continue
        kv = dict(KV.findall(line))
        lines += 1
        p = int(kv.get("pending_prime", 0))
        v1 = int(kv["pending_prime_v1"]) if "pending_prime_v1" in kv else None
        if peak_new is None or p > peak_new[0]:
            peak_new = (p, v1, kv.get("verdict"))
        if v1 is not None and (peak_v1 is None or v1 > peak_v1[0]):
            peak_v1 = (v1, p, kv.get("verdict"))
    fmt = lambda x: "none" if x is None else f"{x[0]}(other={x[1]} verdict={x[2]})"
    print(f"DAY39 READING card={card} boot={name} burst_status={dict(sorted(status.items(), key=str))} "
          f"admit_mem_lines_in_burst={lines} peak_pending_prime={fmt(peak_new)} peak_pending_prime_v1={fmt(peak_v1)}")
