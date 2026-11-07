#!/usr/bin/env python3
"""WP-B day 40 readings (DAY40.md 1.3, no bound): per boot, the burst's HTTP status counts, and over the
`[admit-mem] id=` lines inside the burst window (marks.json, plus 60 s as day33-compare.py reads it): how many lines
the prefix cache's budget cap bound (pending_seed < pending_seed_uncapped) and the largest difference, and the peak
pending_prime with pending_prime_v1 on the same line.

usage: day40-read.py <card> <boots dir>
"""
import glob, json, os, re, sys

card, boots = sys.argv[1], sys.argv[2]
KV = re.compile(r"(\w+)=(\S+)")
for d in sorted(glob.glob(os.path.join(boots, "O*-*"))):
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d)
    try:
        marks = json.load(open(os.path.join(d, "marks.json")))
    except (OSError, ValueError):
        marks = {}
    b0, b1 = marks.get("burst_start_ms"), marks.get("burst_end_ms")
    status = {}
    try:
        for l in open(os.path.join(d, "rows.jsonl")):
            r = json.loads(l)
            if str(r.get("tag", "")).startswith("burst"):
                status[r.get("status")] = status.get(r.get("status"), 0) + 1
    except OSError:
        pass
    lines = bound = 0
    max_diff = 0
    peak = None
    try:
        log = open(os.path.join(d, "server.log"), errors="replace")
    except OSError:
        log = []
    for line in log:
        if "[admit-mem] id=" not in line:
            continue
        m = re.match(r"(\d+) ", line)
        t = int(m.group(1)) if m else None
        if not (b0 and b1 and t is not None and b0 <= t <= b1 + 60000):
            continue
        kv = dict(KV.findall(line))
        lines += 1
        seed, unc = int(kv.get("pending_seed", 0)), int(kv.get("pending_seed_uncapped", 0))
        if seed < unc:
            bound += 1
            max_diff = max(max_diff, unc - seed)
        p = int(kv.get("pending_prime", 0))
        if peak is None or p > peak[0]:
            peak = (p, kv.get("pending_prime_v1"), kv.get("verdict"))
    print(f"DAY40 READING card={card} boot={name} burst_status={dict(sorted(status.items(), key=str))} "
          f"admit_mem_lines_in_burst={lines} seed_cap_bound_lines={bound} max_cap_cut_bytes={max_diff} "
          f"peak_pending_prime={peak}")
