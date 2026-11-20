#!/usr/bin/env python3
"""Per decode-call breakdown from an nsys cuda_gpu_trace CSV of `hy4f-gpu --gen N`.

usage: trace_steps.py TRACE.csv [--skip N]
A call starts at an r_embed_kernel launch. For the decode calls (every call after the first,
the first `--skip` more dropped as warm-up) it prints the median wall (embed start to the next
embed start), GPU busy time (union of kernel + memcpy intervals), and per-kernel time and count
per call, sorted by time.
"""
import csv, sys, statistics, re
from collections import defaultdict

path = sys.argv[1]
skip = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 2
rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        name = r.get("Name") or ""
        start = int(r["Start (ns)"])
        dur = int(r["Duration (ns)"])
        rows.append((start, dur, name, r.get("GrdX", ""), r.get("GrdY", ""), r.get("GrdZ", "")))
rows.sort()
starts = [i for i, r in enumerate(rows) if "r_embed_kernel" in r[2]]
calls = []
for ci, s in enumerate(starts):
    e = starts[ci + 1] if ci + 1 < len(starts) else len(rows)
    calls.append(rows[s:e])
dec = calls[1 + skip:-1]  # drop prefill, warm-up steps and the last (no next start)
print(f"calls {len(calls)}, decode calls analysed {len(dec)}")


def short(n):
    n = re.sub(r"\(.*", "", n)
    n = n.replace("void ", "")
    return n[:60]


walls, busy, kern, cnt, launches = [], [], defaultdict(list), defaultdict(list), []
for c in dec:
    t0 = c[0][0]
    nxt = None
    walls.append(None)
    iv = sorted((r[0], r[0] + r[1]) for r in c)
    tot, cs, ce = 0, None, None
    for a, b in iv:
        if cs is None or a > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    tot += ce - cs
    busy.append(tot)
    launches.append(len(c))
    per = defaultdict(int)
    pc = defaultdict(int)
    for r in c:
        k = short(r[2]) + (f" g{r[3]}x{r[4]}x{r[5]}" if "gemm" in r[2] else "")
        per[short(r[2])] += r[1]
        pc[short(r[2])] += 1
    for k in per:
        kern[k].append(per[k])
        cnt[k].append(pc[k])
# wall: from one call's first op to the next call's first op
for i, c in enumerate(dec):
    idx = calls.index(c)
    walls[i] = calls[idx + 1][0][0] - c[0][0]
med = statistics.median
print(f"step wall p50 {med(walls)/1e6:.2f} ms, GPU busy p50 {med(busy)/1e6:.2f} ms, ops/step {med(launches)}")
tot = sum(med(v) for v in kern.values())
print(f"{'kernel':60s} {'ms/step':>8s} {'n/step':>6s} {'%busy':>6s}")
for k, v in sorted(kern.items(), key=lambda kv: -med(kv[1])):
    print(f"{k:60s} {med(v)/1e6:8.3f} {med(cnt[k]):6.0f} {100*med(v)/med(busy):6.1f}")
