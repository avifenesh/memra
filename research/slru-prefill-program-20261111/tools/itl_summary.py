#!/usr/bin/env python3
"""Two-session cells: session A's token gaps that overlap session B's in-flight window.

usage: itl_summary.py <served dir>
A gap is the time between two consecutive streamed tokens of A. "During B" counts every gap
whose interval overlaps [B sent, B done]; "steady" counts A's gaps over the same-length window
before B was sent. Also the SLRU pairs lock-hold trace of a traced cell, if present.
"""
import json
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


print("| cell | B in flight s | A gaps overlapping B: n, p50 ms, p99 ms, max ms | A gaps before B: n, p50 ms, p99 ms, max ms |")
print("|---|---|---|---|")
for d in sorted(root.glob("*itl*")):
    if not (d / "arrivals.json").exists():
        continue
    j = json.loads((d / "arrivals.json").read_text())
    a, s, e = j["a"], j["b"]["sent"], j["b"]["done"]
    gaps = [(a[i - 1], a[i]) for i in range(1, len(a))]
    during = [(y - x) * 1000 for x, y in gaps if y > s and x < e]
    span = e - s
    before = [(y - x) * 1000 for x, y in gaps if x >= s - span and y <= s]
    f = lambda v: f"{len(v)}, {statistics.median(v):.1f}, {pct(v, .99):.1f}, {max(v):.1f}"
    print(f"| {d.name} | {span:.3f} | {f(during)} | {f(before)} |")

for d in sorted(root.glob("*trace*")):
    log = d / "server.log"
    if not log.exists():
        continue
    rows = [dict(re.findall(r"(\w+)=(\S+)", l)) for l in log.read_text().splitlines()
            if l.startswith("[moe-slru-pairs] il=")]
    if rows:
        h = [int(r["hold_us"]) for r in rows]
        modes = sorted({r["mode"] for r in rows})
        print(f"\n{d.name}: {len(rows)} SLRU pairs layer calls, modes {modes}, lock hold per call "
              f"p50 {statistics.median(h) / 1000:.2f} ms, p99 {pct(h, .99) / 1000:.2f} ms, "
              f"max {max(h) / 1000:.2f} ms, sum {sum(h) / 1000:.1f} ms")
