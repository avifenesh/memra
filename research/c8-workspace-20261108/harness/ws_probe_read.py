#!/usr/bin/env python3
"""Summarize [ws-probe] lines: per call kind and row bucket, the peak pool growth during the call
(USED_MEM_HIGH minus the start), the retained growth, and the slab before and after.

  python3 ws_probe_read.py SERVE.LOG [SERVE.LOG ...] [--json OUT]
"""
import json
import re
import sys
from collections import defaultdict

LINE = re.compile(r"\[ws-probe\] kind=(\w+) rows=(\d+) members=(\d+) per=\[([^\]]*)\] pos0=(\d+) "
                  r"used0=(\d+) peak=(\d+) retained=(-?\d+) slab0=(\d+) slab1=(\d+)")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
if out_json:
    args.remove(out_json)
rows = []
for path in args:
    for line in open(path, errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        kind, r, mem, per, pos0, used0, peak, ret, s0, s1 = m.groups()
        rows.append(dict(log=path, kind=kind, rows=int(r), members=int(mem),
                         per=[int(x) for x in per.split(",") if x.strip()], pos0=int(pos0),
                         used0=int(used0), peak=int(peak), retained=int(ret), slab0=int(s0), slab1=int(s1)))
MB = 1e6


def bucket(n):
    for b in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
        if n <= b:
            return b
    return n


groups = defaultdict(list)
for r in rows:
    groups[(r["kind"], bucket(r["rows"]))].append(r)
print(f"{len(rows)} probe lines from {len(args)} logs")
print("| kind | rows <= | calls | members max | peak MB median | peak MB max | peak bytes/row max (excl. slab growth) "
      "| slab growth MB max | retained MB max | slab after MB max |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (kind, b), g in sorted(groups.items()):
    peaks = sorted(x["peak"] for x in g)
    excl = [(x["peak"] - max(0, x["slab1"] - x["slab0"])) / max(1, x["rows"]) for x in g]
    print(f"| {kind} | {b} | {len(g)} | {max(x['members'] for x in g)} | {peaks[len(peaks)//2]/MB:.0f} | "
          f"{peaks[-1]/MB:.0f} | {max(excl):.0f} | {max(x['slab1'] - x['slab0'] for x in g)/MB:.0f} | "
          f"{max(x['retained'] for x in g)/MB:.0f} | {max(x['slab1'] for x in g)/MB:.0f} |")
if out_json:
    json.dump(rows, open(out_json, "w"))
