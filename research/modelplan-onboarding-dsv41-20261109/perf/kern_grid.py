#!/usr/bin/env python3
"""One kernel's launches in an nsys trace (sqlite export) by grid: count and mean duration.
usage: kern_grid.py TRACE.sqlite NAME_PART"""
import sqlite3
import sys
from collections import defaultdict

con = sqlite3.connect(sys.argv[1])
con.text_factory = lambda b: b.decode("utf-8", "replace")
cur = con.cursor()
names = dict(cur.execute("select id, value from StringIds"))
t = defaultdict(lambda: [0, 0])
for s, e, n, gx, gy in cur.execute("select start, end, shortName, gridX, gridY from CUPTI_ACTIVITY_KIND_KERNEL"):
    if sys.argv[2] in names.get(n, ""):
        t[(gx, gy)][0] += e - s
        t[(gx, gy)][1] += 1
print("| grid | launches | mean us |")
print("|---|---|---|")
for k, (ns, c) in sorted(t.items(), key=lambda x: -x[1][0]):
    print(f"| {k[0]} x {k[1]} | {c} | {ns / c / 1e3:.1f} |")
