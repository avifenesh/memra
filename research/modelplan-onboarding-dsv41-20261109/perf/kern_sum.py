#!/usr/bin/env python3
"""Kernel time by name over a whole nsys trace (sqlite export), per forward-layer and device.
usage: kern_sum.py TRACE.sqlite FORWARDS LAYERS [TOP]"""
import sqlite3
import sys
from collections import defaultdict

con = sqlite3.connect(sys.argv[1])
con.text_factory = lambda b: b.decode("utf-8", "replace")
cur = con.cursor()
fw, layers = int(sys.argv[2]), int(sys.argv[3])
top = int(sys.argv[4]) if len(sys.argv) > 4 else 24
names = dict(cur.execute("select id, value from StringIds"))
t = defaultdict(lambda: [0, 0])
dev = set()
for s, e, n, d in cur.execute("select start, end, shortName, deviceId from CUPTI_ACTIVITY_KIND_KERNEL"):
    k = names.get(n, str(n))
    k = k.split("<")[0].replace("v41_", "").replace("_kernel", "")
    t[k][0] += e - s
    t[k][1] += 1
    dev.add(d)
nd = max(1, len(dev))
tot = sum(v[0] for v in t.values())
den = fw * layers * nd
print(f"{len(dev)} device(s), {fw} forwards x {layers} layers: kernels {tot / den / 1e3:.1f} us a layer a device")
print("| kernel | launches a layer | us a layer | share |")
print("|---|---|---|---|")
for k, (ns, c) in sorted(t.items(), key=lambda x: -x[1][0])[:top]:
    print(f"| {k} | {c / den:.1f} | {ns / den / 1e3:.1f} | {100 * ns / tot:.1f}% |")
