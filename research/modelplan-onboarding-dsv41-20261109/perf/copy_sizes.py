#!/usr/bin/env python3
"""Peer copies of an nsys sqlite export grouped by size (and source/destination device): count a layer and MB a
layer, for one destination device, over the first prefill window (the first `--tokens` prefill of the trace).

usage: copy_sizes.py TRACE.sqlite LAYERS [DST_DEVICE (0)] [--window START_NS END_NS]
"""
import sqlite3
import sys
from collections import defaultdict

db, layers = sys.argv[1], int(sys.argv[2])
dst = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 0
con = sqlite3.connect(db)
cur = con.cursor()
cols = [r[1] for r in cur.execute("pragma table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")]
has_src = "srcDeviceId" in cols
q = "select start, end, bytes, copyKind, deviceId" + (", srcDeviceId, dstDeviceId" if has_src else "") + \
    " from CUPTI_ACTIVITY_KIND_MEMCPY order by start"
rows = cur.execute(q).fetchall()
# the windows: the trace holds two prefills (4,096 then 32,768); split at the largest gap between copies
starts = [r[0] for r in rows]
gaps = sorted(((starts[i + 1] - starts[i], i) for i in range(len(starts) - 1)), reverse=True)[:3]
print("largest gaps between copies (ms, index):", [(round(g / 1e6, 1), i) for g, i in gaps])
cut = max(gaps)[1] if gaps else len(rows) - 1
win = rows[: cut + 1]
t = defaultdict(lambda: [0, 0, 0])
for r in win:
    s, e, b, k, d = r[:5]
    sd, dd = (r[5], r[6]) if has_src else (None, None)
    if k != 10 and not has_src:  # 10 = peer-to-peer in CUPTI's copy kinds
        pass
    key = (k, sd, dd if has_src else d, b)
    t[key][0] += 1
    t[key][1] += b
    t[key][2] += e - s
print(f"first window: {len(win)} copies")
print("| kind | src | dst | bytes a copy | copies a layer | MB a layer | ms a layer |")
print("|---|---|---|---|---|---|---|")
for k, (c, b, ns) in sorted(t.items(), key=lambda kv: -kv[1][1]):
    if (k[2] == dst) and b / layers > 0.2e6:
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {c / layers:.2f} | {b / layers / 1e6:.1f} | {ns / layers / 1e6:.2f} |")
