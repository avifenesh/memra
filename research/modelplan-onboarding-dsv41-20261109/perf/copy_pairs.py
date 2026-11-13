#!/usr/bin/env python3
"""Peer copies of an nsys trace by (source card, destination card): count, GB, median GB/s, best GB/s, for
copies of at least 8 MB. usage: copy_pairs.py TRACE.sqlite"""
import sqlite3
import statistics
import sys

db = sqlite3.connect(sys.argv[1])
rows = db.execute(
    "select srcDeviceId, dstDeviceId, bytes, end - start, start from CUPTI_ACTIVITY_KIND_MEMCPY "
    "where bytes >= 8388608 and srcDeviceId != dstDeviceId"
).fetchall()
by = {}
for s, d, b, t, _ in rows:
    by.setdefault((s, d), []).append(b / t)
print("| src | dst | copies | median GB/s | best GB/s | worst GB/s |")
for k in sorted(by):
    v = by[k]
    print(f"| {k[0]} | {k[1]} | {len(v)} | {statistics.median(v):.2f} | {max(v):.1f} | {min(v):.2f} |")
# how many peer copies overlap in time, on average, weighted by duration
ev = []
for s, d, b, t, st in rows:
    ev.append((st, 1))
    ev.append((st + t, -1))
ev.sort()
cur = 0
last = ev[0][0] if ev else 0
acc = {}
for t, dlt in ev:
    acc[cur] = acc.get(cur, 0) + (t - last)
    cur += dlt
    last = t
tot = sum(v for k, v in acc.items() if k > 0)
if tot:
    print("copies in flight at once (share of the time any is in flight): " + ", ".join(f"{k}: {v / tot:.0%}" for k, v in sorted(acc.items()) if k > 0))
