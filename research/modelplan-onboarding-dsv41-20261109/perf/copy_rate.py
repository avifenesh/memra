#!/usr/bin/env python3
"""Peer copies of an nsys trace by size: count, median duration, median GB/s, and a least-squares fit of
duration = fixed + bytes / rate over the size classes. usage: copy_rate.py TRACE.sqlite"""
import sqlite3
import statistics
import sys

db = sqlite3.connect(sys.argv[1])
tabs = [r[0] for r in db.execute("select name from sqlite_master where type='table' and name like 'CUPTI_ACTIVITY_KIND_MEMCPY%'")]
for t in tabs:
    cols = [r[1] for r in db.execute(f"pragma table_info({t})")]
    print(t, "columns:", ",".join(cols))
    rows = db.execute(f"select bytes, end - start from {t} where bytes >= 1048576").fetchall()
    if not rows:
        continue
    by = {}
    for b, d in rows:
        by.setdefault(round(b / 1e6), []).append((b, d))
    print("| MB | copies | median ms | median GB/s | min ms |")
    pts = []
    for mb in sorted(by):
        v = by[mb]
        ds = [d for _, d in v]
        bs = statistics.median([b for b, _ in v])
        md = statistics.median(ds)
        print(f"| {mb} | {len(v)} | {md / 1e6:.3f} | {bs / md:.2f} | {min(ds) / 1e6:.3f} |")
        if len(v) >= 8:
            pts.append((bs, md))
    if len(pts) >= 2:
        n = len(pts)
        sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
        sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
        den = n * sxx - sx * sx
        if den:
            k = (n * sxy - sx * sy) / den
            c = (sy - k * sx) / n
            print(f"fit: duration = {c / 1e6:.3f} ms + bytes / {1 / k if k else 0:.2f} GB/s (ns per byte {k:.4f})")
