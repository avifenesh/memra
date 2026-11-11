#!/usr/bin/env python3
"""Every host-to-device copy of 20 MB and more in the whole trace (not only the 4k window): is the 331 MB head-slice
upload a one-off, and does the embedding upload recur once a prompt. usage: big_htod_all.py TRACE.sqlite"""
import sqlite3
import sys
from collections import Counter

con = sqlite3.connect(sys.argv[1])
cur = con.cursor()
mk = {r[0]: r[2].lower() for r in cur.execute("select * from ENUM_CUDA_MEM_KIND")}
rows = cur.execute("select start, end, bytes, deviceId, srcKind from CUPTI_ACTIVITY_KIND_MEMCPY "
                   "where copyKind = 1 and bytes >= 20000000 order by start").fetchall()
print(f"{len(rows)} copies >= 20 MB; by (MB, src): " +
      ", ".join(f"{mb} MB {k}: {c}" for (mb, k), c in sorted(Counter((round(r[2] / 1e6, 1), mk.get(r[4])) for r in rows).items())))
t00 = rows[0][0] if rows else 0
for st, en, b, dev, sk in rows:
    if b >= 40_000_000:
        print(f"{(st - t00) / 1e9:9.3f} s  dev {dev}  {b / 1e6:7.1f} MB  {b / (en - st):5.1f} GB/s  {mk.get(sk)}")
