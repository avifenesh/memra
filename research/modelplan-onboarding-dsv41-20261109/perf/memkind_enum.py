#!/usr/bin/env python3
"""The trace's own memory-kind enum (nsys numbers differ from CUPTI's), and the >= 1 MB host-to-device copies of the
4,096-token prefill window by source kind name. usage: memkind_enum.py TRACE.sqlite"""
import sqlite3
import sys

con = sqlite3.connect(sys.argv[1])
cur = con.cursor()
tabs = [r[0] for r in cur.execute("select name from sqlite_master where type = 'table' and name like 'ENUM_CUDA_MEM%'")]
names = {}
for t in tabs:
    rows = cur.execute(f"select * from {t}").fetchall()
    print(t, rows)
    if t == "ENUM_CUDA_MEM_KIND":
        names = {r[0]: r[1] for r in rows}
pc = cur.execute("select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 10 order by start").fetchall()[36:]
acc, w = 0, []
for x in pc:
    if acc >= 27282112512:
        break
    acc += x[2]
    w.append(x)
t0, t1 = w[0][0] - 60_000_000, w[-1][1] + 60_000_000
print("| start ms | device | MB | GB/s | srcKind | name |")
print("|---|---|---|---|---|---|")
for st, en, b, dev, sk in cur.execute(
        "select start, end, bytes, deviceId, srcKind from CUPTI_ACTIVITY_KIND_MEMCPY "
        "where copyKind = 1 and bytes >= 1048576 order by start"):
    if t0 <= st <= t1:
        print(f"| {(st - t0) / 1e6:.1f} | {dev} | {b / 1e6:.1f} | {b / (en - st):.1f} | {sk} | {names.get(sk, '?')} |")
