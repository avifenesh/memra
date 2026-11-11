#!/usr/bin/env python3
"""cuMemAllocAsync calls of 0.5 ms and more in the trace, relative to the first 4k-row mHC pass (the 4k prompt's
window start), with the thread, and the trace's memory-usage tables if any: is a slow allocation the pool growing past
its high-water mark (first use of a size) or something that recurs. usage: alloc_slow.py TRACE.sqlite"""
import sqlite3
import sys
from collections import defaultdict

con = sqlite3.connect(sys.argv[1])
cur = con.cursor()
S = dict(cur.execute("select id, value from StringIds"))
tabs = [r[0] for r in cur.execute("select name from sqlite_master where type = 'table'")]
print("memory tables:", [t for t in tabs if "MEM" in t.upper() and "ENUM" not in t.upper()])
kern = cur.execute("select start, shortName, gridX from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
t0 = next(k[0] for k in kern if "hc_mixes_part" in S.get(k[1], "") and "part1" not in S.get(k[1], "") and k[2] == 256)
ids = {v: k for k, v in S.items()}
aid = [k for k, v in S.items() if v.startswith("cuMemAllocAsync")]
fid = [k for k, v in S.items() if v.startswith("cuMemFreeAsync")]
rows = cur.execute(f"select start, end, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where nameId in ({','.join(map(str, aid))}) "
                   "order by start").fetchall()
tot = defaultdict(lambda: [0, 0])
for st, en, tid in rows:
    b = "before" if st < t0 else ("4k" if st < t0 + 4_034_800_000 else "32k and after")
    tot[b][0] += 1
    tot[b][1] += en - st
print("cuMemAllocAsync by phase: " + "; ".join(f"{k} {c} calls {ns / 1e6:.1f} ms" for k, (c, ns) in tot.items()))
nf = cur.execute(f"select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME where nameId in ({','.join(map(str, fid))})").fetchone()[0]
print(f"cuMemFreeAsync calls in the trace: {nf}")
print("| at ms (from the 4k window start) | ms | thread |")
print("|---|---|---|")
for st, en, tid in rows:
    if en - st >= 500_000:
        print(f"| {(st - t0) / 1e6:+.1f} | {(en - st) / 1e6:.2f} | {tid & 0xFFFFFF} |")
for t in tabs:
    if "MEMORY_USAGE" in t.upper() or "MEMPOOL" in t.upper() or "MEMORY_POOL" in t.upper():
        print(t, cur.execute(f"select count(*) from {t}").fetchone()[0], "rows; columns:",
              [r[1] for r in cur.execute(f"pragma table_info({t})")])
