#!/usr/bin/env python3
"""Host time inside CUDA calls of an nsys trace, by call name: count, total ms, mean us, longest ms.
usage: host_calls.py TRACE.sqlite [TOP]"""
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
top = int(sys.argv[2]) if len(sys.argv) > 2 else 12
rows = db.execute(
    "select s.value, count(*), sum(r.end - r.start), max(r.end - r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r "
    "join StringIds s on s.id = r.nameId group by s.value order by 3 desc limit ?",
    (top,),
).fetchall()
span = db.execute("select min(start), max(end) from CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
print(f"trace span {(span[1] - span[0]) / 1e9:.2f} s")
print("| call | calls | total ms | mean us | longest ms |")
for name, n, tot, mx in rows:
    print(f"| {name} | {n} | {tot / 1e6:.1f} | {tot / n / 1e3:.1f} | {mx / 1e6:.2f} |")
