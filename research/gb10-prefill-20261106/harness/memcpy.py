#!/usr/bin/env python3
"""memcpy.py prof.sqlite: memcpy/memset activity inside the prefill window (longest kernel segment)."""
import collections, sqlite3, sys

con = sqlite3.connect(sys.argv[1])
rows = con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
segs, cur = [], [rows[0]]
for r in rows[1:]:
    if (r[0] - cur[-1][1]) / 1e9 > 2.0:
        segs.append(cur); cur = []
    cur.append(r)
segs.append(cur)
seg = max(segs, key=lambda s: sum(r[1] - r[0] for r in s))
t0, t1 = seg[0][0], max(r[1] for r in seg)
kinds = {r[0]: r[1] for r in con.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER")}
out = collections.defaultdict(lambda: [0, 0, 0])
for s, e, k, b in con.execute("SELECT start, end, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start >= ? AND end <= ?", (t0, t1)):
    o = out[kinds.get(k, k)]; o[0] += 1; o[1] += e - s; o[2] += b
print("memcpy kind          count     total_s        GB    GB/s")
for k, (c, d, b) in sorted(out.items(), key=lambda x: -x[1][1]):
    print(f"{str(k):18s} {c:8d} {d/1e9:11.3f} {b/1e9:9.2f} {b/max(d,1):7.1f}")
try:
    ms = con.execute("SELECT count(*), sum(end-start), sum(bytes) FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start >= ? AND end <= ?", (t0, t1)).fetchone()
    print(f"memset {ms[0]} x, {(ms[1] or 0)/1e9:.3f} s, {(ms[2] or 0)/1e9:.2f} GB")
except sqlite3.OperationalError:
    pass
