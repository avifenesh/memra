#!/usr/bin/env python3
"""r5_gaps.py prof.sqlite TTFT_S: the decode window of the last request (as r5_decode.py): device copies
by kind and size bucket, and every idle gap of 100 us or more between kernels with the kernels on
either side and what copies or runtime calls fall inside it (counts by neighbour pair)."""
import collections, sqlite3, sys

db, ttft = sys.argv[1], float(sys.argv[2])
con = sqlite3.connect(db)
rows = con.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                   "JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName) ORDER BY k.start").fetchall()
start = 0; prev = rows[0][1]
for i in range(1, len(rows)):
    if rows[i][0] - prev > 1e9:
        start = i
    prev = max(prev, rows[i][1])
t0 = rows[start][0] + int(ttft * 1e9)
t_end = max(r[1] for r in rows[start:])
win = [(s, e, n.split("(")[0][:44]) for s, e, n in rows[start:] if s >= t0]
KIND = {1: "HtoD", 2: "DtoH", 3: "HtoA", 4: "AtoH", 5: "AtoA", 6: "AtoD", 7: "DtoA", 8: "DtoD", 9: "HtoH", 10: "PtoP"}
mc = con.execute("SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start >= ? AND end <= ? ORDER BY start", (t0, t_end)).fetchall()
by = collections.defaultdict(lambda: [0, 0.0, 0])
for s, e, b, k in mc:
    bucket = "<4K" if b < 4096 else "<1M" if b < 1 << 20 else "<16M" if b < 16 << 20 else ">=16M"
    x = by[(KIND.get(k, k), bucket)]; x[0] += 1; x[1] += (e - s) / 1e6; x[2] += b
print("copies by kind and size: count, ms, MB")
for k, (c, ms, b) in sorted(by.items()):
    print(f"  {k[0]:5s} {k[1]:6s} {c:7d} {ms:8.1f} ms {b / 1e6:9.1f} MB")
ms_ = con.execute("SELECT COUNT(*), SUM(end - start), SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start >= ? AND end <= ?", (t0, t_end)).fetchone()
print(f"memsets: {ms_[0]}, {(ms_[1] or 0) / 1e6:.1f} ms, {(ms_[2] or 0) / 1e6:.1f} MB")
pairs = collections.defaultdict(lambda: [0, 0.0])
pe, pn = None, None
for s, e, n in win:
    if pe is not None and s - pe >= 100_000:
        x = pairs[(pn, n)]; x[0] += 1; x[1] += (s - pe) / 1e6
    if pe is None or e > pe:
        pe, pn = e, n
print("gaps >= 100 us by (kernel before -> kernel after): count, total ms")
for (a, b), (c, ms) in sorted(pairs.items(), key=lambda kv: -kv[1][1])[:16]:
    print(f"  {c:5d} {ms:8.1f} ms  {a} -> {b}")
