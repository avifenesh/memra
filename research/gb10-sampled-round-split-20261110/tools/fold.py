#!/usr/bin/env python3
"""fold.py prof.sqlite REQUEST FROM_MS TO_MS: the device intervals of one request of the boot, in
start order, between two offsets from the request's first kernel: kernels (K), device copies (C)
and memsets (M), runs of the same op folded into one line (count, total us, MB), and the idle
before a line when it is over 20 us. REQUEST counts from the end: 6 requests in a boot, so the
warm-up is -6 and the closing greedy control is -1 (the driver's order: warm, greedy, t1, chat,
pen, greedy-b)."""
import sqlite3
import sys

db, req, f_ms, t_ms = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
se = con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
segs = []
a = se[0][0]
prev = se[0][1]
for s, e in se[1:]:
    if s - prev > 1e9:
        segs.append(a)
        a = s
    prev = max(prev, e)
segs.append(a)
t0 = segs[req]
f, t = t0 + int(f_ms * 1e6), t0 + int(t_ms * 1e6)
KIND = {1: "HtoD", 2: "DtoH", 8: "DtoD"}
ops = [(s, e, "K " + n.split("(")[0].split("<")[0].replace("void ", ""), 0) for s, e, n in con.execute(
    "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName) "
    "WHERE k.start >= ? AND k.start < ?", (f, t))]
ops += [(s, e, "C " + KIND.get(k, str(k)), by) for s, e, by, k in con.execute(
    "SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start >= ? AND start < ?", (f, t))]
ops += [(s, e, "M memset", by) for s, e, by in con.execute(
    "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start >= ? AND start < ?", (f, t))]
ops.sort()
out = []
for s, e, n, by in ops:
    if out and out[-1][0] == n:
        out[-1][1] += 1
        out[-1][2] += (e - s) / 1e3
        out[-1][3] += by
        out[-1][5] = e
    else:
        out.append([n, 1, (e - s) / 1e3, by, s, e])
print(f"request {req}, +{f_ms} ms to +{t_ms} ms from its first kernel: {len(ops)} intervals")
pe = None
for n, c, us, by, s, e in out:
    gap = f"  idle before {(s - pe) / 1e3:.1f} us" if pe is not None and s - pe > 20e3 else ""
    mb = f" {by / 1e6:8.3f} MB" if by else ""
    print(f"+{(s - t0) / 1e6:10.3f} ms {c:5d} x {n:44s} {us:9.1f} us{mb}{gap}")
    pe = e
