#!/usr/bin/env python3
"""Summarize one nsys sqlite export around each promote: the anchor is the probe's submission (d33t: the one
cuLaunchHostFunc per promote; d32: the first span-sized H2D memcpy burst on the copy stream). For each anchor: the owner
thread's CUDA API calls longer than 0.3 ms in the next 25 ms (name, start offset, duration), and the GPU work in that
window by stream (kernels summed, memcpys by kind and bytes, first start and last end offsets)."""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
tag = sys.argv[2]
names = dict(db.execute("select id, value from StringIds"))
api = db.execute("select start, end, globalTid, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start").fetchall()
def nm(i): return names.get(i, str(i))
mem = db.execute("select start, end, streamId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
try:
    ker = db.execute("select start, end, streamId, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
except Exception:
    ker = []
# owner thread = the thread with the most cuLaunchKernel calls
tcount = collections.Counter(t for s, e, t, n in api if nm(n).startswith("cuLaunchKernel"))
owner = tcount.most_common(1)[0][0] if tcount else None
if tag.startswith("d33"):
    anchors = [s for s, e, t, n in api if nm(n).startswith("cuLaunchHostFunc")]
else:
    # the day-32 span burst: >= 20 H2D copies of >= 256 KiB within 5 ms on one stream
    anchors, last = [], -10**18
    h2d = [m for m in mem if m[4] == 1 and m[3] >= 256 * 1024]
    for i, m in enumerate(h2d):
        burst = [x for x in h2d[i:i + 48] if x[2] == m[2] and x[0] - m[0] < 5_000_000]
        if len(burst) >= 20 and m[0] - last > 50_000_000:
            anchors.append(m[0]); last = m[0]
print(f"{tag}: owner thread tid={owner}; anchors={len(anchors)}")
W = 25_000_000
for k, a in enumerate(anchors[:6]):
    print(f"-- anchor {k}")
    for s, e, t, n in api:
        if t == owner and a - 2_000_000 <= s <= a + W and e - s > 300_000:
            print(f"   owner api {nm(n):28s} start {(s - a) / 1e6:+8.3f} ms dur {(e - s) / 1e6:7.3f} ms")
    by = collections.defaultdict(lambda: [0, 0.0, None, None])
    for s, e, st, n in ker:
        if a <= s <= a + W:
            b = by[("kernel", st)]; b[0] += 1; b[1] += (e - s) / 1e6
            b[2] = s if b[2] is None else min(b[2], s); b[3] = e if b[3] is None else max(b[3], e)
    for s, e, st, byt, kind in mem:
        if a <= s <= a + W:
            b = by[({1: "H2D", 2: "D2H", 8: "D2D"}.get(kind, str(kind)), st)]; b[0] += 1; b[1] += byt / 1e6
            b[2] = s if b[2] is None else min(b[2], s); b[3] = e if b[3] is None else max(b[3], e)
    for (what, st), (cnt, tot, s0, e0) in sorted(by.items(), key=lambda kv: kv[1][2]):
        unit = "ms busy" if what == "kernel" else "MB"
        print(f"   gpu {what:6s} stream {st:4d} n={cnt:4d} {tot:9.2f} {unit} first {(s0 - a) / 1e6:+8.3f} ms last end {(e0 - a) / 1e6:+8.3f} ms")
