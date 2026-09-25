#!/usr/bin/env python3
"""WP-A day 38 section 13e's fifth trace reader (exploratory, written after section 13d's reading): the owner stream's
kernel-to-kernel gap per 250 ms bin across the whole trace, with the receipt kernel markers and every other-stream
activity (kernels and memcpys) marked in their bins, so a step at a marker and a ramp between markers can be told apart.
Also: the pinned host allocations (`cuMemHostAlloc`) and frees (`cuMemFreeHost`) per bin, from any thread. Usage:
reader SQLITE"""
import sqlite3, sys
from collections import Counter, defaultdict

db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))
ker = cur.execute("SELECT start, end, streamId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
owner = Counter(k[2] for k in ker).most_common(1)[0][0]
t0 = ker[0][0]
BIN = 250_000_000
gap = defaultdict(float); nk = Counter(); marks = Counter(); other = defaultdict(float)
own_mem = sorted(s for (s,) in cur.execute(f"SELECT start FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE streamId = {owner}"))
import bisect
prev = None
for s, e, sid, n in ker:
    b = (s - t0) // BIN
    if sid != owner:
        other[b] += (e - s) / 1e6
        if names.get(n) == "d2h_receipt_sha256":
            marks[b] += 1
        continue
    if prev is not None:
        j = bisect.bisect_right(own_mem, prev)
        if not (j < len(own_mem) and own_mem[j] < s):
            gap[b] += max(0, s - prev) / 1e6
            nk[b] += 1
    prev = e
for s, e, sid in cur.execute("SELECT start, end, streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
    if sid != owner:
        other[(s - t0) // BIN] += (e - s) / 1e6
pin = Counter(); unpin = Counter()
for s, nid in cur.execute("SELECT start, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
    n = names.get(nid, "")
    if n.startswith("cuMemHostAlloc"):
        pin[(s - t0) // BIN] += 1
    elif n.startswith("cuMemFreeHost"):
        unpin[(s - t0) // BIN] += 1
for b in sorted(set(nk) | set(marks)):
    if nk[b] < 1000:
        continue
    print(f"NSYSLINE t={b * 0.25:7.2f}s kernels={nk[b]} gap_us_per_kernel={gap[b] / nk[b] * 1e3:.3f} other_ms={other[b]:.2f} "
          f"receipt_marker={marks[b]} host_alloc={pin[b]} host_free={unpin[b]}")
