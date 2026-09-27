#!/usr/bin/env python3
"""WP-A day 38 section 13's trace reader, written before the traces exist. Input: an Nsight Systems sqlite export of one
traced demote boot, and its stall_cell receipt (for the run count). Markers: each launch of `d2h_receipt_sha256` is one
demote's receipt; the interval between two markers covers the next run's tenant decode. The owner stream is the stream
with the most kernels. Per interval: the owner's kernel count, its GPU-busy ms (the union of its kernels), its launch
gaps (idle ms between consecutive owner kernels), the mean duration of its five most frequent kernels (by short name),
and the busy ms of every other stream's kernels and copies inside the interval. Printed one line per interval, then the
same columns' first-three and peak intervals. A reading: nothing here is a clause. Usage: reader SQLITE RECEIPT"""
import json, sqlite3, statistics as st, sys
from collections import Counter, defaultdict

db = sqlite3.connect(sys.argv[1])
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))
ker = [(s, e, sid, names.get(n, str(n))) for s, e, sid, n in
       cur.execute("SELECT start, end, streamId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
try:
    mem = [(s, e, sid) for s, e, sid in cur.execute("SELECT start, end, streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY")]
except sqlite3.OperationalError:
    mem = []
if not ker:
    print("NSYS no kernels"); sys.exit(2)
per_stream = Counter(sid for _, _, sid, _ in ker)
owner = per_stream.most_common(1)[0][0]
marks = [s for s, _, _, n in ker if n == "d2h_receipt_sha256"]
receipt_stream = next((sid for _, _, sid, n in ker if n == "d2h_receipt_sha256"), None)
print(f"NSYS streams kernels={dict(per_stream)} owner={owner} receipt_stream={receipt_stream} markers={len(marks)} "
      f"copies={len(mem)}")
top = [n for n, _ in Counter(n for _, _, sid, n in ker if sid == owner).most_common(5)]
print("NSYS owner top kernels:", top)

def union(iv):
    iv = sorted(iv); tot = 0.0; cs = ce = None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None: tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None: tot += ce - cs
    return tot

rows = []
for k in range(len(marks) - 1):
    a, b = marks[k], marks[k + 1]
    ok = [(s, e, n) for s, e, sid, n in ker if sid == owner and a <= s < b]
    oth = [(s, e) for s, e, sid, _ in ker if sid != owner and a <= s < b] + [(s, e) for s, e, sid in mem if sid != owner and a <= s < b]
    if not ok:
        continue
    busy = union([(s, e) for s, e, _ in ok]) / 1e6
    gaps = sum(max(0, ok[i + 1][0] - ok[i][1]) for i in range(len(ok) - 1)) / 1e6
    means = defaultdict(list)
    for s, e, n in ok:
        if n in top: means[n].append((e - s) / 1e3)
    row = (k + 1, len(ok), busy, gaps, {n: st.mean(v) for n, v in means.items()}, union(oth) / 1e6, (b - a) / 1e6)
    rows.append(row)
    print(f"NSYS interval={row[0]} owner_kernels={row[1]} owner_busy_ms={row[2]:.2f} owner_gaps_ms={row[3]:.2f} "
          f"other_busy_ms={row[5]:.2f} span_ms={row[6]:.1f} top_mean_us=" + ",".join(f"{row[4].get(n, float('nan')):.1f}" for n in top))
if len(rows) >= 4:
    base = rows[:3]; pk = max(rows, key=lambda r: r[2] / max(r[1], 1))
    f = lambda r: f"owner_busy_per_kernel_us={r[2] / r[1] * 1e3:.2f} gaps_per_kernel_us={r[3] / r[1] * 1e3:.2f} other_busy_ms={r[5]:.2f}"
    print("NSYS first3 " + " | ".join(f(r) for r in base))
    print(f"NSYS peak interval={pk[0]} " + f(pk))
