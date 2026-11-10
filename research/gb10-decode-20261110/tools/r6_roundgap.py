#!/usr/bin/env python3
"""r6_roundgap.py prof.sqlite TTFT_S: the gap between one round's last kernel and the next round's first
(kernel gaps of 1 ms or more in the decode window of the last request, as r5_round.py cuts rounds): per
gap its length, the device copies and memsets inside it, and the remainder (host). Prints the median
and the mean of each over the gaps, leaving out the gap that holds the prefix-cache publication (a copy
of 64 MB or more)."""
import sqlite3, statistics, sys
db, ttft = sys.argv[1], float(sys.argv[2])
con = sqlite3.connect(db)
rows = con.execute("SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k ORDER BY k.start").fetchall()
start = 0; prev = rows[0][1]
for i in range(1, len(rows)):
    if rows[i][0] - prev > 1e9:
        start = i
    prev = max(prev, rows[i][1])
t0 = rows[start][0] + int(ttft * 1e9)
win = [(s, e) for s, e in rows[start:] if s >= t0]
ops = con.execute("SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start >= ? UNION ALL "
                  "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start >= ?", (t0, t0)).fetchall()
gaps = []
pe = win[0][1]
for s, e in win[1:]:
    if s - pe >= 1e6:
        inside = [(a, b, n) for a, b, n in ops if a >= pe and b <= s]
        gaps.append(((s - pe) / 1e6, sum(b - a for a, b, n in inside) / 1e6, max([n for a, b, n in inside] or [0])))
    pe = max(pe, e)
steady = [g for g in gaps if g[2] < 64e6]
print(f"{len(gaps)} round gaps, {len(gaps) - len(steady)} with the prefix-cache publication")
for g in gaps:
    if g[2] >= 64e6:
        print(f"  the publication's gap: {g[0]:.1f} ms, {g[1]:.1f} ms of it device copies and memsets")
for name, f in (("gap", lambda g: g[0]), ("copies and memsets inside", lambda g: g[1]), ("host remainder", lambda g: g[0] - g[1])):
    v = [f(g) for g in steady]
    print(f"  {name:28s} median {statistics.median(v):6.2f} ms  mean {statistics.fmean(v):6.2f} ms  min {min(v):6.2f}  max {max(v):6.2f}")
