#!/usr/bin/env python3
"""r6_big.py prof.sqlite TTFT_S [MIN_MB]: in the decode window of the last request (as r5_decode.py), every
device copy and memset of MIN_MB or more (default 8), folded by size: count, total ms, total MB, and the
kernel that ended last before it and the one that starts next after it (counts by neighbour pair)."""
import collections, sqlite3, sys, bisect
db, ttft = sys.argv[1], float(sys.argv[2])
min_mb = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
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
starts = [w[0] for w in win]
out = collections.defaultdict(lambda: [0, 0.0, 0.0, collections.Counter()])
for kind, q in (("copy", "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start >= ? AND end <= ?"),
                ("memset", "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start >= ? AND end <= ?")):
    for s, e, b in con.execute(q, (t0, t_end)):
        if b < min_mb * 1e6:
            continue
        i = bisect.bisect_left(starts, s)
        before = win[i - 1][2] if i > 0 else "-"
        after = win[i][2] if i < len(win) else "-"
        o = out[(kind, round(b / 1e6, 1))]
        o[0] += 1; o[1] += (e - s) / 1e6; o[2] += b / 1e6; o[3][(before, after)] += 1
for (kind, mb), (n, ms, tot, nb) in sorted(out.items()):
    print(f"{kind:7s} {mb:9.1f} MB  x{n:5d}  {ms:8.1f} ms  {tot:9.1f} MB")
    for (b, a), c in nb.most_common(3):
        print(f"            {c:5d}  {b} -> {a}")
