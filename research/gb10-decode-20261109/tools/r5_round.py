#!/usr/bin/env python3
"""r5_round.py prof.sqlite TTFT_S [ROUND_INDEX]: one decode round of the last request, as the device ran
it: kernels, device copies and memsets in start order, runs of the same op folded into one line
(count, total us, bytes), the round taken between two host gaps of 1 ms or more."""
import sqlite3, sys

db, ttft = sys.argv[1], float(sys.argv[2])
ri = int(sys.argv[3]) if len(sys.argv) > 3 else 5
con = sqlite3.connect(db)
rows = con.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                   "JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName) ORDER BY k.start").fetchall()
start = 0; prev = rows[0][1]
for i in range(1, len(rows)):
    if rows[i][0] - prev > 1e9:
        start = i
    prev = max(prev, rows[i][1])
t0 = rows[start][0] + int(ttft * 1e9)
ks = [(s, e, "K " + n.split("(")[0][:50], 0) for s, e, n in rows[start:] if s >= t0]
# round boundaries: kernel gaps of 1 ms or more
cuts = [ks[0][0]]
pe = ks[0][1]
for s, e, n, b in ks[1:]:
    if s - pe >= 1_000_000:
        cuts.append(s)
    pe = max(pe, e)
a, z = cuts[ri], cuts[ri + 1]
ev = [k for k in ks if a <= k[0] < z]
KIND = {1: "HtoD", 2: "DtoH", 8: "DtoD"}
ev += [(s, e, f"C {KIND.get(k, k)}", b) for s, e, b, k in con.execute(
    "SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start >= ? AND start < ?", (a, z))]
ev += [(s, e, "M memset", b) for s, e, b in con.execute(
    "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start >= ? AND start < ?", (a, z))]
ev.sort()
print(f"round {ri}: {(z - a) / 1e6:.2f} ms, {len(ev)} ops ({len(cuts) - 1} rounds in the window)")
out = []
for s, e, n, b in ev:
    if out and out[-1][0] == n:
        out[-1][1] += 1; out[-1][2] += (e - s) / 1e3; out[-1][3] += b; out[-1][5] = e
    else:
        out.append([n, 1, (e - s) / 1e3, b, s, e])
for n, c, us, b, s, e in out:
    print(f"  +{(s - a) / 1e3:9.1f} us  {c:4d} x {n:54s} {us:9.1f} us" + (f"  {b / 1e6:8.2f} MB" if b else ""))
