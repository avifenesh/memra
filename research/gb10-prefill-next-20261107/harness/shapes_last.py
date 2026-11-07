#!/usr/bin/env python3
"""shapes_last.py: launch shapes (grid x block) and time per kernel name in the last served prefill
of an nsys sqlite export (the parts_last.py window).

  shapes_last.py prof.sqlite TTFT_S NAME_SUBSTR [NAME_SUBSTR ...]
"""
import collections, sqlite3, sys

db, ttft, names = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
con = sqlite3.connect(db)
rows = con.execute("""SELECT k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ
                      FROM CUPTI_ACTIVITY_KIND_KERNEL k
                      JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName)
                      ORDER BY k.start""").fetchall()
t1 = max(e for s, e, n, *_ in rows if n.startswith("fa_prefill_qw"))
t0 = t1 - int(ttft * 1e9)
agg = collections.defaultdict(lambda: [0, 0.0])
for s, e, n, gx, gy, gz, bx, by, bz in rows:
    if not (t0 <= s <= t1):
        continue
    short = n.split("(")[0]
    if not any(x in short for x in names):
        continue
    k = (short[:48], f"{gx}x{gy}x{gz}", f"{bx}x{by}x{bz}")
    agg[k][0] += 1
    agg[k][1] += (e - s) / 1e9
for (n, g, b), (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print(f"{t:8.3f} s {c:6d} x {1000 * t / c:8.3f} ms  grid {g:>14s} block {b:>9s}  {n}")
