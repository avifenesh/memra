#!/usr/bin/env python3
"""gaps.py prof.sqlite: GPU idle gaps inside the prefill window, bucketed by size, with the
kernel that follows each large gap (where the host stalled)."""
import collections, re, sqlite3, sys

con = sqlite3.connect(sys.argv[1])
rows = con.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                   "JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName) ORDER BY k.start").fetchall()
segs, cur = [], [rows[0]]
for r in rows[1:]:
    if (r[0] - cur[-1][1]) / 1e9 > 2.0:
        segs.append(cur); cur = []
    cur.append(r)
segs.append(cur)
seg = max(segs, key=lambda s: sum(r[1] - r[0] for r in s))
buckets = collections.Counter(); bsum = collections.Counter(); after = collections.Counter(); asum = collections.Counter()
last = seg[0][1]
for s, e, n in seg[1:]:
    g = s - last
    if g > 0:
        b = "<10us" if g < 1e4 else "<100us" if g < 1e5 else "<1ms" if g < 1e6 else "<10ms" if g < 1e7 else ">=10ms"
        buckets[b] += 1; bsum[b] += g
        if g >= 1e6:
            k = re.sub(r"\(.*", "", re.sub(r"^void ", "", n))[:60]
            after[k] += 1; asum[k] += g
    last = max(last, e)
print("gap bucket      count     total_s")
for b in ["<10us", "<100us", "<1ms", "<10ms", ">=10ms"]:
    print(f"{b:10s} {buckets[b]:10d} {bsum[b]/1e9:11.3f}")
print("\nkernels that follow gaps >= 1 ms")
for k, v in sorted(asum.items(), key=lambda x: -x[1])[:15]:
    print(f"{v/1e9:9.3f} s {after[k]:6d} x  {k}")
