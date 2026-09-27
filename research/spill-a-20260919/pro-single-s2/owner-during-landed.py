import sqlite3, sys, statistics as st
db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True); c = db.cursor()
names = dict(c.execute("SELECT id, value FROM StringIds"))
ker = c.execute("SELECT start, end, streamId, shortName, gridX, gridY FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
sha = [k for k in ker if names.get(k[3]) == "d2h_receipt_sha256"]
copy = sha[0][2]
spans = [k for k in ker if names.get(k[3]) == "span_receipt_digests"]
print("span_receipt_digests launches:", len(spans))
for k in spans[:6]:
    print("  dur_ms=%.3f grid=%sx%s" % ((k[1]-k[0])/1e6, k[4], k[5]))
own = [k for k in ker if k[2] != copy]
# owner kernel time in windows: during each landed launch vs a same-length window 20 ms before
def busy(a, b):
    tot = 0
    for k in own:
        s, e = max(k[0], a), min(k[1], b)
        if e > s: tot += e - s
    n = sum(1 for k in own if k[0] >= a and k[0] < b)
    return tot/1e6, n
land = [k for k in spans if (k[1]-k[0]) > 5e5]
for k in land[:8]:
    a, b = k[0], k[1]
    d = b - a
    print("landed dur=%.2f owner busy in=%.2fms n=%d | before busy=%.2fms n=%d" % (d/1e6, *busy(a,b), *busy(a-d-2e6, a-2e6)))
# owner kernel durations by name: inside landed windows vs outside
inside, outside = {}, {}
wins = [(k[0], k[1]) for k in land]
for k in own:
    nm = names.get(k[3]); dur = (k[1]-k[0])/1e3
    tgt = inside if any(a <= k[0] < b for a, b in wins) else outside
    tgt.setdefault(nm, []).append(dur)
rows = []
for nm, xs in inside.items():
    if nm in outside and len(xs) >= 5 and len(outside[nm]) >= 20:
        rows.append((st.median(xs) - st.median(outside[nm]), nm, st.median(xs), st.median(outside[nm]), len(xs)))
rows.sort(reverse=True)
for r in rows[:10]:
    print("  %s in=%.1fus out=%.1fus delta=%+.1fus n_in=%d" % (r[1][:40], r[2], r[3], r[0], r[4]))
