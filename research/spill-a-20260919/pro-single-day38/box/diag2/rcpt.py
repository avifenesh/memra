import sqlite3, bisect, sys
for arm in ("x1", "x2"):
    db = sqlite3.connect(f"file:nsys-{arm}/trace.sqlite?mode=ro", uri=True); c = db.cursor()
    names = dict(c.execute("SELECT id,value FROM StringIds"))
    if arm == "x1":
        print(arm, [r[1] for r in c.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")])
    ker = c.execute("SELECT start,end,streamId,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    own = [(s, e) for s, e, sid, n in ker if sid == 13]; os_ = [s for s, e in own]
    rec = [(s, e, sid) for s, e, sid, n in ker if names.get(n) == "d2h_receipt_sha256"]
    rsid = rec[0][2]
    mem = c.execute("SELECT start,end,bytes,streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE streamId<>13 ORDER BY start").fetchall()
    def gap(a, b):
        i = bisect.bisect_left(os_, a); j = bisect.bisect_left(os_, b); g = 0; n = 0
        for k in range(max(i, 1), j):
            d = own[k][0] - own[k - 1][1]
            if d < 2000: g += d; n += 1
        return g / n if n else float("nan")
    for r, (s, e, sid) in enumerate(rec):
        lanes = [x for x in mem if x[3] == rsid and x[2] == 1024 and s < x[0] < e + 5e7]
        cps = [x for x in mem if not (x[3] == rsid and x[2] == 1024) and s - 2e7 < x[0] < e + 2e8]
        ld = (lanes[0][0] - e) / 1e3 if lanes else float("nan")
        c0 = (cps[0][0] - s) / 1e6 if cps else 0; c1 = (cps[-1][1] - s) / 1e6 if cps else 0
        print(f"{arm} r{r + 1:2d} k={(e - s) / 1e6:.2f}ms lanes_d2h={ld:7.1f}us copies n={len(cps)} [{c0:+.2f},{c1:+.2f}]ms gap_after={gap(e + 5e8, e + 2e9):.1f}ns")
