#!/usr/bin/env python3
"""WP-A day 38 section 13c's third trace reader, written before it is run on the two banked traces. Input: an Nsight
Systems sqlite export (read-only). Markers and the owner stream as day38-nsys-reading.py. Per interval, for the owner
stream's memcpys (every kind): the count and GPU ms by (copyKind, srcKind, dstKind, bytes bucket); for each owner-stream
memcpy, its LEAD (its start minus the end of the owner stream's previous activity, kernel or memcpy) and its TAIL (the
next owner kernel's start minus its end); the median and p90 of each, and their sums. Also, per interval, every other
stream's memcpys by the same key, and how many owner memcpys overlap a memcpy of another stream in time. For the owner
thread's `cuMemcpyDtoHAsync_v2` calls: the median and p90 API duration. A reading: nothing here is a clause.
Usage: reader SQLITE"""
import sqlite3, statistics as st, sys
from bisect import bisect_right
from collections import Counter, defaultdict

db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))
ker = cur.execute("SELECT start, end, streamId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
owner = Counter(k[2] for k in ker).most_common(1)[0][0]
marks = [k[0] for k in ker if names.get(k[3]) == "d2h_receipt_sha256"]
K = len(marks) - 1
def slot(t):
    i = bisect_right(marks, t) - 1
    return i if 0 <= i < K else None
cols = [r[1] for r in cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")]
want = [c for c in ("start", "end", "streamId", "copyKind", "srcKind", "dstKind", "bytes", "correlationId") if c in cols]
mem = [dict(zip(want, r)) for r in cur.execute(f"SELECT {', '.join(want)} FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start")]
enum = {}
for t in ("ENUM_CUDA_MEMCPY_OPER", "ENUM_CUDA_MEM_KIND"):
    try:
        enum[t] = dict(cur.execute(f"SELECT id, name FROM {t}"))
    except sqlite3.OperationalError:
        enum[t] = {}
def key(m):
    b = m.get("bytes", 0)
    bucket = "<=64B" if b <= 64 else "<=4KiB" if b <= 4096 else "<=1MiB" if b <= 1 << 20 else ">1MiB"
    return (enum["ENUM_CUDA_MEMCPY_OPER"].get(m.get("copyKind"), m.get("copyKind")),
            enum["ENUM_CUDA_MEM_KIND"].get(m.get("srcKind"), m.get("srcKind")),
            enum["ENUM_CUDA_MEM_KIND"].get(m.get("dstKind"), m.get("dstKind")), bucket)
print(f"NSYSCOPY owner={owner} markers={len(marks)} memcpys={len(mem)} columns={want}")
# owner stream timeline: kernels and memcpys merged
ev = sorted([(s, e, "k", None) for s, e, sid, _ in ker if sid == owner] +
            [(m["start"], m["end"], "m", m) for m in mem if m["streamId"] == owner])
others = sorted((m["start"], m["end"]) for m in mem if m["streamId"] != owner)
ostarts = [o[0] for o in others]
lead = [[] for _ in range(K)]; tail = [[] for _ in range(K)]; kinds = [Counter() for _ in range(K)]
gpu = [defaultdict(float) for _ in range(K)]; overlap = [0] * K
for i, (s, e, typ, m) in enumerate(ev):
    if typ != "m":
        continue
    k = slot(s)
    if k is None:
        continue
    kk = key(m); kinds[k][kk] += 1; gpu[k][kk] += (e - s) / 1e6
    if i > 0:
        lead[k].append((s - ev[i - 1][1]) / 1e3)
    nxt = next((x for x in ev[i + 1:i + 4] if x[2] == "k"), None)
    if nxt:
        tail[k].append((nxt[0] - e) / 1e3)
    j = bisect_right(ostarts, e)
    if any(o[1] > s for o in others[max(0, j - 50):j]):
        overlap[k] += 1
oth = [Counter() for _ in range(K)]; oth_ms = [defaultdict(float) for _ in range(K)]
for m in mem:
    if m["streamId"] == owner:
        continue
    k = slot(m["start"])
    if k is not None:
        kk = (m["streamId"],) + key(m); oth[k][kk] += 1; oth_ms[k][kk] += (m["end"] - m["start"]) / 1e6
def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")
for k in range(K):
    print(f"NSYSCOPY interval={k + 1} owner_memcpys={sum(kinds[k].values())} lead_us median={q(lead[k], .5):.1f} p90={q(lead[k], .9):.1f} "
          f"sum_ms={sum(lead[k]) / 1e3:.2f} | tail_us median={q(tail[k], .5):.1f} p90={q(tail[k], .9):.1f} sum_ms={sum(tail[k]) / 1e3:.2f} | "
          f"overlapping_other_copies={overlap[k]} | by kind " +
          "; ".join(f"{kk}: n={n} gpu_ms={gpu[k][kk]:.2f}" for kk, n in kinds[k].most_common(4)))
    print(f"NSYSCOPY interval={k + 1} other-stream copies " +
          "; ".join(f"{kk}: n={n} ms={oth_ms[k][kk]:.2f}" for kk, n in oth[k].most_common(4)))
# the owner thread's D2H API durations
rt = cur.execute("SELECT r.start, r.end, r.globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME r WHERE r.nameId IN "
                 "(SELECT id FROM StringIds WHERE value = 'cuMemcpyDtoHAsync_v2')").fetchall()
tid = Counter(t for _, _, t in rt).most_common(1)[0][0] if rt else None
per = [[] for _ in range(K)]
for s, e, t in rt:
    k = slot(s)
    if t == tid and k is not None:
        per[k].append((e - s) / 1e3)
for k in range(K):
    print(f"NSYSCOPY interval={k + 1} owner cuMemcpyDtoHAsync_v2 calls={len(per[k])} api_us median={q(per[k], .5):.1f} p90={q(per[k], .9):.1f}")
