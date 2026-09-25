#!/usr/bin/env python3
"""WP-A day 38 section 13d's fourth trace reader, written before it is run on the two banked traces. Input: an Nsight
Systems sqlite export (read-only). Markers and the owner stream as day38-nsys-reading.py; the owner thread as
day38-nsys-api-reading.py. For each owner-stream kernel after the first of an interval: its GAP (its start minus the
previous owner kernel's end, only when no owner memcpy lies between them), the kernel's short name, and WAITS, the
number of `cuStreamWaitEvent` calls the owner thread made between the previous kernel's launch call and this kernel's
launch call (by correlationId). Per interval: the gaps' sum and count by size (<2, 2-10, 10-50, >=50 us), the sums for
kernels with WAITS=0 and WAITS>0 (and their counts), and the ten kernel names whose gap sum grows most from the median of
the first three full intervals to the full interval with the most gap. A reading: nothing here is a clause.
Usage: reader SQLITE"""
import sqlite3, statistics as st, sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))
ker = cur.execute("SELECT start, end, streamId, shortName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
owner = Counter(k[2] for k in ker).most_common(1)[0][0]
marks = [k[0] for k in ker if names.get(k[3]) == "d2h_receipt_sha256"]
K = len(marks) - 1
def slot(t):
    i = bisect_right(marks, t) - 1
    return i if 0 <= i < K else None
own = [k for k in ker if k[2] == owner]
del ker
mem_starts = sorted(s for (s,) in cur.execute(f"SELECT start FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE streamId = {owner}"))
try:
    mem_starts += sorted(s for (s,) in cur.execute(f"SELECT start FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE streamId = {owner}"))
    mem_starts.sort()
except sqlite3.OperationalError:
    pass
wid = [i for i, v in names.items() if v.startswith("cuStreamWaitEvent")]
sample = {k[4] for k in own[:: max(1, len(own) // 5000)]}
tid_count = Counter()
launch_at = {}
waits = []
for s, tid, cid, nid in cur.execute("SELECT start, globalTid, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
    if cid in sample:
        tid_count[tid] += 1
owner_tid = tid_count.most_common(1)[0][0]
for s, cid, nid in cur.execute("SELECT start, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid = ?", (owner_tid,)):
    if nid in wid:
        waits.append(s)
    else:
        launch_at[cid] = s
waits.sort()
print(f"NSYSGAP owner={owner} owner_tid={owner_tid} markers={len(marks)} owner_kernels={len(own)} waits={len(waits)} "
      f"wait_names={[names[i] for i in wid]}")
B = ("<2us", "2-10us", "10-50us", ">=50us")
def bucket(g):
    return B[0] if g < 2 else B[1] if g < 10 else B[2] if g < 50 else B[3]
rows = [dict(sum=0.0, n=Counter(), s=defaultdict(float), w0=[0, 0.0], w1=[0, 0.0], by=defaultdict(float)) for _ in range(K)]
for i in range(1, len(own)):
    s, e, _, nid, cid = own[i]
    ps, pe, _, _, pcid = own[i - 1]
    k = slot(s)
    if k is None or slot(ps) != k:
        continue
    j = bisect_right(mem_starts, pe)
    if j < len(mem_starts) and mem_starts[j] < s:
        continue
    g = max(0, s - pe) / 1e3
    r = rows[k]
    r["sum"] += g; b = bucket(g); r["n"][b] += 1; r["s"][b] += g
    la, pla = launch_at.get(cid), launch_at.get(pcid)
    nw = (bisect_left(waits, la) - bisect_right(waits, pla)) if la is not None and pla is not None else 0
    t = r["w1"] if nw > 0 else r["w0"]
    t[0] += 1; t[1] += g
    r["by"][names.get(nid, str(nid))] += g
for k, r in enumerate(rows):
    print(f"NSYSGAP interval={k + 1} gap_ms={r['sum'] / 1e3:.2f} " +
          " ".join(f"{b}:n={r['n'][b]},ms={r['s'][b] / 1e3:.2f}" for b in B) +
          f" | waits=0 n={r['w0'][0]} ms={r['w0'][1] / 1e3:.2f} | waits>0 n={r['w1'][0]} ms={r['w1'][1] / 1e3:.2f}")
spans = [(marks[k + 1] - marks[k]) for k in range(K)]
full = [k for k in range(K) if spans[k] > 0.75 * st.median(spans)]
base = full[:3]; peak = max(full, key=lambda k: rows[k]["sum"])
keys = set().union(*(rows[k]["by"].keys() for k in base + [peak]))
g = sorted(((rows[peak]["by"].get(n, 0) - st.median([rows[k]["by"].get(n, 0) for k in base]), n) for n in keys), reverse=True)
print(f"NSYSGAP base={[k + 1 for k in base]} peak={peak + 1}")
for d, n in g[:10]:
    print(f"NSYSGAP growth kernel={n} base_ms={st.median([rows[k]['by'].get(n, 0) for k in base]) / 1e3:.2f} "
          f"peak_ms={rows[peak]['by'].get(n, 0) / 1e3:.2f} delta_ms={d / 1e3:+.2f}")
