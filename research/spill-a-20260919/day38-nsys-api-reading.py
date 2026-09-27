#!/usr/bin/env python3
"""WP-A day 38 section 13b's second trace reader, written before it is run on the two banked traces (section 13's X1 and
X2). Input: an Nsight Systems sqlite export (opened read-only). Markers as day38-nsys-reading.py: each launch of
`d2h_receipt_sha256` starts one interval. The owner stream is the stream with the most kernels; the OWNER THREAD is the
thread whose runtime calls launched most of the owner stream's kernels (by correlationId over a sample of 5000). Per
interval, for the owner thread: its CUDA API calls by name (count, total ms), its OS runtime calls by name (count, total
ms), and its OFF-API ms (the time between consecutive CUDA API calls, a stand-in for its own host work). Printed: one line
per interval with the totals, then the ten API and the ten OS runtime names whose total ms grows most from the median of
the first three intervals to the interval with the most owner launch gaps (day38-nsys-reading.py's column, recomputed
here), and the same growth table for every other thread with CUDA calls, summed per thread name. A reading: nothing here
is a clause. Usage: reader SQLITE"""
import sqlite3, statistics as st, sys
from bisect import bisect_right
from collections import Counter, defaultdict

db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))
tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
ker = cur.execute("SELECT start, end, streamId, shortName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
per_stream = Counter(k[2] for k in ker)
owner = per_stream.most_common(1)[0][0]
marks = [k[0] for k in ker if names.get(k[3]) == "d2h_receipt_sha256"]
if len(marks) < 5:
    print(f"NSYSAPI markers={len(marks)}: too few"); sys.exit(2)
own = [k for k in ker if k[2] == owner]
sample = {k[4] for k in own[:: max(1, len(own) // 5000)]}
tid_count = Counter()
for (cid, tid) in cur.execute("SELECT correlationId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
    if cid in sample:
        tid_count[tid] += 1
owner_tid = tid_count.most_common(1)[0][0]
tname = {}
if "ThreadNames" in tables:
    for nid, tid in cur.execute("SELECT nameId, globalTid FROM ThreadNames"):
        tname[tid] = names.get(nid, str(nid))
print(f"NSYSAPI owner_stream={owner} owner_tid={owner_tid} ({tname.get(owner_tid, '?')}) sample_hits={dict(tid_count.most_common(3))} "
      f"markers={len(marks)}")
K = len(marks) - 1
def slot(t):
    i = bisect_right(marks, t) - 1
    return i if 0 <= i < K else None
# owner launch gaps per interval (as the first reader)
gaps = [0.0] * K
prev_end = None; prev_slot = None
for s, e, *_ in own:
    k = slot(s)
    if k is not None and prev_slot == k and prev_end is not None:
        gaps[k] += max(0, s - prev_end) / 1e6
    prev_end, prev_slot = e, k
api = [defaultdict(lambda: [0, 0.0]) for _ in range(K)]
off_api = [0.0] * K
other = defaultdict(lambda: [defaultdict(lambda: [0, 0.0]) for _ in range(K)])
last_end = {}
for s, e, tid, nid in cur.execute("SELECT start, end, globalTid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"):
    k = slot(s)
    if k is None:
        last_end[tid] = e
        continue
    n = names.get(nid, str(nid))
    if tid == owner_tid:
        a = api[k][n]; a[0] += 1; a[1] += (e - s) / 1e6
        if tid in last_end:
            off_api[k] += max(0, s - last_end[tid]) / 1e6
    else:
        a = other[tname.get(tid, str(tid))][k][n]; a[0] += 1; a[1] += (e - s) / 1e6
    last_end[tid] = e
osrt = [defaultdict(lambda: [0, 0.0]) for _ in range(K)]
if "OSRT_API" in tables:
    for s, e, nid in cur.execute("SELECT start, end, nameId FROM OSRT_API WHERE globalTid = ?", (owner_tid,)):
        k = slot(s)
        if k is None:
            continue
        a = osrt[k][names.get(nid, str(nid))]; a[0] += 1; a[1] += (e - s) / 1e6
for k in range(K):
    print(f"NSYSAPI interval={k + 1} owner_gaps_ms={gaps[k]:.2f} owner_api_calls={sum(v[0] for v in api[k].values())} "
          f"owner_api_ms={sum(v[1] for v in api[k].values()):.2f} owner_off_api_ms={off_api[k]:.2f} "
          f"owner_osrt_calls={sum(v[0] for v in osrt[k].values())} owner_osrt_ms={sum(v[1] for v in osrt[k].values()):.2f} "
          f"span_ms={(marks[k + 1] - marks[k]) / 1e6:.1f}")
full = [k for k in range(K) if (marks[k + 1] - marks[k]) / 1e6 > 0.75 * st.median([(marks[i + 1] - marks[i]) / 1e6 for i in range(K)])]
base = full[:3]
peak = max(full, key=lambda k: gaps[k])
def growth(tab, label):
    keys = set().union(*(tab[k].keys() for k in base + [peak]))
    rows = []
    for n in keys:
        b_ms = st.median([tab[k][n][1] if n in tab[k] else 0.0 for k in base])
        b_ct = st.median([tab[k][n][0] if n in tab[k] else 0 for k in base])
        p = tab[peak][n] if n in tab[peak] else [0, 0.0]
        rows.append((p[1] - b_ms, n, b_ct, p[0], b_ms, p[1]))
    rows.sort(reverse=True)
    for d, n, bc, pc, bm, pm in rows[:10]:
        print(f"NSYSAPI growth {label} name={n} base(first3 median) calls={bc} ms={bm:.2f} -> peak(interval {peak + 1}) "
              f"calls={pc} ms={pm:.2f} delta_ms={d:+.2f}")
print(f"NSYSAPI base intervals={[k + 1 for k in base]} peak interval={peak + 1} (the most owner launch gaps among full intervals)")
growth(api, "owner-cuda")
growth(osrt, "owner-osrt")
print(f"NSYSAPI owner off-api ms base median={st.median([off_api[k] for k in base]):.2f} peak={off_api[peak]:.2f}")
for name, tab in sorted(other.items()):
    tot_b = st.median([sum(v[1] for v in tab[k].values()) for k in base])
    tot_p = sum(v[1] for v in tab[peak].values())
    print(f"NSYSAPI other-thread name={name} api_ms base={tot_b:.2f} peak={tot_p:.2f}")
    growth(tab, f"thread={name}")
