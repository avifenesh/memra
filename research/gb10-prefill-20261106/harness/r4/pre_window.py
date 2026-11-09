#!/usr/bin/env python3
"""pre_window.py PROF.sqlite [SECONDS]: where the host time before a cold prime's first kernel goes.

Anchors at the first prime chunk kernel of the request (the first `embed_gather_u32_t` after the
longest GPU-idle stretch, i.e. after the server sat idle waiting for the request) and reports, over
the SECONDS (default 0.8) before it: CUDA driver/runtime API time by call, OS runtime (osrt) time
by call, and CPU sampling leaf and caller-chain symbols of the server process."""
import collections
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
win = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))
tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}

ks = cur.execute("SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
# the request's first kernel: the kernel after the longest idle gap in the trace
best, anchor = 0, None
for (s0, e0, _), (s1, _, n1) in zip(ks, ks[1:]):
    if s1 - e0 > best:
        best, anchor = s1 - e0, (s1, names.get(n1, "?"))
t1 = anchor[0]
t0 = t1 - int(win * 1e9)
print(f"anchor: first kernel after the longest idle ({best / 1e9:.3f} s): {anchor[1][:60]} at {t1 / 1e9:.3f} s")
print(f"window: {win:.3f} s before it\n")


def by_name(table, label):
    if table not in tables:
        print(f"{label}: no {table} table\n")
        return
    rows = cur.execute(f"SELECT start, end, nameId FROM {table} WHERE end > ? AND start < ?", (t0, t1)).fetchall()
    agg = collections.defaultdict(lambda: [0, 0.0])
    for s, e, n in rows:
        a = agg[names.get(n, str(n))]
        a[0] += 1
        a[1] += (min(e, t1) - max(s, t0)) / 1e9
    print(f"{label} in the window (calls, seconds):")
    for k, (c, sec) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:20]:
        print(f"  {sec:8.4f} s  {c:6d} x  {k[:90]}")
    print()


by_name("CUPTI_ACTIVITY_KIND_RUNTIME", "CUDA API")
by_name("OSRT_API", "OS runtime")

if "COMPOSITE_EVENTS" in tables and "SAMPLING_CALLCHAINS" in tables:
    samples = cur.execute("SELECT id, globalTid FROM COMPOSITE_EVENTS WHERE start > ? AND start < ?", (t0, t1)).fetchall()
    chains = collections.defaultdict(list)
    for cid, depth, sym in cur.execute("SELECT id, stackDepth, symbol FROM SAMPLING_CALLCHAINS"):
        chains[cid].append((depth, sym))
    leaf = collections.Counter()
    frames = collections.Counter()
    per_thread = collections.Counter()
    for cid, gtid in samples:
        st = sorted(chains.get(cid, []))
        if not st:
            continue
        per_thread[gtid] += 1
        leaf[names.get(st[0][1], str(st[0][1]))] += 1
        for name in {names.get(s, str(s)) for _, s in st[:12]}:
            frames[name] += 1
    n = sum(per_thread.values())
    print(f"CPU samples in the window: {n} over {len(per_thread)} threads (most: {per_thread.most_common(4)})")
    print("leaf symbols:")
    for k, c in leaf.most_common(20):
        print(f"  {c:6d}  {k[:110]}")
    print("symbols on the top 12 frames (inclusive):")
    for k, c in frames.most_common(30):
        print(f"  {c:6d}  {k[:110]}")
else:
    print("no CPU sampling tables")
