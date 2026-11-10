#!/usr/bin/env python3
"""Each host thread's time inside the decode window of an nsys trace of an N-rank decode run.

Per thread, per layer: the CUDA calls it made, the time inside them (and inside the blocking ones:
a device-to-host copy or a synchronize), and the time between two calls by gap size. A gap under a
microsecond is the host code between two launches; a gap of tens of microseconds is the thread
waiting for its next section of the layer (see `decode_window.py` for the window and the streams).

usage: thread_gaps.py PROFILE.sqlite RANKS LAYERS
"""
import sqlite3, sys
from collections import defaultdict
db, ranks, layers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
con = sqlite3.connect(db); con.text_factory = lambda b: b.decode("utf-8", "replace"); cur = con.cursor()
strings = dict(cur.execute("select id, value from StringIds"))
kern = cur.execute("select start, shortName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
marks = sorted(k[0] for k in kern if "hc_mixes_part1" in strings.get(k[1], ""))
per = ranks * layers * 2
t0, t1 = marks[per], marks[-1]
nsteps = (len(marks) - per) / per
rows = cur.execute("select start, end, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where start >= ? and end <= ? order by start", (t0, t1)).fetchall()
by = defaultdict(list)
for st, en, nid, tid in rows:
    by[tid].append((st, en, strings.get(nid, "?")))
edges = [0, 1e3, 3e3, 1e4, 3e4, 1e5, 1e9]
print(f"wall {(t1-t0)/1e3/nsteps/layers:.1f} us/layer")
for tid, ev in sorted(by.items(), key=lambda x: -len(x[1])):
    incall = sum(e - s for s, e, _ in ev)
    block = sum(e - s for s, e, n in ev if "DtoH" in n or "Synchronize" in n)
    hist = [[0, 0.0] for _ in edges]
    after = defaultdict(lambda: [0, 0.0])
    for a, b in zip(ev, ev[1:]):
        g = b[0] - a[1]
        for i in range(len(edges) - 1):
            if edges[i] <= g < edges[i + 1]:
                hist[i][0] += 1; hist[i][1] += g
        if g >= 1e4:
            k = after[(a[2].split("_v")[0], b[2].split("_v")[0])]; k[0] += 1; k[1] += g
    f = nsteps * layers
    print(f"\nthread {tid & 0xffffff}: {len(ev)/f:.0f} calls/layer, in calls {incall/1e3/f:.1f} us/layer (blocking {block/1e3/f:.1f}), between calls {(t1-t0-incall)/1e3/f:.1f} us/layer")
    print("  gaps: " + "; ".join(f"{edges[i]/1e3:g}-{edges[i+1]/1e3:g}us n={hist[i][0]/f:.1f}/layer {hist[i][1]/1e3/f:.1f}us/layer" for i in range(len(edges) - 1)))
    for (a, b), (c, ns) in sorted(after.items(), key=lambda x: -x[1][1])[:8]:
        print(f"    >=10us after {a} before {b}: {c/f:.2f}/layer, {ns/1e3/f:.1f} us/layer, {ns/c/1e3:.0f} us each")
