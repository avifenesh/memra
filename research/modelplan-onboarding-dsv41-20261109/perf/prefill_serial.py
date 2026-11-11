#!/usr/bin/env python3
"""The host's serial part of each prefill in a dsv41-tpbench trace taken with --nvtx (nsys -t cuda,nvtx): from the
cell's start (NVTX "v41 prefill cell") to its first kernel launch, split by the forward's own ranges ("v41 hash": the
Engram hashes of the prompt; "v41 embed": the id check, the ids' upload and the row gather's launch on every rank)
and the bookkeeping around them (the sequence's state, buffers), with the CUDA calls in that span; then the first
kernel's start on a device, the first 16-row mHC pass, the cell's end (TTFT on the host), and over the whole cell the
items the load is meant to have removed: cuMemHostAlloc, cuMemAllocAsync calls of 1 ms and more, host-to-device
copies of 1 MB and more. usage: prefill_serial.py TRACE.sqlite"""
import sqlite3
import sys
from collections import defaultdict

con = sqlite3.connect(sys.argv[1])
con.text_factory = lambda b: b.decode("utf-8", "replace")
cur = con.cursor()
S = dict(cur.execute("select id, value from StringIds"))
tabs = {r[0] for r in cur.execute("select name from sqlite_master where type = 'table'")}
if "NVTX_EVENTS" not in tabs:
    sys.exit("no NVTX_EVENTS: trace with nsys -t cuda,nvtx and dsv41-tpbench --nvtx")
cols = [r[1] for r in cur.execute("pragma table_info(NVTX_EVENTS)")]
tid_col = "textId" if "textId" in cols else "NULL"
nv = []
for st, en, text, tid, gt in cur.execute(f"select start, end, text, {tid_col}, globalTid from NVTX_EVENTS where end is not null"):
    name = text if text is not None else S.get(tid, "")
    if name.startswith("v41 "):
        nv.append((st, en, name, gt))
nv.sort()
cells = [x for x in nv if x[2] == "v41 prefill cell"]
api = cur.execute("select start, end, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME order by start").fetchall()
kern = cur.execute("select start, end, shortName, gridX, deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
mk = {r[0]: r[2].lower() for r in cur.execute("select * from ENUM_CUDA_MEM_KIND")}
copies = cur.execute("select start, end, bytes, deviceId, srcKind from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 1 "
                     "and bytes >= 1048576 order by start").fetchall()
launch = lambda n: n.startswith("cudaLaunchKernel") or n.startswith("cuLaunchKernel")
ms = lambda a, b: (b - a) / 1e6
for ci, (c0, c1, _, cgt) in enumerate(cells):
    first_mhc = next((k for k in kern if c0 <= k[0] <= c1 and "hc_mixes_part" in S.get(k[2], "")
                      and "part1" not in S.get(k[2], "")), None)
    rows = first_mhc[3] * 16 if first_mhc else 0
    inner = {x[2]: x for x in nv if c0 <= x[0] <= c1 and x[2] in ("v41 hash", "v41 embed")}
    fl = next((a for a in api if a[0] >= c0 and launch(S.get(a[2], ""))), None)
    fk = next((k for k in kern if k[0] >= c0), None)
    print(f"\n## cell {ci}: about {rows} rows (first 16-row mHC pass of {first_mhc[3] if first_mhc else 0} columns)")
    print(f"host TTFT (the cell's NVTX range) {ms(c0, c1):.1f} ms")
    print("| span | ms |")
    print("|---|---|")
    h, e = inner.get("v41 hash"), inner.get("v41 embed")
    if h:
        print(f"| cell start -> hashing (the sequence's state, bookkeeping) | {ms(c0, h[0]):.2f} |")
        print(f"| hashing (Engram hashes of the prompt) | {ms(h[0], h[1]):.2f} |")
    if e:
        print(f"| hashing -> embed | {ms(h[1] if h else c0, e[0]):.2f} |")
        print(f"| embed (id check, ids' upload, the row gather launched on every rank) | {ms(e[0], e[1]):.2f} |")
    if fl:
        print(f"| cell start -> first kernel launch call ({S.get(fl[2])}) | {ms(c0, fl[0]):.2f} |")
    if fk:
        print(f"| cell start -> first kernel on a device ({S.get(fk[2], '?')[:40]}, device {fk[4]}) | {ms(c0, fk[0]):.2f} |")
    if first_mhc:
        print(f"| cell start -> first 16-row mHC pass (layer 0) on device {first_mhc[4]} | {ms(c0, first_mhc[0]):.2f} |")
    if fl:
        by = defaultdict(lambda: [0, 0])
        for a in api:
            if c0 <= a[0] < fl[0]:
                x = by[S.get(a[2], "?")]
                x[0] += 1
                x[1] += a[1] - a[0]
        print("CUDA calls before the first launch (all threads): " +
              "; ".join(f"{n} {c} / {t / 1e6:.2f} ms" for n, (c, t) in sorted(by.items(), key=lambda kv: -kv[1][1])[:8]))
    ha = [a for a in api if c0 <= a[0] <= c1 and S.get(a[2], "").startswith("cuMemHostAlloc")]
    sa = [a for a in api if c0 <= a[0] <= c1 and S.get(a[2], "").startswith("cuMemAllocAsync") and a[1] - a[0] >= 1_000_000]
    cp = [x for x in copies if c0 <= x[0] <= c1]
    print(f"in the cell: cuMemHostAlloc {len(ha)} calls / {sum(a[1] - a[0] for a in ha) / 1e6:.1f} ms; cuMemAllocAsync of "
          f">= 1 ms {len(sa)} calls / {sum(a[1] - a[0] for a in sa) / 1e6:.1f} ms; host-to-device copies >= 1 MB: " +
          (", ".join(f"{b / 1e6:.1f} MB {mk.get(k)} dev {d} {ms(s, en):.2f} ms" for s, en, b, d, k in cp) or "none"))
