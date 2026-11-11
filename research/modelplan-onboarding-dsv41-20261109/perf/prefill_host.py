#!/usr/bin/env python3
"""The host side of the 4,096-token prefill in the night pass's trace (sqlite export): (1) the host-to-device
copies of 1 MB and more (source memory kind, size, rate, which layer, the CUDA call and thread that issued them);
(2) the main stream's idle gaps where the next launch was issued after the stream ran dry ("waiting for the host"),
with what the issuing thread was doing in the gap; (3) the host calls by thread.

usage: prefill_host.py TRACE.sqlite [LAYERS (40)]
"""
import sqlite3
import sys
from collections import defaultdict

db = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 40
con = sqlite3.connect(db)
con.text_factory = lambda b: b.decode("utf-8", "replace")
cur = con.cursor()
S = dict(cur.execute("select id, value from StringIds"))
# the 4,096-token prefill: the peer copies of its 27.28 GB (after the 36 copies of the start-up ladder)
pc = cur.execute("select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 10 order by start").fetchall()[36:]
acc, w = 0, []
for x in pc:
    if acc >= 27282112512:
        break
    acc += x[2]
    w.append(x)
# the layer marker: the 16-row mHC pass's first kernel of each layer on device 0 (prefill_window's rule)
k = cur.execute("select start, end, shortName, streamId, deviceId, correlationId, gridX from CUPTI_ACTIVITY_KIND_KERNEL "
                "order by start").fetchall()
t0, t1 = w[0][0] - 60_000_000, w[-1][1] + 60_000_000
marks = [r for r in k if t0 <= r[0] <= t1 and r[4] == 0 and "hc_mixes_part" in S.get(r[2], "")]
lay0 = [m[0] for m in marks[::2]]  # two mHC passes a layer


def layer_of(t):
    i = 0
    while i + 1 < len(lay0) and lay0[i + 1] <= t:
        i += 1
    return i


print(f"prefill window {t0 / 1e9:.3f}-{t1 / 1e9:.3f} s, {len(marks)} mHC marks on device 0, {len(lay0)} layer starts")
api = {}
for st, en, tid, corr, nid in cur.execute("select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
    api[corr] = (st, en, tid, S.get(nid, "?"))
# the export's own numbering (ENUM_CUDA_MEM_KIND), which is not CUPTI's: 0 is pageable, 1 pinned
mk = {r[0]: r[2].lower() for r in cur.execute("select * from ENUM_CUDA_MEM_KIND")}
print("\n(1) host-to-device copies >= 1 MB in the window")
print("| start ms | device | stream | MB | ms | GB/s | src | layer | call (thread) | call ms |")
print("|---|---|---|---|---|---|---|---|---|---|")
tot = defaultdict(lambda: [0, 0, 0])
for st, en, b, dev, strm, corr, sk, dk in cur.execute(
        "select start, end, bytes, deviceId, streamId, correlationId, srcKind, dstKind from CUPTI_ACTIVITY_KIND_MEMCPY "
        "where copyKind = 1 and bytes >= 1048576 order by start"):
    if not (t0 <= st <= t1):
        continue
    a = api.get(corr, (0, 0, 0, "?"))
    print(f"| {(st - t0) / 1e6:.1f} | {dev} | {strm} | {b / 1e6:.1f} | {(en - st) / 1e6:.2f} | {b / (en - st):.1f} | "
          f"{mk.get(sk, sk)} | {layer_of(st)} | {a[3]} ({a[2] & 0xFFFFFF}) | {(a[1] - a[0]) / 1e6:.2f} |")
    t = tot[mk.get(sk, sk)]
    t[0] += 1
    t[1] += b
    t[2] += en - st
for kd, (c, b, ns) in tot.items():
    print(f"total {kd}: {c} copies, {b / 1e6:.0f} MB, {ns / 1e6:.1f} ms of copy time")
# (3) host calls by thread in the window
print("\n(3) host CUDA calls in the window by thread: calls, ms")
byt = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for corr, (st, en, tid, nm) in api.items():
    if t0 <= st <= t1:
        x = byt[tid & 0xFFFFFF][nm]
        x[0] += 1
        x[1] += en - st
for tid, d in sorted(byt.items(), key=lambda kv: -sum(v[1] for v in kv[1].values())):
    top = sorted(d.items(), key=lambda kv: -kv[1][1])[:6]
    print(f"thread {tid}: " + "; ".join(f"{nm} {c} / {ns / 1e6:.1f} ms" for nm, (c, ns) in top))
# (2) the main stream's host-late gaps on device 0 and what the issuing thread did in them
evs = [(r[0], r[1], S.get(r[2], "?"), r[5]) for r in k if r[4] == 0 and t0 <= r[0] <= t1]
st_count = defaultdict(int)
for r in k:
    if r[4] == 0 and t0 <= r[0] <= t1:
        st_count[r[3]] += 1
main = max(st_count, key=st_count.get)
evs = sorted((r[0], r[1], S.get(r[2], "?"), r[5]) for r in k if r[4] == 0 and r[3] == main and t0 <= r[0] <= t1)
for st, en, strm, dev, corr, b in cur.execute(
        "select start, end, streamId, deviceId, correlationId, bytes from CUPTI_ACTIVITY_KIND_MEMCPY"):
    if dev == 0 and strm == main and t0 <= st <= t1:
        evs.append((st, en, f"copy {b >> 20} MB", corr))
evs.sort()
calls_by_tid = defaultdict(list)
for corr, (st, en, tid, nm) in api.items():
    if t0 - 1e8 <= st <= t1:
        calls_by_tid[tid].append((st, en, nm))
for v in calls_by_tid.values():
    v.sort()
late = defaultdict(lambda: [0, 0])
late_layer = defaultdict(float)
for i in range(1, len(evs)):
    prev_end = evs[i - 1][1]
    st, en, nm, corr = evs[i]
    gap = st - prev_end
    a = api.get(corr)
    if gap <= 0 or not a:
        continue
    h = min(gap, max(0, a[1] - prev_end))
    if h <= 0:
        continue
    # the issuing thread's longest call that overlaps (prev_end, its launch)
    best, bl = "host code between calls", 0
    for cs, ce, cn in calls_by_tid[a[2]]:
        if ce < prev_end or cs > a[0]:
            continue
        o = min(ce, a[0]) - max(cs, prev_end)
        if o > bl:
            best, bl = cn, o
    if bl < 0.5 * h:
        best = "host code between calls"
    late[best][0] += 1
    late[best][1] += h
    late_layer[layer_of(st)] += h
print(f"\n(2) device 0 main stream {main}: idle gaps the host caused, by what the issuing thread was in")
print("| the issuing thread was in | gaps | ms a prefill | ms a layer |")
print("|---|---|---|---|")
for nm, (c, ns) in sorted(late.items(), key=lambda kv: -kv[1][1])[:10]:
    print(f"| {nm} | {c} | {ns / 1e6:.1f} | {ns / 1e6 / L:.2f} |")
print("host-late ms by layer: " + ", ".join(f"{l}:{v / 1e6:.1f}" for l, v in sorted(late_layer.items())))
