#!/usr/bin/env python3
"""Where prefill_window.py's "waiting for the host" sits in time: the same window and the same split (a gap is the
host's when the next event's launch call returned after the stream's previous event ended), per stream, cut at the
4k prefill's last kernel before the next prompt's embedding upload, with the gaps of 1 ms and more listed and the
host calls the issuing thread was in during each. usage: window_hostgaps.py TRACE.sqlite --rows 4096 --layers 40"""
import sqlite3
import sys
from collections import defaultdict


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


con = sqlite3.connect(sys.argv[1])
con.text_factory = lambda b: b.decode("utf-8", "replace")
cur = con.cursor()
rows, layers = int(arg("--rows")), int(arg("--layers"))
S = dict(cur.execute("select id, value from StringIds"))
kern = cur.execute("select start, end, shortName, streamId, deviceId, correlationId, gridX from "
                   "CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
marks = [k for k in kern if "hc_mixes_part" in S.get(k[2], "")]
mine = [k for k in marks if "part1" not in S.get(k[2], "") and k[6] == (rows + 15) // 16]
t0 = mine[0][0]
later = [k[0] for k in marks if k[0] > mine[-1][0] and k not in mine]
t1 = later[0] if later else max(k[1] for k in kern)
# the next prompt's start: its cell's NVTX range (tpbench --nvtx), else its embedding upload (the first host-to-device
# copy of 100 MB and more after the last mHC pass; a build with the table on the card has none)
nxt = None
if cur.execute("select count(*) from sqlite_master where name = 'NVTX_EVENTS'").fetchone()[0]:
    ncols = [r[1] for r in cur.execute("pragma table_info(NVTX_EVENTS)")]
    tcol = "textId" if "textId" in ncols else "NULL"
    starts = sorted(st for st, text, ti in cur.execute(f"select start, text, {tcol} from NVTX_EVENTS")
                    if (text if text is not None else S.get(ti, "")) == "v41 prefill cell")
    nxt = next((st for st in starts if st > mine[-1][0]), None)
if nxt is None:
    nxt = cur.execute("select min(start) from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 1 and bytes >= 100000000 "
                      f"and start > {mine[-1][0]} and start < {t1}").fetchone()[0]
nxt = min(nxt or t1, t1)
print(f"window {(t1 - t0) / 1e6:.1f} ms (t0 = first {rows}-row mHC pass); last mHC pass of the prompt at "
      f"+{(mine[-1][0] - t0) / 1e6:.1f} ms; next prompt's upload at +{(nxt - t0) / 1e6:.1f} ms")
api = {}
for st, en, tid, corr, nid in cur.execute("select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
    api[corr] = (st, en, tid & 0xFFFFFF, S.get(nid, "?"))
calls = defaultdict(list)
for corr, (st, en, tid, nm) in api.items():
    if t0 - 1e8 <= st <= t1:
        calls[tid].append((st, en, nm))
for v in calls.values():
    v.sort()
ev = defaultdict(list)
for st, en, nm, stream, dev, corr, _ in kern:
    if t0 <= st and en <= t1:
        ev[(dev, stream)].append((st, en, S.get(nm, "?")[:40], corr))
for st, en, stream, dev, corr, b, kind in cur.execute(
        "select start, end, streamId, deviceId, correlationId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
    if t0 <= st and en <= t1:
        ev[(dev, stream)].append((st, en, f"copy kind {kind} {b >> 20} MB", corr))
lastk = 0
for key in sorted(ev):
    e = sorted(ev[key])
    seg = defaultdict(float)
    big = []
    for i in range(1, len(e)):
        st, en, nm, corr = e[i]
        pe = e[i - 1][1]
        gap = max(0, st - pe)
        a = api.get(corr)
        iss = a[1] if a else pe
        h = min(gap, max(0, iss - pe))
        if h <= 0:
            continue
        part = "in the prompt's layers" if st <= mine[-1][0] else (
            "after its last mHC pass, before the next upload" if st < nxt else "the next prompt's start")
        seg[part] += h
        if h >= 1e6:
            best, bl = "host code between calls", 0
            if a:
                for cs, ce, cn in calls[a[2]]:
                    if ce < pe or cs > a[0]:
                        continue
                    o = min(ce, a[0]) - max(cs, pe)
                    if o > bl:
                        best, bl = cn, o
                if bl < 0.5 * h:
                    best = "host code between calls"
            big.append(((pe - t0) / 1e6, h / 1e6, nm, best))
    tot = sum(seg.values())
    print(f"\nstream {key[0]}/{key[1]}: host-late {tot / 1e6:.1f} ms ({tot / 1e6 / layers:.2f} a layer): " +
          "; ".join(f"{k} {v / 1e6:.1f} ms" for k, v in seg.items()))
    for at, h, nm, best in big:
        print(f"  gap at +{at:.1f} ms: {h:.1f} ms before {nm}; the issuing thread was in {best}")
