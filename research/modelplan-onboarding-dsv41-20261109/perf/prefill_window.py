#!/usr/bin/env python3
"""Where a prefill's wall time goes in an nsys trace of a dsv41-tpbench run (sqlite export).

The window of the ROWS-token prefill is the span from the first to the last launch of the 16-token
mHC pass whose grid is ceil(ROWS / 16) columns, extended to the next launch of that kernel with
another grid (or of the one-token kernel, or the end of the trace). Inside it, per stream: kernel
time by kernel, copies by kind and size with their rate, and the idle time before a copy (a gather
waiting for a peer) and before a kernel, split into the host's (the launch call returned after the
stream's previous event ended) and the device's. Then the host's CUDA calls.

usage: prefill_window.py PROFILE.sqlite --rows ROWS --layers L
"""
import bisect
import sqlite3
import sys
from collections import defaultdict


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    con = sqlite3.connect(sys.argv[1])
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    cur = con.cursor()
    rows, layers = int(arg("--rows")), int(arg("--layers"))
    S = dict(cur.execute("select id, value from StringIds"))
    kern = cur.execute("select start, end, shortName, streamId, deviceId, correlationId, gridX from "
                       "CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    name = lambda i: S.get(i, "?").replace("v41_", "").replace("_kernel", "")
    marks = [k for k in kern if "hc_mixes_part" in S.get(k[2], "")]
    mine = [k for k in marks if "part1" not in S.get(k[2], "") and k[6] == (rows + 15) // 16]
    if not mine:
        sys.exit(f"no 16-token mHC pass with {(rows + 15) // 16} columns")
    t0 = mine[0][0]
    later = [k[0] for k in marks if k[0] > mine[-1][0] and k not in mine]
    t1 = later[0] if later else max(k[1] for k in kern)
    # a trace with tpbench's --nvtx: the window ends with the prompt's own cell (its head and readback), not at the
    # next prompt's first mHC pass (which drags the next prompt's start-up, its upload and allocations, into this one)
    edge = "the next prompt's first mHC pass (no NVTX cell ranges: the next prompt's start-up is inside)"
    if cur.execute("select count(*) from sqlite_master where name = 'NVTX_EVENTS'").fetchone()[0]:
        cols = [r[1] for r in cur.execute("pragma table_info(NVTX_EVENTS)")]
        tid = "textId" if "textId" in cols else "NULL"
        for st, en, text, ti in cur.execute(f"select start, end, text, {tid} from NVTX_EVENTS where end is not null"):
            if (text if text is not None else S.get(ti, "")) == "v41 prefill cell" and st <= t0 <= en:
                t1, edge = min(t1, en), "the end of the prompt's own cell (NVTX)"
    wall = (t1 - t0) / 1e6
    print(f"prefill of {rows} tokens: {wall:.1f} ms wall, {wall / layers:.2f} ms a layer over {layers} layers "
          f"(from its first mHC pass to {edge})")
    issued = dict(cur.execute("select correlationId, end from CUPTI_ACTIVITY_KIND_RUNTIME"))
    ev = defaultdict(list)
    for st, en, nm, stream, dev, corr, _ in kern:
        if t0 <= st and en <= t1:
            ev[(dev, stream)].append((st, en, name(nm), corr, 0))
    kinds = {1: "host>dev", 2: "dev>host", 8: "dev>dev", 10: "peer"}
    for st, en, stream, dev, corr, b, kind in cur.execute(
            "select start, end, streamId, deviceId, correlationId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
        if t0 <= st and en <= t1:
            ev[(dev, stream)].append((st, en, f"copy {kinds.get(kind, kind)}", corr, b))
    for key in sorted(ev):
        e = sorted(ev[key])
        busy = defaultdict(lambda: [0, 0.0, 0])
        hostw = peerw = devw = 0.0
        for i, (st, en, nm, corr, b) in enumerate(e):
            lab = nm if not nm.startswith("copy") else f"{nm} {'<1MB' if b < 1 << 20 else '>=1MB'}"
            x = busy[lab]
            x[0] += 1
            x[1] += en - st
            x[2] += b
            if i:
                gap = max(0, st - e[i - 1][1])
                h = min(gap, max(0, issued.get(corr, e[i - 1][1]) - e[i - 1][1]))
                hostw += h
                if nm.startswith("copy"):
                    peerw += gap - h
                else:
                    devw += gap - h
        tot = sum(v[1] for v in busy.values())
        print(f"\nstream {key[0]}/{key[1]}: busy {tot / 1e6 / layers:.2f} ms a layer; idle a layer: waiting for the "
              f"host {hostw / 1e6 / layers:.2f}, before a copy {peerw / 1e6 / layers:.2f}, before a kernel "
              f"{devw / 1e6 / layers:.2f}")
        print("| kernel or copy | a layer | ms a layer | GB a layer | GB/s |")
        print("|---|---|---|---|---|")
        for lab, (c, ns, b) in sorted(busy.items(), key=lambda x: -x[1][1])[:16]:
            rate = f"{b / ns:.1f}" if b and ns else ""
            print(f"| {lab} | {c / layers:.1f} | {ns / 1e6 / layers:.3f} | {b / 1e9 / layers:.3f} | {rate} |")
    api = defaultdict(lambda: [0, 0.0])
    for st, en, nid in cur.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        if t0 <= st and en <= t1:
            a = api[S.get(nid, "?")]
            a[0] += 1
            a[1] += en - st
    print("\n| host CUDA call | calls a layer | ms a layer |")
    print("|---|---|---|")
    for nm, (c, ns) in sorted(api.items(), key=lambda x: -x[1][1])[:10]:
        print(f"| {nm} | {c / layers:.1f} | {ns / 1e6 / layers:.3f} |")


if __name__ == "__main__":
    main()
