#!/usr/bin/env python3
"""Per-op GPU time of a `dsv41-bench layer --nvtx` profile (nsys sqlite export).

Every GPU operation (kernel, memcpy, memset) is attributed to the NVTX ranges open on the host
thread when it was launched (by correlation id): the outer range `L<layer>/<phase>` names the
layer and phase, the innermost op range (`qr`, `wo_b`, `experts`, ...) the op. Reported per outer
range instance (one prefill rep or one decode step): GPU busy time per op, the host wall of the
outer range, and the op's kernels.

usage: prof_attr.py PROFILE.sqlite [--json OUT.json] [--md OUT.md]
"""
import json
import re
import sqlite3
import sys
from bisect import bisect_right
from collections import defaultdict

OUTER = re.compile(r"^L\d+/(prefill|decode)$")


def table_exists(cur, name):
    return cur.execute("select 1 from sqlite_master where type='table' and name=?", (name,)).fetchone() is not None


def main():
    db = sys.argv[1]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    out_md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    con = sqlite3.connect(db)
    con.text_factory = lambda b: b.decode("utf-8", "replace")  # an export can carry a non-UTF-8 string id
    cur = con.cursor()
    strings = dict(cur.execute("select id, value from StringIds"))

    # NVTX push/pop ranges per thread
    rows = cur.execute(
        "select start, end, text, textId, globalTid from NVTX_EVENTS where eventType = 59 and end is not null"
    ).fetchall()
    by_tid = defaultdict(list)
    for st, en, text, tid_text, gtid in rows:
        name = text if text is not None else strings.get(tid_text, "?")
        by_tid[gtid].append((st, en, name))
    # ranges nest per thread: give each its parent, so the stack at time t is the innermost range
    # that started before t and contains it, then its ancestors
    parent = {}
    for t, v in by_tid.items():
        v.sort(key=lambda r: (r[0], -r[1]))
        stk = []
        par = []
        for r in v:
            while stk and stk[-1][1] < r[0]:
                stk.pop()
            par.append(stk[-1] if stk else None)
            stk.append(r)
        parent[t] = dict(zip(v, par))
    starts = {t: [r[0] for r in v] for t, v in by_tid.items()}

    def stack_at(gtid, t):
        v = by_tid.get(gtid)
        if not v:
            return []
        i = bisect_right(starts[gtid], t) - 1
        if i < 0:
            return []
        r = v[i]
        # the latest-starting range before t may have closed; climb to one that contains t
        while r is not None and r[1] < t:
            r = parent[gtid][r]
        out = []
        while r is not None:
            out.append(r)
            r = parent[gtid][r]
        return out[::-1]

    # launches: correlation id -> (host time, thread); host API calls per thread for the host-time table
    launch = {}
    api = []
    for tbl in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table_exists(cur, tbl):
            for st, en, gtid, cid, nid in cur.execute(
                f"select start, end, globalTid, correlationId, nameId from {tbl}"
            ):
                launch[cid] = (st, gtid)
                api.append((st, en, gtid, strings.get(nid, "?")))

    ops = []
    if table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        for st, en, cid, nid in cur.execute(
            "select start, end, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            ops.append((st, en, cid, strings.get(nid, "?")))
    if table_exists(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for st, en, cid, kind in cur.execute(
            "select start, end, correlationId, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"
        ):
            ops.append((st, en, cid, f"memcpy[{kind}]"))
    if table_exists(cur, "CUPTI_ACTIVITY_KIND_MEMSET"):
        for st, en, cid in cur.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_MEMSET"):
            ops.append((st, en, cid, "memset"))

    # outer instance key: (outer name, outer start) -> accumulators
    inst = {}
    for st, en, cid, kname in ops:
        if cid not in launch:
            continue
        t, gtid = launch[cid]
        stk = stack_at(gtid, t)
        outer = None
        for r in stk:
            if OUTER.match(r[2]):
                outer = r
        if outer is None:
            continue
        inner = [r[2] for r in stk if r[0] >= outer[0] and r is not outer]
        op = "/".join(inner) if inner else "(unranged)"
        key = (outer[2], outer[0])
        d = inst.setdefault(key, {"wall_ns": outer[1] - outer[0], "ops": defaultdict(float),
                                  "kern": defaultdict(lambda: [0.0, 0]), "gpu_first": None, "gpu_last": None})
        d["ops"][op] += en - st
        k = d["kern"][(op, kname)]
        k[0] += en - st
        k[1] += 1
        d["gpu_first"] = st if d["gpu_first"] is None else min(d["gpu_first"], st)
        d["gpu_last"] = en if d["gpu_last"] is None else max(d["gpu_last"], en)

    # host API time per outer instance, by call name
    host = {}
    for st, en, gtid, name in api:
        stk = stack_at(gtid, st)
        outer = None
        for r in stk:
            if OUTER.match(r[2]):
                outer = r
        if outer is None:
            continue
        key = (outer[2], outer[0])
        h = host.setdefault(key, defaultdict(lambda: [0.0, 0]))
        h[name][0] += en - st
        h[name][1] += 1

    # per outer name: mean over its instances
    agg = defaultdict(list)
    for (name, _), d in sorted(inst.items(), key=lambda x: x[0][1]):
        agg[name].append(d)
    report = {}
    md = []
    for name, lst in agg.items():
        n = len(lst)
        ops_mean = defaultdict(float)
        kern_mean = defaultdict(lambda: [0.0, 0.0])
        for d in lst:
            for op, ns in d["ops"].items():
                ops_mean[op] += ns / n
            for (op, kn), (ns, cnt) in d["kern"].items():
                kern_mean[(op, kn)][0] += ns / n
                kern_mean[(op, kn)][1] += cnt / n
        wall = sum(d["wall_ns"] for d in lst) / n
        busy = sum(ops_mean.values())
        hmean = defaultdict(lambda: [0.0, 0.0])
        keys = [k for k in inst if k[0] == name]
        for k in keys:
            for nm, (ns, cnt) in host.get(k, {}).items():
                hmean[nm][0] += ns / len(keys)
                hmean[nm][1] += cnt / len(keys)
        report[name] = {
            "instances": n,
            "wall_ms": wall / 1e6,
            "gpu_busy_ms": busy / 1e6,
            "ops_ms": {k: v / 1e6 for k, v in sorted(ops_mean.items(), key=lambda x: -x[1])},
            "kernels": [
                {"op": op, "kernel": kn, "ms": ns / 1e6, "launches": cnt}
                for (op, kn), (ns, cnt) in sorted(kern_mean.items(), key=lambda x: -x[1][0])
            ],
            "host_api": [
                {"call": nm, "ms": ns / 1e6, "calls": cnt}
                for nm, (ns, cnt) in sorted(hmean.items(), key=lambda x: -x[1][0])
            ],
        }
        md.append(f"### {name} ({n} instances): wall {wall / 1e6:.3f} ms, GPU busy {busy / 1e6:.3f} ms\n")
        md.append("| op | GPU ms | share of busy |")
        md.append("|---|---|---|")
        for op, ns in sorted(ops_mean.items(), key=lambda x: -x[1]):
            md.append(f"| {op} | {ns / 1e6:.3f} | {100 * ns / max(busy, 1):.1f}% |")
        hs = sorted(hmean.items(), key=lambda x: -x[1][0])
        md.append("")
        md.append(f"host API calls: {sum(c for _, (_, c) in hs):.0f} calls, {sum(ns for _, (ns, _) in hs) / 1e6:.3f} ms; top: "
                  + ", ".join(f"{nm} {ns / 1e6:.3f} ms x{c:.0f}" for nm, (ns, c) in hs[:6]))
        md.append("")
    text = "\n".join(md)
    print(text)
    if out_json:
        with open(out_json, "w") as f:
            json.dump(report, f, indent=1)
    if out_md:
        with open(out_md, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
