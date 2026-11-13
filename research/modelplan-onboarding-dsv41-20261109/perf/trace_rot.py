#!/usr/bin/env python3
"""Kernel time by name of a mixed-width trace against the sum of its widths traced alone (nsys sqlite
exports of dsv41-tpbench runs with the same rounds).
usage: trace_rot.py MIXED.sqlite ALONE.sqlite... [--top N]
Prints per trace: kernels, their summed time, the busy span (first launch to last end) and what the device
did besides kernels (memsets, copies, memory events); then per kernel name the mixed trace's time against
the alone traces' sum, sorted by the difference."""
import sqlite3
import sys
from collections import defaultdict

args = [a for a in sys.argv[1:] if not a.startswith("--")]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 24
if "--top" in sys.argv:
    args.remove(str(top))


def tables(cur):
    return {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}


def load(path):
    con = sqlite3.connect(path)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    cur = con.cursor()
    names = dict(cur.execute("select id, value from StringIds"))
    t = defaultdict(lambda: [0, 0])
    lo, hi = None, 0
    for s, e, n in cur.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
        k = names.get(n, str(n)).split("<")[0].replace("v41_", "").replace("_kernel", "")
        t[k][0] += e - s
        t[k][1] += 1
        lo = s if lo is None else min(lo, s)
        hi = max(hi, e)
    tb = tables(cur)
    extra = {}
    for name in ("CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        if name in tb:
            n, ns = cur.execute(f"select count(*), coalesce(sum(end - start), 0) from {name}").fetchone()
            extra[name.rsplit("_", 1)[1].lower()] = (n, ns)
    for name in ("CUDA_GPU_MEMORY_USAGE_EVENT",):
        if name in tb:
            extra["memory events"] = (cur.execute(f"select count(*) from {name}").fetchone()[0], 0)
    # the host's driver calls by name
    host = defaultdict(lambda: [0, 0])
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tb:
        for s, e, n in cur.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
            k = names.get(n, str(n))
            host[k][0] += e - s
            host[k][1] += 1
    return t, (hi - (lo or 0)), extra, host


mixed = load(args[0])
alone = [load(p) for p in args[1:]]
for label, (t, span, extra, _) in [("mixed", mixed)] + [(f"alone{i + 1}", a) for i, a in enumerate(alone)]:
    n = sum(v[1] for v in t.values())
    ns = sum(v[0] for v in t.values())
    ex = " ".join(f"{k}: {v[0]} ({v[1] / 1e6:.1f} ms)" for k, v in extra.items())
    print(f"{label}: {n} kernels, {ns / 1e6:.1f} ms in kernels, span {span / 1e6:.1f} ms; {ex}")
if alone:
    names = set(mixed[0])
    for a in alone:
        names |= set(a[0])
    rows = []
    for k in names:
        m = mixed[0].get(k, [0, 0])
        s = [sum(a[0].get(k, [0, 0])[i] for a in alone) for i in (0, 1)]
        rows.append((m[0] - s[0], k, m, s))
    tm = sum(v[0] for v in mixed[0].values())
    ts = sum(sum(v[0] for v in a[0].values()) for a in alone)
    print(f"kernels: mixed {tm / 1e6:.1f} ms against the alone traces' {ts / 1e6:.1f} ms ({tm / max(ts, 1):.3f})")
    print("| kernel | launches mixed / alone | ms mixed | ms alone | ratio |")
    print("|---|---|---|---|---|")
    for d, k, m, s in sorted(rows, key=lambda x: -abs(x[0]))[:top]:
        print(f"| {k} | {m[1]} / {s[1]} | {m[0] / 1e6:.2f} | {s[0] / 1e6:.2f} | {m[0] / max(s[0], 1):.3f} |")
    hn = set(mixed[3])
    for a in alone:
        hn |= set(a[3])
    hrows = []
    for k in hn:
        m = mixed[3].get(k, [0, 0])
        s = [sum(a[3].get(k, [0, 0])[i] for a in alone) for i in (0, 1)]
        hrows.append((m[0] - s[0], k, m, s))
    print("| host call | calls mixed / alone | ms mixed | ms alone |")
    print("|---|---|---|---|")
    for d, k, m, s in sorted(hrows, key=lambda x: -abs(x[0]))[:12]:
        print(f"| {k} | {m[1]} / {s[1]} | {m[0] / 1e6:.2f} | {s[0] / 1e6:.2f} |")
