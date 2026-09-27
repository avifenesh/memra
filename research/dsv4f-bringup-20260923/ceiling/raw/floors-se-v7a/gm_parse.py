#!/usr/bin/env python3
"""Average each GPU metric per device over that device's kernel span in an nsys sqlite export
(memra #710 ceiling). Usage: gm_parse.py <export.sqlite> [steps]. Prints one line per (device,
metric) with the span mean, and for the DRAM throughput metrics the implied bytes per step at the
1.792 TB/s spec peak."""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 0
cur = db.cursor()
tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
names = {}
for row in cur.execute("select typeId, metricId, metricName from TARGET_INFO_GPU_METRICS"):
    names[(row[0], row[1])] = row[2]
spans = {}
busy = {}
for tab in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
    if tab in tables and not spans:
        for dev, lo, hi, tot, n in cur.execute(f"select deviceId, min(start), max(end), sum(end-start), count(*) from {tab} group by deviceId"):
            spans[dev] = (lo, hi)
            busy[dev] = (tot, n)
        print("span table", tab, {d: {"span_ms": round((b - a) / 1e6, 3), "busy_ms": round(busy[d][0] / 1e6, 3), "records": busy[d][1]} for d, (a, b) in spans.items()})
types = sorted({t for t, _ in names})
print("types", types, "kernel spans (ms)", {d: round((b - a) / 1e6, 3) for d, (a, b) in spans.items()})
# nsys orders GPU metric typeIds by device; pair them with the kernel devices in order.
devs = sorted(spans)
for i, t in enumerate(types):
    dev = devs[i] if i < len(devs) else None
    lo, hi = spans.get(dev, (None, None))
    q = "select metricId, avg(value), count(*) from GPU_METRICS where typeId=?"
    args = [t]
    if lo is not None:
        q += " and timestamp between ? and ?"
        args += [lo, hi]
    q += " group by metricId"
    for mid, avg, n in cur.execute(q, args):
        name = names.get((t, mid), str(mid))
        line = f"GM type={t} dev={dev} {name}: mean={avg:.2f} samples={n}"
        if "DRAM" in name and "%" in name and lo is not None and steps:
            secs = (hi - lo) / 1e9
            line += f" bytes_per_step={avg / 100 * 1.792e12 * secs / steps / 1e9:.3f}GB"
        print(line)
