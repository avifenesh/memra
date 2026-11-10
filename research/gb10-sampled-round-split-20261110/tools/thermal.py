#!/usr/bin/env python3
"""thermal.py CELL_DIR...: the thermal regime of each request of a cell, from the cell's 250 ms
nvidia-smi telemetry: samples inside the request's wall-clock window (reqs.jsonl t_start, t_end),
temperature median and maximum, SM clock median, power median, throttle reasons seen. The
telemetry's local timestamps take the UTC offset of the first line of guard.log."""
import csv
import datetime
import json
import pathlib
import statistics
import sys

print("| cell | request | samples | temp median / max (C) | SM clock median (MHz) | power median (W) | throttle reasons |")
print("|---|---|---:|---|---:|---:|---|")
for d in map(pathlib.Path, sys.argv[1:]):
    tz = datetime.datetime.fromisoformat((d / "guard.log").read_text().split()[0]).tzinfo
    rows = []
    for r in list(csv.reader(open(d / "telemetry.csv")))[1:]:
        if len(r) < 7:
            continue
        try:
            t = datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=tz).timestamp()
            rows.append((t, float(r[1]), float(r[2].split()[0]), float(r[4].split()[0]), r[6].strip()))
        except ValueError:
            continue
    for q in (json.loads(l) for l in open(d / "reqs.jsonl") if l.strip()):
        w = [x for x in rows if q["t_start"] <= x[0] <= q["t_end"]]
        if not w:
            print(f"| {d.name} | {q['class']} | 0 | | | | |")
            continue
        print(f"| {d.name} | {q['class']} | {len(w)} | {statistics.median(x[1] for x in w):.0f} / {max(x[1] for x in w):.0f} | "
              f"{statistics.median(x[2] for x in w):.0f} | {statistics.median(x[3] for x in w):.1f} | {', '.join(sorted({x[4] for x in w}))} |")
