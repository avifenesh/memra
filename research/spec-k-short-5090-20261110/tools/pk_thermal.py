#!/usr/bin/env python3
"""Per-boot thermal summary from pk_ab.py's 250 ms nvidia-smi telemetry.
usage: pk_thermal.py DIR"""
import csv, pathlib, re, statistics, sys
D = pathlib.Path(sys.argv[1])
print("| boot | arm | samples busy | temp median / max (C) | busy SM clock median (MHz) | power median (W) | throttle reasons seen |")
print("|---:|---|---:|---|---:|---:|---|")
for f in sorted(D.glob("telemetry-*.csv"), key=lambda p: int(re.search(r"telemetry-(\d+)-", p.name).group(1))):
    n, arm = re.search(r"telemetry-(\d+)-([\w-]+)\.csv", f.name).groups()
    rows = list(csv.reader(open(f)))[1:]
    busy = []
    reasons = set()
    for r in rows:
        if len(r) < 7:
            continue
        try:
            temp = float(r[1]); sm = float(r[2].split()[0]); pw = float(r[4].split()[0]); util = float(r[5].split()[0])
        except ValueError:
            continue
        reasons.add(r[6].strip())
        if util >= 50:
            busy.append((temp, sm, pw))
    if not busy:
        print(f"| {n} | {arm} | 0 | | | | |"); continue
    print(f"| {n} | {arm} | {len(busy)} | {statistics.median(b[0] for b in busy):.0f} / {max(b[0] for b in busy):.0f} | "
          f"{statistics.median(b[1] for b in busy):.0f} | {statistics.median(b[2] for b in busy):.0f} | {', '.join(sorted(reasons))} |")
