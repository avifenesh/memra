#!/usr/bin/env python3
"""The step model's table from the probe's `step` runs: per build and rows, a four-rank layer's dense set (a PRO
6000's share per SM, P:dense4) and its split MoE (plan + both halves), graph-replay us, medians over runs, and the
step's sum over 40 layers. usage: step_table.py RUN_DIR"""
import glob
import re
import statistics as st
import sys

d = sys.argv[1]
pat = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+n\s+(\d+)\s+k\s+(\d+)\s+rows\s+(\d+)\s+eager\s+([\d.]+) us\s+host\s+([\d.]+) us\s+graph\s+([\d.]+) us")
v = {}
for f in glob.glob(d + "/run*-*.txt"):
    for line in open(f):
        m = pat.match(line)
        if m:
            v.setdefault((m[1], m[3], int(m[6])), []).append(float(m[9]))
builds = [b for b in ["k3", "cutlass", "new", "new5"] if any(k[0] == b for k in v)]
rows = sorted({k[2] for k in v if k[1] == "plan+all"})
med = lambda b, c, r: st.median(v[(b, c, r)]) if (b, c, r) in v else float("nan")
print("| rows | " + " | ".join(f"{b}: dense / MoE / layer" for b in builds) + " | " +
      " | ".join(f"{b} - k3, 40 layers (ms)" for b in builds if b != "k3") + " |")
print("|" + "---|" * (1 + len(builds) + len(builds) - 1))
for r in rows:
    cells, lay = [], {}
    for b in builds:
        dn, mo = med(b, "P:dense4", r), med(b, "plan+all", r)
        lay[b] = dn + mo
        cells.append(f"{dn:.1f} / {mo:.1f} / {dn + mo:.1f}")
    deltas = [f"{(lay[b] - lay['k3']) * 40 / 1000:+.2f}" for b in builds if b != "k3"]
    print(f"| {r} | " + " | ".join(cells) + " | " + " | ".join(deltas) + " |")
print("\nsteps between rows (layer us, 8 -> 9 and 9 -> 16 a row):")
for b in builds:
    l = {r: med(b, "P:dense4", r) + med(b, "plan+all", r) for r in rows}
    if 8 in l and 9 in l and 16 in l:
        print(f"  {b}: 8 -> 9 {l[9] - l[8]:+.1f} us, 9 -> 16 {(l[16] - l[9]) / 7:+.1f} us a row, 1 -> 8 {(l[8] - l[1]) / 7:+.1f} us a row")
