#!/usr/bin/env python3
"""Probe runs -> one table a cell kind: graph-replay us a call, median over the runs of each build."""
import glob
import re
import statistics as st
import sys

d = sys.argv[1]
kinds = sys.argv[2].split(",") if len(sys.argv) > 2 else ["gemm", "linear", "set", "moe"]
pat = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+n\s+(\d+)\s+k\s+(\d+)\s+rows\s+(\d+)\s+eager\s+([\d.]+) us\s+host\s+([\d.]+) us\s+graph\s+([\d.]+) us")
cells = {}
builds = []
for f in sorted(glob.glob(d + "/run*-*.txt")):
    for line in open(f):
        m = pat.match(line)
        if not m:
            continue
        b, kind, name, n, k, rows, eager, host, graph = m.groups()
        if b not in builds:
            builds.append(b)
        cells.setdefault((kind, name, int(n), int(k), int(rows)), {}).setdefault(b, []).append(float(graph))
order = [b for b in ["k3", "cutlass", "new"] if b in builds]
for kind in kinds:
    keys = [c for c in cells if c[0] == kind]
    if not keys:
        continue
    print(f"\n### {kind}: graph-replay us a call (median of runs; n runs)")
    hdr = "| shape | n | k | rows | " + " | ".join(order) + " | new / old | new / k3 |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    seen = []
    for c in sorted(keys, key=lambda c: (list(dict.fromkeys(x[1] for x in keys)).index(c[1]), c[4])):
        v = cells[c]
        med = {b: st.median(v[b]) for b in order if b in v}
        row = f"| {c[1]} | {c[2]} | {c[3]} | {c[4]} | " + " | ".join(
            f"{med[b]:.2f} ({len(v[b])})" if b in med else "-" for b in order)
        r1 = f"{med['new'] / med['cutlass']:.3f}" if "new" in med and "cutlass" in med else "-"
        r2 = f"{med['new'] / med['k3']:.3f}" if "new" in med and "k3" in med else "-"
        print(row + f" | {r1} | {r2} |")
