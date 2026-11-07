#!/usr/bin/env python3
"""Per-kernel, per-depth medians across probe_rounds.sh rounds, and the ratio to the reference.

usage: probe_table.py LOG [LOG...]
"""
import re
import statistics
import sys
from collections import defaultdict

rows = defaultdict(list)   # (kernel, tkv) -> [ms]
tf = defaultdict(list)
bad = defaultdict(int)
regs = {}
ref = None
for path in sys.argv[1:]:
    for line in open(path):
        m = re.match(r"kernel (\S+)\s+regs (\d+) local (\d+) B smem (\d+) B warps (\d+)", line)
        if m:
            regs[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
        m = re.match(r"T (\d+) Tkv\s+(\d+) (\S+)\s+([\d.]+) ms\s+([\d.]+) TFLOPS .* mismatch (\d+) nonfinite (\d+)", line)
        if m:
            k, tkv = m.group(3), int(m.group(2))
            rows[(k, tkv)].append(float(m.group(4)))
            tf[(k, tkv)].append(float(m.group(5)))
            bad[(k, tkv)] += int(m.group(6)) + int(m.group(7))
        m = re.match(r"# round 1 order (\S+)", line)
        if m and ref is None:
            ref = m.group(1).split(",")[0]
kernels = sorted({k for k, _ in rows}, key=lambda k: (k != ref, k))
depths = sorted({t for _, t in rows})
print(f"{'kernel':10s} {'regs':>4s} {'local':>5s} {'smem':>6s} {'warps':>5s} " +
      " ".join(f"{d:>16d}" for d in depths) + "  mismatch")
for k in kernels:
    r = regs.get(k, ("?", "?", "?", "?"))
    cells = []
    nb = 0
    for d in depths:
        if (k, d) not in rows:
            cells.append(f"{'':>16s}")
            continue
        t = statistics.median(tf[(k, d)])
        base = statistics.median(tf[(ref, d)]) if (ref, d) in tf else None
        rel = f"{100 * (t / base - 1):+5.1f}%" if base and k != ref else "      "
        cells.append(f"{t:6.2f} {rel:>8s} ")
        nb += bad[(k, d)]
    print(f"{k:10s} {r[0]:>4} {r[1]:>5} {r[2]:>6} {r[3]:>5} " + " ".join(cells) + f"  {nb}")
print(f"(TFLOPS, median over rounds of each round's median of 3; % vs {ref}; n rounds = "
      f"{max(len(v) for v in rows.values())})")
