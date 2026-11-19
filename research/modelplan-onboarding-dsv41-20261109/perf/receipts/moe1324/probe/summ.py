#!/usr/bin/env python3
"""Summarize probe1324 outputs: per (rows, shape) main's time and the best variants (median over files)."""
import re
import sys
from collections import defaultdict
from statistics import median

top = 6
files = [f for f in sys.argv[1:] if not f.startswith("--")]
for a in sys.argv[1:]:
    if a.startswith("--top="):
        top = int(a[6:])
main = defaultdict(list)
sw = defaultdict(list)
var = defaultdict(lambda: defaultdict(list))
occ = {}
bad = defaultdict(int)
base = defaultdict(list)
for f in files:
    for line in open(f):
        m = re.match(r"cell main rows\s+(\d+) (\w+)\s+n\s+\d+ k\s+\d+ main\s+([\d.]+) us", line)
        if m:
            main[(int(m[1]), m[2])].append(float(m[3]))
            continue
        m = re.match(r"cell main rows\s+(\d+) swiglu\s+([\d.]+) us", line)
        if m:
            sw[int(m[1])].append(float(m[2]))
            continue
        m = re.match(r"cell base rows\s+(\d+) w13\+swiglu\s+([\d.]+) us w2\s+([\d.]+) us", line)
        if m:
            base[int(m[1])].append((float(m[2]), float(m[3])))
            continue
        m = re.match(r"var rows\s+(\d+) (\w+)\s+(\S+)\s+([\d.]+) us\s+/main [\d.]+\s+occ (\d+)\s+regs (\d+)\s+bytes differ (\d+)", line)
        if m:
            k = (int(m[1]), m[2])
            var[k][m[3]].append(float(m[4]))
            occ[(k, m[3])] = f"{m[5]} r{m[6]}"
            bad[(k, m[3])] += int(m[7])
rows = sorted({k[0] for k in main} | set(base))
for r in rows:
    b = base.get(r)
    bs = ""
    if b:
        bs = f"base w13+swiglu {median(x[0] for x in b):.2f} w2 {median(x[1] for x in b):.2f} (n {len(b)})"
    print(f"== rows {r}: swiglu {median(sw[r]) if sw[r] else float('nan'):.2f}  {bs}")
    for shp in ("w1w3", "w2"):
        k = (r, shp)
        if k not in main:
            continue
        mm = median(main[k])
        print(f"  {shp}: main {mm:.2f} us (n {len(main[k])})")
        vs = sorted(((median(v), n) for n, v in var[k].items()), key=lambda x: x[0])
        for t, n in vs[:top]:
            print(f"    {n:22s} {t:8.2f}  {t / mm:.3f}  occ {occ[(k, n)]}  differ {bad[(k, n)]}")
nb = sum(bad.values())
print(f"# byte differences over all variants and cells: {nb}")
